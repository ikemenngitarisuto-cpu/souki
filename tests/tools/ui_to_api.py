# -*- coding: utf-8 -*-
"""画面用ワークフロー（UI 形式）を API 形式にする、テスト用の簡単な変換。

ComfyUI のフロントがやっていることのうち、テストに要る分だけ:
  - サブグラフの中のノードを "外:中" の番号で平らに並べる
  - サブグラフの入口につながった線は、外側の線か、外側のノードの値に置き換える
  - フロント専用のノード（MarkdownNote など）と、止めてあるノードは除く
ウィジェットの名前は、名前つきの値（widgets_values_named）か、下の表から読む。

  python tests/tools/ui_to_api.py 入力.ui.json 出力.api.json
"""
import json
import sys

FRONT_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode", "ImageCompare"}
CONTROL = {"fixed", "increment", "decrement", "randomize"}
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "ComfySwitchNode": ["switch"],
    "QwenImage21Cache": ["device", "dtype"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "PrimitiveStringMultiline": ["value"],
    "TextGenerate": ["prompt", "max_length", "sampling_mode", "sampling_mode.temperature",
                     "sampling_mode.top_k", "sampling_mode.top_p", "sampling_mode.min_p",
                     "sampling_mode.repetition_penalty", "sampling_mode.seed", "thinking",
                     "use_default_template", "keep_loaded", "device"],
    "LoadImage": ["image", "upload"],
    "SaveImage": ["filename_prefix"],
    "SaveImageAdvanced": ["filename_prefix", "format", "bit_depth", "color_space"],
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CLIPTextEncode": ["text"],
    "ResolutionSelector": ["aspect_ratio", "megapixels"],
}


def link_table(links):
    out = {}
    for l in links or []:
        if isinstance(l, dict):
            out[l["id"]] = (l["origin_id"], l["origin_slot"], l["target_id"], l["target_slot"])
        else:
            out[l[0]] = (l[1], l[2], l[3], l[4])
    return out


def widget_values(nd):
    named = nd.get("widgets_values_named")
    if isinstance(named, dict):
        return {k: v for k, v in named.items() if k != "control_after_generate"}
    wv = nd.get("widgets_values")
    if isinstance(wv, dict):
        return dict(wv)
    names = WIDGETS.get(nd["type"])
    if not isinstance(wv, list) or names is None:
        return {}
    out, i = {}, 0
    for n in names:
        if i >= len(wv):
            break
        out[n] = wv[i]
        i += 1
        if i < len(wv) and isinstance(out[n], int) and wv[i] in CONTROL:
            i += 1
    return out


def convert(g):
    subs = {sg["id"]: sg for sg in (g.get("definitions") or {}).get("subgraphs", [])}
    api = {}

    def flatten(nodes, links, prefix, outer_resolve):
        """outer_resolve(slot) -> 線 [番号, 出口] か ("value", 値)。サブグラフの入口用。"""
        lt = link_table(links)
        by_id = {nd["id"]: nd for nd in nodes}

        def src(link_id):
            o_id, o_slot, _, _ = lt[link_id]
            if o_id == -10:
                return outer_resolve(o_slot)
            o = by_id.get(o_id)
            if o is None or o.get("mode") in (2, 4):
                return None
            if o["type"] in subs:
                sg = subs[o["type"]]
                inner = sg["outputs"][o_slot]["linkIds"][0]
                ilt = link_table(sg["links"])
                io_id, io_slot, _, _ = ilt[inner]
                return [f"{prefix}{o_id}:{io_id}", io_slot]
            return [f"{prefix}{o_id}", o_slot]

        for nd in nodes:
            if nd["type"] in FRONT_ONLY or nd.get("mode") in (2, 4):
                continue
            nid = f"{prefix}{nd['id']}"
            if nd["type"] in subs:
                sg = subs[nd["type"]]
                vals = widget_values(nd)
                ins = {i["name"]: i for i in nd.get("inputs", [])}

                def res(slot, sg=sg, vals=vals, ins=ins):
                    name = sg["inputs"][slot]["name"]
                    i = ins.get(name)
                    if i and i.get("link") is not None:
                        return src(i["link"])
                    if name in vals:
                        return ("value", vals[name])
                    return None
                flatten(sg["nodes"], sg["links"], f"{nid}:", res)
                continue
            inputs = dict(widget_values(nd))
            for i in nd.get("inputs", []):
                if i.get("link") is None:
                    continue
                s = src(i["link"])
                if s is None:
                    inputs.pop(i["name"], None)
                elif isinstance(s, tuple):
                    inputs[i["name"]] = s[1]
                else:
                    inputs[i["name"]] = s
            api[nid] = {"inputs": inputs, "class_type": nd["type"],
                        "_meta": {"title": nd.get("title") or nd["type"]}}

    flatten(g["nodes"], g.get("links"), "", lambda slot: None)
    return api


if __name__ == "__main__":
    g = json.load(open(sys.argv[1], encoding="utf-8"))
    json.dump(convert(g), open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
