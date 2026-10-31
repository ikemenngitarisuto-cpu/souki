// 想起の受け口: 想起の「ComfyUI で開く」から届いたワークフローを、この画面に読み込む。
// 画像をドラッグしたときと同じように、新しいタブ（ワークフロー）として開く。
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function notify(summary, detail, severity) {
  try {
    const t = app.extensionManager && app.extensionManager.toast;
    if (t && t.add) { t.add({ severity: severity || "info", summary, detail, life: 4000 }); return; }
  } catch (e) {}
  console.log("[想起]", summary, detail || "");
}

async function openFromSouki(d) {
  if (!d || (!d.workflow && !d.api)) return;
  const name = d.name || "想起";
  try {
    if (d.workflow) {
      await app.loadGraphData(d.workflow, true, true, name);
    } else if (app.loadApiJson) {
      await app.loadApiJson(d.api, name);
    } else {
      notify("想起", "この ComfyUI では API 形式のワークフローを開けません", "warn");
      return;
    }
    notify("想起から開きました", name, "success");
  } catch (e) {
    console.error("[想起] ワークフローを開けませんでした", e);
    notify("想起", "ワークフローを開けませんでした: " + (e && e.message ? e.message : e), "error");
  }
}

app.registerExtension({
  name: "souki.bridge",
  async setup() {
    api.addEventListener("souki.open", (ev) => openFromSouki(ev.detail));
    // ComfyUI の画面が開いていなかったときに預けたものを受け取る
    try {
      const r = await api.fetchApi("/souki/pending");
      if (r.ok) {
        const d = await r.json();
        if (d && (d.workflow || d.api)) setTimeout(() => openFromSouki(d), 300);
      }
    } catch (e) {}
  },
});
