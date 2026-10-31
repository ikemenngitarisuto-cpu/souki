# 想起（Souki）— 開発メモ

このリポで作業する Claude 向けのメモ。仕様そのものは docs/ にある。

## 想起とは

ComfyUI が出力した PNG に入っているワークフロー（API 形式の prompt と、画面用の workflow）を読み取って SQLite の台帳にし、
ブラウザの画面で絞り込み・再投入（キャラ・服の差し替えつき）をするツール。Windows で動かす。
顔でキャラを見分けるキャラ帳がついている。

| ファイル | 役目 |
|---|---|
| souki.py | サーバ本体（台帳・API・ComfyUI 連携・差し替え・保存名のひな形） |
| chara.py | キャラ帳（顔の振り分け、キャラの特徴語） |
| vision.py | 顔の検出と特徴（ONNX Runtime） |
| ui.html | 台帳の画面 |
| chara.html | キャラ帳の画面 |
| 起動.bat | いつもの起動口（runtime\venv の Python で `souki.py serve`） |
| setup/ | 最初の準備（setup.bat）、作り直し（環境を作り直す.bat）、requirements |
| comfyui_node/souki_bridge/ | ComfyUI 側に入れるカスタムノード（/souki/ping・/souki/open・/souki/pending と web/souki_bridge.js） |
| souki.ico, souki_256.png, chara.ico, chara_256.png | アイコン |
| docs/仕様_v2.4.md | 次の 2 回分（v2.4.0・v2.5.0）の確定仕様 |
| tests/fixtures/ | テスト用のワークフロー（下の「テスト」） |

設定は config.json、台帳やログは data/、Python 環境は runtime/（どれもリポに入れない。.gitignore 済み）。
サーバの既定ポートは 8765（`serve --port` で変えられる）、ComfyUI の既定は http://127.0.0.1:8188。

## いまの状態と次の作業

- 今の版は v2.4.0（docs/仕様_v2.4.md の第 1 弾）。
- 次は同じ仕様書の「第 2 弾（v2.5.0）」。仕様は作者と確定済みなので、そのとおりに作る。
  書いていないことで迷ったら作者に聞く。
- 版を上げるときは souki.py・chara.py・vision.py の `VERSION`、ui.html の `UI_VERSION`、chara.html の `PAGE_VERSION` を全部そろえる
  （画面とサーバの版が違うと警告が出る）。souki.py の `BRIDGE_VERSION` はブリッジのノードを変えたときだけ上げる。
- 利用者向けの説明 README.md も版ごとに直す。

## 作者とのやりとり・納品のきまり

- 日本語で話す。専門用語はかみくだき、仕様書と同じ言い方（「窓」「線」「保存名のひな形」など）を使う。
- 大きめの変更は、先に仕様を決めてから作る。選択肢を出すときは、おすすめを先頭に置いて理由をひとこと添える。
- できあがったら GitHub に push し、同じ中身の zip（`souki_v<版>.zip`）も渡す。
  - 日本語のファイル名が化けないよう、zip は UTF-8 フラグを立てて作る（Python の zipfile で `ZipInfo.flag_bits |= 0x800`）。
  - 入れるもの: 起動.bat, souki.py, chara.py, vision.py, ui.html, chara.html, README.md, souki.ico, souki_256.png, chara.ico,
    chara_256.png, setup/setup.bat, setup/環境を作り直す.bat, setup/requirements.txt, setup/requirements.in,
    comfyui_node/souki_bridge/__init__.py, comfyui_node/souki_bridge/web/souki_bridge.js
    （config.json・data/・docs/・tests/・CLAUDE.md は入れない）
- アイコン画像は「いつもどおり」必ず入れる。
- bat ファイルは ASCII だけ、改行は CRLF、`if (...)` を何行にもまたがるブロックで書かない（1 行の if と goto で書く）。

## 技術メモ（v2.3.0 の時点で分かっていること）

### ComfyUI のワークフロー

- API 形式（PNG の prompt）: `{ノード番号: {class_type, inputs, _meta: {title}}}`。線は `[ノード番号, 出口番号]`。
  サブグラフの中のノードは `"外:中"`（例 `"459:474"`）という番号で平らに並ぶ。
- 画面用（PNG の workflow）: `nodes[]` の `widgets_values`（並び順の値）と、新しいフロントの `widgets_values_named`（名前つきの値。
  設定 Comfy.Workflow.NamedValuesRestore）。`links` は `[線番号, 元ノード, 元の出口, 先ノード, 先の入口, 型]` の配列。
  サブグラフの中身は `definitions.subgraphs[]` にある。
- 投入: POST /prompt `{prompt, client_id, front, extra_data: {extra_pnginfo: {workflow}}}`。`front: true` でキューの先頭へ。
  画像の送り込みは POST /upload/image、確認は /view?type=input、フォルダの場所は /internal/folder_paths。
- バックエンドは保存名の `%date:…%` を展開しない（%year% などだけ）。だから想起が投入前に自分で展開する（souki.py の `expand_dates`）。

### v2.3.0 で見つかった不具合（第 1 弾で直す）

- 着替工房（作者の別ツール。ComfyUI のワークフローと専用ノード）で作った絵で、キャラ・服の差し替えが効かない。
  - 原因: souki.py の `_collect_text` と `find_positive_nodes` が、`value` キー（PrimitiveStringMultiline）や StringConcatenate・
    RegexReplace・KigaeKoboPrompt を通ってたどれない。たどれないと「最初の文字入り CLIPTextEncode」に逃げるので、
    **ネガティブ**を拾っていた（差し替えの語がネガティブに入る。台帳のポジティブ欄にもネガティブが入る）。
  - 専用ノード版は SaveImage の filename_prefix が線（KigaeKoboPrompt の「保存名」出口）なのに、`has_save_prefix` が文字の値しか見ないので、
    保存名のひな形が効かない。
- 再現用のワークフローは tests/fixtures/kigae_*.json。直す前は、どちらも台帳のポジティブ＝ネガティブの文、
  差し替え先＝ネガティブのノード（専用ノード版は 5、標準版は 16）になる。

### 着替工房の専用ノード KigaeKoboPrompt

- 入力: 品質・画風／キャラ／服（1 行に 1 パターン。行頭の【名前】がラベル。`//` か `#` で始まる行は無効。`{a|b}` はランダム）／
  ポーズ・表情／背景・構図／服の出し方（全部まとめて／番号の行だけ）／番号（control_after_generate つき）／保存名の頭。
- 出口: プロンプト・ラベル・保存名・一覧。OUTPUT_IS_LIST は (True, True, True, False)。
- 保存名の出口は `着替工房/01_メイド` のような形で、ファイルは `着替工房\01_メイド_00001_.png` になる。01 は服の窓の行番号。

### Qwen-Image 2.1

- 公式テンプレート `image_qwen_image_2_1_image_edit.json`（ほかに t2i・turbo・background_removal）。
  ComfyUI のテンプレート集（GitHub の Comfy-Org/workflow_templates）にある。
- サブグラフ「Image Edit (Qwen Image 2.1)」の中に UNETLoader（qwen_image_2.1_int8_convrot.safetensors）、
  CLIPLoader（qwen3vl_8b…、type qwen_image）、VAELoader、QwenImage21Cache、KSampler。
- TextEncodeQwenImage21: 入力 clip, prompt, negative_prompt, vae, resolution と参照画像 `images.image_1`〜（増やせる。テンプレートでは 12 枠）。
  出口 0＝positive、1＝negative、2＝latent。
- プロンプト強化は ComfySwitchNode（on_false／on_true／switch）で、TextGenerate（system_prompt は PrimitiveStringMultiline
  「Text (System Prompt)」）を通すかどうかを切り替える。
- 参照画像は上の階層の LoadImage から images.image_1・image_2 へ入る。ほかに BatchImagesNode、SaveImageAdvanced
  （widgets `['Qwen_image_2.1','png','8-bit','sRGB']`）。
- プロンプトの中では `<image1>` のように枠を指す。image_1 が編集する絵。

### 専用の窓（作者の別ツール「台割工房」と同じ方式）

- msedge.exe を探す: `%ProgramFiles(x86)%`・`%ProgramFiles%`・`%LOCALAPPDATA%` の `Microsoft\Edge\Application\msedge.exe`、
  なければレジストリの App Paths。
- 起動: `msedge --app=URL --user-data-dir=<data>\edge --no-first-run --no-default-browser-check --window-size=幅,高さ`
  （subprocess.Popen に DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP）。Edge が無いときは webbrowser.open。
- 窓の大きさと位置は保存して次に使う。黒い窓を出さないよう pythonw で動かす。

## v2.4.0 で入れたしくみ（souki.py）

- プロンプトのたどり方は `Tracer`（`read_prompts`）。窓は `{node, key, line, title, text, role}`。
  role は窓の名前（_meta.title）から（`window_role`）。ネガティブ側にも出てくる欄は書き換えの対象から外す。
- 差し替えは `build_swap`（窓ごと）。キャラの窓があれば、ほかの窓からは元の子の語だけ外す。
- 着替工房の番号は「お休みの行（// #）を除いた、上から何番目の服か」（1 から）として扱っている。
- 保存名は `save_nodes` / `apply_save_template`（線の保存名は API で文字に、画面用は線を外す）。`{服}` 対応。
- 画面用ワークフローの書き換えは `UIGraph`（名前つきの値・サブグラフの入口・線を外す）。
- 参照画像: 表 `refs`・`ref_files`、控えは data/refs/<sha 先頭2>/<sha>.<拡張子>。`capture_refs`・`restore_refs`。
- 台帳の読み直し: `reparse_ledger`。`PARSE_VERSION` を上げると、次の起動で 1 回読み直す。
- 専用の窓: `serve --app`（起動.bat が pythonw で呼ぶ）。画面は 5 秒ごとに /api/alive、閉じるとき /api/bye。
  `Alive`・`watch_window` が、知らせが途絶えたら仕事の終わりを待って止める。窓の大きさは config.json の "window"。

## テスト

- `python -m unittest discover -s tests -v`（リポの一番上で）。一時フォルダに部品を写し、偽の ComfyUI（tests/fake_comfy.py）と
  想起を立てて API を確かめる。画面は tests/ui_check.js（node の playwright）。`SOUKI_KEEP_TMP=1` で一時フォルダを残す。
- zip は `python tests/tools/make_zip.py`（dist/souki_v<版>.zip）。
- ComfyUI 本体は無いので、/prompt・/upload/image・/view・/object_info・/internal/folder_paths・/souki/* に答える小さな偽サーバを書き、
  `python souki.py serve --port …` を立てて API を確かめる。画面は Playwright で確かめる（クラウドの作業場所には Chromium と一緒に入っている）。
- tests/fixtures/
  - kigae_special.ui.json / kigae_special.api.json … 着替工房の専用ノード版（画面用／API 形式）
  - kigae_standard.ui.json / kigae_standard.api.json … 着替工房の標準ノード版
  - API 形式は画面用から変換したもの（MarkdownNote などフロント専用のノードは除いてある）。中の文は見本の文。
  - qwen21_edit.ui.json / qwen21_edit.api.json … Qwen-Image 2.1 の公式テンプレート（Image Edit）。
    API 形式は tests/tools/ui_to_api.py で作った（サブグラフの中は `"外:中"` の番号に平らにする）。
