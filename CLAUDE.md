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
| fuerutama/ | 想起とは別のソフト「ふえる玉」（いちばん下の節）。想起の zip には入れない |

設定は config.json、台帳やログは data/、Python 環境は runtime/（どれもリポに入れない。.gitignore 済み）。
サーバの既定ポートは 8765（`serve --port` で変えられる）、ComfyUI の既定は http://127.0.0.1:8188。

## いまの状態と次の作業

- 今の版は v2.5.0（docs/仕様_v2.4.md の第 1 弾・第 2 弾ともに済み）。
- 次の作業は作者と決める（仕様書にあるものは全部作った）。書いていないことで迷ったら作者に聞く。
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
- タスクバー（v2.4.2）: Edge（Chromium）は --app の窓に AppUserModelID「MSEdge.<ホスト>_<パス>.<設定フォルダ名>.Default」を付け
  （chrome/browser/shell_integration_win.cc の GetAppUserModelIdForApp・web_app_helpers.cc の GenerateApplicationNameFromURL。
  ポートは入らない。設定フォルダ名は英数字だけ残す）、同じ名札の窓をタスクバーでまとめる。台割工房（127.0.0.1:<ポート>/ と edge）と
  分けるため、想起は `/souki`（→ / へ 302）と `data\souki_edge` で開く（`app_window_url`・`edge_profile_dir`・`chromium_app_id`）。
  2 段目として `watch_window_identity` が、題が「想起」で始まる Edge の専用窓に名札 `Souki.Ledger`・起動.bat・souki.ico を付け、
  Edge に戻されたら付け直す（Edge はプロファイルのアイコンができたときに付け直す）。Windows の呼び出しは作業環境では動かせない
  （`identity_pass` の流れだけ偽物でテストしている）。
- Python の控え（v2.4.2）: zip の日時が版ごとに同じだと、版の数字しか変わらない chara.py・vision.py（大きさも同じ）で
  Python が古い .pyc を使う。`fresh_bytecode`（souki.py）と `_fresh_vision_bytecode`（chara.py）が、控えを中身（ハッシュ）で
  確かめる形（CHECKED_HASH）に作り直す。make_zip.py は日時を「作った時刻」にする。

## v2.5.0 で入れたしくみ（souki.py）

- 差し替えの記録: 投入のとき extra_pnginfo の `souki`（`prepare` の `record`）。台帳では表 `lineage`（`lineage_rows`・`write_lineage`）。
  元の絵は `resolve_image`（場所 → 絵柄と名前 → 名前と大きさ）で探す。`made_from`・`made_here`。参照画像の元の絵も同じ表（kind="ref"）。
- 2 人以上の絵: `build_swap(child=…)`。2 人以上（`is_multi_prompt` か顔が 2 人以上）なら、その子の特徴語・トリガー語・
  台帳から学んだその子の語だけ外す（体の語をまとめて外さない）。保存名は `save_name_for` がその子の名前だけ入れ替える。
- LoRA: 表 `c_lora`（手で決めたもの）。無ければ `infer_lora`（その子の絵の 3 割以上・全体の 2 倍以上に偏る LoRA、
  トリガー語は LoRA を使った絵の 8 割以上・全体の 3 倍以上に偏る語。髪・目・服・体の語は除く）。`character_lora` が使う方を返す。
  入れ替えは `plan_lora_swap`（入れ替え／強さ 0 で止める／Power Lora Loader に足す／`_model_loader` のすぐ後ろに LoraLoader を足す）。
  画面用ワークフローには `UIGraph.apply_ops`（set・power・power_add・insert_lora）で写す。新しいノードの番号は画面用の最大より大きくする。
  Power Lora Loader は API で `lora_N: {on, lora, strength}`、画面用は widgets_values の中の dict（`PowerLoraLoaderHeaderWidget` の後ろ）。
- 参照画像の差し替え: `slot_info`（枠の子は、控えの元の絵のキャラ）・`ref_candidates`・`plan_ref_swaps`（おまかせは文章の絵だけ）・
  `ensure_ref_uploads`（`souki_<sha 16 桁>.png` で送る）。文章のプロンプト（`prose`）は語を入れ替えず、LoRA も触らない。
- まとめて差し替え: `api_swap_batch`（/api/swap/batch）。1 人ずつ `run_submit`。全員同じ seed は 1 つの seed を "fixed" で使う。

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

## 別のソフト: ふえる玉（fuerutama/）

作者が X で見た「円の中で玉が挟まれて分かれ、増えていく」動画（#satisfying 系）を、延々と眺めるためのソフト。
作者と決めたこと: ブラウザで開く HTML 1 枚、想起のリポの別フォルダに置く、名前は「ふえる玉」（仮）。想起とはつながっていない。
v1.1 で作者と決めたこと: 増え方は理論どおり（上限まで重なりながら増やす）、刃の数・形を選べる（プレス＋十字＝寄せ刃）、
減るもの 4 つ（食べる玉・穴・寿命・消し刃）、生まれた玉ははじき出して次の 1 回は守る、込み入ったものは設定に逃がす。
v2.0 で作者と決めたこと（fuerutama/docs/仕様_v2.0.md）: 動き・見た目・音・設定を参考動画に寄せて一から作り直す（上の決めごとは残す）。
減るものは「行ったり来たりの勝負」にする（食べる玉が育つ・穴が開いたり閉じたり・寿命が見える・消し刃の赤が広がる）。

| ファイル | 役目 |
|---|---|
| fuerutama/fuerutama.html | 本体（これ 1 枚で動く。外のファイル・通信なし。アイコンも中に埋め込み） |
| fuerutama/起動.bat | Edge の専用の窓（`--app=file:///…`）で開く。Edge が無ければいつものブラウザ |
| fuerutama/README.md | 利用者向けの説明 |
| fuerutama/docs/仕様_v2.0.md | v2.0 の仕様（作者と決めたこと・作ってみて足したこと） |
| fuerutama/fuerutama.ico, fuerutama_256.png | アイコン（`python fuerutama/tools/make_icon.py` で作り直す。HTML の中の絵も入れ替わる） |
| fuerutama/tools/check.js | 試験（`node fuerutama/tools/check.js`。Playwright で 仕掛けと刃の形・動きの芯・守り・混んだとき・減るもの 4 つ・全滅と記録・金色・3 万個・描き方・設定の引き継ぎ・キーとスマホ幅。`FT_BIG=1` で 10 万個も） |
| fuerutama/tools/make_zip.py | 納品の zip（`python fuerutama/tools/make_zip.py` → fuerutama/dist/fuerutama_v<版>.zip） |

- 版は fuerutama.html の `VERSION` だけ。README.md の見出しの版もそろえる。
- zip に入れるもの: 起動.bat, fuerutama.html, README.md, fuerutama.ico, fuerutama_256.png（UTF-8 の印つき。docs/・tools/ は入れない）。
- 起動.bat: 日本語を含む場所でも開けるよう、PowerShell の `(New-Object System.Uri …).AbsoluteUri` で file:/// の住所を作る
  （Chromium は `--app` の値が ASCII でないと開かない）。専用の設定フォルダは `fuerutama\data\edge`（.gitignore 済み）。
  `--autoplay-policy=no-user-gesture-required` で最初から音が出る。窓の大きさは最初の 1 回だけ渡す（あとは Edge が覚える）。

### しくみ（fuerutama.html。v2.0）

ファイルの中は、見出しのコメントで分かれている（上から: 設定の土台／世界（玉と当たり）／仕掛け／減るもの／1 回ぶんの進め方／画面／音／操作）。

- 世界は円の半径 R=500 の単位。玉の半径は `SIZES`（特大 20・大 13・中 9・小 6.3・極小 4.2）。玉は生まれた順に並べ（うしろほど新しい＝上に描く）、
  消えた玉は印をつけて、こまの終わりに `compact` で詰める。`ST` の印: 1 消えた／2 挟まれた（はじける前の白いかたまり）／4 穴から出ていく途中／8 消し刃で消える／16 寿命でしぼんでいる。
- 増え方: 「円 1 杯」＝面積の 74%（`PACK`）。決めた数 `RD.target`（おまかせは 3 倍。キリのいい数にそろえる）ちょうどまで。玉の数の上限 `MAXN`＝101,000。
- 動きの芯: 仕掛けが合図を出す（`rigEvent`）。close（寄りはじめ。生まれる玉の色を次へ）→ seal（囲いこんだ）→ impact（閉じきった）。
  別の組（`grp`）の刃 2 枚に逆向きに押された玉（`contact` → `pinch`）に「挟まれた」印がつき、`doomedPass` で `rig.lump` に集まる。
  impact で `splitAll` がいっせいに分け、`rig.exitInfo` の出口から `spawnBlob`（かたまり）か `spawnInPlace`（プレス。すき間に沿って）で生む。
  生まれたては `SC0` 倍の大きさで、`GROW_T` 秒でふくらむ。守り: `IMM`（`guardTime`。12 個までは守らず、40 個までにだんだん長く）。
  守られた玉が挟まれると `GHO`（しばらく刃をすり抜ける）。刃が開いていくとき（`rig.live` が偽）に挟まった玉も同じ。
- 物理: 升目（当たりの直径の大きさ）で近くの玉を探し、重なりを押し戻す（位置で直す）。1/180 秒きざみ（設定で 1/120・1/240。重いと自動で 1/60 まで下げる）。
  - 混んだら、当たりの半径 `RC` を数に合わせて縮める（`updateRadius`。詰め具合 `PHI_C`=0.8 を保つ）。見た目の半径 `RV` は設定「混んできたら」しだい（重なる＝縮めない）。
  - `spread`: 粗い升目で混み具合を見て、差の大きい所だけ、混んだほうから空いたほうへ少しずつ流す（升目のあいだはなめらかにつなぐ）。
    これがないと、刃が通ったあとに黒いすき間が残る。円がいっぱいに近いとき（当たりで 5 割以上）だけ働く。
  - 玉どうしのはね返り `E_BB` は 1（勢いを減らさない）。0.9 だと、混んだ所でぶつかるたびに勢いを失い、ふちに固まった輪になって仕掛けへ来なくなった（増えが止まる）。
  - 円のふちではね返るたびに、少し真ん中へ向きを寄せる（まん丸の器では、ふち寄りを回る玉が真ん中へ来ないため）。
  - 重さ: 詰まった升目は間引く（`CELL_MAX`・`CELL_SOLO`）。5 万個より上は、升目を市松もようにして 1 歩おきに調べる。
  - 出だしの助け: 玉が少ないうちは、仕掛けが玉を待ってから閉じる（`noPrey`。8 の字はゆっくり回る）。4 秒ひとつも挟まれなければ、`herd` が玉を囲いの真ん中（`rig.bait`）へ寄せる。
- 仕掛け（`Rig` を継ぐ。刃は太さのある線分 `SEG` で、1 本 12 個: 両端・太さ・1 歩前の両端・種類 0 押すだけ／1 刃／2 消し刃・組・刃の番号。速い刃が玉を飛び越えたら、もとの側へ戻す）:
  - `Yose`（寄せ刃）: 2 枚はプレス（待つ → 閉じる → すぐ開く）。3 枚以上は円の外から寄り、`dLock` で多角形に閉じ、締める。
    玉が増えると、刃は `up`（浮いていて当たらない。うすく描く）のまま寄ってきて、`grabAt` の所で降りる（全部かき集めると増え方が速すぎ、円の中が空っぽに見える）。
  - `Loops`（8 の字）: 2 組の弧が逆向きに回り続ける。角度 `phi` が `stp` をまたぐと輪ができ、そこから `thc` 回るとつぶれる。
  - `Box`（はさみ箱）: 動きの段取り `Phases` で、すべって入る → 羽根が折れる → 横に締まる → 逆の順で戻る。
  - どれも `heat`（白く光る度合い）・`contains`・`exitFor`・`exitInfo`・`lump`・`vise`（挟まれた食べる玉の置き場所とつぶれる向き）・`bait`・`prey` を持つ。
- 減るもの（`makeFoe`。強さごとの数字は `FOE_TUNE`）。どれも終わりがある（`foeOver`）。玉が 0 個になれば `foeWins`:
  - `Eater`（食べる玉）: 食べる速さとおなかの大きさは、出てきたときの玉の数に合わせる。吸いこんで、触れた玉を食べる（`feed`）。小さいうちは囲いからとび出してよける（`hop`）。
    おなかいっぱい（`full`）になると `rig.bait` へ寄っていき、挟まれて（trap）、impact ではじけ（`pop`）、食べた玉の何割かを出口から吐き出す。
  - `Hole`（穴）: `GAPS` を ひび → 開く → 閉じる。2 回目からは「決めた数の何割（`marks`）」まで増えてから開く（増えなければ 12 秒で）。穴へ吸い出す風（混んでいるほど強い）。開いている間は `cam.zoom` を 0.9 に。
  - `Life`（寿命）: `DIE`。長さは「挟まれる割合 `RD.fhat`・分かれる数・守り」から逆算し、時間とともに長くする。色あせは `LIFE_L` で描く。
  - `RedTide`（消し刃）: `RED.set` に 1 枚ずつ足して、引いていく。いちばん多いときの枚数は、分かれる数から決める（赤の割合が 1 − 1/分かれる数 を超えると減る）。
  - ケンカ中は、相手が強いほど仕掛けが玉を待たない（`ASSIST`）。勝ち負けは 20 回ずつ数えて合わせた（減るものの勝ち: 弱い 0%・ふつう 0〜15%・強い 25〜40%）。
- 1 回の段取り（`RD.phase`）: intro → grow → warn（ふちに白い粒）→ burst（割れる）→ clear → 次の回。金色は grow → gold → clear。全滅は grow → gone → clear。
  増えも減りもしなくなったら fizzle（ケンカ中は打ち切らない）。組み合わせは `planRound`（同じものが続かないよう袋から引く。📌 の固定は `SET.pin`）。
  特大の玉は 8 の字 4 つまで。小・極小のおまかせは「2 つに分かれる」を出さない（1 回が 2 分近くになるため）。
- 絵: 玉は WebGL でまとめて描く（`GLR`。1 個 8 つの値で、四角を玉の数だけ描く。影・黒いふちどり・光の点・白く光る・色あせ・赤く焼ける・光のにじみ）。
  GPU のない PC（WebGL がソフトウェア描画になる）では、2D で色ごとの絵を 1 個ずつ貼る（`drawBalls2D`。そのほうが何倍も速い）。`?gl=1` で WebGL を無理に使う、`?gl=0` で 2D。
  円・進み具合の輪・刃・減るもの・演出（`FX`）は、上に重ねたキャンバスに描く。
- 音（`AU`）: ファイルを使わずその場で作る。はじける音は回ごとに音程が上がる（`SCALE` を 3 段。15 回でひとまわり）。粒の音は、数が増えると雑音 1 本にまとめる。
  仕掛けが動く低い音、閉じる音、割れる前に上がっていく音、穴から噴き出す音。
- 設定は `OPTS`（見出し・名前・表示・選択肢）から欄を作る。プリセット `PRESETS`、記録 `REC`（localStorage の "fuerutama.rec"）。v1.0 の保存値（mech: press／cross）は寄せ刃 2／4 枚に読みかえる。
- 試験用: `?seed=` `?mech=yose|loops|box` `?yn=` `?ys=` `?ln=` `?ls=` `?size=` `?split=` `?speed=` `?limit=`（数も可）`?crowd=` `?guard=` `?eject=` `?foe=` `?fl=` `?q=`
  `?gold=0|1` `?gl=0|1` `?pause=1` `?smart=0` `?rec=1`、`window.FT`（state・run・next・pause・rec）。`FT.run(秒)` は 1/60 秒きざみで進む（それより短い値では進まない）。
- クラウドの作業場所は GPU がないので、見た目を確かめるときは `?gl=1` をつける（つけないと 2D の絵になる）。
  音は耳で聞けないので、`?rec=1` で開いて `FT.rec(ミリ秒)` で録り、ffmpeg で波形と周波数を見る。
