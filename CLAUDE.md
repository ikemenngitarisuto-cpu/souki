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

| ファイル | 役目 |
|---|---|
| fuerutama/fuerutama.html | 本体（これ 1 枚で動く。外のファイル・通信なし。アイコンも中に埋め込み） |
| fuerutama/起動.bat | Edge の専用の窓（`--app=file:///…`）で開く。Edge が無ければいつものブラウザ |
| fuerutama/README.md | 利用者向けの説明 |
| fuerutama/fuerutama.ico, fuerutama_256.png | アイコン（`python fuerutama/tools/make_icon.py` で作り直す。HTML の中の絵も入れ替わる） |
| fuerutama/tools/check.js | 試験（`node fuerutama/tools/check.js`。Playwright で仕掛け・刃の形・減るもの・全滅・金色・3 万個・2D・読みかえ・キー・スマホ幅） |
| fuerutama/tools/make_zip.py | 納品の zip（`python fuerutama/tools/make_zip.py` → fuerutama/dist/fuerutama_v<版>.zip） |

- 版は fuerutama.html の `VERSION` だけ。README.md の見出しの版もそろえる。
- zip に入れるもの: 起動.bat, fuerutama.html, README.md, fuerutama.ico, fuerutama_256.png（UTF-8 の印つき）。
- 起動.bat: 日本語を含む場所でも開けるよう、PowerShell の `(New-Object System.Uri …).AbsoluteUri` で file:/// の住所を作る
  （Chromium は `--app` の値が ASCII でないと開かない）。専用の設定フォルダは `fuerutama\data\edge`（.gitignore 済み）。
  `--autoplay-policy=no-user-gesture-required` で最初から音が出る。窓の大きさは最初の 1 回だけ渡す（あとは Edge が覚える）。

### しくみ（fuerutama.html）

- 世界は円の半径 R=500 の単位。玉の大きさは 4 段（12・8.5・6・4.1）。「円 1 杯」＝面積の 74%（`PACK`）を物差しにして、
  上限（設定「どこまで増やす」。おまかせは 3 倍、キリのいい数にそろえる）まで増やす。玉の数の上限は `MAXN`＝101,000。
- 増え方: つぶれた玉は必ず k 個に分かれる（上限ちょうどで止める）。生まれた玉は円の中心から外向きにはじき出し（`spawnKids`）、
  `IMM` で仕掛け 1 回分（`mech.cycle`×設定の回数）つぶれない。色は刃が閉じ始める合図 `bev:"close"` ごとに次へ。
- 物理: 升目（玉の直径の大きさ）で近くの玉を探し、重なりを押し戻す（位置で直す方式）。1 こまに 3 歩（細かさ・重さで 2〜4 歩）。
  - いっぱいを超えたら重なりを許す: 升目の玉の数をならした「混み具合」で、混んだほうから空いたほうへ押す（`relax`）。
    1 つの升目に 6 個より多いときは升目の中どうしだけ押し合う（`CELL_SOLO`）。円の 0.9 杯を超えたら、はじける動き（`kickOn`）は止める。
  - 仕掛けは「太さのある線分」の集まり（`SEG` に 1 本 10 個: 両端・太さ・1 歩前の両端・種類 0 押すだけ／1 刃／2 消し刃）。
    1 歩前の位置から面の速さを出して押し返す。速い棒が玉を飛び越えたら、もとの側へ戻す（すり抜け防止）。
  - つぶれる条件: (1) 2 つの面から逆向き（内積 < -0.45。3 枚が 120° で寄るときも入る）に押され、めり込みの合計が半径の 14% を超えた
    (2) 閉じていく刃に押され、まわりの玉との重なりが「直径の 1.2 倍」と「全体の平均の 2.6 倍」の大きいほうを超えた。真ん中の軸ではつぶさない。
- 仕掛け: `Yose`（寄せ刃 N 枚。刃の形は線分の点列 `lp`。ギザギザは 2 枚のとき上下がかみ合う形、3 枚以上は刃の真ん中に歯の先）、
  `Loops`（8 の字 N 個。弧＝半円 2 枚、まっすぐ＝ひし形の半分 2 枚）、`Box`（はさみ箱）。動きは `Seq`（行き先・時間・動き方の段取り）。
- 減るもの（30 個で出てくる。`activateFoe`）: 食べる玉 `EAT`（うろうろ・吸いこみ・食べると大きく。1 秒に食べられる数 `rate` に限りがあり、
  満腹 `full` だと遅くなる。混んだ所で食べ放題にすると決着がつかないため）、穴 `DRN`（ふちのすき間。
  出ていく玉は `ESC`、縁に押すだけの柱）、寿命 `DIE`、消し刃（`setRed`。赤い刃だけに挟まれたら消える、ふつうの刃とも挟まれたら半々）。
  消えた玉は `RM`、つぶれた玉は `CR` に印をつけ、`processEvents` でまとめて片づける。
- 1 回の段取り: intro → grow（増える）→ warn（ふちに白い粒）→ burst（割れる）→ clear → 次の回。金色の回は grow → gold → clear。
  全滅・減るものの勝ちは grow → gone → clear（`foeWins`。名札「〇〇の勝ち」）。30 秒（減るものがいる回は 22 秒）増えないとき・
  N<6 で 14 秒つぶれないときも終わる（そのとき上限の 5% 未満で減るものがいれば、減るものの勝ち）。
- 玉は WebGL の点でまとめて描く（`GLR`。1 個 6 つの値: 場所・大きさ・色・光・濃さ）。使えないときは 2D キャンバスで 1 個ずつ（`drawBalls`）。
- 音はファイルを使わずその場で作る（`AU.synth`）。
- 設定は `OPTS`（見出し・名前・表示・選択肢）から設定の欄を作る。v1.0 の保存値（mech: press／cross）は寄せ刃 2／4 枚に読みかえる。
- 試験用: `?seed=` `?mech=yose|loops|box` `?yn=` `?ys=` `?ln=` `?ls=` `?size=` `?split=` `?limit=`（数も可）`?foe=` `?fl=` `?gold=0|1`
  `?gl=0` `?pause=1`、`window.FT`（state・run・next・pause）。
