// 想起の画面のテスト（Playwright）。tests/test_souki.py から呼ばれる。
//   node tests/ui_check.js http://127.0.0.1:ポート '{"files": {"ファイル名": id, ...}, "chars": {"名前": id}, "made": [元の絵, できた絵]}'
const { chromium } = require("playwright");

const base = process.argv[2];
const arg = JSON.parse(process.argv[3]);
const ids = arg.files, chars = arg.chars;
const fail = (m) => { console.error("NG: " + m); process.exitCode = 1; };

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  let alive = 0;
  page.on("request", (r) => { if (r.url().endsWith("/api/alive")) alive++; });

  // 1. 参照画像が並ぶ（Qwen-Image 2.1 の絵）
  await page.goto(`${base}/?image=${ids["Qwen_image_2.1_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });
  await page.waitForSelector(".refs .ref img", { timeout: 15000 });
  const refs = await page.$$eval(".refs .ref", (els) => els.map((e) => e.querySelector(".ref-slot").textContent));
  if (refs.join(",") !== "image_1,image_2") fail("参照画像の枠: " + refs.join(","));
  const src = await page.$$eval(".refs .ref button", (els) => els.map((e) => e.textContent));
  if (src.length !== 1) fail("元の絵のボタン: " + src.length);
  if ((await page.$$(".taglist .tag")).length && (await page.textContent("#dbody")).includes("TAG ("))
    fail("文章のプロンプトなのにタグが出ている");

  // 2. 窓ごとの下見（着替工房の標準ノード版）
  await page.goto(`${base}/?image=${ids["標準_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#swapsel option").length > 1, null, { timeout: 15000 });
  await page.selectOption("#swapsel", "ピナ");
  await page.waitForFunction(() => {
    const b = document.querySelector("#swapprev");
    return b && b.style.display === "block" && b.textContent.includes("窓：");
  }, null, { timeout: 15000 });
  const prev = await page.textContent("#swapprev");
  if (!prev.includes("キャラ")) fail("下見に窓の名前が出ていない: " + prev);
  if (!prev.includes("+ pink hair")) fail("下見に入れる語が出ていない: " + prev);
  if (prev.includes("ネガティブ")) fail("ネガティブの窓が下見に出ている");

  // 4. 2 人の絵: 差し替える子を選べる（いちばん大きく写っている子が最初）
  await page.goto(`${base}/?image=${ids["2025-3-14illuだれか_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#childseg button").length === 2, null, { timeout: 15000 });
  const kids = await page.$$eval("#childseg button", (els) => els.map((e) => e.textContent + (e.classList.contains("on") ? "*" : "")));
  if (kids.join(",") !== "ピナ,ルル*") fail("差し替える子: " + kids.join(","));

  // 5. LoRA の入れ替えが下見に出る
  await page.goto(`${base}/?image=${ids["lora_ルル_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#swapsel option").length > 1, null, { timeout: 15000 });
  await page.selectOption("#swapsel", "ピナ");
  await page.waitForFunction(() => (document.querySelector("#swapprev") || {}).textContent.includes("LoRA"), null, { timeout: 15000 });
  const lp = await page.textContent("#swapprev");
  if (!lp.includes("入れ替え: rulu_v1 → pina_xl")) fail("LoRA の下見: " + lp);
  if ((await page.$$("#childseg button")).length) fail("1 人の絵に「差し替える子」が出ている");

  // 6. 参照画像の枠（Qwen）: おまかせで差し替え先の子の絵、ほかの絵も選べる
  await page.goto(`${base}/?image=${ids["Qwen_image_2.1_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#refrow .refsel").length === 2, null, { timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll("#swapsel option").length > 1, null, { timeout: 15000 });
  await page.selectOption("#swapsel", "ピナ");
  await page.waitForFunction(() => (document.querySelector("#swapprev") || {}).textContent.includes("image_1:"), null, { timeout: 15000 });
  const rp = await page.textContent("#swapprev");
  if (!rp.includes("lora_ピナ_00001_.png") && !rp.includes("ピナ")) fail("参照画像の下見: " + rp);
  await page.selectOption("#refrow .refsel:nth-child(3) select", "pick");          // image_2（服の枠）
  await page.waitForSelector(".mdl .refgrid", { timeout: 15000 });
  const nc = (await page.$$(".mdl .refcand")).length;
  if (nc) {
    await page.click(".mdl .refcand");
    await page.waitForFunction(() => (document.querySelector("#swapprev") || {}).textContent.includes("image_2:"), null, { timeout: 15000 });
  } else {
    await page.click(".mdl .mdl-b button");
  }

  // 7. まとめて差し替えの欄
  await page.click("#batchtoggle");
  await page.waitForFunction(() => document.querySelectorAll("#batchchars input").length > 0, null, { timeout: 15000 });
  if (!(await page.isDisabled("#batchgo"))) fail("誰も選んでいないのに投入できる");
  await page.check("#batchchars input >> nth=0");
  if (await page.isDisabled("#batchgo")) fail("選んだのに投入できない");
  if (!(await page.textContent("#batchgo")).includes("1 人")) fail("人数が出ていない");

  // 8. つながり（元の絵 ⇄ この絵から作った絵）
  if (arg.made) {
    await page.goto(`${base}/?image=${arg.made[0]}`);
    await page.waitForSelector("#drawer.open", { timeout: 15000 });
    await page.waitForSelector(".lineage .lthumb", { timeout: 15000 });
    await page.goto(`${base}/?image=${arg.made[1]}`);
    await page.waitForSelector("#drawer.open", { timeout: 15000 });
    await page.waitForSelector(".lineage .lrow button", { timeout: 15000 });
  }

  // 9. キャラ帳: LoRA の欄（台帳から推し量った LoRA が入っている）
  await page.goto(`${base}/chara#/char/${chars["ルル"]}`);
  await page.waitForSelector(".lorabox input.txt", { timeout: 20000 });
  const lv = await page.$eval(".lorabox input.txt", (e) => e.value);
  if (lv !== "chara/rulu_v1.safetensors") fail("キャラ帳の LoRA: " + lv);
  await page.goto(`${base}/?image=${ids["標準_00001_.png"]}`);
  await page.waitForSelector("#drawer.open", { timeout: 15000 });

  // 3. キャラ帳へは同じ窓で（新しいタブを開かない）
  const pages0 = browser.contexts()[0].pages().length;
  await page.evaluate(() => document.querySelector("#kaolink").click());
  await page.waitForURL(/\/chara/, { timeout: 15000 });
  if (browser.contexts()[0].pages().length !== pages0) fail("キャラ帳が別のタブで開いた");
  await page.waitForTimeout(500);

  if (alive < 2) fail("画面が開いていることを知らせていない: " + alive);
  if (errors.length) fail("画面のエラー: " + errors.join(" / "));
  await browser.close();
  if (!process.exitCode) console.log("OK");
})().catch((e) => { console.error(e); process.exit(1); });
