// 想起の画面のテスト（Playwright）。tests/test_souki.py から呼ばれる。
//   node tests/ui_check.js http://127.0.0.1:ポート '{"ファイル名": id, ...}'
const { chromium } = require("playwright");

const base = process.argv[2];
const ids = JSON.parse(process.argv[3]);
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
