// ふえる玉の確かめ。リポの一番上で:  node fuerutama/tools/check.js
// 終わりに「合格」と出れば OK（不合格の数が終了コードになる）。FT_SHOTS=フォルダ で画面の写しを残す。
// クラウドの作業場所などに入っている Playwright（require('playwright')）と Chromium を使う。
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const HTML = 'file://' + path.resolve(__dirname, '..', 'fuerutama.html');
const SHOTS = process.env.FT_SHOTS || '';
let failures = 0;
const ok = (cond, msg) => { console.log((cond ? '  ok   ' : '  NG   ') + msg); if (!cond) failures++; };

async function open(browser, query, vp) {
  const page = await browser.newPage({ viewport: vp || { width: 540, height: 960 } });
  const errs = [];
  page.on('pageerror', e => errs.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto(HTML + (query ? '?' + query : ''));
  await page.waitForTimeout(300);
  return { page, errs };
}

// 1 回ぶんを最後まで進めて、通った段取りを返す
async function oneRound(page, limit) {
  return page.evaluate((limit) => {
    FT.pause(true);
    const seen = [], first = {};
    let t = 0, maxN = 0, s = FT.state();
    while (t < limit) {
      s = FT.run(0.25); t += 0.25;
      if (seen[seen.length - 1] !== s.phase) seen.push(s.phase);
      if (s.phase === 'grow' || s.phase === 'full' || s.phase === 'gold') maxN = Math.max(maxN, s.N);
      for (const n of [10, 100, 1000]) if (first[n] === undefined && s.N >= n) first[n] = t;
      if (s.rounds > 1) break;
    }
    return { seen, maxN, first, t, last: s };
  }, limit);
}

(async () => {
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });

  console.log('■ 4 つの仕掛けが、1 個から増えて、終わって、次の回へ進む');
  for (const [mech, seed] of [['press', 101], ['cross', 102], ['loops', 103], ['box', 104]]) {
    const { page, errs } = await open(browser, `seed=${seed}&mech=${mech}&size=m&split=3`);
    const before = await page.evaluate(() => FT.state());
    const r = await oneRound(page, 150);
    const order = r.seen.join(' → ');
    ok(before.mech === mech && before.cap > 1500, `${mech}: 中の玉で上限 ${before.cap} 個`);
    ok(r.maxN >= before.target * 0.99, `${mech}: 上限まで増えた（最大 ${r.maxN} 個、1,000 個まで ${r.first[1000]} 秒）`);
    ok(r.first[1000] !== undefined && r.first[1000] < 70, `${mech}: 増え方が遅すぎない`);
    ok(/grow → (full → warn → burst|gold) → clear → intro/.test(order), `${mech}: 段取り ${order}`);
    ok(r.last.rounds === 2, `${mech}: 次の回が始まった（${r.t} 秒で 1 回）`);
    ok(errs.length === 0, `${mech}: エラーなし ${errs.length ? JSON.stringify(errs.slice(0, 3)) : ''}`);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, `check_${mech}.png`) });
    await page.close();
  }

  console.log('■ 極小の玉は 10,000 個ちょうどで金色になる');
  {
    const { page, errs } = await open(browser, 'seed=105&mech=press&size=t&split=3', { width: 1280, height: 720 });
    const r = await page.evaluate(() => {
      FT.pause(true); let s, t = 0;
      while (t < 150) { s = FT.run(0.25); t += 0.25; if (s.phase === 'gold') break; }
      return { s, t };
    });
    ok(r.s.phase === 'gold' && r.s.N === 10000, `金色: ${r.s.phase} ${r.s.N} 個（${r.t} 秒）`);
    const st = await page.evaluate(() => { FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 1500)); });
    ok(st.gl === true, 'WebGL で描いている');
    ok(st.fps >= 10, `1 万個でも動いている（この環境で ${st.fps} こま/秒、計算 ${st.physMs} ms）`);
    ok(errs.length === 0, `エラーなし ${errs.length ? JSON.stringify(errs.slice(0, 3)) : ''}`);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'check_gold.png') });
    await page.close();
  }

  console.log('■ WebGL が使えないときも描ける');
  {
    const { page, errs } = await open(browser, 'seed=106&mech=cross&size=s&gl=0');
    const s = await page.evaluate(() => { FT.pause(true); FT.run(20); FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 600)); });
    ok(s.gl === false && s.N > 50, `2D で描く（${s.N} 個）`);
    ok(errs.length === 0, `エラーなし ${errs.length ? JSON.stringify(errs.slice(0, 3)) : ''}`);
    await page.close();
  }

  console.log('■ キーと画面の大きさ');
  {
    const { page, errs } = await open(browser, 'seed=107&mech=auto', { width: 390, height: 844 });
    await page.mouse.click(195, 400);
    const r0 = await page.evaluate(() => FT.state());
    await page.keyboard.press('n');
    const r1 = await page.evaluate(() => FT.state());
    ok(r1.rounds === r0.rounds + 1, 'N で次の回へ');
    await page.keyboard.press(' ');
    ok((await page.evaluate(() => FT.state())).paused === true, 'スペースで止まる');
    await page.keyboard.press(' ');
    ok((await page.evaluate(() => FT.state())).paused === false, 'もう一度スペースで動く');
    await page.keyboard.press('2');
    const r2 = await page.evaluate(() => FT.state());
    ok(r2.mech === 'cross' && r2.setMech === 'cross', '2 で十字になる');
    await page.keyboard.press('0');
    ok((await page.evaluate(() => FT.state())).setMech === 'auto', '0 でおまかせに戻る');
    const m0 = (await page.evaluate(() => FT.state())).mute;
    await page.keyboard.press('m');
    ok((await page.evaluate(() => FT.state())).mute === !m0, 'M で消音が切りかわる');
    await page.keyboard.press('m');
    const lay = await page.evaluate(() => {
      const b = document.querySelector('#bar').getBoundingClientRect();
      return { sw: document.documentElement.scrollWidth, iw: innerWidth, l: b.left, r: b.right, h: b.height };
    });
    ok(lay.sw <= lay.iw && lay.l >= 15 && lay.r <= lay.iw - 15, `スマホ幅で横にはみ出さない（帯 ${Math.round(lay.l)}〜${Math.round(lay.r)}）`);
    ok(lay.h < 60, `スマホ幅でも操作の帯が 1 段（高さ ${Math.round(lay.h)}）`);
    ok(errs.length === 0, `エラーなし ${errs.length ? JSON.stringify(errs.slice(0, 3)) : ''}`);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'check_phone.png') });
    await page.close();
  }

  await browser.close();
  console.log(failures ? `不合格 ${failures} 件` : '合格');
  process.exit(failures);
})().catch(e => { console.error(e); process.exit(99); });
