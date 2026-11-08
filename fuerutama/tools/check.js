// ふえる玉の確かめ。リポの一番上で:  node fuerutama/tools/check.js
// 終わりに「合格」と出れば OK（不合格の数が終了コードになる）。FT_SHOTS=フォルダ で画面の写しを残す。
// クラウドの作業場所などに入っている Playwright（require('playwright')）と Chromium を使う。
const { chromium } = require('playwright');
const path = require('path');

const HTML = 'file://' + path.resolve(__dirname, '..', 'fuerutama.html');
const SHOTS = process.env.FT_SHOTS || '';
let failures = 0;
const ok = (cond, msg) => { console.log((cond ? '  ok   ' : '  NG   ') + msg); if (!cond) failures++; };

async function open(browser, query, vp, init) {
  const page = await browser.newPage({ viewport: vp || { width: 540, height: 960 } });
  if (init) await page.addInitScript(init);
  const errs = [];
  page.on('pageerror', e => errs.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto(HTML + '?pause=1&' + (query || ''));
  await page.waitForTimeout(200);
  return { page, errs };
}
const noErr = (errs, what) => ok(errs.length === 0, `${what}: エラーなし ${errs.length ? JSON.stringify(errs.slice(0, 3)) : ''}`);

// 1 回ぶんを最後まで進めて、通った段取りを返す
async function oneRound(page, limit) {
  return page.evaluate((limit) => {
    const seen = [], first = {};
    let t = 0, maxN = 0, s = FT.state();
    while (t < limit) {
      s = FT.run(0.25); t += 0.25;
      if (seen[seen.length - 1] !== s.phase) seen.push(s.phase);
      if (s.phase === 'grow' || s.phase === 'gold' || s.phase === 'warn') maxN = Math.max(maxN, s.N);
      for (const n of [10, 100, 1000]) if (first[n] === undefined && s.N >= n) first[n] = t;
      if (s.rounds > 1) break;
    }
    return { seen, maxN, first, t, last: s };
  }, limit);
}

(async () => {
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });

  console.log('■ 仕掛けと刃の形: 1 個から上限ちょうどまで増えて、割れて、次の回へ（上限は「円の 2 倍」）');
  const cases = [['yose', 'yn=2&ys=bar', 101], ['yose', 'yn=4&ys=saw', 102], ['yose', 'yn=6&ys=arc', 103], ['yose', 'yn=8&ys=chev', 104],
                 ['loops', 'ln=2&ls=arc', 105], ['loops', 'ln=3&ls=bar', 106], ['box', '', 107]];
  for (const [mech, extra, seed] of cases) {
    const name = `${mech}${extra ? ' ' + extra : ''}`;
    const { page, errs } = await open(browser, `seed=${seed}&mech=${mech}&${extra}&size=m&split=3&foe=none&limit=x2&gold=0`);
    const before = await page.evaluate(() => FT.state());
    const r = await oneRound(page, 160);
    ok(before.target > before.cap * 1.8, `${name}: 上限 ${before.target} 個（円 1 杯 ${before.cap} 個の約 2 倍）`);
    ok(r.maxN === before.target, `${name}: 上限ちょうどまで増えた（最大 ${r.maxN} 個、1,000 個まで ${r.first[1000]} 秒）`);
    ok(/grow → warn → burst → clear → intro/.test(r.seen.join(' → ')), `${name}: 段取り ${r.seen.join(' → ')}`);
    ok(r.last.rounds === 2, `${name}: 次の回が始まった（${r.t} 秒で 1 回）`);
    noErr(errs, name);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, `check_${mech}_${seed}.png`) });
    await page.close();
  }

  console.log('■ 生まれた玉は、次に刃が閉じる 1 回はつぶれない（守る）');
  {
    const { page, errs } = await open(browser, 'seed=111&mech=yose&yn=2&size=m&split=3&foe=none&guard=1');
    const r = await page.evaluate(() => {
      let s; for (let i = 0; i < 40; i++){ s = FT.run(.25); if (s.N > 1) break; }
      const born = []; for (let i = 0; i < N; i++) born.push(+(IMM[i] - BORN[i]).toFixed(2));
      return { N, born, cycle: +mech.cycle.toFixed(2) };
    });
    ok(r.N === 3 && r.born.every(v => Math.abs(v - (r.cycle + .1)) < .02), `守る時間＝仕掛け 1 回分（${r.cycle} 秒）: ${JSON.stringify(r.born)}`);
    noErr(errs, '守る');
    await page.close();
  }

  console.log('■ 減るもの 4 つ: 30 個で出てきて、玉を減らす');
  for (const [foe, stat, q] of [['eater', 'eaten', 'mech=yose&yn=2'], ['hole', 'drained', 'mech=loops&ln=3&fl=3'], ['life', 'expired', 'mech=yose&yn=4'], ['red', 'erased', 'mech=yose&yn=4&fl=3']]) {
    const { page, errs } = await open(browser, `seed=121&${q}&size=m&split=3&foe=${foe}&gold=0`);
    const r = await page.evaluate((stat) => {
      let s, t = 0, onAt = null;
      while (t < 90){ s = FT.run(.5); t += .5; if (s.foeOn && onAt === null) onAt = { t, N: s.N }; if (s[stat] > 30 || s.rounds > 1) break; }
      return { s, t, onAt };
    }, stat);
    ok(r.onAt && r.onAt.N >= 30 && r.onAt.N < 200, `${foe}: ${r.onAt ? r.onAt.N + ' 個のときに出てきた' : '出てこない'}`);
    ok(r.s[stat] > 30, `${foe}: 玉が減っている（${stat} ${r.s[stat]} 個、${r.t} 秒）`);
    noErr(errs, foe);
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, `check_foe_${foe}.png`) });
    await page.close();
  }

  console.log('■ 全滅したら、次の回へ');
  {
    const { page, errs } = await open(browser, 'seed=131&mech=yose&yn=4&size=m&foe=eater');
    const r = await page.evaluate(() => {
      FT.run(6);
      for (let i = 0; i < N; i++) RM[i] = 1;    // 全部食べられたことにする
      const seen = []; let s, t = 0;
      while (t < 10){ s = FT.run(.1); t += .1; if (seen[seen.length - 1] !== s.phase) seen.push(s.phase); if (s.rounds > 1) break; }
      return { seen, rounds: s.rounds };
    });
    ok(r.seen.join(' → ').startsWith('gone → clear → intro') && r.rounds === 2, `全滅: ${r.seen.join(' → ')}`);
    noErr(errs, '全滅');
    await page.close();
  }

  console.log('■ 極小の玉の金色は、上限ちょうど（1 万個）');
  {
    const { page, errs } = await open(browser, 'seed=141&mech=yose&yn=2&size=t&split=3&foe=none&limit=10000&gold=1', { width: 1280, height: 720 });
    const r = await page.evaluate(() => { let s, t = 0; while (t < 150){ s = FT.run(.25); t += .25; if (s.phase === 'gold') break; } return { s, t }; });
    ok(r.s.phase === 'gold' && r.s.N === 10000, `金色: ${r.s.phase} ${r.s.N} 個（${r.t} 秒）`);
    const st = await page.evaluate(() => { FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 1500)); });
    ok(st.gl === true, 'WebGL で描いている');
    ok(st.fps >= 8, `1 万個でも動いている（この環境で ${st.fps} こま/秒、計算 ${st.physMs} ms）`);
    noErr(errs, '金色');
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'check_gold.png') });
    await page.close();
  }

  console.log('■ 3 万個（円の約 3 倍に重なる）でも止まらない');
  {
    const { page, errs } = await open(browser, 'seed=151&mech=loops&ln=3&size=t&split=4&foe=none&limit=30000&gold=0', { width: 1280, height: 720 });
    const r = await page.evaluate(() => { let s, t = 0; while (t < 150){ s = FT.run(.25); t += .25; if (s.phase !== 'grow' && s.phase !== 'intro') break; }
      const a = performance.now(); for (let i = 0; i < 10; i++) substep(HS); return { s, t, ms: +((performance.now() - a) / 10).toFixed(1) }; });
    ok(r.s.N === 30000, `3 万個まで増えた（${r.t} 秒、計算 1 歩 ${r.ms} ms）`);
    noErr(errs, '3 万個');
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'check_30000.png') });
    await page.close();
  }

  console.log('■ WebGL が使えないときも描ける');
  {
    const { page, errs } = await open(browser, 'seed=161&mech=yose&yn=4&size=s&foe=hole&gl=0');
    const s = await page.evaluate(() => { FT.run(25); FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 600)); });
    ok(s.gl === false && s.N > 50, `2D で描く（${s.N} 個）`);
    noErr(errs, '2D');
    await page.close();
  }

  console.log('■ v1.0 の設定（プレス）を読みかえる');
  {
    const { page, errs } = await open(browser, 'seed=171', null, () => { try { localStorage.setItem('fuerutama', JSON.stringify({ mech: 'press', speed: 1, vol: 40 })); } catch (e) {} });
    const s = await page.evaluate(() => FT.state());
    ok(s.setMech === 'yose' && s.mech === 'yose' && s.nb === 2 && s.vol === 40, `プレス → 寄せ刃 2 枚（${s.mech} ${s.nb} 枚、音量 ${s.vol}）`);
    noErr(errs, '読みかえ');
    await page.close();
  }

  console.log('■ キー・設定の欄・画面の大きさ');
  {
    const { page, errs } = await open(browser, 'seed=181', { width: 390, height: 844 });
    await page.evaluate(() => FT.pause(false));
    await page.mouse.click(195, 400);
    const r0 = await page.evaluate(() => FT.state());
    await page.keyboard.press('n');
    ok((await page.evaluate(() => FT.state())).rounds === r0.rounds + 1, 'N で次の回へ');
    await page.keyboard.press(' ');
    ok((await page.evaluate(() => FT.state())).paused === true, 'スペースで止まる');
    await page.keyboard.press(' ');
    ok((await page.evaluate(() => FT.state())).paused === false, 'もう一度スペースで動く');
    for (const [key, mech] of [['1', 'yose'], ['2', 'loops'], ['3', 'box']]) {
      await page.keyboard.press(key);
      const s = await page.evaluate(() => FT.state());
      ok(s.mech === mech && s.setMech === mech, `${key} で ${mech}`);
    }
    await page.keyboard.press('0');
    ok((await page.evaluate(() => FT.state())).setMech === 'auto', '0 でおまかせに戻る');
    const m0 = (await page.evaluate(() => FT.state())).mute;
    await page.keyboard.press('m');
    ok((await page.evaluate(() => FT.state())).mute === !m0, 'M で消音が切りかわる');
    await page.keyboard.press('m');
    await page.click('#bSet');
    await page.selectOption('#o_yoseN', '6');
    await page.selectOption('#o_yoseShape', 'saw');
    await page.keyboard.press('1');
    const s6 = await page.evaluate(() => FT.state());
    ok(s6.mech === 'yose' && s6.nb === 6 && s6.shape === 'saw', `設定の欄で 刃 6 枚・ギザギザ（${s6.nb} 枚・${s6.shape}）`);
    const lay = await page.evaluate(() => {
      const b = document.querySelector('#bar').getBoundingClientRect(), p = document.querySelector('#panel');
      return { sw: document.documentElement.scrollWidth, iw: innerWidth, l: b.left, r: b.right, h: b.height, ph: p.getBoundingClientRect().height, scroll: p.scrollHeight > p.clientHeight };
    });
    ok(lay.sw <= lay.iw && lay.l >= 15 && lay.r <= lay.iw - 15, `スマホ幅で横にはみ出さない（帯 ${Math.round(lay.l)}〜${Math.round(lay.r)}）`);
    ok(lay.h < 60, `スマホ幅でも操作の帯が 1 段（高さ ${Math.round(lay.h)}）`);
    ok(lay.ph < 844 - 100, `設定の欄は画面に収まり、中で巻き取れる（高さ ${Math.round(lay.ph)}${lay.scroll ? '・巻き取りあり' : ''}）`);
    noErr(errs, 'キーと画面');
    if (SHOTS) await page.screenshot({ path: path.join(SHOTS, 'check_phone.png') });
    await page.close();
  }

  await browser.close();
  console.log(failures ? `不合格 ${failures} 件` : '合格');
  process.exit(failures);
})().catch(e => { console.error(e); process.exit(99); });
