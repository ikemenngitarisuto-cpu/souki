// ふえる玉 v2 の確かめ。リポの一番上で:  node fuerutama/tools/check.js
// 終わりに「合格」と出れば OK（不合格の数が終了コードになる）。
//   FT_SHOTS=フォルダ … 画面の写しを残す    FT_BIG=1 … 10 万個の確かめも足す（1〜2 分よけいにかかる）
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
const shot = async (page, name) => {
  if (!SHOTS) return;
  await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
  await page.screenshot({ path: path.join(SHOTS, name + '.png') });
};

// 1 回ぶんを最後まで進めて、通った段取りを返す
async function oneRound(page, limit) {
  return page.evaluate((limit) => {
    const seen = [], first = {};
    let t = 0, maxN = 0, s = FT.state(), exact = true;
    while (t < limit) {
      s = FT.run(0.25); t += 0.25;
      if (seen[seen.length - 1] !== s.phase) seen.push(s.phase);
      if (s.phase === 'grow' || s.phase === 'gold' || s.phase === 'warn') maxN = Math.max(maxN, s.N);
      if (s.phase === 'grow' && s.rounds === 1 && s.N !== 1 + s.crush * (s.k - 1) && s.N < s.target) exact = false;   // 理論どおり: 1 個つぶれるたびに (分かれる数 - 1) 個ふえる
      for (const n of [10, 100, 1000]) if (first[n] === undefined && s.N >= n) first[n] = t;
      if (s.rounds > 1) break;
    }
    return { seen, maxN, first, t, exact, last: s };
  }, limit);
}

(async () => {
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });

  console.log('■ 仕掛けと刃の形: 1 個から決めた数ちょうどまで増えて、割れて、次の回へ（上限は「円の 2 倍」）');
  const cases = [['yose', 'yn=2&ys=bar', 101], ['yose', 'yn=2&ys=saw', 102], ['yose', 'yn=3&ys=saw', 103], ['yose', 'yn=4&ys=bar', 104], ['yose', 'yn=6&ys=arc', 105],
                 ['yose', 'yn=8&ys=chev', 106], ['loops', 'ln=2&ls=arc', 107], ['loops', 'ln=3&ls=bar', 108], ['loops', 'ln=5&ls=arc', 109], ['box', '', 110]];
  for (const [mech, extra, seed] of cases) {
    const name = `${mech}${extra ? ' ' + extra : ''}`;
    const { page, errs } = await open(browser, `seed=${seed}&mech=${mech}&${extra}&size=m&split=3&foe=none&limit=x2&gold=0${seed % 4 === 1 ? '&gl=1' : ''}`);
    const before = await page.evaluate(() => FT.state());
    const r = await oneRound(page, 220);
    ok(before.target > before.cap * 1.8, `${name}: 上限 ${before.target} 個（円 1 杯 ${before.cap} 個の約 2 倍）`);
    ok(r.maxN === before.target, `${name}: 上限ちょうどまで増えた（最大 ${r.maxN} 個。10 個 ${r.first[10]} 秒・100 個 ${r.first[100]} 秒・1,000 個 ${r.first[1000]} 秒）`);
    ok(r.exact, `${name}: 理論どおりに増えた（つぶれた数 ×（分かれる数 − 1）＋ 1 ＝ 玉の数）`);
    ok(/grow → warn → burst → clear → intro/.test(r.seen.join(' → ')), `${name}: 段取り ${r.seen.join(' → ')}`);
    ok(r.last.rounds === 2, `${name}: 次の回が始まった（${r.t} 秒で 1 回）`);
    noErr(errs, name);
    await page.close();
  }

  console.log('■ 動きの芯: 囲まれると白く光る → 閉じきるといっせいに分かれる → 出口から小さく生まれて飛び出す');
  {
    const { page, errs } = await open(browser, 'seed=201&mech=yose&yn=4&ys=bar&size=m&split=4&foe=none&gold=0');
    const r = await page.evaluate(() => {
      let s, n = 0;
      while (N < 40 && n++ < 4000) s = FT.run(1 / 60);
      // 締まっていくところまで進める
      n = 0; while (!(rig.q > .5 && rig.q < 1) && n++ < 2000) FT.run(1 / 60);
      let hot = 0; for (let i = 0; i < M; i++) if (!(ST[i] & 1) && HEAT[i] > .3) hot++;
      const q = +rig.q.toFixed(2), n0 = N, doomed0 = doomedN;
      // はじけるまで
      n = 0; while (N === n0 && n++ < 2000) FT.run(1 / 60);
      const fresh = []; for (let i = 0; i < M; i++) if (!(ST[i] & 1) && T - BORN[i] < .03) fresh.push({ sc: +SC[i].toFixed(2), d: Math.hypot(X[i], Y[i]) });
      const k = RD.k, grew = N - n0;
      FT.run(.6);
      let big = 0; for (let i = 0; i < M; i++) if (!(ST[i] & 1) && SC[i] >= 1) big++;
      return { q, hot, doomed0, grew, k, fresh: fresh.length, scMax: Math.max(...fresh.map(f => f.sc)), dMin: Math.min(...fresh.map(f => f.d)), dC: rig.dCn, big, N, sc0: SC0 };
    });
    ok(r.hot >= 1, `締まるとき（締まり具合 ${r.q}）、囲まれた玉が白く光る（${r.hot} 個）`);
    ok(r.grew > 0 && r.grew % (r.k - 1) === 0, `閉じきった瞬間にまとめて分かれる（＋${r.grew} 個＝${r.grew / (r.k - 1)} 個がそれぞれ ${r.k} つに）`);
    ok(r.fresh > 0 && r.scMax <= r.sc0 + .25, `生まれたては小さい（大きさ ${r.scMax} 倍まで。生まれたて ${r.fresh} 個）`);
    ok(r.dMin > r.dC, `刃の外（出口）に生まれる（真ん中から ${Math.round(r.dMin)} 以上。閉じた刃は ${Math.round(r.dC)}）`);
    ok(r.big === r.N, `0.6 秒後には、ふつうの大きさにふくらんでいる（${r.big}/${r.N} 個）`);
    noErr(errs, '動きの芯');
    await shot(page, 'check_core');
    await page.close();
  }

  console.log('■ 生まれた玉は、次に刃が閉じる 1 回は守られる（挟まれても刃の下をすり抜ける）');
  {
    const { page, errs } = await open(browser, 'seed=211&mech=yose&yn=2&size=m&split=3&foe=none&guard=1&gold=0');
    const r = await page.evaluate(() => {
      let s, n = 0;
      while (N < 2 && n++ < 3000) s = FT.run(1 / 60);
      const early = []; for (let i = 0; i < M; i++) if (!(ST[i] & 1)) early.push(+(IMM[i] - BORN[i]).toFixed(2));
      const nFirst = N;
      n = 0; while (N < 120 && n++ < 6000) s = FT.run(1 / 60);
      let newest = 0; for (let i = 0; i < M; i++) if (BORN[i] > BORN[newest]) newest = i;
      const g = +(IMM[newest] - BORN[newest]).toFixed(2), cyc = +rig.cycle.toFixed(2);
      FT.run(6);
      return { nFirst, early, g, cyc, ghost: STAT.ghost };
    });
    ok(r.nFirst === 3, `最初の 1 個が 3 つに分かれた（${r.nFirst} 個）`);
    ok(r.early.every(v => v <= .25), `玉が少ないうちは守らない（出だしを待たせない）: ${JSON.stringify(r.early)} 秒`);
    ok(Math.abs(r.g - (r.cyc + .12)) < .3, `40 個からは、仕掛け 1 回分（${r.cyc} 秒）守る: ${r.g} 秒`);
    ok(r.ghost > 0, `守られた玉は、挟まれても分かれずにすり抜けた（${r.ghost} 回）`);
    noErr(errs, '守る');
    await page.close();
  }

  console.log('■ 混んできたら: 当たりの大きさを縮めて、重なる／小さくなる');
  for (const [crowd, label] of [['over', '重なる'], ['shrink', '小さくなる']]) {
    const { page, errs } = await open(browser, `seed=221&mech=loops&ln=2&size=l&split=4&foe=none&limit=x2&gold=0&crowd=${crowd}`);
    const s = await page.evaluate(() => { let s, t = 0; while (t < 200){ s = FT.run(.25); t += .25; if (s.phase === 'warn') break; } return s; });
    ok(s.phase === 'warn' && s.rc < s.rad * .85, `${label}: 円の 2 倍（${s.N} 個）で、当たりの半径 ${s.rc}（もとは ${s.rad}）`);
    ok(crowd === 'over' ? s.rv === s.rad : Math.abs(s.rv - s.rc) < s.rad * .03, `${label}: 見た目の半径 ${s.rv}`);
    noErr(errs, label);
    await shot(page, 'check_crowd_' + crowd);
    await page.close();
  }

  console.log('■ 減るもの: 食べる玉（食べるほど大きくなる → 挟まれてはじけ、食べた玉を吐き出す）');
  {
    const { page, errs } = await open(browser, 'seed=4&mech=box&size=m&split=4&foe=eater&fl=2&gold=0&limit=x1');
    const r = await page.evaluate(() => {
      let s, t = 0, onAt = null, rMax = 0, r0 = 0, trap = false, full = false, hop = false, left0 = 0, leftMin = 99, popLabel = '';
      while (t < 200){
        s = FT.run(1 / 30); t += 1 / 30;
        if (s.foeOn && !onAt){ onAt = { N: s.N, t: +t.toFixed(1) }; left0 = s.foeInfo.left; }
        if (s.foeInfo && s.foeInfo.list){
          for (const e of s.foeInfo.list){ if (!r0) r0 = e.r; rMax = Math.max(rMax, e.r); if (e.state === 'trap') trap = true; if (e.state === 'hop') hop = true; if (e.full) full = true; }
          leftMin = Math.min(leftMin, s.foeInfo.left);
        }
        if (s.label.includes('はじけた')) popLabel = s.label;
        if (s.phase !== 'grow' && s.phase !== 'intro') break;
      }
      return { s, t: +t.toFixed(1), onAt, rMax, r0, trap, full, hop, left0, leftMin, popLabel };
    });
    ok(r.onAt && r.onAt.N >= 30 && r.onAt.N < 200, `玉が ${r.onAt ? r.onAt.N : '?'} 個のときに出てきた（${r.left0} 匹ぶん）`);
    ok(r.s.eaten > 20 && r.rMax > r.r0 * 1.5, `食べて大きくなった（食べた ${r.s.eaten} 個、半径 ${r.r0} → ${r.rMax}）`);
    ok(r.full && r.trap, 'おなかいっぱいになって、囲いに挟まれた');
    ok(r.s.spilled > 0 && r.leftMin < r.left0, `はじけて、食べた玉を吐き出した（吐き出した ${r.s.spilled} 個、のこり ${r.left0} → ${r.leftMin}）${r.popLabel ? '「' + r.popLabel + '」' : ''}`);
    ok(r.s.phase === 'warn' || r.s.phase === 'gone', `決着がついた（${r.s.phase}・${r.s.label}・${r.t} 秒）`);
    noErr(errs, '食べる玉');
    await page.close();
  }

  console.log('■ 減るもの: 穴（ひび → 開く → 閉じる。2 回目からは、玉が増えてから開く）');
  {
    const { page, errs } = await open(browser, 'seed=6&mech=yose&yn=2&size=m&split=4&foe=hole&fl=2&gold=0');
    const r = await page.evaluate(() => {
      let s, t = 0, onAt = null, states = new Set(), zoomMin = 1, opens = [], wasOpen = false, left0 = 0;
      while (t < 240){
        s = FT.run(.1); t += .1;
        if (s.foeOn && !onAt){ onAt = { N: s.N }; left0 = s.foeInfo.left; }
        const gaps = s.foeInfo && s.foeInfo.gaps ? s.foeInfo.gaps : [];
        for (const g of gaps) states.add(g.state);
        const open = gaps.some(g => g.state === 'open');
        if (open && !wasOpen) opens.push(s.N);
        wasOpen = open;
        if (open) zoomMin = Math.min(zoomMin, cam.zoom);
        if (s.phase !== 'grow' && s.phase !== 'intro') break;
      }
      return { s, t: +t.toFixed(1), onAt, states: [...states], zoomMin: +zoomMin.toFixed(2), opens, left0 };
    });
    ok(r.onAt && r.onAt.N >= 40 && r.onAt.N < 300, `玉が ${r.onAt ? r.onAt.N : '?'} 個のときに出てきた（${r.left0} 回ぶん）`);
    ok(['crack', 'open', 'shut'].every(x => r.states.includes(x)), `ひび → 開く → 閉じる（${r.states.join('・')}）`);
    ok(r.s.drained > 20, `穴から玉がこぼれた（${r.s.drained} 個）`);
    ok(r.opens.length >= 2 && r.opens[r.opens.length - 1] > r.opens[0] * 3, `あとの穴ほど、玉が増えてから開く（開いたときの数: ${r.opens.join('・')}）`);
    ok(r.zoomMin < .97, `開いているあいだは、少し引いて映す（${r.zoomMin} 倍）`);
    ok(r.s.phase === 'warn' || r.s.phase === 'gone', `決着がついた（${r.s.phase}・${r.s.label}・${r.t} 秒）`);
    noErr(errs, '穴');
    await page.close();
  }

  console.log('■ 減るもの: 寿命（色あせて消える。だんだん長くなって、最後はなくなる）');
  {
    const { page, errs } = await open(browser, 'seed=3&mech=loops&ln=3&size=m&split=3&foe=life&fl=2&gold=0&limit=x1');
    const r = await page.evaluate(() => {
      let s, t = 0, onAt = null, L0 = 0, L1 = 0, gray = 0, doneLabel = '';
      while (t < 240){
        s = FT.run(.1); t += .1;
        if (s.foeOn && !onAt){ onAt = { N: s.N }; L0 = s.foeInfo.L; }
        if (s.foeInfo && !s.foeInfo.done){ L1 = Math.max(L1, s.foeInfo.L); let g = 0; for (let i = 0; i < M; i++) if (!(ST[i] & 1) && DIE[i] - T < .5 * LIFE_L) g++; gray = Math.max(gray, g); }
        if (s.foeDone && !doneLabel) doneLabel = s.label;
        if (s.phase !== 'grow' && s.phase !== 'intro') break;
      }
      let limited = 0; for (let i = 0; i < M; i++) if (!(ST[i] & 1) && DIE[i] !== Infinity) limited++;
      return { s, t: +t.toFixed(1), onAt, L0, L1, gray, doneLabel, limited };
    });
    ok(r.onAt && r.onAt.N >= 30 && r.onAt.N < 200, `玉が ${r.onAt ? r.onAt.N : '?'} 個のときに出てきた`);
    ok(r.s.expired > 10 && r.gray > 0, `年をとった玉が色あせて消えた（消えた ${r.s.expired} 個、色あせていた玉 最大 ${r.gray} 個）`);
    ok(r.L1 > r.L0 * 1.15, `寿命はだんだん長くなる（${r.L0} 秒 → ${r.L1} 秒）`);
    ok(r.s.phase === 'gone' || (r.doneLabel.includes('寿命がなくなった') && r.limited === 0), `終わり方: ${r.s.phase === 'gone' ? r.s.label : '「' + r.doneLabel + '」のあとは寿命なし'}`);
    ok(r.s.phase === 'warn' || r.s.phase === 'gone', `決着がついた（${r.s.phase}・${r.t} 秒）`);
    noErr(errs, '寿命');
    await page.close();
  }

  console.log('■ 減るもの: 消し刃（赤い刃が 1 枚ずつ広がり、やがて引く。赤い刃に挟まれた玉は消える）');
  {
    const { page, errs } = await open(browser, 'seed=3&mech=yose&yn=5&ys=arc&size=m&split=3&foe=red&fl=2&gold=0&limit=x1');
    const r = await page.evaluate(() => {
      let s, t = 0, onAt = null, nMax = 0, P = 0, U = 0, seq = [], doneLabel = '';
      while (t < 240){
        s = FT.run(.1); t += .1;
        if (s.foeOn && !onAt) onAt = { N: s.N };
        if (s.foeInfo && s.foeInfo.P){ P = s.foeInfo.P; U = s.foeInfo.U; nMax = Math.max(nMax, s.foeInfo.n); if (seq[seq.length - 1] !== s.foeInfo.n) seq.push(s.foeInfo.n); }
        if (s.foeDone && !doneLabel) doneLabel = s.label;
        if (s.phase !== 'grow' && s.phase !== 'intro') break;
      }
      return { s, t: +t.toFixed(1), onAt, nMax, P, U, seq, doneLabel, redOn: RED.on, redK: Math.max(...rig.redK) };
    });
    ok(r.onAt && r.onAt.N >= 50 && r.onAt.N < 300, `玉が ${r.onAt ? r.onAt.N : '?'} 個のときに出てきた`);
    ok(r.nMax === r.P && r.P >= 1 && r.P <= r.U, `赤い刃は 1 枚ずつ増えて ${r.P}/${r.U} 枚まで（${r.seq.join(' → ')}）`);
    ok(r.s.erased > 10, `赤い刃に挟まれた玉が消えた（${r.s.erased} 個）`);
    ok(r.s.phase === 'gone' || (r.doneLabel.includes('赤がひいた') && !r.redOn), `終わり方: ${r.s.phase === 'gone' ? r.s.label : '「' + r.doneLabel + '」'}`);
    ok(r.s.phase === 'warn' || r.s.phase === 'gone', `決着がついた（${r.s.phase}・${r.t} 秒）`);
    noErr(errs, '消し刃');
    await page.close();
  }

  console.log('■ 全滅したら「〇〇の勝ち」と出て、次の回へ。記録にも残る');
  {
    const { page, errs } = await open(browser, 'seed=231&mech=yose&yn=4&size=m&foe=red&gold=0', null, () => { try { localStorage.removeItem('fuerutama.rec'); } catch (e) {} });
    const r = await page.evaluate(() => {
      let s, t = 0;
      while (t < 90){ s = FT.run(.25); t += .25; if (s.foeOn) break; }   // 消し刃が出てくるまで
      for (let i = 0; i < M; i++) kill(i);                                 // 全部消されたことにする
      const seen = []; let label = '';
      t = 0;
      while (t < 12){
        s = FT.run(.1); t += .1;
        if (seen[seen.length - 1] !== s.phase){ seen.push(s.phase); if (s.phase === 'gone') label = s.label; }
        if (s.rounds > 1) break;
      }
      return { seen, rounds: s.rounds, label, rec: JSON.parse(JSON.stringify(REC)), stats: document.querySelector('#stats').textContent };
    });
    ok(r.seen.join(' → ').startsWith('gone → clear → intro') && r.rounds === 2, `全滅: ${r.seen.join(' → ')}`);
    ok(r.label.includes('消し刃の勝ち'), `全滅の名札: ${r.label}`);
    ok(r.rec.rounds === 1 && r.rec.lose.red === 1 && /消し刃 と 0 勝 1 敗/.test(r.stats), `記録: ${r.stats}`);
    noErr(errs, '全滅');
    await page.close();
  }

  console.log('■ 極小の玉の金色は、決めた数ちょうど（1 万個）');
  {
    const { page, errs } = await open(browser, 'seed=141&mech=yose&yn=2&size=t&split=3&foe=none&limit=10000&gold=1', { width: 1280, height: 720 });
    const r = await page.evaluate(() => { let s, t = 0; while (t < 200){ s = FT.run(.25); t += .25; if (s.phase === 'gold') break; } return { s, t }; });
    ok(r.s.phase === 'gold' && r.s.N === 10000, `金色: ${r.s.phase} ${r.s.N} 個（${r.t} 秒）`);
    const st = await page.evaluate(() => { FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 1500)); });
    ok(st.fps >= 8, `1 万個でも動いている（この環境で ${st.fps} こま/秒、計算 ${st.physMs} ms、${st.gl ? 'WebGL' : '2D'} で描く）`);
    noErr(errs, '金色');
    await shot(page, 'check_gold');
    await page.close();
  }

  console.log('■ 3 万個（円の約 3 倍に重なる）でも止まらない');
  {
    const { page, errs } = await open(browser, 'seed=151&mech=loops&ln=3&size=t&split=4&foe=none&limit=30000&gold=0', { width: 1280, height: 720 });
    const r = await page.evaluate(() => { let s, t = 0; while (t < 220){ s = FT.run(.25); t += .25; if (s.phase !== 'grow' && s.phase !== 'intro') break; }
      const a = performance.now(); for (let i = 0; i < 10; i++) substep(HS); return { s, t, ms: +((performance.now() - a) / 10).toFixed(1) }; });
    ok(r.s.N === 30000, `3 万個まで増えた（${r.t} 秒、計算 1 歩 ${r.ms} ms）`);
    noErr(errs, '3 万個');
    await shot(page, 'check_30000');
    await page.close();
  }

  if (process.env.FT_BIG === '1') {
    console.log('■ 10 万個（プリセット「大洪水」と同じ設定）');
    const { page, errs } = await open(browser, 'seed=152&mech=yose&yn=2&size=t&split=4&speed=1.5&limit=100000&eject=2&foe=none&q=0&gold=0', { width: 1280, height: 720 });
    const r = await page.evaluate(() => { let s, t = 0; while (t < 300){ s = FT.run(.25); t += .25; if (s.phase !== 'grow' && s.phase !== 'intro') break; }
      const a = performance.now(); for (let i = 0; i < 6; i++) advance(1 / 60); return { s, t, ms: +((performance.now() - a) / 6).toFixed(1) }; });
    ok(r.s.N === 100000, `10 万個まで増えた（${r.t} 秒、計算 1 こま ${r.ms} ms）`);
    noErr(errs, '10 万個');
    await shot(page, 'check_100000');
    await page.close();
  }

  console.log('■ 描き方: WebGL（?gl=1 で GPU がなくても使う）と、WebGL なしの 2D');
  {
    // GPU のない環境では、ふだんは 2D になる（ソフトウェア描画の WebGL は遅いため）。ここでは WebGL を無理に使って、絵が出ることを確かめる
    const { page, errs } = await open(browser, 'seed=161&mech=box&size=l&split=4&foe=red&fl=2&gl=1');
    const r = await page.evaluate(() => {
      let s, n = 0; while (N < 300 && n++ < 6000) s = FT.run(1 / 30);
      render();
      const gl = GLR.gl, w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let solid = 0; for (let i = 3; i < px.length; i += 4) if (px[i] > 200) solid++;
      return { gl: s.gl, N: s.N, solid, area: Math.round(N * Math.PI * (RV * VIEW.s) ** 2 * .2) };   // 重なりや、生まれたての小さい玉があるので、玉の面積の 2 割より多ければよい
    });
    ok(r.gl === true, 'WebGL で描いている（影つきの玉の絵をその場で作る。試験では GPU がなくても使う）');
    ok(r.solid > r.area, `WebGL の画面に玉が描かれている（${r.N} 個・塗られた点 ${r.solid}）`);
    noErr(errs, 'WebGL');
    await shot(page, 'check_webgl');
    await page.close();
  }
  {
    const { page, errs } = await open(browser, 'seed=161&mech=yose&yn=4&size=s&foe=hole&gl=0');
    const s = await page.evaluate(() => { FT.run(25); FT.pause(false); return new Promise(res => setTimeout(() => res(FT.state()), 600)); });
    ok(s.gl === false && s.N > 50, `2D で描く（${s.N} 個）`);
    noErr(errs, '2D');
    await shot(page, 'check_2d');
    await page.close();
  }

  console.log('■ 前の版の設定を引き継ぐ（v1.0 のプレス → 寄せ刃 2 枚。v1.1 の設定はそのまま）');
  {
    let o = await open(browser, 'seed=171', null, () => { try { localStorage.setItem('fuerutama', JSON.stringify({ mech: 'press', speed: 1, vol: 40 })); } catch (e) {} });
    let s = await o.page.evaluate(() => FT.state());
    ok(s.setMech === 'yose' && s.mech === 'yose' && s.nb === 2 && s.vol === 40, `v1.0 プレス → 寄せ刃 2 枚（${s.mech} ${s.nb} 枚、音量 ${s.vol}）`);
    noErr(o.errs, '読みかえ v1.0');
    await o.page.close();
    o = await open(browser, '', null, () => { try { localStorage.setItem('fuerutama', JSON.stringify({ mech: 'yose', yoseN: '6', yoseShape: 'saw', size: 's', split: '4', foe: 'hole', foeLv: '3', limit: 'x5', guard: '2', vol: 25, mute: true, count: false })); } catch (e) {} });
    s = await o.page.evaluate(() => ({ st: FT.state(), set: { limit: SET.limit, guard: SET.guard, count: SET.count } }));
    ok(s.st.mech === 'yose' && s.st.nb === 6 && s.st.shape === 'saw' && s.st.size === 's' && s.st.k === 4 && s.st.foe === 'hole' && s.st.foeLv === 3 && s.st.vol === 25 && s.st.mute === true
       && s.set.limit === 'x5' && s.set.guard === '2' && s.set.count === false, `v1.1 の設定がそのまま使われる（${s.st.mech} ${s.st.nb} 枚 ${s.st.shape}・${s.st.size}・${s.st.k} つ・${s.st.foe} 強さ ${s.st.foeLv}・上限 ${s.set.limit}）`);
    noErr(o.errs, '読みかえ v1.1');
    await o.page.close();
  }

  console.log('■ キー・設定の欄・プリセット・固定・スマホ幅');
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
    // 固定（P）: 次の回も同じ組み合わせ
    await page.keyboard.press('p');
    const rc1 = await page.evaluate(() => JSON.stringify(RD.rc));
    await page.keyboard.press('n');
    const p1 = await page.evaluate(() => ({ rc: JSON.stringify(RD.rc), pin: FT.state().pin, on: document.querySelector('#bPin').classList.contains('on') }));
    ok(p1.pin && p1.on && p1.rc === rc1, `P で今の組み合わせを固定（次の回も ${rc1}）`);
    await page.keyboard.press('p');
    ok((await page.evaluate(() => FT.state())).pin === false, 'もう一度 P で固定を外す');
    // 設定の欄とプリセット
    await page.click('#bSet');
    await page.selectOption('#o_yoseN', '6');
    await page.selectOption('#o_yoseShape', 'saw');
    await page.keyboard.press('1');
    const s6 = await page.evaluate(() => FT.state());
    ok(s6.mech === 'yose' && s6.nb === 6 && s6.shape === 'saw', `設定の欄で 刃 6 枚・ギザギザ（${s6.nb} 枚・${s6.shape}）`);
    await page.click('#presets button:nth-child(2)');      // 動画ふう
    const sv = await page.evaluate(() => ({ st: FT.state(), set: { guard: SET.guard, yoseN: SET.yoseN }, on: document.querySelector('#presets button:nth-child(2)').classList.contains('on') }));
    ok(sv.st.size === 'xl' && sv.st.k === 4 && sv.st.foe === null && sv.set.guard === '0' && sv.set.yoseN === 'auto' && sv.on, `プリセット「動画ふう」: 特大・4 つ・守らない・減るものなし（ほかは既定に戻る）`);
    await page.click('#presets button:nth-child(1)');      // おまかせ
    const lay = await page.evaluate(() => {
      const b = document.querySelector('#bar').getBoundingClientRect(), p = document.querySelector('#panel'), pr = p.getBoundingClientRect();
      return { sw: document.documentElement.scrollWidth, iw: innerWidth, l: b.left, r: b.right, h: b.height, top: b.top, ph: pr.height, pb: pr.bottom, scroll: p.scrollHeight > p.clientHeight };
    });
    ok(lay.sw <= lay.iw && lay.l >= 15 && lay.r <= lay.iw - 15, `スマホ幅で横にはみ出さない（帯 ${Math.round(lay.l)}〜${Math.round(lay.r)}）`);
    ok(lay.h < 60, `スマホ幅でも操作の帯が 1 段（高さ ${Math.round(lay.h)}）`);
    ok(lay.pb <= lay.top && lay.ph <= 844 - 90, `設定の欄は帯の上に収まり、中で巻き取れる（高さ ${Math.round(lay.ph)}${lay.scroll ? '・巻き取りあり' : ''}）`);
    noErr(errs, 'キーと画面');
    await shot(page, 'check_phone');
    await page.close();
  }

  await browser.close();
  console.log(failures ? `不合格 ${failures} 件` : '合格');
  process.exit(failures);
})().catch(e => { console.error(e); process.exit(99); });
