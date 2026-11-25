// Tiny SVG helpers shared by all pages. No dependencies.
const C = {fg:'#111', mut:'#4a5058', line:'#e4e1da', acc:'#e8590c', good:'#2f9e44', blue:'#1c7ed6', tint:'#fff4ec'};
// [file, nav label, built]
const PAGES = [['index','Home',1],['00-mdp','0 MDP',1],['01-return','1 Return',1],['02-value','2 Value',1],['03-policy','3 Policy',1],
  ['04-gradient','4 Gradient',1],['05-baseline','5 Baseline',0],['06-gae','6 GAE',0],['07-trust','7 Trust region',0],['08-clip','8 Clip',0],
  ['09-rslrl','9 rsl_rl',0],['10-networks','10 Networks',0],['11-mini-ppo','11 Mini-PPO',0],['12-read','12 Read data',0],
  ['13-tune','13 Tune',0],['14-rewards','14 Rewards',0],['15-build','15 Build',0]];
const $ = id => document.getElementById(id);

function nav(current) {
  const built = PAGES.filter(p => p[2]), i = built.findIndex(p => p[0] === current);
  $('nav').innerHTML = PAGES.map(([f,t,ok]) => ok ? `<a href="${f}.html"${f===current?' class="on"':''}>${t}</a>` : `<a class="soon" title="coming next">${t}</a>`).join('');
  const link = (p, txt) => p ? `<a href="${p[0]}.html">${txt.replace('*', p[1])}</a>` : '<span></span>';
  $('pn').innerHTML = link(built[i-1], '← *') + link(built[i+1], '* →');
}

// Predict-first box: question, choices, the right index, the explanation shown after a pick.
function predict(id, q, choices, right, why) {
  const el = $(id);
  el.className = 'predict';
  el.innerHTML = `<b class="q">Predict first: ${q}</b>` + choices.map((c,i) => `<button class="pick" data-i="${i}">${c}</button>`).join('') + '<div class="verdict"></div>';
  el.querySelectorAll('.pick').forEach(b => b.onclick = () => {
    el.querySelectorAll('.pick').forEach(x => x.classList.toggle('right', +x.dataset.i === right));
    if (+b.dataset.i !== right) b.classList.add('wrong');
    el.querySelector('.verdict').innerHTML = (+b.dataset.i === right ? '<b class="ok">Right.</b> ' : '<b class="ko">Not quite.</b> ') + why;
  });
}

// Shared timeline engine: frames = number of steps, render(k) draws step k and returns its caption.
function timeline(id, frames, render, o={}) {
  const el = $(id); let k = 0, timer = null;
  el.insertAdjacentHTML('afterbegin', `<div class="tl"><button data-a="first">⏮</button><button data-a="prev">◀</button><button data-a="play">Play</button><button data-a="next">▶</button><button data-a="last">⏭</button><span class="cap"></span></div>`);
  const cap = el.querySelector('.cap'), play = el.querySelector('[data-a=play]');
  const draw = () => { cap.innerHTML = `step ${k}/${frames()-1} · ` + (render(k) || ''); };
  const stop = () => { clearInterval(timer); timer = null; play.textContent = 'Play'; };
  el.querySelector('.tl').onclick = e => {
    const a = e.target.dataset.a; if (!a) return;
    if (a === 'play') { if (timer) return stop(); play.textContent = 'Pause'; timer = setInterval(() => { if (k >= frames()-1) return stop(); k++; draw(); }, o.ms || 160); return; }
    stop(); k = a === 'first' ? 0 : a === 'last' ? frames()-1 : Math.max(0, Math.min(frames()-1, k + (a === 'next' ? 1 : -1))); draw();
  };
  // first draw happens on reset()/redraw(), once the caller has built its data
  return { redraw: () => { k = Math.min(k, frames()-1); draw(); }, reset: () => { stop(); k = 0; draw(); } };
}

const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
const T = (x,y,s,o={}) => `<text x="${x}" y="${y}" font-size="${o.size||12}" text-anchor="${o.a||'middle'}" fill="${o.col||C.fg}"${o.b?' font-weight="700"':''}>${esc(s)}</text>`;
const R = (x,y,w,h,o={}) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx??8}" fill="${o.fill||'#fff'}" stroke="${o.col||C.fg}" stroke-width="${o.sw||1.4}"/>`;
const L = (pts,o={}) => `<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${o.col||C.fg}" stroke-width="${o.sw||1.5}"${o.dash?' stroke-dasharray="5 4"':''}${o.arrow?` marker-end="url(#ah${(o.col||C.fg).slice(1)})"`:''}/>`;
const DEFS = '<defs>' + [C.fg,C.acc,C.good,C.blue,C.mut].map(c => `<marker id="ah${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${c}"/></marker>`).join('') + '</defs>';
// box with centred lines of text
function box(x,y,w,h,lines,o={}) {
  const y0 = y + h/2 - (lines.length-1)*8 + 4;
  return R(x,y,w,h,o) + lines.map((s,i) => T(x+w/2, y0+i*16, s, {b:i===0, size:i===0?13:11.5, col:i===0?C.fg:C.mut})).join('');
}
function svg(id, w, h, inner) { $(id).innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img">${DEFS}${inner}</svg>`; }

const fmt = v => v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) < 0.001 ? v.toExponential(1) : +v.toPrecision(3) + '';

// Line chart. series: [{pts:[[x,y]], col, label, dash}]. o: {xl, yl, ymin, ymax, xmin, xmax, vlines:[{x,label,col}], log, hover}
function chart(id, series, o={}) {
  const W=720, H=o.h||250, m={l:58,r:14,t:22,b:34};
  const all = series.flatMap(s => s.pts), ty = v => o.log ? Math.log10(v) : v;
  const xs = all.map(p=>p[0]), ys = all.map(p=>ty(p[1]));
  const x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs);
  let y0 = o.ymin != null ? ty(o.ymin) : Math.min(...ys), y1 = o.ymax != null ? ty(o.ymax) : Math.max(...ys);
  if (y1 - y0 < 1e-12) { y1 += 1; y0 -= 1; }
  const X = x => m.l + (x-x0)/(x1-x0)*(W-m.l-m.r), Y = y => H-m.b - (ty(y)-y0)/(y1-y0)*(H-m.t-m.b);
  const Yr = v => o.log ? 10**v : v;
  let s = '';
  for (let k=0;k<=4;k++) {
    const v = y0 + (y1-y0)*k/4, py = H-m.b-(H-m.t-m.b)*k/4, x = x0+(x1-x0)*k/4;
    s += L([[m.l,py],[W-m.r,py]],{col:C.line,sw:1}) + T(m.l-6,py+4,fmt(Yr(v)),{a:'end',size:11,col:C.mut});
    s += T(X(x),H-m.b+15,fmt(Math.abs(x)<1e-9?0:x),{size:11,col:C.mut});
  }
  for (const v of o.vlines||[]) s += L([[X(v.x),m.t],[X(v.x),H-m.b]],{col:v.col||C.mut,dash:true,sw:1.2}) + T(X(v.x),m.t-6,v.label,{size:11,col:v.col||C.mut});
  series.forEach((se,i) => {
    s += L(se.pts.map(p=>[X(p[0]).toFixed(1),Y(p[1]).toFixed(1)]),{col:se.col||C.fg,sw:2,dash:se.dash});
    if (se.dots) s += se.pts.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="${se.col||C.fg}"/>`).join('');
    if (se.label) s += L([[m.l+10+i*190,H-6],[m.l+30+i*190,H-6]],{col:se.col||C.fg,sw:3,dash:se.dash}) + T(m.l+36+i*190,H-2,se.label,{a:'start',size:11});
  });
  if (o.xl) s += T(W-m.r,H-m.b-6,o.xl,{a:'end',size:11,col:C.mut});
  if (o.yl) s += T(4,12,o.yl,{a:'start',size:11,col:C.mut});
  if (o.hover) s += `<line class="hv" y1="${m.t}" y2="${H-m.b}" stroke="${C.acc}" opacity="0"/><text class="ht" y="12" x="${W-m.r}" text-anchor="end" font-size="12" font-weight="700" fill="${C.acc}"></text>`;
  svg(id, W, H + (series.some(se=>se.label)?4:0), s);
  if (o.hover) {
    const el = $(id).querySelector('svg'), pts = series[0].pts;
    el.onpointermove = e => {
      const r = el.getBoundingClientRect(), x = x0 + ((e.clientX-r.left)/r.width*W - m.l)/(W-m.l-m.r)*(x1-x0);
      const p = pts.reduce((a,b) => Math.abs(b[0]-x) < Math.abs(a[0]-x) ? b : a);
      const l = el.querySelector('.hv'); l.setAttribute('x1',X(p[0])); l.setAttribute('x2',X(p[0])); l.setAttribute('opacity',.7);
      el.querySelector('.ht').textContent = `iteration ${p[0]}: ${fmt(p[1])}`;
    };
  }
}

// Bind range/select inputs: calls draw(values) now and on every change; writes <output id="NAME-o">.
function bind(ids, draw) {
  const run = () => {
    const v = {};
    for (const id of ids) {
      const el = $(id); v[id] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : +el.value;
      const out = $(id+'-o'); if (out) out.textContent = el.value;
    }
    draw(v);
  };
  ids.forEach(id => $(id).addEventListener('input', run));
  run(); return run;
}

// Vertical bars. vals: numbers; o: {labels, max, col, hi (index to highlight), fmtv, h}
function bars(id, vals, o={}) {
  const W=720, H=o.h||220, m={l:44,r:10,t:20,b:34}, n=vals.length, bw=(W-m.l-m.r)/n;
  const mx = o.max ?? Math.max(...vals.map(Math.abs), 1e-9), Y = v => H-m.b - v/mx*(H-m.t-m.b);
  let s = L([[m.l,H-m.b],[W-m.r,H-m.b]],{col:C.line});
  for (let k=0;k<=4;k++){ const v=mx*k/4; s += T(m.l-6,Y(v)+4,fmt(v),{a:'end',size:11,col:C.mut}); }
  vals.forEach((v,i) => {
    const x = m.l+i*bw+bw*0.15, w = bw*0.7, col = (o.cols && o.cols[i]) || (i===o.hi ? C.acc : o.col || C.blue);
    s += `<rect x="${x}" y="${Math.min(Y(v),H-m.b)}" width="${w}" height="${Math.abs(H-m.b-Y(v))}" fill="${col}" opacity=".85" rx="2"/>`;
    s += T(x+w/2, Y(v)-5, (o.fmtv||fmt)(v), {size:11});
    if (o.labels) s += T(x+w/2, H-m.b+15, o.labels[i], {size:11, col:C.mut});
  });
  if (o.xl) s += T(W-m.r, H-4, o.xl, {a:'end', size:11, col:C.mut});
  svg(id, W, H, s);
}
const randn = () => Math.sqrt(-2*Math.log(1-Math.random()))*Math.cos(2*Math.PI*Math.random());
