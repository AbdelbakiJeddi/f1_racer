/* figures.js — reusable canvas figures. Needs tracksim.js (window.TS) loaded first.
 *
 *   Fig.camera(canvas, frame, overlay)      camera frame (or mask) upscaled, optional bands/markers/centreline
 *   Fig.detection(canvas, frame, det, cam)  camera frame + full lab-detector overlay (bands, lines, centres)
 *   Fig.profile(canvas, series, opts)       1-D band profile plot with level line and line markers
 *   Fig.topDown(canvas, track, opts)        bird's-eye view: track lines, car, near/far points, heading chord
 *
 * Colours come from CSS variables, so figures follow light/dark theme. Redraw on "themechange".
 */
(function () {
  "use strict";
  const TS = window.TS;
  const C = () => ({
    ink: TS.cssVar("--ink", "#1b1e24"), ink2: TS.cssVar("--ink-2", "#4a505b"), rule: TS.cssVar("--rule", "#c5cad2"),
    near: TS.cssVar("--near", "#0a8fae"), far: TS.cssVar("--far", "#d96d12"), yellow: TS.cssVar("--yellow", "#ffd100"),
    bad: TS.cssVar("--bad", "#c8323c"), good: TS.cssVar("--good", "#23824e"), card: TS.cssVar("--card", "#f7f8f9"),
    mono: TS.cssVar("--mono", "monospace"),
  });

  function sizeCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return dpr;
  }

  // overlay = { bands:[{rows:[r0,r1], color:"near"|"far"|css, lines:[{u}], centreU, v}], midline, centreLine:[[u,v],[u,v]], cross:[u,v] }
  function camera(canvas, frame, overlay = {}) {
    sizeCanvas(canvas);
    const ctx = TS.blitFrame(canvas, frame);
    const s = canvas.width / frame.w, c = C();
    const col = (k) => (k === "near" ? c.near : k === "far" ? c.far : k || c.yellow);
    ctx.save();
    ctx.lineWidth = Math.max(1, s * 0.25);
    if (overlay.midline) {
      ctx.setLineDash([s, s]); ctx.strokeStyle = "rgba(255,255,255,0.45)";
      ctx.beginPath(); ctx.moveTo((frame.w / 2) * s, 0); ctx.lineTo((frame.w / 2) * s, canvas.height); ctx.stroke(); ctx.setLineDash([]);
    }
    if (overlay.horizon != null) {
      ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.setLineDash([s * 2, s]);
      ctx.beginPath(); ctx.moveTo(0, overlay.horizon * s); ctx.lineTo(canvas.width, overlay.horizon * s); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const b of overlay.bands || []) {
      ctx.strokeStyle = col(b.color); ctx.lineWidth = Math.max(1.5, s * 0.35);
      ctx.strokeRect(0.5, b.rows[0] * s, canvas.width - 1, (b.rows[1] - b.rows[0]) * s);
      const v = b.v != null ? b.v : (b.rows[0] + b.rows[1]) / 2;
      for (const l of b.lines || []) cross(ctx, l.u * s, v * s, s * 1.4, "#fff", Math.max(1.5, s * 0.4));
      if (b.centreU != null) { ctx.fillStyle = col(b.color); dot(ctx, b.centreU * s, v * s, Math.max(4, s * 1.3)); }
    }
    if (overlay.centreLine) {
      const [[u0, v0], [u1, v1]] = overlay.centreLine;
      ctx.strokeStyle = c.yellow; ctx.lineWidth = Math.max(2, s * 0.5);
      ctx.beginPath(); ctx.moveTo(u0 * s, v0 * s); ctx.lineTo(u1 * s, v1 * s); ctx.stroke();
    }
    if (overlay.cross) cross(ctx, overlay.cross[0] * s, overlay.cross[1] * s, s * 2, c.yellow, Math.max(2, s * 0.5));
    ctx.restore();
    return s;
  }

  function detection(canvas, frame, det, cam, extra = {}) {
    const pts = [[cam.toPixelU(det.xNear, det.near.v), det.near.v], [cam.toPixelU(det.xFar, det.far.v), det.far.v]];
    return camera(canvas, frame, Object.assign({
      midline: true,
      bands: [
        { rows: det.near.rows, color: "near", lines: det.near.used, centreU: det.near.u, v: det.near.v },
        { rows: det.far.rows, color: "far", lines: det.far.used, centreU: det.far.u, v: det.far.v },
      ],
      centreLine: pts,
    }, extra));
  }

  function dot(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
  function cross(ctx, x, y, r, color, w) {
    ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke();
  }

  // series: [{data: Float32Array, color: "near"|"far"|css, fill:bool}], opts: {level, ymax, lines:[{u,color,label}], xmin, xmax, xlabel, split}
  function profile(canvas, series, opts = {}) {
    const dpr = sizeCanvas(canvas), ctx = canvas.getContext("2d"), c = C();
    const W = canvas.width, H = canvas.height, pad = { l: 34 * dpr, r: 8 * dpr, t: 10 * dpr, b: 22 * dpr };
    const n = series[0].data.length, xmin = opts.xmin ?? 0, xmax = opts.xmax ?? n, ymax = opts.ymax ?? 1;
    const X = (u) => pad.l + ((u - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
    const Y = (y) => H - pad.b - (y / ymax) * (H - pad.t - pad.b);
    const col = (k) => (k === "near" ? c.near : k === "far" ? c.far : k || c.ink);
    ctx.clearRect(0, 0, W, H);
    ctx.font = `${10 * dpr}px ${c.mono}`; ctx.fillStyle = c.ink2; ctx.strokeStyle = c.rule; ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(String(ymax), pad.l - 4 * dpr, pad.t + 8 * dpr); ctx.fillText("0", pad.l - 4 * dpr, H - pad.b);
    ctx.textAlign = "center";
    for (let u = xmin; u <= xmax; u += opts.xtick || 32) ctx.fillText(String(u), X(u), H - 6 * dpr);
    if (opts.xlabel) { ctx.textAlign = "right"; ctx.fillText(opts.xlabel, W - pad.r, H - 6 * dpr); }
    if (opts.split != null) { ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.strokeStyle = c.ink2; ctx.beginPath(); ctx.moveTo(X(opts.split), pad.t); ctx.lineTo(X(opts.split), H - pad.b); ctx.stroke(); ctx.setLineDash([]); }
    for (const s of series) {
      ctx.strokeStyle = col(s.color); ctx.lineWidth = 1.8 * dpr;
      ctx.beginPath();
      for (let i = 0; i < s.data.length; i++) { const x = X(i + 0.5), y = Y(Math.min(s.data[i], ymax)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
      if (s.fill) { ctx.globalAlpha = 0.18; ctx.lineTo(X(s.data.length - 0.5), Y(0)); ctx.lineTo(X(0.5), Y(0)); ctx.fillStyle = col(s.color); ctx.fill(); ctx.globalAlpha = 1; }
    }
    if (opts.level != null) {
      ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = c.bad; ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(opts.level)); ctx.lineTo(W - pad.r, Y(opts.level)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.bad; ctx.textAlign = "left"; ctx.fillText("level " + opts.level, pad.l + 4 * dpr, Y(opts.level) - 4 * dpr);
    }
    (opts.lines || []).forEach((l, idx) => {
      ctx.strokeStyle = col(l.color || c.ink); ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.moveTo(X(l.u), pad.t); ctx.lineTo(X(l.u), H - pad.b); ctx.stroke();
      // stagger labels so neighbouring markers don't overprint each other
      if (l.label) { ctx.fillStyle = col(l.color || c.ink); ctx.textAlign = "center"; ctx.fillText(l.label, X(l.u), pad.t + (8 + (idx % 2) * 12) * dpr); }
    });
    return { X, Y };
  }

  // Bird's-eye view. x lateral (+ right) horizontal, z forward vertical. Car at bottom centre.
  // opts: {xRange:0.8, zMax:1.3, near:{z,x}, far:{z,x}, chord:true, truth:true, label:true}
  function topDown(canvas, track, opts = {}) {
    const dpr = sizeCanvas(canvas), ctx = canvas.getContext("2d"), c = C();
    const W = canvas.width, H = canvas.height, xr = opts.xRange || 0.8, zMax = opts.zMax || 1.3;
    const k = Math.min(W / (2 * xr), (H - 40 * dpr) / zMax);
    const X = (x) => W / 2 + x * k, Z = (z) => H - 30 * dpr - z * k;
    ctx.clearRect(0, 0, W, H);
    // asphalt ribbon
    ctx.fillStyle = TS.cssVar("--panel", "#e1e4e8");
    ctx.beginPath();
    for (let z = 0; z <= zMax; z += 0.02) { const xc = track.centreX(z); const x = X(xc - track.halfWidth); z ? ctx.lineTo(x, Z(z)) : ctx.moveTo(x, Z(z)); }
    for (let z = zMax; z >= 0; z -= 0.02) ctx.lineTo(X(track.centreX(z) + track.halfWidth), Z(z));
    ctx.fill();
    // boundary lines
    ctx.strokeStyle = c.yellow; ctx.lineWidth = 4 * dpr;
    for (const s of [-1, 1]) {
      if ((s < 0 && !track.drawLeft) || (s > 0 && !track.drawRight)) continue;
      ctx.beginPath();
      for (let z = 0; z <= zMax; z += 0.02) { const x = X(track.centreX(z) + s * track.halfWidth); z ? ctx.lineTo(x, Z(z)) : ctx.moveTo(x, Z(z)); }
      ctx.stroke();
    }
    // true centreline
    if (opts.truth !== false) {
      ctx.strokeStyle = c.ink2; ctx.lineWidth = 1.2 * dpr; ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath();
      for (let z = 0; z <= zMax; z += 0.02) { const x = X(track.centreX(z)); z ? ctx.lineTo(x, Z(z)) : ctx.moveTo(x, Z(z)); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    // car heading axis
    ctx.strokeStyle = c.bad; ctx.lineWidth = 1.2 * dpr;
    ctx.beginPath(); ctx.moveTo(X(0), Z(0)); ctx.lineTo(X(0), Z(zMax)); ctx.stroke();
    // car
    ctx.fillStyle = c.ink;
    ctx.beginPath(); ctx.moveTo(X(0), Z(0.09)); ctx.lineTo(X(0.05), Z(-0.05)); ctx.lineTo(X(-0.05), Z(-0.05)); ctx.closePath(); ctx.fill();
    ctx.font = `${10 * dpr}px ${c.mono}`;
    // near / far estimates
    const pts = [];
    for (const [key, color] of [["near", c.near], ["far", c.far]]) {
      const p = opts[key];
      if (!p) continue;
      ctx.strokeStyle = color; ctx.globalAlpha = 0.35; ctx.lineWidth = dpr;
      ctx.beginPath(); ctx.moveTo(X(-xr), Z(p.z)); ctx.lineTo(X(xr), Z(p.z)); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = color; dot(ctx, X(p.x), Z(p.z), 5 * dpr);
      if (opts.label !== false) { ctx.textAlign = "left"; ctx.fillText(`${key} z=${p.z.toFixed(2)} m`, 4 * dpr, Z(p.z) - 4 * dpr); }
      pts.push(p);
    }
    if (opts.chord !== false && pts.length === 2) {
      const [a, b] = pts;
      ctx.strokeStyle = c.ink; ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.moveTo(X(a.x), Z(a.z)); ctx.lineTo(X(b.x), Z(b.z)); ctx.stroke();
      // heading angle arc, measured from straight ahead
      const ang = Math.atan2(b.x - a.x, b.z - a.z), r = 26 * dpr;
      ctx.strokeStyle = c.ink2; ctx.lineWidth = 1 * dpr; ctx.setLineDash([2 * dpr, 2 * dpr]);
      ctx.beginPath(); ctx.moveTo(X(a.x), Z(a.z)); ctx.lineTo(X(a.x), Z(a.z) - r * 1.6); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(X(a.x), Z(a.z), r, -Math.PI / 2, -Math.PI / 2 + ang, ang < 0); ctx.stroke();
    }
    return { X, Z, k };
  }

  function onTheme(fn) { document.addEventListener("themechange", fn); matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn); window.addEventListener("resize", fn); }

  window.Fig = { camera, detection, profile, topDown, onTheme, sizeCanvas, colors: C };
})();
