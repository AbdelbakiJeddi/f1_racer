/* tracksim.js — browser port of the project's vision code, for the course.
 *
 *   SyntheticTrack   <- labs/helpers/vision_preview.py   (renders 128x128 camera frames)
 *   Camera           <- labs/lanedet/camera.py           (pixels <-> metres, flat ground)
 *   labMask, bandProfile, findLines, estimateCentre, LaneDetector
 *                    <- labs/lanedet/{mask,scan,centre,detector}.py
 *   ProdDetector     <- src/leatherback_racer/vision/track_detector.py  (the GPU detector, batch of 1)
 *
 * Plain script (no ES modules) so lessons work when opened straight from disk (file://).
 * Everything lives on window.TS.
 * Keep the maths 1:1 with the Python; check-port.mjs compares the two.
 */
(function (root) {
  "use strict";

  const YELLOW = [255, 209, 0];
  const ASPHALT = [31, 31, 31];
  const BACKGROUND = [62, 68, 82];

  const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  // Small seeded RNG (mulberry32) + Box-Muller, for repeatable noise.
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    const uni = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = () => {
      const u = Math.max(uni(), 1e-12), v = uni();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return { uni, gauss };
  }

  // ------------------------------------------------------------------ frames
  // A frame is { w, h, data: Uint8ClampedArray(w*h*3) } RGB, row-major.
  function makeFrame(w, h) { return { w, h, data: new Uint8ClampedArray(w * h * 3) }; }
  function cloneFrame(f) { return { w: f.w, h: f.h, data: new Uint8ClampedArray(f.data) }; }
  function px(f, c, r) { const i = (r * f.w + c) * 3; return [f.data[i], f.data[i + 1], f.data[i + 2]]; }

  // ------------------------------------------------------------------ synthetic track
  const TRACK_DEFAULTS = {
    width: 128, height: 128,
    focal: 64.0,        // px (~90 deg HFOV at 128 px)
    camHeight: 0.25,    // m
    horizon: 0.28,      // horizon row as fraction of height
    halfWidth: 0.30,    // m, centre to line centre
    offset: 0.0,        // m, track centre lateral position at the car (+ right)
    yaw: 0.0,           // rad, track direction relative to car
    curvature: 0.0,     // 1/m, + bends right
    lineWidth: 0.03,    // m
    drawLeft: true, drawRight: true,
    brightness: 1.0,
    noise: 0.0,         // gaussian std, 0..255
    asphalt: ASPHALT, lineColor: YELLOW, background: BACKGROUND,
  };

  class SyntheticTrack {
    constructor(opts = {}) { Object.assign(this, TRACK_DEFAULTS, opts); }
    with(opts) { return new SyntheticTrack(Object.assign({}, this, opts)); }
    get vHorizon() { return this.horizon * this.height; }
    distance(v) { return (this.focal * this.camHeight) / (v - this.vHorizon); }
    centreX(z) { return this.offset + Math.tan(this.yaw) * z + 0.5 * this.curvature * z * z; }
    camera() { return new Camera(this.width, this.height, this.focal, this.camHeight, this.vHorizon); }

    render(seed = null) {
      const W = this.width, H = this.height;
      const img = new Float32Array(W * H * 3);
      const hRow = Math.ceil(this.vHorizon);
      for (let r = 0; r < H; r++) {
        const col = r < hRow ? this.background : this.asphalt;
        for (let c = 0; c < W; c++) { const i = (r * W + c) * 3; img[i] = col[0]; img[i + 1] = col[1]; img[i + 2] = col[2]; }
      }
      const lc = this.lineColor;
      for (let r = hRow; r < H; r++) {
        const v = r + 0.5;
        if (v - this.vHorizon < 0.5) continue;
        const z = this.distance(v);
        const xc = this.centreX(z);
        const lines = [];
        if (this.drawLeft) lines.push(xc - this.halfWidth);
        if (this.drawRight) lines.push(xc + this.halfWidth);
        for (const x of lines) {
          const u0 = W / 2 + (this.focal * (x - this.lineWidth / 2)) / z;
          const u1 = W / 2 + (this.focal * (x + this.lineWidth / 2)) / z;
          const cLo = Math.max(0, Math.floor(u0)), cHi = Math.min(W - 1, Math.ceil(u1));
          for (let c = cLo; c <= cHi; c++) {
            const cover = clip(Math.min(c + 1, u1) - Math.max(c, u0), 0, 1);
            if (cover <= 0) continue;
            const i = (r * W + c) * 3;
            for (let k = 0; k < 3; k++) img[i + k] = img[i + k] * (1 - cover) + lc[k] * cover;
          }
        }
      }
      const g = this.noise > 0 ? rng(seed == null ? 1 : seed + 1) : null;
      const out = makeFrame(W, H);
      for (let i = 0; i < img.length; i++) {
        let x = img[i] * this.brightness;
        if (g) x += g.gauss() * this.noise;
        out.data[i] = Math.floor(clip(x, 0, 255)); // numpy astype(uint8) truncates
      }
      return out;
    }
  }

  // The lab's demo drive: car weaves while the track bends left and right.
  function driveState(i, n = 150) {
    const t = (2 * Math.PI * i) / n;
    return { offset: 0.10 * Math.sin(t), yaw: 0.12 * Math.sin(t + 1.2), curvature: 0.7 * Math.sin(2 * t) };
  }

  // ------------------------------------------------------------------ camera (block 3)
  class Camera {
    constructor(width, height, focal, camHeight, vHorizon) {
      Object.assign(this, { width, height, focal, camHeight, vHorizon });
    }
    get cx() { return this.width / 2; }
    distance(v) { return (this.focal * this.camHeight) / (v - this.vHorizon); }
    lateral(u, v) { return (this.camHeight * (u - this.cx)) / (v - this.vHorizon); }
    toPixelU(x, v) { return this.cx + (x * (v - this.vHorizon)) / this.camHeight; }
    widthPx(m, v) { return (m * (v - this.vHorizon)) / this.camHeight; }
    rowAt(z) { return this.vHorizon + (this.focal * this.camHeight) / z; }
  }

  // ------------------------------------------------------------------ mask (block 1)
  // Returns Float32Array(w*h) in 0..1.
  function yellownessLab(r, g, b) { return Math.min(r, g) - b; }          // min(R,G) - B
  function yellownessProd(r, g, b) { return clip(r + g - 2 * b, 0, 1); }  // R + G - 2B, clamped

  function labMask(frame, low = 0.15, high = 0.40) {
    const n = frame.w * frame.h, m = new Float32Array(n), d = frame.data;
    for (let i = 0; i < n; i++) {
      const y = yellownessLab(d[3 * i] / 255, d[3 * i + 1] / 255, d[3 * i + 2] / 255);
      m[i] = clip((y - low) / (high - low), 0, 1);
    }
    return m;
  }

  function prodMask(frame, threshold = 0.25) {
    const n = frame.w * frame.h, m = new Float32Array(n), d = frame.data;
    for (let i = 0; i < n; i++) {
      const c = yellownessProd(d[3 * i] / 255, d[3 * i + 1] / 255, d[3 * i + 2] / 255);
      m[i] = clip(Math.max(c - threshold, 0) / (1 - threshold + 1e-6), 0, 1);
    }
    return m;
  }

  // ------------------------------------------------------------------ scan (block 2)
  function bandProfile(mask, w, rows) {
    const [r0, r1] = rows, p = new Float32Array(w), n = r1 - r0;
    for (let r = r0; r < r1; r++) for (let c = 0; c < w; c++) p[c] += mask[r * w + c];
    for (let c = 0; c < w; c++) p[c] /= n;
    return p;
  }

  // Runs of the profile above `level` -> {u, mass, clipped, start, end}, left to right.
  function findLines(profile, level = 0.1, minMass = 1.0) {
    const lines = [], W = profile.length;
    let c = 0;
    while (c < W) {
      if (profile[c] > level) {
        const start = c;
        while (c < W && profile[c] > level) c++;
        const end = c;
        let mass = 0, mom = 0;
        for (let k = start; k < end; k++) { mass += profile[k]; mom += profile[k] * (k + 0.5); }
        if (mass >= minMass) lines.push({ u: mom / mass, mass, clipped: start === 0 || end === W, start, end });
      } else c++;
    }
    return lines;
  }

  // ------------------------------------------------------------------ centre (block 4)
  // Sides a lone line cannot be: the other boundary would be in view, but the profile is empty there.
  // "left" = "this line is the left boundary" (the other one would sit at u + 2*halfPx).
  function refutedSides(line, halfPx, profile, level = 0.1, margin = 3, win = 4) {
    const out = new Set(), W = profile.length;
    for (const [side, other] of [["left", line.u + 2 * halfPx], ["right", line.u - 2 * halfPx]]) {
      if (other < margin || other > W - margin) continue;
      let mx = 0;
      for (let c = Math.max(0, Math.floor(other) - win); c < Math.min(W, Math.floor(other) + win + 1); c++) mx = Math.max(mx, profile[c]);
      if (mx < level) out.add(side);
    }
    return out;
  }
  function sideFromRefuted(r) { return r.size !== 1 ? null : r.has("right") ? "left" : "right"; }

  function estimateCentre(lines, halfPx, priorU, tolerance = 0.35, singleConfidence = 0.5, side = null) {
    let best = null;
    for (let i = 0; i < lines.length; i++)
      for (let j = i + 1; j < lines.length; j++) {
        const a = lines[i], b = lines[j];
        const err = Math.abs(b.u - a.u - 2 * halfPx) / (2 * halfPx);
        if (err < tolerance && (best === null || err < best.err)) best = { err, a, b };
      }
    if (best) {
      let conf = 1 - best.err / tolerance;
      if (best.a.clipped || best.b.clipped) conf *= 0.8;
      return { u: (best.a.u + best.b.u) / 2, case: "pair", confidence: conf, used: [best.a, best.b] };
    }
    let cands = [];
    for (const l of lines) {
      cands.push({ d: Math.abs(l.u + halfPx - priorU), u: l.u + halfPx, side: "left", l });
      cands.push({ d: Math.abs(l.u - halfPx - priorU), u: l.u - halfPx, side: "right", l });
    }
    if (side && lines.length === 1) cands = cands.filter((c) => c.side === side);
    if (cands.length) {
      let m = cands[0];
      for (const c of cands) if (c.d < m.d) m = c; // first minimum, like Python min()
      return { u: m.u, case: m.side, confidence: singleConfidence * (m.l.clipped ? 0.8 : 1), used: [m.l] };
    }
    return { u: null, case: "none", confidence: 0, used: [] };
  }

  // ------------------------------------------------------------------ detector (block 5)
  const LAB_DEFAULTS = {
    halfWidth: 0.30, nearRows: [68, 75], farRows: [52, 58],
    maskLow: 0.15, maskHigh: 0.40, lineLevel: 0.1, minMass: 1.0, tolerance: 0.35,
    stateful: true,
  };

  class LaneDetector {
    constructor(camera, opts = {}) { this.camera = camera; Object.assign(this, LAB_DEFAULTS, opts); this.reset(); }
    reset() { this.prevX = { near: 0, far: 0 }; this.prevValid = false; }
    static bandRow(rows) { return (rows[0] + rows[1]) / 2; }
    scan(mask, rows) {
      const v = LaneDetector.bandRow(rows);
      const profile = bandProfile(mask, this.camera.width, rows);
      const lines = findLines(profile, this.lineLevel, this.minMass);
      const halfPx = this.camera.widthPx(this.halfWidth, v);
      const refuted = lines.length === 1 ? refutedSides(lines[0], halfPx, profile, this.lineLevel) : new Set();
      return { v, rows, profile, lines, halfPx, refuted, side: sideFromRefuted(refuted) };
    }
    detect(frame) {
      if (!this.stateful) this.reset();
      const mask = labMask(frame, this.maskLow, this.maskHigh);
      const s = { near: this.scan(mask, this.nearRows), far: this.scan(mask, this.farRows) };
      const est = {};
      for (const n of ["near", "far"]) {
        const b = s[n], priorU = this.camera.toPixelU(this.prevX[n], b.v);
        est[n] = Object.assign(estimateCentre(b.lines, b.halfPx, priorU, this.tolerance), b, { priorU, rule: "memory" });
      }
      // A lone line is placed by, in order: a pair on the other band, last frame's centre, the missing-line test.
      // The test comes last because worn or shadowed paint also looks like "no line there".
      for (const [n, o] of [["near", "far"], ["far", "near"]]) {
        const b = est[n];
        if (b.case === "pair") b.rule = "pair";
        if (b.case !== "left" && b.case !== "right") continue;
        if (est[o].case === "pair") {
          const priorU = this.camera.toPixelU(this.camera.lateral(est[o].u, est[o].v), b.v);
          Object.assign(b, estimateCentre(b.lines, b.halfPx, priorU, this.tolerance), { priorU, rule: "pair on other band" });
        } else if (!this.prevValid) {
          let side = b.side;
          // a lone line in both bands is the same boundary, so an undecided band borrows the other's answer
          if (!side && s[o].lines.length === 1) side = s[o].side;
          Object.assign(b, estimateCentre(b.lines, b.halfPx, b.priorU, this.tolerance, 0.5, side), { side, rule: side ? "missing-line test" : "no memory: image centre" });
        }
      }
      const near = est.near, far = est.far;
      const xNear = near.u === null ? this.prevX.near : this.camera.lateral(near.u, near.v);
      const xFar = far.u === null ? this.prevX.far : this.camera.lateral(far.u, far.v);
      this.prevX = { near: xNear, far: xFar };
      this.prevValid = this.prevValid || near.u !== null || far.u !== null;
      const zNear = this.camera.distance(near.v), zFar = this.camera.distance(far.v);
      return {
        eLat: clip(xNear / this.halfWidth, -1, 1),
        eHeading: Math.atan2(xFar - xNear, zFar - zNear),
        confidence: (near.confidence + far.confidence) / 2,
        xNear, xFar, zNear, zFar, near, far, mask,
      };
    }
  }

  // ------------------------------------------------------------------ production detector
  // src/leatherback_racer/vision/track_detector.py, one image instead of a batch of N.
  const PROD_DEFAULTS = {
    imgH: 128, imgW: 128,
    nearRoi: [0.46, 0.52], farRoi: [0.40, 0.45],
    yellowThreshold: 0.25,
    nearHalfWidth: 0.525, farHalfWidth: 0.990,
    longitudinalDeltaD: 0.532, halfTrackWidth: 0.550,
    nearHalfSpan: 0.80, farHalfSpan: 0.55,
  };

  function linspace(a, b, n) { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = n === 1 ? a : a + ((b - a) * i) / (n - 1); return o; }

  // _find_scanline_center: split the profile in halves, centre of mass per half.
  function prodScanlineCentre(profile, halfSpan, prev = null) {
    const W = profile.length, uMid = Math.floor(W / 2);
    const gl = linspace(-1, 0, uMid), gr = linspace(0, 1, W - uMid);
    let mL = 0, mR = 0, sL = 0, sR = 0;
    for (let i = 0; i < uMid; i++) { mL += profile[i]; sL += profile[i] * gl[i]; }
    for (let i = uMid; i < W; i++) { mR += profile[i]; sR += profile[i] * gr[i - uMid]; }
    const xL = sL / (mL + 1e-6), xR = sR / (mR + 1e-6);
    const minLineMass = 2.0;
    const cL = clip(mL / minLineMass, 0, 1), cR = clip(mR / minLineMass, 0, 1);
    const both = cL > 0.6 && cR > 0.6, onlyL = cL > 0.6 && !both, onlyR = cR > 0.6 && !both;
    let x, kase;
    if (both) { x = (xL + xR) * 0.5; kase = "pair"; }
    else if (onlyL) { x = xL + halfSpan; kase = "left"; }
    else if (onlyR) { x = xR - halfSpan; kase = "right"; }
    else { x = prev === null ? 0 : prev; kase = "none"; }
    x = clip(x, -1, 1);
    const conf = both ? (cL + cR) * 0.5 : Math.max(cL, cR) * 0.85;
    return { x, confidence: conf, case: kase, xL, xR, massL: mL, massR: mR, confL: cL, confR: cR };
  }

  class ProdDetector {
    constructor(opts = {}) { Object.assign(this, PROD_DEFAULTS, opts); }
    rows(roi) { return [Math.floor(this.imgH * roi[0]), Math.floor(this.imgH * roi[1])]; }
    detect(frame) {
      const mask = prodMask(frame, this.yellowThreshold);
      const nr = this.rows(this.nearRoi), fr = this.rows(this.farRoi);
      const pN = bandProfile(mask, frame.w, nr), pF = bandProfile(mask, frame.w, fr);
      const near = prodScanlineCentre(pN, this.nearHalfSpan);
      const far = prodScanlineCentre(pF, this.farHalfSpan, near.x);
      const yNear = near.x * this.nearHalfWidth, yFar = far.x * this.farHalfWidth;
      const eLat = clip(yNear / this.halfTrackWidth, -1, 1);
      const eHeading = Math.atan2(yFar - yNear, this.longitudinalDeltaD);
      return {
        eLat, eHeading, confidence: (near.confidence + far.confidence) / 2,
        near: Object.assign(near, { rows: nr, v: Math.floor((nr[0] + nr[1]) / 2), profile: pN }),
        far: Object.assign(far, { rows: fr, v: Math.floor((fr[0] + fr[1]) / 2), profile: pF }),
        yNear, yFar, mask,
      };
    }
  }

  // ------------------------------------------------------------------ stress conditions (labs/08_tune)
  const STRESS = {
    clean: (f) => f,
    shadow(f) { // tree shadow over the right half
      const o = cloneFrame(f);
      for (let r = 55; r < o.h; r++) for (let c = 60; c < o.w; c++) for (let k = 0; k < 3; k++) { const i = (r * o.w + c) * 3 + k; o.data[i] = Math.floor(o.data[i] * 0.15); }
      return o;
    },
    glare(f) { const o = cloneFrame(f); for (let i = 0; i < o.data.length; i++) o.data[i] = Math.floor(clip(f.data[i] * 0.3 + 170, 0, 255)); return o; },
    redCurb(f) { const o = cloneFrame(f); for (let r = 70; r < o.h; r++) for (let c = 0; c < 10; c++) { const i = (r * o.w + c) * 3; o.data[i] = 200; o.data[i + 1] = 30; o.data[i + 2] = 30; } return o; },
    cone(f) { const o = cloneFrame(f); for (let r = 62; r < 78; r++) for (let c = 60; c < 67; c++) { const i = (r * o.w + c) * 3; o.data[i] = 255; o.data[i + 1] = 150; o.data[i + 2] = 0; } return o; },
    noise(f, seed = 7) { const g = rng(seed), o = cloneFrame(f); for (let i = 0; i < o.data.length; i++) o.data[i] = Math.floor(clip(f.data[i] + g.gauss() * 30, 0, 255)); return o; },
  };

  // GPU detector constants recomputed for a given Camera (so it can run fairly on the synthetic track).
  function prodConfigFor(cam, halfWidth = 0.30, base = PROD_DEFAULTS) {
    const p = new ProdDetector(base), nr = p.rows(p.nearRoi), fr = p.rows(p.farRoi);
    const vn = (nr[0] + nr[1]) / 2, vf = (fr[0] + fr[1]) / 2, hw = (v) => (cam.camHeight * cam.cx) / (v - cam.vHorizon);
    return {
      nearHalfWidth: hw(vn), farHalfWidth: hw(vf), longitudinalDeltaD: cam.distance(vf) - cam.distance(vn),
      halfTrackWidth: halfWidth, nearHalfSpan: halfWidth / hw(vn), farHalfSpan: halfWidth / hw(vf),
    };
  }

  // One lap of the synthetic drive -> RMSE against ground truth (labs/08_tune.ipynb `evaluate`).
  // detector: object with detect(frame) returning {eLat, eHeading, near:{case}, far:{case}}
  // bandsZ: {zNear, zFar} distances where truth is evaluated.
  function evaluateLap(detector, bandsZ, { n = 120, noise = 8, transform = null, base = new SyntheticTrack() } = {}) {
    if (detector.reset) detector.reset();
    let sl = 0, sh = 0, paired = 0, seen = 0;
    const trace = [];
    for (let i = 0; i < n; i++) {
      const tr = base.with(Object.assign({ noise }, driveState(i, n)));
      let f = tr.render(i);
      if (transform) f = transform(f, i);
      const d = detector.detect(f);
      const xn = tr.centreX(bandsZ.zNear), xf = tr.centreX(bandsZ.zFar);
      const tl = clip(xn / tr.halfWidth, -1, 1), th = Math.atan2(xf - xn, bandsZ.zFar - bandsZ.zNear);
      sl += (d.eLat - tl) ** 2; sh += (d.eHeading - th) ** 2;
      if (d.near.case === "pair" && d.far.case === "pair") paired++;
      if (d.near.case !== "none" || d.far.case !== "none") seen++;
      trace.push({ eLat: d.eLat, tLat: tl, eHeading: d.eHeading, tHeading: th });
    }
    return { latRmse: Math.sqrt(sl / n), headingRmseDeg: (Math.sqrt(sh / n) * 180) / Math.PI, paired: paired / n, seen: seen / n, trace };
  }

  // ------------------------------------------------------------------ drawing helpers
  // Draw a frame (RGB) or a mask (0..1) onto a canvas, nearest-neighbour upscaled to fill it.
  function blitFrame(canvas, frame) {
    const off = blitFrame._off || (blitFrame._off = document.createElement("canvas"));
    off.width = frame.w; off.height = frame.h;
    const octx = off.getContext("2d"), id = octx.createImageData(frame.w, frame.h);
    for (let i = 0, j = 0; i < frame.data.length; i += 3, j += 4) { id.data[j] = frame.data[i]; id.data[j + 1] = frame.data[i + 1]; id.data[j + 2] = frame.data[i + 2]; id.data[j + 3] = 255; }
    octx.putImageData(id, 0, 0);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    return ctx;
  }
  function maskToFrame(mask, w, h, tint = [255, 255, 255]) {
    const f = makeFrame(w, h);
    for (let i = 0; i < w * h; i++) for (let k = 0; k < 3; k++) f.data[3 * i + k] = Math.round(mask[i] * tint[k]);
    return f;
  }
  // Size canvas backing store to CSS size x devicePixelRatio. Returns scale (canvas px per image px).
  function fitCanvas(canvas, imgW) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || cssW;
    if (canvas.width !== Math.round(cssW * dpr)) { canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr); }
    return canvas.width / imgW;
  }
  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  root.TS = {
    YELLOW, ASPHALT, BACKGROUND, clip, rng, linspace,
    makeFrame, cloneFrame, px,
    SyntheticTrack, driveState, Camera,
    yellownessLab, yellownessProd, labMask, prodMask,
    bandProfile, findLines, estimateCentre, LaneDetector, LAB_DEFAULTS,
    prodScanlineCentre, ProdDetector, PROD_DEFAULTS, prodConfigFor,
    evaluateLap, STRESS, blitFrame, maskToFrame, fitCanvas, cssVar,
  };
  if (typeof module !== "undefined") module.exports = root.TS;
})(typeof window !== "undefined" ? window : globalThis);
