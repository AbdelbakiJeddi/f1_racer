# Detector presentation (demo)

White "fiche" style demo of the track/centreline detector. No questions; every figure is live.
Open `index.html` (works from file://).

- Pages: `index.html`, `01-mask` … `08-stress-demo.html`; each page stands alone.
- Page list / nav: `PAGES` in `assets/demo.js`. Style: `assets/demo.css`.
- Detector maths + figures are shared with the course: `../course/assets/tracksim.js`, `figures.js`
  (verified against Python with `node ../course/tools/check-port.mjs`).
- Colours: blue = near band, violet = far band, yellow = paint, orange = problem/highlight, green = correct.
