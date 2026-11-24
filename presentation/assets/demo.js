/* demo.js — page chrome + small helpers for the presentation pages.
 * A page needs: <body data-page="mask"> and <main> ... <section id="..."><h2>Title</h2>...</section> ... </main>
 * It gets: page pills (top), section pills (built from the h2s), prev/next pager.
 * Edit PAGES to add / reorder / rename pages.
 */
(function () {
  "use strict";
  const PAGES = [
    { id: "home", file: "index.html", short: "Overview" },
    { id: "mask", file: "01-mask.html", short: "1 · Mask" },
    { id: "scan", file: "02-scan.html", short: "2 · Scan" },
    { id: "camera", file: "03-camera.html", short: "3 · Camera" },
    { id: "centre", file: "04-centre.html", short: "4 · Centre" },
    { id: "errors", file: "05-errors.html", short: "5 · Errors" },
    { id: "gpu", file: "06-gpu-detector.html", short: "6 · GPU detector" },
    { id: "example", file: "07-worked-example.html", short: "7 · Worked example" },
    { id: "stress", file: "08-stress-demo.html", short: "8 · Stress demo" },
  ];
  const here = document.body.dataset.page;
  const main = document.querySelector("main");

  function chrome() {
    const sub = main.querySelector(".sub");
    const pages = document.createElement("nav");
    pages.className = "pages"; pages.setAttribute("aria-label", "Pages");
    pages.innerHTML = PAGES.map((p) => `<a href="${p.file}"${p.id === here ? ' aria-current="page"' : ""}>${p.short}</a>`).join("");
    const toc = document.createElement("nav");
    toc.className = "toc"; toc.setAttribute("aria-label", "On this page");
    toc.innerHTML = [...main.querySelectorAll("section[id] > h2")].map((h) => `<a href="#${h.parentElement.id}">${h.firstChild.textContent.trim()}</a>`).join("");
    sub.after(pages, toc);
    const i = PAGES.findIndex((p) => p.id === here), prev = PAGES[i - 1], next = PAGES[i + 1];
    const pager = document.createElement("div");
    pager.className = "pager";
    pager.innerHTML = (prev ? `<a href="${prev.file}">← ${prev.short}</a>` : "<span></span>") + (next ? `<a href="${next.file}">${next.short} →</a>` : "<span></span>");
    main.appendChild(pager);
  }

  // Animation that runs only while visible. Reduced motion: starts paused, the Play button still works.
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function animate(target, frameFn, button = null) {
    let playing = !reduced, visible = true, raf = 0, last = 0, t = 0;
    const tick = (now) => {
      raf = 0;
      if (!playing || !visible) return;
      const dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
      last = now; t += dt; frameFn(t);
      raf = requestAnimationFrame(tick);
    };
    const kick = () => { if (!raf && playing && visible) { last = 0; raf = requestAnimationFrame(tick); } };
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; kick(); }).observe(target);
    const label = () => { if (button) button.textContent = playing ? "Pause" : "Play"; };
    const ctl = {
      play() { playing = true; label(); kick(); }, pause() { playing = false; label(); },
      toggle() { playing ? ctl.pause() : ctl.play(); }, redraw() { frameFn(t); },
      get t() { return t; }, set t(v) { t = v; frameFn(t); },
    };
    if (button) { button.type = "button"; button.classList.add("play"); button.addEventListener("click", ctl.toggle); label(); }
    frameFn(0); kick();
    return ctl;
  }

  // <input type=range id=x> + <output for=x>; returns a getter.
  function slider(id, fmt, onInput) {
    const input = document.getElementById(id), out = document.querySelector(`output[for="${id}"]`);
    const show = () => { if (out) out.textContent = fmt(Number(input.value)); };
    input.addEventListener("input", () => { show(); onInput && onInput(Number(input.value)); });
    show();
    return () => Number(input.value);
  }

  // Toggle-button group. Returns a getter for the current choice.
  function chips(id, names, initial, onChange) {
    const box = document.getElementById(id);
    let cur = initial;
    for (const n of names) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = n; b.setAttribute("aria-pressed", n === cur);
      b.addEventListener("click", () => { cur = n; box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b)); onChange && onChange(n); });
      box.appendChild(b);
    }
    return () => cur;
  }

  const sgn = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
  const deg = (r, d = 1) => sgn((r * 180) / Math.PI, d) + "°";
  const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };

  chrome();
  window.Demo = { PAGES, animate, slider, chips, sgn, deg, set, reduced };
})();
