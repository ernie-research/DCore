"use strict";
// Particle "fireworks" for accepted draft runs.
// One fixed, click-through canvas covers the page. Each burst is anchored to its cell's
// scrolling .out element in content coordinates, so it follows the text when the cell
// auto-scrolls, and it is clipped to the cell's box.
//
// Performance notes (the effect runs while tokens stream in, so it must not force layout):
//  - getBoundingClientRect() results are cached per element (WeakMap); the cell box is
//    fixed by the CSS grid, so re-measuring every frame was pure layout thrash.
//    `refresh()` (fires on resize / layout changes) drops the cache.
//  - Particles are pre-rendered soft-glow sprites drawn with drawImage(), instead of
//    beginPath/arc/fill per particle; twinkle uses a deterministic per-particle phase.
const Fireworks = (() => {
  const MAX_PARTICLES = 1600;
  const GRAVITY = 150;   // px/s^2
  const DRAG = 2.4;      // 1/s, exponential velocity decay
  const GLOW = 3.2;      // sprite diameter = particle size * GLOW
  const SPRITE = 48;     // sprite canvas resolution
  const reduced = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };
  let canvas = null, ctx = null, dpr = 1, W = 0, H = 0, raf = null, lastT = 0, live = 0;
  let bursts = [];
  let liveMode = false;  // while streaming: fewer, shorter particles (see setLiveMode)
  let rectCache = new WeakMap();   // element -> {left, top, width, height}
  const sprites = new Map();       // "r,g,b" -> canvas

  function hexToRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
    const n = m ? parseInt(m[1], 16) : 0xffffff;
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const WHITE = [255, 255, 255];

  function rectOf(el) {
    let r = rectCache.get(el);
    if (!r) {
      const b = el.getBoundingClientRect();
      r = { left: b.left, top: b.top, width: b.width, height: b.height };
      rectCache.set(el, r);
    }
    return r;
  }

  function refresh() { rectCache = new WeakMap(); }

  // Soft-glow dot: hot white core falling off to the method colour.
  function getSprite(rgb) {
    const key = `${rgb[0]},${rgb[1]},${rgb[2]}`;
    let s = sprites.get(key);
    if (s) return s;
    const c = document.createElement("canvas");
    c.width = c.height = SPRITE;
    const g = c.getContext("2d");
    const grad = g.createRadialGradient(SPRITE / 2, SPRITE / 2, 0, SPRITE / 2, SPRITE / 2, SPRITE / 2);
    grad.addColorStop(0, "rgba(255,255,255,.9)");
    grad.addColorStop(0.28, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.85)`);
    grad.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
    g.fillStyle = grad;
    g.fillRect(0, 0, SPRITE, SPRITE);
    sprites.set(key, c);
    return c;
  }

  function ensureCanvas() {
    if (canvas) return;
    canvas = document.createElement("canvas");
    canvas.id = "fxCanvas";
    canvas.setAttribute("aria-hidden", "true");
    document.body.appendChild(canvas);
    // desynchronized: skip the canvas compositing fence where supported (Chrome/Edge)
    ctx = canvas.getContext("2d", { desynchronized: true });
    resize();
    window.addEventListener("resize", resize);
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    // clientWidth/Height match the fixed canvas box (viewport minus scrollbars)
    W = document.documentElement.clientWidth;
    H = document.documentElement.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    rectCache = new WeakMap();
  }

  // x, y: viewport coordinates of the accepted run; ratio in [0, 1] scales the burst.
  function burst({ anchor, clip, x, y, accepted, ratio, color }) {
    if (reduced.matches || !anchor) return;
    ensureCanvas();
    const r = Math.max(0, Math.min(1, ratio || 0));
    const base = hexToRgb(color);
    const palette = [base, mix(base, WHITE, 0.45), mix(base, WHITE, 0.8)];
    const sprs = [getSprite(palette[0]), getSprite(palette[1]), getSprite(palette[2])];
    // Keep only a couple of concurrent bursts. With 8 cells streaming, the old
    // global pool saturated on the first methods, so later ones lost their
    // firework entirely (n < 4 -> return) while the pool kept rendering at full
    // load, dragging the frame rate down. Evicting the oldest burst caps the
    // particle load and guarantees every new step gets an effect.
    const maxBursts = liveMode ? 2 : 3;
    while (bursts.length >= maxBursts) {
      const old = bursts.shift();
      live -= old.parts.length;
    }
    // While a live race streams into the DOM, keep the particle load light: fewer,
    // shorter-lived particles, no crackle cascades. The feedback elements
    // (flash / ring / +N label) render even with zero particles left.
    const crowd = liveMode ? 0.5 : 1;
    const n = Math.max(0, Math.min(MAX_PARTICLES - live, Math.round((8 + 72 * r) * crowd)));
    const speed = 70 + 270 * r;
    const life = (0.45 + 0.9 * r) * (liveMode ? 0.8 : 1);
    const parts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.35;
      // two shells: a dense outer ring plus a softer inner fill
      const s = speed * (i % 3 === 0 ? 0.3 + 0.35 * Math.random() : 0.75 + 0.25 * Math.random());
      parts.push({
        x: 0, y: 0, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 25 * r,
        age: 0, life: life * (0.7 + 0.5 * Math.random()),
        size: 1.6 + 1.8 * r * Math.random(),
        spr: sprs[i % 3 === 2 ? 2 : (i & 1)],
        phase: Math.random() * Math.PI * 2,
        crackle: r >= 0.7 && Math.random() < (liveMode ? 0.12 : 0.35),
      });
    }
    live += n;
    const rect = rectOf(anchor);
    bursts.push({
      anchor, clip: clip || anchor, r, parts, age: 0,
      ax: x - rect.left + anchor.scrollLeft,
      ay: y - rect.top + anchor.scrollTop,
      flash: 14 + 40 * r,
      ring: r >= 0.4 ? { R: 0.85 * speed / DRAG, life: 0.4 + 0.2 * r } : null,
      label: accepted >= 3
        ? { text: `+${accepted}`, life: 0.7 + 0.5 * r, size: Math.round(13 + 11 * r) } : null,
      light: css(palette[1]), sparkSpr: sprs[2],
    });
    if (!raf) { lastT = performance.now(); raf = requestAnimationFrame(frame); }
  }

  function spark(p, spr) {
    const a = Math.random() * Math.PI * 2, s = 30 + 40 * Math.random();
    return { x: p.x, y: p.y, vx: p.vx + Math.cos(a) * s, vy: p.vy + Math.sin(a) * s,
      age: 0, life: 0.25 + 0.15 * Math.random(), size: 1.2, spr,
      phase: Math.random() * Math.PI * 2, crackle: false };
  }

  function drawFlash(b, ox, oy) {
    if (b.age >= 0.18) return;
    const a = 1 - b.age / 0.18;
    const g = ctx.createRadialGradient(ox, oy, 0, ox, oy, b.flash);
    g.addColorStop(0, `rgba(255,255,255,${(0.6 * a).toFixed(3)})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.globalAlpha = 1;
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(ox, oy, b.flash, 0, Math.PI * 2); ctx.fill();
  }

  function drawRing(b, ox, oy) {
    if (!b.ring || b.age >= b.ring.life) return;
    const t = b.age / b.ring.life, e = 1 - (1 - t) ** 3;
    ctx.globalAlpha = 0.7 * (1 - t);
    ctx.strokeStyle = b.light;
    ctx.lineWidth = 2 + 2.5 * b.r * (1 - t);
    ctx.beginPath(); ctx.arc(ox, oy, 4 + b.ring.R * e, 0, Math.PI * 2); ctx.stroke();
  }

  function drawLabel(b, ox, oy) {
    if (!b.label || b.age >= b.label.life) return;
    const t = b.age / b.label.life;
    const ly = oy - 10 - 26 * (1 - (1 - t) ** 2);
    // pop-in: bigger on the first frames, settling to the final size
    const fs = Math.round(b.label.size * (1 + 0.35 * Math.max(0, 1 - t * 6)));
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = t < 0.15 ? t / 0.15 : 1 - Math.max(0, (t - 0.5) / 0.5);
    ctx.font = `700 ${fs}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(15,17,21,.85)";
    ctx.strokeText(b.label.text, ox, ly);
    ctx.fillStyle = b.light;
    ctx.fillText(b.label.text, ox, ly);
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const keep = [];
    for (const b of bursts) {
      b.age += dt;
      if (!b.anchor.isConnected) { live -= b.parts.length; continue; }
      const rect = rectOf(b.anchor);
      const ox = rect.left - b.anchor.scrollLeft + b.ax;
      const oy = rect.top - b.anchor.scrollTop + b.ay;
      const c = rectOf(b.clip);
      ctx.save();
      ctx.beginPath(); ctx.rect(c.left, c.top, c.width, c.height); ctx.clip();
      ctx.globalCompositeOperation = "lighter";
      drawFlash(b, ox, oy);
      drawRing(b, ox, oy);
      const decay = Math.exp(-DRAG * dt), born = [];
      let alive = 0;
      for (const p of b.parts) {
        if (p.age >= p.life) continue;
        p.age += dt;
        p.vx *= decay; p.vy = p.vy * decay + GRAVITY * dt;
        p.x += p.vx * dt; p.y += p.vy * dt;
        const t = p.age / p.life;
        if (t >= 1) {
          if (p.crackle && live + born.length < MAX_PARTICLES) {
            for (let k = 0; k < 3; k++) born.push(spark(p, b.sparkSpr));
          }
          continue;
        }
        alive += 1;
        // twinkle near the end of life (deterministic, no per-frame random)
        const a = (1 - t) * (1 - t) * (0.7 + 0.3 * Math.sin(p.phase + t * 25));
        if (a < 0.02) continue;
        const d = p.size * (1 - 0.45 * t) * GLOW;
        ctx.globalAlpha = a;
        ctx.drawImage(p.spr, ox + p.x - d / 2, oy + p.y - d / 2, d, d);
      }
      drawLabel(b, ox, oy);
      ctx.restore();
      if (born.length) { b.parts.push(...born); live += born.length; alive += born.length; }
      // compact dead particles so long-lived bursts do not grow unbounded
      if (b.parts.length > 2 * alive + 16) {
        const before = b.parts.length;
        b.parts = b.parts.filter((p) => p.age < p.life);
        live -= before - b.parts.length;
      }
      const overlay = Math.max(b.ring ? b.ring.life : 0, b.label ? b.label.life : 0, 0.18);
      if (alive > 0 || b.age < overlay) keep.push(b);
      else live -= b.parts.length;
    }
    bursts = keep;
    if (bursts.length) raf = requestAnimationFrame(frame);
    else { raf = null; live = 0; ctx.clearRect(0, 0, W, H); }
  }

  function clear() {
    bursts = []; live = 0;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    if (ctx) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); }
  }

  // Toggle the lighter particle profile used while a race streams into the DOM.
  function setLiveMode(v) { liveMode = !!v; }

  return { burst, clear, refresh, setLiveMode, get reduced() { return reduced.matches; } };
})();