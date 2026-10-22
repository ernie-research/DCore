"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  meta: {},          // method key -> {label, color, ours}
  rec: null,         // current recording (full, with runs)
  cells: {},         // method key -> cell DOM handles + render state
  mode: "idle",      // idle | live | replay
  preset: "0:0",
  raf: null,
  abort: null,
};

function fmt(x, d = 1) { return (x === null || x === undefined || !isFinite(x)) ? "–" : x.toFixed(d); }
function setMsg(text, err = false) { $("msg").textContent = text; $("msg").classList.toggle("err", err); }

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json();
}

// ---------------------------------------------------------------- setup ----

async function init() {
  const cfg = await api("/api/config");
  state.config = cfg;
  for (const m of [...cfg.methods, cfg.baseline]) state.meta[m.key] = m;

  $("model").innerHTML = cfg.models.map((m) => `<option>${m}</option>`).join("");
  $("budget").innerHTML = cfg.budgets.map((b) => `<option value="${b}">${b}</option>`).join("");
  $("budget").value = "16";
  const seg = $("preset");
  for (const p of cfg.presets) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = p; b.dataset.preset = p;
    b.setAttribute("aria-pressed", p === state.preset ? "true" : "false");
    b.onclick = () => setPreset(p);
    seg.appendChild(b);
  }
  const pick = $("methodsPick");
  for (const m of cfg.methods) {
    const l = document.createElement("label");
    l.style.setProperty("--c", m.color);
    l.innerHTML = `<input type="checkbox" value="${m.key}" checked> <span class="swatch"></span> ${m.label}`;
    pick.appendChild(l);
  }
  const bench = $("bench"), q = $("q");
  const fillQuestions = () => {
    const b = cfg.prompts[+bench.value];
    q.innerHTML = `<option value="">— custom —</option>` +
      (b ? b.questions.map((p, i) => `<option value="${i}">${escapeHtml(p.title || p.prompt.slice(0, 40))}</option>`).join("") : "");
    if (b && b.questions.length) { q.value = "0"; $("prompt").value = b.questions[0].prompt; }
  };
  bench.innerHTML = cfg.prompts.map((b, i) => `<option value="${i}">${escapeHtml(b.benchmark)}</option>`).join("");
  bench.onchange = fillQuestions;
  q.onchange = () => { const b = cfg.prompts[+bench.value]; const i = q.value; if (b && i !== "") $("prompt").value = b.questions[+i].prompt; };
  if (cfg.prompts.length) { bench.value = "0"; fillQuestions(); }
  $("model").onchange = $("budget").onchange = refreshAvailability;
  $("runBtn").onclick = runLive;
  $("cancelBtn").onclick = cancelRun;
  $("replayBtn").onclick = () => replay(state.rec);
  $("loadBtn").onclick = loadSelected;
  $("refreshBtn").onclick = refreshRecordings;
  $("stopServersBtn").onclick = stopServers;
  $("presentBtn").onclick = togglePresentation;
  $("entShow").onchange = syncEntShow;
  syncEntShow();
  $("layout").onchange = () => { document.body.classList.toggle("layout-row", $("layout").value === "row"); Fireworks.refresh(); };
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input" || tag === "select") return;
    if (e.key === "h" || e.key === "H") togglePresentation();
    if (e.key === "f" || e.key === "F") toggleFx();
    if (e.key === "e" || e.key === "E") { $("entShow").checked = !$("entShow").checked; syncEntShow(); }
    if (e.key === "r" || e.key === "R") { if (state.rec && state.mode !== "live") replay(state.rec); }
  });

  refreshAvailability();
  await refreshRecordings();
  buildCells(currentMethods(), $("baseline").checked);
  updateSubtitle({ model: $("model").value, budget: +$("budget").value, preset: state.preset });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setPreset(p) {
  state.preset = p;
  for (const b of $("preset").children) b.setAttribute("aria-pressed", b.dataset.preset === p ? "true" : "false");
}

function currentMethods() {
  return [...$("methodsPick").querySelectorAll("input:checked")].map((i) => i.value);
}

function refreshAvailability() {
  const av = state.config.availability[$("model").value];
  const budget = $("budget").value;
  for (const input of $("methodsPick").querySelectorAll("input")) {
    const ok = av && av.checkpoints[`${input.value}:${budget}`];
    input.parentElement.title = ok ? "" : "checkpoint not found on disk";
    input.parentElement.style.opacity = ok ? "1" : ".45";
  }
  if (av && !av.target_exists) setMsg(`Target model not found: ${av.target}`, true);
}

async function refreshRecordings() {
  const list = await api("/api/recordings");
  const sel = $("recSelect");
  sel.innerHTML = `<option value="">—</option>` + list.map((r) =>
    `<option value="${escapeHtml(r.id)}">${escapeHtml(`${r.model} b${r.budget} T${r.preset} · ${r.prompt.slice(0, 48)}`)}</option>`).join("");
}

function togglePresentation() { document.body.classList.toggle("presentation"); Fireworks.refresh(); }

function updateSubtitle(r) {
  state.budget = r.budget;  // verify window of the race on screen; scales the fireworks
  const label = (p) => ({ "0:0": "greedy", "1:0": "sampled target, greedy draft", "1:1": "sampled target & draft" }[p] || "");
  const gpus = r.gpus
    ? (r.parallel ? `parallel on ${r.gpus.length} GPUs` : "1 GPU")
    : `GPUs: ${state.config && state.config.gpus ? state.config.gpus : "auto"}`;
  $("subtitle").textContent = `${r.model} · budget ${r.budget} · T(target:draft) = ${r.preset} (${label(r.preset)}) · thinking off · ${gpus}`;
}

// ---------------------------------------------------------------- cells ----

const BASELINE = "baseline";

function buildCells(methods, includeBaseline) {
  // Same order as the backend runner: Target AR (baseline) first, then
  // EAGLE-3, then the remaining methods in registry order.
  const order = state.config.methods.map((m) => m.key).filter((k) => methods.includes(k));
  if (order.includes("eagle3")) {
    order.splice(order.indexOf("eagle3"), 1);
    order.unshift("eagle3");
  }
  if (includeBaseline) order.unshift(BASELINE);
  const grid = $("grid"), board = $("board");
  grid.innerHTML = ""; board.innerHTML = "";
  const cap = document.createElement("div"); cap.className = "cap";
  cap.innerHTML = `<i class="sw-tps"></i>tok/s (thin) &nbsp;·&nbsp; <i class="sw-tau"></i>τ accept length (wide)`;
  board.appendChild(cap);
  state.cells = {};
  state.baselineTokS = null;
  for (const key of order) {
    const m = state.meta[key];
    const root = document.createElement("article");
    root.className = "cell" + (m.ours ? " ours" : "");
    root.style.setProperty("--c", m.color);
    root.setAttribute("aria-label", m.label);
    root.innerHTML = `
      <div class="cell-head"><span class="title">${escapeHtml(m.label)}</span>
        <span class="chip" aria-live="polite">queued</span><span class="rank"></span></div>
      <div class="stats">
        <span>tokens<b data-k="tokens">–</b></span><span>τ<b data-k="tau">–</b></span>
        <span>tok/s<b data-k="toks">–</b></span><span class="speedup">speedup<b data-k="speedup">–</b></span>
      </div>
      <div class="ent-stats" hidden></div>
      <div class="out" tabindex="0"></div>`;
    grid.appendChild(root);

    const name = document.createElement("div"); name.className = "name"; name.textContent = m.label;
    const track = document.createElement("div"); track.className = "track";
    const barTau = document.createElement("div"); barTau.className = "bar";
    const bar = document.createElement("div"); bar.className = "bar tps";  // unified --hl colour
    track.append(barTau, bar);
    const val = document.createElement("div"); val.className = "val"; val.textContent = "–";
    board.append(name, track, val);

    state.cells[key] = {
      key, root, out: root.querySelector(".out"), chip: root.querySelector(".chip"),
      rank: root.querySelector(".rank"), bar, barTau, val,
      st: Object.fromEntries([...root.querySelectorAll("[data-k]")].map((b) => [b.dataset.k, b])),
      tokens: 0, decodeTokens: 0, verifySteps: 0, time: 0, done: false, summary: null,
      fxUntil: 0, fxLast: 0, tokIdx: 0, entropy: null, entStats: null,
      entBox: root.querySelector(".ent-stats"),
    };
  }
  Fireworks.clear();
  $("timer").textContent = "0.00 s";
}

function setChip(cell, text, cls = "") {
  cell.chip.textContent = text;
  cell.chip.title = text;
  cell.chip.className = "chip" + (cls ? " " + cls : "");
}

// -------------------------------------------------- streaming render ----
// Verify steps arrive from 8 cells in parallel; appending each one to the DOM
// synchronously forced layout ~7 times per step (scroll reads, style writes,
// stats, bars). Instead queue steps and flush them once per animation frame:
// each cell touches the DOM at most once per frame, in a single appendChild
// inside a DocumentFragment, with one stick/scroll and one stats pass.
let pendingSteps = [];        // [{cell, ev}] accumulated for the next frame
let stepsScheduled = false;

function queueStep(cell, ev) {
  pendingSteps.push({ cell, ev });
  if (!stepsScheduled) {
    stepsScheduled = true;
    requestAnimationFrame(() => { stepsScheduled = false; flushSteps(); });
  }
}

function flushSteps() {
  const q = pendingSteps; pendingSteps = [];
  if (!q.length) return;
  const byCell = new Map();
  for (const { cell, ev } of q) {
    let evs = byCell.get(cell);
    if (!evs) byCell.set(cell, evs = []);
    evs.push(ev);
  }
  let lastT = 0;
  const touched = [];
  for (const [cell, evs] of byCell) {
    const t = renderBatch(cell, evs);
    if (t > lastT) lastT = t;
    updateStats(cell, evs[evs.length - 1].t);
    touched.push(cell);
  }
  // After every cell has been appended: fireworks measure client rects, whose
  // first read forces layout, so batching all cells into one pass costs a
  // single layout instead of one per cell (8 cells -> 1).
  flushCellCelebrations(touched);
  if (lastT) { updateBars(); $("timer").textContent = `${lastT.toFixed(2)} s`; }
}

// Render a batch of steps into the cell with a single DOM write and one
// scroll decision.
function renderBatch(cell, evs) {
  const frag = document.createDocumentFragment();
  let lastT = 0;
  for (const ev of evs) { lastT = ev.t; renderStep(cell, ev, frag); }
  if (frag.childNodes.length) {
    const stick = cell.out.scrollHeight - cell.out.scrollTop - cell.out.clientHeight < 40;
    cell.out.appendChild(frag);
    if (stick) cell.out.scrollTop = cell.out.scrollHeight;
  }
  return lastT;
}

// Fireworks need their span attached to the DOM before measuring its rects, so
// they are queued during renderStep and fired here, after the batch is in.
function flushCellCelebrations(cells) {
  for (const cell of cells) {
    if (!cell._celebrate) continue;
    const q = cell._celebrate; cell._celebrate = null;
    for (const [span, accepted] of q) celebrate(cell, span, accepted);
  }
}

function renderStep(cell, ev, frag) {
  const span = document.createElement("span");
  const isAR = cell.key === BASELINE;
  if (ev.kind !== "prefill") cell.verifySteps += 1;
  span.className = "step new" + (cell.verifySteps % 2 ? "" : " alt");
  const last = ev.ids.length - 1;
  const base = cell.tokIdx;
  cell.tokIdx += ev.ids.length;
  ev.pieces.forEach((piece, i) => {
    if (!piece) return;  // special token or incomplete UTF-8 byte sequence
    const t = document.createElement("span");
    t.textContent = piece;
    t.dataset.i = base + i;
    if (cell.entropy) paintEntropy(t, cell.entropy[base + i]);
    if (ev.kind === "merged") t.className = "merged";
    else if (ev.kind !== "prefill" && !isAR) t.className = i < last ? "acc" : "bonus";
    span.appendChild(t);
  });
  (frag || cell.out).appendChild(span);
  cell.tokens += ev.ids.length;
  if (ev.kind !== "prefill") cell.decodeTokens += ev.ids.length;
  // fireworks need the span attached to the DOM to measure its rects; defer
  // until renderBatch has appended the fragment.
  if (ev.kind === "verify" && !isAR) (cell._celebrate ||= []).push([span, ev.ids.length - 1]);
}

// --------------------------------------------------------- fireworks ----

const FX_COOLDOWN_MS = 150;  // per cell; a strictly longer run may still fire inside it

function fxThreshold() { return +$("fx").value; }  // fraction of the window, 0 = off

function toggleFx() {
  const sel = $("fx");
  if (+sel.value > 0) { state.fxPrev = sel.value; sel.value = "0"; }
  else sel.value = state.fxPrev || "0.5";
}

// Burst over the accepted run of one verify step; size grows with accepted / (window - 1).
function celebrate(cell, span, accepted) {
  const thr = fxThreshold();
  const maxAcc = Math.max(1, (state.budget || 16) - 1);
  const ratio = Math.min(1, accepted / maxAcc);
  if (!thr || accepted < 1 || ratio < thr) return;
  const now = performance.now();
  if (now < cell.fxUntil && accepted <= cell.fxLast) return;
  const accs = span.querySelectorAll(".acc");
  if (!accs.length) return;  // accepted tokens were all special / partial UTF-8
  // centre of the last line of the run, where the eye already is
  const rects = accs[accs.length - 1].getClientRects();
  const r = rects[rects.length - 1];
  if (!r) return;
  const box = cell.out.getBoundingClientRect();
  if (r.bottom < box.top || r.top > box.bottom) return;  // run is scrolled out of view
  cell.fxUntil = now + FX_COOLDOWN_MS;
  cell.fxLast = accepted;
  Fireworks.burst({
    anchor: cell.out, clip: cell.root,
    x: r.left + r.width / 2, y: r.top + r.height / 2,
    accepted, ratio, color: state.meta[cell.key].color,
  });
}

function cellTokS(cell) {
  if (cell.done && cell.summary) return cell.summary.tok_s;
  return cell.time > 0.15 ? cell.tokens / cell.time : null;
}

function cellTau(cell) {
  // tokens committed per verify step; the baseline is a plain AR (τ = 1) model.
  return cell.key === BASELINE ? 1 : (cell.verifySteps ? cell.decodeTokens / cell.verifySteps : null);
}

function updateStats(cell, t) {
  cell.time = t;
  const toks = cellTokS(cell);
  const tau = cellTau(cell);
  cell.st.tokens.textContent = cell.tokens;
  cell.st.tau.textContent = fmt(tau, 2);
  cell.st.toks.textContent = fmt(toks, 1);
  const sp = (toks && state.baselineTokS) ? toks / state.baselineTokS : null;
  cell.st.speedup.textContent = sp ? `${sp.toFixed(2)}×` : "–";
  cell.val.innerHTML = toks
    ? `<b>${toks.toFixed(0)}</b> tok/s<span class="sub">τ ${fmt(tau, 2)}${sp ? ` · ${sp.toFixed(2)}×` : ""}</span>`
    : "–";
}

function updateBars(refMax) {
  const cells = Object.values(state.cells);
  const tps = (c) => cellTokS(c) || 0;
  const tau = (c) => cellTau(c) || 0;
  // Each bar is normalised to the best method for its own metric, so a short
  // run behind a long one never hides it: the pale τ bar spans the full track
  // height while the thin tok/s bar floats in front (see style.css).
  const maxTps = refMax || Math.max(1, ...cells.map(tps));
  const maxTau = Math.max(1, ...cells.map(tau));
  for (const c of cells) {
    c.barTau.style.width = `${Math.min(100, 100 * tau(c) / (maxTau * 1.05))}%`;
    c.bar.style.width = `${Math.min(100, 100 * tps(c) / (maxTps * 1.05))}%`;
  }
}

function finishCell(cell, summary) {
  cell.done = true;
  cell.summary = summary;
  if (summary) {
    cell.time = summary.e2e;
    setChip(cell, `done · ${summary.e2e.toFixed(2)} s`);
    showEntStats(cell);
    if (summary.merged_steps) cell.chip.title += ` (${summary.merged_steps} coalesced chunks)`;
  }
  updateStats(cell, cell.time);
}

function finalizeRanks() {
  const done = Object.values(state.cells).filter((c) => c.done && c.summary && c.key !== BASELINE);
  done.sort((a, b) => b.summary.tok_s - a.summary.tok_s);
  done.forEach((c, i) => { c.rank.textContent = `#${i + 1}`; });
}

function refreshAllStats() { for (const c of Object.values(state.cells)) updateStats(c, c.time); }

// ----------------------------------------------------------- entropy ----

// Target entropy H(p_target(. | prefix)) at T=1, in nats, scored after the run.
// Colour: low -> purple, high -> bright green, on a log1p scale up to ENT_MAX so the
// many near-zero tokens still separate from each other.
const ENT_MAX = 4;
const ENT_STOPS = [[0, [123, 63, 196]], [0.5, [38, 140, 84]], [1, [140, 255, 110]]];

function entColor(h) {
  const t = Math.max(0, Math.min(1, Math.log1p(Math.max(0, h)) / Math.log1p(ENT_MAX)));
  let k = 1;
  while (k < ENT_STOPS.length - 1 && t > ENT_STOPS[k][0]) k++;
  const [t0, a] = ENT_STOPS[k - 1], [t1, b] = ENT_STOPS[k];
  const u = (t - t0) / (t1 - t0);
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(",")})`;
}

function paintEntropy(el, h) {
  if (h === null || h === undefined) return;
  el.style.setProperty("--h", entColor(h));
  el.dataset.h = h.toFixed(2);
  el.title = `target entropy ${h.toFixed(3)} nats`;
}

// Live mode: entropies arrive after the run, so colour the tokens already on screen.
function applyEntropy(cell, ent, stats) {
  cell.entropy = ent;
  cell.entStats = stats || null;
  for (const el of cell.out.querySelectorAll("[data-i]")) paintEntropy(el, ent[+el.dataset.i]);
}

function showEntStats(cell) {
  const s = cell.entStats;
  if (!s) { cell.entBox.hidden = true; return; }
  const f = (x) => fmt(x, 2);
  const swatch = (x) => (x === null || x === undefined) ? "" : `<i style="background:${entColor(x)}"></i>`;
  cell.entBox.innerHTML = cell.key === BASELINE
    ? `H̄ all ${swatch(s.all)}<b>${f(s.all)}</b>`
    : `H̄ accepted ${swatch(s.accepted)}<b>${f(s.accepted)}</b>`
      + ` · at rejection ${swatch(s.rejected)}<b>${f(s.rejected)}</b>`
      + ` · all ${swatch(s.all)}<b>${f(s.all)}</b>`;
  cell.entBox.title = cell.key === BASELINE ? `${s.n_all} tokens, nats`
    : `nats; ${s.n_accepted} accepted drafts, ${s.n_rejected} rejection points, `
      + `${s.n_full_bonus} fully-accepted steps (bonus H̄ ${f(s.full_bonus)})`;
  cell.entBox.hidden = false;
}

function syncEntShow() { document.body.classList.toggle("ent", $("entShow").checked); }

// ----------------------------------------------------------------- live ----

function setBusy(busy) {
  $("runBtn").disabled = busy;
  $("cancelBtn").disabled = !busy;
  $("loadBtn").disabled = busy;
  $("stopServersBtn").disabled = busy;
  $("replayBtn").disabled = busy || !state.rec;
}

async function runLive() {
  stopReplay();
  const prompt = $("prompt").value.trim();
  if (!prompt) { setMsg("Prompt is empty", true); return; }
  const body = {
    model: $("model").value,
    budget: +$("budget").value,
    preset: state.preset,
    prompt,
    max_new_tokens: +$("maxTokens").value || 8192,
    seed: $("seed").value === "" ? null : +$("seed").value,
    methods: currentMethods(),
    baseline: $("baseline").checked,
    entropy: $("entropy").checked,
  };
  state.mode = "live";
  state.rec = null;
  Fireworks.setLiveMode(true);  // lighter particles while steps stream in
  setBusy(true);
  setMsg("Methods run one after another on the GPU; press Replay afterwards to race them side by side.");
  const ctrl = new AbortController();
  state.abort = ctrl;
  try {
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: ctrl.signal,
    });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (_) { /* not json */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of block.split("\n")) {
          if (line.startsWith("data: ")) await handleEvent(JSON.parse(line.slice(6)));
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") setMsg(`Run failed: ${e.message}`, true);
  } finally {
    state.mode = "idle";
    state.abort = null;
    Fireworks.setLiveMode(false);
    setBusy(false);
  }
}

async function handleEvent(ev) {
  const cell = ev.method ? state.cells[ev.method] : null;
  switch (ev.type) {
    case "start": {
      const r = ev.recording;
      state.rec = { ...r, runs: {} };
      pendingSteps = [];
      buildCells(r.methods.filter((k) => k !== BASELINE), r.methods.includes(BASELINE));
      updateSubtitle(r);
      break;
    }
    case "status":
      if (cell) setChip(cell, ev.message, ev.state === "running" ? "running" : "");
      break;
    case "step": {
      if (!cell) break;
      const run = (state.rec.runs[ev.method] ||= { method: ev.method, status: "running", steps: [] });
      run.steps.push(ev);
      queueStep(cell, ev);
      break;
    }
    case "done":
      if (!cell) break;
      flushSteps();  // flush any queued render before finalising stats
      state.rec.runs[ev.method].status = "done";
      state.rec.runs[ev.method].summary = ev.summary;
      if (ev.method === BASELINE) state.baselineTokS = ev.summary.tok_s;
      finishCell(cell, ev.summary);
      refreshAllStats();
      updateBars();
      break;
    case "error":
      if (!cell) break;
      setChip(cell, "error", "error");
      cell.out.insertAdjacentHTML("beforeend", `<div class="err">${escapeHtml(ev.message)}</div>`);
      break;
    case "finished":
      flushSteps();  // drain the last frame's steps before recording handlers
      try { state.rec = await api(`/api/recordings/${encodeURIComponent(ev.recording_id)}`); } catch (_) { /* keep local copy */ }
      finalizeRanks();
      await refreshRecordings();
      $("recSelect").value = ev.recording_id;
      setMsg(`Recorded ${ev.recording_id}. Press "Replay race" (or R) to race all methods side by side.`);
      break;
    case "phase":
      setMsg(ev.message);
      break;
    case "entropy":
      if (!cell) break;
      if (state.rec.runs[ev.method]) {
        state.rec.runs[ev.method].entropy = ev.entropy;
        state.rec.runs[ev.method].entropy_stats = ev.stats;
      }
      applyEntropy(cell, ev.entropy, ev.stats);
      showEntStats(cell);
      break;
    case "fatal":
      setMsg(ev.message, true);
      break;
  }
}

async function cancelRun() {
  try { await api("/api/cancel", { method: "POST" }); } catch (e) { setMsg(e.message, true); }
  if (state.abort) state.abort.abort();
  setMsg("Cancelled.");
}

async function stopServers() {
  try { await api("/api/servers/stop", { method: "POST" }); setMsg("sglang servers stopped; GPU memory released."); }
  catch (e) { setMsg(e.message, true); }
}

// --------------------------------------------------------------- replay ----

function stopReplay() {
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = null;
  $("timer").classList.remove("running");
  if (state.mode === "replay") state.mode = "idle";
}

function applyRecordingToControls(rec) {
  $("model").value = rec.model;
  $("budget").value = String(rec.budget);
  setPreset(rec.preset);
  $("prompt").value = rec.prompt;
  $("q").value = "";  // recording's prompt may not come from the selected question
  $("maxTokens").value = rec.max_new_tokens;
  if (rec.seed !== null && rec.seed !== undefined) $("seed").value = rec.seed;
  refreshAvailability();
}

async function loadSelected() {
  const id = $("recSelect").value;
  if (!id) return;
  try {
    state.rec = await api(`/api/recordings/${encodeURIComponent(id)}`);
  } catch (e) { setMsg(e.message, true); return; }
  applyRecordingToControls(state.rec);
  $("replayBtn").disabled = false;
  replay(state.rec);
}

function replay(rec) {
  if (!rec || state.mode === "live") return;
  stopReplay();
  Fireworks.setLiveMode(false);
  const order = rec.methods || Object.keys(rec.runs);
  buildCells(order.filter((k) => k !== BASELINE), order.includes(BASELINE));
  updateSubtitle(rec);
  state.mode = "replay";

  const runs = [];
  for (const key of order) {
    const run = rec.runs[key], cell = state.cells[key];
    if (!cell) continue;
    if (!run || run.status !== "done") {
      setChip(cell, run ? "error" : "not run", "error");
      if (run && run.error) cell.out.insertAdjacentHTML("beforeend", `<div class="err">${escapeHtml(run.error)}</div>`);
      continue;
    }
    setChip(cell, "running", "running");
    cell.entropy = run.entropy || null;
    cell.entStats = run.entropy_stats || null;
    runs.push({ key, run, cell, ptr: 0 });
  }
  const base = rec.runs[BASELINE];
  state.baselineTokS = base && base.status === "done" ? base.summary.tok_s : null;
  const refMax = Math.max(1, ...runs.map((r) => r.run.summary.tok_s || 0));
  const speed = +$("speed").value || 1;
  const lead = 0.8;  // seconds of empty screen before the start
  const t0 = performance.now();
  $("timer").classList.add("running");

  const frame = (now) => {
    const el = ((now - t0) / 1000 - lead) * speed;
    let pending = 0;
    const touched = [];
    for (const r of runs) {
      if (r.cell.done) continue;
      const steps = r.run.steps;
      const evs = [];
      while (r.ptr < steps.length && steps[r.ptr].t <= el) { evs.push(steps[r.ptr]); r.ptr++; }
      if (evs.length) { renderBatch(r.cell, evs); touched.push(r.cell); }
      if (r.ptr >= steps.length && el >= r.run.summary.e2e) finishCell(r.cell, r.run.summary);
      else { pending += 1; updateStats(r.cell, Math.max(el, 0)); }
    }
    if (touched.length) flushCellCelebrations(touched);
    updateBars(refMax);
    $("timer").textContent = `${Math.max(el, 0).toFixed(2)} s`;
    if (pending > 0) {
      state.raf = requestAnimationFrame(frame);
    } else {
      finalizeRanks();
      stopReplay();
    }
  };
  state.raf = requestAnimationFrame(frame);
}

init().catch((e) => setMsg(`Init failed: ${e.message}`, true));
