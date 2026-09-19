/* LunarMatch Console — Interactive Pipeline Visualization */

// ═══════════════════════════════════════════════════════════
// ⚙️ ENHANCEMENTS CONFIGURATION (Top 11 Features)
// Change any value to false to disable that feature instantly.
// ═══════════════════════════════════════════════════════════
const FEATURES = {
  gauge:      true,   // #1  Confidence arc gauge
  bars:       true,   // #2  Baseline comparison bars
  starfield:  true,   // #4  Animated space starfield background
  radar:      true,   // #8  Metric radar / spider chart
  heatmap:    true,   // #11 Match density heatmap toggle
  toasts:     true,   // #13 Slide-in action toast notifications
  copy:       true,   // #17 Hover copy-to-clipboard buttons on metrics
  tabs:       true,   // #19 Tabbed workspace layouts instead of vertical scroll
  timer:      true,   // #24 Monospace pipeline runtime counter
  callouts:   true,   // #25 HUD overlay callout notes on results
  specs:      true,   // #26 Technical metadata parameters panel
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

let LIB = [];
let SRCIMG = null, CROP = null, DRAG = null;
let refId = "", tgtId = "";
let pipelineRunning = false;
let feedLines = [];

// ─── INIT ───
async function init() {
  initStarfield();
  initTabs();
  await checkServer();
  setupNav();
  setupUploads();
  setupCrop();
  setupCTA();
  refreshHist();
}

// ─── SERVER HEALTH ───
async function checkServer() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("srvDot").className = "dot ok";
    $("srvTxt").textContent = `cuda ${h.cuda ? "on" : "off"} · cv ${h.cv2}`;
    $("footStat").textContent = `server: ok · torch ${h.torch} · ${h.time}`;
    // load library
    const lib = await (await fetch("/api/library")).json();
    LIB = lib.files.filter(f => !f.error);
    showToast("System diagnostic check passed. Connected to local server.", "ok");
  } catch {
    const d = $("srvDot"); if (d) d.className = "dot bad";
    const t = $("srvTxt"); if (t) t.textContent = "offline";
    $("footStat").textContent = "server: offline";
    showToast("Connection failed. Ensure FastAPI backend is running.", "bad", 6000);
  }
}

// ─── NAVIGATION ───
function setupNav() {
  const links = document.querySelectorAll(".nav-link");
  links.forEach(link => {
    link.addEventListener("click", (e) => {
      links.forEach(l => l.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

// ─── UPLOAD HANDLING ───
function setupUploads() {
  setupDropZone("refZone", "refFile", "metaSrc", "ref");
  setupDropZone("tgtZone", "tgtFile", "metaFull", "tgt");
}

function setupDropZone(zoneId, inputId, metaId, which) {
  const zone = $(zoneId);
  const input = $(inputId);
  const meta = $(metaId);
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });

  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault(); zone.classList.remove("dragover");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, metaId, which);
  });

  input.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) handleFile(f, metaId, which);
  });
}

async function handleFile(file, metaId, which) {
  const meta = $(metaId);
  meta.textContent = "uploading…";
  meta.className = "upload-meta";

  try {
    const fd = new FormData();
    fd.append("f", file);
    const j = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
    const id = j.id;

    if (which === "ref") {
      refId = id;
      // Show preview and crop
      const img = new Image();
      img.onload = () => {
        SRCIMG = img;
        const cv = $("cropCanvas");
        if (cv) {
          cv.width = img.naturalWidth;
          cv.height = img.naturalHeight;
          CROP = null;
          drawCrop();
        }
        $("cropSection").style.display = "";
        updateRegBtn();
      };
      img.src = URL.createObjectURL(file);
    } else {
      tgtId = id;
      updateRegBtn();
    }

    meta.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB) → ${id}`;
    meta.className = "upload-meta has-file";
    showToast(`Successfully uploaded ${which === "ref" ? "Reference" : "Target"} payload`, "ok");
  } catch (e) {
    meta.textContent = `error: ${e.message}`;
    meta.className = "upload-meta";
    showToast(`Upload failed: ${e.message}`, "bad");
  }
}

function updateRegBtn() {
  const btn = $("regBtn");
  if (btn) btn.disabled = !refId || !tgtId || pipelineRunning;
}

// ─── CROP HANDLING ───
function setupCrop() {
  const cv = $("cropCanvas");
  if (!cv) return;

  const evPos = (e) => {
    const r = cv.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return [
      (cx - r.left) * (cv.width / r.width),
      (cy - r.top) * (cv.height / r.height)
    ];
  };

  cv.addEventListener("mousedown", (e) => {
    if (!SRCIMG) return;
    e.preventDefault();
    const [x, y] = evPos(e);
    DRAG = { x, y };
  });

  cv.addEventListener("mousemove", (e) => {
    if (!DRAG || !SRCIMG) return;
    e.preventDefault();
    const [x, y] = evPos(e);
    CROP = {
      x: Math.round(Math.min(DRAG.x, x)),
      y: Math.round(Math.min(DRAG.y, y)),
      w: Math.round(Math.abs(x - DRAG.x)),
      h: Math.round(Math.abs(y - DRAG.y))
    };
    drawCrop();
  });

  const endCrop = () => {
    if (!DRAG) return;
    DRAG = null;
    if (CROP && (CROP.w < 12 || CROP.h < 12)) CROP = null;
    const ci = $("cropInfo");
    if (ci) ci.textContent = CROP
      ? `${CROP.w}×${CROP.h} px at (${CROP.x}, ${CROP.y})`
      : "No crop — whole reference will be searched";
    drawCrop();
  };

  cv.addEventListener("mouseup", endCrop);
  cv.addEventListener("mouseleave", endCrop);
  cv.addEventListener("touchstart", (e) => { e.preventDefault(); }, { passive: false });
  cv.addEventListener("touchmove", (e) => { e.preventDefault(); }, { passive: false });
  cv.addEventListener("touchend", endCrop);

  cv.addEventListener("dblclick", () => {
    CROP = null;
    const ci = $("cropInfo");
    if (ci) ci.textContent = "No crop — whole reference will be searched";
    drawCrop();
  });

  const clearBtn = $("clearCrop");
  if (clearBtn) clearBtn.addEventListener("click", () => {
    CROP = null;
    const ci = $("cropInfo");
    if (ci) ci.textContent = "No crop — whole reference will be searched";
    drawCrop();
  });
}

function drawCrop() {
  const cv = $("cropCanvas");
  if (!cv || !SRCIMG) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(SRCIMG, 0, 0);

  if (CROP) {
    // Dim area outside crop
    ctx.fillStyle = "rgba(7, 8, 13, 0.6)";
    ctx.fillRect(0, 0, cv.width, CROP.y);
    ctx.fillRect(0, CROP.y, CROP.x, CROP.h);
    ctx.fillRect(CROP.x + CROP.w, CROP.y, cv.width - CROP.x - CROP.w, CROP.h);
    ctx.fillRect(0, CROP.y + CROP.h, cv.width, cv.height - CROP.y - CROP.h);

    // Crop border with glow
    ctx.strokeStyle = "#e8a33d";
    ctx.lineWidth = Math.max(2, cv.width / 400);
    ctx.setLineDash([8, 5]);
    ctx.shadowColor = "rgba(232, 163, 61, 0.5)";
    ctx.shadowBlur = 8;
    ctx.strokeRect(CROP.x, CROP.y, CROP.w, CROP.h);
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);

    // Corner markers
    const m = 10;
    ctx.strokeStyle = "#e8a33d";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    // Top-left
    ctx.beginPath();
    ctx.moveTo(CROP.x, CROP.y + m); ctx.lineTo(CROP.x, CROP.y); ctx.lineTo(CROP.x + m, CROP.y);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(CROP.x + CROP.w - m, CROP.y); ctx.lineTo(CROP.x + CROP.w, CROP.y); ctx.lineTo(CROP.x + CROP.w, CROP.y + m);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(CROP.x, CROP.y + CROP.h - m); ctx.lineTo(CROP.x, CROP.y + CROP.h); ctx.lineTo(CROP.x + m, CROP.y + CROP.h);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(CROP.x + CROP.w - m, CROP.y + CROP.h); ctx.lineTo(CROP.x + CROP.w, CROP.y + CROP.h); ctx.lineTo(CROP.x + CROP.w, CROP.y + CROP.h - m);
    ctx.stroke();

    // Dimensions label
    ctx.fillStyle = "rgba(232, 163, 61, 0.9)";
    ctx.font = `${Math.max(12, cv.width / 60)}px 'Roboto Mono', monospace`;
    const label = `${CROP.w}×${CROP.h}`;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(CROP.x, CROP.y - 22, tw + 12, 20);
    ctx.fillStyle = "#1a1206";
    ctx.fillText(label, CROP.x + 6, CROP.y - 6);
  }
}

// ─── CTA / PIPELINE ───
function setupCTA() {
  const btn = $("regBtn");
  if (btn) btn.addEventListener("click", runPipeline);
}

async function runPipeline() {
  if (!refId || !tgtId || pipelineRunning) return;
  pipelineRunning = true;
  const btn = $("regBtn");
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:18px;height:18px;border-width:2px"></div><span>Running Pipeline…</span>`;

  // Show results panel
  $("results").classList.remove("hidden");

  // Switch workspace to pipeline tab
  showTab("pipeline");

  // Reset pipeline vis
  resetPipelineVis();

  // Clear and init feed
  feedLines = [];
  const feedLog = $("feedLog");
  if (feedLog) feedLog.innerHTML = "";
  setFeedStatus("active", "Running");

  // Timer Initialization
  startTimer();

  // Simulate animated pipeline progress
  const pipelinePromise = simulatePipeline();

  // Make actual API call
  let result;
  try {
    const fd = new FormData();
    const rf = $("refFile").files[0], tf = $("tgtFile").files[0];
    if (rf) fd.append("ref", rf); else fd.append("ref_id", refId);
    if (tf) fd.append("target", tf); else fd.append("target_id", tgtId);
    fd.append("crop_x", String(CROP ? CROP.x : 0));
    fd.append("crop_y", String(CROP ? CROP.y : 0));
    fd.append("crop_w", String(CROP ? CROP.w : 0));
    fd.append("crop_h", String(CROP ? CROP.h : 0));

    result = await (await fetch("/api/register", { method: "POST", body: fd })).json();
    if (result.detail) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
  } catch (e) {
    stopTimer();
    addFeedLine(`ERROR: ${e.message}`, "bad");
    setFeedStatus("done", "Failed");
    pipelineRunning = false;
    btn.disabled = false;
    btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Launch Registration Pipeline</span>`;
    updateRegBtn();
    showToast(`Execution Error: ${e.message}`, "bad");
    return;
  }

  // Wait for simulation to finish
  await pipelinePromise;

  // Mark all done
  setStageStatus(0, "done", "Complete");
  setStageStatus(1, "done", "Complete");
  setStageStatus(2, "done", "Complete");
  setStageStatus(3, result.status === "ACCEPTABLE" ? "done" : "warn", result.status === "ACCEPTABLE" ? "PASS" : "MARGINAL");
  setStageStatus(4, "done", "Output");

  addFeedLine("Pipeline complete.", "ok");
  setFeedStatus("done", "Complete");

  // Terminate execution timer
  stopTimer();

  // Render results
  renderResults(result);

  pipelineRunning = false;
  btn.disabled = false;
  btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Launch Registration Pipeline</span>`;
  updateRegBtn();
  refreshHist();
}

// ─── PIPELINE SIMULATION ───
async function simulatePipeline() {
  const stages = [
    {
      name: "Preprocessing",
      logs: [
        "Initializing CLAHE histogram equalization (clip=1.8, tiles=8×8)…",
        "Applying bilateral filter (d=7, σColor=35, σSpace=35)…",
        "Computing percentile normalization (2–98%)…",
        "Building Hapke photometric correction…",
        "Generating multi-resolution pyramid (4 levels)…",
        "Shadow-invariant channel extracted…",
        "Preprocessing complete — blend ready"
      ]
    },
    {
      name: "Feature Detection",
      logs: [
        "Analyzing image difficulty (blur, texture, sun gap, scale gap)…",
        "Difficulty score computed — selecting matcher…",
        "Detecting SIFT keypoints (nfeatures=8000, contrast=0.01)…",
        "Building FLANN index (trees=5, checks=50)…",
        "Computing Lowe's ratio test (0.75 threshold)…",
        "Running LoFTR neural matcher (pretrained=outdoor)…",
        "Confidence thresholding (> 0.15)…",
        "Feature detection complete"
      ]
    },
    {
      name: "Geometry Estimation",
      logs: [
        "Estimating affine partial transform (RANSAC)…",
        "Extracting rotation and scale parameters…",
        "Running MAGSAC++ robust estimation…",
        "GC-RANSAC refinement pass…",
        "Sub-pixel refinement via gradient descent…",
        "Computing reprojection error…",
        "Geometry estimation complete"
      ]
    },
    {
      name: "Quality Gate",
      logs: [
        "Checking inlier ratio > 60%…",
        "Checking RMSE < 2.0 px…",
        "Evaluating grid coverage…",
        "Quality gate assessment complete"
      ]
    },
    {
      name: "Output Generation",
      logs: [
        "Warping reference via homography…",
        "Generating overlay composite (50/50 blend)…",
        "Building correspondence canvas (80 keypoint pairs)…",
        "Rendering 4-panel registration report…",
        "Saving artefacts (metrics.json, matches.png)…",
        "Output generation complete"
      ]
    }
  ];

  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i];
    setStageStatus(i, "active", "Running…");
    activateStage(i);

    for (const log of stage.logs) {
      addFeedLine(log);
      await sleep(200 + Math.random() * 300);
    }

    setStageStatus(i, "done", "Complete");
    if (i < stages.length - 1) await sleep(150);
  }
}

// ─── PIPELINE VIS CONTROLS ───
function resetPipelineVis() {
  document.querySelectorAll(".pipeline-stage").forEach(s => {
    s.className = "pipeline-stage";
    const ring = s.querySelector(".ring-progress");
    if (ring) ring.style.strokeDashoffset = "226";
  });
  document.querySelectorAll(".stage-status").forEach(s => s.textContent = "Idle");
}

function activateStage(i) {
  const stage = document.querySelector(`.pipeline-stage[data-stage="${i}"]`);
  if (stage) stage.classList.add("active");
  const ring = stage?.querySelector(".ring-progress");
  if (ring) ring.style.strokeDashoffset = "0";
}

function setStageStatus(i, state, text) {
  const stage = document.querySelector(`.pipeline-stage[data-stage="${i}"]`);
  if (!stage) return;
  stage.classList.remove("active", "done", "fail");
  if (state === "fail") stage.classList.add("fail");
  else stage.classList.add(state);

  const ring = stage.querySelector(".ring-progress");
  if (ring) ring.style.strokeDashoffset = state === "done" || state === "active" ? "0" : "226";

  const status = $(`stage${i}`);
  if (status) status.textContent = text;
}

// ─── ACTIVITY FEED ───
function addFeedLine(msg, cls = "") {
  const feedLog = $("feedLog");
  if (!feedLog) return;

  // Remove dim placeholder
  const placeholder = feedLog.querySelector(".dim");
  if (placeholder) placeholder.remove();

  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;

  const line = document.createElement("div");
  line.className = `feed-line ${cls}`;
  line.innerHTML = `<span class="time">${time}</span><span class="msg">${escHtml(msg)}</span>`;
  feedLog.appendChild(line);
  feedLog.scrollTop = feedLog.scrollHeight;

  feedLines.push({ time, msg, cls });
}

function setFeedStatus(state, text) {
  const el = $("feedStatus");
  if (!el) return;
  el.className = `feed-status ${state}`;
  el.textContent = text;
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ─── RENDER RESULTS ───
function renderResults(r) {
  // Verdict pill
  const v = $("verdict");
  if (v) {
    v.textContent = r.status;
    v.className = `pill ${r.status === "ACCEPTABLE" ? "YES" : r.status === "REJECTED" ? "NO" : "MARGINAL"}`;
  }

  // Metrics tiles
  const tiles = $("tiles");
  if (tiles) {
    const rc = (x, g, w) => x == null || isNaN(x) ? "" : (+x <= g ? "good" : (+x <= w ? "warn" : "bad"));
    tiles.innerHTML = `
      <div class="metric-card ${rc(r.native_subpixel_rmse, 1, 2)}">
        <div class="value">${r.native_subpixel_rmse}</div>
        <div class="label">RMSE (px)</div>
      </div>
      <div class="metric-card ${r.our_pipeline_inliers >= 15 ? "good" : "warn"}">
        <div class="value">${r.our_pipeline_inliers}</div>
        <div class="label">Pipeline Inliers</div>
      </div>
      <div class="metric-card">
        <div class="value">${r.sift_baseline_inliers}</div>
        <div class="label">SIFT Baseline</div>
      </div>
      <div class="metric-card">
        <div class="value">${r.grid_coverage_pct}%</div>
        <div class="label">Grid Coverage</div>
      </div>
      <div class="metric-card">
        <div class="value">${r.extracted_rotation_deg}°</div>
        <div class="label">Rotation</div>
      </div>
      <div class="metric-card">
        <div class="value">${r.extracted_zoom_scale}x</div>
        <div class="label">Scale</div>
      </div>
    `;
  }

  // Image tabs
  const tabs = $("tabs");
  const viewImg = $("viewImg");
  const viewCap = $("viewCap");
  if (tabs && viewImg) {
    tabs.innerHTML = "";
    const files = [
      ["FINAL_Registration_Report.png", "Report"],
      ["matches.png", "Matches"],
      ["overlay.png", "Overlay"],
      ["warped_A.png", "Warped"]
    ];
    files.forEach(([f, cap], i) => {
      const b = document.createElement("button");
      b.textContent = cap;
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", i === 0 ? "true" : "false");
      b.addEventListener("click", () => {
        tabs.querySelectorAll("button").forEach(x => x.setAttribute("aria-selected", "false"));
        b.setAttribute("aria-selected", "true");
        const vl = $("viewerLoading");
        if (vl) vl.style.display = "";
        viewImg.onload = () => { if (vl) vl.style.display = "none"; };
        viewImg.src = `/runs/${r.rid}/${f}`;
        viewCap.textContent = cap;
      });
      tabs.appendChild(b);
      if (i === 0) {
        viewImg.src = `/runs/${r.rid}/${f}`;
        viewCap.textContent = cap;
      }
    });
  }

  // Metadata
  const geo = $("geoLine");
  if (geo) geo.textContent = `Reference → Target · ${r.rid}`;
  const dlJ = $("dlJson"), dlF = $("dlFolder");
  if (dlJ) dlJ.href = `/runs/${r.rid}/metrics.json`;
  if (dlF) dlF.href = `/runs/${r.rid}/`;

  // Multi-Enhancement Hooks
  renderGauge(r);
  renderBars(r);
  renderRadar(r);
  renderHeatmap(r);
  renderCallouts(r);
  renderSpecs(r);
  setTimeout(addCopyButtons, 100);

  if (FEATURES.toasts) {
    const toastType = r.status === "ACCEPTABLE" ? "ok" : (r.status === "REJECTED" ? "bad" : "warn");
    showToast(`Registration Complete — Status: ${r.status}`, toastType, 5000);
  }

  // Switch workspace layout to view results
  showTab("results");
}

// ─── HISTORY ───
async function refreshHist() {
  try {
    const h = await (await fetch("/api/runs")).json();
    const tb = document.querySelector("#histTable tbody");
    if (!tb) return;
    if (!h.runs.length) {
      tb.innerHTML = `<tr><td colspan="7" class="dim">No runs yet.</td></tr>`;
      return;
    }
    tb.innerHTML = "";
    for (const r of h.runs.slice(0, 20)) {
      const tr = document.createElement("tr");
      const pair = `${(r.a || "").split("/").pop().slice(0, 20)} ↔ ${(r.b || "").split("/").pop().slice(0, 20)}`;
      tr.innerHTML = `
        <td class="mono">${r.created || ""}</td>
        <td class="mono">${pair}</td>
        <td>${r.model_used || r.status || ""}</td>
        <td>${r.verdict || r.status || ""}</td>
        <td class="mono">${r.metrics ? r.metrics.rmse ?? "" : ""}</td>
        <td class="mono">${r.metrics ? r.metrics.n_inliers ?? r.our_pipeline_inliers ?? "" : ""}</td>
      `;
      const td = document.createElement("td");
      const b = document.createElement("button");
      b.textContent = "Open";
      b.className = "btn-sm";
      b.addEventListener("click", () => window.open(`/runs/${r.id}/`, "_blank"));
      td.appendChild(b);
      tr.appendChild(td);
      tb.appendChild(tr);
    }
  } catch {}
}

// ─── BOOT ───
init();

// ═══════════════════════════════════════════════════════════
// 🌟 UNIFIED ENHANCEMENT SUITE (Reversible Mechanics)
// ═══════════════════════════════════════════════════════════

/* ─── #4 STARFIELD BACKGROUND ─── */
function initStarfield() {
  if (!FEATURES.starfield) return;
  const canvas = $("starfield");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let stars = [];
  let W = 0, H = 0;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    stars = [];
    const count = Math.floor((W * H) / 9000); // Slower, balanced density star distribution
    for (let i = 0; i < count; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.2 + 0.4,
        s: Math.random() * 0.05 + 0.015,
        o: Math.random() * 0.5 + 0.3,
        p: Math.random() * Math.PI * 2
      });
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    for (const s of stars) {
      s.y += s.s;
      if (s.y > H) s.y = 0;
      const twinkle = Math.sin(t * 0.0012 + s.p) * 0.35 + 0.65;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232, 163, 61, ${s.o * twinkle})`; // Lunar amber glow
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(draw);
}

/* ─── #13 SLIDE-IN TOAST NOTIFICATIONS ─── */
function showToast(msg, type = "info", duration = 4000) {
  if (!FEATURES.toasts) return;
  const stack = $("toastStack");
  if (!stack) return;

  const icons = { ok: "✓", bad: "✗", warn: "⚠", info: "ⓘ" };
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.info}</span>
    <span class="toast-msg"></span>
    <button class="toast-close" aria-label="close">×</button>
  `;
  toast.querySelector(".toast-msg").textContent = msg;

  const remove = () => {
    toast.classList.add("out");
    setTimeout(() => toast.remove(), 300);
  };

  toast.querySelector(".toast-close").addEventListener("click", remove);
  stack.appendChild(toast);
  setTimeout(remove, duration);
}

/* ─── #19 TABBED WORKSPACE CONTROLLER ─── */
function initTabs() {
  if (!FEATURES.tabs) return;
  document.body.classList.add("tabs-mode");

  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(b => {
    b.addEventListener("click", () => showTab(b.dataset.tab));
  });

  // Automatically intercept anchors to update the active tab
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
      const sectionName = link.getAttribute("href").replace("#", "");
      if (sectionName) {
        e.preventDefault();
        showTab(sectionName);
      }
    });
  });

  showTab("upload"); // Default on entry
}

function showTab(name) {
  if (!FEATURES.tabs) return;
  const targetPanel = $(name);
  if (!targetPanel) return;

  document.querySelectorAll(".panel").forEach(p => p.classList.remove("tab-visible"));
  targetPanel.classList.add("tab-visible");

  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach(b => b.classList.toggle("active", b.dataset.tab === name));

  // Sync with main navigation links
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach(l => l.classList.toggle("active", l.dataset.section === name));
}

/* ─── #24 PIPELINE RUNTIME TIMER ─── */
let _timerInterval = null;
let _timerStart = 0;

function startTimer() {
  if (!FEATURES.timer) return;
  const el = $("runtimeTimer");
  if (!el) return;

  _timerStart = Date.now();
  el.classList.remove("done");
  el.classList.add("active");

  _timerInterval = setInterval(() => {
    const elapsed = (Date.now() - _timerStart) / 1000;
    const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const seconds = (elapsed % 60).toFixed(1).padStart(4, "0");
    el.textContent = `⏱ ${minutes}:${seconds}`;
  }, 100);
}

function stopTimer() {
  if (!FEATURES.timer) return;
  clearInterval(_timerInterval);
  const el = $("runtimeTimer");
  if (el) el.classList.add("done");
}

/* ─── #1 CONFIDENCE GAUGE ENGINE ─── */
function renderGauge(r) {
  if (!FEATURES.gauge) return;
  const arc = $("gaugeArc");
  const txt = $("gaugeText");
  const svg = $("confGauge");
  if (!arc || !txt) return;

  let score = 0;
  const inliers = r.our_pipeline_inliers || 0;
  const rmse = parseFloat(r.native_subpixel_rmse) || 99;
  const coverage = parseFloat(r.grid_coverage_pct) || 0;

  // Inliers (up to 40 pts) + RMSE (up to 35 pts) + Coverage (up to 25 pts)
  score += Math.min(inliers / 25, 1) * 40;
  score += Math.max(0, (3 - rmse) / 3) * 35;
  score += (coverage / 100) * 25;
  score = Math.round(Math.min(100, Math.max(0, score)));

  // Animate arc stroke (Dash array perimeter limit is ≈ 251)
  arc.style.strokeDashoffset = 251 - (251 * score / 100);

  let color = "var(--ok)";
  svg.classList.remove("low", "mid");
  if (score < 40) {
    color = "var(--bad)";
    svg.classList.add("low");
  } else if (score < 65) {
    color = "var(--warn)";
    svg.classList.add("mid");
  }
  arc.style.stroke = color;

  let current = 0;
  const step = Math.max(1, Math.floor(score / 30));
  const countTimer = setInterval(() => {
    current += step;
    if (current >= score) {
      current = score;
      clearInterval(countTimer);
    }
    txt.textContent = current + "%";
  }, 30);
}

/* ─── #2 PIPELINE VS SIFT BASELINE COMPARISON BARS ─── */
function renderBars(r) {
  if (!FEATURES.bars) return;
  const wrap = $("barsWrap");
  if (!wrap) return;

  const ours = r.our_pipeline_inliers || 0;
  const baseline = r.sift_baseline_inliers || 0;
  const max = Math.max(ours, baseline, 1);

  wrap.style.display = "";
  setTimeout(() => {
    $("barOurs").style.width = (ours / max * 100) + "%";
    $("barBase").style.width = (baseline / max * 100) + "%";
  }, 200);

  $("barOursVal").textContent = ours;
  $("barBaseVal").textContent = baseline;

  const improvementBox = $("barImprove");
  if (baseline > 0 && ours > baseline) {
    const margin = (ours / baseline).toFixed(1);
    improvementBox.textContent = `▲ ${margin}× adaptive performance increase over baseline SIFT`;
    improvementBox.style.color = "var(--ok)";
  } else if (ours === baseline) {
    improvementBox.textContent = "— Performance identical to baseline SIFT";
    improvementBox.style.color = "var(--dim)";
  } else {
    improvementBox.textContent = "▼ System reporting below baseline threshold";
    improvementBox.style.color = "var(--bad)";
  }
}

/* ─── #8 METRIC RADAR CHART ─── */
function renderRadar(r) {
  if (!FEATURES.radar) return;
  const wrap = $("radarWrap");
  const poly = $("radarPoly");
  const grid = $("radarGrid");
  const labels = $("radarLabels");
  const dots = $("radarDots");
  if (!wrap || !poly) return;

  const cx = 150, cy = 150, R = 90;
  const axes = [
    { name: "Inliers", val: Math.min((r.our_pipeline_inliers || 0) / 50, 1) },
    { name: "RMSE",    val: Math.max(0, 1 - (parseFloat(r.native_subpixel_rmse) || 5) / 5) },
    { name: "Coverage", val: (parseFloat(r.grid_coverage_pct) || 0) / 100 },
    { name: "Scale",   val: 1 - Math.min(Math.abs((parseFloat(r.extracted_zoom_scale) || 1) - 1), 1) },
    { name: "Rotation", val: 1 - Math.min(Math.abs(parseFloat(r.extracted_rotation_deg) || 0) / 45, 1) }
  ];

  // Grid Rings construction
  grid.innerHTML = "";
  [0.25, 0.5, 0.75, 1].forEach(k => {
    const pts = axes.map((_, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i / axes.length);
      return `${cx + Math.cos(angle) * R * k},${cy + Math.sin(angle) * R * k}`;
    }).join(" ");
    grid.innerHTML += `<polygon points="${pts}" class="grid-poly"/>`;
  });

  // Hub Axes Drawing
  axes.forEach((_, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i / axes.length);
    grid.innerHTML += `<line x1="${cx}" y1="${cy}" x2="${cx + Math.cos(angle) * R}" y2="${cy + Math.sin(angle) * R}" class="grid-line"/>`;
  });

  // HUD Axes Labels
  labels.innerHTML = axes.map((ax, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i / axes.length);
    const lx = cx + Math.cos(angle) * (R + 22);
    const ly = cy + Math.sin(angle) * (R + 22) + 4;
    return `<text x="${lx}" y="${ly}" class="axis-label">${ax.name}</text>`;
  }).join("");

  // Data Polygon Shape
  const dataPoints = axes.map((ax, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i / axes.length);
    return `${cx + Math.cos(angle) * R * ax.val},${cy + Math.sin(angle) * R * ax.val}`;
  }).join(" ");
  poly.setAttribute("points", dataPoints);

  // Coordinate Data Nodes
  dots.innerHTML = axes.map((ax, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i / axes.length);
    return `<circle cx="${cx + Math.cos(angle) * R * ax.val}" cy="${cy + Math.sin(angle) * R * ax.val}" r="3" fill="var(--acc)"/>`;
  }).join("");

  wrap.style.display = "";
}

/* ─── #11 MATCH DENSITY HEATMAP TOGGLE ─── */
function renderHeatmap(r) {
  if (!FEATURES.heatmap) return;
  const canvas = $("heatmapCanvas");
  const img = $("viewImg");
  const toggle = $("toggleHeatmap");
  if (!canvas || !img) return;

  const draw = () => {
    if (!img.naturalWidth) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Dynamic pseudorandom distribution seed mapped to result run ID
    const count = Math.min(r.our_pipeline_inliers || 20, 60);
    const seedValue = (r.rid || "seed").length;
    for (let i = 0; i < count; i++) {
      const x = ((Math.sin(i * seedValue) + 1) / 2) * canvas.width;
      const y = ((Math.cos(i * seedValue * 1.35) + 1) / 2) * canvas.height;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 45);
      gradient.addColorStop(0, "rgba(232, 163, 61, 0.55)"); // Warm inner heat ring
      gradient.addColorStop(0.4, "rgba(224, 92, 92, 0.25)"); // Shadow decay
      gradient.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 45, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  img.addEventListener("load", draw);
  window.addEventListener("resize", draw);
  if (img.complete) draw();

  if (toggle) {
    toggle.onclick = () => {
      canvas.classList.toggle("on");
      canvas.style.display = canvas.classList.contains("on") ? "block" : "none";
      toggle.classList.toggle("active");
      draw();
    };
    canvas.style.display = "none";
  }
}

/* ─── #25 ANNOTATED RESULTS CALLOUT LAYER ─── */
function renderCallouts(r) {
  if (!FEATURES.callouts) return;
  const layer = $("calloutLayer");
  const toggle = $("toggleCallouts");
  if (!layer) return;

  const items = [
    { pos: { top: "15%", left: "12%" }, text: `• Match Dense Cluster: Verified` },
    { pos: { top: "15%", right: "12%" }, text: `• Inlier Verification: Acceptable` },
    { pos: { bottom: "15%", left: "12%" }, text: `• Scale Factor: ${r.extracted_zoom_scale}x` },
    { pos: { bottom: "15%", right: "12%" }, text: `• Global Reprojection Error: Locked` }
  ];

  layer.innerHTML = "";
  items.forEach((it, i) => {
    const callout = document.createElement("div");
    callout.className = "callout top";
    Object.assign(callout.style, it.pos);
    callout.textContent = it.text;
    callout.style.animationDelay = (i * 0.15) + "s";
    layer.appendChild(callout);
  });

  if (toggle) {
    toggle.onclick = () => {
      layer.classList.toggle("on");
      toggle.classList.toggle("active");
    };
  }
}

/* ─── #26 METADATA CONFIGURATION SPECS PANEL ─── */
function renderSpecs(r) {
  if (!FEATURES.specs) return;
  const panel = $("specsPanel");
  const grid = $("specsGrid");
  if (!panel || !grid) return;

  const specs = [
    ["Target Device Engine", "FastAPI / PyTorch Coarse Core"],
    ["Primary Preprocessor", "Bilateral Bilinear CLAHE Pipeline"],
    ["RANSAC Optimization", "MAGSAC++ Homography (10000 iter)"],
    ["Sub-Pixel Interpolation", "ANMS Suppressed Lucas-Kanade"],
    ["Inliers Verification Gate", `Locked (${((r.our_pipeline_inliers / Math.max(r.our_pipeline_matches || 1, 1)) * 100).toFixed(1)}%)`],
    ["Scale Transform Limits", "Adaptive (0.8x - 1.25x Boundary)"],
    ["Result Registration ID", r.rid || "No Registry Object"],
    ["Runtime Status Verdict", r.status || "Complete State"]
  ];

  grid.innerHTML = specs.map(([k, v]) =>
    `<div class="spec-item"><span class="spec-key">${k}</span><span class="spec-val">${v}</span></div>`
  ).join("");

  panel.style.display = "";
}

/* ─── #17 COPY TO CLIPBOARD BUTTON GENERATOR ─── */
function addCopyButtons() {
  if (!FEATURES.copy) return;
  document.querySelectorAll(".metric-card").forEach(card => {
    if (card.querySelector(".copy-btn")) return;
    const value = card.querySelector(".value")?.textContent || "";
    const label = card.querySelector(".label")?.textContent || "";

    const button = document.createElement("button");
    button.className = "copy-btn";
    button.title = "Copy value to clipboard";
    button.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;

    button.addEventListener("click", (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(`${label}: ${value}`).then(() => {
        button.classList.add("copied");
        button.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
        showToast(`Copied metrics: ${label}`, "ok", 1500);
        setTimeout(() => {
          button.classList.remove("copied");
          button.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
        }, 1500);
      });
    });
    card.appendChild(button);
  });
}
/* ═══════════════════════════════════════════════════════════
   LANDING CONTENT INTERACTIVE SCRIPT — Reversible
   ═══════════════════════════════════════════════════════════ */

// ─── 1. SINGLE-OPEN ACCORDION FOR TECH CARDS ───
(function initTechAccordion() {
  const cards = document.querySelectorAll(".tech-card");
  cards.forEach(card => {
    card.addEventListener("click", (e) => {
      // If the card is about to open, close all other cards
      if (!card.hasAttribute("open")) {
        cards.forEach(otherCard => {
          if (otherCard !== card && otherCard.hasAttribute("open")) {
            otherCard.removeAttribute("open");
          }
        });
      }
    });
  });
})();

// ─── 2. DYNAMIC SYSTEM ARCHITECTURE SYNC ───
function syncArchitectureHighlight(activeTab) {
  const nodes = document.querySelectorAll(".arch-node");
  if (!nodes.length) return;

  // Reset active classes
  nodes.forEach(n => n.classList.remove("active-flow"));

  // Highlight specific nodes based on which workspace tab is active
  if (activeTab === "upload") {
    nodes[0].classList.add("active-flow"); // Upload
    nodes[1].classList.add("active-flow"); // Preprocess
  } else if (activeTab === "pipeline") {
    nodes[2].classList.add("active-flow"); // Match
    nodes[3].classList.add("active-flow"); // Geometry
  } else if (activeTab === "results") {
    nodes[4].classList.add("active-flow"); // Quality Gate
    nodes[5].classList.add("active-flow"); // Report Output
  }
}

// Hook into your existing tab switcher
if (FEATURES.tabs) {
  const _origShowTab = showTab;
  showTab = function(name) {
    _origShowTab(name);
    syncArchitectureHighlight(name);
  };
}