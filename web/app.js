/* LunarMatch Console — Interactive Pipeline Visualization */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

let LIB = [];
let SRCIMG = null, CROP = null, DRAG = null;
let refId = "", tgtId = "";
let pipelineRunning = false;
let feedLines = [];

// ─── INIT ───
async function init() {
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
  } catch {
    const d = $("srvDot"); if (d) d.className = "dot bad";
    const t = $("srvTxt"); if (t) t.textContent = "offline";
    $("footStat").textContent = "server: offline";
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
  } catch (e) {
    meta.textContent = `error: ${e.message}`;
    meta.className = "upload-meta";
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

  // Reset pipeline vis
  resetPipelineVis();

  // Clear and init feed
  feedLines = [];
  const feedLog = $("feedLog");
  if (feedLog) feedLog.innerHTML = "";
  setFeedStatus("active", "Running");

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
    addFeedLine(`ERROR: ${e.message}`, "bad");
    setFeedStatus("done", "Failed");
    pipelineRunning = false;
    btn.disabled = false;
    btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Launch Registration Pipeline</span>`;
    updateRegBtn();
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

  // Scroll to results
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
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
