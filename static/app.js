/* FaceLab UI — multi-model attendance + benchmarking harness.
 *
 * Two workspaces share one set of static assets:
 *   • Attendance: enroll students, pick a detector/recognizer pair, scan frames,
 *     and re-score the saved scan instantly as the threshold sliders move.
 *   • Benchmark: build ground-truth datasets and compare model grids on
 *     calibrated F1, size-stratified recall, and latency.
 *
 * Neural inference is only ever triggered by a scan or a benchmark run; every
 * slider move re-uses the cached embeddings via /api/results/{id}/sweep.
 */

const STORAGE_KEY = "facelab.selection.v3";
const TOKEN_KEY = "facelab.token";

const state = {
  // roster / registry
  people: [],
  registry: { detectors: [], recognizers: [] },
  packs: [],
  // current selections
  detector: "insightface:scrfd_10g_bnkps",
  recognizer: "insightface:w600k_r50",
  // saved artifacts
  results: [],
  datasets: [],
  benchmarks: [],
  // scans
  scan: null,
  selectedFrame: 0,
  frameZoom: null,
  inspector: null,
  liveSweep: null,
  sweepTimer: null,
  // benchmark display
  benchmarkRun: null,
  benchSort: { key: "f1", dir: -1 },
  benchSelected: null,
  // transient UI
  faceFiles: [],
  frameFiles: [],
  datasetFiles: [],
  datasetPresent: new Set(),
  datasetBoxes: {},
  busy: false,
  studentsOpen: false,
  toastTimer: null,
  toastRemaining: 0,
  toastDeadline: 0,
  settings: {
    acceptThreshold: 0.42,
    reviewThreshold: 0.34,
    minMargin: 0.04,
    detSize: 1280,
    detThresh: 0.35,
    nmsThresh: 0.45,
    tilingMode: "none",
    assignmentMode: "hungarian",
    templateStrategy: "top_k_mean",
    fusionMode: "k_of_n_voting",
    fusionK: 2,
    minFaceQuality: 0,
    maxFrames: 12,
    maxPhotosPerUpload: 8,
    maxPhotosPerPerson: 12,
    templateTopK: 2,
    benchmarkDetSizes: [640, 1024, 1280],
    defaultDetector: "insightface:scrfd_10g_bnkps",
    defaultRecognizer: "insightface:w600k_r50",
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

/* ------------------------------------------------------------------ utils */

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      default: return "&#39;";
    }
  });
}

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const plural = (count, singular, pluralForm = `${singular}s`) =>
  `${count} ${count === 1 ? singular : pluralForm}`;

const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const fixed = (value, digits = 3) => (Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—");
// Like fixed(), but null/undefined render as a dash instead of 0.000.
const dash = (value, digits = 3) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(digits);

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatWhen(value) {
  if (!value) return "unknown time";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function debounce(fn, delay) {
  let handle = null;
  return (...args) => {
    window.clearTimeout(handle);
    handle = window.setTimeout(() => fn(...args), delay);
  };
}

function storedToken() {
  try { return window.localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

async function api(path, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  const token = storedToken();
  if (token) headers.set("X-API-Token", token);
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401 && retry) {
    const entered = window.prompt("This server requires an API token:");
    if (entered) {
      try { window.localStorage.setItem(TOKEN_KEY, entered.trim()); } catch { /* ignore */ }
      return api(path, options, false);
    }
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail = typeof body.detail === "string"
          ? body.detail
          : Array.isArray(body.detail)
            ? body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ")
            : JSON.stringify(body.detail);
      }
    } catch { /* keep the status text */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

const jsonRequest = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function _scheduleToastHide(ms) {
  window.clearTimeout(state.toastTimer);
  state.toastRemaining = ms;
  state.toastDeadline = Date.now() + ms;
  state.toastTimer = window.setTimeout(() => {
    $("#toast").hidden = true;
    state.toastRemaining = 0;
  }, ms);
}
function pauseToast() {
  if ($("#toast").hidden || !state.toastRemaining) return;
  window.clearTimeout(state.toastTimer);
  state.toastRemaining = Math.max(400, state.toastDeadline - Date.now());
}
function resumeToast() {
  if ($("#toast").hidden || !state.toastRemaining || document.hidden) return;
  _scheduleToastHide(state.toastRemaining);
}
function toast(message, kind = "") {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast ${kind}`.trim();
  element.hidden = false;
  _scheduleToastHide(4600);
}
function openStudentsDrawer() {
  state.studentsOpen = true;
  $("#students-drawer").hidden = false;
  $("#drawer-backdrop").hidden = false;
  $("#students-toggle")?.setAttribute("aria-expanded", "true");
  document.body.classList.add("drawer-open");
}
function closeStudentsDrawer() {
  if (!state.studentsOpen) return;
  state.studentsOpen = false;
  $("#students-drawer").hidden = true;
  $("#drawer-backdrop").hidden = true;
  $("#students-toggle")?.setAttribute("aria-expanded", "false");
  document.body.classList.remove("drawer-open");
}

function remember() {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ detector: state.detector, recognizer: state.recognizer }),
    );
  } catch { /* localStorage may be blocked */ }
}

function recall() {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null") || {};
  } catch {
    return {};
  }
}

/* --------------------------------------------------------------- controls */

function slider(id, outputId, value, formatter = (v) => Number(v).toFixed(2)) {
  const input = $(`#${id}`);
  input.value = String(value);
  const update = () => { $(`#${outputId}`).textContent = formatter(input.value); };
  input.addEventListener("input", update);
  update();
}

function readNumber(id, fallback = 0) {
  const value = Number($(`#${id}`).value);
  return Number.isFinite(value) ? value : fallback;
}

function readSelection() {
  return {
    detector: $("#detector-select").value,
    recognizer: $("#recognizer-select").value,
    detSize: parseInt($("#det-size").value, 10) || state.settings.detSize || 1280,
    tilingMode: $("#tiling-mode").value,
    detThresh: readNumber("det-thresh", 0.35),
    nmsThresh: readNumber("nms-thresh", 0.45),
    acceptThreshold: readNumber("accept-threshold", 0.42),
    reviewThreshold: readNumber("review-threshold", 0.34),
    minMargin: readNumber("min-margin", 0.04),
    minQuality: readNumber("min-quality", 0),
    assignmentMode: $("#assignment-mode").value,
    templateStrategy: $("#template-strategy").value,
    fusionMode: $("#fusion-mode").value,
    fusionK: Math.max(1, Math.round(readNumber("fusion-k", 2))),
    flipAug: $("#flip-aug").checked,
    lowresAug: $("#lowres-aug").checked,
    withQuality: $("#with-quality").checked,
    withPose: $("#with-pose").checked,
  };
}

/* ----------------------------------------------------------- model status */

function setModelPill(kind, text, title = "") {
  const pill = $("#model-pill");
  pill.classList.remove("loading", "ready", "error");
  pill.classList.add(kind);
  $("#model-status").textContent = text;
  pill.title = title;
}

function fillSelect(select, entries, value, labelFor) {
  select.innerHTML = "";
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.id;
    const unavailable = entry.available === false;
    option.textContent = `${labelFor(entry)}${unavailable ? " · unavailable" : ""}`;
    option.disabled = unavailable;
    select.append(option);
  }
  if (value && !entries.some((entry) => entry.id === value)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  select.value = value;
}

function constrainDetSize() {
  const select = $("#det-size");
  if (!select) return;
  const entry = state.registry.detectors.find((item) => item.id === state.detector);
  const allowed = entry?.input_sizes;
  let firstAllowed = null;
  for (const option of [...select.options]) {
    const ok = !allowed || allowed.includes(Number(option.value));
    option.disabled = !ok;
    if (ok && firstAllowed === null) firstAllowed = option.value;
  }
  if (allowed && !allowed.includes(Number(select.value)) && firstAllowed) {
    select.value = firstAllowed;
  }
}

function renderModelPickers() {
  const { detectors, recognizers } = state.registry;
  fillSelect($("#detector-select"), detectors, state.detector, (entry) => entry.label || entry.id);
  fillSelect($("#recognizer-select"), recognizers, state.recognizer, (entry) => entry.label || entry.id);
  const recognizerEntry = recognizers.find((entry) => entry.id === state.recognizer);
  $("#scan-recognizer").textContent = recognizerEntry?.label || state.recognizer;
  constrainDetSize();
}

let modelStatusSeq = 0;

async function refreshModelStatus() {
  const seq = ++modelStatusSeq;
  setModelPill("loading", "Checking models…");
  try {
    const info = await api(
      `/api/model?detector=${encodeURIComponent(state.detector)}&recognizer=${encodeURIComponent(state.recognizer)}`,
    );
    if (seq !== modelStatusSeq) return;
    if (info.ready) {
      const ep = (info.providers || []).length ? ` · EP: ${info.providers.join(", ")}` : "";
      setModelPill(
        "ready",
        `${info.detector?.label || state.detector} · ready`,
        `${info.detectorId} + ${info.recognizerId}${ep}`,
      );
      $("#model-banner").hidden = true;
    } else {
      setModelPill("error", "Models unavailable", info.error || "");
      $("#model-banner").hidden = false;
      $("#model-banner").textContent = info.error || "The selected models could not load.";
    }
  } catch (error) {
    if (seq !== modelStatusSeq) return;
    setModelPill("error", "Models unavailable", error.message);
    $("#model-banner").hidden = false;
    $("#model-banner").textContent = error.message;
  }
}

/* ---------------------------------------------------------------- roster */

function personById(id) {
  return state.people.find((person) => person.id === Number(id));
}

function renderModelUsage() {
  const list = $("#model-usage-list");
  const usage = {};
  for (const person of state.people) {
    for (const [recognizerId, count] of Object.entries(person.recognizers || {})) {
      usage[recognizerId] = usage[recognizerId] || { templates: 0, people: 0 };
      usage[recognizerId].templates += count;
      usage[recognizerId].people += 1;
    }
  }
  const ids = Object.keys(usage);
  if (!ids.length) {
    list.innerHTML = `<div class="empty small"><strong>No embeddings yet</strong><span>Enroll a student to create templates.</span></div>`;
    return;
  }
  list.innerHTML = ids
    .map((id) => {
      const entry = state.registry.recognizers.find((item) => item.id === id);
      const active = id === state.recognizer;
      return `
        <div class="model-row${active ? " is-active" : ""}">
          <div class="model-main"><span class="model-name">${esc(entry?.label || id)}</span></div>
          <div class="model-stats"><b>${usage[id].templates}</b> templates · <b>${usage[id].people}</b> students</div>
        </div>`;
    })
    .join("");
}

function renderPeople() {
  const list = $("#people-list");
  list.innerHTML = "";
  $("#student-count").textContent = String(state.people.length);
  const btnCount = $("#students-btn-count");
  if (btnCount) btnCount.textContent = String(state.people.length);
  const templates = state.people.reduce((total, person) => total + person.templates, 0);
  $("#template-total").textContent = plural(templates, "template");
  renderModelUsage();

  if (!state.people.length) {
    list.innerHTML = `<li class="empty"><strong>No students yet</strong><span>Add a name and one or more face photos to get started.</span></li>`;
    return;
  }

  list.innerHTML = state.people
    .map((person) => {
      const thumbs = person.photos.length
        ? person.photos
            .map(
              (photo) => `
                <div class="thumb" title="${esc(photo.quality || "Face photo")} · ${photo.recognizers?.length || 0} embedder(s)">
                  <img src="${esc(photo.url)}" alt="" loading="lazy">
                  <button class="thumb-x" data-action="remove-photo" data-photo-id="${esc(photo.id)}" title="Remove this photo" type="button">×</button>
                </div>`,
            )
            .join("")
        : `<div class="thumb-empty">No photos — add one before scanning.</div>`;
      return `
        <li class="person" data-id="${person.id}">
          <div class="person-top">
            <span class="num">${String(person.number).padStart(2, "0")}</span>
            <div class="person-title">
              <div class="person-name" title="Double-click to rename">${esc(person.name)}</div>
              <div class="person-meta">${plural(person.templates, "face photo")}</div>
            </div>
            <div class="person-actions">
              <button class="btn ghost small" data-action="add-photos" type="button">+ Photos</button>
              <button class="icon-btn" data-action="remove-person" title="Remove student" type="button">×</button>
            </div>
          </div>
          <div class="thumbs">${thumbs}</div>
          <input class="photo-input" type="file" accept="image/*" multiple hidden>
        </li>`;
    })
    .join("");
}

async function addPerson(event) {
  event.preventDefault();
  const errorElement = $("#add-error");
  errorElement.hidden = true;
  if (!state.faceFiles.length) {
    errorElement.hidden = false;
    errorElement.textContent = "Add at least one face photo.";
    return;
  }
  const button = $("#add-person-btn");
  if (state.busy || button.disabled) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Adding…";
  state.busy = true;
  const form = new FormData();
  form.append("name", $("#person-name").value.trim());
  for (const file of state.faceFiles) form.append("images", file, file.name);
  try {
    const data = await api("/api/people", { method: "POST", body: form });
    applyState(data.state);
    state.faceFiles = [];
    $("#person-name").value = "";
    renderFaceFiles();
    toast(`Student #${data.person.number} added (${data.person.name})`, "ok");
  } catch (error) {
    errorElement.hidden = false;
    errorElement.textContent = error.message;
  } finally {
    state.busy = false;
    button.textContent = original;
    button.disabled = false;
  }
}

async function uploadPersonPhotos(personId, files) {
  const form = new FormData();
  for (const file of files) form.append("images", file, file.name);
  try {
    const data = await api(`/api/people/${personId}/photos`, { method: "POST", body: form });
    applyState(data.state);
    toast(`Added ${plural(files.length, "photo")} to ${data.person.name}`, "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function handlePeopleClick(event) {
  const actionElement = event.target.closest("[data-action]");
  if (!actionElement) return;
  const card = actionElement.closest(".person");
  if (!card) return;
  const personId = Number(card.dataset.id);
  const action = actionElement.dataset.action;

  if (action === "add-photos") { card.querySelector(".photo-input").click(); return; }

  if (action === "remove-photo") {
    try {
      const data = await api(`/api/people/${personId}/photos/${actionElement.dataset.photoId}`, { method: "DELETE" });
      applyState(data.state);
      toast("Photo removed", "ok");
    } catch (error) { toast(error.message, "error"); }
    return;
  }

  if (action === "remove-person") {
    const person = personById(personId);
    if (!person || !window.confirm(`Remove ${person.name} and all their face photos?`)) return;
    try {
      const data = await api(`/api/people/${personId}`, { method: "DELETE" });
      applyState(data.state);
      toast(`${person.name} removed`, "ok");
    } catch (error) { toast(error.message, "error"); }
  }
}

async function handlePeopleChange(event) {
  if (!event.target.classList.contains("photo-input")) return;
  const card = event.target.closest(".person");
  const files = [...event.target.files];
  event.target.value = "";
  if (!card || !files.length) return;
  await uploadPersonPhotos(Number(card.dataset.id), files);
}

async function handleRename(event) {
  const nameElement = event.target.closest(".person-name");
  if (!nameElement) return;
  const card = nameElement.closest(".person");
  const person = personById(card?.dataset.id);
  if (!person) return;
  const next = window.prompt("Rename student", person.name);
  if (next === null) return;
  const clean = next.trim();
  if (!clean || clean === person.name) return;
  try {
    const data = await api(`/api/people/${person.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: clean }),
    });
    applyState(data.state);
    toast("Student renamed", "ok");
  } catch (error) { toast(error.message, "error"); }
}

function applyState(payload) {
  if (payload.people) state.people = payload.people;
  if (payload.registry) state.registry = payload.registry;
  renderPeople();
  renderModelPickers();
  renderPresentPicker();
  renderBenchOptions();
}

async function reembedAll() {
  const button = $("#reembed-btn");
  if (state.busy || button.disabled) return;
  const original = button.textContent;
  button.disabled = true;
  state.busy = true;
  button.textContent = "Embedding…";
  try {
    const data = await api("/api/students/re-embed", jsonRequest({ recognizer_ids: [] }));
    applyState(data.state);
    const summary = Object.entries(data.results || {})
      .map(([id, row]) => `${id.split(":").pop()} +${row.updated}`)
      .join(" · ");
    toast(summary ? `Embeddings updated: ${summary}` : "No photos needed re-embedding", "ok");
    if (data.errors?.length) toast(`${data.errors.length} photo(s) skipped`, "error");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.busy = false;
    button.textContent = original;
    button.disabled = false;
  }
}

/* ------------------------------------------------------------ file chips */

function renderFileChips(containerId, files, onRemove) {
  const wrap = $(`#${containerId}`);
  wrap.innerHTML = "";
  files.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    const image = document.createElement("img");
    const url = URL.createObjectURL(file);
    image.src = url;
    image.alt = "";
    image.onload = () => URL.revokeObjectURL(url);
    image.onerror = () => URL.revokeObjectURL(url);
    const label = document.createElement("span");
    label.textContent = file.name;
    label.title = `${file.name} · ${formatBytes(file.size)}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => onRemove(index));
    chip.append(image, label, remove);
    wrap.append(chip);
  });
}

const renderFaceFiles = () =>
  renderFileChips("face-files", state.faceFiles, (index) => {
    state.faceFiles.splice(index, 1);
    renderFaceFiles();
  });

const renderFrameFiles = () =>
  renderFileChips("frame-files", state.frameFiles, (index) => {
    state.frameFiles.splice(index, 1);
    renderFrameFiles();
  });

const renderDatasetFiles = () =>
  renderFileChips("dataset-files", state.datasetFiles, (index) => {
    state.datasetFiles.splice(index, 1);
    renderDatasetFiles();
  });

function addFiles(bucket, files, limit, label) {
  const incoming = [...files].filter((file) => file.type.startsWith("image/"));
  if (!incoming.length) return;
  const room = limit - bucket.files.length;
  if (room <= 0) { toast(`Up to ${limit} ${label}`, "error"); return; }
  if (incoming.length > room) toast(`Only ${room} more ${label} can be added`, "error");
  bucket.files.push(...incoming.slice(0, room));
  bucket.render();
}

function wireDropzone(dropzone, handler) {
  for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  }
  dropzone.addEventListener("dragleave", (event) => {
    if (event.target === dropzone) dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    handler(event.dataTransfer.files);
  });
}

/* ------------------------------------------------------------------ scan */

function scanForm() {
  const selection = readSelection();
  const form = new FormData();
  form.append("detector", selection.detector);
  form.append("recognizer", selection.recognizer);
  form.append("detSize", String(selection.detSize));
  form.append("tilingMode", selection.tilingMode);
  form.append("detThresh", String(selection.detThresh));
  form.append("nmsThresh", String(selection.nmsThresh));
  form.append("acceptThreshold", String(selection.acceptThreshold));
  form.append("reviewThreshold", String(selection.reviewThreshold));
  form.append("minMargin", String(selection.minMargin));
  form.append("minQuality", String(selection.minQuality));
  form.append("assignmentMode", selection.assignmentMode);
  form.append("templateStrategy", selection.templateStrategy);
  form.append("fusionMode", selection.fusionMode);
  form.append("fusionK", String(selection.fusionK));
  form.append("flipAug", String(selection.flipAug));
  form.append("lowresAug", String(selection.lowresAug));
  form.append("withQuality", String(selection.withQuality));
  form.append("withPose", String(selection.withPose));
  form.append("labelUnknown", String($("#label-unknown").checked));
  for (const file of state.frameFiles) form.append("images", file, file.name);
  return form;
}

async function runScan() {
  if (state.busy) return;
  setBanner("#scan-error", "");
  $("#scan-warnings").hidden = true;
  if (!state.frameFiles.length) { setBanner("#scan-error", "Add at least one classroom frame."); return; }

  const button = $("#scan-btn");
  const original = button.textContent;
  const status = $("#scan-status");
  const startedAt = performance.now();
  button.textContent = "Scanning…";
  button.disabled = true;
  status.hidden = false;
  state.busy = true;
  const timer = window.setInterval(() => {
    $("#scan-status-text").textContent =
      `Analyzing ${plural(state.frameFiles.length, "frame")} · ${((performance.now() - startedAt) / 1000).toFixed(1)}s`;
  }, 120);

  try {
    const result = await api("/api/scan", { method: "POST", body: scanForm() });
    state.scan = result;
    state.liveSweep = null;
    state.frameZoom = null;
    closeInspector();
    state.selectedFrame = result.summary.bestIndex >= 0 ? result.summary.bestIndex : 0;
    renderResults();
    loadResults().catch((error) => toast(error.message, "error"));
    toast(
      `Scanned ${plural(result.summary.frames, "frame")} · ${result.summary.facesDetected} faces · ${(result.summary.elapsedMs / 1000).toFixed(1)}s`,
      "ok",
    );
  } catch (error) {
    setBanner("#scan-error", error.message);
    toast(error.message, "error");
  } finally {
    window.clearInterval(timer);
    status.hidden = true;
    button.textContent = original;
    button.disabled = false;
    state.busy = false;
  }
}

let sweepSeq = 0;

/* Re-score the loaded scan from cached embeddings — no inference. */
async function liveSweep() {
  if (!state.scan || state.busy) return;
  const scan = state.scan;
  const seq = ++sweepSeq;
  const selection = readSelection();
  const payload = {
    detector: scan.detectorId,
    recognizer: scan.recognizerId,
    det_size: scan.summary?.detSize || selection.detSize,
    tiling_mode: scan.summary?.tilingMode || selection.tilingMode,
    det_thresh: selection.detThresh,
    nms_thresh: selection.nmsThresh,
    flip_aug: scan.summary?.flipAug ?? selection.flipAug,
    lowres_aug: scan.summary?.lowresAug ?? false,
    with_quality: selection.withQuality,
    with_pose: selection.withPose,
    variant: scan.summary?.variant,
    accept_threshold: selection.acceptThreshold,
    review_threshold: selection.reviewThreshold,
    min_margin: selection.minMargin,
    assignment_mode: selection.assignmentMode,
    template_strategy: selection.templateStrategy,
    top_k: scan.summary?.topK ?? state.settings.templateTopK ?? 2,
    fusion_mode: selection.fusionMode,
    k: Math.round(selection.fusionK),
    min_quality: selection.minQuality,
  };
  try {
    const data = await api(`/api/results/${encodeURIComponent(scan.scanId)}/sweep`, jsonRequest(payload));
    if (seq !== sweepSeq || state.scan !== scan) return;
    state.liveSweep = data;
    const frames = scan.frames;
    data.faces.forEach((faces, index) => { if (frames[index]) frames[index].faces = faces; });
    delete scan.roster;
    // Rebuild the roster from the current people list, not just the rows the
    // original scan saved: a student enrolled after the scan must still appear.
    const byId = new Map((scan.people || []).map((row) => [Number(row.personId), row]));
    for (const person of state.people) {
      if (!byId.has(person.id)) {
        byId.set(person.id, {
          personId: person.id,
          number: person.number,
          name: person.name,
          status: "absent",
          confidence: 0,
          framesHit: 0,
          enrolled: Boolean(person.recognizers?.[scan.recognizerId]),
          photoUrl: person.photos?.[0]?.url || null,
          templates: person.recognizers?.[scan.recognizerId] || 0,
        });
      }
    }
    scan.people = [...byId.values()]
      .sort((a, b) => (a.number ?? 0) - (b.number ?? 0))
      .map((row) => {
        const record = data.students[String(row.personId)];
        return record
          ? { ...row, status: record.status, confidence: record.score ?? 0, framesHit: record.frames_hit ?? 0 }
          : { ...row, status: "absent", confidence: 0, framesHit: 0 };
      });
    scan.summary.matchedCount = data.matchedCount;
    if (typeof data.needsReview === "number") scan.summary.needsReview = data.needsReview;
    if (typeof data.unknown === "number") scan.summary.unknown = data.unknown;
    scan.summary.templateStrategy = payload.template_strategy;
    scan.summary.acceptThreshold = payload.accept_threshold;
    scan.summary.reviewThreshold = payload.review_threshold;
    scan.summary.minMargin = payload.min_margin;
    scan.summary.assignmentMode = payload.assignment_mode;
    scan.summary.fusionMode = payload.fusion_mode;
    scan.summary.fusionK = payload.k;
    scan.summary.minQuality = payload.min_quality;
    const zoomedPerson = state.frameZoom?.personId ?? null;
    renderSummary();
    renderRoster();
    renderFrame();
    if (zoomedPerson != null) {
      const frame = scan.frames[state.selectedFrame];
      const idx = frame?.faces?.findIndex((f) => Number(f.personId) === Number(zoomedPerson)) ?? -1;
      if (idx >= 0) zoomFrameToFace(idx, zoomedPerson);
      else resetFrameZoom();
    }
    if (state.lightbox && state.lightbox.frames === scan.frames) renderLightbox();
    if (state.inspector) {
      openInspector(state.inspector.faceIndex, state.inspector.frame, state.inspector.scope);
    }
    toast("Re-scored from cache — no re-inference", "");
  } catch (error) {
    if (seq !== sweepSeq) return;
    toast(`Live sweep needs a fresh scan: ${error.message}`, "error");
  }
}

const scheduleSweep = debounce(liveSweep, 220);

/* --------------------------------------------------------------- results */

function rosterRows(scan) {
  if (scan.roster) {
    return scan.roster.map((row) => {
      const person = personById(row.personId);
      return {
        personId: row.personId,
        number: person?.number ?? row.number ?? 0,
        name: person?.name || row.name || `#${row.personId}`,
        status: row.status === "absent" ? "absent" : row.status,
        confidence: row.score ?? row.confidence ?? 0,
        framesHit: row.frames_hit ?? row.framesHit ?? 0,
      };
    });
  }
  return scan.people || [];
}

function summaryCard(label, value, sub, kind = "") {
  return `<div class="summary-card ${kind}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub">${esc(sub)}</div></div>`;
}

function renderSummary() {
  const scan = state.scan;
  if (!scan) return;
  const summary = scan.summary;
  const rows = rosterRows(scan);
  const present = rows.filter((row) => row.status === "present").length;
  const review = rows.filter((row) => row.status === "needs_review").length;
  const unmatched = (scan.frames[state.selectedFrame]?.faces || []).filter((face) => face.status === "unmatched").length;

  $("#results-meta").textContent = [
    scan.summary?.detectorId || scan.detectorId,
    "+",
    scan.summary?.recognizerId || scan.recognizerId,
    `· ${plural(summary.frames, "frame")}`,
    `· accept ${fixed(summary.acceptThreshold, 2)}`,
    `· det ${summary.detSize}px`,
    `· ${(summary.elapsedMs / 1000).toFixed(2)}s`,
  ].filter(Boolean).join(" ");

  $("#summary-grid").innerHTML = [
    summaryCard("Present", present, `of ${plural(rows.length, "student")}`, present ? "good" : ""),
    summaryCard("Review", review, "ambiguous matches", review ? "warn" : ""),
    summaryCard("Faces", summary.facesDetected, `${unmatched} unknown in frame ${state.selectedFrame + 1}`),
    summaryCard(
      "Latency",
      `${(Number(summary.elapsedMs || 0) / 1000).toFixed(1)}s`,
      `${plural(summary.frames, "frame")} · ${summary.cachedFrames || 0} cached`,
    ),
  ].join("");
}

function renderRoster() {
  const scan = state.scan;
  if (!scan) return;
  const rows = rosterRows(scan);
  const list = $("#matches-list");
  const present = rows
    .filter((row) => row.status === "present" || row.status === "needs_review")
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const absent = rows.filter((row) => row.status !== "present" && row.status !== "needs_review");
  $("#matched-count").textContent = `${present.length} matched`;

  if (!rows.length) {
    list.innerHTML = `<div class="empty"><strong>No students enrolled</strong><span>Faces are detected and counted, but cannot be identified yet.</span></div>`;
    $("#absent-wrap").innerHTML = "";
    return;
  }
  if (!present.length) {
    list.innerHTML = `<div class="empty"><strong>No students matched</strong><span>Try lowering the accept threshold or enrolling clearer reference photos.</span></div>`;
  } else {
    list.innerHTML = present
      .map(
        (row) => `
          <div class="match-row ${row.status}" data-person-id="${row.personId}" title="${esc(row.name)}">
            <span class="num sm">${String(row.number).padStart(2, "0")}</span>
            <div class="match-main">
              <div class="match-name">${esc(row.name)} <span class="status-tag ${row.status}">${row.status === "present" ? "present" : "review"}</span></div>
              <div class="bar"><span style="width:${Math.round(clamp(row.confidence || 0, 0, 1) * 100)}%"></span></div>
            </div>
            <span class="score">${fixed(row.confidence, 3)}</span>
          </div>`,
      )
      .join("");
  }

  for (const row of $$(".match-row", list)) {
    const pid = Number(row.dataset.personId);
    row.classList.toggle("is-focused", state.frameZoom?.personId === pid);
    row.title = `${row.title} · click to zoom in frame`;
    row.addEventListener("mouseenter", () => setHighlight(row.dataset.personId));
    row.addEventListener("mouseleave", () => setHighlight(null));
    row.addEventListener("click", () => focusStudentInFrame(pid));
  }

  $("#absent-wrap").innerHTML = absent.length
    ? `<details class="absent-block"><summary><span>Absent / undetected</span><span>${absent.length}</span></summary><div class="absent-list">${absent
        .map((row) => `<div class="absent-row"><span class="num">${String(row.number).padStart(2, "0")}</span><span>${esc(row.name)}</span><span>absent</span></div>`)
        .join("")}</div></details>`
    : "";
}

function renderFrameTabs() {
  const scan = state.scan;
  state.selectedFrame = clamp(state.selectedFrame, 0, Math.max(0, scan.frames.length - 1));
  const tabs = $("#frame-tabs");
  tabs.innerHTML = "";
  scan.frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `frame-tab${index === state.selectedFrame ? " active" : ""}`;
    button.title = frame.sourceName || `Frame ${index + 1}`;
    button.innerHTML = `Frame <b>${index + 1}</b> <span>${plural(frame.faces.length, "face")}</span>`;
    button.addEventListener("click", () => {
      state.selectedFrame = index;
      state.frameZoom = null;
      closeInspector();
      renderFrameTabs();
      renderFrame();
      renderSummary();
    });
    tabs.append(button);
  });
}

function faceClass(status) {
  if (status === "present") return "matched";
  if (status === "needs_review") return "review";
  return "unknown";
}

function renderFrameInto(container, frame, index = 0) {
  container.innerHTML = "";
  delete container.dataset.highlight;

  const media = document.createElement("div");
  media.className = "frame-media";
  const image = document.createElement("img");
  image.src = frame.imageUrl;
  image.alt = `Classroom frame ${index + 1}`;
  image.draggable = false;
  media.append(image);
  const layer = document.createElement("div");
  layer.className = "box-layer";
  media.append(layer);
  container.append(media);

  frame.faces.forEach((face, faceIndex) => {
    const [x1, y1, x2, y2] = face.box;
    const left = clamp((x1 / frame.width) * 100, 0, 100);
    const top = clamp((y1 / frame.height) * 100, 0, 100);
    const width = clamp(((x2 - x1) / frame.width) * 100, 0, 100 - left);
    const height = clamp(((y2 - y1) / frame.height) * 100, 0, 100 - top);
    if (width <= 0 || height <= 0) return;

    const box = document.createElement("button");
    box.type = "button";
    box.className = `face-box ${faceClass(face.status)}`;
    box.style.left = `${left}%`;
    box.style.top = `${top}%`;
    box.style.width = `${width}%`;
    box.style.height = `${height}%`;
    box.dataset.person = face.personId == null ? "" : String(face.personId);
    box.dataset.face = String(faceIndex);
    box.title = `${face.name || "Unknown"} · ${face.status} · score ${fixed(face.confidence)} · margin ${fixed(face.margin)} · ${face.sizeBucket || "?"}`;
    if (width < 7 || height < 7) box.classList.add("compact");
    const label = document.createElement("span");
    label.className = "box-label";
    const tag = face.status === "present" ? "" : "? ";
    label.textContent = `${tag}${face.personId != null ? face.name : "Unknown"} · ${fixed(face.confidence, 2)}`;
    if (top < 7) label.classList.add("inside");
    if (left > 60) label.classList.add("right");
    box.append(label);
    box.addEventListener("click", () => openInspector(faceIndex, frame, container));
    layer.append(box);
  });

  return media;
}

function renderFrame() {
  const scan = state.scan;
  if (!scan || !scan.frames.length) return;
  const index = clamp(state.selectedFrame, 0, scan.frames.length - 1);
  const frame = scan.frames[index];
  renderFrameInto($("#frame-view"), frame, index);
  const resetBtn = $("#frame-zoom-reset");
  if (resetBtn) resetBtn.hidden = true;

  $("#frame-meta").textContent =
    `${frame.sourceName || `frame ${index + 1}`} · ${frame.width}×${frame.height} · ${plural(frame.faces.length, "face")}` +
    (frame.cached ? " · cached" : "");
  const link = $("#annotated-link");
  link.hidden = !frame.annotatedUrl;
  link.href = frame.annotatedUrl || "#";
}

function applyFrameZoom() {
  const zoom = state.frameZoom;
  const view = $("#frame-view");
  const media = view?.querySelector(".frame-media");
  const frame = state.scan?.frames?.[state.selectedFrame];
  if (!zoom || !view || !media || !frame) return;
  const face = frame.faces?.[zoom.faceIndex];
  if (!face) { resetFrameZoom(); return; }
  const img = media.querySelector("img");
  if (!img || !img.complete || !img.naturalWidth || !media.offsetWidth) {
    img?.addEventListener("load", () => applyFrameZoom(), { once: true });
    return;
  }
  const mw = media.offsetWidth;
  const mh = media.offsetHeight;
  const vw = view.clientWidth;
  const vh = view.clientHeight;
  const [x1, y1, x2, y2] = face.box;
  const bw = Math.max(4, ((x2 - x1) / frame.width) * mw);
  const bh = Math.max(4, ((y2 - y1) / frame.height) * mh);
  const cx = ((x1 + x2) / 2 / frame.width) * mw;
  const cy = ((y1 + y2) / 2 / frame.height) * mh;
  // Fit a 3.2x padded context region around the face so surroundings stay legible.
  const scale = clamp(Math.min(vw / (bw * 3.2), vh / (bh * 3.2)), 1.5, 6.5);
  const offX = media.offsetLeft;
  const offY = media.offsetTop;
  let tx = vw / 2 - offX - cx * scale;
  let ty = vh / 2 - offY - cy * scale;
  if (mw * scale >= vw) tx = clamp(tx, vw - offX - mw * scale, -offX);
  else tx = (vw - mw * scale) / 2 - offX;
  if (mh * scale >= vh) ty = clamp(ty, vh - offY - mh * scale, -offY);
  else ty = (vh - mh * scale) / 2 - offY;
  view.classList.add("is-zoomed");
  delete view.dataset.highlight;
  media.style.setProperty("--inv", String(1 / scale));
  media.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${scale.toFixed(3)})`;
  for (const box of $$(".face-box", media)) {
    box.classList.toggle("focused", Number(box.dataset.face) === zoom.faceIndex);
  }
  for (const row of $$("#matches-list .match-row")) {
    row.classList.toggle("is-focused", Number(row.dataset.personId) === Number(zoom.personId));
  }
  const resetBtn = $("#frame-zoom-reset");
  if (resetBtn) resetBtn.hidden = false;
}
function zoomFrameToFace(faceIndex, personId) {
  state.frameZoom = { faceIndex, personId: Number(personId) };
  applyFrameZoom();
}
function resetFrameZoom() {
  state.frameZoom = null;
  const view = $("#frame-view");
  const media = view?.querySelector(".frame-media");
  view?.classList.remove("is-zoomed");
  if (media) {
    media.style.transform = "";
    media.style.setProperty("--inv", "1");
    for (const box of $$(".face-box", media)) box.classList.remove("focused");
  }
  for (const row of $$("#matches-list .match-row")) row.classList.remove("is-focused");
  const resetBtn = $("#frame-zoom-reset");
  if (resetBtn) resetBtn.hidden = true;
}
function focusStudentInFrame(personId) {
  const scan = state.scan;
  if (!scan?.frames?.length) return;
  if (state.frameZoom && Number(state.frameZoom.personId) === Number(personId)) {
    resetFrameZoom();
    return;
  }
  const current = scan.frames[state.selectedFrame];
  let targetFrame = state.selectedFrame;
  let faceIndex = current?.faces?.findIndex((f) => Number(f.personId) === Number(personId)) ?? -1;
  if (faceIndex < 0) {
    // Pick the frame where this student matched with the highest confidence.
    let bestConf = -1;
    scan.frames.forEach((frame, fIdx) => {
      (frame.faces || []).forEach((face, idx) => {
        if (Number(face.personId) === Number(personId) && (face.confidence ?? 0) > bestConf) {
          bestConf = face.confidence ?? 0;
          targetFrame = fIdx;
          faceIndex = idx;
        }
      });
    });
  }
  if (faceIndex < 0) {
    toast("No face box for this student in the current scan", "");
    return;
  }
  if (targetFrame !== state.selectedFrame) {
    state.selectedFrame = targetFrame;
    closeInspector();
    renderFrameTabs();
    renderFrame();
    renderSummary();
  }
  zoomFrameToFace(faceIndex, personId);
}
function setHighlight(personId) {
  if (state.frameZoom) return;
  const view = $("#frame-view");
  if (!personId) {
    delete view.dataset.highlight;
    for (const box of $$(".face-box", view)) box.classList.remove("active");
    return;
  }
  view.dataset.highlight = String(personId);
  for (const box of $$(".face-box", view)) {
    box.classList.toggle("active", box.dataset.person === String(personId));
  }
}

function renderResults() {
  const scan = state.scan;
  $("#results").hidden = !scan;
  $("#empty-results").hidden = Boolean(scan);
  if (!scan) return;
  renderSummary();
  renderRoster();
  renderFrameTabs();
  renderFrame();

  const warnings = [...(scan.warnings || [])];
  if (scan.summary && scan.summary.templateCount === 0) {
    warnings.unshift("No templates exist for this recognizer — enroll students with it active.");
  }
  const warningElement = $("#scan-warnings");
  warningElement.hidden = warnings.length === 0;
  warningElement.textContent = warnings.join("\n");
}

/* -------------------------------------------------------------- inspector */

function topCandidatesHtml(candidates) {
  if (!candidates?.length) return `<div class="muted small">no candidates</div>`;
  return candidates
    .map((candidate, index) => {
      const name = candidate.name || personById(candidate.studentId)?.name || `#${candidate.studentId}`;
      return `<div class="cand-row${index === 0 ? " best" : ""}"><span>${esc(name)}</span><span class="mono">${fixed(candidate.score)}</span></div>`;
    })
    .join("");
}

function landmarksSvg(points) {
  if (!points?.length) return "";
  const dots = points
    .map(([x, y]) => `<circle cx="${Number(x).toFixed(1)}" cy="${Number(y).toFixed(1)}" r="2.4" class="landmark-dot" />`)
    .join("");
  return `<svg class="landmark-layer" viewBox="0 0 112 112" preserveAspectRatio="none" aria-hidden="true">${dots}</svg>`;
}

function openInspector(faceIndex, frame = null, scope = null) {
  frame = frame || state.scan?.frames[state.selectedFrame];
  const face = frame?.faces[faceIndex];
  if (!face) return;
  const container = scope || $("#frame-view");
  state.inspector = { faceIndex, frame, scope: container };
  const wrap = $("#inspector");
  wrap.hidden = false;

  const pose = face.headPose
    ? `yaw ${face.headPose.yaw}° · pitch ${face.headPose.pitch}° · roll ${face.headPose.roll}°`
    : "not computed";
  const qualityText = face.quality != null ? `${fixed(face.quality, 3)} (${face.qualitySource || "n/a"})` : "n/a";

  $("#inspector-body").innerHTML = `
    <div class="inspect-crop-wrap">
      ${face.cropUrl ? `<img class="inspect-crop" src="${esc(face.cropUrl)}" alt="Aligned face crop">` : `<div class="inspect-crop placeholder">no aligned crop</div>`}
      ${landmarksSvg(face.landmarksCrop)}
    </div>
    <div class="inspect-title">
      <span class="status-tag ${face.status}">${esc(face.status)}</span>
      <strong>${esc(face.name || "Unknown")}</strong>
      <span class="muted">conf ${fixed(face.confidence)} · margin ${fixed(face.margin)}</span>
    </div>
    <div class="inspect-grid">
      <div><span class="muted">Size bucket</span><b>${esc(face.sizeBucket || "?")}</b></div>
      <div><span class="muted">Face height</span><b>${face.faceHeightPx != null ? Math.round(face.faceHeightPx) + " px" : "—"}</b></div>
      <div><span class="muted">Detector score</span><b>${fixed(face.detScore)}</b></div>
      <div><span class="muted">Quality</span><b>${esc(qualityText)}</b></div>
      <div class="wide"><span class="muted">Head pose</span><b>${esc(pose)}</b></div>
    </div>
    <div class="inspect-section">
      <span class="field-label">Top candidates</span>
      <div class="cand-list">${topCandidatesHtml(face.topCandidates)}</div>
    </div>
    <div class="inspect-section">
      <span class="field-label">Box (x1, y1, x2, y2)</span>
      <div class="mono small">${face.box.map((value) => Math.round(value)).join(", ")}</div>
    </div>`;
  for (const box of $$(".face-box", container)) {
    box.classList.toggle("active", box.dataset.face === String(faceIndex));
  }
}

function closeInspector() {
  state.inspector = null;
  $("#inspector").hidden = true;
  for (const box of $$(".face-box")) box.classList.remove("active");
}

/* ----------------------------------------------------- view prefs + lightbox */

const VIEW_KEY = "facelab.view";
const LIGHTBOX_MIN_SCALE = 1;
const LIGHTBOX_MAX_SCALE = 6;

function viewPrefs() {
  try { return JSON.parse(window.localStorage.getItem(VIEW_KEY) || "{}") || {}; } catch { return {}; }
}

function saveViewPrefs(patch) {
  try {
    window.localStorage.setItem(VIEW_KEY, JSON.stringify({ ...viewPrefs(), ...patch }));
  } catch { /* localStorage may be blocked */ }
}

function setUnknownLabels(on) {
  document.body.classList.toggle("show-unknown-labels", on);
  const input = $("#label-unknown");
  if (input) input.checked = on;
}

function setRosterCollapsed(on) {
  $("#results-body")?.classList.toggle("roster-collapsed", on);
  const button = $("#roster-toggle");
  if (button) {
    button.setAttribute("aria-pressed", String(on));
    button.textContent = on ? "Show roster" : "Hide roster";
  }
}

function setBenchSetupCollapsed(on) {
  $("#panel-benchmark")?.classList.toggle("setup-collapsed", on);
  const button = $("#bench-setup-toggle");
  if (button) {
    button.setAttribute("aria-pressed", String(on));
    button.textContent = on ? "Show setup" : "Hide setup";
  }
}
function applyViewPrefs() {
  const prefs = viewPrefs();
  setUnknownLabels(Boolean(prefs.labelUnknown));
  setRosterCollapsed(Boolean(prefs.rosterCollapsed));
  setBenchSetupCollapsed(Boolean(prefs.benchSetupCollapsed));
}

function lightboxFrame() {
  return state.lightbox?.frames?.[state.lightbox.index] || null;
}

function fitLightboxStage() {
  const lb = state.lightbox;
  const frame = lightboxFrame();
  const viewport = $("#lightbox-viewport");
  if (!lb || !frame || !viewport) return;
  const pad = 28;
  const vw = Math.max(1, viewport.clientWidth - pad * 2);
  const vh = Math.max(1, viewport.clientHeight - pad * 2);
  const aspect = (frame.width || 1) / (frame.height || 1);
  let w = vw;
  let h = vw / aspect;
  if (h > vh) { h = vh; w = vh * aspect; }
  lb.fit = { w, h };
  const stage = $("#lightbox-stage");
  stage.style.width = `${w}px`;
  stage.style.height = `${h}px`;
}

// Panning only updates transform; updating --inv on every pointermove would
// recalculate styles on every child .face-box and .box-label.
function applyLightboxPan() {
  const lb = state.lightbox;
  const stage = $("#lightbox-stage");
  if (!lb || !stage) return;
  stage.style.transform = `translate(${lb.tx}px, ${lb.ty}px) scale(${lb.scale})`;
}
function applyLightboxZoom() {
  const lb = state.lightbox;
  const stage = $("#lightbox-stage");
  if (!lb || !stage) return;
  applyLightboxPan();
  stage.style.setProperty("--inv", String(1 / lb.scale));
  stage.classList.toggle("zoomed", lb.scale >= 1.9);
}

function centerLightbox() {
  const lb = state.lightbox;
  const viewport = $("#lightbox-viewport");
  if (!lb || !lb.fit || !viewport) return;
  lb.tx = (viewport.clientWidth - lb.fit.w * lb.scale) / 2;
  lb.ty = (viewport.clientHeight - lb.fit.h * lb.scale) / 2;
}

function renderLightbox() {
  const lb = state.lightbox;
  const frame = lightboxFrame();
  if (!lb || !frame) return;
  const stage = $("#lightbox-stage");
  fitLightboxStage();
  renderFrameInto(stage, frame, lb.index);
  centerLightbox();
  applyLightboxZoom();
  $("#lightbox-title").textContent =
    `${frame.sourceName || frame.frameId || `Frame ${lb.index + 1}`} · ${frame.width}×${frame.height} · ${plural(frame.faces.length, "face")}`;
  $("#lightbox-prev").disabled = lb.index <= 0;
  $("#lightbox-next").disabled = lb.index >= lb.frames.length - 1;
}

function openLightbox(frames, index = 0) {
  if (!frames?.length) return;
  state.lightbox = {
    frames,
    index: clamp(index, 0, frames.length - 1),
    scale: 1,
    tx: 0,
    ty: 0,
    fit: null,
  };
  document.body.classList.add("lightbox-open");
  $("#frame-lightbox").hidden = false;
  renderLightbox();
}

function closeLightbox() {
  if (!state.lightbox) return;
  const stage = $("#lightbox-stage");
  if (state.inspector && state.inspector.scope === stage) closeInspector();
  state.lightbox = null;
  document.body.classList.remove("lightbox-open");
  $("#frame-lightbox").hidden = true;
  stage.innerHTML = "";
}

function lightboxStep(delta) {
  const lb = state.lightbox;
  if (!lb) return;
  const next = clamp(lb.index + delta, 0, lb.frames.length - 1);
  if (next === lb.index) return;
  if (state.inspector) closeInspector();
  lb.index = next;
  lb.scale = 1;
  renderLightbox();
}

function lightboxReset() {
  const lb = state.lightbox;
  if (!lb) return;
  lb.scale = 1;
  centerLightbox();
  applyLightboxZoom();
}

function openScanLightbox() {
  if (state.scan?.frames?.length) openLightbox(state.scan.frames, state.selectedFrame);
}

/* ------------------------------------------------------------- history */

function renderHistory() {
  const list = $("#history-list");
  const runs = state.results;
  $("#history-count").textContent = plural(runs.length, "run");
  if (!runs.length) {
    list.innerHTML = `<div class="empty small"><strong>No saved runs yet</strong><span>Scan a classroom frame and the result is stored here.</span></div>`;
    return;
  }
  list.innerHTML = runs
    .map((run) => {
      const current = state.scan && state.scan.scanId === run.scanId;
      const thumb = run.thumbUrl ? `<img src="${esc(run.thumbUrl)}" alt="" loading="lazy">` : "";
      return `
        <div class="history-row${current ? " is-current" : ""}" data-scan="${esc(run.scanId)}">
          <div class="history-thumb">${thumb}</div>
          <div class="history-main">
            <div class="history-title">
              <span class="model-tag">${esc((run.detectorId || "?").split(":").pop())}</span>
              <span class="model-tag alt">${esc((run.recognizerId || run.model || "?").split(":").pop())}</span>
              <span class="muted">${esc(formatWhen(run.createdAt))}</span>
            </div>
            <div class="history-meta">
              ${plural(run.facesDetected ?? 0, "face")} · ${run.matchedCount ?? 0} matched · ${plural(run.frames ?? 0, "frame")} · ${(Number(run.elapsedMs || 0) / 1000).toFixed(2)}s
            </div>
          </div>
          <div class="history-actions">
            <button class="btn ghost small" type="button" data-action="load-result">Load</button>
            <button class="icon-btn" type="button" data-action="delete-result" title="Delete run">×</button>
          </div>
        </div>`;
    })
    .join("");
}

async function loadResult(scanId) {
  try {
    const data = await api(`/api/results/${encodeURIComponent(scanId)}`);
    state.scan = data;
    state.liveSweep = null;
    state.frameZoom = null;
    closeInspector();
    state.selectedFrame = Number.isInteger(data.summary?.bestIndex) && data.summary.bestIndex >= 0 ? data.summary.bestIndex : 0;
    switchTab("attendance");
    renderResults();
    renderHistory();
    $("#results").scrollIntoView({ behavior: "smooth", block: "nearest" });
    toast(`Loaded run ${scanId}`, "ok");
  } catch (error) { toast(error.message, "error"); }
}

async function deleteResult(scanId) {
  if (!window.confirm(`Delete saved run ${scanId}? The annotated frames are removed too.`)) return;
  try {
    const data = await api(`/api/results/${encodeURIComponent(scanId)}`, { method: "DELETE" });
    state.results = data.results || [];
    if (state.scan?.scanId === scanId) { state.scan = null; state.frameZoom = null; closeInspector(); renderResults(); }
    renderHistory();
    toast("Saved run deleted", "ok");
  } catch (error) { toast(error.message, "error"); }
}

async function handleHistoryClick(event) {
  const row = event.target.closest(".history-row");
  if (!row) return;
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "delete-result") { await deleteResult(row.dataset.scan); return; }
  await loadResult(row.dataset.scan);
}

/* --------------------------------------------------------------- datasets */

function renderPresentPicker() {
  const wrap = $("#dataset-present");
  if (!wrap) return;
  if (!state.people.length) {
    wrap.innerHTML = `<div class="muted small">Enroll students first, then mark who is present.</div>`;
    return;
  }
  wrap.innerHTML = state.people
    .map((person) => {
      const on = state.datasetPresent.has(person.id);
      return `<button type="button" class="present-chip${on ? " on" : ""}" data-person="${person.id}">
        <span class="num sm">${String(person.number).padStart(2, "0")}</span>${esc(person.name)}
      </button>`;
    })
    .join("");
}

function renderDatasets() {
  const list = $("#dataset-list");
  $("#dataset-count").textContent = String(state.datasets.length);
  $("#bench-dataset-count").textContent = String(state.datasets.length);
  if (!state.datasets.length) {
    list.innerHTML = `<div class="empty"><strong>No datasets yet</strong><span>Build one on the left to unlock calibrated benchmarks.</span></div>`;
    return;
  }
  list.innerHTML = state.datasets
    .map((dataset) => {
      const id = dataset.datasetId || dataset.dataset_id;
      return `
      <div class="dataset-row" data-dataset="${esc(id)}">
        <div class="history-thumb">${dataset.thumbUrl ? `<img src="${esc(dataset.thumbUrl)}" alt="" loading="lazy">` : ""}</div>
        <div class="history-main">
          <div class="history-title"><strong>${esc(dataset.name)}</strong></div>
          <div class="history-meta">${plural(dataset.frames, "frame")} · ${dataset.present} labelled present · ${dataset.annotatedBoxes} boxes · ${esc(formatWhen(dataset.createdAt))}</div>
        </div>
        <div class="history-actions">
          <button class="btn ghost small" type="button" data-action="use-dataset">Benchmark</button>
          <button class="icon-btn" type="button" data-action="delete-dataset" title="Delete dataset">×</button>
        </div>
      </div>`;
    })
    .join("");
}

async function createDataset(event) {
  event.preventDefault();
  const errorElement = $("#dataset-error");
  errorElement.hidden = true;
  if (!state.datasetFiles.length) {
    errorElement.hidden = false;
    errorElement.textContent = "Add at least one frame.";
    return;
  }
  const button = $("#dataset-btn");
  if (button.disabled) return;
  const original = button.textContent;
  button.textContent = "Saving…";
  button.disabled = true;
  const form = new FormData();
  form.append("name", $("#dataset-name").value.trim());
  form.append("notes", $("#dataset-notes").value.trim());
  form.append("presentIds", JSON.stringify([...state.datasetPresent]));
  const boxes = Object.values(state.datasetBoxes);
  form.append("boxes", JSON.stringify(boxes));
  for (const file of state.datasetFiles) form.append("images", file, file.name);
  try {
    const data = await api("/api/datasets", { method: "POST", body: form });
    state.datasets = data.datasets || [];
    state.datasetFiles = [];
    state.datasetPresent = new Set();
    state.datasetBoxes = {};
    $("#dataset-name").value = "";
    $("#dataset-notes").value = "";
    renderDatasetFiles();
    renderPresentPicker();
    renderDatasets();
    renderBenchDatasets();
    toast(`Dataset saved (${data.dataset.frames.length} frames)`, "ok");
  } catch (error) {
    errorElement.hidden = false;
    errorElement.textContent = error.message;
  } finally {
    button.textContent = original;
    button.disabled = false;
  }
}

async function handleDatasetListClick(event) {
  const row = event.target.closest(".dataset-row");
  if (!row) return;
  const action = event.target.closest("[data-action]")?.dataset.action;
  const datasetId = row.dataset.dataset;
  if (action === "delete-dataset") {
    if (!window.confirm("Delete this dataset and its frames?")) return;
    try {
      const data = await api(`/api/datasets/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
      state.datasets = data.datasets || [];
      renderDatasets();
      renderBenchDatasets();
      toast("Dataset deleted", "ok");
    } catch (error) { toast(error.message, "error"); }
    return;
  }
  if (action === "use-dataset") {
    switchTab("benchmark");
    const input = $(`#bench-datasets input[value="${CSS.escape(datasetId)}"]`);
    if (input) input.checked = true;
  }
}

/* -------------------------------------------------------------- benchmark */

function renderBenchOptions() {
  const prior = (id) => {
    const inputs = $$(`#${id} input`);
    return inputs.length ? new Set(inputs.filter((i) => i.checked).map((i) => i.value)) : null;
  };
  const keep = { det: prior("bench-detectors"), rec: prior("bench-recognizers"), size: prior("bench-det-sizes"), tile: prior("bench-tilings") };
  const on = (set, value, fallback) => ((set ? set.has(String(value)) : fallback) ? "checked" : "");

  const detectors = state.registry.detectors.filter((entry) => entry.available);
  const recognizers = state.registry.recognizers.filter((entry) => entry.available);
  $("#bench-detectors").innerHTML = detectors.length
    ? detectors.map((entry) => {
        const note = entry.input_sizes ? ` · ${entry.input_sizes.join("/")}px only` : "";
        return `<label class="check"><input type="checkbox" value="${esc(entry.id)}" ${on(keep.det, entry.id, true)}><span>${esc(entry.label)}<span class="muted">${esc(note)}</span></span></label>`;
      }).join("")
    : `<div class="muted small">No detectors installed.</div>`;
  $("#bench-recognizers").innerHTML = recognizers.length
    ? recognizers.map((entry) => `<label class="check"><input type="checkbox" value="${esc(entry.id)}" ${on(keep.rec, entry.id, true)}><span>${esc(entry.label)}</span></label>`).join("")
    : `<div class="muted small">No recognizers installed.</div>`;

  const sizes = [...new Set([...(state.settings.benchmarkDetSizes || []), 640, 1024, 1280])].sort((a, b) => a - b);
  $("#bench-det-sizes").innerHTML = sizes
    .map((size) => `<label class="chip"><input type="checkbox" value="${size}" ${on(keep.size, size, size === 640 || size === 1280)}><span>${size}px</span></label>`)
    .join("");
  $("#bench-tilings").innerHTML = [
    ["none", "Off"],
    ["sahi_upper_band", "Upper band"],
    ["sahi_full", "Full SAHI"],
  ]
    .map(([value, label]) => `<label class="chip"><input type="checkbox" value="${value}" ${on(keep.tile, value, value === "none")}><span>${label}</span></label>`)
    .join("");
}

function renderBenchDatasets() {
  const wrap = $("#bench-datasets");
  if (!wrap) return;
  $("#bench-dataset-count").textContent = String(state.datasets.length);
  if (!state.datasets.length) {
    wrap.innerHTML = `<div class="muted small">No datasets yet — build one in the Datasets tab.</div>`;
    return;
  }
  const current = new Set([...wrap.querySelectorAll("input:checked")].map((input) => input.value));
  wrap.innerHTML = state.datasets
    .map((dataset) => {
      const id = dataset.datasetId || dataset.dataset_id;
      return `<label class="check"><input type="checkbox" value="${esc(id)}" ${current.has(id) ? "checked" : ""}><span>${esc(dataset.name)} <span class="muted">${plural(dataset.frames, "frame")}</span></span></label>`;
    })
    .join("");
}

function checkedValues(containerId) {
  return $$(`#${containerId} input:checked`).map((input) => input.value);
}

async function runBenchmark() {
  const datasetIds = checkedValues("bench-datasets");
  if (!datasetIds.length) { setBanner("#bench-error", "Select at least one dataset."); return; }
  const detectors = checkedValues("bench-detectors");
  const recognizers = checkedValues("bench-recognizers");
  if (!detectors.length || !recognizers.length) { setBanner("#bench-error", "Select at least one detector and recognizer."); return; }

  setBanner("#bench-error", "");
  const acceptRaw = $("#bench-accept").value.trim();
  const payload = {
    dataset_ids: datasetIds,
    detectors,
    recognizers,
    det_sizes: checkedValues("bench-det-sizes").map(Number),
    tiling_modes: checkedValues("bench-tilings"),
    det_thresh: readNumber("bench-det-thresh", 0.35),
    nms_thresh: readNumber("bench-nms-thresh", state.settings.nmsThresh ?? 0.45),
    flip_aug: $("#bench-flip").checked,
    lowres_aug: $("#bench-lowres").checked,
    with_quality: $("#bench-quality").checked,
    with_pose: $("#bench-pose").checked,
    accept_threshold: acceptRaw ? Number(acceptRaw) : null,
    review_threshold: readNumber("bench-review", 0.34),
    min_margin: readNumber("bench-margin", 0.04),
    assignment_mode: $("#bench-assignment").value,
    template_strategy: $("#bench-template-strategy")?.value || state.settings.templateStrategy || "top_k_mean",
    top_k: Math.max(1, Math.round(readNumber("bench-top-k", state.settings.templateTopK ?? 2))),
    fusion_mode: $("#bench-fusion").value,
    k: Math.max(1, Math.round(readNumber("bench-fusion-k", state.settings.fusionK ?? 2))),
    min_quality: readNumber("bench-min-quality", state.settings.minFaceQuality ?? 0),
    force_fresh: $("#bench-force-fresh").checked,
    label_unknown: $("#bench-label-unknown").checked,
  };

  if (!payload.det_sizes.length || !payload.tiling_modes.length) {
    setBanner("#bench-error", "Select at least one resolution and one tiling mode.");
    return;
  }

  const button = $("#bench-run-btn");
  if (state.busy || button.disabled) return;
  const original = button.textContent;
  button.textContent = "Running…";
  button.disabled = true;
  state.busy = true;
  $("#bench-status").hidden = false;
  const startedAt = performance.now();
  const timer = window.setInterval(() => {
    $("#bench-status-text").textContent = `Warming models and measuring… ${((performance.now() - startedAt) / 1000).toFixed(0)}s`;
  }, 250);

  try {
    const run = await api("/api/benchmark/run", jsonRequest(payload));
    state.benchmarkRun = run;
    state.benchSelected = null;
    state.benchSort = { key: "f1", dir: -1 };
    $("#bench-detail").hidden = true;
    renderBenchmark(run);
    loadBenchmarks();
    const ok = run.rows.filter((row) => !row.error).length;
    toast(`Benchmark finished · ${ok}/${run.rows.length} combinations scored`, "ok");
  } catch (error) {
    setBanner("#bench-error", error.message);
    toast(error.message, "error");
  } finally {
    window.clearInterval(timer);
    $("#bench-status").hidden = true;
    button.textContent = original;
    button.disabled = false;
    state.busy = false;
  }
}

function metricClass(value) {
  if (value >= 0.9) return "good";
  if (value >= 0.7) return "warn";
  return "bad";
}

const BENCH_COLUMNS = [
  { key: "detector", label: "Detector", text: (row) => row.detector.split(":").pop(), value: (row) => row.detector },
  { key: "recognizer", label: "Recognizer", text: (row) => row.recognizer.split(":").pop(), value: (row) => row.recognizer },
  { key: "detSize", label: "Res", text: (row) => row.detSize, value: (row) => row.detSize },
  { key: "tiling", label: "Tiling", text: (row) => row.tilingMode, value: (row) => row.tilingMode },
  {
    key: "f1",
    label: "F1",
    text: (row) => fixed(row.metrics?.f1),
    // F1 dominates; similarity margin breaks ties.
    value: (row) => (row.metrics?.f1 ?? -1) * 1000 + Math.max(0, row.simMargin ?? 0),
    cls: (row) => metricClass(row.metrics?.f1 ?? 0),
  },
  { key: "precision", label: "Prec", text: (row) => fixed(row.metrics?.precision), value: (row) => row.metrics?.precision ?? -1 },
  { key: "recall", label: "Rec", text: (row) => fixed(row.metrics?.recall), value: (row) => row.metrics?.recall ?? -1 },
  { key: "tauF1", label: "τ*", text: (row) => fixed(row.tauF1, 2), value: (row) => row.tauF1 ?? -1 },
  { key: "meanSim", label: "Mean Sim", text: (row) => dash(row.meanPresentSim), value: (row) => row.meanPresentSim ?? -1 },
  { key: "impostor", label: "Impostor", text: (row) => dash(row.maxImpostorSim), value: (row) => row.maxImpostorSim ?? -1 },
  {
    key: "buckets",
    label: "L/M/S recall",
    // Only meaningful with ground-truth boxes; otherwise show a dash, not 0.000.
    text: (row) => {
      if (!row.detection?.has_annotations) return "—";
      const b = row.detection?.buckets || {};
      return `${fixed(b.large?.identification_recall)} / ${fixed(b.medium?.identification_recall)} / ${fixed(b.small?.identification_recall)}`;
    },
    value: (row) =>
      row.detection?.has_annotations
        ? row.detection?.buckets?.medium?.identification_recall ?? -1
        : -1,
  },
  {
    key: "faces",
    label: "Faces L/M/S",
    text: (row) => {
      const b = row.detectedBuckets || row.detection?.detected_buckets || {};
      return `${b.large || 0} / ${b.medium || 0} / ${b.small || 0}`;
    },
    value: (row) => {
      const b = row.detectedBuckets || row.detection?.detected_buckets || {};
      return (b.large || 0) + (b.medium || 0) + (b.small || 0);
    },
  },
  {
    key: "fps",
    label: "FPS",
    text: (row) => `${fixed(row.fps, 1)}${row.cachedFrames ? "*" : ""}`,
    value: (row) => row.fps ?? -1,
  },
  { key: "detP50", label: "det p50", text: (row) => fixed(row.latency?.detect_ms_p50, 0), value: (row) => row.latency?.detect_ms_p50 ?? -1 },
  { key: "totalP50", label: "total p50", text: (row) => fixed(row.latency?.total_inference_ms_p50, 0), value: (row) => row.latency?.total_inference_ms_p50 ?? -1 },
];

const comboKey = (row) => `${row.detector}|${row.recognizer}|${row.detSize}|${row.tilingMode}`;

function renderBenchmark(run) {
  $("#bench-row-count").textContent = String(run.rows.length);
  const wrap = $("#bench-matrix");
  if (!run.rows.length) {
    wrap.innerHTML = `<div class="empty"><strong>No combinations</strong><span>Widen the grid and run again.</span></div>`;
    return;
  }

  const sort = state.benchSort;
  const column = BENCH_COLUMNS.find((item) => item.key === sort.key);
  const rows = [...run.rows];
  if (column?.value) {
    rows.sort((a, b) => {
      const left = column.value(a);
      const right = column.value(b);
      if (left === right) return 0;
      return (left > right ? 1 : -1) * sort.dir;
    });
  }

  const header = BENCH_COLUMNS.map((item) => {
    const sorted = sort.key === item.key ? (sort.dir === 1 ? " ▲" : " ▼") : "";
    return `<th class="sortable" data-sort="${item.key}">${item.label}${sorted}</th>`;
  }).join("");

  const body = rows
    .map((row) => {
      if (row.error) {
        return `<tr class="error-row"><td>${esc(row.detector.split(":").pop())}</td><td>${esc(row.recognizer.split(":").pop())}</td><td>${row.detSize ?? "—"}</td><td>${esc(row.tilingMode || "—")}</td><td colspan="${BENCH_COLUMNS.length - 4}">${esc(row.error)}</td></tr>`;
      }
      const key = comboKey(row);
      const cells = BENCH_COLUMNS.map((item) => {
        if (item.key === "detector" || item.key === "recognizer") {
          return `<td title="${esc(item.key === "detector" ? row.detector : row.recognizer)}">${esc(item.text(row))}</td>`;
        }
        const cls = item.cls ? ` ${item.cls(row)}` : "";
        return `<td class="mono${cls}">${esc(String(item.text(row)))}</td>`;
      }).join("");
      return `<tr data-combo="${esc(key)}" class="${state.benchSelected === key ? "selected" : ""}">${cells}</tr>`;
    })
    .join("");

  wrap.innerHTML = `<table class="matrix"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;

  for (const th of $$("th.sortable", wrap)) {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (state.benchSort.key === key) state.benchSort.dir *= -1;
      else state.benchSort = { key, dir: key === "detector" || key === "recognizer" ? 1 : -1 };
      renderBenchmark(run);
    });
  }
  for (const tr of $$("tbody tr[data-combo]", wrap)) {
    tr.addEventListener("click", () => {
      const row = run.rows.find((item) => !item.error && comboKey(item) === tr.dataset.combo);
      showBenchmarkRow(row, tr);
    });
  }
  if (state.benchSelected) {
    const row = run.rows.find((item) => !item.error && comboKey(item) === state.benchSelected);
    if (row) {
      const tr = $(`tbody tr[data-combo="${CSS.escape(state.benchSelected)}"]`, wrap);
      showBenchmarkRow(row, tr, { scroll: false });
    }
  }
}

function curveSvg(curve) {
  if (!curve?.length) return "";
  const width = 520;
  const height = 130;
  const pad = 10;
  const denom = Math.max(1, curve.length - 1);
  const x = (index) => pad + (index / denom) * (width - pad * 2);
  const y = (value) => height - pad - clamp(value, 0, 1) * (height - pad * 2);
  const path = (key) =>
    curve.map((point, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(1)} ${y(point[key]).toFixed(1)}`).join(" ");
  // Match backend pick_operating_points(): highest (f1, precision, threshold).
  const isBetter = (a, b) =>
    a.f1 !== b.f1
      ? a.f1 > b.f1
      : a.precision !== b.precision
        ? a.precision > b.precision
        : a.threshold > b.threshold;
  const bestIndex = curve.reduce((best, point, index) => (isBetter(point, curve[best]) ? index : best), 0);
  const best = curve[bestIndex];
  const grid = [0.25, 0.5, 0.75]
    .map((tick) => `<line x1="${pad}" y1="${y(tick).toFixed(1)}" x2="${width - pad}" y2="${y(tick).toFixed(1)}" class="curve-grid" />`)
    .join("");
  return `
    <div class="curve-wrap">
      <svg class="curve" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="precision, recall and F1 against threshold">
        ${grid}
        <line x1="${x(bestIndex).toFixed(1)}" y1="${pad}" x2="${x(bestIndex).toFixed(1)}" y2="${height - pad}" class="curve-marker" />
        <path d="${path("precision")}" class="curve-line precision" />
        <path d="${path("recall")}" class="curve-line recall" />
        <path d="${path("f1")}" class="curve-line f1" />
      </svg>
      <span class="curve-note">F1 ${fixed(best.f1)} @ τ ${fixed(best.threshold, 2)}</span>
      <div class="curve-axis"><span>τ ${fixed(curve[0].threshold, 2)}</span><span>τ ${fixed(curve[curve.length - 1].threshold, 2)}</span></div>
    </div>
    <div class="curve-legend"><span class="curve-key precision"></span>precision <span class="curve-key recall"></span>recall <span class="curve-key f1"></span>F1</div>`;
}

function benchmarkRecipe(row) {
  const cfg = state.benchmarkRun?.config || {};
  // Runs saved before the recipe field existed only have row-level fields; fill
  // the shared request params from the run config.
  const base = row.recipe || {
    detector: row.detector,
    recognizer: row.recognizer,
    detSize: row.detSize,
    tilingMode: row.tilingMode,
    detThresh: cfg.detThresh,
    nmsThresh: cfg.nmsThresh,
    acceptThreshold: row.acceptThreshold,
    reviewThreshold: row.reviewThreshold,
    minMargin: row.minMargin,
    assignmentMode: row.assignmentMode,
    templateStrategy: row.templateStrategy,
    fusionMode: row.fusionMode,
    flipAug: cfg.flipAug,
    lowresAug: cfg.lowresAug,
    withQuality: cfg.withQuality,
    withPose: cfg.withPose,
    variant: row.variant,
  };
  return {
    ...base,
    benchmark: {
      runId: state.benchmarkRun?.runId || null,
      datasetIds: state.benchmarkRun?.config?.datasetIds || [],
      metrics: row.metrics || null,
      tauF1: row.tauF1 ?? null,
      far0: row.far0 ? { threshold: row.far0.threshold, precision: row.far0.precision, recall: row.far0.recall } : null,
      prec98: row.prec98 ? { threshold: row.prec98.threshold, precision: row.prec98.precision, recall: row.prec98.recall } : null,
      meanPresentSim: row.meanPresentSim ?? null,
      maxImpostorSim: row.maxImpostorSim ?? null,
      fps: row.fps ?? null,
    },
    exportedAt: new Date().toISOString(),
  };
}

function applyRecipeToScan(row) {
  const recipe = row.recipe || benchmarkRecipe(row);
  state.detector = recipe.detector;
  state.recognizer = recipe.recognizer;
  remember();
  renderModelPickers();
  setSelectValue("det-size", recipe.detSize, `${recipe.detSize} px`);
  setSelectValue("tiling-mode", recipe.tilingMode);
  const setRange = (id, value) => {
    const element = $(`#${id}`);
    if (!element || value === undefined || value === null) return;
    element.value = String(value);
    element.dispatchEvent(new Event("input"));
  };
  setRange("det-thresh", recipe.detThresh);
  setRange("nms-thresh", recipe.nmsThresh);
  setRange("accept-threshold", recipe.acceptThreshold);
  setRange("review-threshold", recipe.reviewThreshold);
  setRange("min-margin", recipe.minMargin);
  setRange("min-quality", recipe.minQuality);
  setSelectValue("assignment-mode", recipe.assignmentMode);
  setSelectValue("template-strategy", recipe.templateStrategy);
  setSelectValue("fusion-mode", recipe.fusionMode);
  const fusionK = $("#fusion-k");
  if (fusionK && recipe.fusionK != null) fusionK.value = String(recipe.fusionK);
  const setToggle = (id, on) => {
    const element = $(`#${id}`);
    if (element && on !== undefined && on !== null) element.checked = Boolean(on);
  };
  setToggle("flip-aug", recipe.flipAug);
  setToggle("lowres-aug", recipe.lowresAug);
  setToggle("with-quality", recipe.withQuality);
  setToggle("with-pose", recipe.withPose);
  refreshModelStatus();
  switchTab("attendance");
  toast(
    `Applied ${recipe.detector.split(":").pop()} + ${recipe.recognizer.split(":").pop()} to the scan panel`,
    "ok",
  );
}

async function copyRecipe(row) {
  const text = JSON.stringify(benchmarkRecipe(row), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    toast("Benchmark settings copied to clipboard", "ok");
  } catch {
    window.prompt("Copy the benchmark settings:", text);
  }
}

function downloadRecipe(row) {
  const recipe = benchmarkRecipe(row);
  const name = `${recipe.detector.split(":").pop()}-${recipe.recognizer.split(":").pop()}-${recipe.detSize}`
    .replace(/[^a-z0-9._-]+/gi, "-");
  const blob = new Blob([JSON.stringify(recipe, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${name}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  toast(`Downloaded ${name}.json`, "ok");
}

function showBenchmarkRow(row, tr, options = {}) {
  if (!row) return;
  state.benchSelected = comboKey(row);
  for (const other of $$("tbody tr")) other.classList.remove("selected");
  tr?.classList.add("selected");
  const detail = $("#bench-detail");
  detail.hidden = false;
  if (row.error) {
    detail.innerHTML = `<div class="bench-detail-head"><strong>${esc(row.detector)}</strong> + <strong>${esc(row.recognizer)}</strong></div><div class="banner error">${esc(row.error)}</div>`;
    return;
  }
  const buckets = row.detection?.buckets || {};
  const hasGt = Boolean(row.detection?.has_annotations);
  const detected = row.detectedBuckets || row.detection?.detected_buckets || {};
  const reading = (key) =>
    ["large", "medium", "small"]
      .map((name) => `<div class="bucket"><span class="muted">${name}</span><b class="mono">${fixed(buckets[name]?.[key])}</b><span class="muted small">${buckets[name]?.gt || 0} gt</span></div>`)
      .join("");
  const detectedReading = ["large", "medium", "small"]
    .map((name) => `<div class="bucket"><span class="muted">${name}</span><b class="mono">${detected[name] || 0}</b></div>`)
    .join("");
  const weightBits = [
    row.detectorSha256 ? `det ${row.detectorSha256.slice(0, 10)}` : "",
    row.recognizerSha256 ? `rec ${row.recognizerSha256.slice(0, 10)}` : "",
    (row.providers || []).length ? `EP: ${(row.providers || []).join(", ")}` : "",
  ].filter(Boolean).join(" · ");
  const frames = row.frames || [];
  const timingSource = row.cachedFrames
    ? `${row.cachedFrames}/${row.frameCount ?? frames.length} cached`
    : "fresh";
  const fpsText = row.fps == null ? "—" : `${fixed(row.fps, 1)} fps${row.cachedFrames ? "*" : ""}`;
  const framesHtml = frames.length
    ? `<div class="inspect-section">
        <span class="field-label">Annotated frames (${frames.length}) · click to open viewer</span>
        <div class="bench-frames">${frames
          .map((frame, index) => `
            <button type="button" class="bench-frame-thumb" data-frame-index="${index}" title="${esc(frame.frameId || `Frame ${index + 1}`)}">
              <img src="${esc(frame.annotatedUrl || frame.imageUrl)}" alt="" loading="lazy">
              <span class="thumb-meta">${esc(frame.frameId || `Frame ${index + 1}`)} · ${plural(frame.faces?.length || 0, "face")}</span>
            </button>`)
          .join("")}</div>
      </div>`
    : `<div class="inspect-section"><span class="field-label">Annotated frames</span><div class="bench-frames-empty">No saved frames for this run.</div></div>`;
  detail.innerHTML = `
    <div class="bench-detail-head">
      <strong>${esc(row.detector)}</strong> + <strong>${esc(row.recognizer)}</strong>
      <span class="muted">${row.detSize}px · ${esc(row.tilingMode)} · ${esc(row.variant)} · ${esc(row.assignmentMode)}</span>
      <div class="bench-detail-actions">
        <button class="btn ghost small" type="button" data-action="apply-scan">Apply to scan</button>
        <button class="btn ghost small" type="button" data-action="copy-recipe">Copy settings</button>
        <button class="btn ghost small" type="button" data-action="download-recipe">Download JSON</button>
      </div>
    </div>
    <div class="inspect-grid">
      <div><span class="muted">Best F1</span><b>${fixed(row.bestF1)}</b></div>
      <div><span class="muted">τ* (F1)</span><b>${fixed(row.tauF1, 2)}</b></div>
      <div><span class="muted">τ (FAR=0)</span><b>${row.far0 ? fixed(row.far0.threshold, 2) : "—"}</b></div>
      <div><span class="muted">τ (P≥0.98)</span><b>${row.prec98 ? fixed(row.prec98.threshold, 2) : "—"}</b></div>
      <div><span class="muted">Mean present sim</span><b>${dash(row.meanPresentSim)}</b></div>
      <div><span class="muted">Max impostor sim</span><b>${dash(row.maxImpostorSim)}</b></div>
      <div><span class="muted">Throughput</span><b>${fpsText}</b></div>
      <div><span class="muted">Total (mean)</span><b>${fixed(row.latency?.total_inference_ms_mean, 0)} ms</b></div>
      <div><span class="muted">Latency p50 / p95</span><b>${fixed(row.latency?.total_inference_ms_p50, 0)} / ${fixed(row.latency?.total_inference_ms_p95, 0)} ms</b></div>
      <div><span class="muted">Detected / expected</span><b>${row.facesDetected} / ${row.expectedFaces}</b></div>
      <div><span class="muted">Timing source</span><b>${esc(timingSource)}</b></div>
      ${weightBits ? `<div class="wide"><span class="muted">Weights / provider</span><b class="mono small">${esc(weightBits)}</b></div>` : ""}
    </div>
    ${curveSvg(row.curve)}
    <div class="inspect-section">
      <span class="field-label">Faces detected by size (all frames)</span>
      <div class="bucket-row">${detectedReading}</div>
    </div>
    ${hasGt
      ? `<div class="inspect-section">
          <span class="field-label">Identification recall by face size</span>
          <div class="bucket-row">${reading("identification_recall")}</div>
        </div>
        <div class="inspect-section">
          <span class="field-label">Detection recall by face size</span>
          <div class="bucket-row">${reading("detection_recall")}</div>
        </div>`
      : `<div class="inspect-section"><span class="field-label">Recall by face size</span><div class="muted small">No ground-truth boxes in this dataset — add annotated boxes to compute size-stratified recall.</div></div>`}
    ${framesHtml}`;
  for (const button of $$("[data-action]", detail)) {
    button.addEventListener("click", () => {
      if (button.dataset.action === "apply-scan") applyRecipeToScan(row);
      else if (button.dataset.action === "copy-recipe") copyRecipe(row);
      else if (button.dataset.action === "download-recipe") downloadRecipe(row);
    });
  }
  for (const thumb of $$(".bench-frame-thumb", detail)) {
    thumb.addEventListener("click", () => openLightbox(frames, Number(thumb.dataset.frameIndex)));
  }
  if (options.scroll !== false) detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadBenchmarks() {
  try {
    const data = await api("/api/benchmarks");
    state.benchmarks = data.benchmarks || [];
    renderBenchmarkHistory();
  } catch (error) { toast(error.message, "error"); }
}

function renderBenchmarkHistory() {
  const list = $("#bench-history");
  $("#bench-history-count").textContent = plural(state.benchmarks.length, "run");
  if (!state.benchmarks.length) {
    list.innerHTML = `<div class="empty small"><strong>No benchmark runs yet</strong><span>Configure a grid and run it.</span></div>`;
    return;
  }
  list.innerHTML = state.benchmarks
    .map((run) => {
      const config = run.config || {};
      return `
        <div class="history-row" data-run="${esc(run.runId)}">
          <div class="history-main">
            <div class="history-title"><strong>${esc(run.runId)}</strong><span class="muted">${esc(formatWhen(run.createdAt))}</span></div>
            <div class="history-meta">${run.rows} rows · ${(config.detectors || []).length} detectors × ${(config.recognizers || []).length} recognizers · ${(config.detSizes || []).join("/")}px</div>
          </div>
          <div class="history-actions"><button class="btn ghost small" type="button" data-action="load-run">Load</button></div>
        </div>`;
    })
    .join("");
}

async function handleBenchHistoryClick(event) {
  const row = event.target.closest(".history-row");
  if (!row) return;
  try {
    const run = await api(`/api/benchmarks/${encodeURIComponent(row.dataset.run)}`);
    state.benchmarkRun = run;
    state.benchSelected = null;
    $("#bench-detail").hidden = true;
    renderBenchmark(run);
    toast(`Loaded benchmark ${run.runId}`, "ok");
  } catch (error) { toast(error.message, "error"); }
}

/* ------------------------------------------------------------- tab + reset */

const TABS = ["attendance", "benchmark", "datasets"];

function switchTab(name, options = {}) {
  if (!TABS.includes(name)) name = "attendance";
  if (state.lightbox) closeLightbox();
  for (const tab of $$("#tabs .tab")) tab.classList.toggle("active", tab.dataset.tab === name);
  for (const panel of $$(".workspace[data-panel]")) panel.hidden = panel.dataset.panel !== name;
  if (name === "datasets") { renderDatasets(); renderPresentPicker(); }
  if (name === "benchmark") { renderBenchDatasets(); renderBenchOptions(); }
  if (!options.silent && window.location.hash.slice(1) !== name) {
    window.location.hash = name;
  }
}

function setBanner(selector, message) {
  const element = $(selector);
  element.hidden = !message;
  element.textContent = message || "";
}

async function resetEverything() {
  if (!window.confirm("Clear all students, scans, datasets, and results? Model packs are kept.")) return;
  try {
    const data = await api("/api/reset", { method: "POST" });
    state.people = data.state.people;
    state.scan = null;
    state.frameZoom = null;
    closeStudentsDrawer();
    closeInspector();
    closeLightbox();
    state.results = [];
    state.datasets = [];
    state.benchmarks = [];
    state.faceFiles = [];
    state.frameFiles = [];
    state.datasetFiles = [];
    state.datasetPresent = new Set();
    state.benchmarkRun = null;
    state.benchSelected = null;
    state.liveSweep = null;
    state.datasetBoxes = {};
    state.selectedFrame = 0;
    $("#bench-row-count").textContent = "0";
    renderFaceFiles();
    renderFrameFiles();
    renderDatasetFiles();
    renderPeople();
    renderModelPickers();
    renderHistory();
    renderResults();
    renderDatasets();
    renderBenchDatasets();
    renderBenchmarkHistory();
    $("#bench-matrix").innerHTML = "";
    $("#bench-detail").hidden = true;
    await loadDatasets();
    toast("Reset: everything cleared", "ok");
  } catch (error) { toast(error.message, "error"); }
}

/* ------------------------------------------------------------------- load */

function setSelectValue(id, value, label = String(value)) {
  const select = $(`#${id}`);
  const text = String(value ?? "");
  if (text && ![...select.options].some((option) => option.value === text)) {
    select.append(new Option(label, text));
  }
  if (text) select.value = text;
}

function applySettings(settings) {
  if (!settings) return;
  state.settings = { ...state.settings, ...settings };
  $("#max-frames").textContent = state.settings.maxFrames;
  setSelectValue("det-size", state.settings.detSize, `${state.settings.detSize} px`);
  setSelectValue("tiling-mode", state.settings.tilingMode);
  $("#det-thresh").value = String(state.settings.detThresh);
  $("#nms-thresh").value = String(state.settings.nmsThresh);
  $("#accept-threshold").value = String(state.settings.acceptThreshold);
  $("#review-threshold").value = String(state.settings.reviewThreshold);
  $("#min-margin").value = String(state.settings.minMargin);
  $("#min-quality").value = String(state.settings.minFaceQuality);
  setSelectValue("assignment-mode", state.settings.assignmentMode);
  setSelectValue("template-strategy", state.settings.templateStrategy);
  setSelectValue("fusion-mode", state.settings.fusionMode);
  $("#fusion-k").value = String(state.settings.fusionK);
  if ($("#bench-review")) $("#bench-review").value = String(state.settings.reviewThreshold);
  if ($("#bench-margin")) $("#bench-margin").value = String(state.settings.minMargin);
  if ($("#bench-det-thresh")) $("#bench-det-thresh").value = String(state.settings.detThresh);
  if ($("#bench-nms-thresh")) $("#bench-nms-thresh").value = String(state.settings.nmsThresh);
  if ($("#bench-min-quality")) $("#bench-min-quality").value = String(state.settings.minFaceQuality);
  if ($("#bench-assignment")) setSelectValue("bench-assignment", state.settings.assignmentMode);
  if ($("#bench-template-strategy")) setSelectValue("bench-template-strategy", state.settings.templateStrategy);
  if ($("#bench-top-k")) $("#bench-top-k").value = String(state.settings.templateTopK ?? 2);
  if ($("#bench-fusion")) setSelectValue("bench-fusion", state.settings.fusionMode);
  if ($("#bench-fusion-k")) $("#bench-fusion-k").value = String(state.settings.fusionK ?? 2);
  for (const id of ["det-thresh", "nms-thresh", "accept-threshold", "review-threshold", "min-margin", "min-quality"]) {
    $(`#${id}`).dispatchEvent(new Event("input"));
  }
  $("#ephemeral-banner").hidden = !state.settings.ephemeral;
}

async function loadState() {
  const data = await api("/api/state");
  state.people = data.people || [];
  if (data.registry) state.registry = data.registry;
  applySettings(data.settings);
  renderPeople();
  renderModelPickers();
  renderPresentPicker();
}

async function loadModels() {
  const data = await api("/api/models");
  state.packs = data.packs || [];
  if (data.detectors) state.registry.detectors = data.detectors;
  if (data.recognizers) state.registry.recognizers = data.recognizers;
  const remembered = recall();
  const pick = (entries, rememberedId, fallback) => {
    const usable = (id) => id && entries.some((entry) => entry.id === id && entry.available !== false);
    return (usable(rememberedId) && rememberedId) || (usable(fallback) && fallback) ||
      entries.find((entry) => entry.available !== false)?.id || entries[0]?.id || fallback;
  };
  state.detector = pick(state.registry.detectors, remembered.detector, state.settings.defaultDetector);
  state.recognizer = pick(state.registry.recognizers, remembered.recognizer, state.settings.defaultRecognizer);
  renderModelPickers();
  renderModelUsage();
  renderBenchOptions();
}

async function loadResults() {
  const data = await api("/api/results");
  state.results = data.results || [];
  renderHistory();
}

async function loadDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets || [];
  renderDatasets();
  renderBenchDatasets();
}

/* ------------------------------------------------------------- wire + init */

function wireEvents() {
  $("#tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) switchTab(tab.dataset.tab);
  });
  window.addEventListener("hashchange", () => switchTab(window.location.hash.slice(1), { silent: true }));

  $("#add-person-form").addEventListener("submit", addPerson);
  $("#face-input").addEventListener("change", (event) => {
    addFiles({ files: state.faceFiles, render: renderFaceFiles }, event.target.files, state.settings.maxPhotosPerUpload, "photos per upload");
    event.target.value = "";
  });
  $("#frame-input").addEventListener("change", (event) => {
    addFiles({ files: state.frameFiles, render: renderFrameFiles }, event.target.files, state.settings.maxFrames, "frames");
    event.target.value = "";
  });
  $("#dataset-input").addEventListener("change", (event) => {
    addFiles({ files: state.datasetFiles, render: renderDatasetFiles }, event.target.files, state.settings.maxFrames, "frames");
    event.target.value = "";
  });
  wireDropzone($("#face-drop"), (files) => addFiles({ files: state.faceFiles, render: renderFaceFiles }, files, state.settings.maxPhotosPerUpload, "photos per upload"));
  wireDropzone($("#frame-drop"), (files) => addFiles({ files: state.frameFiles, render: renderFrameFiles }, files, state.settings.maxFrames, "frames"));
  wireDropzone($("#dataset-drop"), (files) => addFiles({ files: state.datasetFiles, render: renderDatasetFiles }, files, state.settings.maxFrames, "frames"));

  // Detector thresholds change detection itself, so they need a fresh scan;
  // only the embedding-space knobs can be swept from the cache.
  for (const id of ["accept-threshold", "review-threshold", "min-margin", "min-quality", "fusion-k", "assignment-mode", "template-strategy", "fusion-mode"]) {
    $(`#${id}`).addEventListener("change", scheduleSweep);
  }

  $("#detector-select").addEventListener("change", () => {
    state.detector = $("#detector-select").value;
    remember();
    renderModelPickers();
    refreshModelStatus();
  });
  $("#recognizer-select").addEventListener("change", () => {
    state.recognizer = $("#recognizer-select").value;
    remember();
    renderModelPickers();
    refreshModelStatus();
  });

  $("#scan-btn").addEventListener("click", runScan);
  $("#reset-btn").addEventListener("click", resetEverything);
  $("#reembed-btn").addEventListener("click", reembedAll);
  $("#students-toggle").addEventListener("click", () => {
    if (state.studentsOpen) closeStudentsDrawer();
    else openStudentsDrawer();
  });
  $("#students-close").addEventListener("click", closeStudentsDrawer);
  $("#drawer-backdrop").addEventListener("click", closeStudentsDrawer);
  $("#bench-setup-toggle").addEventListener("click", () => {
    const collapsed = $("#panel-benchmark").classList.contains("setup-collapsed");
    setBenchSetupCollapsed(!collapsed);
    saveViewPrefs({ benchSetupCollapsed: !collapsed });
  });
  $("#frame-zoom-reset").addEventListener("click", resetFrameZoom);
  $("#frame-view").addEventListener("dblclick", () => { if (state.frameZoom) resetFrameZoom(); });
  $("#toast").addEventListener("pointerenter", pauseToast);
  $("#toast").addEventListener("pointerleave", resumeToast);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseToast();
    else resumeToast();
  });
  $("#inspector-close").addEventListener("click", closeInspector);

  $("#people-list").addEventListener("click", handlePeopleClick);
  $("#people-list").addEventListener("change", handlePeopleChange);
  $("#people-list").addEventListener("dblclick", handleRename);
  $("#history-list").addEventListener("click", handleHistoryClick);

  $("#dataset-form").addEventListener("submit", createDataset);
  $("#dataset-present").addEventListener("click", (event) => {
    const chip = event.target.closest(".present-chip");
    if (!chip) return;
    const id = Number(chip.dataset.person);
    if (state.datasetPresent.has(id)) state.datasetPresent.delete(id);
    else state.datasetPresent.add(id);
    renderPresentPicker();
  });
  $("#dataset-list").addEventListener("click", handleDatasetListClick);

  $("#bench-run-btn").addEventListener("click", runBenchmark);
  $("#bench-history").addEventListener("click", handleBenchHistoryClick);

  // Frame viewer: roster collapse, unknown-label toggle, fullscreen lightbox.
  $("#roster-toggle").addEventListener("click", () => {
    const collapsed = $("#results-body").classList.contains("roster-collapsed");
    setRosterCollapsed(!collapsed);
    saveViewPrefs({ rosterCollapsed: !collapsed });
  });
  $("#label-unknown").addEventListener("change", (event) => {
    setUnknownLabels(event.target.checked);
    saveViewPrefs({ labelUnknown: event.target.checked });
  });
  $("#frame-fullscreen").addEventListener("click", openScanLightbox);
  $("#lightbox-close").addEventListener("click", closeLightbox);
  $("#lightbox-reset").addEventListener("click", lightboxReset);
  $("#lightbox-prev").addEventListener("click", () => lightboxStep(-1));
  $("#lightbox-next").addEventListener("click", () => lightboxStep(1));

  const lightboxViewport = $("#lightbox-viewport");
  lightboxViewport.addEventListener("wheel", (event) => {
    const lb = state.lightbox;
    if (!lb) return;
    event.preventDefault();
    const rect = lightboxViewport.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const next = clamp(lb.scale * factor, LIGHTBOX_MIN_SCALE, LIGHTBOX_MAX_SCALE);
    const ratio = next / lb.scale;
    lb.tx = px - (px - lb.tx) * ratio;
    lb.ty = py - (py - lb.ty) * ratio;
    lb.scale = next;
    applyLightboxZoom();
  }, { passive: false });

  let lightboxDrag = null;
  let suppressLightboxClick = false;
  lightboxViewport.addEventListener("pointerdown", (event) => {
    if (!state.lightbox || event.button !== 0 || lightboxDrag) return;
    suppressLightboxClick = false;
    lightboxDrag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      tx: state.lightbox.tx,
      ty: state.lightbox.ty,
      moved: false,
    };
    try { lightboxViewport.setPointerCapture(event.pointerId); } catch { /* ignore */ }
    lightboxViewport.classList.add("dragging");
  });
  window.addEventListener("pointermove", (event) => {
    if (!lightboxDrag || !state.lightbox || event.pointerId !== lightboxDrag.pointerId) return;
    const dx = event.clientX - lightboxDrag.x;
    const dy = event.clientY - lightboxDrag.y;
    if (!lightboxDrag.moved && Math.hypot(dx, dy) > 4) lightboxDrag.moved = true;
    state.lightbox.tx = lightboxDrag.tx + dx;
    state.lightbox.ty = lightboxDrag.ty + dy;
    applyLightboxPan();
  });
  const endLightboxDrag = (event, cancelled = false) => {
    if (!lightboxDrag || event.pointerId !== lightboxDrag.pointerId) return;
    if (!cancelled && lightboxDrag.moved) {
      suppressLightboxClick = true;
      // Clear on the next tick so releasing outside the viewport (no click
      // fired on lightboxViewport) never swallows the next real box click.
      window.setTimeout(() => { suppressLightboxClick = false; }, 0);
    }
    try { lightboxViewport.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
    lightboxDrag = null;
    lightboxViewport.classList.remove("dragging");
  };
  window.addEventListener("pointerup", (event) => endLightboxDrag(event, false));
  window.addEventListener("pointercancel", (event) => endLightboxDrag(event, true));
  // A pan that ends over a face box must not be treated as a click -> inspector.
  lightboxViewport.addEventListener("click", (event) => {
    if (!suppressLightboxClick) return;
    suppressLightboxClick = false;
    event.preventDefault();
    event.stopPropagation();
  }, true);
  lightboxViewport.addEventListener("dblclick", lightboxReset);
  window.addEventListener("resize", () => {
    if (state.frameZoom) applyFrameZoom();
    if (!state.lightbox) return;
    fitLightboxStage();
    lightboxReset();
  });
  window.addEventListener("keydown", (event) => {
    const target = event.target;
    const typing =
      target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable);
    if (event.key === "Escape") {
      if (state.inspector) { closeInspector(); return; }
      if (state.lightbox) { closeLightbox(); return; }
      if (state.frameZoom) { resetFrameZoom(); return; }
      if (state.studentsOpen) { closeStudentsDrawer(); return; }
    }
    // Never hijack browser shortcuts (Ctrl/Cmd/Alt+F), and only act on the tab
    // that owns the shortcut.
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (!state.lightbox) {
      const onAttendance = !$("#panel-attendance")?.hidden;
      if (onAttendance && (event.key === "f" || event.key === "F") && state.scan?.frames?.length) {
        openScanLightbox();
      }
      return;
    }
    if (event.key === "ArrowLeft") lightboxStep(-1);
    if (event.key === "ArrowRight") lightboxStep(1);
    if (event.key === "0") lightboxReset();
  });
}

async function init() {
  wireEvents();
  applyViewPrefs();
  slider("det-thresh", "det-thresh-value", 0.35);
  slider("nms-thresh", "nms-thresh-value", 0.45);
  slider("accept-threshold", "accept-threshold-value", 0.42);
  slider("review-threshold", "review-threshold-value", 0.34);
  slider("min-margin", "min-margin-value", 0.04);
  slider("min-quality", "min-quality-value", 0.0);
  renderFaceFiles();
  renderFrameFiles();
  renderDatasetFiles();
  renderPresentPicker();
  renderHistory();
  renderResults();

  for (const step of [loadState, loadModels]) {
    try { await step(); } catch (error) { toast(`Startup problem: ${error.message}`, "error"); }
  }
  const settled = await Promise.allSettled([loadResults(), loadDatasets(), loadBenchmarks(), refreshModelStatus()]);
  const failed = settled.find((item) => item.status === "rejected");
  if (failed) toast(`Startup problem: ${failed.reason.message}`, "error");
  switchTab(window.location.hash.slice(1) || "attendance", { silent: true });
}

document.addEventListener("DOMContentLoaded", init);
