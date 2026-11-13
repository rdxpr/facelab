const ACTIVE_MODEL_KEY = "facelab.activeModel";

const state = {
  people: [],
  models: [],
  results: [],
  activeModel: "buffalo_l",
  settings: {
    threshold: 0.5,
    detSize: 960,
    maxFrames: 12,
    maxPhotosPerUpload: 8,
    defaultModel: "buffalo_l",
  },
  modelReady: false,
  modelInfo: null,
  faceFiles: [],
  frameFiles: [],
  scan: null,
  selectedFrame: 0,
  busy: false,
  toastTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatWhen(value) {
  if (!value) return "unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail =
          typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // not JSON; keep the status text
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function toast(message, kind = "") {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast ${kind}`.trim();
  element.hidden = false;
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    element.hidden = true;
  }, 4600);
}

function setModelPill(kind, text) {
  const pill = $("#model-pill");
  pill.classList.remove("loading", "ready", "error");
  pill.classList.add(kind);
  $("#model-status").textContent = text;
}

function setAddError(message) {
  const element = $("#add-error");
  element.hidden = !message;
  element.textContent = message || "";
}

function setScanError(message) {
  const element = $("#scan-error");
  element.hidden = !message;
  element.textContent = message || "";
}

function updateActionState() {
  const ready = state.modelReady && !state.busy;
  $("#add-person-btn").disabled = !ready;
  $("#scan-btn").disabled = !ready;
  const reason = state.modelReady ? "" : "Waiting for the active model to load";
  $("#add-person-btn").title = reason;
  $("#scan-btn").title = reason;
}

function rememberModel(name) {
  try {
    window.localStorage.setItem(ACTIVE_MODEL_KEY, name);
  } catch {
    // localStorage can be blocked; the in-memory value still works.
  }
}

function readRememberedModel() {
  try {
    return window.localStorage.getItem(ACTIVE_MODEL_KEY);
  } catch {
    return null;
  }
}

async function loadState() {
  try {
    const data = await api("/api/state");
    state.people = data.people;
    state.settings = { ...state.settings, ...data.settings };
    $("#max-frames").textContent = state.settings.maxFrames;
    $("#threshold").value = String(state.settings.threshold);
    $("#threshold-value").textContent = Number(state.settings.threshold).toFixed(2);
    const sizeSelect = $("#det-size");
    const target = String(state.settings.detSize);
    if (![...sizeSelect.options].some((option) => option.value === target)) {
      const option = document.createElement("option");
      option.value = target;
      option.textContent = `${target} px`;
      sizeSelect.append(option);
    }
    sizeSelect.value = target;
    renderPeople();
  } catch (error) {
    toast(`Could not load roster: ${error.message}`, "error");
  }
}

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    if (!state.activeModel) {
      state.activeModel = readRememberedModel() || data.defaultModel || "buffalo_l";
    }
    renderModels();
  } catch (error) {
    toast(`Could not list models: ${error.message}`, "error");
  }
}

async function loadModel(name) {
  setModelPill("loading", `${name} loading...`);
  try {
    const info = await api(`/api/model?name=${encodeURIComponent(name)}`);
    state.modelInfo = info;
    state.modelReady = Boolean(info.ready);
    if (info.ready) {
      const loadTime =
        info.loadMs >= 30 ? ` · loaded in ${(info.loadMs / 1000).toFixed(1)}s` : "";
      setModelPill("ready", `${info.name} · ready${loadTime}`);
      $("#model-pill").title = `Model: ${info.path}`;
      $("#model-banner").hidden = true;
    } else {
      setModelPill("error", `${info.name} unavailable`);
      $("#model-pill").title = info.error || "Model unavailable";
      $("#model-banner").hidden = false;
      $("#model-banner").textContent = info.installed === false
        ? `Model pack "${info.name}" is not installed. Run "python scripts/fetch_models.py --model ${info.name}" or add its ONNX files to the models directory.`
        : `Face model could not load: ${info.error}`;
    }
  } catch (error) {
    state.modelReady = false;
    setModelPill("error", "Model unavailable");
    $("#model-banner").hidden = false;
    $("#model-banner").textContent = `Face model could not load: ${error.message}`;
  } finally {
    updateActionState();
  }
}

async function loadResults() {
  try {
    const data = await api("/api/results");
    state.results = data.results || [];
    renderHistory();
  } catch (error) {
    toast(`Could not load saved runs: ${error.message}`, "error");
  }
}

function selectModel(name) {
  if (!name) return;
  state.activeModel = name;
  rememberModel(name);
  $("#model-name").textContent = name;
  $("#enroll-model").textContent = name;
  $("#scan-model").textContent = name;
  const select = $("#model-select");
  if (select.value !== name) select.value = name;
  renderModels();
  updateActionState();
  return loadModel(name);
}

function renderModels() {
  const select = $("#model-select");
  const models = state.models;
  select.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.installed ? model.name : `${model.name} · missing`;
    select.append(option);
  }
  if (!models.some((model) => model.name === state.activeModel)) {
    const option = document.createElement("option");
    option.value = state.activeModel;
    option.textContent = state.activeModel;
    select.append(option);
  }
  select.value = state.activeModel;

  const list = $("#model-usage-list");
  if (!models.length) {
    list.innerHTML = `
      <div class="empty small">
        <strong>No model packs found</strong>
        <span>Run <code>python scripts/fetch_models.py</code> to download one.</span>
      </div>`;
    return;
  }

  list.innerHTML = models
    .map((model) => {
      const active = model.name === state.activeModel;
      const badges = [
        model.default ? `<span class="badge default">default</span>` : "",
        model.installed ? "" : `<span class="badge missing">missing</span>`,
        active ? `<span class="badge active">active</span>` : "",
      ].join("");
      return `
        <div class="model-row${active ? " is-active" : ""}" data-model="${esc(model.name)}">
          <div class="model-main">
            <span class="model-name">${esc(model.name)}</span>
            ${badges}
          </div>
          <div class="model-stats">
            <b>${model.templates}</b> ${model.templates === 1 ? "template" : "templates"}
            · <b>${model.people}</b> ${model.people === 1 ? "student" : "students"}
          </div>
          <button class="btn ghost small" type="button" data-action="use-model" ${active ? "disabled" : ""}>
            ${active ? "Active" : "Use"}
          </button>
        </div>`;
    })
    .join("");
}

function renderPeople() {
  const list = $("#people-list");
  list.innerHTML = "";
  $("#student-count").textContent = String(state.people.length);
  const templates = state.people.reduce((total, person) => total + person.templates, 0);
  $("#template-total").textContent = plural(templates, "template");

  if (!state.people.length) {
    list.innerHTML = `
      <li class="empty">
        <strong>No students yet</strong>
        <span>Add a name and one or more face photos to get started.</span>
      </li>`;
    return;
  }

  list.innerHTML = state.people
    .map((person) => {
      const thumbs = person.photos.length
        ? person.photos
            .map(
              (photo) => `
                <div class="thumb" title="${esc(photo.quality || "Face photo")} · ${esc(photo.model)}">
                  <img src="${esc(photo.url)}" alt="" loading="lazy">
                  <span class="thumb-model">${esc(photo.model)}</span>
                  <button class="thumb-x" data-action="remove-photo" data-photo-id="${esc(photo.id)}" title="Remove this photo" type="button">×</button>
                </div>`,
            )
            .join("")
        : `<div class="thumb-empty">No photos — add one before scanning.</div>`;

      const modelBits = Object.entries(person.models || {})
        .map(([name, count]) => `${esc(name)} ${count}`)
        .join(" · ");

      return `
        <li class="person" data-id="${person.id}">
          <div class="person-top">
            <span class="num">${String(person.number).padStart(2, "0")}</span>
            <div class="person-title">
              <div class="person-name" title="Double-click to rename">${esc(person.name)}</div>
              <div class="person-meta">${plural(person.templates, "face photo")}${modelBits ? ` · ${modelBits}` : ""}</div>
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

function renderHistory() {
  const list = $("#history-list");
  const runs = state.results;
  $("#history-count").textContent = plural(runs.length, "run");
  if (!runs.length) {
    list.innerHTML = `
      <div class="empty small">
        <strong>No saved runs yet</strong>
        <span>Scan a classroom frame and the result is stored here.</span>
      </div>`;
    return;
  }

  list.innerHTML = runs
    .map((run) => {
      const current = state.scan && state.scan.scanId === run.scanId;
      const thumb = run.thumbUrl
        ? `<img src="${esc(run.thumbUrl)}" alt="" loading="lazy">`
        : "";
      return `
        <div class="history-row${current ? " is-current" : ""}" data-scan="${esc(run.scanId)}">
          <div class="history-thumb">${thumb}</div>
          <div class="history-main">
            <div class="history-title">
              <span class="model-tag">${esc(run.model || "?")}</span>
              <span class="muted">${esc(formatWhen(run.createdAt))}</span>
            </div>
            <div class="history-meta">
              ${plural(run.facesDetected ?? 0, "face")} ·
              ${run.matchedCount ?? 0} matched ·
              ${plural(run.frames ?? 0, "frame")} ·
              threshold ${Number(run.threshold ?? 0).toFixed(2)} ·
              ${(Number(run.elapsedMs || 0) / 1000).toFixed(2)}s
            </div>
          </div>
          <div class="history-actions">
            <button class="btn ghost small" type="button" data-action="load-result">Load</button>
            <a class="btn ghost small" href="/api/results/${encodeURIComponent(run.scanId)}" target="_blank" rel="noopener">JSON</a>
            <button class="icon-btn" type="button" data-action="delete-result" title="Delete run">×</button>
          </div>
        </div>`;
    })
    .join("");
}

function renderFaceFiles() {
  const wrap = $("#face-files");
  wrap.innerHTML = "";
  state.faceFiles.forEach((file, index) => {
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
    remove.addEventListener("click", () => {
      state.faceFiles.splice(index, 1);
      renderFaceFiles();
    });

    chip.append(image, label, remove);
    wrap.append(chip);
  });
}

function renderFrameFiles() {
  const wrap = $("#frame-files");
  wrap.innerHTML = "";
  state.frameFiles.forEach((file, index) => {
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
    remove.addEventListener("click", () => {
      state.frameFiles.splice(index, 1);
      renderFrameFiles();
    });

    chip.append(image, label, remove);
    wrap.append(chip);
  });
}

function addFaceFiles(files) {
  const incoming = [...files].filter((file) => file.type.startsWith("image/"));
  if (!incoming.length) return;
  const room = state.settings.maxPhotosPerUpload - state.faceFiles.length;
  if (room <= 0) {
    toast(`Up to ${state.settings.maxPhotosPerUpload} photos per upload`, "error");
    return;
  }
  if (incoming.length > room) {
    toast(`Only ${room} more photo(s) can be added to this upload`, "error");
  }
  state.faceFiles.push(...incoming.slice(0, room));
  renderFaceFiles();
}

function addFrameFiles(files) {
  const incoming = [...files].filter((file) => file.type.startsWith("image/"));
  if (!incoming.length) return;
  const room = state.settings.maxFrames - state.frameFiles.length;
  if (room <= 0) {
    toast(`Up to ${state.settings.maxFrames} frames per scan`, "error");
    return;
  }
  if (incoming.length > room) {
    toast(`Only ${room} more frame(s) can be added`, "error");
  }
  state.frameFiles.push(...incoming.slice(0, room));
  renderFrameFiles();
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

async function addPerson(event) {
  event.preventDefault();
  setAddError("");
  if (!state.modelReady) {
    setAddError(`The active model (${state.activeModel}) is still loading.`);
    return;
  }
  if (!state.faceFiles.length) {
    setAddError("Add at least one face photo.");
    return;
  }

  const button = $("#add-person-btn");
  const originalText = button.textContent;
  state.busy = true;
  updateActionState();
  button.textContent = "Adding...";

  const form = new FormData();
  form.append("name", $("#person-name").value.trim());
  form.append("model", state.activeModel);
  for (const file of state.faceFiles) form.append("images", file, file.name);

  try {
    const data = await api("/api/people", { method: "POST", body: form });
    state.people = data.state.people;
    state.faceFiles = [];
    $("#person-name").value = "";
    renderFaceFiles();
    renderPeople();
    await loadModels();
    setAddError("");
    toast(`Student #${data.person.number} added (${data.person.name})`, "ok");
  } catch (error) {
    setAddError(error.message);
    toast(error.message, "error");
  } finally {
    state.busy = false;
    button.textContent = originalText;
    updateActionState();
  }
}

async function uploadPersonPhotos(personId, files) {
  const form = new FormData();
  form.append("model", state.activeModel);
  for (const file of files) form.append("images", file, file.name);
  state.busy = true;
  updateActionState();
  try {
    const data = await api(`/api/people/${personId}/photos`, {
      method: "POST",
      body: form,
    });
    state.people = data.state.people;
    renderPeople();
    await loadModels();
    toast(
      `Added ${plural(files.length, "photo")} to ${data.person.name} (${state.activeModel})`,
      "ok",
    );
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.busy = false;
    updateActionState();
  }
}

async function handlePeopleClick(event) {
  const actionElement = event.target.closest("[data-action]");
  if (!actionElement) return;
  const card = actionElement.closest(".person");
  if (!card) return;
  const personId = Number(card.dataset.id);
  const action = actionElement.dataset.action;

  if (action === "add-photos") {
    card.querySelector(".photo-input").click();
    return;
  }

  if (action === "remove-photo") {
    const photoId = actionElement.dataset.photoId;
    try {
      const data = await api(`/api/people/${personId}/photos/${photoId}`, {
        method: "DELETE",
      });
      state.people = data.state.people;
      renderPeople();
      await loadModels();
      toast("Photo removed", "ok");
    } catch (error) {
      toast(error.message, "error");
    }
    return;
  }

  if (action === "remove-person") {
    const person = state.people.find((item) => item.id === personId);
    if (!person) return;
    if (!window.confirm(`Remove ${person.name} and all their face photos?`)) return;
    try {
      const data = await api(`/api/people/${personId}`, { method: "DELETE" });
      state.people = data.state.people;
      renderPeople();
      await loadModels();
      toast(`${person.name} removed`, "ok");
    } catch (error) {
      toast(error.message, "error");
    }
  }
}

async function handlePeopleChange(event) {
  if (!event.target.classList.contains("photo-input")) return;
  const card = event.target.closest(".person");
  if (!card) return;
  const personId = Number(card.dataset.id);
  const files = [...event.target.files];
  event.target.value = "";
  if (!files.length) return;
  await uploadPersonPhotos(personId, files);
}

async function handleRename(event) {
  const nameElement = event.target.closest(".person-name");
  if (!nameElement) return;
  const card = nameElement.closest(".person");
  if (!card) return;
  const personId = Number(card.dataset.id);
  const person = state.people.find((item) => item.id === personId);
  if (!person) return;
  const next = window.prompt("Rename student", person.name);
  if (next === null) return;
  const clean = next.trim();
  if (!clean || clean === person.name) return;
  try {
    const data = await api(`/api/people/${personId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: clean }),
    });
    state.people = data.state.people;
    renderPeople();
    toast("Student renamed", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

function summaryCard(label, value, sub, kind = "") {
  return `
    <div class="summary-card ${kind}">
      <div class="label">${esc(label)}</div>
      <div class="value">${esc(value)}</div>
      <div class="sub">${esc(sub)}</div>
    </div>`;
}

function renderResults() {
  const scan = state.scan;
  if (!scan) return;
  const summary = scan.summary;

  $("#results").hidden = false;
  $("#results-meta").textContent = [
    scan.model || summary.model || "model",
    plural(summary.frames, "frame"),
    `threshold ${Number(summary.threshold).toFixed(2)}`,
    `detector ${summary.detSize}px`,
    `${(summary.elapsedMs / 1000).toFixed(2)}s`,
    scan.createdAt ? formatWhen(scan.createdAt) : "",
  ]
    .filter(Boolean)
    .join(" · ");

  $("#summary-grid").innerHTML = [
    summaryCard(
      "Faces detected",
      summary.facesDetected,
      `max in one frame · ${plural(summary.templateCount, "template")}`,
    ),
    summaryCard(
      "Matched",
      summary.matchedCount,
      `of ${plural(summary.peopleCount, "student")}`,
      summary.matchedCount ? "good" : "",
    ),
    summaryCard("Unknown", summary.unknown, "in the busiest frame", summary.unknown ? "bad" : ""),
    summaryCard(
      "Frames",
      summary.frames,
      `${summary.uploadedFrames} uploaded · ${plural(summary.templateCount, "template")} compared`,
    ),
  ].join("");

  renderMatches(scan);
  renderFrameTabs(scan);
  renderSelectedFrame();

  const warnings = [...(scan.warnings || [])];
  if (!scan.people.length) {
    warnings.unshift(
      `No students are enrolled for ${scan.model || summary.model}. Add students with that model active, or pick a model that has templates.`,
    );
  }
  const warningElement = $("#scan-warnings");
  warningElement.hidden = warnings.length === 0;
  warningElement.textContent = warnings.join("\n");
}

function renderMatches(scan) {
  const present = scan.people
    .filter((person) => person.status === "Present")
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const absent = scan.people.filter((person) => person.status !== "Present");
  const list = $("#matches-list");
  $("#matched-count").textContent = `${present.length} present`;

  if (!scan.people.length) {
    list.innerHTML = `
      <div class="empty">
        <strong>No students enrolled for this model</strong>
        <span>Faces are detected and counted, but cannot be identified yet.</span>
      </div>`;
    $("#absent-wrap").innerHTML = "";
    return;
  }

  if (!present.length) {
    list.innerHTML = `
      <div class="empty">
        <strong>No students matched</strong>
        <span>Try lowering the threshold, adding clearer reference photos, or using a larger frame.</span>
      </div>`;
  } else {
    list.innerHTML = present
      .map(
        (person) => `
          <div class="match-row" data-person-id="${person.personId}" title="${esc(person.name)}">
            <span class="num sm">${String(person.number).padStart(2, "0")}</span>
            <div class="match-main">
              <div class="match-name">${esc(person.name)}</div>
              <div class="bar"><span style="width:${Math.round(clamp(person.confidence || 0, 0, 1) * 100)}%"></span></div>
            </div>
            <span class="score">${Number(person.confidence || 0).toFixed(3)}</span>
          </div>`,
      )
      .join("");
  }

  for (const row of $$(".match-row", list)) {
    row.addEventListener("mouseenter", () => setHighlight(row.dataset.personId));
    row.addEventListener("mouseleave", () => setHighlight(null));
  }

  if (absent.length) {
    $("#absent-wrap").innerHTML = `
      <details class="absent-block">
        <summary><span>Not detected</span><span>${absent.length}</span></summary>
        <div class="absent-list">
          ${absent
            .map(
              (person) => `
                <div class="absent-row">
                  <span class="num">${String(person.number).padStart(2, "0")}</span>
                  <span>${esc(person.name)}</span>
                  <span>absent</span>
                </div>`,
            )
            .join("")}
        </div>
      </details>`;
  } else {
    $("#absent-wrap").innerHTML = "";
  }
}

function renderFrameTabs(scan) {
  state.selectedFrame = clamp(
    state.selectedFrame,
    0,
    Math.max(0, scan.frames.length - 1),
  );
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
      renderSelectedFrame();
      $$(".frame-tab", tabs).forEach((tab, tabIndex) => {
        tab.classList.toggle("active", tabIndex === index);
      });
    });
    tabs.append(button);
  });
}

function renderSelectedFrame() {
  const scan = state.scan;
  if (!scan || !scan.frames.length) return;
  const index = clamp(state.selectedFrame, 0, scan.frames.length - 1);
  state.selectedFrame = index;
  const frame = scan.frames[index];
  const view = $("#frame-view");
  view.innerHTML = "";
  delete view.dataset.highlight;

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
  view.append(media);

  for (const face of frame.faces) {
    const [x1, y1, x2, y2] = face.box;
    const left = clamp((x1 / frame.width) * 100, 0, 100);
    const top = clamp((y1 / frame.height) * 100, 0, 100);
    const width = clamp(((x2 - x1) / frame.width) * 100, 0, 100 - left);
    const height = clamp(((y2 - y1) / frame.height) * 100, 0, 100 - top);
    if (width <= 0 || height <= 0) continue;

    const box = document.createElement("div");
    box.className = `face-box ${face.matched ? "matched" : "unknown"}`;
    box.style.left = `${left}%`;
    box.style.top = `${top}%`;
    box.style.width = `${width}%`;
    box.style.height = `${height}%`;
    box.dataset.person = face.personId == null ? "" : String(face.personId);
    box.title = face.matched
      ? `${face.name} · confidence ${Number(face.confidence).toFixed(3)} · detector ${Number(face.detScore).toFixed(3)}`
      : `Unknown · best similarity ${Number(face.confidence).toFixed(3)} · detector ${Number(face.detScore).toFixed(3)}`;

    const tooSmallForLabel = !face.matched && (width < 8 || height < 8);
    if (tooSmallForLabel) box.classList.add("compact");

    const label = document.createElement("span");
    label.className = "box-label";
    label.textContent = face.matched
      ? `${face.name} · ${Number(face.confidence).toFixed(2)}`
      : `Unknown · ${Number(face.confidence).toFixed(2)}`;
    if (top < 7) label.classList.add("inside");
    if (left > 62) label.classList.add("right");
    box.append(label);
    layer.append(box);
  }

  $("#frame-meta").textContent = `${frame.sourceName || `frame ${index + 1}`} · ${frame.width}×${frame.height} · ${plural(frame.faces.length, "face")}`;
  $("#annotated-link").href = frame.annotatedUrl || "#";
}

function setHighlight(personId) {
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

async function runScan() {
  setScanError("");
  $("#scan-warnings").hidden = true;
  if (!state.modelReady) {
    setScanError(`The active model (${state.activeModel}) is still loading.`);
    return;
  }
  if (!state.frameFiles.length) {
    setScanError("Add at least one classroom frame.");
    return;
  }

  const button = $("#scan-btn");
  const originalText = button.textContent;
  const status = $("#scan-status");
  const statusText = $("#scan-status-text");
  const startedAt = performance.now();
  state.busy = true;
  updateActionState();
  button.textContent = "Scanning...";
  status.hidden = false;
  statusText.textContent = `Analyzing ${plural(state.frameFiles.length, "frame")} at ${$("#det-size").value}px...`;

  const timer = window.setInterval(() => {
    const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    statusText.textContent = `Analyzing ${plural(state.frameFiles.length, "frame")} at ${$("#det-size").value}px · ${seconds}s`;
  }, 120);

  const form = new FormData();
  form.append("model", state.activeModel);
  for (const file of state.frameFiles) form.append("images", file, file.name);
  form.append("threshold", $("#threshold").value);
  form.append("detSize", $("#det-size").value);

  try {
    const result = await api("/api/scan", { method: "POST", body: form });
    state.scan = result;
    state.selectedFrame =
      result.summary.bestIndex >= 0 ? result.summary.bestIndex : 0;
    renderResults();
    await loadResults();
    renderHistory();
    toast(
      `Scanned ${plural(result.summary.frames, "frame")} with ${result.model} in ${(result.summary.elapsedMs / 1000).toFixed(1)}s`,
      "ok",
    );
  } catch (error) {
    setScanError(error.message);
    toast(error.message, "error");
  } finally {
    window.clearInterval(timer);
    status.hidden = true;
    button.textContent = originalText;
    state.busy = false;
    updateActionState();
  }
}

async function loadResult(scanId) {
  try {
    const data = await api(`/api/results/${encodeURIComponent(scanId)}`);
    state.scan = data;
    state.selectedFrame =
      Number.isInteger(data.summary?.bestIndex) && data.summary.bestIndex >= 0
        ? data.summary.bestIndex
        : 0;
    renderResults();
    renderHistory();
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
    toast(`Loaded saved run ${scanId}`, "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function deleteResult(scanId) {
  if (!window.confirm(`Delete saved run ${scanId}? The annotated frames are removed too.`)) {
    return;
  }
  try {
    const data = await api(`/api/results/${encodeURIComponent(scanId)}`, {
      method: "DELETE",
    });
    state.results = data.results || [];
    if (state.scan && state.scan.scanId === scanId) {
      state.scan = null;
      $("#results").hidden = true;
      $("#scan-warnings").hidden = true;
    }
    renderHistory();
    toast("Saved run deleted", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function handleHistoryClick(event) {
  const row = event.target.closest(".history-row");
  if (!row) return;
  const scanId = row.dataset.scan;
  const actionElement = event.target.closest("[data-action]");
  const action = actionElement?.dataset.action;

  if (action === "delete-result") {
    await deleteResult(scanId);
    return;
  }
  if (action === "load-result" || !event.target.closest("a")) {
    await loadResult(scanId);
  }
}

async function resetPlayground() {
  if (
    !window.confirm(
      "Clear all students, face templates, saved runs, and their frames? Model packs are kept.",
    )
  ) {
    return;
  }
  try {
    const data = await api("/api/reset", { method: "POST" });
    state.people = data.state.people;
    state.faceFiles = [];
    state.frameFiles = [];
    state.scan = null;
    state.selectedFrame = 0;
    state.results = [];
    $("#person-name").value = "";
    $("#results").hidden = true;
    $("#scan-warnings").hidden = true;
    setAddError("");
    setScanError("");
    renderFaceFiles();
    renderFrameFiles();
    renderPeople();
    renderHistory();
    await loadModels();
    toast("Reset: roster, results, and frames cleared", "ok");
  } catch (error) {
    toast(error.message, "error");
  }
}

function wireEvents() {
  $("#add-person-form").addEventListener("submit", addPerson);
  $("#face-input").addEventListener("change", (event) => {
    addFaceFiles(event.target.files);
    event.target.value = "";
  });
  $("#frame-input").addEventListener("change", (event) => {
    addFrameFiles(event.target.files);
    event.target.value = "";
  });
  $("#threshold").addEventListener("input", (event) => {
    $("#threshold-value").textContent = Number(event.target.value).toFixed(2);
  });
  $("#model-select").addEventListener("change", (event) => {
    selectModel(event.target.value);
  });
  $("#model-usage-list").addEventListener("click", (event) => {
    const row = event.target.closest(".model-row");
    if (!row) return;
    if (!event.target.closest("[data-action]")) return;
    selectModel(row.dataset.model);
  });
  $("#scan-btn").addEventListener("click", runScan);
  $("#reset-btn").addEventListener("click", resetPlayground);
  $("#people-list").addEventListener("click", handlePeopleClick);
  $("#people-list").addEventListener("change", handlePeopleChange);
  $("#people-list").addEventListener("dblclick", handleRename);
  $("#history-list").addEventListener("click", handleHistoryClick);
  wireDropzone($("#face-drop"), addFaceFiles);
  wireDropzone($("#frame-drop"), addFrameFiles);
}

async function init() {
  wireEvents();
  renderPeople();
  renderFaceFiles();
  renderFrameFiles();
  renderHistory();
  updateActionState();

  await loadState();
  await loadModels();
  await loadResults();

  const remembered = readRememberedModel();
  const installed = state.models.filter((model) => model.installed).map((model) => model.name);
  const initial =
    (remembered && state.models.some((model) => model.name === remembered) && remembered) ||
    (state.settings.defaultModel && state.models.some((m) => m.name === state.settings.defaultModel)
      ? state.settings.defaultModel
      : null) ||
    installed[0] ||
    state.models[0]?.name ||
    state.activeModel;

  selectModel(initial);
}

document.addEventListener("DOMContentLoaded", init);
