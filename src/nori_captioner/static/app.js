const state = {
  page: 1,
  perPage: 50,
  total: 0,
  filter: "all",
  status: null,
  previousStatus: null,
  saveTimers: new Map(),
  cards: new Map(),
  loadRequestId: 0,
};

const grid = document.getElementById("grid");
const template = document.getElementById("card-template");
const emptyState = document.getElementById("empty-state");
const pagers = [
  document.getElementById("pager-top"),
  document.getElementById("pager-bottom"),
];
const pageLabels = [
  document.getElementById("page-label-top"),
  document.getElementById("page-label"),
];
const uploadInput = document.getElementById("upload-input");
const uploadStatus = document.getElementById("upload-status");
const dropZone = document.getElementById("drop-zone");
const userPromptInput = document.getElementById("user-prompt");
const saveUserPromptBtn = document.getElementById("save-user-prompt");
const settingsStatus = document.getElementById("settings-status");

function updatePagerUi(total) {
  const label = `Page ${state.page} (${total})`;
  for (const pageLabel of pageLabels) {
    pageLabel.textContent = label;
  }
}

async function goToPreviousPage() {
  if (state.page > 1) {
    state.page -= 1;
    await loadFiles();
  }
}

async function goToNextPage() {
  const maxPage = Math.max(1, Math.ceil(state.total / state.perPage));
  if (state.page < maxPage) {
    state.page += 1;
    await loadFiles();
  }
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(path, {
    headers,
    ...options,
  });
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || `HTTP ${res.status}`);
  }
  return res.json();
}

function applyCaptionState(card, text) {
  const hasCaption = text.trim().length > 0;
  card.dataset.state = hasCaption ? "captioned" : "idle";
}

function setStatusLine(status) {
  document.getElementById("model").textContent = `Model: ${status.model_id || "none"}`;
  document.getElementById("queue").textContent = `Queue: ${status.queue_len}`;
  document.getElementById("done").textContent = `Done: ${status.done}`;
  document.getElementById("errors").textContent = `Errors: ${status.errors}`;
}

function formatFps(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }
  return Number.isInteger(value) ? `${value} fps` : `${value.toFixed(2)} fps`;
}

function formatDuration(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }

  const totalSeconds = Math.max(0, Math.round(value));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function buildMetaStatsText(item) {
  const parts = [];

  if (item.width && item.height) {
    parts.push(`${item.width}x${item.height}px`);
  }

  if (item.media_type === "video") {
    const durationText = formatDuration(item.duration_seconds);
    if (durationText) {
      parts.push(durationText);
    }
    if (item.frame_count !== null && item.frame_count !== undefined) {
      parts.push(`${item.frame_count} frames`);
    }
    const fpsText = formatFps(item.fps);
    if (fpsText) {
      parts.push(fpsText);
    }
  }

  return parts.join(" | ");
}

function setMetaLines(card, item) {
  const meta = card.querySelector(".meta");
  meta.innerHTML = "";

  const pathLine = document.createElement("div");
  pathLine.className = "meta-path";
  pathLine.textContent = item.rel_path;
  meta.appendChild(pathLine);

  const statsText = buildMetaStatsText(item);
  if (statsText) {
    const statsLine = document.createElement("div");
    statsLine.className = "meta-stats";
    statsLine.textContent = statsText;
    meta.appendChild(statsLine);
  }
}

function makeMediaEl(item) {
  const wrap = document.createElement("div");
  wrap.className = "media-wrap";

  if (item.media_type === "video") {
    const el = document.createElement("video");
    el.controls = true;
    el.src = `/media/${item.id}`;
    el.preload = "metadata";
    wrap.appendChild(el);
  } else {
    const el = document.createElement("img");
    el.src = `/media/${item.id}`;
    el.loading = "lazy";
    wrap.appendChild(el);
  }

  return wrap;
}

async function loadCaption(fileId) {
  const data = await api(`/api/file/${fileId}/caption`);
  return data.text || "";
}

function scheduleSave(fileId, textarea, stateEl) {
  if (state.saveTimers.has(fileId)) {
    clearTimeout(state.saveTimers.get(fileId));
  }
  const timer = setTimeout(async () => {
    stateEl.textContent = "saving...";
    try {
      await api(`/api/file/${fileId}/caption`, {
        method: "PUT",
        body: JSON.stringify({ text: textarea.value }),
      });
      stateEl.textContent = "saved";
      applyCaptionState(stateEl.closest(".card"), textarea.value);
      await loadStatus();
    } catch (err) {
      stateEl.textContent = `save failed`;
      console.error(err);
    }
  }, 800);
  state.saveTimers.set(fileId, timer);
}

function setCardState(card, item) {
  card.dataset.state = item.state || (item.has_caption ? "captioned" : "idle");
}

async function refreshCompletedCaptions(nextStatus) {
  const previous = state.previousStatus;
  state.previousStatus = nextStatus;
  if (!previous) {
    return;
  }

  const hadRunning = previous.running_id;
  const hasRunning = nextStatus.running_id;
  if (hadRunning === null || hadRunning === hasRunning) {
    return;
  }

  const cardRef = state.cards.get(hadRunning);
  if (!cardRef) {
    await loadFiles();
    return;
  }

  try {
    const text = await loadCaption(hadRunning);
    cardRef.textarea.value = text;
    cardRef.stateEl.textContent = "captioned";
    cardRef.card.dataset.state = "captioned";
  } catch (err) {
    console.error(err);
  }
}

async function renderItems(items, requestId) {
  if (requestId !== state.loadRequestId) {
    return;
  }

  grid.innerHTML = "";
  state.cards.clear();

  if (items.length === 0) {
    emptyState.hidden = false;
    for (const pager of pagers) {
      pager.hidden = true;
    }
    return;
  }

  emptyState.hidden = true;
  for (const pager of pagers) {
    pager.hidden = false;
  }
  for (const item of items) {
    if (requestId !== state.loadRequestId) {
      return;
    }

    const card = template.content.firstElementChild.cloneNode(true);
    setMetaLines(card, item);
    card.querySelector(".media-wrap")?.remove();
    card.prepend(makeMediaEl(item));

    const textarea = card.querySelector("textarea");
    const saveBtn = card.querySelector(".save");
    const autoBtn = card.querySelector(".auto");
    const deleteBtn = card.querySelector(".delete");
    const stateEl = card.querySelector(".state");

    const captionText = await loadCaption(item.id);
    if (requestId !== state.loadRequestId) {
      return;
    }

    textarea.value = captionText;
    setCardState(card, item);
    stateEl.textContent = item.state;
  state.cards.set(item.id, { card, textarea, stateEl });

    textarea.addEventListener("input", () => scheduleSave(item.id, textarea, stateEl));
    saveBtn.addEventListener("click", async () => {
      stateEl.textContent = "saving...";
      try {
        await api(`/api/file/${item.id}/caption`, {
          method: "PUT",
          body: JSON.stringify({ text: textarea.value }),
        });
        stateEl.textContent = "saved";
        applyCaptionState(card, textarea.value);
        await loadStatus();
      } catch (err) {
        stateEl.textContent = "save failed";
        alert(`Save failed: ${err.message}`);
      }
    });

    autoBtn.addEventListener("click", async () => {
      autoBtn.disabled = true;
      stateEl.textContent = "queued";
      card.dataset.state = "queued";
      try {
        await api(`/api/autocaption/${item.id}`, { method: "POST" });
        // Update global counters without re-rendering the whole grid.
        await loadStatus();
      } catch (err) {
        stateEl.textContent = "error";
        card.dataset.state = "error";
        alert(`Auto-caption failed: ${err.message}`);
      } finally {
        autoBtn.disabled = false;
      }
    });

    deleteBtn.addEventListener("click", async () => {
      const confirmed = window.confirm(`Delete ${item.rel_path}? This also removes its .txt caption.`);
      if (!confirmed) {
        return;
      }

      deleteBtn.disabled = true;
      try {
        await api(`/api/file/${item.id}`, { method: "DELETE" });
        await refresh();
      } catch (err) {
        alert(`Delete failed: ${err.message}`);
      } finally {
        deleteBtn.disabled = false;
      }
    });

    grid.appendChild(card);
  }
}

async function loadFiles() {
  const requestId = ++state.loadRequestId;
  const data = await api(`/api/files?page=${state.page}&per_page=${state.perPage}&filter=${state.filter}`);

  if (requestId !== state.loadRequestId) {
    return;
  }

  state.total = data.total;
  updatePagerUi(data.total);
  await renderItems(data.items, requestId);
}

async function loadStatus() {
  const nextStatus = await api("/api/status");
  state.status = nextStatus;
  setStatusLine(nextStatus);
  await refreshCompletedCaptions(nextStatus);
}

async function refresh() {
  try {
    await loadStatus();
    await loadFiles();
  } catch (err) {
    console.error(err);
  }
}

async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    userPromptInput.value = settings.prompt || "";
    settingsStatus.textContent = "";
  } catch (err) {
    settingsStatus.textContent = "Failed to load settings";
    console.error(err);
  }
}

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) {
    uploadStatus.textContent = "Select files first";
    return;
  }

  const form = new FormData();
  for (const file of fileList) {
    form.append("files", file);
  }

  uploadInput.disabled = true;
  uploadStatus.textContent = "Uploading...";
  try {
    const result = await api("/api/upload", {
      method: "POST",
      body: form,
    });
    const skippedCount = result.skipped?.length || 0;
    uploadStatus.textContent = `Uploaded ${result.uploaded_count}, skipped ${skippedCount}`;
    uploadInput.value = "";
    state.page = 1;
    await refresh();
  } catch (err) {
    uploadStatus.textContent = "Upload failed";
    alert(`Upload failed: ${err.message}`);
  } finally {
    uploadInput.disabled = false;
  }
}

document.getElementById("filter").addEventListener("change", async (e) => {
  state.filter = e.target.value;
  state.page = 1;
  await loadFiles();
});

const captionAllBtn = document.getElementById("caption-all");

captionAllBtn.addEventListener("click", async () => {
  captionAllBtn.disabled = true;

  for (const cardRef of state.cards.values()) {
    if (cardRef.card.dataset.state === "idle") {
      cardRef.card.dataset.state = "queued";
      cardRef.stateEl.textContent = "queued";
    }
  }

  try {
    await api("/api/autocaption/batch", {
      method: "POST",
      body: JSON.stringify({ filter: "uncaptioned" }),
    });
    // Keep current DOM stable: update counters without full list re-render.
    await loadStatus();
  } catch (err) {
    alert(`Batch queue failed: ${err.message}`);
  } finally {
    captionAllBtn.disabled = false;
  }
});

document.getElementById("prev").addEventListener("click", goToPreviousPage);
document.getElementById("prev-top").addEventListener("click", goToPreviousPage);
document.getElementById("next").addEventListener("click", goToNextPage);
document.getElementById("next-top").addEventListener("click", goToNextPage);

uploadInput.addEventListener("change", async () => {
  await uploadFiles(uploadInput.files);
});

dropZone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dropZone.classList.add("active");
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("active");
});

dropZone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dropZone.classList.remove("active");
});

dropZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropZone.classList.remove("active");
  const files = event.dataTransfer?.files;
  await uploadFiles(files);
});

saveUserPromptBtn.addEventListener("click", async () => {
  saveUserPromptBtn.disabled = true;
  settingsStatus.textContent = "Saving user prompt...";
  try {
    await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify({
        prompt: userPromptInput.value,
      }),
    });
    settingsStatus.textContent = "User prompt saved";
  } catch (err) {
    settingsStatus.textContent = "Failed to save user prompt";
    alert(`Failed to save user prompt: ${err.message}`);
  } finally {
    saveUserPromptBtn.disabled = false;
  }
});

loadSettings();
refresh();
setInterval(loadStatus, 2000);
