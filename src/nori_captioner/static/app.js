const state = {
  page: 1,
  perPage: 50,
  total: 0,
  filter: "all",
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
const userPromptInput = document.getElementById("user-prompt");
const saveUserPromptBtn = document.getElementById("save-user-prompt");
const settingsStatus = document.getElementById("settings-status");
const modelStatusEl = document.getElementById("model");
const queueStatusEl = document.getElementById("queue");
const doneStatusEl = document.getElementById("done");
const errorsStatusEl = document.getElementById("errors");
let dragDepth = 0;

function ensureDragOverlay() {
  const existing = document.getElementById("drag-overlay");
  if (existing) {
    return existing;
  }

  const overlay = document.createElement("div");
  overlay.id = "drag-overlay";
  overlay.className = "drag-overlay";
  overlay.setAttribute("aria-hidden", "true");
  overlay.hidden = true;

  const inner = document.createElement("div");
  inner.className = "drag-overlay-inner";
  inner.textContent = "Drop files to upload";
  overlay.appendChild(inner);

  document.body.prepend(overlay);
  return overlay;
}

const dragOverlay = ensureDragOverlay();

function isFileDrag(event) {
  const transfer = event.dataTransfer;
  if (!transfer) {
    return false;
  }

  const types = transfer.types;
  if (!types) {
    return Boolean(transfer.files?.length);
  }

  const hasType = (value) => {
    if (typeof types.includes === "function" && types.includes(value)) {
      return true;
    }
    if (typeof types.contains === "function" && types.contains(value)) {
      return true;
    }
    return Array.from(types).includes(value);
  };

  return hasType("Files") || hasType("application/x-moz-file") || Boolean(transfer.files?.length);
}

function setDropActive(active) {
  dragOverlay.hidden = !active;
}

function hasCaptionText(text) {
  return text.trim().length > 0;
}

function updatePagerUi(total) {
  const label = `Page ${state.page} (${total})`;
  for (const pageLabel of pageLabels) {
    pageLabel.textContent = label;
  }
}

function syncGridVisibility() {
  const hasCards = grid.children.length > 0;
  emptyState.hidden = hasCards;
  for (const pager of pagers) {
    pager.hidden = !hasCards;
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
  if (card.dataset.captionLocked === "true") {
    card.dataset.state = "locked";
    return;
  }
  card.dataset.state = hasCaptionText(text) ? "captioned" : "idle";
}

function setStatusLine(status) {
  modelStatusEl.textContent = `Model: ${status.model_id || "none"}`;
  queueStatusEl.textContent = `Queue: ${status.queue_len}`;
  doneStatusEl.textContent = `Done: ${status.done}`;
  errorsStatusEl.textContent = `Errors: ${status.errors}`;
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

  if (item.caption_locked) {
    parts.push("caption locked");
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

function setCaptionLockControls(button, textarea, saveBtn, autoBtn, item) {
  button.textContent = item.caption_locked ? "Unlock Caption" : "Lock Caption";
  button.classList.toggle("is-locked", item.caption_locked);
  const card = button.closest(".card");
  if (card) {
    card.dataset.captionLocked = String(item.caption_locked);
    if (card.dataset.state !== "queued" && card.dataset.state !== "running") {
      card.dataset.state = item.caption_locked
        ? "locked"
        : hasCaptionText(textarea.value)
          ? "captioned"
          : "idle";
    }
  }
  textarea.disabled = item.caption_locked;
  saveBtn.disabled = item.caption_locked;
  autoBtn.disabled = item.caption_locked;
}

async function toggleCaptionLock(fileId) {
  const cardRef = state.cards.get(fileId);
  if (!cardRef) {
    return;
  }

  const { item, textarea, saveBtn, autoBtn, captionLockBtn, card } = cardRef;
  captionLockBtn.disabled = true;
  try {
    const result = await api(`/api/file/${fileId}/caption-lock`, {
      method: "PUT",
      body: JSON.stringify({ locked: !item.caption_locked }),
    });
    item.caption_locked = result.caption_locked;
    setCaptionLockControls(captionLockBtn, textarea, saveBtn, autoBtn, item);
    setMetaLines(card, item);
  } catch (err) {
    alert(`Caption lock update failed: ${err.message}`);
  } finally {
    captionLockBtn.disabled = false;
  }
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

async function saveCaption(fileId, text) {
  await api(`/api/file/${fileId}/caption`, {
    method: "PUT",
    body: JSON.stringify({ text }),
  });
}

function scheduleSave(fileId, textarea, stateEl) {
  // Clear stale state while the user is actively editing.
  stateEl.textContent = "";
  const card = stateEl.closest(".card");
  if (card && card.dataset.state !== "queued" && card.dataset.state !== "running") {
    card.dataset.state = "editing";
  }

  if (state.saveTimers.has(fileId)) {
    clearTimeout(state.saveTimers.get(fileId));
  }
  const timer = setTimeout(async () => {
    stateEl.textContent = "saving...";
    try {
      await saveCaption(fileId, textarea.value);
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
  card.dataset.captionLocked = String(Boolean(item.caption_locked));
  card.dataset.state = item.state || (item.caption_locked ? "locked" : item.has_caption ? "captioned" : "idle");
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
    // Keep viewport stable: avoid full re-render when completed item is off-page.
    return;
  }

  try {
    const text = await loadCaption(hadRunning);
    cardRef.textarea.value = text;
    cardRef.stateEl.textContent = cardRef.item.caption_locked ? "locked" : "captioned";
    cardRef.card.dataset.state = cardRef.item.caption_locked ? "locked" : "captioned";
  } catch (err) {
    console.error(err);
  }
}

async function renderItems(items, requestId) {
  if (requestId !== state.loadRequestId) {
    return;
  }

  if (items.length === 0) {
    grid.replaceChildren();
    state.cards.clear();
    syncGridVisibility();
    return;
  }

  const captions = await Promise.all(
    items.map(async (item) => {
      try {
        return await loadCaption(item.id);
      } catch (err) {
        console.error(err);
        return "";
      }
    })
  );

  if (requestId !== state.loadRequestId) {
    return;
  }

  const previousHeight = grid.getBoundingClientRect().height;
  if (previousHeight > 0) {
    grid.style.minHeight = `${Math.round(previousHeight)}px`;
  }

  const fragment = document.createDocumentFragment();
  const nextCards = new Map();

  for (const [idx, item] of items.entries()) {
    const card = template.content.firstElementChild.cloneNode(true);
    card.dataset.fileId = String(item.id);
    setMetaLines(card, item);
    card.querySelector(".media-wrap")?.remove();
    card.prepend(makeMediaEl(item));

    const textarea = card.querySelector("textarea");
    const captionLockBtn = card.querySelector(".caption-lock");
    const saveBtn = card.querySelector(".save");
    const autoBtn = card.querySelector(".auto");
    const deleteBtn = card.querySelector(".delete");
    const stateEl = card.querySelector(".state");

    textarea.value = captions[idx] || "";
    setCardState(card, item);
    stateEl.textContent = item.state;
    setCaptionLockControls(captionLockBtn, textarea, saveBtn, autoBtn, item);
    nextCards.set(item.id, {
      item,
      card,
      textarea,
      stateEl,
      saveBtn,
      autoBtn,
      captionLockBtn,
    });

    textarea.addEventListener("input", () => scheduleSave(item.id, textarea, stateEl));
    saveBtn.addEventListener("click", async () => {
      stateEl.textContent = "saving...";
      try {
        await saveCaption(item.id, textarea.value);
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

        state.cards.delete(item.id);
        card.remove();
        state.total = Math.max(0, state.total - 1);
        updatePagerUi(state.total);
        syncGridVisibility();

        if (grid.children.length === 0 && state.page > 1) {
          state.page -= 1;
          await loadFiles();
        }

        await loadStatus();
      } catch (err) {
        alert(`Delete failed: ${err.message}`);
      } finally {
        deleteBtn.disabled = false;
      }
    });

    fragment.appendChild(card);
  }

  if (requestId !== state.loadRequestId) {
    return;
  }

  grid.replaceChildren(fragment);
  state.cards.clear();
  for (const [id, ref] of nextCards) {
    state.cards.set(id, ref);
  }
  syncGridVisibility();
  requestAnimationFrame(() => {
    grid.style.minHeight = "";
  });
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

async function loadFilesPreserveScroll() {
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  await loadFiles();
  window.scrollTo(scrollX, scrollY);
}

async function loadStatus() {
  const nextStatus = await api("/api/status");
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
    const previousTotal = state.total;
    const previousMaxPage = Math.max(1, Math.ceil(previousTotal / state.perPage));
    const isOnLastPageBeforeUpload = state.page === previousMaxPage;
    const hadNoVisibleCards = grid.children.length === 0;

    const result = await api("/api/upload", {
      method: "POST",
      body: form,
    });
    const uploadedCount = result.uploaded_count || 0;
    const skippedCount = result.skipped?.length || 0;
    uploadStatus.textContent = `Uploaded ${uploadedCount}, skipped ${skippedCount}`;
    uploadInput.value = "";

    // Keep current cards stable. Newly uploaded files show on next paging/filter/reload.
    if (state.filter === "all" || state.filter === "uncaptioned" || state.filter === "unlocked") {
      state.total += uploadedCount;
      updatePagerUi(state.total);
      syncGridVisibility();

      // Refresh only when the visible page can change due to new uploads.
      if (uploadedCount > 0 && (hadNoVisibleCards || isOnLastPageBeforeUpload)) {
        await loadFilesPreserveScroll();
      }
    }
    await loadStatus();
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

const captionAllUncaptionedBtn = document.getElementById("caption-all-uncaptioned");
const captionAllUnlockedBtn = document.getElementById("caption-all-unlocked");

async function queueBatch(filterMode, button) {
  button.disabled = true;

  for (const cardRef of state.cards.values()) {
    if (cardRef.item.caption_locked) {
      continue;
    }
    if (filterMode === "uncaptioned" && hasCaptionText(cardRef.textarea.value)) {
      continue;
    }
    if (cardRef.card.dataset.state !== "queued" && cardRef.card.dataset.state !== "running") {
      cardRef.card.dataset.state = "queued";
      cardRef.stateEl.textContent = "queued";
    }
  }

  try {
    await api("/api/autocaption/batch", {
      method: "POST",
      body: JSON.stringify({ filter: filterMode }),
    });
    // Keep current DOM stable: update counters without full list re-render.
    await loadStatus();
  } catch (err) {
    alert(`Batch queue failed: ${err.message}`);
  } finally {
    button.disabled = false;
  }
}

captionAllUncaptionedBtn.addEventListener("click", async () => {
  await queueBatch("uncaptioned", captionAllUncaptionedBtn);
});

captionAllUnlockedBtn.addEventListener("click", async () => {
  await queueBatch("unlocked", captionAllUnlockedBtn);
});

document.getElementById("prev").addEventListener("click", goToPreviousPage);
document.getElementById("prev-top").addEventListener("click", goToPreviousPage);
document.getElementById("next").addEventListener("click", goToNextPage);
document.getElementById("next-top").addEventListener("click", goToNextPage);

uploadInput.addEventListener("change", async () => {
  await uploadFiles(uploadInput.files);
});

grid.addEventListener("click", async (event) => {
  const button = event.target.closest(".caption-lock");
  if (!button) {
    return;
  }

  event.preventDefault();
  event.stopPropagation();
  const card = button.closest(".card");
  const fileId = Number(card?.dataset.fileId);
  if (!Number.isInteger(fileId)) {
    return;
  }

  await toggleCaptionLock(fileId);
});

window.addEventListener("dragenter", (event) => {
  if (!isFileDrag(event)) {
    return;
  }
  event.preventDefault();
  dragDepth += 1;
  setDropActive(true);
});

window.addEventListener("dragover", (event) => {
  if (!isFileDrag(event)) {
    return;
  }
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
  }
  setDropActive(true);
});

window.addEventListener("dragleave", (event) => {
  if (!isFileDrag(event)) {
    return;
  }
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    setDropActive(false);
  }
});

window.addEventListener("drop", async (event) => {
  if (!isFileDrag(event)) {
    return;
  }
  event.preventDefault();
  dragDepth = 0;
  setDropActive(false);
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
