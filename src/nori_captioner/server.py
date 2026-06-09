from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .captioner import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, CaptionerService
from .scanner import (
    MediaFile,
    caption_sidecar_path,
    has_nonempty_caption,
    is_supported,
    media_type_for,
    paginate,
    probe_media_metadata,
    read_caption,
    scan_directory,
    thumbnail_bytes,
)


@dataclass(slots=True)
class AppConfig:
    root: Path
    model: str | None = None
    quantize: int | None = None
    device: str = "auto"
    frames: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt: str = DEFAULT_PROMPT


class CaptionUpdate(BaseModel):
    text: str


class BatchRequest(BaseModel):
    filter: str = "uncaptioned"


class SettingsUpdate(BaseModel):
    system_prompt: str | None = None
    prompt: str | None = None


class CaptionLockUpdate(BaseModel):
    locked: bool


def _settings_file(root: Path) -> Path:
    return root / ".nori-captioner.settings.json"


def _load_saved_settings(root: Path) -> dict[str, str | None]:
    path = _settings_file(root)
    if not path.exists():
        return {"system_prompt": None, "prompt": None, "caption_locks": set()}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"system_prompt": None, "prompt": None, "caption_locks": set()}

    caption_locks = data.get("caption_locks")
    if isinstance(caption_locks, list):
        cleaned_caption_locks = {
            rel_path for rel_path in caption_locks if isinstance(rel_path, str)
        }
    elif isinstance(caption_locks, dict):
        cleaned_caption_locks = {
            str(rel_path)
            for rel_path, locked in caption_locks.items()
            if isinstance(rel_path, str) and bool(locked)
        }
    else:
        cleaned_caption_locks = set()

    return {
        "system_prompt": data.get("system_prompt"),
        "prompt": data.get("prompt"),
        "caption_locks": cleaned_caption_locks,
    }


def _save_settings(
    root: Path,
    *,
    system_prompt: str,
    prompt: str,
    caption_locks: set[str],
) -> None:
    path = _settings_file(root)
    payload = {
        "system_prompt": system_prompt,
        "prompt": prompt,
        "caption_locks": sorted(caption_locks),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dedupe_destination(root: Path, requested_name: str) -> Path:
    candidate_name = Path(requested_name).name
    candidate = root / candidate_name
    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while candidate.exists() or caption_sidecar_path(candidate).exists():
        candidate = root / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def _serialize_file(item: MediaFile, status: dict) -> dict:
    queued_ids = set(status["queued_ids"])
    running_id = status["running_id"]
    caption_locked = item.rel_path in status["caption_locked_paths"]

    state = "idle"
    if item.id == running_id:
        state = "running"
    elif item.id in queued_ids:
        state = "queued"
    elif caption_locked:
        state = "locked"
    elif item.has_caption:
        state = "captioned"

    return {
        "id": item.id,
        "rel_path": item.rel_path,
        "media_type": item.media_type,
        "has_caption": item.has_caption,
        "state": state,
        "caption_locked": caption_locked,
        "width": item.width,
        "height": item.height,
        "frame_count": item.frame_count,
        "fps": item.fps,
        "duration_seconds": item.duration_seconds,
    }


def create_app(config: AppConfig) -> FastAPI:
    saved_settings = _load_saved_settings(config.root)
    if saved_settings["system_prompt"]:
        config.system_prompt = saved_settings["system_prompt"]
    if saved_settings["prompt"]:
        config.prompt = saved_settings["prompt"]

    media_files = scan_directory(config.root)
    captioner = CaptionerService(
        media_files=media_files,
        model_spec=config.model,
        system_prompt=config.system_prompt,
        prompt=config.prompt,
        caption_locks=saved_settings["caption_locks"],
        frames_per_video=config.frames,
        quantize=config.quantize,
        device=config.device,
    )

    def persist_settings() -> None:
        _save_settings(
            config.root,
            system_prompt=captioner.system_prompt,
            prompt=captioner.prompt,
            caption_locks=captioner.caption_locks,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await captioner.preload_model()
        await captioner.start()
        try:
            yield
        finally:
            await captioner.stop()

    app = FastAPI(title="nori-captioner", lifespan=lifespan)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/files")
    async def list_files(
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=50, ge=1, le=500),
        filter: str = Query(default="all"),
    ) -> JSONResponse:
        status = captioner.status() | {"caption_locked_paths": set(captioner.caption_locks)}
        queued_ids = set(status["queued_ids"])
        locked_paths = status["caption_locked_paths"]

        filtered = [m for m in media_files if m is not None]
        if filter == "captioned":
            filtered = [m for m in media_files if m is not None and m.has_caption]
        elif filter == "uncaptioned":
            filtered = [m for m in media_files if m is not None and not m.has_caption]
        elif filter == "locked":
            filtered = [m for m in media_files if m is not None and m.rel_path in locked_paths]
        elif filter == "unlocked":
            filtered = [m for m in media_files if m is not None and m.rel_path not in locked_paths]
        elif filter == "queued":
            filtered = [
                m
                for m in media_files
                if m is not None and (
                    m.id in queued_ids or m.id == status["running_id"]
                )
            ]

        page_items, total = paginate(filtered, page=page, per_page=per_page)
        return JSONResponse(
            {
                "items": [_serialize_file(item, status) for item in page_items],
                "page": page,
                "per_page": per_page,
                "total": total,
            }
        )

    @app.get("/api/file/{file_id}/caption")
    async def get_caption(file_id: int) -> JSONResponse:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")
        media = media_files[file_id]
        return JSONResponse({"text": read_caption(media.abs_path)})

    @app.put("/api/file/{file_id}/caption")
    async def put_caption(file_id: int, update: CaptionUpdate) -> JSONResponse:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")
        from .scanner import write_caption

        media = media_files[file_id]
        if captioner.is_caption_locked(media):
            raise HTTPException(status_code=409, detail="Caption is locked")
        write_caption(media.abs_path, update.text)
        media.has_caption = has_nonempty_caption(media.abs_path)
        return JSONResponse({"ok": True})

    @app.delete("/api/file/{file_id}")
    async def delete_file(file_id: int) -> JSONResponse:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")

        if captioner.running_id is not None or captioner.queue.qsize() > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot delete while caption queue is active. "
                    "Wait for completion or clear queue."
                ),
            )

        media = media_files[file_id]
        caption_path = caption_sidecar_path(media.abs_path)
        captioner.clear_caption_lock(media)

        try:
            media.abs_path.unlink(missing_ok=True)
            caption_path.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

        media_files[file_id] = None
        persist_settings()

        return JSONResponse({"ok": True})

    @app.put("/api/file/{file_id}/caption-lock")
    async def put_caption_lock(file_id: int, update: CaptionLockUpdate) -> JSONResponse:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")

        media = media_files[file_id]
        if update.locked:
            captioner.set_caption_lock(media)
        else:
            captioner.clear_caption_lock(media)

        try:
            persist_settings()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {exc}") from exc

        return JSONResponse({"ok": True, "caption_locked": captioner.is_caption_locked(media)})

    @app.post("/api/autocaption/batch")
    async def queue_batch(request: BatchRequest) -> JSONResponse:
        if captioner.model_id is None:
            raise HTTPException(status_code=400, detail="No model configured. Start with --model")
        mode = (
            request.filter
            if request.filter in {"all", "uncaptioned", "unlocked"}
            else "uncaptioned"
        )
        queued_count = captioner.enqueue_batch(mode)
        return JSONResponse({"queued_count": queued_count})

    @app.post("/api/autocaption/{file_id}")
    async def queue_one(file_id: int) -> JSONResponse:
        if captioner.model_id is None:
            raise HTTPException(status_code=400, detail="No model configured. Start with --model")
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")
        media = media_files[file_id]
        if captioner.is_caption_locked(media):
            raise HTTPException(status_code=409, detail="Caption is locked")
        queued = captioner.enqueue(file_id)
        if not queued:
            raise HTTPException(status_code=400, detail="File already queued/running or invalid")
        return JSONResponse({"queued": True})

    @app.delete("/api/autocaption/queue")
    async def clear_queue() -> JSONResponse:
        removed = captioner.clear_queue()
        return JSONResponse({"removed": removed})

    @app.get("/api/status")
    async def get_status() -> JSONResponse:
        return JSONResponse(captioner.status())

    @app.get("/api/settings")
    async def get_settings() -> JSONResponse:
        return JSONResponse(
            {
                "system_prompt": captioner.system_prompt,
                "prompt": captioner.prompt,
            }
        )

    @app.put("/api/settings")
    async def put_settings(update: SettingsUpdate) -> JSONResponse:
        if update.system_prompt is not None:
            captioner.system_prompt = update.system_prompt
        if update.prompt is not None:
            captioner.prompt = update.prompt

        try:
            persist_settings()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {exc}") from exc

        return JSONResponse(
            {
                "ok": True,
                "system_prompt": captioner.system_prompt,
                "prompt": captioner.prompt,
            }
        )

    @app.post("/api/upload")
    async def upload_media(files: Annotated[list[UploadFile], File(...)]) -> JSONResponse:
        uploaded: list[dict] = []
        skipped: list[dict] = []

        for incoming in files:
            filename = (incoming.filename or "").strip()
            if not filename:
                skipped.append({"filename": "", "reason": "missing filename"})
                continue

            destination = _dedupe_destination(config.root, filename)
            if not is_supported(destination):
                skipped.append({"filename": filename, "reason": "unsupported extension"})
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("wb") as out_file:
                    shutil.copyfileobj(incoming.file, out_file)
            finally:
                await incoming.close()

            media_type = media_type_for(destination)
            if media_type is None:
                skipped.append({"filename": filename, "reason": "unsupported media type"})
                destination.unlink(missing_ok=True)
                continue

            media = MediaFile(
                id=len(media_files),
                abs_path=destination,
                rel_path=str(destination.relative_to(config.root)),
                media_type=media_type,
                has_caption=has_nonempty_caption(destination),
                **probe_media_metadata(destination, media_type),
            )
            media_files.append(media)
            uploaded.append({"id": media.id, "rel_path": media.rel_path})

        return JSONResponse(
            {
                "uploaded_count": len(uploaded),
                "uploaded": uploaded,
                "skipped": skipped,
            }
        )

    @app.get("/media/{file_id}")
    async def get_media(file_id: int) -> FileResponse:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")
        path = media_files[file_id].abs_path
        return FileResponse(path)

    @app.get("/thumbnail/{file_id}")
    async def get_thumbnail(file_id: int) -> Response:
        if file_id < 0 or file_id >= len(media_files) or media_files[file_id] is None:
            raise HTTPException(status_code=404, detail="File not found")
        media = media_files[file_id]

        def _thumb() -> bytes:
            return thumbnail_bytes(media.abs_path, media.media_type, max_width=420)

        try:
            data = await asyncio.get_running_loop().run_in_executor(None, _thumb)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return Response(content=data, media_type="image/jpeg")

    return app
