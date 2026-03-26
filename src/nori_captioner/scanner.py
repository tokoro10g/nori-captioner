from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import av
from PIL import Image

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
}
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}


@dataclass(slots=True)
class MediaFile:
    id: int
    abs_path: Path
    rel_path: str
    media_type: str
    has_caption: bool
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None


def probe_media_metadata(path: Path, media_type: str) -> dict[str, int | float | None]:
    if media_type == "image":
        try:
            with Image.open(path) as img:
                width, height = img.size
        except OSError:
            return {
                "width": None,
                "height": None,
                "frame_count": None,
                "fps": None,
                "duration_seconds": None,
            }
        return {
            "width": width,
            "height": height,
            "frame_count": 1,
            "fps": None,
            "duration_seconds": None,
        }

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            width = int(stream.width) if stream.width else None
            height = int(stream.height) if stream.height else None

            fps_value: float | None = None
            rate = stream.average_rate or stream.base_rate
            if rate:
                fps_value = float(rate)

            duration_seconds: float | None = None
            if stream.duration and stream.time_base:
                duration_seconds = float(stream.duration * stream.time_base)
            elif container.duration:
                duration_seconds = float(container.duration / av.time_base)

            frame_count = int(stream.frames) if stream.frames else None
            if frame_count is None and fps_value and duration_seconds:
                if duration_seconds > 0:
                    frame_count = int(round(duration_seconds * fps_value))
    except (av.AVError, IndexError, OSError, ValueError):
        return {
            "width": None,
            "height": None,
            "frame_count": None,
            "fps": None,
            "duration_seconds": None,
        }

    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps_value,
        "duration_seconds": duration_seconds,
    }


def is_supported(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in IMAGE_EXTS or ext in VIDEO_EXTS


def media_type_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def caption_sidecar_path(media_path: Path) -> Path:
    return media_path.with_suffix(".txt")


def read_caption(media_path: Path) -> str | None:
    caption_path = caption_sidecar_path(media_path)
    if not caption_path.exists():
        return None
    return caption_path.read_text(encoding="utf-8").strip() or ""


def has_nonempty_caption(media_path: Path) -> bool:
    caption_text = read_caption(media_path)
    return bool(caption_text and caption_text.strip())


def write_caption(media_path: Path, text: str) -> None:
    caption_path = caption_sidecar_path(media_path)
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=caption_path.parent,
    ) as tmp_file:
        tmp_file.write(text)
        tmp_name = tmp_file.name
    Path(tmp_name).replace(caption_path)


def scan_directory(root: Path) -> list[MediaFile]:
    root = root.resolve()
    found: list[MediaFile] = []

    # Prune hidden directories (e.g. .git, .venv, .cache) from traversal.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        base = Path(dirpath)
        for filename in sorted(filenames):
            path = base / filename
            if not is_supported(path):
                continue
            media_type = media_type_for(path)
            if media_type is None:
                continue
            metadata = probe_media_metadata(path, media_type)
            rel_path = str(path.relative_to(root))
            found.append(
                MediaFile(
                    id=len(found),
                    abs_path=path,
                    rel_path=rel_path,
                    media_type=media_type,
                    has_caption=has_nonempty_caption(path),
                    width=metadata["width"],
                    height=metadata["height"],
                    frame_count=metadata["frame_count"],
                    fps=metadata["fps"],
                    duration_seconds=metadata["duration_seconds"],
                )
            )
    return found


def paginate(items: list[MediaFile], page: int, per_page: int) -> tuple[list[MediaFile], int]:
    per_page = max(1, per_page)
    page = max(1, page)
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], total


def extract_evenly_spaced_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    n_frames = max(1, n_frames)
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    total_frames = stream.frames
    if total_frames and total_frames > 0:
        if total_frames <= n_frames:
            target_idxs = list(range(total_frames))
        else:
            target_idxs = [int(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]

        selected: dict[int, Image.Image] = {}
        for idx, frame in enumerate(container.decode(stream)):
            if idx in target_idxs:
                selected[idx] = frame.to_image().convert("RGB")
            if len(selected) == len(target_idxs):
                break
        frames = [selected[i] for i in sorted(selected)]
        return frames[:n_frames] if frames else []

    decoded = [f.to_image().convert("RGB") for f in container.decode(stream)]
    if not decoded:
        return []
    if len(decoded) <= n_frames:
        return decoded
    return [decoded[int(i * (len(decoded) - 1) / (n_frames - 1))] for i in range(n_frames)]


def _thumbnail_from_image(image: Image.Image, max_width: int = 400) -> bytes:
    image = image.convert("RGB")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def thumbnail_bytes(media_path: Path, media_type: str, max_width: int = 400) -> bytes:
    if media_type == "image":
        with Image.open(media_path) as img:
            return _thumbnail_from_image(img, max_width=max_width)

    frames = extract_evenly_spaced_frames(media_path, 1)
    if not frames:
        raise ValueError(f"Could not extract frame from {media_path}")
    return _thumbnail_from_image(frames[0], max_width=max_width)


def refresh_caption_flags(media_files: Iterable[MediaFile | None]) -> None:
    for item in media_files:
        if item is not None:
            item.has_caption = has_nonempty_caption(item.abs_path)
