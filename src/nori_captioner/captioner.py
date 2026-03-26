from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .models import GemmaCaptioner, GenericVisionCaptioner, QwenVLCaptioner
from .scanner import (
    MediaFile,
    extract_evenly_spaced_frames,
    has_nonempty_caption,
    refresh_caption_flags,
    write_caption,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert dataset captioner for vision-language training data. "
    "Produce precise, concrete, visual-only descriptions. Do not include "
    "safety disclaimers, moralizing, or speculative claims about facts that "
    "are not observable in the image/video."
)

DEFAULT_PROMPT = (
    "Write one ultra-detailed, factual training caption for this media as "
    "a single dense paragraph. Describe the main subjects, appearance, "
    "clothing or materials, pose, actions, expression, and composition. For "
    "people, include visibly inferable attributes such as apparent ethnicity, "
    "facial features and structure, skin tone, hairstyle, body build and "
    "proportions, age range, and posture when these are clearly visible. "
    "Include camera angle, framing, background, environment, lighting, "
    "color, mood, and style, explicitly naming the visual style category "
    "(for example photorealistic photo, cinematic still, 3D rendering/CGI, "
    "anime, manga, digital illustration, painting, sketch, or mixed media), "
    "plus visible text and notable objects or spatial "
    "relationships. Be specific and exhaustive, avoid hedging, avoid bullet "
    "points, and do not invent details that are not visible. Aim for roughly "
    "120 words but not limited by that when enough visual detail is present."
)

LOGGER = logging.getLogger("nori_captioner.captioner")


@dataclass(slots=True)
class CaptionTask:
    file_id: int


class CaptionerService:
    def __init__(
        self,
        media_files: list[MediaFile],
        model_spec: str | None,
        system_prompt: str,
        prompt: str,
        frames_per_video: int,
        quantize: int | None,
        device: str,
    ) -> None:
        self.media_files = media_files
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.prompt = prompt or DEFAULT_PROMPT
        self.frames_per_video = max(1, frames_per_video)
        self.quantize = quantize
        self.device = device
        self.model_spec = model_spec
        self.model_id = self._resolve_model_id(model_spec) if model_spec else None
        self.captioner = None

        self.queue: asyncio.Queue[CaptionTask] = asyncio.Queue()
        self.queued_ids: set[int] = set()
        self.running_id: int | None = None
        self.done_count = 0
        self.error_count = 0
        self.last_error: str | None = None
        self._worker_task: asyncio.Task[Any] | None = None

    def _resolve_model_id(self, model_spec: str | None) -> str | None:
        if not model_spec:
            return None

        model_spec = model_spec.strip()
        alias_map = {
            "gemma3:4b": "google/gemma-3-4b-it",
            "gemma3:12b": "google/gemma-3-12b-it",
            "gemma3:27b": "google/gemma-3-27b-it",
            "qwen2-vl:2b": "Qwen/Qwen2-VL-2B-Instruct",
            "qwen2-vl:7b": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen2-vl:72b": "Qwen/Qwen2-VL-72B-Instruct",
            "qwen2.5-vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
            "qwen3-vl:2b": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl:4b": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl:8b": "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3-vl:32b": "Qwen/Qwen3-VL-32B-Instruct",
            "qwen3-vl:30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        }
        if model_spec in alias_map:
            return alias_map[model_spec]

        if "/" in model_spec:
            return model_spec

        raise ValueError(
            f"Unknown model spec '{model_spec}'. Use aliases like qwen3-vl:8b "
            "or a raw HF model id org/name."
        )

    def _build_quant_config(self):
        if self.quantize is None:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        if self.quantize == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if self.quantize == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError("quantize must be 4, 8, or None")

    def _device_map(self) -> str:
        if self.device == "auto":
            return "auto"
        return self.device

    def load_model(self) -> None:
        if not self.model_id:
            self.captioner = None
            return
        if self.captioner is not None:
            return

        quant_config = self._build_quant_config()
        model_id = self.model_id

        # Dispatch known model families first, then fallback to generic loader.
        if "gemma-3" in model_id:
            self.captioner = GemmaCaptioner(
                model_id,
                device_map=self._device_map(),
                quantization_config=quant_config,
            )
            return

        if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id:
            self.captioner = QwenVLCaptioner(
                model_id,
                device_map=self._device_map(),
                quantization_config=quant_config,
            )
            return

        self.captioner = GenericVisionCaptioner(
            model_id,
            device_map=self._device_map(),
            quantization_config=quant_config,
        )

    async def preload_model(self) -> None:
        if not self.model_id:
            return

        LOGGER.info("Preloading model at startup: %s", self.model_id)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.load_model)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            LOGGER.exception("Model preload failed for %s", self.model_id)
            raise
        else:
            self.last_error = None
            LOGGER.info("Model ready: %s", self.model_id)

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    def enqueue(self, file_id: int) -> bool:
        if file_id < 0 or file_id >= len(self.media_files):
            return False
        if self.media_files[file_id] is None:
            return False
        if file_id in self.queued_ids or self.running_id == file_id:
            return False
        self.queued_ids.add(file_id)
        self.queue.put_nowait(CaptionTask(file_id=file_id))
        return True

    def enqueue_batch(self, mode: str = "uncaptioned") -> int:
        added = 0
        for media in self.media_files:
            if media is None:
                continue
            caption_exists = has_nonempty_caption(media.abs_path)
            if mode == "uncaptioned" and caption_exists:
                continue
            if self.enqueue(media.id):
                added += 1
        return added

    def clear_queue(self) -> int:
        removed = 0
        while True:
            try:
                task = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.queued_ids.discard(task.file_id)
            self.queue.task_done()
            removed += 1
        return removed

    def status(self) -> dict[str, Any]:
        refresh_caption_flags(self.media_files)
        return {
            "model_spec": self.model_spec,
            "model_id": self.model_id,
            "model_loaded": self.captioner is not None,
            "queue_len": self.queue.qsize(),
            "queued_ids": sorted(self.queued_ids),
            "running_id": self.running_id,
            "done": self.done_count,
            "errors": self.error_count,
            "last_error": self.last_error,
        }

    async def _run_caption_task(self, media: MediaFile) -> None:
        if self.captioner is None:
            self.load_model()
        if self.captioner is None:
            raise RuntimeError("No model loaded")

        loop = asyncio.get_running_loop()

        if media.media_type == "image":
            with Image.open(media.abs_path) as img:
                image = img.convert("RGB")
            caption = await loop.run_in_executor(
                None,
                lambda: self.captioner.caption_image(
                    image,
                    self.system_prompt,
                    self.prompt,
                ),
            )
        else:
            frames = extract_evenly_spaced_frames(Path(media.abs_path), self.frames_per_video)
            caption = await loop.run_in_executor(
                None,
                lambda: self.captioner.caption_video(
                    frames,
                    self.system_prompt,
                    self.prompt,
                ),
            )

        write_caption(media.abs_path, caption)
        media.has_caption = has_nonempty_caption(media.abs_path)

    async def _worker_loop(self) -> None:
        while True:
            task = await self.queue.get()
            media = self.media_files[task.file_id]
            self.queued_ids.discard(task.file_id)
            if media is None:
                self.queue.task_done()
                continue
            self.running_id = task.file_id
            try:
                await self._run_caption_task(media)
                self.done_count += 1
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                self.last_error = str(exc)
                LOGGER.exception(
                    "Auto-caption failed for %s (file_id=%s, media_type=%s, model=%s)",
                    media.rel_path,
                    media.id,
                    media.media_type,
                    self.model_id,
                )
            finally:
                self.running_id = None
                self.queue.task_done()
