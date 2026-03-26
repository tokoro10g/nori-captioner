from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL.Image import Image


class BaseCaptioner(ABC):
    @abstractmethod
    def caption_image(
        self,
        image: Image,
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def caption_video(
        self,
        frames: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        raise NotImplementedError
