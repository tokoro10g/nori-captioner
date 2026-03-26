from __future__ import annotations

from typing import Sequence

from PIL.Image import Image

from .base import BaseCaptioner


class GemmaCaptioner(BaseCaptioner):
    def __init__(self, model_id: str, device_map: str = "auto", quantization_config=None):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self._torch = torch
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            quantization_config=quantization_config,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _generate(
        self,
        content: list[dict],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": content + [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0].strip()

    def caption_image(
        self,
        image: Image,
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        return self._generate(
            [{"type": "image", "image": image}],
            system_prompt,
            prompt,
            max_new_tokens,
        )

    def caption_video(
        self,
        frames: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        if not frames:
            raise ValueError("No frames provided")
        content = [{"type": "image", "image": frame} for frame in frames]
        return self._generate(content, system_prompt, prompt, max_new_tokens)
