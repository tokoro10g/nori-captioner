from __future__ import annotations

from typing import Sequence

from PIL.Image import Image

from .base import BaseCaptioner


class GenericVisionCaptioner(BaseCaptioner):
    def __init__(self, model_id: str, device_map: str = "auto", quantization_config=None):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._torch = torch
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            quantization_config=quantization_config,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _generate(
        self,
        images: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def caption_image(
        self,
        image: Image,
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        return self._generate([image], system_prompt, prompt, max_new_tokens)

    def caption_video(
        self,
        frames: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        if not frames:
            raise ValueError("No frames provided")
        return self._generate(frames, system_prompt, prompt, max_new_tokens)
