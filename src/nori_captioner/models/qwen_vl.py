from __future__ import annotations

from typing import Sequence

from PIL.Image import Image

from .base import BaseCaptioner


class QwenVLCaptioner(BaseCaptioner):
    def __init__(self, model_id: str, device_map: str = "auto", quantization_config=None):
        import torch
        from transformers import AutoConfig, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)

        config = AutoConfig.from_pretrained(model_id)
        model_type = getattr(config, "model_type", "")
        self.model_type = model_type

        if model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                dtype="auto",
                quantization_config=quantization_config,
            ).eval()
        else:
            from transformers import Qwen2VLForConditionalGeneration

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype="auto",
                quantization_config=quantization_config,
            ).eval()

    def _messages_for_images(
        self,
        images: Sequence[Image],
        system_prompt: str,
        prompt: str,
    ) -> list[dict]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {"role": "user", "content": content},
        ]

    def _caption_qwen3(
        self,
        images: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = self._messages_for_images(images, system_prompt, prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)

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

    def _caption_qwen2(
        self,
        images: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        messages = self._messages_for_images(images, system_prompt, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

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
        if self.model_type == "qwen3_vl":
            return self._caption_qwen3([image], system_prompt, prompt, max_new_tokens)
        return self._caption_qwen2([image], system_prompt, prompt, max_new_tokens)

    def caption_video(
        self,
        frames: Sequence[Image],
        system_prompt: str,
        prompt: str,
        max_new_tokens: int = 384,
    ) -> str:
        if not frames:
            raise ValueError("No frames provided")
        if self.model_type == "qwen3_vl":
            return self._caption_qwen3(frames, system_prompt, prompt, max_new_tokens)
        return self._caption_qwen2(frames, system_prompt, prompt, max_new_tokens)
