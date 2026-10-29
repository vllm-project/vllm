# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import cast

from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs import TokPrompt
from vllm.renderers.params import TokenizeParams


class UnlimitedOCR2Renderer(HfRenderer):
    """Validate page layouts before requests enter the engine worker."""

    def _validate_pages(self, ids: Sequence[int], offsets: Sequence[int]) -> None:
        page_id = self.model_config.hf_config.sparse_attention_config["page_token_id"]
        if (
            offsets
            and ids.count(page_id) == len(offsets)
            and all(offset > 0 and ids[offset - 1] == page_id for offset in offsets)
        ):
            return
        raise VLLMValidationError(
            "Unlimited-OCR-2 requires <PAGE><image> for every page and "
            "at least one image.",
            parameter="prompt",
        )

    def validate_token_ids(
        self, token_ids: Sequence[int], *, parameter: str | None = None
    ) -> None:
        super().validate_token_ids(token_ids, parameter=parameter)
        image_id = self.model_config.hf_config.sparse_attention_config["image_token_id"]
        # Pre-rendered scale-out inputs contain expanded image-token runs.
        self._validate_pages(
            token_ids,
            [
                i
                for i, token in enumerate(token_ids)
                if token == image_id and (i == 0 or token_ids[i - 1] != image_id)
            ],
        )

    def _validate_prompt(self, prompt: TokPrompt) -> None:
        # Validate before multimodal processing updates the sender cache.
        ids = cast(list[int], prompt.get("prompt_token_ids", []))
        image_id = self.model_config.hf_config.sparse_attention_config["image_token_id"]
        self._validate_pages(
            ids, [i for i, token in enumerate(ids) if token == image_id]
        )

    def _validate_result(self, inputs: EngineInput) -> EngineInput:
        if inputs["type"] == "multimodal":
            images = inputs["mm_placeholders"].get("image", [])
            self._validate_pages(
                inputs["prompt_token_ids"], [image.offset for image in images]
            )
        else:
            self._validate_pages([], [])
        return inputs

    def process_for_engine(
        self,
        prompt: TokPrompt,
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
        tok_params: TokenizeParams | None = None,
    ) -> EngineInput:
        self._validate_prompt(prompt)
        return self._validate_result(
            super().process_for_engine(
                prompt, arrival_time, skip_mm_cache=skip_mm_cache, tok_params=tok_params
            )
        )

    async def process_for_engine_async(
        self,
        prompt: TokPrompt,
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
        tok_params: TokenizeParams | None = None,
    ) -> EngineInput:
        self._validate_prompt(prompt)
        return self._validate_result(
            await super().process_for_engine_async(
                prompt, arrival_time, skip_mm_cache=skip_mm_cache, tok_params=tok_params
            )
        )
