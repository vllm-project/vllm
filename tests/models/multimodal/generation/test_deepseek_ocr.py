# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard tests for DeepSeek-OCR _encode_local_features.

_encode_local_features must skip processing only for an empty-element
patches tensor, not for a zero-sum one.

Regression for #64075: the old guard `torch.sum(patches).item() == 0`
returned None for a 1536x768 half-white/half-black image whose two
normalised crop patches cancel to sum==0, causing a shape mismatch in
_merge_multimodal_embeddings and a fatal EngineCore crash.

The correct guard is `patches.numel() == 0`.
"""

import pytest
import torch

from vllm.model_executor.models.deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.deepseek_ocr2 import DeepseekOCR2ForCausalLM


class TestDeepseekOCREncodeLocalFeaturesGuard:
    def test_ocr2_empty_patches_returns_none(self):
        """Zero-element tensor → skip local encoding (non-tiled image)."""
        empty = torch.empty(0, 3, 224, 224)
        assert DeepseekOCR2ForCausalLM._encode_local_features(None, empty) is None

    def test_ocr2_zero_sum_patches_enters_model_body(self):
        """Non-empty patches that happen to sum to zero must NOT return None.

        The method will raise AttributeError when it tries to call
        self.sam_model on a None self – that is the expected outcome here
        because it proves the guard was passed, not that encoding succeeded.
        The real crash was a shape-mismatch ValueError that killed EngineCore.
        """
        # Two patches that cancel: white (1.0) + black (-1.0) after Normalize.
        wb = torch.cat([torch.ones(1, 3, 4, 4), -torch.ones(1, 3, 4, 4)])
        assert wb.numel() > 0
        assert wb.sum().item() == 0.0
        with pytest.raises(AttributeError):
            DeepseekOCR2ForCausalLM._encode_local_features(None, wb)

    def test_ocr1_empty_patches_returns_none(self):
        """OCR-1: zero-element tensor → skip local encoding."""
        empty = torch.empty(0, 3, 224, 224)
        crop_shape = torch.tensor([0, 0])
        assert (
            DeepseekOCRForCausalLM._encode_local_features(None, empty, crop_shape)
            is None
        )

    def test_ocr1_zero_sum_patches_enters_model_body(self):
        """OCR-1: non-empty zero-sum patches must pass the guard."""
        wb = torch.cat([torch.ones(1, 3, 4, 4), -torch.ones(1, 3, 4, 4)])
        crop_shape = torch.tensor([1, 2])
        with pytest.raises(AttributeError):
            DeepseekOCRForCausalLM._encode_local_features(None, wb, crop_shape)


@pytest.mark.core_model
@pytest.mark.parametrize(
    "model_id, expected_tokens",
    [("deepseek-ai/DeepSeek-OCR", 273), ("deepseek-ai/DeepSeek-OCR-2", 257)],
)
def test_deepseek_ocr_request_processor_kwargs(vllm_runner, model_id, expected_tokens):
    """Generate with a request override and verify the processed image span."""
    from PIL import Image

    from vllm import SamplingParams

    with vllm_runner(
        model_id,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        kv_cache_memory_bytes=1 << 29,
        attention_config={"backend": "TRITON_ATTN"},
        mm_encoder_attn_backend="TORCH_SDPA",
        limit_mm_per_prompt={"image": 1},
    ) as runner:
        result = runner.llm.generate(
            {
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {"image": Image.new("RGB", (1600, 1200))},
                "mm_processor_kwargs": {"crop_mode": False},
            },
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )[0]
        assert result.outputs[0].finish_reason == "length"
        assert len(result.outputs[0].token_ids) == 1
        image_token_id = runner.llm.get_tokenizer().convert_tokens_to_ids("<image>")
        assert result.prompt_token_ids.count(image_token_id) == expected_tokens
