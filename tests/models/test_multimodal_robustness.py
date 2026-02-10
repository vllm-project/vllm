# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

try:
    import pytest
except ImportError:
    pytest = MagicMock()
import torch

from vllm.model_executor.models.transformers.multimodal import (
    MultiModalProcessingInfo,
    MultiModalProcessor,
)


def test_robustness_attribute_error():
    # Mock context and config
    ctx = MagicMock()
    ctx.model_config.multimodal_config.mm_processor_kwargs = {}

    # Mock processor that raises AttributeError in _get_num_multimodal_tokens
    processor = MagicMock()
    processor._get_num_multimodal_tokens.side_effect = AttributeError("test error")

    info = MultiModalProcessingInfo(ctx)
    info.get_max_image_size = MagicMock(return_value=(100, 100))
    info.get_hf_processor = MagicMock(return_value=processor)

    # Should not crash and return 0
    tokens = info.get_max_image_tokens()
    assert tokens == 0


def test_robustness_attribute_error_apply():
    # Mock info and dummy inputs
    info = MagicMock()
    info.ctx.model_config.multimodal_config.mm_processor_kwargs = {}
    dummy_inputs = MagicMock()

    # Mock processor that raises AttributeError
    hf_processor = MagicMock()
    hf_processor._get_num_multimodal_tokens.side_effect = AttributeError("test error")

    processor = MultiModalProcessor(info, dummy_inputs)
    processor.get_hf_processor = MagicMock(return_value=hf_processor)

    # Mock _apply_hf_processor_text_mm to avoid text tokenization logic
    # We use torch.zeros here
    processor._apply_hf_processor_text_mm = MagicMock(
        return_value=(
            [1, 2, 3],  # prompt_ids
            {
                "mm_token_type_ids": torch.zeros((1, 3), dtype=torch.long)
            },  # processed_data
            None,  # mm_uuids
        )
    )

    mm_items = MagicMock()
    # Mock images.get_image_size(item_idx)
    images = MagicMock()
    images.__len__.return_value = 1
    images.get_image_size.return_value = MagicMock(height=100, width=100)

    mm_items.get_items.return_value = images

    # Mock _hash_mm_items
    processor._hash_mm_items = MagicMock(return_value={})

    # Mock _get_mm_fields_config
    processor._get_mm_fields_config = MagicMock(return_value={})

    # Should not crash during _get_num_multimodal_tokens call
    # It might crash later due to empty dict if it tries
    # to access keys, but our fix is there.
    # Actually, in apply:
    # mm_tokens_per_modality = { "num_image_tokens": [], "num_image_patches": [] }
    # split_sizes = []
    # Nothing will happen in if split_sizes:
    # processed_data["num_image_patches"] = torch.tensor([])
    # This should actually complete apply successfully if mocked correctly.

    res = processor.apply("text", mm_items)
    assert res is not None
    assert "num_image_patches" in res.mm_kwargs
    print("test_robustness_attribute_error_apply passed")


if __name__ == "__main__":
    test_robustness_attribute_error()
    test_robustness_attribute_error_apply()
    print("All tests passed!")
