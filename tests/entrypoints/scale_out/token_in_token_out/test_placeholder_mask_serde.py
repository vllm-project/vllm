# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.multimodal.inputs import PlaceholderRange


def test_mm_features_preserve_sparse_placeholder_mask():
    mask = [False, True, False, True]
    placeholder = PlaceholderRange(
        offset=2,
        length=len(mask),
        is_embed=torch.tensor(mask, dtype=torch.bool),
    )
    features = MultiModalFeatures(
        mm_hashes={"image": ["abc123"]},
        mm_placeholders={
            "image": [PlaceholderRangeInfo.from_placeholder_range(placeholder)]
        },
    )

    decoded = MultiModalFeatures.model_validate_json(features.model_dump_json())

    assert decoded.mm_placeholders["image"][0].is_embed == mask
    decoded_placeholder = decoded.mm_placeholders["image"][0].to_placeholder_range()
    assert torch.equal(decoded_placeholder.is_embed, placeholder.is_embed)


def test_mm_features_reject_placeholder_mask_length_mismatch():
    with pytest.raises(ValueError, match="is_embed.*length"):
        PlaceholderRangeInfo(offset=2, length=4, is_embed=[True])
