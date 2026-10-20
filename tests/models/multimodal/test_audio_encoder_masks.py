# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.audio_padding_mask import (
    broadcast_audio_key_padding_mask,
)

_FLASH_NAMES = (
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "kernels-community/flash-attn2",
    "kernels-community/vllm-flash-attn3",
    "kernels-community/aiter-flash-attn",
    "kernels-community/flash-attn4",
    "paged|flash_attention_2",
)


def _keep() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )


def test_sdpa_and_default_are_compact_bool():
    keep = _keep()
    for name in (None, "sdpa", "paged|sdpa"):
        mask = broadcast_audio_key_padding_mask(
            keep,
            name,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert mask.dtype == torch.bool
        assert mask.shape == (2, 1, 1, 5)
        assert torch.equal(mask, keep.view(2, 1, 1, 5))
        assert mask.shape[-2] != mask.shape[-1]


@pytest.mark.parametrize("name", _FLASH_NAMES)
def test_flash_names_are_long_key_mask(name: str):
    mask = broadcast_audio_key_padding_mask(
        _keep(),
        name,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.dtype == torch.long
    assert mask.shape == (2, 5)
    assert torch.equal(mask, _keep().to(torch.long))


def test_eager_uses_additive_mask_and_all_pad_row_is_neg_inf():
    keep = _keep()
    mask = broadcast_audio_key_padding_mask(
        keep,
        "eager",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 1, 1, 5)
    assert mask.dtype == torch.float32
    assert torch.equal(mask[0, 0, 0, :3], torch.zeros(3))
    assert torch.isneginf(mask[0, 0, 0, 3:]).all()

    all_pad = torch.zeros(1, 4, dtype=torch.bool)
    padded = broadcast_audio_key_padding_mask(
        all_pad,
        "eager",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert padded.shape == (1, 1, 1, 4)
    assert torch.isneginf(padded).all()


def test_flash_rejects_an_all_pad_row():
    keep = torch.tensor([[True, False], [False, False]])
    with pytest.raises(ValueError, match="all-pad"):
        broadcast_audio_key_padding_mask(
            keep,
            "flash_attention_2",
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("name", ["paged|paged|sdpa", "flex_attention", "unknown"])
def test_unsupported_names_raise(name: str):
    with pytest.raises(ValueError, match="unsupported audio attention"):
        broadcast_audio_key_padding_mask(
            _keep(),
            name,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_compact_sdpa_matches_square_mask_on_kept_rows():
    torch.manual_seed(0)
    query = torch.randn(2, 1, 5, 8)
    key = torch.randn(2, 1, 5, 8)
    value = torch.randn(2, 1, 5, 8)
    keep = _keep()
    square = torch.zeros(2, 1, 5, 5)
    square.masked_fill_(~keep.view(2, 1, 1, 5), float("-inf"))
    compact = keep.view(2, 1, 1, 5)
    square_out = F.scaled_dot_product_attention(query, key, value, attn_mask=square)
    compact_out = F.scaled_dot_product_attention(query, key, value, attn_mask=compact)
    kept_rows = keep.view(2, 1, 5, 1).expand_as(square_out)
    delta = torch.abs(square_out - compact_out)[kept_rows]
    assert torch.max(delta).item() == 0
