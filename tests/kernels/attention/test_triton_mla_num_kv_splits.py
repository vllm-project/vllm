# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-count selection of the TRITON_MLA decode backend (no GPU needed)."""

import pytest

from vllm.v1.attention.backends.mla.triton_mla import (
    _num_decode_kv_splits,
    _supports_one_wave_cap,
)

# RTX 5090: 170 SMs. A full-cudagraph capture at max_model_len 65536 sizes the
# default split count at 128.
SMS = 170
MAX_SEQ_LEN = 65536


@pytest.mark.parametrize(
    "num_rows, num_heads, want",
    [
        (1, 16, 128),  # 128 programs already fit one wave
        (2, 16, 64),  # one request verifying a draft token
        (1, 20, 64),  # GLM-4.7-Flash, two head tiles
        (2, 20, 32),
        (3, 20, 128),  # 6 programs per split over two tiles: default kept
        (1, 128, 128),
    ],
)
def test_lone_request_splits_fit_one_wave(num_rows, num_heads, want):
    splits = _num_decode_kv_splits(MAX_SEQ_LEN, 1, num_rows, num_heads, SMS, SMS)
    assert splits == want


@pytest.mark.parametrize("one_wave_sm_count, num_reqs", [(SMS, 2), (None, 1)])
def test_default_splits_kept(one_wave_sm_count, num_reqs):
    """A batch shares one split count, and a one-wave count sized for its shape
    would starve its longest request; GPUs without the cap keep the default."""
    splits = _num_decode_kv_splits(
        MAX_SEQ_LEN, num_reqs, num_reqs, 20, SMS, one_wave_sm_count
    )
    assert splits == 128


@pytest.mark.parametrize(
    "kv_lora_rank, qk_rope_head_dim, shared_memory_per_block, want",
    [
        (512, 64, 101376, True),  # SM86 / SM89 / SM120
        (512, 64, 232448, False),  # H100: several programs per SM
        (256, 64, 101376, False),  # smaller latent tile: several per SM
        (512, 128, 101376, False),
        (512, 64, 163840, False),  # A100
    ],
)
def test_one_wave_cap_gate(
    kv_lora_rank, qk_rope_head_dim, shared_memory_per_block, want
):
    got = _supports_one_wave_cap(
        kv_lora_rank, qk_rope_head_dim, shared_memory_per_block
    )
    assert got is want
