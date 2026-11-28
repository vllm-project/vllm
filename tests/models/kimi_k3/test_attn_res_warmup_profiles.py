# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for the Kimi-K3 AttnRes Triton warmup profile contract.

get_attn_res_triton_warmup_profiles() must pre-compile every
(num_blocks, has_delta, block_write_idx, apply_output_norm) key that
KimiDecoderLayer feeds to attn_res() at runtime; otherwise the first prefill
JIT-compiles the missing variants. These tests enumerate the runtime keys from
(num_hidden_layers, attn_res_block_size) and assert the warmup profile set
covers them.
"""

import pytest

from vllm.models.kimi_k3.nvidia.ops.attn_res import (
    get_attn_res_triton_warmup_profiles,
)
from vllm.utils.math_utils import cdiv


def _runtime_attn_res_keys(
    num_hidden_layers: int,
    attn_res_block_size: int,
) -> set[tuple[int, bool, int, bool]]:
    """Enumerate the attn_res() keys KimiDecoderLayer produces at runtime.

    Mirrors vllm/models/kimi_k3/nvidia/model.py:
      - is_block_write_layer = layer_idx % attn_res_block_size == 0
      - block_write_idx      = layer_idx // attn_res_block_size, or -1
      - prev_valid_blocks    = cdiv(layer_idx, attn_res_block_size)
      - pre-attn passes delta=hidden_states, which is None only for layer 0
        (the decoder loop is entered with hidden_states=None and the embedding
        as prefix_sum), so layer 0 is a block write with no delta
      - post-attn passes delta=None iff the layer is a block-write layer, with
        num_blocks = prev_valid_blocks + is_block_write_layer
      - the final output pre-norm uses num_blocks = max_blocks with delta
        present, block_write_idx = -1 and no output norm.
    """
    keys: set[tuple[int, bool, int, bool]] = set()
    for layer_idx in range(num_hidden_layers):
        is_block_write = layer_idx % attn_res_block_size == 0
        block_write_idx = layer_idx // attn_res_block_size if is_block_write else -1
        prev_valid_blocks = cdiv(layer_idx, attn_res_block_size)

        # Pre-attn norm.
        keys.add((prev_valid_blocks, layer_idx != 0, block_write_idx, True))

        # Post-attn norm.
        mlp_valid_blocks = prev_valid_blocks + int(is_block_write)
        keys.add((mlp_valid_blocks, not is_block_write, -1, True))

    # Final output pre-norm.
    keys.add((cdiv(num_hidden_layers, attn_res_block_size), True, -1, False))
    return keys


@pytest.mark.parametrize(
    ("num_hidden_layers", "attn_res_block_size"),
    [
        pytest.param(93, 16, id="kimi-k3"),
        pytest.param(1, 16, id="single-layer"),
        pytest.param(16, 16, id="exactly-one-block"),
        pytest.param(17, 16, id="one-block-plus-one-layer"),
    ],
)
def test_attn_res_warmup_profiles_cover_runtime_keys(
    num_hidden_layers: int,
    attn_res_block_size: int,
) -> None:
    max_blocks = cdiv(num_hidden_layers, attn_res_block_size)
    profiles = set(get_attn_res_triton_warmup_profiles(max_blocks))
    assert _runtime_attn_res_keys(
        num_hidden_layers,
        attn_res_block_size,
    ) <= profiles


@pytest.mark.parametrize("max_blocks", [1, 2, 6, 8])
def test_attn_res_warmup_profiles_exclude_impossible_keys(max_blocks: int) -> None:
    """Profiles must not contain keys the runtime can never produce.

    These are impossible for any (num_hidden_layers, attn_res_block_size):
      - (0, True, 0, True): layer 0 always receives delta=None, so the
        num_blocks == 0 block write can never fuse a delta.
      - (k, False, k, True) for k >= 1: every block-write layer other than
        layer 0 always has delta present in pre-attn.
    """
    profiles = set(get_attn_res_triton_warmup_profiles(max_blocks))
    assert (0, True, 0, True) not in profiles
    assert all(
        (block_write_idx, False, block_write_idx, True) not in profiles
        for block_write_idx in range(1, max_blocks)
    )
