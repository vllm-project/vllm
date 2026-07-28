# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page shapes and dtypes for the FlashInfer ReplaySSM speculative-decode ring.

FlashInfer derives its logical replay window from the ring length
(``max_window = x_cache.size(2) - max_seqlen``), so unlike the Triton path the
ring must be exactly ``B + T`` rather than the next power of two. These are pure
shape/dtype calculations and run on CPU.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

# Nemotron-H-like geometry, small enough to reason about by hand.
INTERMEDIATE_SIZE = 512
NUM_HEADS = 8
HEAD_DIM = 64
STATE_SIZE = 128
N_GROUPS = 4
CONV_KERNEL = 4


def _base_shapes(tp_world_size: int, num_spec: int):
    return MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=tp_world_size,
        intermediate_size=INTERMEDIATE_SIZE,
        n_groups=N_GROUPS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
        num_spec=num_spec,
    )


def _fi_shapes(buffer_len: int, num_spec: int, tp_world_size: int = 1):
    return MambaStateShapeCalculator.append_replayssm_spec_flashinfer_ring(
        _base_shapes(tp_world_size, num_spec),
        N_GROUPS,
        tp_world_size,
        buffer_len,
        num_spec,
    )


@pytest.mark.parametrize(
    "buffer_len,num_spec,expected_ring",
    [
        (16, 3, 20),  # B=16, T=4  -> 20, not next_pow2(20)=32
        (16, 1, 18),
        (16, 7, 24),
        (8, 3, 12),
        (8, 7, 16),  # coincidentally a power of two
    ],
)
def test_flashinfer_ring_len_is_exactly_b_plus_t(buffer_len, num_spec, expected_ring):
    assert (
        MambaStateShapeCalculator.replayssm_spec_flashinfer_ring_len(
            buffer_len, num_spec
        )
        == expected_ring
    )


def test_triton_ring_len_stays_power_of_two():
    """The Triton path is a fixed boundary: it keeps its next_pow2 mask ring."""
    assert MambaStateShapeCalculator.replayssm_spec_ring_len(16, 3) == 32
    assert MambaStateShapeCalculator.replayssm_spec_flashinfer_ring_len(16, 3) == 20


def test_flashinfer_page_is_five_tensors_in_contract_order():
    conv, ssm, x_cache, b_cache, dt_cache = _fi_shapes(buffer_len=16, num_spec=3)
    local_heads, head_dim, dstate = ssm
    ring = 20

    assert (local_heads, head_dim, dstate) == (NUM_HEADS, HEAD_DIM, STATE_SIZE)
    assert x_cache == (NUM_HEADS, ring, HEAD_DIM)
    assert b_cache == (N_GROUPS, ring, STATE_SIZE)
    assert dt_cache == (NUM_HEADS, ring)
    # The conv page is untouched by the ring append.
    assert conv == _base_shapes(1, 3)[0]


@pytest.mark.parametrize("tp_world_size", [1, 2, 4])
def test_flashinfer_shapes_are_tp_local(tp_world_size):
    _, ssm, x_cache, b_cache, dt_cache = _fi_shapes(
        buffer_len=16, num_spec=3, tp_world_size=tp_world_size
    )
    local_heads = NUM_HEADS // tp_world_size
    local_groups = N_GROUPS // tp_world_size
    ring = 20

    assert ssm[0] == local_heads
    assert x_cache == (local_heads, ring, HEAD_DIM)
    assert b_cache == (local_groups, ring, STATE_SIZE)
    assert dt_cache == (local_heads, ring)
    # heads_per_group is what FlashInfer JIT-stamps; it must stay integral.
    assert local_heads % local_groups == 0


def test_flashinfer_group_sharding_matches_the_mixer():
    """G must match `self.n_groups // tp_size` as the decode path computes it.

    `MambaMixer2` pre-extends `n_groups` so it divides the TP size, and
    `extra_groups_for_head_shards` is a no-op on an already-extended count, so
    the calculator agrees whether it is handed the raw or extended value.
    """
    tp_world_size = 4
    n_groups_raw = 3
    extra = MambaStateShapeCalculator.extra_groups_for_head_shards(
        n_groups_raw, tp_world_size
    )
    n_groups_ext = n_groups_raw + extra

    base = MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=tp_world_size,
        intermediate_size=INTERMEDIATE_SIZE,
        n_groups=n_groups_ext,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
        num_spec=3,
    )
    _, _, _, b_cache, _ = (
        MambaStateShapeCalculator.append_replayssm_spec_flashinfer_ring(
            base, n_groups_ext, tp_world_size, 16, 3
        )
    )
    assert b_cache[0] == n_groups_ext // tp_world_size


def test_flashinfer_dtypes_are_activation_activation_fp32():
    base = MambaStateDtypeCalculator.mamba2_state_dtype(
        model_dtype=torch.bfloat16,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="float32",
    )
    dtypes = MambaStateDtypeCalculator.append_replayssm_spec_flashinfer_ring(
        base, torch.bfloat16
    )
    assert len(dtypes) == 5
    assert dtypes[:2] == base
    # x_cache and B_cache ride the activation dtype; dt stays fp32 because the
    # replayed decay is recomputed from it.
    assert dtypes[2] == torch.bfloat16
    assert dtypes[3] == torch.bfloat16
    assert dtypes[4] == torch.float32


def test_flashinfer_page_strides_match_the_kernel_contract():
    """Materialise one page and pin the strides FlashInfer indexes with."""
    _, _, x_cache, b_cache, dt_cache = _fi_shapes(buffer_len=16, num_spec=3)
    x = torch.empty(x_cache)
    b = torch.empty(b_cache)
    dt = torch.empty(dt_cache)

    assert x.stride(-1) == 1 and x.stride(-2) == HEAD_DIM
    assert b.stride(-1) == 1 and b.stride(-2) == STATE_SIZE
    assert dt.stride(-1) == 1


def _validation_stub(buffer_len: int, algorithm: str = "auto"):
    """Minimal stand-in for the fields _validate_replayssm_spec_flashinfer reads.

    Building a real VllmConfig needs model inspection, which is not available in
    a plain unit-test environment; the surrounding pydantic validator is covered
    by the engine tests.
    """
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            replayssm_buffer_len=buffer_len,
            mamba_ssm_cache_dtype="float32",
        ),
        mamba_config=SimpleNamespace(replayssm_spec_algorithm=algorithm),
    )


def test_flashinfer_rejects_buffer_len_above_the_kernel_window():
    with pytest.raises(ValueError, match="at most 16 cached tokens"):
        VllmConfig._validate_replayssm_spec_flashinfer(_validation_stub(32), 4)


def test_flashinfer_rejects_spec_window_larger_than_buffer():
    with pytest.raises(ValueError, match=r"1 \+ num_speculative_tokens"):
        VllmConfig._validate_replayssm_spec_flashinfer(_validation_stub(4), 8)


def test_flashinfer_validation_does_not_import_flashinfer():
    """Config validation must stay import-free: importing flashinfer here would
    initialise CUDA during config construction (see has_flashinfer's find_spec).
    """
    assert "flashinfer" not in sys.modules
    with pytest.raises(ValueError):
        # B > 16 fails before the availability probe; either way nothing may
        # import the package.
        VllmConfig._validate_replayssm_spec_flashinfer(_validation_stub(32), 4)
    assert "flashinfer" not in sys.modules
