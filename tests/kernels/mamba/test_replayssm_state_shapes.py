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


def _alloc_scratch(algorithm: str, buffer_len: int = 16, scratch_bs: int = 8):
    """Run the builder's scratch allocator against a stub Mamba page."""
    from vllm.v1.attention.backends.mamba_attn import (
        BaseMambaAttentionMetadataBuilder,
    )

    spec = SimpleNamespace(
        shapes=_fi_shapes(buffer_len=buffer_len, num_spec=3),
        dtypes=(
            torch.float32,
            torch.float32,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
        ),
    )
    # The real builder seeds these to None before allocating.
    stub = SimpleNamespace(
        replayssm_buffer_len=buffer_len,
        decode_spec_fi_cb_scaled=None,
        decode_spec_fi_cumadt=None,
        decode_spec_fi_cb_old=None,
    )
    BaseMambaAttentionMetadataBuilder._init_replayssm_spec_flashinfer_scratch(
        stub, spec, scratch_bs, algorithm, torch.device("cpu")
    )
    return stub


@pytest.mark.parametrize("algorithm", ["auto", "two-kernel"])
def test_two_kernel_scratch_shapes_match_the_mma_fragment_layout(algorithm):
    from vllm.v1.attention.backends.mamba_attn import (
        _MMA_FRAG_SIZE,
        _MMA_M_TILE,
        _MMA_WARP_SIZE,
    )

    stub = _alloc_scratch(algorithm)
    scratch_bs, buffer_len = 8, 16
    k_old = ((buffer_len + 7) // 8) * 8

    assert stub.decode_spec_fi_cb_scaled.shape == (
        scratch_bs,
        NUM_HEADS,
        _MMA_WARP_SIZE,
        _MMA_FRAG_SIZE,
    )
    assert stub.decode_spec_fi_cumadt.shape == (scratch_bs, NUM_HEADS, _MMA_M_TILE)
    assert stub.decode_spec_fi_cb_old.shape == (
        scratch_bs,
        NUM_HEADS,
        _MMA_WARP_SIZE,
        k_old // 2,
    )
    # cb_scaled/cb_old ride the activation dtype; cumAdt_vec is always fp32.
    assert stub.decode_spec_fi_cb_scaled.dtype == torch.bfloat16
    assert stub.decode_spec_fi_cb_old.dtype == torch.bfloat16
    assert stub.decode_spec_fi_cumadt.dtype == torch.float32


def test_monolith_allocates_no_scratch():
    """FlashInfer routes on cb_scaled != nullptr, so 'monolith' must pass none."""
    stub = _alloc_scratch("monolith")
    assert stub.decode_spec_fi_cb_scaled is None
    assert stub.decode_spec_fi_cumadt is None
    assert stub.decode_spec_fi_cb_old is None


def test_scratch_allocator_rejects_the_triton_four_tensor_page():
    from vllm.v1.attention.backends.mamba_attn import (
        BaseMambaAttentionMetadataBuilder,
    )

    triton_page = SimpleNamespace(
        shapes=((64, 3), (NUM_HEADS, HEAD_DIM, STATE_SIZE), (32, 640), (NUM_HEADS, 32)),
        dtypes=(torch.float32,) * 4,
    )
    with pytest.raises(ValueError, match="5-tensor Mamba2"):
        BaseMambaAttentionMetadataBuilder._init_replayssm_spec_flashinfer_scratch(
            SimpleNamespace(replayssm_buffer_len=16),
            triton_page,
            8,
            "auto",
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "enabled,configured,expected_enabled,expected_rounds",
    [
        (False, 0, False, 0),
        (False, 5, False, 0),  # rounds are ignored when SR is off
        (True, 0, True, 10),  # 0 means "backend default"
        (True, 5, True, 5),
        (True, 10, True, 10),
    ],
)
def test_rounding_policy_resolution(
    enabled, configured, expected_enabled, expected_rounds
):
    """philox_rounds must be settled once, at backend construction.

    It is a kernel template parameter, so the value the decode path passes and
    the value the warmup compiles must agree or the first real request JITs a
    second specialisation. FlashInfer also forces it to 0 without a seed and
    asserts it is positive with one.
    """
    from vllm.model_executor.layers.mamba.ops.replayssm_spec_flashinfer import (
        _resolve_rounding_policy,
    )

    policy = _resolve_rounding_policy(
        SimpleNamespace(
            enable_stochastic_rounding=enabled,
            stochastic_rounding_philox_rounds=configured,
        )
    )
    assert policy.enabled is expected_enabled
    assert policy.philox_rounds == expected_rounds


def _validation_stub(
    buffer_len: int,
    algorithm: str = "auto",
    ssm_dtype: str = "float32",
    stochastic_rounding: bool = False,
):
    """Minimal stand-in for the fields _validate_replayssm_spec_flashinfer reads.

    Building a real VllmConfig needs model inspection, which is not available in
    a plain unit-test environment; the surrounding pydantic validator is covered
    by the engine tests.
    """
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            replayssm_buffer_len=buffer_len,
            mamba_ssm_cache_dtype=ssm_dtype,
        ),
        mamba_config=SimpleNamespace(
            replayssm_spec_algorithm=algorithm,
            enable_stochastic_rounding=stochastic_rounding,
        ),
    )


def test_flashinfer_stochastic_rounding_requires_float16_state():
    """The kernel's Philox path is gated on state_t == __half, so SR is only
    meaningful with an fp16 SSM checkpoint."""
    with pytest.raises(ValueError, match="float16"):
        VllmConfig._validate_replayssm_spec_flashinfer(
            _validation_stub(16, ssm_dtype="bfloat16", stochastic_rounding=True), 4
        )


def test_flashinfer_stochastic_rounding_accepted_with_float16_state():
    """Reaches the availability probe, which is the next check after the dtype
    gate; on a box without flashinfer that surfaces as the package error."""
    stub = _validation_stub(16, ssm_dtype="float16", stochastic_rounding=True)
    with pytest.raises(ValueError, match="flashinfer-python package"):
        VllmConfig._validate_replayssm_spec_flashinfer(stub, 4)


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
