# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `VLLM_ROCM_MOE_PADDING` weight memory-layout padding trick.

`VLLM_ROCM_MOE_PADDING` (see `UnquantizedFusedMoEMethod._maybe_pad_weight`)
is a memory-stride trick, not a logical-shape one: when eligible, it enlarges
a weight tensor's storage so per-expert allocations land farther apart in
HBM. Shape and values are unaffected, but `stride(-2)` increases by
`num_pad` elements -- that's the whole mechanism. For the **AITER** path
tested here, the padded storage never persists on `w13_weight`/`w2_weight`:
`convert_to_unquantized_kernel_format` reallocates via
`rocm_aiter_ops.shuffle_weights`. This is AITER-specific -- on TRITON-on-ROCm,
`.contiguous()` is skipped specifically to keep this same padding, so its
persisted parameters *do* stay padded.

This file verifies:
1. `_maybe_pad_weight` in isolation (any ROCm backend, AITER not required):
   grows storage/`stride(-2)` only when eligible (512-byte-aligned) and the
   flag is on -- covering no padding, hidden-only, intermediate-only, and
   both-dimensions-padded, since w13 and w2 key their alignment gate off
   different dims; a true no-op (identical object) otherwise.
2. End to end, AITER only (per #54966's "Test padding" ask):
   `AiterExperts.apply()` matches an independent reference regardless of
   the flag, and persisted parameter storage size is unaffected by it.

Logical `hidden_dim_unpadded`/`intermediate_size_per_partition_unpadded`
padding and HIP-graph token-padding are separate mechanisms, covered in
#59333 and #59334.

See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").
"""

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

aiter_available = is_aiter_found_and_supported()

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="VLLM_ROCM_MOE_PADDING only takes effect on ROCm",
)
requires_aiter = pytest.mark.skipif(
    not aiter_available,
    reason="requires AITER to exercise AiterExperts.apply()",
)

DEVICE = current_platform.device_type
DTYPE = torch.bfloat16
NUM_EXPERTS = 4
TOPK = 2
NUM_TOKENS = 16
# Mirrors `_maybe_pad_weight`'s `num_pad` constant in unquantized_fused_moe_method.py.
PAD_BYTES = 256
# 256 bf16 elems * 2 bytes = 512 bytes: eligible for `_maybe_pad_weight` on
# both w13 and w2.
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256


def _set_padding_env(monkeypatch: pytest.MonkeyPatch, padding: bool) -> None:
    """Sets the flag only; `_maybe_pad_weight` reads it lazily, no AITER needed."""
    monkeypatch.setenv("VLLM_ROCM_MOE_PADDING", "1" if padding else "0")


def _set_padding_env_with_aiter(monkeypatch: pytest.MonkeyPatch, padding: bool) -> None:
    """Also enables AITER, for tests routing through RoutedExperts's backend
    selection."""
    _set_padding_env(monkeypatch, padding)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()


def _make_moe_config(
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype = DTYPE,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=dtype,
        device=DEVICE,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=NUM_TOKENS,
    )


def _make_routed_experts(hidden_size: int, intermediate_size: int) -> RoutedExperts:
    moe_config = _make_moe_config(hidden_size, intermediate_size)
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=NUM_TOKENS,
        top_k=TOPK,
        global_num_experts=NUM_EXPERTS,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=moe_config.moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )
    return RoutedExperts(
        "experts",
        DTYPE,
        moe_config,
        quant_config=None,
        expert_map_manager=expert_map_manager,
    )


def _expert_weight_iterator(seed: int, hidden_size: int, intermediate_size: int):
    set_random_seed(seed)
    for expert_id in range(NUM_EXPERTS):
        yield (
            f"{expert_id}.gate_proj.weight",
            torch.randn(
                intermediate_size,
                hidden_size,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.up_proj.weight",
            torch.randn(
                intermediate_size,
                hidden_size,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.down_proj.weight",
            torch.randn(
                hidden_size,
                intermediate_size,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )


def _load_and_process_weights(
    layer: RoutedExperts, seed: int, hidden_size: int, intermediate_size: int
) -> None:
    loaded = set(
        layer.load_weights(
            _expert_weight_iterator(seed, hidden_size, intermediate_size)
        )
    )
    assert {"w13_weight", "w2_weight"} <= loaded
    layer.quant_method.process_weights_after_loading(layer)
    torch.accelerator.synchronize()


def _make_static_inputs(
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_random_seed(2026)
    x = torch.randn(
        NUM_TOKENS,
        hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )
    topk_ids = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    topk_ids = torch.cat(
        [topk_ids % NUM_EXPERTS, (topk_ids + 1) % NUM_EXPERTS],
        dim=1,
    )
    topk_weights = torch.full(
        (NUM_TOKENS, TOPK),
        1.0 / TOPK,
        device=DEVICE,
        dtype=DTYPE,
    )
    return x, topk_weights, topk_ids


def _forward(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return layer.forward_modular(
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )


def _assert_backend_is_aiter(layer: RoutedExperts) -> None:
    selected = layer.quant_method.unquantized_backend
    assert selected is UnquantizedMoeBackend.AITER, (
        f"expected AiterExperts to be selected, got backend={selected!r}"
    )


# --- 1. `_maybe_pad_weight` in isolation -----------------------------------


@pytest.mark.parametrize(
    "padding,hidden_size,intermediate_size,dtype,expect_padded",
    [
        # eligible shape (512 bytes): padding fires iff the env var is on.
        (True, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, True),
        (False, HIDDEN_SIZE, INTERMEDIATE_SIZE, DTYPE, False),
        # ineligible shape (400 bytes, not 512-aligned): never fires.
        (True, 200, 200, DTYPE, False),
        # fp32: alignment threshold and num_pad both scale with
        # element_size() (128 fp32 elems * 4 bytes == 512).
        (True, 128, 128, torch.float32, True),
    ],
    ids=["eligible-on", "eligible-off", "ineligible-on", "eligible-on-fp32"],
)
def test_maybe_pad_weight_transparent(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    padding: bool,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    expect_padded: bool,
) -> None:
    """Shape/values never change; storage only grows (`stride(-2) += num_pad`)
    when eligible and the flag is on. Otherwise it's a true no-op (identical
    object)."""
    assert default_vllm_config is not None
    _set_padding_env(monkeypatch, padding)
    assert envs.VLLM_ROCM_MOE_PADDING is padding

    moe_config = _make_moe_config(hidden_size, intermediate_size, dtype=dtype)
    method = UnquantizedFusedMoEMethod(moe_config)

    original = torch.randn(
        NUM_EXPERTS,
        2 * intermediate_size,
        hidden_size,
        device=DEVICE,
        dtype=dtype,
    )
    result = method._maybe_pad_weight(original)

    assert result.shape == original.shape
    assert torch.equal(result, original), "padding must not alter weight values"

    if expect_padded:
        num_pad = PAD_BYTES // original.element_size()
        assert result.stride(-1) == 1
        assert result.stride(-2) == original.stride(-2) + num_pad
        assert result.untyped_storage().nbytes() > original.untyped_storage().nbytes()
        assert result.data_ptr() != original.data_ptr()
    else:
        assert result is original


@pytest.mark.parametrize(
    "hidden_size,intermediate_size,expect_w13_padded,expect_w2_padded",
    [
        # only hidden_size aligned: only w13 (last dim hidden_size) eligible.
        (HIDDEN_SIZE, 200, True, False),
        # only intermediate_size aligned: only w2 (last dim
        # intermediate_size) eligible.
        (200, HIDDEN_SIZE, False, True),
    ],
    ids=["only-w13-eligible", "only-w2-eligible"],
)
def test_maybe_pad_weight_asymmetric_w13_w2_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    hidden_size: int,
    intermediate_size: int,
    expect_w13_padded: bool,
    expect_w2_padded: bool,
) -> None:
    """w13 and w2 key their alignment gate off different dims (hidden_size
    vs. intermediate_size), so they're independently eligible."""
    assert default_vllm_config is not None
    _set_padding_env(monkeypatch, True)

    moe_config = _make_moe_config(hidden_size, intermediate_size)
    method = UnquantizedFusedMoEMethod(moe_config)

    w13 = torch.randn(
        NUM_EXPERTS, 2 * intermediate_size, hidden_size, device=DEVICE, dtype=DTYPE
    )
    w2 = torch.randn(
        NUM_EXPERTS, hidden_size, intermediate_size, device=DEVICE, dtype=DTYPE
    )

    w13_result = method._maybe_pad_weight(w13)
    w2_result = method._maybe_pad_weight(w2)

    assert (w13_result is not w13) == expect_w13_padded
    assert (w2_result is not w2) == expect_w2_padded


def _reference_moe_forward(
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """A minimal, independent (no AITER) reference MoE forward, so the E2E
    test below can't pass merely because both flag settings are wrong the
    same way."""
    weights: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in _expert_weight_iterator(seed, hidden_size, intermediate_size):
        expert_id_str, proj_name = name.split(".", 1)
        weights.setdefault(int(expert_id_str), {})[proj_name] = tensor

    output = torch.zeros_like(x)
    for expert_id, proj in weights.items():
        gate = F.silu(x @ proj["gate_proj.weight"].T)
        up = x @ proj["up_proj.weight"].T
        expert_out = (gate * up) @ proj["down_proj.weight"].T

        mask = topk_ids == expert_id
        weight = (topk_weights * mask).sum(dim=1, keepdim=True)
        output = output + weight * expert_out
    return output


# --- 2. End-to-end numerical transparency through AiterExperts -------------


@requires_aiter
@torch.inference_mode()
def test_aiter_moe_padding_numerically_transparent(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    workspace_init,
) -> None:
    """`AiterExperts.apply()` output must match an independent reference
    (not just itself across flag settings) for an eligible weight shape,
    regardless of `VLLM_ROCM_MOE_PADDING`. Persisted parameter storage size
    must also be unaffected by the flag (see module docstring)."""
    assert default_vllm_config is not None
    assert workspace_init is None

    outputs: dict[bool, torch.Tensor] = {}
    storage_nbytes: dict[bool, tuple[int, int]] = {}
    for padding in (True, False):
        _set_padding_env_with_aiter(monkeypatch, padding)

        with torch.device(DEVICE):
            layer = _make_routed_experts(HIDDEN_SIZE, INTERMEDIATE_SIZE)
            _load_and_process_weights(
                layer,
                seed=1,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
            )
        _assert_backend_is_aiter(layer)
        storage_nbytes[padding] = (
            layer.w13_weight.data.untyped_storage().nbytes(),
            layer.w2_weight.data.untyped_storage().nbytes(),
        )

        x, topk_weights, topk_ids = _make_static_inputs(HIDDEN_SIZE)
        outputs[padding] = _forward(layer, x, topk_weights, topk_ids).clone()

    assert storage_nbytes[True] == storage_nbytes[False], (
        "VLLM_ROCM_MOE_PADDING must not change the persisted parameters' "
        "storage size for the AITER path"
    )

    reference = _reference_moe_forward(
        seed=1,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    # Same AITER kernel, byte-identical weights and inputs either way (only
    # a transient weight-loading buffer differs): bitwise-identical results.
    torch.testing.assert_close(outputs[True], outputs[False], atol=0, rtol=0)

    # Cosine similarity, not elementwise assert_close, for the reference
    # comparison: two independently computed bf16 GEMM chains can have
    # near-zero elements where rounding noise blows up elementwise relative
    # error. Same pattern as test_b12x.py / test_mxfp4_moe.py.
    for padded, output in outputs.items():
        cos_sim = F.cosine_similarity(
            output.flatten().float(), reference.flatten().float(), dim=0
        )
        assert cos_sim > 0.99, (
            f"padding={padded} output diverged from reference "
            f"(cosine similarity {cos_sim:.4f})"
        )
