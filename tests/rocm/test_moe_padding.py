# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `VLLM_ROCM_MOE_PADDING` weight memory-layout padding trick.

`VLLM_ROCM_MOE_PADDING` (see `UnquantizedFusedMoEMethod._maybe_pad_weight` in
`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`) is a
memory-stride trick, not a logical-shape one: when eligible, it enlarges a
weight tensor's underlying storage (via `F.pad(...)[..., :-num_pad]`) so
per-expert allocations land farther apart in HBM, but the tensor's visible
shape, stride, and values are unaffected. `RoutedExperts.process_weights_after_
loading` immediately copies the padded view's values back into the original
(unpadded-storage) parameter via `.data.copy_()` -- to preserve the parameter's
storage address for CUDA-graph capture -- so the padding never persists on
`w13_weight`/`w2_weight` themselves. This file verifies:

1. `_maybe_pad_weight` in isolation: it only grows storage when the tensor's
   last-but-one dim is 512-byte aligned and the flag is on, and never changes
   the tensor's shape or values either way.
2. End to end: `AiterExperts.apply()` produces numerically identical output
   whether `VLLM_ROCM_MOE_PADDING` is enabled or disabled, for weight shapes
   that are eligible for the padding trick.

See https://github.com/vllm-project/vllm/issues/54966 ("Test padding").
"""

import importlib.util

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="ROCm MoE padding tests require ROCm with AITER installed",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_EXPERTS = 4
TOPK = 2
NUM_TOKENS = 16
# 256 bf16 elements * 2 bytes = 512 bytes: eligible for `_maybe_pad_weight`
# on both w13 (stride(-2) == HIDDEN_SIZE) and w2 (stride(-2) == INTERMEDIATE_SIZE).
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256


def _set_padding_env(monkeypatch: pytest.MonkeyPatch, padding: bool) -> None:
    monkeypatch.setenv("VLLM_ROCM_MOE_PADDING", "1" if padding else "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()


def _make_moe_config(hidden_size: int, intermediate_size: int) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=DTYPE,
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
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    for expert_id in range(NUM_EXPERTS):
        yield (
            f"{expert_id}.gate_proj.weight",
            torch.randn(
                intermediate_size,
                hidden_size,
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.up_proj.weight",
            torch.randn(
                intermediate_size,
                hidden_size,
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.down_proj.weight",
            torch.randn(
                hidden_size,
                intermediate_size,
                generator=generator,
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
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(2026)
    x = torch.randn(
        NUM_TOKENS,
        hidden_size,
        generator=generator,
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
    selected = str(
        getattr(
            layer.quant_method.unquantized_backend,
            "value",
            layer.quant_method.unquantized_backend,
        )
    )
    assert selected == "ROCm AITER", (
        f"expected AiterExperts to be selected, got backend={selected!r}"
    )


# --- 1. `_maybe_pad_weight` in isolation -----------------------------------


@pytest.mark.parametrize(
    "padding,hidden_size,intermediate_size,expect_padded",
    [
        # eligible shape (256 bf16 elems * 2 bytes == 512): padding fires iff
        # the env var is on.
        (True, HIDDEN_SIZE, INTERMEDIATE_SIZE, True),
        (False, HIDDEN_SIZE, INTERMEDIATE_SIZE, False),
        # ineligible shape (200 bf16 elems * 2 bytes == 400, not 512-aligned):
        # padding never fires regardless of the env var.
        (True, 200, 200, False),
    ],
    ids=["eligible-on", "eligible-off", "ineligible-on"],
)
def test_maybe_pad_weight_transparent(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    padding: bool,
    hidden_size: int,
    intermediate_size: int,
    expect_padded: bool,
) -> None:
    """`_maybe_pad_weight` must never change a weight's shape or values, and
    must only grow the tensor's storage when the flag is on AND the tensor's
    last-but-one-dim stride is 512-byte aligned."""
    assert default_vllm_config is not None
    _set_padding_env(monkeypatch, padding)
    assert envs.VLLM_ROCM_MOE_PADDING is padding

    moe_config = _make_moe_config(hidden_size, intermediate_size)
    method = UnquantizedFusedMoEMethod(moe_config)

    original = torch.randn(
        NUM_EXPERTS,
        2 * intermediate_size,
        hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )
    result = method._maybe_pad_weight(original)

    assert result.shape == original.shape
    assert torch.equal(result, original), "padding must not alter weight values"

    original_nbytes = original.untyped_storage().nbytes()
    result_nbytes = result.untyped_storage().nbytes()
    if expect_padded:
        assert result_nbytes > original_nbytes, (
            "expected `_maybe_pad_weight` to grow storage for an eligible "
            "shape with VLLM_ROCM_MOE_PADDING enabled"
        )
        assert result.data_ptr() != original.data_ptr()
    else:
        assert result_nbytes == original_nbytes
        assert result.data_ptr() == original.data_ptr()


# --- 2. End-to-end numerical transparency through AiterExperts -------------


@torch.inference_mode()
def test_aiter_moe_padding_numerically_transparent(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    workspace_init,
) -> None:
    """`AiterExperts.apply()` must produce the same output regardless of
    `VLLM_ROCM_MOE_PADDING`, for a weight shape that is eligible for the
    padding trick -- the flag only changes an intermediate tensor's storage
    layout, never the values a real `RoutedExperts` layer ends up loading."""
    assert default_vllm_config is not None
    assert workspace_init is None

    outputs: dict[bool, torch.Tensor] = {}
    for padding in (True, False):
        _set_padding_env(monkeypatch, padding)

        with torch.device(DEVICE):
            layer = _make_routed_experts(HIDDEN_SIZE, INTERMEDIATE_SIZE)
            _load_and_process_weights(
                layer,
                seed=1,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
            )
        _assert_backend_is_aiter(layer)

        x, topk_weights, topk_ids = _make_static_inputs(HIDDEN_SIZE)
        outputs[padding] = _forward(layer, x, topk_weights, topk_ids).clone()

    torch.testing.assert_close(outputs[True], outputs[False], rtol=2e-2, atol=2e-2)
