# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay MoE weights after sleep/wake and verify captured graph replay.

This test exercises the failure mode directly at the MoE layer boundary:

1. Build a small ROCm unquantized `RoutedExperts` layer.
2. Load deterministic random expert weights and run post-load processing.
3. Capture a CUDA graph for `RoutedExperts.forward_modular`.
4. Put the CuMem weights pool through level-2 sleep/wake semantics.
5. Replay the same checkpoint-style weights through the layer weight loader.
6. Verify graph replay is deterministic across sleep/wake and weight replay.

The important invariant is that post-load processing must not replace parameter
storage after CUDA graph capture, because captured kernels hold device
addresses for the weight tensors.
"""

import contextlib
import gc
import importlib.util
from collections.abc import Iterable

import pytest
import torch

from vllm.device_allocator.cumem import CuMemAllocator, cumem_available
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
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="MoE replay test requires ROCm with AITER installed",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_EXPERTS = 4
TOPK = 2
NUM_TOKENS = 16
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256


def _moe_topk_softmax_available() -> bool:
    try:
        importlib.import_module("vllm._moe_C_stable_libtorch")
    except ImportError:
        return False
    return hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "topk_softmax")


def _set_backend_env(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("VLLM_ROCM_MOE_PADDING", "1")
    if backend == "aiter":
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    elif backend == "triton":
        monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    else:
        raise AssertionError(f"unknown backend {backend}")

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=DTYPE,
        device=DEVICE,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=NUM_TOKENS,
    )


def _make_expert_map_manager(moe_config: FusedMoEConfig) -> ExpertMapManager:
    return ExpertMapManager(
        max_num_batched_tokens=NUM_TOKENS,
        top_k=TOPK,
        global_num_experts=NUM_EXPERTS,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=moe_config.moe_parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )


def _make_routed_experts() -> RoutedExperts:
    moe_config = _make_moe_config()
    return RoutedExperts(
        "experts",
        DTYPE,
        moe_config,
        quant_config=None,
        expert_map_manager=_make_expert_map_manager(moe_config),
    )


def _expert_weight_iterator(seed: int) -> Iterable[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    for expert_id in range(NUM_EXPERTS):
        yield (
            f"{expert_id}.gate_proj.weight",
            torch.randn(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.up_proj.weight",
            torch.randn(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )
        yield (
            f"{expert_id}.down_proj.weight",
            torch.randn(
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE,
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ),
        )


def _load_expert_weights(layer: RoutedExperts, seed: int) -> set[str]:
    loaded = set(layer.load_weights(_expert_weight_iterator(seed)))
    assert {"w13_weight", "w2_weight"} <= loaded
    return loaded


def _process_weights_after_loading(layer: RoutedExperts) -> None:
    layer.quant_method.process_weights_after_loading(layer)
    torch.accelerator.synchronize()


def _make_static_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(2026)
    x = torch.randn(
        NUM_TOKENS,
        HIDDEN_SIZE,
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


def _capture_forward_graph(
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    # Warm up kernel dispatch/JIT before capture so the captured region is only
    # the steady-state forward path.
    for _ in range(3):
        _forward(layer, x, topk_weights, topk_ids)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = _forward(layer, x, topk_weights, topk_ids)
    torch.accelerator.synchronize()
    return graph, graph_output


def _weight_pointers(layer: RoutedExperts) -> dict[str, int]:
    return {
        "w13_weight": layer.w13_weight.data_ptr(),
        "w2_weight": layer.w2_weight.data_ptr(),
    }


def _assert_backend(layer: RoutedExperts, backend: str) -> None:
    selected = str(
        getattr(
            layer.quant_method.unquantized_backend,
            "value",
            layer.quant_method.unquantized_backend,
        )
    )
    if backend == "aiter":
        assert selected == "ROCm AITER"
    else:
        assert selected == "TRITON"


@pytest.mark.parametrize("backend", ["triton", "aiter"])
@torch.inference_mode()
def test_moe_cuda_graph_replay_after_sleep_wake_weight_update(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    workspace_init,
    backend: str,
) -> None:
    assert default_vllm_config is not None
    assert workspace_init is None
    if not cumem_available:
        pytest.skip("sleep mode replay requires the cumem allocator extension")
    if not _moe_topk_softmax_available():
        pytest.skip("MoE C extension with topk_softmax is required")

    _set_backend_env(monkeypatch, backend)

    allocator = CuMemAllocator.get_instance()
    layer = graph = graph_output = None
    try:
        with allocator.use_memory_pool("weights"), torch.device(DEVICE):
            layer = _make_routed_experts()
            _load_expert_weights(layer, seed=1)
            _process_weights_after_loading(layer)
        _assert_backend(layer, backend)

        x, topk_weights, topk_ids = _make_static_inputs()
        graph, graph_output = _capture_forward_graph(layer, x, topk_weights, topk_ids)
        captured_pointers = _weight_pointers(layer)

        graph.replay()
        torch.accelerator.synchronize()
        first_graph_output = graph_output.clone()

        # Match vLLM level-2 sleep semantics: weights are discarded, then the
        # same virtual addresses are remapped on wake and refilled by replay.
        allocator.sleep(offload_tags=tuple())
        allocator.wake_up()
        _load_expert_weights(layer, seed=1)
        _process_weights_after_loading(layer)

        assert _weight_pointers(layer) == captured_pointers

        graph_output.zero_()
        graph.replay()
        torch.accelerator.synchronize()

        torch.testing.assert_close(
            graph_output, first_graph_output, rtol=2e-2, atol=2e-2
        )
    finally:
        del graph_output
        del graph
        del layer
        gc.collect()
        with contextlib.suppress(Exception):
            allocator.release_pools()
        torch.accelerator.empty_cache()
