# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU tests for the after_eplb_rearrangement hook on NVFP4 MoE."""

from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_state import _run_after_eplb_rearrangement_hooks
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.platforms import current_platform

EPLB_NVFP4_BACKENDS = ["flashinfer_cutedsl", "flashinfer_trtllm"]

NUM_EXPERTS = 8
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
EXPERT_PERMUTATION = [3, 0, 5, 1, 7, 2, 6, 4]

DERIVED_SCALE_NAMES = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)


class _RoutedExpertsStub(torch.nn.Module):
    """Minimal ``RoutedExperts`` stand-in for NVFP4 weight processing."""

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        super().__init__()
        self.moe_config = moe_config
        self.activation = moe_config.activation

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return None


def _make_moe_config(backend: str) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=True,
    )
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=2,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
    )


def _make_raw_weights(
    device: torch.device, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Random raw NVFP4 checkpoint tensors, indexable per-expert on dim 0."""
    e, h, i = NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device=device, generator=generator
        )

    def block_scale(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 128, shape, dtype=torch.uint8, device=device, generator=generator
        ).view(torch.float8_e4m3fn)

    def global_scale(*shape: int) -> torch.Tensor:
        return torch.rand(shape, device=device, generator=generator) + 0.5

    return {
        "w13_weight_packed": packed(e, 2 * i, h // 2),
        "w2_weight_packed": packed(e, h, i // 2),
        "w13_weight_scale": block_scale(e, 2 * i, h // 16),
        "w2_weight_scale": block_scale(e, h, i // 16),
        "w13_weight_global_scale": global_scale(e, 2),
        "w2_weight_global_scale": global_scale(e),
        "w13_input_global_scale": global_scale(e, 2),
        "w2_input_global_scale": global_scale(e),
    }


def _build_processed_layer(
    backend: str, raw: dict[str, torch.Tensor], device: torch.device
) -> tuple[CompressedTensorsW4A4Nvfp4MoEMethod, _RoutedExpertsStub]:
    """Create the method + layer, load ``raw`` and run the real
    process_weights_after_loading (which builds the flashinfer kernel)."""
    moe_config = _make_moe_config(backend)
    method = CompressedTensorsW4A4Nvfp4MoEMethod(moe_config, "layer.0", use_a16=False)
    layer = _RoutedExpertsStub(moe_config).to(device)
    method.create_weights(
        layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer = layer.to(device)
    with torch.no_grad():
        for name, value in raw.items():
            getattr(layer, name).copy_(value)
    method.process_weights_after_loading(layer)
    return method, layer


def _assert_tensors_equal(
    actual: torch.Tensor, expected: torch.Tensor, name: str
) -> None:
    if actual.dtype == torch.float8_e4m3fn:
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name
    elif actual.dtype == torch.uint8:
        assert torch.equal(actual, expected), name
    else:
        torch.testing.assert_close(actual, expected, msg=lambda m: f"{name}: {m}")


def _iter_derived_scales(
    layer: _RoutedExpertsStub,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name in DERIVED_SCALE_NAMES:
        yield name, getattr(layer, name)


@pytest.mark.parametrize("backend", EPLB_NVFP4_BACKENDS)
def test_nvfp4_after_eplb_rearrangement_matches_reload(backend: str) -> None:
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 CuteDSL/TRTLLM MoE backends require Blackwell (SM100).")

    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device)
    perm = torch.tensor(EXPERT_PERMUTATION, device=device)

    with set_current_vllm_config(VllmConfig()):
        generator = torch.Generator(device=device).manual_seed(1234)
        raw = _make_raw_weights(device, generator)
        raw_permuted = {name: value[perm].contiguous() for name, value in raw.items()}

        try:
            method, routed_experts = _build_processed_layer(backend, raw, device)
        except ValueError as exc:
            pytest.skip(f"{backend} NVFP4 MoE backend unavailable: {exc}")
        _, reloaded = _build_processed_layer(backend, raw_permuted, device)

        before = {n: t.clone() for n, t in _iter_derived_scales(routed_experts)}
        method.after_eplb_rearrangement(routed_experts)
        for name, tensor in _iter_derived_scales(routed_experts):
            _assert_tensors_equal(before[name], tensor, f"hook not a no-op: {name}")

        # Simulate EPLB rearrangement.
        with torch.no_grad():
            for _, param in routed_experts.named_parameters():
                param.copy_(param[perm])
        # quant_method lives on the RoutedExperts; EplbState reaches it via
        # moe_layer.routed_experts.quant_method (a MoERunner has no public one).
        routed_experts.quant_method = method
        assert not hasattr(MoERunner, "quant_method")
        moe_layer = SimpleNamespace(routed_experts=routed_experts)
        assert not hasattr(moe_layer, "quant_method")
        model = SimpleNamespace(moe_layers=[moe_layer])
        _run_after_eplb_rearrangement_hooks(model)

        # After rearrangement, the state must equal what we would get by
        # loading the permuted experts directly.
        ref_params = dict(routed_experts.named_parameters())
        new_params = dict(reloaded.named_parameters())
        assert ref_params.keys() == new_params.keys()
        for name in ref_params:
            _assert_tensors_equal(ref_params[name], new_params[name], name)
        for name, tensor in _iter_derived_scales(routed_experts):
            _assert_tensors_equal(tensor, getattr(reloaded, name), name)
