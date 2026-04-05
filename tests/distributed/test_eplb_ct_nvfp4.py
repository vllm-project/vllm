# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test EPLB (Expert-Level Load Balancing) through the compressed tensors
NVFp4 path (CompressedTensorsW4A4Nvfp4MoEMethod).

Verifies that shuffling expert weights via rearrange_expert_weights_inplace
preserves model output when the logical→physical mapping is updated
accordingly.  Uses data-parallel size as the entry point for expert parallelism.
"""

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
)
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_dp_group,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)

from .eplb_utils import distributed_run, set_env_vars_and_device


@dataclass(frozen=True)
class TestConfig:
    num_layers: int
    num_experts: int
    num_local_experts: int
    num_topk: int
    hidden_size: int
    intermediate_size: int
    num_tokens: int


def _make_nvfp4_quant_config() -> CompressedTensorsConfig:
    """
    Build a minimal CompressedTensorsConfig whose scheme triggers the
    CompressedTensorsW4A4Nvfp4MoEMethod path (W4A4 nvfp4).
    """
    nvfp4_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
        symmetric=True,
        dynamic=False,
    )

    # The target_scheme_map must cover the FusedMoE layer.  Using "Linear"
    # as the key mirrors real checkpoint configs; CompressedTensorsConfig
    # will copy it to "FusedMoE" internally when get_quant_method is called.
    target_scheme_map: dict[str, Any] = {
        "Linear": {
            "weights": nvfp4_args,
            "input_activations": nvfp4_args,
            "format": "float-quantized",
        },
    }

    return CompressedTensorsConfig(
        target_scheme_map=target_scheme_map,
        ignore=[],
        quant_format="float-quantized",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )


def make_fused_moe_layer(
    rank: int,
    layer_idx: int,
    test_config: TestConfig,
) -> FusedMoE:
    """
    Construct a FusedMoE layer quantised with compressed-tensors NVFp4 and
    fill all expert tensors with reproducible random data.
    """
    device = torch.device(f"cuda:{rank}")
    quant_config = _make_nvfp4_quant_config()

    fml = FusedMoE(
        num_experts=test_config.num_experts,
        top_k=test_config.num_topk,
        hidden_size=test_config.hidden_size,
        intermediate_size=test_config.intermediate_size,
        prefix=f"dummy_layer_{layer_idx}",
        activation="silu",
        is_act_and_mul=True,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
    )

    # create_weights is called automatically via get_quant_method →
    # CompressedTensorsW4A4Nvfp4MoEMethod → create_weights during __init__
    # Move everything to the target device.
    fml = fml.to(device)

    # --- Fill packed weights and scales with reproducible random data ---
    # seed per (layer, rank) for determinism across processes
    gen = torch.Generator(device=device)
    gen.manual_seed(42 + layer_idx * 1000 + rank)

    # Packed uint8 weights
    fml.w13_weight_packed.data = torch.randint(
        0,
        256,
        fml.w13_weight_packed.data.shape,
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )
    fml.w2_weight_packed.data = torch.randint(
        0,
        256,
        fml.w2_weight_packed.data.shape,
        dtype=torch.uint8,
        device=device,
        generator=gen,
    )

    # Per-block weight scales (fp8)
    fml.w13_weight_scale.data = (
        torch.randn(
            fml.w13_weight_scale.data.shape,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        .abs()
        .clamp(min=0.01)
        .to(torch.float8_e4m3fn)
    )
    fml.w2_weight_scale.data = (
        torch.randn(
            fml.w2_weight_scale.data.shape,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        .abs()
        .clamp(min=0.01)
        .to(torch.float8_e4m3fn)
    )

    # Global weight scales
    fml.w13_weight_global_scale.data = torch.rand(
        fml.w13_weight_global_scale.data.shape,
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).clamp(min=0.01)
    fml.w2_weight_global_scale.data = torch.rand(
        fml.w2_weight_global_scale.data.shape,
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).clamp(min=0.01)

    # Global input scales
    fml.w13_input_global_scale.data = torch.rand(
        fml.w13_input_global_scale.data.shape,
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).clamp(min=0.01)
    fml.w2_input_global_scale.data = torch.rand(
        fml.w2_input_global_scale.data.shape,
        device=device,
        dtype=torch.float32,
        generator=gen,
    ).clamp(min=0.01)

    # process_weights_after_loading converts packed weights into kernel
    # format and builds the modular kernel.
    fml.quant_method.process_weights_after_loading(fml)

    # Initialize the modular kernel (may be a no-op if process_weights
    # already created it internally).
    fml.maybe_init_modular_kernel()

    return fml


def _test_eplb_ct_nvfp4(env: dict[str, str], world_size: int, test_config: TestConfig):
    """Worker function executed by every rank."""
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        ep_group = get_dp_group().cpu_group
        ep_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{ep_rank}")

        fused_moe_layers = [
            make_fused_moe_layer(ep_rank, layer_idx, test_config).to(device)
            for layer_idx in range(test_config.num_layers)
        ]
        rank_expert_weights = [fml.get_expert_weights() for fml in fused_moe_layers]

        hidden_states = []
        router_logits = []
        for _ in range(test_config.num_layers):
            hidden_states.append(
                torch.randn(
                    (test_config.num_tokens, test_config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
            router_logits.append(
                torch.randn(
                    (test_config.num_tokens, test_config.num_experts),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )

        # ---- Forward BEFORE shuffle ----
        out_before_shuffle = []
        with set_forward_context(
            {},
            num_tokens=test_config.num_tokens,
            num_tokens_across_dp=torch.tensor(
                [test_config.num_tokens] * world_size,
                device="cpu",
                dtype=torch.int,
            ),
            vllm_config=vllm_config,
        ):
            for lidx, fml in enumerate(fused_moe_layers):
                out_before_shuffle.append(
                    fml(
                        hidden_states[lidx].clone(),
                        router_logits[lidx].clone(),
                    )
                )

        # ---- Build identity → shuffled mapping ----
        num_global_experts = test_config.num_experts
        indices = torch.zeros(
            test_config.num_layers, num_global_experts, dtype=torch.long
        )
        for lidx in range(test_config.num_layers):
            indices[lidx] = torch.arange(num_global_experts)

        shuffled_indices = torch.zeros_like(indices)
        for lidx in range(test_config.num_layers):
            shuffled_indices[lidx] = torch.randperm(num_global_experts)

        # ---- Rearrange expert weights via EPLB ----
        rearrange_expert_weights_inplace(
            indices,
            shuffled_indices,
            rank_expert_weights,
            ep_group,
            is_profile=False,
        )

        # ---- Build logical→physical map and set EPLB state ----
        logical_to_physical_map_list = []
        for lidx in range(test_config.num_layers):
            physical_to_logical_map = shuffled_indices[lidx].to(device)
            logical_to_physical_map = torch.empty(
                (num_global_experts,), dtype=torch.int32, device=device
            )
            logical_to_physical_map[physical_to_logical_map] = torch.arange(
                0, num_global_experts, dtype=torch.int32, device=device
            )
            logical_to_physical_map_list.append(
                logical_to_physical_map.reshape(num_global_experts, 1)
            )

        logical_to_physical_map = torch.stack(logical_to_physical_map_list)

        should_record = torch.ones((), dtype=torch.bool, device=device)
        for lidx, fml in enumerate(fused_moe_layers):
            logical_replica_count = torch.ones(
                (test_config.num_layers, num_global_experts),
                dtype=torch.int32,
                device=device,
            )
            fml.enable_eplb = True
            fml.set_eplb_state(
                lidx,
                torch.zeros(
                    (test_config.num_layers, num_global_experts),
                    dtype=torch.int32,
                    device=device,
                ),
                logical_to_physical_map,
                logical_replica_count,
                should_record,
            )

        # ---- Recompute derived quant state after rearrangement ----
        for fml in fused_moe_layers:
            fml.post_weight_rearrangement()

        # ---- Forward AFTER shuffle ----
        out_after_shuffle = []
        with set_forward_context(
            {},
            num_tokens=test_config.num_tokens,
            num_tokens_across_dp=torch.tensor(
                [test_config.num_tokens] * world_size,
                device="cpu",
                dtype=torch.int,
            ),
            vllm_config=vllm_config,
        ):
            for lidx, fml in enumerate(fused_moe_layers):
                out_after_shuffle.append(
                    fml(
                        hidden_states[lidx].clone(),
                        router_logits[lidx].clone(),
                    )
                )

        # ---- Assert outputs match ----
        for lidx in range(test_config.num_layers):
            torch.testing.assert_close(
                out_before_shuffle[lidx],
                out_after_shuffle[lidx],
                atol=1e-1,
                rtol=1e-1,
            )


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_layers", [8])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("intermediate_size", [2048])
@pytest.mark.parametrize("num_tokens", [256])
@pytest.mark.parametrize("backend", ["latency", "throughput"])
def test_eplb_ct_nvfp4(
    world_size: int,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    backend: str,
    monkeypatch,
):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", backend)

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    num_local_experts = num_experts // world_size
    num_topk = 4

    test_config = TestConfig(
        num_layers=num_layers,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        num_topk=num_topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_tokens=num_tokens,
    )

    distributed_run(
        _test_eplb_ct_nvfp4,
        world_size,
        test_config,
    )
