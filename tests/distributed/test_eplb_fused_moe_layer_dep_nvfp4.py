# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Test that the interaction between EPLB and FusedMoE Layer is okay for DP w/ NVFP4

from dataclasses import dataclass

import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_dp_group,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)

from .eplb_utils import distributed_run, set_env_vars_and_device


@dataclass
class TestConfig:
    num_layers: int
    num_experts: int
    num_local_experts: int
    num_topk: int
    hidden_size: int
    intermediate_size: int
    num_tokens: int


def make_fused_moe_layer(
    rank: int,
    layer_idx: int,
    test_config: TestConfig,
) -> FusedMoE:
    quant_config = None

    device = torch.device(f"cuda:{rank}")

    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

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

    nvfp4_fused_moe = ModelOptNvFp4FusedMoE(quant_config, fml)
    nvfp4_fused_moe.create_weights(
        fml,
        test_config.num_local_experts,
        test_config.hidden_size,
        test_config.intermediate_size,
        params_dtype=torch.uint8,
        global_num_experts=test_config.num_experts,
    )

    fml = fml.to(device)
    w1_q, w2_q, quant_config = make_test_quant_config(
        test_config.num_local_experts,
        test_config.intermediate_size,
        test_config.hidden_size,
        in_dtype=torch.bfloat16,
        quant_dtype="nvfp4",
        block_shape=None,
        per_act_token_quant=False,
    )

    fml.w13_weight.data = w1_q
    fml.w2_weight.data = w2_q

    fml.w2_input_scale.data = torch.randn_like(fml.w2_input_scale.data) / 5
    fml.w13_input_scale.data = torch.randn_like(fml.w13_input_scale.data) / 5
    fml.w2_weight_scale_2.data = torch.randn_like(fml.w2_weight_scale_2.data) / 5
    fml.w13_weight_scale_2.data = torch.randn_like(fml.w13_weight_scale_2.data) / 5
    fml.w2_weight_scale.data = (
        torch.randn(fml.w2_weight_scale.data.shape, device=device) / 5
    ).to(fml.w2_weight_scale.data.dtype)
    fml.w13_weight_scale.data = (
        torch.randn(fml.w13_weight_scale.data.shape, device=device) / 5
    ).to(fml.w13_weight_scale.data.dtype)

    nvfp4_fused_moe.process_weights_after_loading(fml)

    fml.maybe_init_modular_kernel()

    return fml


def _test_eplb_fml(env, world_size: int, test_config: TestConfig):
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        ep_group = get_dp_group().cpu_group
        ep_rank = torch.distributed.get_rank()

        device = torch.device(f"cuda:{ep_rank}")

        fml_layers = [
            make_fused_moe_layer(ep_rank, layer_idx, test_config).to(device)
            for layer_idx in range(test_config.num_layers)
        ]
        rank_expert_weights = [fml.get_expert_weights() for fml in fml_layers]

        hidden_states = []
        router_logits = []
        for layer_idx in range(test_config.num_layers):
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

        out_before_shuffle = []
        with set_forward_context(
            {},
            num_tokens=test_config.num_tokens,
            num_tokens_across_dp=torch.tensor(
                [test_config.num_tokens] * world_size, device="cpu", dtype=torch.int
            ),
            vllm_config=vllm_config,
        ):
            for lidx, fml in enumerate(fml_layers):
                out_before_shuffle.append(
                    fml(hidden_states[lidx].clone(), router_logits[lidx].clone())
                )

        indices = torch.zeros(
            test_config.num_layers, test_config.num_experts, dtype=torch.long
        )
        for lidx in range(test_config.num_layers):
            indices[lidx] = torch.Tensor(range(test_config.num_experts))

        shuffled_indices = torch.zeros_like(indices)
        for lidx in range(test_config.num_layers):
            shuffled_indices[lidx] = torch.randperm(test_config.num_experts)

        rearrange_expert_weights_inplace(
            indices,
            shuffled_indices,
            rank_expert_weights,
            ep_group,
            is_profile=False,
        )

        num_global_experts = test_config.num_experts

        logical_to_physical_map_list = []
        for lidx, fml in enumerate(fml_layers):
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

        for lidx, fml in enumerate(fml_layers):
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
            )

        out_after_shuffle = []
        with set_forward_context(
            {},
            num_tokens=test_config.num_tokens,
            num_tokens_across_dp=torch.tensor(
                [test_config.num_tokens] * world_size, device="cpu", dtype=torch.int
            ),
            vllm_config=vllm_config,
        ):
            for lidx, fml in enumerate(fml_layers):
                out_after_shuffle.append(
                    fml(hidden_states[lidx].clone(), router_logits[lidx].clone())
                )

        for lidx in range(test_config.num_layers):
            torch.testing.assert_close(
                out_before_shuffle[lidx], out_after_shuffle[lidx], atol=1e-1, rtol=1e-1
            )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_layers", [8])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("num_tokens", [256])
@pytest.mark.parametrize("backend", ["latency", "throughput"])
def test_eplb_fml(
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

    if torch.cuda.device_count() < world_size:
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
        _test_eplb_fml,
        world_size,
        test_config,
    )
