# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test Mooncake EP dispatch/combine and modular MoE execution."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.moe.utils import make_test_weights
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.mooncake_ep import (
    MooncakeEPPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.utils.import_utils import has_mooncake_ep
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

requires_mooncake_ep = pytest.mark.skipif(
    not has_mooncake_ep(), reason="Requires the Mooncake EP package"
)


@dataclasses.dataclass
class TestConfig:
    m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int


def _make_mooncake_prepare_finalize(
    pg: ProcessGroup, pgi: ProcessGroupInfo, config: TestConfig
) -> MooncakeEPPrepareAndFinalize:
    from mooncake.mooncake_ep_buffer import Buffer

    max_tokens_per_rank = config.m
    buffer_size = Buffer.get_ep_buffer_size_hint(
        max_tokens_per_rank,
        config.hidden_size,
        pgi.world_size,
        config.num_experts,
    )
    return MooncakeEPPrepareAndFinalize(
        Buffer(pg, buffer_size),
        max_tokens_per_rank=max_tokens_per_rank,
        num_dispatchers=pgi.world_size,
    )


def _local_expert_map(
    pgi: ProcessGroupInfo, num_experts: int
) -> torch.Tensor:
    num_local_experts = num_experts // pgi.world_size
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    start = pgi.rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device="cuda"
    )
    return expert_map


def _reference_local_experts(
    expert_x: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    expert_out = torch.zeros_like(expert_x)
    for expert_id in range(w1.size(0)):
        num_tokens = int(expert_num_tokens[expert_id].item())
        if num_tokens:
            x = expert_x[expert_id, :num_tokens]
            gate_up = F.linear(x, w1[expert_id])
            y = F.silu(gate_up[..., : gate_up.size(-1) // 2]) * gate_up[
                ..., gate_up.size(-1) // 2 :
            ]
            expert_out[expert_id, :num_tokens] = F.linear(y, w2[expert_id])
    return expert_out


def _run_mooncake(
    pgi: ProcessGroupInfo,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    use_modular_kernel: bool,
) -> None:
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(pgi.local_rank)
    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)
    w1 = w1.to(device=device)
    w2 = w2.to(device=device)
    num_local_experts = config.num_experts // pgi.world_size
    start = pgi.rank * num_local_experts
    local_w1 = w1[start : start + num_local_experts]
    local_w2 = w2[start : start + num_local_experts]

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    tokens = torch.randn(
        (config.m, config.hidden_size), device=device, dtype=torch.bfloat16
    ) / 10
    logits = torch.randn(
        config.m, config.num_experts, device=device, dtype=torch.float32
    )
    topk_weights, topk_ids = torch.topk(logits, config.topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    with set_current_vllm_config(VllmConfig()):
        expected = torch_experts(
            tokens,
            w1,
            w2,
            topk_weights,
            topk_ids,
            apply_router_weights_on_input=False,
        )
        prepare_finalize = _make_mooncake_prepare_finalize(pg, pgi, config)
        if use_modular_kernel:
            moe_config = make_dummy_moe_config(
                num_experts=config.num_experts,
                num_local_experts=num_local_experts,
                experts_per_token=config.topk,
                hidden_dim=config.hidden_size,
                intermediate_size=config.intermediate_size,
                max_num_tokens=config.m,
            )
            experts = BatchedTritonExperts(
                max_num_tokens=config.m,
                num_dispatchers=pgi.world_size,
                moe_config=moe_config,
                quant_config=prepare_finalize_quant_config(),
            )
            kernel = FusedMoEKernel(prepare_finalize, experts)
            actual = kernel.apply(
                tokens,
                local_w1,
                local_w2,
                topk_weights,
                topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=config.num_experts,
                expert_map=_local_expert_map(pgi, config.num_experts),
                apply_router_weight_on_input=False,
            )
        else:
            expert_x, _, expert_tokens, _, _ = prepare_finalize.prepare(
                tokens,
                topk_weights,
                topk_ids,
                config.num_experts,
                None,
                False,
                prepare_finalize_quant_config(),
                False,
            )
            expert_out = _reference_local_experts(
                expert_x, expert_tokens.expert_num_tokens, local_w1, local_w2
            )
            actual = torch.empty_like(tokens)
            prepare_finalize.finalize(
                actual,
                expert_out,
                topk_weights,
                topk_ids,
                False,
                TopKWeightAndReduceDelegate(),
            )

    torch.testing.assert_close(expected, actual, atol=6e-2, rtol=6e-2)


def prepare_finalize_quant_config():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    return FusedMoEQuantConfig.make(quant_dtype=None)


@pytest.mark.parametrize("m", [1, 37, 100])
@pytest.mark.parametrize("use_modular_kernel", [False, True])
@multi_gpu_test(num_gpus=2)
@requires_mooncake_ep
def test_mooncake_ep_bf16_moe(m: int, use_modular_kernel: bool):
    set_random_seed(7)
    config = TestConfig(
        m=m,
        hidden_size=2048,
        intermediate_size=4096,
        num_experts=8,
        topk=4,
    )
    (_, w1, _, _), (_, w2, _, _) = make_test_weights(
        config.num_experts, config.intermediate_size, config.hidden_size
    )
    parallel_launch(2, _run_mooncake, config, w1, w2, use_modular_kernel)
