# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.v1.worker.workspace import init_workspace_manager

from .modular_kernel_tools.common import Config, make_modular_kernel
from .modular_kernel_tools.mk_objects import TestMoEQuantConfig
from .modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)


def _target_scale_reference(
    x: torch.Tensor, a1_scale: float, a2_scale: float
) -> torch.Tensor:
    a1 = torch.tensor(a1_scale, device=x.device)
    a2 = torch.tensor(a2_scale, device=x.device)
    a1_q = torch.round(x.float() / a1).clamp(-128, 127)
    w13 = (a1_q * a1).to(x.dtype)
    activated = (F.silu(w13) * w13).to(x.dtype)
    a2_q = torch.round(activated.float() / a2).clamp(-128, 127)
    return (a2_q * a2).to(x.dtype)


def _worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    _cpu_group,
    config: Config,
) -> None:
    init_workspace_manager(pgi.device)
    hidden = config.K
    target_a1_scales = (0.25, 0.5)
    target_a2_scales = (0.125, 0.25)

    eye = torch.eye(hidden, device=pgi.device, dtype=torch.int8)
    w1 = torch.cat((eye, eye)).unsqueeze(0)
    w2 = eye.unsqueeze(0)
    quant_config = FusedMoEQuantConfig.make(
        torch.int8,
        w1_scale=torch.ones(1, device=pgi.device),
        w2_scale=torch.ones(1, device=pgi.device),
        a1_scale=torch.tensor(target_a1_scales[pgi.rank], device=pgi.device),
        a2_scale=torch.tensor(target_a2_scales[pgi.rank], device=pgi.device),
        per_act_token_quant=False,
        per_out_ch_quant=False,
    )
    kernel = make_modular_kernel(config, vllm_config, quant_config)
    assert kernel.fused_experts.expects_unquantized_inputs

    expert_map = torch.full((2,), -1, dtype=torch.int32, device=pgi.device)
    expert_map[pgi.rank] = 0

    def run(token_counts: tuple[int, int], target_expert: int) -> torch.Tensor:
        num_tokens = token_counts[pgi.rank]
        x = torch.full(
            (num_tokens, hidden), 0.7, device=pgi.device, dtype=torch.bfloat16
        )
        ids = torch.full(
            (num_tokens, 1), target_expert, device=pgi.device, dtype=torch.int32
        )
        weights = torch.ones((num_tokens, 1), device=pgi.device)
        counts = torch.tensor(token_counts, dtype=torch.int32)
        torch.distributed.barrier()
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=counts,
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            with dp_metadata.sp_local_sizes(sequence_parallel_size=1, use_ep=True):
                return kernel.apply(
                    hidden_states=x,
                    w1=w1,
                    w2=w2,
                    topk_weights=weights,
                    topk_ids=ids,
                    activation=MoEActivation.SILU,
                    global_num_experts=2,
                    expert_map=expert_map,
                    apply_router_weight_on_input=False,
                )

    # Rank 0 has no local input but must still receive and execute rank 1's token.
    out = run((0, 1), target_expert=0)
    if pgi.rank == 0:
        assert out.shape == (0, hidden)
    else:
        expected = _target_scale_reference(
            torch.full_like(out, 0.7), target_a1_scales[0], target_a2_scales[0]
        )
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)

    # Both ranks route to the peer. Each result must use the target rank's
    # distinct static scales, not the sender rank's scales.
    target = 1 - pgi.rank
    out = run((1, 1), target_expert=target)
    expected = _target_scale_reference(
        torch.full_like(out, 0.7),
        target_a1_scales[target],
        target_a2_scales[target],
    )
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_static_int8_naive_dp_ep_uses_target_rank_scales() -> None:
    config = Config(
        Ms=1,
        K=128,
        N=128,
        E=2,
        topks=1,
        dtype=torch.bfloat16,
        quant_config=TestMoEQuantConfig(torch.int8, False, False, None),
        activation=MoEActivation.SILU,
        prepare_finalize_type=MoEPrepareAndFinalizeNaiveDPEPModular,
        fused_experts_type=TritonExperts,
        world_size=2,
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(enforce_eager=True, is_moe=True)
    vllm_config.parallel_config.data_parallel_size = 2
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.all2all_backend = "allgather_reducescatter"
    env = {"VLLM_USE_DEEP_GEMM": "0"}
    parallel_launch_with_config(2, _worker, vllm_config, env, config)
