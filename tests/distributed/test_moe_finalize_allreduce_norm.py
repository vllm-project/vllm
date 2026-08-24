# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE tail-fusion consumer must equal the unfused MoE tail.

Guards `vllm.moe_finalize_allreduce_rms_norm`, which folds the top-k reduction
over an unfinalized MoE output, the shared-expert add, the tensor-parallel
all-reduce, the residual add and the RMSNorm into one kernel.
"""

import queue
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.model_executor.layers.fused_moe_finalize_norm as fmfn
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    destroy_fi_ar_workspace,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

HIDDEN_SIZE = 2048
NUM_TOKENS = 19
TOP_K = 4
RMS_EPS = 1e-6
# The consumer's workspace compiles for a capacity, not for this batch.
MAX_NUM_TOKENS = 64


def _reference_tail(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx: torch.Tensor,
    shared_output: torch.Tensor | None,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    weight_bias: float,
    routed_scaling_factor: float,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unfused tail, in fp32."""
    idx = expanded_idx.long()
    rows = gemm2_permuted.index_select(0, idx.clamp_min(0).view(-1)).float()
    rows = rows.view(idx.shape[0], idx.shape[1], -1)
    weights = torch.where(
        idx >= 0, expert_weights.float(), torch.zeros_like(idx).float()
    )
    routed = (rows * weights.unsqueeze(-1)).sum(dim=1) * routed_scaling_factor
    if shared_output is not None:
        routed = routed + shared_output.float()
    routed = routed.to(residual.dtype)
    dist.all_reduce(routed, group=group)

    residual_out = routed + residual
    var = residual_out.float().pow(2).mean(dim=-1, keepdim=True)
    normed = residual_out.float() * torch.rsqrt(var + RMS_EPS)
    return (normed * (rms_gamma.float() + weight_bias)).to(residual.dtype), residual_out


def _worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(12340 + world_size),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        torch.manual_seed(1234 + local_rank)

        m.setenv("VLLM_ENABLE_MOE_TAIL_FUSION", "1")
        if not fmfn.moe_tail_fusion_applies(dtype):
            q.put("MoE tail fusion is unsupported on this setup.")
            return

        # A padded permuted buffer, as the trtllm-gen kernel hands back: more
        # rows than num_tokens * top_k, referenced only through the permute map.
        num_permuted = NUM_TOKENS * TOP_K + 37
        group = get_tp_group().device_group

        for weight_bias, with_shared in ((0.0, False), (1.0, True)):
            # The workspace is compiled around weight_bias and whether a shared
            # expert is folded in, and one process holds one of them -- as a
            # model does, since its norms and its expert layout do not vary per
            # layer. Drop it so the next configuration builds its own.
            destroy_fi_ar_workspace()

            gemm2_permuted = torch.randn(num_permuted, HIDDEN_SIZE, dtype=dtype) * 0.1
            expert_weights = torch.rand(NUM_TOKENS, TOP_K, dtype=dtype)
            expanded_idx = torch.randperm(NUM_TOKENS * TOP_K, device=device)[
                : NUM_TOKENS * TOP_K
            ].view(NUM_TOKENS, TOP_K)
            # -1 marks an expert that is not local to this rank.
            expanded_idx = torch.where(
                torch.rand(NUM_TOKENS, TOP_K, device=device) < 0.25,
                torch.full_like(expanded_idx, -1),
                expanded_idx,
            ).to(torch.int32)
            residual = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype)
            rms_gamma = torch.randn(HIDDEN_SIZE, dtype=dtype) * 0.1
            shared_output = (
                torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype) * 0.1
                if with_shared
                else None
            )
            # Every rank must norm with the same weights and residual.
            dist.broadcast(residual, src=0, group=group)
            dist.broadcast(rms_gamma, src=0, group=group)

            norm_out, residual_out = torch.ops.vllm.moe_finalize_allreduce_rms_norm(
                gemm2_permuted,
                expert_weights,
                expanded_idx,
                shared_output,
                residual,
                rms_gamma,
                RMS_EPS,
                weight_bias,
                1.0,
                MAX_NUM_TOKENS,
            )
            ref_norm, ref_residual = _reference_tail(
                gemm2_permuted,
                expert_weights,
                expanded_idx,
                shared_output,
                residual,
                rms_gamma,
                weight_bias,
                1.0,
                group,
            )
            torch.testing.assert_close(
                residual_out.float(), ref_residual.float(), atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                norm_out.float(), ref_norm.float(), atol=2e-2, rtol=2e-2
            )

        # The public entry point: a MoEOutput whose routed half is unfinalized,
        # closed out into a model's own Gemma-style norm, with a routed scale to
        # apply.
        destroy_fi_ar_workspace()
        gemm2_permuted = torch.randn(num_permuted, HIDDEN_SIZE, dtype=dtype) * 0.1
        expert_weights = torch.rand(NUM_TOKENS, TOP_K, dtype=dtype)
        expanded_idx = (
            torch.randperm(NUM_TOKENS * TOP_K, device=device)
            .view(NUM_TOKENS, TOP_K)
            .to(torch.int32)
        )
        shared_output = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype) * 0.1
        residual = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype)
        rms_gamma = torch.randn(HIDDEN_SIZE, dtype=dtype) * 0.1
        dist.broadcast(residual, src=0, group=group)
        dist.broadcast(rms_gamma, src=0, group=group)

        class ModelOwnGemmaRMSNorm(torch.nn.Module):
            rms_weight_bias = 1.0

            def __init__(self):
                super().__init__()
                self.weight = rms_gamma
                self.variance_epsilon = RMS_EPS

        moe_output = MoEOutput(
            routed=UnfinalizedMoEOutput(
                gemm2_permuted=gemm2_permuted,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx,
            ),
            shared_output=shared_output,
            routed_scaling_factor=2.0,
            max_num_tokens=MAX_NUM_TOKENS,
        )
        ref_norm, ref_residual = _reference_tail(
            gemm2_permuted,
            expert_weights,
            expanded_idx,
            shared_output,
            residual,
            rms_gamma,
            1.0,
            2.0,
            group,
        )

        norm_out, residual_out = fmfn.fused_moe_finalize_allreduce_rms_norm(
            moe_output, residual, ModelOwnGemmaRMSNorm()
        )
        torch.testing.assert_close(
            residual_out.float(), ref_residual.float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            norm_out.float(), ref_norm.float(), atol=2e-2, rtol=2e-2
        )

        q.put(None)


def test_rms_norm_weight_bias(default_vllm_config):
    """A Gemma-style norm must be recognized however it is spelled.

    Missing the `1 +` is a silent numerical error, and models carrying their own
    Gemma-style RMSNorm do not subclass `GemmaRMSNorm`.
    """
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

    class ModelOwnGemmaRMSNorm(torch.nn.Module):
        rms_weight_bias = 1.0

    assert fmfn.rms_norm_weight_bias(GemmaRMSNorm(8)) == 1.0
    assert fmfn.rms_norm_weight_bias(ModelOwnGemmaRMSNorm()) == 1.0
    assert fmfn.rms_norm_weight_bias(RMSNorm(8)) == 0.0


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MoE tail fusion targets Blackwell",
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_moe_finalize_allreduce_rms_norm(world_size, monkeypatch):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    # The fusion is opt-in; without this the workers skip. It carries its own
    # mnnvl CuTe DSL workspace, so VLLM_FLASHINFER_ALLREDUCE_BACKEND is not part
    # of reaching it.
    monkeypatch.setenv("VLLM_ENABLE_MOE_TAIL_FUSION", "1")

    q: mp.Queue = mp.get_context("spawn").Queue()
    mp.spawn(_worker, args=(world_size, q), nprocs=world_size)
    try:
        reason = q.get(timeout=1)
    except queue.Empty:
        reason = None
    if reason is not None:
        pytest.skip(typing.cast(str, reason))
    cleanup_dist_env_and_memory()
