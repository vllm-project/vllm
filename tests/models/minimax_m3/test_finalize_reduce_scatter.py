# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused MoE finalize + reduce-scatter + norm of MiniMax-M3's token-sharded
batches against the unfused computation: the top-k finalize and shared add in
BF16, the TP sum in rank order rounded to BF16, then the Triton input norm on
this rank's rows. Routing weights are powers of two, so the finalize sums are
exact and their order does not matter. Outputs (MXFP8 codes and scales, or
BF16) and the residual must match bitwise, eagerly and under CUDA-graph
replay."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

HIDDEN = 6144
TOP_K = 4


def _inputs(num_tokens: int, rank: int, device):
    gen = torch.Generator(device=device).manual_seed(1000 * rank + num_tokens)
    rows = num_tokens * TOP_K + 37
    routed = torch.randn(rows, HIDDEN, generator=gen, device=device)
    perm = torch.randperm(rows, generator=gen, device=device)[: num_tokens * TOP_K]
    perm = perm.view(num_tokens, TOP_K).to(torch.int32)
    # Experts on other ranks: no local row.
    perm[torch.rand(perm.shape, generator=gen, device=device) < 0.3] = -1
    # Power-of-two weights: every product and partial sum below is exact in
    # FP32, so the summation order cannot change a bit.
    exponents = torch.randint(-2, 2, (num_tokens, TOP_K), generator=gen, device=device)
    weights = torch.pow(2.0, exponents.float())
    shared = torch.randn(num_tokens, HIDDEN, generator=gen, device=device)
    return routed.bfloat16(), weights.bfloat16(), perm, shared.bfloat16()


def _partial(routed, weights, perm, shared):
    """The unfused TP-partial row: finalize, then the shared add, in BF16."""
    rows = routed[perm.clamp(min=0).long()].float()
    w = torch.where(perm >= 0, weights.float(), 0.0)
    finalized = torch.zeros_like(shared, dtype=torch.float32)
    for k in range(TOP_K):
        finalized += w[:, k, None] * rows[:, k]
    finalized = finalized.bfloat16()
    return (finalized.float() + shared.float()).bfloat16()


@ray.remote(num_gpus=1, max_calls=1)
def _matches_unfused(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.fused_moe.moe_output import (
        MoEOutput,
        UnfinalizedMoEOutput,
    )
    from vllm.models.minimax_m3.nvidia.ops.finalize_reduce_scatter import (
        FinalizeReduceScatterGemmaRMSNorm,
    )
    from vllm.models.minimax_m3.nvidia.ops.norm_mxfp8 import gemma_rmsnorm_mxfp8

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        op = FinalizeReduceScatterGemmaRMSNorm(
            hidden_size=HIDDEN, max_num_tokens=2048, device=device
        )
        norm_weight = (torch.randn(HIDDEN, device=device) * 0.1).bfloat16()
        eps = 1e-6
        for num_tokens, generation in ((256 * tp_size, 0), (1000, 1), (1024, 0)):
            shard = (num_tokens + tp_size - 1) // tp_size
            routed, weights, perm, shared = _inputs(num_tokens, rank, device)
            moe_output = MoEOutput(
                routed=UnfinalizedMoEOutput(
                    gemm2_permuted=routed,
                    expert_weights=weights,
                    expanded_idx_to_permuted_idx=perm,
                ),
                shared_output=shared,
            )
            # Every rank's partial rows, summed in rank order.
            partials = tensor_model_parallel_all_gather(
                _partial(routed, weights, perm, shared)[None], 0
            )
            total = torch.zeros_like(partials[0], dtype=torch.float32)
            for partial in partials:  # rank order
                total += partial.float()
            total = total.bfloat16()
            total = torch.nn.functional.pad(
                total, (0, 0, 0, shard * tp_size - num_tokens)
            )
            mine = total[rank * shard : (rank + 1) * shard]
            gen = torch.Generator(device=device).manual_seed(7 + num_tokens)
            residual = torch.randn(
                tp_size * shard, HIDDEN, generator=gen, device=device
            ).bfloat16()[rank * shard : (rank + 1) * shard]
            for mxfp8 in (shard % 128 == 0, False):
                want_res = residual.clone()
                want, want_sf = gemma_rmsnorm_mxfp8(
                    mine, norm_weight, eps, residual=want_res, mxfp8=mxfp8
                )
                for graphed in (False, True):
                    res = residual.clone()

                    def run(
                        res=res,
                        mxfp8=mxfp8,
                        generation=generation,
                        moe_output=moe_output,
                    ):
                        return op(
                            moe_output,
                            res,
                            norm_weight,
                            eps,
                            mxfp8=mxfp8,
                            generation=generation,
                        )

                    if graphed:
                        run()  # compile outside the capture
                        res.copy_(residual)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out, _ = run()
                        res.copy_(residual)
                        graph.replay()
                    else:
                        out, _ = run()
                    torch.accelerator.synchronize()
                    assert torch.equal(res, want_res)
                    if mxfp8:
                        assert torch.equal(
                            out.data.view(torch.uint8), want.view(torch.uint8)
                        )
                        assert torch.equal(out.scale, want_sf)
                    else:
                        assert torch.equal(out, want)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Needs SM100-family GPUs with peer access",
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_finalize_reduce_scatter(monkeypatch: pytest.MonkeyPatch, tp_size: int):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs.")
    multi_process_parallel(monkeypatch, tp_size, 1, _matches_unfused)
