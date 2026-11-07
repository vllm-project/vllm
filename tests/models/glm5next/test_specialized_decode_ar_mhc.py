# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused TP all-reduce + mHC boundary against an
all-gather and the stock PyTorch reference of mHC post, pre and RMSNorm."""

import pytest
import ray
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

HIDDEN, HC, SINKHORN, EPS, TOP_K = 4096, 4, 20, 1e-6, 8


def _reference(peers, i):
    # The fused kernel sums the ranks in FP32 and rounds once, as one-shot
    # all-reduces do.
    x = (
        (peers[0].float() + peers[1].float()) + (peers[2].float() + peers[3].float())
    ).bfloat16()
    residual = mhc_post_torch(x, i["residual"], i["post"], i["comb"])
    post, comb, collapsed = mhc_pre_torch(
        residual, i["fn"].float(), i["scale"], i["base"],
        EPS, EPS, EPS, 2.0, SINKHORN,
    )  # fmt: skip
    v = collapsed.float()
    rstd = torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS)
    return residual, post, comb, (v * rstd * i["norm"].float()).bfloat16()


def _inputs(tokens, rank, device):
    g = torch.Generator(device=device).manual_seed(tokens)

    def randn(*shape, generator=g):
        return torch.randn(*shape, generator=generator, device=device)

    x = randn(tokens, HIDDEN, generator=torch.Generator(device).manual_seed(rank))
    # Packed +0/-0 pairs collide with the Lamport sentinel.
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    return dict(
        x=x.bfloat16(),
        residual=randn(tokens, HC, HIDDEN).bfloat16(),
        post=torch.rand(tokens, HC, 1, generator=g, device=device) * 2,
        comb=torch.softmax(randn(tokens, HC, HC), -1),
        fn=(randn(2 * HC + HC * HC, HC * HIDDEN) * 0.02).bfloat16(),
        scale=torch.tensor([0.5, 0.5, 1.0], device=device),
        base=randn(2 * HC + HC * HC) * 0.1,
        norm=(1 + 0.1 * randn(HIDDEN)).bfloat16(),
    )


@ray.remote(num_gpus=1, max_calls=1)
def _boundary(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.models.glm5next.nvidia.specialized_decode.all_reduce_mhc import (
        AllReduceMhcBoundary,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        # Larger batches share each cluster between tokens; 37 and 64 end on a
        # short group.
        op = AllReduceMhcBoundary(
            hidden_size=HIDDEN, max_num_tokens=64, top_k=TOP_K, rms_eps=EPS,
            hc_eps=EPS, post_mult=2.0, sinkhorn_repeat=SINKHORN, norm_eps=EPS,
            device=device, cluster_tokens={6: 2, 16: 2, 37: 3, 48: 4, 64: 5},
        )  # fmt: skip

        def check(outputs, i):
            peers = get_tp_group().all_gather(i["x"], dim=0).view(tp_size, -1, HIDDEN)
            ref = _reference(peers, i)
            # BF16 outputs are one rounding apart at most; the mixes are FP32
            # and differ only in summation order.
            torch.testing.assert_close(outputs[0], ref[0], atol=2**-6, rtol=2**-7)
            torch.testing.assert_close(outputs[1], ref[1], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(outputs[2], ref[2], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(outputs[3], ref[3], atol=2**-5, rtol=2**-7)

        # Changing shapes cover shrinking and growing batches in one mailbox.
        for tokens in (1, 6, 16, 3, 2, 64, 37, 48, 8):
            i = _inputs(tokens, rank, device)
            args = [i[k] for k in ("x", "residual", "post", "comb", "fn")]
            args += [i[k] for k in ("scale", "base", "norm")]
            eager = op(*args)
            op.join()
            check(eager, i)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):
                    op(*args)
                captured = op(*args)
                op.join()
            for _ in range(10):
                graph.replay()
            check(captured, i)
            # Replays pick up in-place input changes.
            i["x"].mul_(0.5)
            i["residual"].mul_(2)
            graph.replay()
            check(captured, i)

            # A MoE output with its top-k reduction open: the fused finalize
            # must match finalizing first (one FP32 FMA per route in route
            # order, the shared add, one BF16 rounding).
            rows = tokens * TOP_K + 5
            gemm2 = torch.randn(rows, HIDDEN, device=device).bfloat16()
            permuted = torch.randperm(rows, device=device)[: tokens * TOP_K]
            permuted = permuted.view(tokens, TOP_K).int()
            permuted[0, -1] = -1  # an expert this rank does not hold
            weights = torch.rand(tokens, TOP_K, device=device)
            routed = UnfinalizedMoEOutput(gemm2, weights, permuted)
            shared = i["x"]
            moe = MoEOutput(routed=routed, shared_output=shared)
            fused = op(moe, *args[1:])
            acc = torch.zeros(tokens, HIDDEN, device=device)
            for k in range(TOP_K):
                valid = (permuted[:, k] >= 0).unsqueeze(-1)
                rows_k = gemm2[permuted[:, k].clamp_min(0).long()].float()
                torch.addcmul(
                    acc,
                    torch.where(valid, rows_k, 0.0),
                    torch.where(valid, weights[:, k : k + 1], 0.0),
                    out=acc,
                )
            plain = op((acc + shared.float()).bfloat16(), *args[1:])
            op.join()
            for a, b in zip(fused, plain):
                assert torch.equal(a, b)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
def test_all_reduce_mhc_boundary_matches_reference(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _boundary)
