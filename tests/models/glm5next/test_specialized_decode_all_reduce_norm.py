# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP draft's fused vocab-parallel embedding all-reduce + enorm / hnorm
against the gathered embedding and a naive PyTorch ``fused_eh_norm``."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

HIDDEN, EPS, VOCAB, SHARD = 4096, 1e-5, 1000, 256  # last shard: 232 rows + padding


def _rms_norm(x, weight):
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight.double()


@ray.remote(num_gpus=1, max_calls=1)
def _embed_norm(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm.models.glm5next.nvidia.ops.fused_eh_norm import fused_eh_norm
    from vllm.models.glm5next.nvidia.specialized_decode.all_reduce_norm import (
        AllReduceNorm,
    )

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        op = AllReduceNorm(
            hidden_size=HIDDEN, max_num_tokens=64, top_k=8, eps=EPS, device=device
        )
        g = torch.Generator(device=device).manual_seed(0)
        table = torch.randn(tp_size * SHARD, HIDDEN, generator=g, device=device)
        table = table.bfloat16()
        table[3, :16] = -0.0  # packed -0 pairs collide with the Lamport sentinel
        start = rank * SHARD
        rows = min(SHARD, VOCAB - start)
        weight = table[start : start + SHARD].clone()
        weight[rows:] = float("nan")  # padding rows are never read
        enorm = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
        hnorm = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
        op.prepare_embedding(weight, start, rows, EPS)

        def check(output, ids, positions, previous):
            embedding = torch.where(
                (positions == 0).unsqueeze(-1), 0.0, table[ids.long()].double()
            )
            ref = torch.cat(
                [_rms_norm(embedding, enorm), _rms_norm(previous, hnorm)], -1
            )
            # One BF16 rounding of an FP32 result away from the FP64 reference,
            # as the stock Triton fused_eh_norm.
            torch.testing.assert_close(output.double(), ref, atol=2**-7, rtol=2**-7)
            stock = fused_eh_norm(
                positions, table[ids.long()], previous, enorm, hnorm, EPS
            )
            error = (output.double() - ref).abs().max()
            assert error <= (stock.double() - ref).abs().max() * 1.05

        for tokens in (1, 8, 3, 64, 16):
            ids = torch.randint(0, VOCAB, (tokens,), generator=g, device=device)
            ids = ids.int()
            ids[0] = 3
            positions = torch.randint(0, 9, (tokens,), generator=g, device=device)
            previous = torch.randn(tokens, HIDDEN, generator=g, device=device)
            previous = previous.bfloat16()
            args = (ids, positions, previous, weight, enorm, hnorm)
            check(op.embed_norm(*args), ids, positions, previous)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(3):
                    op.embed_norm(*args)
                captured = op.embed_norm(*args)
            for _ in range(5):
                graph.replay()
            check(captured, ids, positions, previous)
            # Replays pick up in-place input changes.
            ids.copy_(ids.flip(0))
            positions.add_(1)
            graph.replay()
            check(captured, ids, positions, previous)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)
def test_embed_norm_matches_reference(monkeypatch):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs with NVLink multicast")
    multi_process_parallel(monkeypatch, 4, 1, _embed_norm)
