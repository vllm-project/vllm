# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: fused Triton dcp_split_q vs PyTorch reference
(eager / compiled / CUDAGraph).

Usage:
    python tests/v1/attention/bench_dcp_split_q.py
"""

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.dcp_split_q import dcp_split_q


def _pytorch_reference(
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_decode_tokens = num_decodes * tokens_per_req
    offsets = torch.arange(
        1 - tokens_per_req,
        1,
        device=global_seq_lens.device,
        dtype=global_seq_lens.dtype,
    )
    per_token_global = (global_seq_lens[:num_decodes].unsqueeze(1) + offsets).flatten()
    virtual = dcp_world_size * interleave
    base = per_token_global // virtual * interleave
    remainder = (
        (per_token_global - base * dcp_world_size - dcp_rank * interleave)
        .clamp(min=0)
        .clamp(max=interleave)
    )
    seq_lens = base + remainder
    block_tables = (
        block_table[:num_decodes]
        .unsqueeze(1)
        .expand(-1, tokens_per_req, -1)
        .reshape(num_decode_tokens, -1)
    )
    return seq_lens, block_tables


def _make_compiled(dcp_world_size: int, dcp_rank: int, interleave: int):
    @torch.compile(fullgraph=True, mode="max-autotune")
    def _fn(
        global_seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_decodes: int,
        tokens_per_req: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _pytorch_reference(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
        )

    return _fn


def _make_cudagraph(
    global_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    dcp_rank: int,
    interleave: int,
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _pytorch_reference(
                global_seq_lens,
                block_table,
                num_decodes,
                tokens_per_req,
                dcp_world_size,
                dcp_rank,
                interleave,
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_sl, out_bt = _pytorch_reference(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
        )

    def replay():
        g.replay()
        return out_sl, out_bt

    return replay


configs = [
    triton.testing.Benchmark(
        x_names=["num_decodes"],
        x_vals=[
            1,
            4,
            16,
            64,
            128,
            256,
            512,
            1024,
            1234,
            2000,
            3000,
            5000,
            5120,
        ],
        line_arg="provider",
        line_vals=["triton", "pytorch", "compiled", "cudagraph"],
        line_names=[
            "Triton fused",
            "PyTorch eager",
            "torch.compile",
            "CUDAGraph",
        ],
        styles=[
            ("blue", "-"),
            ("red", "--"),
            ("green", "-."),
            ("orange", ":"),
        ],
        ylabel="us",
        plot_name=(f"dcp_split_q (tokens_per_req={tpr}, dcp={dcp}, interleave={il})"),
        args={
            "tokens_per_req": tpr,
            "dcp_world_size": dcp,
            "interleave": il,
            "ncols": 128,
        },
    )
    for tpr in [4]
    for dcp in [2, 4]
    for il in [1]
]


@triton.testing.perf_report(configs)
def bench_dcp_split_q(
    num_decodes: int,
    tokens_per_req: int,
    dcp_world_size: int,
    interleave: int,
    ncols: int,
    provider: str,
) -> float:
    device = "cuda"
    dcp_rank = 0

    global_seq_lens = torch.randint(
        tokens_per_req,
        4096,
        (num_decodes,),
        device=device,
        dtype=torch.int32,
    )
    block_table = torch.randint(
        0,
        1000,
        (num_decodes, ncols),
        device=device,
        dtype=torch.int32,
    )

    total = num_decodes * tokens_per_req
    out_sl = torch.empty(total, device=device, dtype=torch.int32)
    out_bt = torch.empty(total, ncols, device=device, dtype=torch.int32)

    if provider == "triton":
        fn = lambda: dcp_split_q(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
            out_sl,
            out_bt,
        )
    elif provider == "pytorch":
        fn = lambda: _pytorch_reference(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
        )
    elif provider == "compiled":
        compiled_fn = _make_compiled(dcp_world_size, dcp_rank, interleave)
        for _ in range(3):
            compiled_fn(
                global_seq_lens,
                block_table,
                num_decodes,
                tokens_per_req,
            )
        fn = lambda: compiled_fn(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
        )
    elif provider == "cudagraph":
        replay = _make_cudagraph(
            global_seq_lens,
            block_table,
            num_decodes,
            tokens_per_req,
            dcp_world_size,
            dcp_rank,
            interleave,
        )
        fn = replay
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = triton.testing.do_bench(fn, warmup=100, rep=500)
    return ms * 1000  # us


if __name__ == "__main__":
    bench_dcp_split_q.run(print_data=True, show_plots=False)
