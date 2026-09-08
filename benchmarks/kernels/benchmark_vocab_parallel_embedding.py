# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the TP > 1 input-embedding path of VocabParallelEmbedding.

Two providers, both computing one rank's partial embedding (the all-reduce
that follows is identical for each and is not measured):

  unfused   mask + id shift + int64 cast + F.embedding + masked_fill_
  fused     the single fused CUDA kernel, _C::vocab_parallel_embedding

With --use-cuda-graph each provider is captured into a CUDA graph and the
replay is timed, which is what a decode step actually costs.
"""

import itertools

import torch

from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
    pad_vocab_size,
)
from vllm.triton_utils import triton


def get_shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size):
    return VocabParallelEmbedding._get_indices(
        pad_vocab_size(vocab_size),
        pad_vocab_size(org_vocab_size),
        vocab_size,
        org_vocab_size,
        tp_rank,
        tp_size,
    )


def embedding_reference(input_ids, weight, shard_indices):
    masked_input, input_mask = get_masked_input_and_mask(
        input_ids,
        shard_indices.org_vocab_start_index,
        shard_indices.org_vocab_end_index,
        shard_indices.num_org_vocab_padding,
        shard_indices.added_vocab_start_index,
        shard_indices.added_vocab_end_index,
    )
    output = torch.nn.functional.embedding(masked_input.long(), weight)
    return output.masked_fill_(input_mask.unsqueeze(-1), 0)


def embedding_fused(input_ids, weight, shard_indices):
    return vllm_ops.vocab_parallel_embedding(
        input_ids,
        weight,
        shard_indices.org_vocab_start_index,
        shard_indices.org_vocab_end_index,
        shard_indices.num_org_vocab_padding,
        shard_indices.added_vocab_start_index,
        shard_indices.added_vocab_end_index,
    )


PROVIDERS = {
    "unfused": embedding_reference,
    "fused": embedding_fused,
}


def make_inputs(vocab_size, hidden_size, tp_size, num_tokens, dtype, tp_rank=0):
    shard_indices = get_shard_indices(vocab_size, vocab_size, tp_rank, tp_size)
    weight = torch.randn(
        pad_vocab_size(vocab_size) // tp_size,
        hidden_size,
        dtype=dtype,
        device="cuda",
    )
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    return input_ids, weight, shard_indices


def graph_replay(fn):
    """Capture fn into a CUDA graph and return its replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def calculate_diff(vocab_size, hidden_size, tp_size, num_tokens):
    """The fused kernel must be bit-exact with the path it replaces, on every
    rank of the shard."""
    dtype = torch.bfloat16
    for tp_rank in range(tp_size):
        input_ids, weight, shard_indices = make_inputs(
            vocab_size, hidden_size, tp_size, num_tokens, dtype, tp_rank
        )
        output_reference = embedding_reference(input_ids, weight, shard_indices)
        output_fused = embedding_fused(input_ids, weight, shard_indices)
        if not torch.equal(output_reference, output_fused):
            print(f"❌ rank {tp_rank}: fused output differs")
            return
    print(f"✅ fused matches the reference path on all {tp_size} ranks")


tp_size_range = [2, 4, 8]
num_tokens_range = [1, 8, 64, 512, 2048, 8192, 32768]
configs = list(itertools.product(tp_size_range, num_tokens_range))


def get_benchmark(vocab_size, hidden_size, dtype, use_cuda_graph):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["tp_size", "num_tokens"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=list(PROVIDERS),
            line_names=["Unfused PyTorch", "Fused kernel"],
            styles=[("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name=(
                f"vocab-parallel-embedding-perf-vocab{vocab_size}-"
                f"hidden{hidden_size}{'-cudagraph' if use_cuda_graph else ''}"
            ),
            args={},
        )
    )
    def benchmark(tp_size, num_tokens, provider):
        input_ids, weight, shard_indices = make_inputs(
            vocab_size, hidden_size, tp_size, num_tokens, dtype
        )
        fn = PROVIDERS[provider]

        def run():
            return fn(input_ids, weight, shard_indices)

        if use_cuda_graph:
            run = graph_replay(run)

        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=[0.5, 0.2, 0.8])
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-size", type=int, default=128256, help="Vocabulary size"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=4096, help="Embedding dimension"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Embedding table dtype"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="TP size used for the correctness check",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Time CUDA graph replay instead of eager launches",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save benchmark results",
    )
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    calculate_diff(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        tp_size=args.tp_size,
        num_tokens=1024,
    )

    benchmark = get_benchmark(
        args.vocab_size, args.hidden_size, dtype, args.use_cuda_graph
    )
    benchmark.run(print_data=True, save_path=args.save_path)
