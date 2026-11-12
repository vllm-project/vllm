# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare either hierarchical BF16 backend against a dense BF16 head."""

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.hierarchical_lm_head import HierarchicalLMHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--num-centroids", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--sidecar", help="Optional trained router; otherwise use random centroids"
    )
    parser.add_argument("--backend", choices=("compiled", "moe"), default="compiled")
    parser.add_argument(
        "--moe-routing",
        choices=("auto", "unsorted", "sorted"),
        default="unsorted",
        help="MoE assignment policy; sorted forces grouping across tokens at any C",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    head = HierarchicalLMHead(
        args.hidden_size,
        args.vocab_size,
        args.num_centroids,
        args.top_k,
        backend=args.backend,
        moe_routing=args.moe_routing,
    ).to(device="cuda", dtype=torch.bfloat16)
    if args.sidecar:
        head.load_sidecar(args.sidecar)
    weight = torch.randn(
        args.vocab_size, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    head.prepare_weights(weight)
    print(f"backend={args.backend}")
    print("vocabulary_precision=bf16")
    if args.backend == "moe":
        print(f"moe_routing={args.moe_routing}")
    print("batch,dense_us,hierarchical_us,speedup")
    with torch.inference_mode():
        for batch in args.batch_sizes:
            hidden = torch.randn(
                batch, args.hidden_size, device="cuda", dtype=torch.bfloat16
            )

            def dense(hidden=hidden):
                return F.linear(hidden, weight).argmax(-1)

            def sparse(hidden=hidden):
                return head.get_top_tokens(hidden, weight)

            # Compile outside CUDA graph capture, including batch=1 specialization.
            sparse()
            torch.cuda.synchronize()
            dense_ms = do_bench_cudagraph(dense)
            sparse_ms = do_bench_cudagraph(sparse)
            row = (
                f"{batch},{dense_ms * 1000:.3f},{sparse_ms * 1000:.3f},"
                f"{dense_ms / sparse_ms:.3f}"
            )
            print(row)


if __name__ == "__main__":
    main()
