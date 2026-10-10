# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer KV checksum cost: the Triton kernel vs the PyTorch path.

Checksums ``--num-blocks`` blocks of one layer's KV cache, as a consumer does
for a finished load and a producer for a finished prefill (multiply by the
number of layers for a whole model).
"""

import argparse
from unittest.mock import patch

import torch

from vllm.distributed.kv_transfer.kv_checksum import worker as checksum_worker
from vllm.triton_utils import triton


def main(args: argparse.Namespace) -> None:
    # [blocks, heads, tokens, K and V of head_size] bf16, as 8-byte pieces.
    cache = torch.randn(
        args.cache_blocks,
        args.heads,
        args.block_size,
        2 * args.head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pieces = cache.view(torch.int64).unflatten(0, (args.cache_blocks, 1))
    block_ids = torch.randperm(args.cache_blocks, device="cuda")[: args.num_blocks]
    first_tokens = torch.arange(args.num_blocks, device="cuda") * args.block_size
    valid_tokens = torch.full_like(block_ids, args.block_size)
    gigabytes = block_ids.numel() * cache[0].numel() * cache.element_size() / 1e9

    results = []
    for use_triton in (False, True):
        # The PyTorch path is the one taken without Triton.
        with patch.object(checksum_worker, "HAS_TRITON", use_triton):
            view = checksum_worker._LayerView(pieces, first_slice=7, replica_key=(1, 0))
        out = torch.zeros_like(block_ids)
        view.accumulate(block_ids, first_tokens, valid_tokens, out)
        results.append(out.clone())
        ms = triton.testing.do_bench(
            lambda view=view, out=out: view.accumulate(
                block_ids, first_tokens, valid_tokens, out
            )
        )
        name = "triton" if use_triton else "pytorch"
        print(f"{name:8s} {ms:8.3f} ms  {gigabytes / ms * 1e3:8.1f} GB/s of KV read")
    assert torch.equal(results[0], results[1]), "Triton and PyTorch paths differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=128)
    main(parser.parse_args())
