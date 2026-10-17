# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksum cost: the Triton kernel vs the PyTorch path.

Checksums ``--num-blocks`` blocks of ``--num-layers`` layers of one layout, as
a consumer does for a finished load and a producer for a finished prefill. The
Triton kernel takes one launch for all the layers; the PyTorch path one pass
per layer.
"""

import argparse
from unittest.mock import patch

import torch

from vllm.distributed.kv_transfer.kv_checksum import worker as checksum_worker
from vllm.distributed.kv_transfer.kv_checksum.checksum import MAX_KV_HEADS_PER_LAYER
from vllm.triton_utils import triton


def main(args: argparse.Namespace) -> None:
    # Per layer: [blocks, heads, tokens, K and V of head_size] bf16, as 8-byte
    # pieces.
    caches = [
        torch.randn(
            args.cache_blocks,
            args.heads,
            args.block_size,
            2 * args.head_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(args.num_layers)
    ]
    pieces = [
        cache.view(torch.int64).unflatten(0, (args.cache_blocks, 1)) for cache in caches
    ]
    block_ids = torch.randperm(args.cache_blocks, device="cuda")[: args.num_blocks]
    first_tokens = torch.arange(args.num_blocks, device="cuda") * args.block_size
    valid_tokens = torch.full_like(block_ids, args.block_size)
    layer_bytes = caches[0][0].numel() * caches[0].element_size()
    gigabytes = block_ids.numel() * layer_bytes * args.num_layers / 1e9

    results = []
    for use_triton in (False, True):
        # The PyTorch path is the one taken without Triton.
        with patch.object(checksum_worker, "HAS_TRITON", use_triton):
            views = [
                checksum_worker._LayerView(
                    layer_pieces,
                    first_slice=layer * MAX_KV_HEADS_PER_LAYER,
                    replica_key=(1, 0),
                )
                for layer, layer_pieces in enumerate(pieces)
            ]
            launch = checksum_worker._FusedLayers(views, columns=[0] * len(views))
        out = torch.zeros_like(block_ids).unsqueeze(1)
        launch.accumulate(block_ids, first_tokens, valid_tokens, out)
        results.append(out.clone())
        ms = triton.testing.do_bench(
            lambda launch=launch, out=out: launch.accumulate(
                block_ids, first_tokens, valid_tokens, out
            )
        )
        name = "triton" if use_triton else "pytorch"
        print(f"{name:8s} {ms:8.3f} ms  {gigabytes / ms * 1e3:8.1f} GB/s of KV read")
    assert torch.equal(results[0], results[1]), "Triton and PyTorch paths differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=128)
    main(parser.parse_args())
