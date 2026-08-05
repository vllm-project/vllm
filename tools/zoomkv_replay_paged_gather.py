# SPDX-License-Identifier: Apache-2.0
"""Standalone stress/replay tool for ZoomKV's fused Triton paged gather."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vllm.v1.attention.ops.zoomkv.paged import (
    _gather_kv_from_topk_batch_reference,
)
from vllm.v1.attention.ops.zoomkv.paged_triton import (
    paged_gather_kv_from_topk_batch,
)


def _synthetic_case(
    max_batch: int,
    heads: int,
    head_dim: int,
    max_seq_len: int,
) -> dict[str, object]:
    block_size = 16
    max_blocks = (max_seq_len + block_size - 1) // block_size
    kv = torch.randn(
        max_blocks,
        heads,
        block_size,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key, value = kv.transpose(1, 2).split(head_dim, dim=-1)
    block_table = torch.arange(
        max_blocks, device="cuda", dtype=torch.int32
    ).expand(max_batch, -1).contiguous()
    return {
        "key_cache": key,
        "value_cache": value,
        "block_table": block_table,
        "block_size": block_size,
        "sink_size": 64,
        "local_size": 256,
        "output_bthd": True,
    }


def _load_snapshot(path: Path) -> dict[str, object]:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    saved_k = snapshot["key_cache"]
    saved_v = snapshot["value_cache"]
    blocks, block_size, heads, head_dim = saved_k.shape
    kv = torch.empty(
        blocks,
        heads,
        block_size,
        2 * head_dim,
        dtype=saved_k.dtype,
        device="cuda",
    )
    key, value = kv.transpose(1, 2).split(head_dim, dim=-1)
    key.copy_(saved_k.cuda())
    value.copy_(saved_v.cuda())
    snapshot["key_cache"] = key
    snapshot["value_cache"] = value
    for name in ("block_table", "seq_lens", "topk"):
        snapshot[name] = snapshot[name].cuda()
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    case = (
        _load_snapshot(args.snapshot)
        if args.snapshot
        else _synthetic_case(8, args.heads, args.head_dim, 66560)
    )
    key = case["key_cache"]
    value = case["value_cache"]
    block_table = case["block_table"]
    block_size = int(case["block_size"])
    sink_size = int(case["sink_size"])
    local_size = int(case["local_size"])
    max_batch = int(block_table.shape[0])
    heads = int(key.shape[2])
    topk_len = 100
    seq_sweep = (65536, 65871, 65872, 66560)
    saved_seq = case.get("seq_lens")
    saved_topk = case.get("topk")

    print(
        {
            "key_shape": tuple(key.shape),
            "key_stride": tuple(key.stride()),
            "block_table_shape": tuple(block_table.shape),
            "block_table_stride": tuple(block_table.stride()),
            "repeats": args.repeats,
        },
        flush=True,
    )
    batch_sizes = sorted({1, 2, 4, 8, max_batch})
    for batch in batch_sizes:
        if batch > max_batch:
            continue
        inputs = []
        if saved_seq is not None and saved_topk is not None:
            inputs.append(("snapshot", saved_seq[:batch], saved_topk[:batch]))
        else:
            for seq_len in seq_sweep:
                seq = torch.full(
                    (batch,), seq_len, dtype=torch.int32, device="cuda"
                )
                low = sink_size
                high = max(low + 1, seq_len - local_size)
                backing = torch.randint(
                    low,
                    high,
                    (max(batch * 2, 8), heads, topk_len),
                    dtype=torch.int64,
                    device="cuda",
                )
                inputs.append((str(seq_len), seq, backing[:batch]))
        for label, seq, topk in inputs:
            bt = block_table[:batch]
            ref_k, ref_v = _gather_kv_from_topk_batch_reference(
                key,
                value,
                bt,
                seq,
                topk,
                block_size,
                sink_size,
                local_size,
                output_bthd=True,
            )
            for _ in range(args.repeats):
                tri_k, tri_v = paged_gather_kv_from_topk_batch(
                    key,
                    value,
                    bt,
                    seq,
                    topk,
                    block_size,
                    sink_size,
                    local_size,
                    output_bthd=True,
                )
                torch.cuda.synchronize()
            torch.testing.assert_close(tri_k, ref_k, rtol=0, atol=0)
            torch.testing.assert_close(tri_v, ref_v, rtol=0, atol=0)
            print(
                f"PASS batch={batch} seq_len={label} "
                f"topk_stride={tuple(topk.stride())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
