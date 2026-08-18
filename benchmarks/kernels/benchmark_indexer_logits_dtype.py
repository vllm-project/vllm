# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark DeepGEMM paged-indexer output precision on captured inputs."""

import argparse
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from vllm.utils import deep_gemm

_PAGE_SIZE = 64
_TOPK = 2048


def time_launch(launch, repeats: int) -> float:
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def run_logits(
    dg,
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    metadata: torch.Tensor,
    max_model_len: int,
    indices: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dg.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_pages,
        weights,
        seq_lens,
        block_table,
        metadata,
        max_model_len,
        False,
        dtype,
        indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--batches", default="1,8,32,128,1024")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    dg = deep_gemm._import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is required")
    dg.set_pdl(True)

    captured = torch.load(args.capture, map_location="cuda", weights_only=True)
    for name in ("q_values", "weights", "logits"):
        if not torch.isfinite(captured[name].float()).all():
            raise ValueError(f"captured {name} contains NaN or Inf: {args.capture}")
    kv_pages = captured["kv_pages"].contiguous()
    seq_len = int(captured["seq_len"])
    num_pages = kv_pages.shape[0]
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    results: dict[tuple[int, torch.dtype], list[float]] = defaultdict(list)

    for batch in map(int, args.batches.split(",")):
        q = captured["q_values"].unsqueeze(0).repeat(batch, 1, 1, 1)
        weights = captured["weights"].repeat(batch, 1)
        seq_lens = torch.full((batch, 1), seq_len, dtype=torch.int32, device="cuda")
        block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").repeat(
            batch, 1
        )
        indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        metadata = dg.get_paged_mqa_logits_metadata(
            seq_lens, _PAGE_SIZE, dg.get_num_sms(), indices=indices
        )

        launch = partial(
            run_logits,
            dg,
            q,
            kv_pages,
            weights,
            seq_lens,
            block_table,
            metadata,
            math.ceil(seq_len / _PAGE_SIZE) * _PAGE_SIZE,
            indices,
        )

        reference = launch(torch.float32)
        reference_indices = reference[:, :seq_len].topk(_TOPK).indices
        for dtype in dtypes[1:]:
            logits = launch(dtype)
            reduced_indices = logits[:, :seq_len].topk(_TOPK).indices
            overlap = (
                sum(
                    len(set(old.tolist()) & set(new.tolist())) / _TOPK
                    for old, new in zip(
                        reference_indices.cpu(), reduced_indices.cpu(), strict=True
                    )
                )
                / batch
            )
            max_error = (
                (logits[:, :seq_len].float() - reference[:, :seq_len])
                .abs()
                .max()
                .item()
            )
            print(
                f"batch={batch} dtype={dtype} max_error={max_error:.6g} "
                f"fp32-index-overlap={overlap:.6%}",
                flush=True,
            )

        for round_index in range(args.rounds):
            order = dtypes if round_index % 2 == 0 else tuple(reversed(dtypes))
            for dtype in order:
                latency = time_launch(partial(launch, dtype), args.repeats)
                results[batch, dtype].append(latency)
                print(
                    f"batch={batch} round={round_index} dtype={dtype} "
                    f"latency={latency:.3f} us",
                    flush=True,
                )

    print("aggregate")
    for batch in map(int, args.batches.split(",")):
        means = {
            dtype: sum(results[batch, dtype]) / len(results[batch, dtype])
            for dtype in dtypes
        }
        print(
            f"batch={batch} fp32={means[torch.float32]:.3f} us "
            f"fp16={means[torch.float16]:.3f} us "
            f"bf16={means[torch.bfloat16]:.3f} us "
            f"fp16-speedup={means[torch.float32] / means[torch.float16]:.4f}x "
            f"bf16-speedup={means[torch.float32] / means[torch.bfloat16]:.4f}x"
        )


if __name__ == "__main__":
    main()
