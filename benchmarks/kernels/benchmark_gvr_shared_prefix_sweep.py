# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sweep exact shared-prefix indexer scheduling over prefix fractions."""

import argparse
from functools import partial
from pathlib import Path

import torch
from benchmark_gvr_shared_prefix import (
    _full_logits,
    _shared_prefix_stitched_logits,
    _time,
)

from vllm.utils import deep_gemm

_PAGE_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--batches", default="128,1024")
    parser.add_argument("--fractions", default="0.1,0.25,0.5,0.75,0.9,0.99")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    dg = deep_gemm._import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is required")
    dg.set_pdl(True)

    captured = torch.load(args.capture, map_location="cuda", weights_only=True)
    kv_pages = captured["kv_pages"].contiguous()
    seq_len = int(captured["seq_len"])
    num_pages = kv_pages.shape[0]

    for batch in map(int, args.batches.split(",")):
        q = captured["q_values"].unsqueeze(0).repeat(batch, 1, 1, 1)
        weights = captured["weights"].repeat(batch, 1)
        seq_lens = torch.full((batch, 1), seq_len, dtype=torch.int32, device="cuda")
        unique_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        grouped_indices = torch.zeros_like(unique_indices)

        for fraction in map(float, args.fractions.split(",")):
            common_tokens = int(seq_len * fraction) // _PAGE_SIZE * _PAGE_SIZE
            common_pages = common_tokens // _PAGE_SIZE
            tail_width = (seq_len - common_tokens + _PAGE_SIZE - 1) // _PAGE_SIZE
            tail_width *= _PAGE_SIZE

            block_table = torch.arange(
                num_pages, dtype=torch.int32, device="cuda"
            ).repeat(batch, 1)
            suffix_pages = block_table.shape[1] - common_pages
            row_offsets = (
                torch.arange(batch, dtype=torch.int32, device="cuda") * 104729
            )[:, None]
            block_table[:, common_pages:] = (
                block_table[:, common_pages:] + row_offsets
            ) % num_pages
            tail_block_table = block_table[:, common_pages:].contiguous()

            shared_lens = torch.full_like(seq_lens, common_tokens)
            tail_lens = seq_lens - common_tokens
            full_meta = dg.get_paged_mqa_logits_metadata(
                seq_lens, _PAGE_SIZE, dg.get_num_sms(), indices=unique_indices
            )
            shared_meta = dg.get_paged_mqa_logits_metadata(
                shared_lens, _PAGE_SIZE, dg.get_num_sms(), indices=grouped_indices
            )
            tail_meta = dg.get_paged_mqa_logits_metadata(
                tail_lens, _PAGE_SIZE, dg.get_num_sms(), indices=unique_indices
            )

            full_launch = partial(
                _full_logits,
                dg,
                q,
                kv_pages,
                weights,
                seq_lens,
                block_table,
                full_meta,
                unique_indices,
            )
            stitched_launch = partial(
                _shared_prefix_stitched_logits,
                dg,
                q,
                kv_pages,
                weights,
                shared_lens,
                tail_lens,
                block_table,
                tail_block_table,
                shared_meta,
                tail_meta,
                grouped_indices,
                unique_indices,
                batch,
                common_tokens,
                tail_width,
            )

            reference = full_launch()
            actual = stitched_launch()
            error = (actual[:, :seq_len] - reference[:, :seq_len]).abs().max()
            full_us = _time(full_launch, args.repeats)
            stitched_us = _time(stitched_launch, args.repeats)
            print(
                f"batch={batch} prefix={fraction:.2f} "
                f"shared_tokens={common_tokens} suffix_pages={suffix_pages} "
                f"full={full_us:.3f} us stitched={stitched_us:.3f} us "
                f"speedup={full_us / stitched_us:.3f}x max_error={error:.3g}"
            )


if __name__ == "__main__":
    main()
