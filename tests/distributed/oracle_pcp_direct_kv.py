#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PCP fused oracle entry: fp8_ds_mla + Indexer-K on a packed backing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.distributed.test_pcp_symm_kv import run_fp8_ds_mla_indexer_oracle


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    run_fp8_ds_mla_indexer_oracle()
    print(f"rank{rank} fp8_ds_mla_indexer_match=True", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
