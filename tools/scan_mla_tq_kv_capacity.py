# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""§3.1 KV-cache capacity scan: how many decode tokens fit per dtype.

For each kv_cache_dtype, init vLLM at fixed max_model_len + gpu_mem_util,
read the resulting GPU KV cache size (tokens) and per-token bytes from
vllm.v1.core.kv_cache_utils, then write a markdown row.

Usage:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    MODEL=/data/modelscope/DeepSeek-V2-Lite \
        python tools/scan_mla_tq_kv_capacity.py \
            --dtypes auto turboquant_k8v4 turboquant_4bit_nc turboquant_3bit_nc
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPORT = Path(
    os.environ.get(
        "REPORT",
        str(Path(__file__).resolve().parent / "reports" / "p3_1_kv_capacity.md"),
    )
)
MODEL_PATH = os.environ.get("MODEL", "/data/modelscope/DeepSeek-V2-Lite")


def probe(dtype: str, tp: int, max_model_len: int, gpu_mem: float) -> dict:
    from vllm import LLM

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        kv_cache_dtype=dtype,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=os.environ.get("ENFORCE_EAGER", "1") == "1",
    )

    # Pull cache config from engine
    cfg = llm.llm_engine.vllm_config
    kv_cfg = cfg.cache_config
    num_gpu_blocks = kv_cfg.num_gpu_blocks or 0
    block_size = kv_cfg.block_size
    total_tokens = num_gpu_blocks * block_size
    res = dict(
        dtype=dtype,
        max_model_len=max_model_len,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        total_tokens=total_tokens,
        max_concurrency=total_tokens / max_model_len if max_model_len else 0.0,
    )

    del llm
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=[
            "auto",
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_3bit_nc",
        ],
    )
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    rows = []
    for dt in args.dtypes:
        print(f">>> probe dtype={dt}")
        try:
            r = probe(dt, args.tp, args.max_model_len, args.gpu_memory_utilization)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            r = dict(
                dtype=dt,
                max_model_len=args.max_model_len,
                num_gpu_blocks=0,
                block_size=0,
                total_tokens=0,
                max_concurrency=0.0,
            )
        print(
            f"  blocks={r['num_gpu_blocks']} bs={r['block_size']} "
            f"toks={r['total_tokens']} conc={r['max_concurrency']:.1f}x"
        )
        rows.append(r)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    new = not REPORT.exists()
    with REPORT.open("a") as f:
        if new:
            f.write("# §3.1 KV-cache capacity per dtype\n\n")
        f.write(f"\n## Run @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.note:
            f.write(f"\n_{args.note}_\n")
        f.write(
            f"\nmax_model_len={args.max_model_len}  tp={args.tp}  "
            f"gpu_mem_util={args.gpu_memory_utilization}\n\n"
        )
        f.write(
            "| dtype | gpu blocks | block_size | total tokens | max concurrency |\n"
        )
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['dtype']} | {r['num_gpu_blocks']} | {r['block_size']} | "
                f"{r['total_tokens']} | {r['max_concurrency']:.1f}x |\n"
            )
    print(f"\nReport appended → {REPORT}")


if __name__ == "__main__":
    main()
