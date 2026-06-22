# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P1 baseline benchmark for MLA + TurboQuant decode throughput.

Measures TTFT, decode tok/s, and peak HBM across kv_cache_dtype x batch.

Usage:
    conda run -n vllm-mla-tq python tools/bench_mla_tq_decode.py \
        --dtypes auto turboquant_k8v4 turboquant_4bit_nc \
                 turboquant_k3v4_nc turboquant_3bit_nc \
        --batches 1 4 16 --prompt-len 512 --gen-len 256

Output:
    tools/reports/p1_baseline.md (markdown table, appended)
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402

REPORT_PATH = Path(
    os.environ.get(
        "REPORT",
        str(Path(__file__).resolve().parent / "reports" / "p1_baseline.md"),
    )
)
MODEL_PATH = os.environ.get("MODEL", "/path/to/DeepSeek-V2-Lite")


def make_prompt(prompt_len: int, tokenizer) -> str:
    # Pick a deterministic, prose-like prompt so token count is stable.
    base = (
        "The history of artificial intelligence began in antiquity with myths "
        "and stories of artificial beings endowed with intelligence. "
    )
    text = base * 200
    ids = tokenizer.encode(text)[:prompt_len]
    return tokenizer.decode(ids)


def bench_one(
    dtype: str,
    batch: int,
    prompt_len: int,
    gen_len: int,
    tp: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    chunked_prefill: bool = False,
    prefix_caching: bool = False,
) -> dict:
    from vllm import LLM, SamplingParams

    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        kv_cache_dtype=dtype,
        enforce_eager=enforce_eager,
        enable_prefix_caching=prefix_caching,
        enable_chunked_prefill=chunked_prefill,
        max_model_len=max_model_len,
        max_num_seqs=max(batch, 1),
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    tok = llm.get_tokenizer()
    prompt = make_prompt(prompt_len, tok)
    prompts = [prompt] * batch

    # ---------- TTFT pass: prompt + 1 token ----------
    sp_ttft = SamplingParams(temperature=0.0, max_tokens=1)
    # warmup
    llm.generate(prompts, sp_ttft, use_tqdm=False)
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sp_ttft, use_tqdm=False)
    torch.accelerator.synchronize()
    ttft_s = time.perf_counter() - t0

    # ---------- decode pass: prompt + gen_len ----------
    sp_dec = SamplingParams(temperature=0.0, max_tokens=gen_len)
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp_dec, use_tqdm=False)
    torch.accelerator.synchronize()
    total_s = time.perf_counter() - t0

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    decode_only_s = max(total_s - ttft_s, 1e-6)
    decode_tps = gen_tokens / decode_only_s
    peak_gb = torch.accelerator.max_memory_allocated() / (1024**3)

    res = dict(
        dtype=dtype,
        batch=batch,
        ttft_ms=ttft_s * 1000.0,
        decode_tps=decode_tps,
        gen_tokens=gen_tokens,
        peak_hbm_gb=peak_gb,
    )

    # release model
    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    return res


def append_report(rows: list[dict], note: str = "") -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not REPORT_PATH.exists()
    with REPORT_PATH.open("a") as f:
        if new:
            f.write("# P1 baseline — MLA + TurboQuant decode\n\n")
        f.write(f"\n## Run @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if note:
            f.write(f"\n_{note}_\n")
        f.write(
            "\n| dtype | batch | TTFT (ms) | decode tok/s | "
            "gen toks | peak HBM (GB) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['dtype']} | {r['batch']} | "
                f"{r['ttft_ms']:.1f} | {r['decode_tps']:.2f} | "
                f"{r['gen_tokens']} | {r['peak_hbm_gb']:.2f} |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=[
            "auto",
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    )
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Disable enforce_eager (lets vLLM use PIECEWISE/FULL CG)",
    )
    ap.add_argument(
        "--chunked-prefill", action="store_true", help="Enable chunked prefill (P3-3)"
    )
    ap.add_argument(
        "--prefix-caching", action="store_true", help="Enable prefix caching (P3-3)"
    )
    args = ap.parse_args()

    enforce_eager = not args.no_enforce_eager
    max_model_len = args.prompt_len + args.gen_len + 16
    rows = []
    for dtype in args.dtypes:
        for batch in args.batches:
            print(f"\n>>> bench dtype={dtype} batch={batch} eager={enforce_eager}")
            try:
                r = bench_one(
                    dtype,
                    batch,
                    args.prompt_len,
                    args.gen_len,
                    args.tp,
                    max_model_len,
                    args.gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    chunked_prefill=args.chunked_prefill,
                    prefix_caching=args.prefix_caching,
                )
            except Exception as e:
                print(f"  FAILED: {e!r}")
                r = dict(
                    dtype=dtype,
                    batch=batch,
                    ttft_ms=float("nan"),
                    decode_tps=float("nan"),
                    gen_tokens=0,
                    peak_hbm_gb=float("nan"),
                )
            print(
                f"  TTFT={r['ttft_ms']:.1f}ms  "
                f"decode={r['decode_tps']:.2f} tok/s  "
                f"peak={r['peak_hbm_gb']:.2f} GB"
            )
            rows.append(r)

    append_report(rows, args.note)
    print(f"\nReport appended → {REPORT_PATH}")


if __name__ == "__main__":
    main()
