# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end decode benchmark: SketchSSM vs ReplaySSM vs the standard SSM decode.

Loads a hybrid Mamba-2 model, replicates one prompt across the batch, and times
a long greedy decode (CUDA graphs on) with each mode in its own subprocess. All
modes use an FP32 SSM cache; standard and SketchSSM run on Model Runner V2, which
SketchSSM requires, and ReplaySSM (Triton) on Model Runner V1, which it requires.

Example:
    python benchmarks/sketchssm/e2e_decode_speedup.py \\
        --model-id nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \\
        --sketchssm SketchSSM/Nemotron-3-Nano-4B-BF16 --sketchssm-mean-rank 8

"""

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_PROMPT = "My cat wrote all this CUDA code for a new language model and"
MODES = {"standard": "standard", "replayssm": "ReplaySSM", "sketchssm": "SketchSSM"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-id", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    p.add_argument("--sketchssm", default="SketchSSM/Nemotron-3-Nano-4B-BF16")
    p.add_argument("--sketchssm-mean-rank", type=float, default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--buffer-len", type=int, default=16, help="Window length W.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--worker", choices=list(MODES), help=argparse.SUPPRESS)
    return p.parse_args()


def run_worker(args) -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0" if args.worker == "replayssm" else "1"
    import torch

    from vllm import LLM, SamplingParams

    max_model_len = args.num_steps + 256
    llm_kwargs = dict(
        model=args.model_id,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(max_model_len, args.batch_size * 64),
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mamba_ssm_cache_dtype="float32",
    )
    if args.worker == "replayssm":
        llm_kwargs.update(use_replayssm=True, replayssm_buffer_len=args.buffer_len)
    elif args.worker == "sketchssm":
        llm_kwargs.update(
            sketchssm=args.sketchssm,
            sketchssm_mean_rank=args.sketchssm_mean_rank,
            replayssm_buffer_len=args.buffer_len,
        )
    llm = LLM(**llm_kwargs)
    prompts = [args.prompt] * args.batch_size

    def timed_generate(n_tokens: int) -> float:
        params = SamplingParams(
            temperature=0.0, ignore_eos=True, min_tokens=n_tokens, max_tokens=n_tokens
        )
        torch.accelerator.synchronize()
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        torch.accelerator.synchronize()
        assert min(len(o.outputs[0].token_ids) for o in outs) == n_tokens
        return time.perf_counter() - start

    timed_generate(args.warmup_steps)
    elapsed = min(timed_generate(args.num_steps) for _ in range(args.repeats))
    print(
        "RESULT_JSON "
        + json.dumps(
            dict(
                mode=args.worker,
                per_step_ms=elapsed / args.num_steps * 1e3,
                tok_s=args.batch_size * args.num_steps / elapsed,
            )
        ),
        flush=True,
    )


def run_mode(args, mode: str) -> dict:
    cmd = [sys.executable, __file__, "--worker", mode]
    for name in (
        "model_id",
        "sketchssm",
        "sketchssm_mean_rank",
        "prompt",
        "batch_size",
        "num_steps",
        "warmup_steps",
        "repeats",
        "buffer_len",
        "gpu_memory_utilization",
    ):
        value = getattr(args, name)
        if value is not None:
            cmd += [f"--{name.replace('_', '-')}", str(value)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mode {mode!r} exited with {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    raise RuntimeError(f"mode {mode!r} produced no result")


def main() -> None:
    args = parse_args()
    if args.worker is not None:
        run_worker(args)
        return
    print(
        f"model={args.model_id} batch_size={args.batch_size} "
        f"steps={args.num_steps} W={args.buffer_len} "
        f"sketchssm={args.sketchssm} mean_rank={args.sketchssm_mean_rank}"
    )
    results = [run_mode(args, mode) for mode in args.modes]
    base = results[0]["per_step_ms"]
    print(f"{'mode':<12}{'ms/step':>10}{'tok/s':>14}{'speedup':>10}")
    for r in results:
        print(
            f"{MODES[r['mode']]:<12}{r['per_step_ms']:>10.3f}"
            f"{r['tok_s']:>14,.0f}{base / r['per_step_ms']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
