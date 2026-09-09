# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end autoregressive decode benchmark: ReplaySSM vs the standard SSM kernel.

Loads a hybrid Mamba2 model, replicates one prompt across the batch, and times a
long greedy decode (CUDA graphs on) once with the standard kernel and once with
ReplaySSM, then reports the per-step / throughput speedup. The two modes run in
separate subprocesses so each gets a clean CUDA context.

The FlashInfer FP4-MoE autotuner is disabled by default (it is unstable under
CUDA-graph capture on the pre-release Blackwell FP4 path); pass
--no-disable-flashinfer-autotune for non-FP4 models.

Examples:
    python e2e_decode_speedup.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    python e2e_decode_speedup.py --dtype auto --buffer-len 16 \
        --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4   # B300 NVFP4
"""

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_PROMPT = "My cat wrote all this CUDA code for a new language model and"

MODE_LABEL = {"standard": "standard", "replayssm": "ReplaySSM"}


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E decode speedup: ReplaySSM vs the standard SSM kernel."
    )
    p.add_argument("--model-id", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--buffer-len", type=int, default=16, help="ReplaySSM input-buffer length."
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument(
        "--disable-flashinfer-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable the FlashInfer FP4-MoE autotuner (default: on). "
        "It is unstable under CUDA-graph capture on the "
        "pre-release Blackwell FP4 path; pass "
        "--no-disable-flashinfer-autotune for non-FP4 models.",
    )
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="SSM state dtype (both modes). 'auto' = config-driven; "
        "'float32' = fp32 state, 'bfloat16' = s16 state.",
    )
    p.add_argument(
        "--baseline-ssm-config",
        default="",
        help="Pin the STANDARD baseline's SSM launch config as "
        "'bsm,nw' via override_ssm_config (forces the in-process "
        "engine so the override reaches the kernel). Empty = off.",
    )
    p.add_argument(
        "--worker",
        choices=["standard", "replayssm"],
        default=None,
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def resolve_max_model_len(args) -> int:
    if args.max_model_len is not None:
        return args.max_model_len
    return args.num_steps + 256


def run_worker(args):
    # override_ssm_config is a module global; it only reaches the model if the
    # engine runs in-process (default V1 spawns a separate EngineCore). Force it.
    if args.worker == "standard" and args.baseline_ssm_config:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch

    from vllm import LLM, SamplingParams

    mode = args.worker
    max_model_len = resolve_max_model_len(args)

    llm_kwargs = dict(
        model=args.model_id,
        tensor_parallel_size=1,
        dtype=args.dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(max_model_len, args.batch_size * 64),
        enforce_eager=False,
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # SSM state dtype (applies to both standard and ReplaySSM).
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
    )
    if args.disable_flashinfer_autotune:
        # FP4-MoE autotuner is unstable under CUDA-graph capture on Blackwell;
        # re-enable (--no-disable-flashinfer-autotune) only for non-FP4 models.
        llm_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    if mode == "replayssm":
        llm_kwargs.update(use_replayssm=True, replayssm_buffer_len=args.buffer_len)

    _ssm_cm = None
    if mode == "standard" and args.baseline_ssm_config:
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import override_ssm_config

        _bsm, _nw = (int(x) for x in args.baseline_ssm_config.split(","))
        _ssm_cm = override_ssm_config((_bsm, _nw))
        _ssm_cm.__enter__()  # active through LLM() graph capture + decode
        print(
            f"[{mode}] override_ssm_config -> (BLOCK_SIZE_M={_bsm}, num_warps={_nw})",
            flush=True,
        )

    llm = LLM(**llm_kwargs)
    prompts = [args.prompt] * args.batch_size

    def timed_generate(n_tokens):
        sp = SamplingParams(
            n=1,
            temperature=0.0,
            ignore_eos=True,
            min_tokens=n_tokens,
            max_tokens=n_tokens,
        )
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        elapsed = time.perf_counter() - t0
        produced = min(len(o.outputs[0].token_ids) for o in outs)
        assert produced == n_tokens, f"expected {n_tokens} tokens, got {produced}"
        return elapsed

    timed_generate(args.warmup_steps)

    best = None
    for _ in range(args.repeats):
        elapsed = timed_generate(args.num_steps)
        tok_s = args.batch_size * args.num_steps / elapsed
        per_step_ms = elapsed / args.num_steps * 1e3
        print(
            f"[{mode}] {elapsed:.3f}s  {tok_s:,.0f} tok/s  {per_step_ms:.3f} ms/step",
            flush=True,
        )
        if best is None or elapsed < best["elapsed_s"]:
            best = {
                "mode": mode,
                "elapsed_s": elapsed,
                "tok_s": tok_s,
                "per_step_ms": per_step_ms,
            }

    print("RESULT_JSON " + json.dumps(best), flush=True)
    if _ssm_cm is not None:
        _ssm_cm.__exit__(None, None, None)


def run_one_mode(args, mode) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        mode,
        "--model-id",
        args.model_id,
        "--prompt",
        args.prompt,
        "--batch-size",
        str(args.batch_size),
        "--num-steps",
        str(args.num_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--repeats",
        str(args.repeats),
        "--buffer-len",
        str(args.buffer_len),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--mamba-ssm-cache-dtype",
        args.mamba_ssm_cache_dtype,
        "--baseline-ssm-config",
        args.baseline_ssm_config,
    ]
    cmd.append(
        "--disable-flashinfer-autotune"
        if args.disable_flashinfer_autotune
        else "--no-disable-flashinfer-autotune"
    )
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]

    result = None
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON ") :])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"mode '{mode}' worker exited with {proc.returncode}")
    if result is None:
        raise RuntimeError(f"mode '{mode}' produced no RESULT_JSON line")
    return result


def main():
    args = parse_args()
    if args.worker is not None:
        run_worker(args)
        return

    print(
        f"model={args.model_id}  batch_size={args.batch_size}  "
        f"steps={args.num_steps}  buffer_len={args.buffer_len}  dtype={args.dtype}"
    )

    std = run_one_mode(args, "standard")
    fla = run_one_mode(args, "replayssm")
    speedup = std["per_step_ms"] / fla["per_step_ms"]

    print()
    header = f"{'mode':<10}{'ms/step':>12}{'tok/s':>16}{'wall (s)':>12}"
    print(header)
    print("-" * len(header))
    for r in (std, fla):
        print(
            f"{MODE_LABEL[r['mode']]:<10}{r['per_step_ms']:>12.3f}"
            f"{r['tok_s']:>16,.0f}{r['elapsed_s']:>12.3f}"
        )
    print("-" * len(header))
    print(f"speedup (standard / ReplaySSM, per step): {speedup:.3f}x")


if __name__ == "__main__":
    main()
