# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end speculative-decode throughput: AR vs standard spec vs ReplaySSM spec.

Real GSM8K prompts (chat-formatted) at a fixed batch size, CUDA graphs on, with
ignore_eos so every mode emits exactly --max-tokens tokens per sequence, making
tokens/s directly comparable. Reports throughput, mean acceptance length, and
the ReplaySSM-spec speedup over both baselines.

  ar       : no speculative decoding
  standard : vLLM native spec decoding (one recurrent state per draft token)
  cache    : ReplaySSM cached spec decoding (use_replayssm_spec)

Each mode runs in its own subprocess for a clean CUDA context.

The FlashInfer FP4-MoE autotuner is disabled by default (it is unstable under
CUDA-graph capture on the pre-release Blackwell FP4 path); pass
--no-disable-flashinfer-autotune for non-FP4 models.

Examples (B300):
    python e2e_spec_decode_throughput.py --batch-size 512 \
        --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
    python e2e_spec_decode_throughput.py --batch-size 512 \
        --model-id nvidia/Qwen3.5-122B-A10B-NVFP4 --spec-method qwen3_next_mtp \
        --moe-backend triton
"""

import argparse
import json
import subprocess
import sys
import time

MODE_LABEL = {"ar": "AR", "standard": "standard-spec", "cache": "ReplaySSM-spec"}


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E spec-decode throughput: AR vs standard vs ReplaySSM."
    )
    p.add_argument("--model-id",
                   default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-spec", type=int, default=3,
                   help="Draft tokens per step (spec window = num_spec + 1).")
    p.add_argument("--spec-method", default="mtp")
    p.add_argument("--buffer-len", type=int, default=16,
                   help="ReplaySSM buffer length (power of two, >= 1 + num_spec).")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--warmup-s", type=float, default=30.0,
                   help="Sustained full-batch decode before timing, to ramp the "
                        "GPU SM clock to its steady-state boost.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Keep reasoning mode on (default off for short outputs).")
    p.add_argument("--disable-flashinfer-autotune",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Disable the FlashInfer FP4-MoE autotuner (default: on). "
                        "It is unstable under CUDA-graph capture on the "
                        "pre-release Blackwell FP4 path; pass "
                        "--no-disable-flashinfer-autotune for non-FP4 models.")
    p.add_argument("--modes", default="ar,standard,cache",
                   help="Comma-separated subset of {ar,standard,cache}.")
    p.add_argument("--moe-backend", default=None,
                   help="Override the draft-model MoE backend (e.g. triton). On "
                        "Blackwell the draft's bf16 MoE hangs on flashinfer_trtllm; "
                        "pass triton. Main-model MoE is left at auto.")
    p.add_argument("--worker", choices=["ar", "standard", "cache"], default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def gsm8k_messages(batch_size):
    from datasets import load_dataset

    questions = [r["question"]
                 for r in load_dataset("openai/gsm8k", "main", split="test")]
    return [[{"role": "user", "content": questions[i % len(questions)]}]
            for i in range(batch_size)]


def _sum_counter(metrics, name):
    return sum(
        m.value for m in metrics
        if getattr(m, "name", None) == name and hasattr(m, "value")
    )


def run_worker(args):
    import torch

    from vllm import LLM, SamplingParams

    mode = args.worker
    spec_window = 1 if mode == "ar" else 1 + args.num_spec

    llm_kwargs = dict(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        disable_log_stats=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        # Avoid the FlashInfer GDN-prefill cutlass-DSL JIT stall on Blackwell
        # (same default as the decode benchmark).
        additional_config={"gdn_prefill_backend": "triton"},
        compilation_config={
            "max_cudagraph_capture_size": max(8, args.batch_size * spec_window)
        },
    )
    if args.disable_flashinfer_autotune:
        # FP4-MoE autotuner is unstable under CUDA-graph capture on Blackwell;
        # re-enable (--no-disable-flashinfer-autotune) only for non-FP4 models.
        llm_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    if mode != "ar":
        spec_cfg = {
            "method": args.spec_method,
            "num_speculative_tokens": args.num_spec,
        }
        # Override only the draft MoE backend (the trtllm hang is in the draft).
        if args.moe_backend:
            spec_cfg["moe_backend"] = args.moe_backend
        llm_kwargs["speculative_config"] = spec_cfg
    if mode == "cache":
        llm_kwargs["use_replayssm_spec"] = True
        llm_kwargs["replayssm_buffer_len"] = args.buffer_len

    llm = LLM(**llm_kwargs)
    messages = gsm8k_messages(args.batch_size)
    chat_kwargs = {"enable_thinking": args.enable_thinking}
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens,
                        ignore_eos=True, seed=0)

    def timed_chat():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.chat(messages, sp, chat_template_kwargs=chat_kwargs,
                        use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, outs

    deadline = time.perf_counter() + args.warmup_s
    while time.perf_counter() < deadline:
        timed_chat()

    pre_acc = _sum_counter(llm.get_metrics(), "vllm:spec_decode_num_accepted_tokens")
    pre_dft = _sum_counter(llm.get_metrics(), "vllm:spec_decode_num_drafts")

    elapsed, outs = timed_chat()
    produced = min(len(o.outputs[0].token_ids) for o in outs)
    assert produced == args.max_tokens, f"expected {args.max_tokens}, got {produced}"
    tok_s = args.batch_size * args.max_tokens / elapsed

    accept_len = None
    if mode != "ar":
        drafts = _sum_counter(llm.get_metrics(),
                              "vllm:spec_decode_num_drafts") - pre_dft
        accepted = _sum_counter(llm.get_metrics(),
                                "vllm:spec_decode_num_accepted_tokens") - pre_acc
        accept_len = 1.0 + accepted / drafts if drafts else None

    result = {"mode": mode, "elapsed_s": elapsed, "tok_s": tok_s,
              "accept_len": accept_len}
    print(f"[{mode}] {elapsed:.2f}s  {tok_s:,.0f} tok/s"
          + (f"  accept_len={accept_len:.2f}" if accept_len else ""), flush=True)
    print("RESULT_JSON " + json.dumps(result), flush=True)


def run_one_mode(args, mode):
    cmd = [
        sys.executable, __file__, "--worker", mode,
        "--model-id", args.model_id,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--batch-size", str(args.batch_size), "--num-spec", str(args.num_spec),
        "--spec-method", args.spec_method, "--buffer-len", str(args.buffer_len),
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len), "--dtype", args.dtype,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--warmup-s", str(args.warmup_s),
    ]
    if args.enable_thinking:
        cmd.append("--enable-thinking")
    if args.moe_backend:
        cmd += ["--moe-backend", args.moe_backend]
    cmd.append("--disable-flashinfer-autotune" if args.disable_flashinfer_autotune
               else "--no-disable-flashinfer-autotune")

    result = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
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

    print(f"model={args.model_id}  tp={args.tensor_parallel_size}  "
          f"batch_size={args.batch_size}  num_spec={args.num_spec}  "
          f"buffer_len={args.buffer_len}  max_tokens={args.max_tokens}")

    modes = [m for m in args.modes.split(",") if m]
    results = {m: run_one_mode(args, m) for m in modes}

    print()
    header = f"{'mode':<16}{'tok/s':>14}{'accept_len':>12}{'wall (s)':>12}"
    print(header)
    print("-" * len(header))
    for m in modes:
        r = results[m]
        al = f"{r['accept_len']:.2f}" if r["accept_len"] else "-"
        print(f"{MODE_LABEL[m]:<16}{r['tok_s']:>14,.0f}{al:>12}{r['elapsed_s']:>12.2f}")
    print("-" * len(header))
    if "cache" in results and "standard" in results:
        print(f"ReplaySSM / standard : "
              f"{results['cache']['tok_s'] / results['standard']['tok_s']:.2f}x")
    if "cache" in results and "ar" in results:
        print(f"ReplaySSM / AR       : "
              f"{results['cache']['tok_s'] / results['ar']['tok_s']:.2f}x")


if __name__ == "__main__":
    main()
