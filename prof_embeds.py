#!/usr/bin/env python3
"""Profile Prefill: input_ids (text) vs prompt_embeds at matched input length.

Example: Latency only:

    VLLM_SHM_EMBEDS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MAX_JOBS=1 \\
    PROF_EAGER=1 PROF_N=24000 PROF_K=3 CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    uv run --no-project python prof_embeds.py

Example: With an nsys capture (adds the .nsys-rep):

    VLLM_SHM_EMBEDS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MAX_JOBS=1 \\
    PROF_EAGER=1 PROF_N=24000 PROF_K=3 CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    nsys profile --trace=cuda,nvtx,osrt -s none \\
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \\
        -o prof_updated \\
        uv run --no-project python prof_embeds.py
"""

import os
import random
import statistics
import time

import torch

from vllm import LLM, SamplingParams, TokensPrompt

N = int(os.environ.get("PROF_N", "24000"))  # prompt length (tokens)
K = int(os.environ.get("PROF_K", "4"))  # repeats per phase
MODEL = os.environ.get("PROF_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
TP = int(os.environ.get("PROF_TP", "4"))
MAXLEN = int(os.environ.get("PROF_MAXLEN", "131072"))

from transformers import AutoConfig  # noqa: E402

_cfg = AutoConfig.from_pretrained(MODEL)
H = _cfg.hidden_size  # hidden size for the embeds tensor
VOCAB = _cfg.vocab_size  # vocab size for the model


def tok_prompt():
    return TokensPrompt(
        prompt_token_ids=[random.randrange(1, VOCAB - 100) for _ in range(N)]
    )


def emb_prompt():
    # bf16 [N, H]; values don't matter for kernel timing, only shape/dtype/path.
    return {"prompt_embeds": torch.randn(N, H, dtype=torch.bfloat16)}


def main():
    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=MAXLEN,
        enable_prompt_embeds=True,
        enable_prefix_caching=False,  # Force full prefill every request for a clean comparison.
        gpu_memory_utilization=0.80,
        enforce_eager=bool(int(os.environ.get("PROF_EAGER", "0"))),
    )
    if "Nemotron" in MODEL:  # hybrid-mamba + FP8 MoE specific args.
        kwargs.update(
            kv_cache_dtype="fp8",
            mamba_ssm_cache_dtype="float16",
            enable_expert_parallel=True,
        )
    llm = LLM(**kwargs)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup each path BEFORE the capture range.
    print(">>> Warmup", flush=True)
    llm.generate([tok_prompt()], sp, use_tqdm=False)
    llm.generate([emb_prompt()], sp, use_tqdm=False)

    torch.cuda.set_device(0)
    _ctx = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    # Pre-build ALL prompts OUTSIDE the timed window.
    text_prompts = [tok_prompt() for _ in range(K)]
    embed_prompts = [emb_prompt() for _ in range(K)]

    print(">>> START PROFILE", flush=True)
    torch.cuda.profiler.start()
    text_t, emb_t = [], []

    # Text prefill
    for i in range(K):
        torch.cuda.nvtx.range_push("TEXT_PREFILL")
        t = time.perf_counter()
        llm.generate([text_prompts[i]], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        torch.cuda.nvtx.range_pop()
        text_t.append(dt)
        print(f"text  {i}: {dt:.3f}s", flush=True)

    # Embed prefill
    for i in range(K):
        torch.cuda.nvtx.range_push("EMBED_PREFILL")
        t = time.perf_counter()
        llm.generate([embed_prompts[i]], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        torch.cuda.nvtx.range_pop()
        emb_t.append(dt)
        print(f"embed {i}: {dt:.3f}s", flush=True)
    torch.cuda.profiler.stop()

    print(
        f">>> RESULT N={N} text_median={statistics.median(text_t):.3f}s "
        f"embed_median={statistics.median(emb_t):.3f}s "
        f"premium={statistics.median(emb_t) / statistics.median(text_t):.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
