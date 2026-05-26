"""
A/B benchmark: Triton vs CDNA-HIP per-token-head attention kernels.

Run the script twice — once with VLLM_USE_CDNA_PTH_ATTN=1 and once unset.
The dispatch path will log "[CDNA-PTH] dispatching ..." on first use so we
know the kernel actually engaged.
"""
import os
import time

import torch

from vllm import LLM, SamplingParams

MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
KV = os.environ.get("BENCH_KV", "int8_per_token_head")
TAG = "CDNA" if os.environ.get("VLLM_USE_CDNA_PTH_ATTN") == "1" else "TRITON"
SEED = 0

# Prefill regime: longish prompt, short generation.
PREFILL_PROMPT_LEN = 1024
PREFILL_NEW_TOKENS = 1
PREFILL_BATCH = 8

# Decode regime: short prompt, longer generation.
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 256
DECODE_BATCH = 8

NUM_WARMUP = 2
NUM_ITERS = 5


def make_prompts(tokenizer, n_tokens, batch):
    # Build a prompt that tokenises to roughly n_tokens. We use a simple
    # repeated-string trick — exact length isn't important, only that all
    # prompts in the batch are the same length so attention is uniform.
    seed = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer.encode(seed)
    rep = max(1, (n_tokens + len(ids) - 1) // len(ids))
    full = tokenizer.decode(ids * rep)
    # Now trim/pad to exactly n_tokens.
    full_ids = tokenizer.encode(full)[:n_tokens]
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return [text] * batch


def run_regime(llm, prompts, new_tokens, label):
    sp = SamplingParams(
        temperature=0.0, max_tokens=new_tokens, ignore_eos=True,
    )
    # Warmup.
    for _ in range(NUM_WARMUP):
        llm.generate(prompts, sp, use_tqdm=False)
    # Time.
    times = []
    for _ in range(NUM_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    total_new = len(prompts) * new_tokens
    print(
        f"[{TAG}][{label}] median={median*1000:.2f}ms "
        f"min={min(times)*1000:.2f}ms max={max(times)*1000:.2f}ms "
        f"thr={total_new/median:.1f} tok/s "
        f"(batch={len(prompts)} new_tokens={new_tokens})"
    )
    # Sanity: print first generation to confirm output non-trivial.
    print(f"[{TAG}][{label}] sample: {outs[0].outputs[0].text[:60]!r}")


def main():
    print(f"=== {TAG} run | model={MODEL} | kv={KV} ===")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=KV,
        calculate_kv_scales=True,
        max_model_len=2048,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
        seed=SEED,
        attention_config={"backend": "TRITON_ATTN"},
    )
    tok = llm.get_tokenizer()

    prefill_prompts = make_prompts(tok, PREFILL_PROMPT_LEN, PREFILL_BATCH)
    decode_prompts = make_prompts(tok, DECODE_PROMPT_LEN, DECODE_BATCH)

    run_regime(llm, prefill_prompts, PREFILL_NEW_TOKENS, "prefill")
    run_regime(llm, decode_prompts, DECODE_NEW_TOKENS, "decode")


if __name__ == "__main__":
    main()
