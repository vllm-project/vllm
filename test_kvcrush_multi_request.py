"""
test_kvcrush_multi_request.py — Targeted test for KVCrush compression with
multiple concurrent requests through the full vLLM pipeline.

Exercises the KVCrush compression loop in flash_attn.py's forward() with:
  - Multiple requests in a single batch (tests occupied_slot_mapping partitioning)
  - Sequences long enough to trigger compression
  - Mix of short (skip compression) and long (trigger compression) prompts
  - Multiple decode steps after compression (verifies post-compression decode)

Usage :
    VLLM_V1_KVC_BUDGET=128 \
    VLLM_V1_KVCRUSH_RATIO=0 \
    VLLM_V1_KVCRUSH_START_SIZE=4 \
    VLLM_V1_KVCRUSH_RECENT_SIZE=8 \
    python test_kvcrush_multi_request.py

    # Quick smoke test (2 requests, short):
    VLLM_TARGET_DEVICE=xpu \
    VLLM_V1_KVC_BUDGET=64 \
    VLLM_V1_KVCRUSH_START_SIZE=4 \
    VLLM_V1_KVCRUSH_RECENT_SIZE=8 \
    python test_kvcrush_multi_request.py --scenario smoke

    # All scenarios:
    python test_kvcrush_multi_request.py --scenario all
"""
import argparse
import os
import sys
import time

# Set KVCrush env vars BEFORE importing vllm (they're read at import time)
os.environ.setdefault("VLLM_V1_KVC_BUDGET", "128")
os.environ.setdefault("VLLM_V1_KVCRUSH_RATIO", "0")
os.environ.setdefault("VLLM_V1_KVCRUSH_START_SIZE", "4")
os.environ.setdefault("VLLM_V1_KVCRUSH_RECENT_SIZE", "8")
os.environ.setdefault("VLLM_V1_KVCRUSH_WINDOW_SIZE", "0")


def parse_args():
    p = argparse.ArgumentParser(
        description="Targeted KVCrush multi-request test")
    p.add_argument("--model", type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="Model to use (small model recommended)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--scenario", type=str, default="all",
                   choices=["smoke", "multi_basic", "mixed_lengths",
                            "concurrent_compress", "post_compress_decode",
                            "all"],
                   help="Which test scenario to run")
    p.add_argument("--output-len", type=int, default=16,
                   help="Decode tokens per request")
    return p.parse_args()


def build_prompt(tokenizer, target_len: int) -> str:
    """Build a prompt that tokenizes to approximately target_len tokens."""
    filler = "The quick brown fox jumps over the lazy dog. "
    filler_tokens = len(tokenizer.encode(filler, add_special_tokens=False))
    repeats = max(1, (target_len // filler_tokens) + 1)
    long_text = filler * repeats
    token_ids = tokenizer.encode(long_text,
                                 add_special_tokens=False)[:target_len]
    return tokenizer.decode(token_ids)


def get_kvc_threshold() -> int:
    """Minimum seq_len for compression to trigger."""
    budget = int(os.environ.get("VLLM_V1_KVC_BUDGET", "128"))
    start = int(os.environ.get("VLLM_V1_KVCRUSH_START_SIZE", "32"))
    recent = int(os.environ.get("VLLM_V1_KVCRUSH_RECENT_SIZE", "128"))
    return budget + start + recent


def run_scenario(name, llm, tokenizer, prompts, output_len, expect_compress):
    """Run a scenario and report results."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")
    print(f"  Requests: {len(prompts)}")
    for i, p in enumerate(prompts):
        prompt_len = len(tokenizer.encode(p))
        print(f"  Prompt[{i}]: {prompt_len} tokens")
    print(f"  Output tokens: {output_len}")
    print(f"  Compression threshold: {get_kvc_threshold()} tokens")
    print(f"  Expect compression: {expect_compress}")

    t0 = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling_params)
        t1 = time.perf_counter()

        print(f"  Time: {t1 - t0:.2f}s")
        all_ok = True
        for i, out in enumerate(outputs):
            n_out = len(out.outputs[0].token_ids)
            text_preview = out.outputs[0].text[:80].replace('\n', ' ')
            status = "OK" if n_out > 0 else "FAIL"
            if n_out == 0:
                all_ok = False
            print(f"  Output[{i}]: {n_out} tokens, status={status}, "
                  f"text='{text_preview}...'")

        if all_ok:
            print(f"  RESULT: PASS")
        else:
            print(f"  RESULT: FAIL (some outputs empty)")
        return all_ok

    except Exception as e:
        print(f"  RESULT: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    budget = int(os.environ.get("VLLM_V1_KVC_BUDGET", "128"))
    print(f"KVCrush config:")
    print(f"  VLLM_V1_KVC_BUDGET={budget}")
    print(f"  VLLM_V1_KVCRUSH_RATIO={os.environ.get('VLLM_V1_KVCRUSH_RATIO')}")
    print(f"  VLLM_V1_KVCRUSH_START_SIZE="
          f"{os.environ.get('VLLM_V1_KVCRUSH_START_SIZE')}")
    print(f"  VLLM_V1_KVCRUSH_RECENT_SIZE="
          f"{os.environ.get('VLLM_V1_KVCRUSH_RECENT_SIZE')}")
    print(f"  Compression threshold: {get_kvc_threshold()} tokens")
    print()

    if budget == 0:
        print("ERROR: VLLM_V1_KVC_BUDGET=0 — compression disabled. "
              "Set it to e.g. 128.")
        sys.exit(1)

    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Initialize LLM once, reuse across scenarios
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # max_num_seqs=8,
        disable_log_stats=False,
    )
    print("Model loaded.\n")

    threshold = get_kvc_threshold()
    # Prompts above threshold will trigger compression on first decode step
    long_len = threshold + 64   # comfortably above threshold
    short_len = threshold // 2  # below threshold, no compression

    long_prompt = build_prompt(tokenizer, long_len)
    short_prompt = build_prompt(tokenizer, short_len)

    scenarios_to_run = []

    if args.scenario in ("smoke", "all"):
        # Scenario 1: Smoke test — 2 identical long requests
        scenarios_to_run.append((
            "smoke: 2 identical long requests",
            [long_prompt, long_prompt],
            args.output_len,
            True,
        ))

    if args.scenario in ("multi_basic", "all"):
        # Scenario 2: Multiple requests (4), all long enough to compress
        scenarios_to_run.append((
            "multi_basic: 4 long requests",
            [long_prompt] * 4,
            args.output_len,
            True,
        ))

    if args.scenario in ("mixed_lengths", "all"):
        # Scenario 3: Mix of short (no compress) and long (compress)
        # Tests that occupied_slot_start tracking works when some
        # requests skip and some compress
        scenarios_to_run.append((
            "mixed_lengths: short + long + short + long",
            [short_prompt, long_prompt, short_prompt, long_prompt],
            args.output_len,
            True,  # long ones should compress
        ))

    if args.scenario in ("concurrent_compress", "all"):
        # Scenario 4: Many concurrent requests, all compressible
        # Stress test for the per-request compression loop
        scenarios_to_run.append((
            "concurrent_compress: 8 long requests",
            [long_prompt] * 8,
            args.output_len,
            True,
        ))

    if args.scenario in ("post_compress_decode", "all"):
        # Scenario 5: Long decode after compression
        # Tests that decode steps after compression work correctly
        # (compressed KV cache + relocated decode token)
        scenarios_to_run.append((
            "post_compress_decode: 2 long requests, 64 decode tokens",
            [long_prompt, long_prompt],
            64,  # override output_len for this scenario
            True,
        ))

    # Run all selected scenarios
    results = {}
    for name, prompts, out_len, expect in scenarios_to_run:
        ok = run_scenario(name, llm, tokenizer, prompts, out_len, expect)
        results[name] = ok

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
