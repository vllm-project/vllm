# SPDX-License-Identifier: Apache-2.0
"""E2E test for LMCache CacheBlend on ROCm with Triton sparse attention.

Based on examples/blend_kv_v1/blend.py but adapted for ROCm:
- Uses FlashAttention backend (not FlashInfer)
- enable_sparse routes to LMCTritonSparseBackend automatically
"""

# Standard
from dataclasses import asdict
import contextlib
import os
import time

# Third Party
import pytest

# Skip entire module when vLLM is not installed (e.g. in UT environments)
vllm = pytest.importorskip("vllm", reason="vLLM required for E2E test")

# Third Party
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME  # noqa: E402
from lmcache.v1.cache_engine import LMCacheEngineBuilder  # noqa: E402


def setup_env():
    """Set up LMCache environment variables for CacheBlend."""
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = " # # "
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    # Enable sparse attention (routes to Triton backend on ROCm)
    os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

    # On ROCm, do NOT set VLLM_ATTENTION_BACKEND=FLASHINFER
    # The Triton backend works with any vLLM attention impl


@contextlib.contextmanager
def build_llm(model: str):
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def timed_generate(llm, prompt, sampling_params, label):
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
    )
    elapsed = time.time() - start
    text = outputs[0].outputs[0].text if outputs else ""
    print(f"[{label}] {elapsed:.2f}s | Generated: {text!r}")
    return elapsed


def main():
    # Standard
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    model = args.model
    print("=== LMCache CacheBlend ROCm E2E Test ===")
    print(f"Model: {model}")
    print("Backend: Triton sparse attention (auto-detected on ROCm)")

    setup_env()
    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm(model) as llm:
        # Build prompts with shared chunks (non-prefix overlap)
        blend_sep = tokenizer.encode(os.environ["LMCACHE_BLEND_SPECIAL_STR"])[
            1:
        ]  # skip BOS

        chunk_a = tokenizer.encode("The quick brown fox jumps. " * 200)[1:]
        chunk_b = tokenizer.encode("Machine learning is exciting. " * 200)[1:]
        chunk_c = tokenizer.encode("AMD GPUs are powerful for AI. " * 200)[1:]

        sys_tokens = tokenizer.encode("You are a helpful assistant. ")

        # Request 1: sys + A + B + C + question
        prompt1 = (
            sys_tokens
            + blend_sep
            + chunk_a
            + blend_sep
            + chunk_b
            + blend_sep
            + chunk_c
            + blend_sep
            + tokenizer.encode("Summarize the above.")[1:]
        )

        # Request 2: sys + B + A + C + question (reordered — non-prefix!)
        prompt2 = (
            sys_tokens
            + blend_sep
            + chunk_b
            + blend_sep
            + chunk_a
            + blend_sep
            + chunk_c
            + blend_sep
            + tokenizer.encode("What is the main topic?")[1:]
        )

        # Request 3: sys + B + A + C + different question
        prompt3 = (
            sys_tokens
            + blend_sep
            + chunk_b
            + blend_sep
            + chunk_a
            + blend_sep
            + chunk_c
            + blend_sep
            + tokenizer.encode("Tell me more about GPUs.")[1:]
        )

        sp = SamplingParams(temperature=0, max_tokens=32)

        print(
            f"\nPrompt lengths: p1={len(prompt1)}, p2={len(prompt2)}, p3={len(prompt3)}"
        )

        # Warmup
        warmup = tokenizer.encode("Hello world " * 100)[1:]
        timed_generate(llm, warmup, sp, "warmup")

        # Request 1: cold cache
        t1 = timed_generate(llm, prompt1, sp, "req1-cold")

        time.sleep(0.5)

        # Request 2: non-prefix overlap → CacheBlend should reuse chunks
        t2 = timed_generate(llm, prompt2, sp, "req2-blend")

        time.sleep(0.5)

        # Request 3: same prefix as req2 → should be faster
        t3 = timed_generate(llm, prompt3, sp, "req3-blend")

        print("\n=== Results ===")
        print(f"Cold (req1):  {t1:.2f}s")
        print(f"Blend (req2): {t2:.2f}s")
        print(f"Blend (req3): {t3:.2f}s")

        if t2 < t1 * 0.9:
            print("✅ CacheBlend speedup observed!")
        else:
            print("⚠️  No significant speedup (may need longer prompts)")

        print("\n✅ E2E test completed without errors!")


if __name__ == "__main__":
    main()
