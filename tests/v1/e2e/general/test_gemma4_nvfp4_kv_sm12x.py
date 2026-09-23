# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end NVFP4 KV correctness for Gemma 4 on consumer Blackwell (sm120/sm121).

Gemma 4's global-attention layers have head_dim 512, which the FA2 NVFP4 path
runs as a two-pass VO split through the prefill wrapper for every request. Two
regressions only showed up end to end, under the *default* compilation config,
and were invisible to the kernel and CPU unit tests:

* FULL cudagraph decode capture around the per-step-planned VO-split prefill
  wrapper replayed stale plan data: every Gemma 4 decoded garbage with NVFP4 KV
  unless ``enforce_eager`` was set (chat needle retrieval 0/8 vs 8/8).
* KV-sharing layers (E-series ``num_kv_shared_layers``) dequantized the target
  layer's cache with k/v scale 1.0.

This test serves a Gemma-4-E2B checkpoint that carries calibrated 4-bit KV
scales (public "NVFP4" Gemma 4 checkpoints have none and are expected to
produce garbage), with cudagraphs on, and checks needle retrieval through a
chat-template prompt at several filler lengths. A model with calibrated NVFP4
KV scales can be pointed at with ``VLLM_TEST_GEMMA4_NVFP4KV_MODEL``.
"""

import os
import random

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

MODEL = os.environ.get(
    "VLLM_TEST_GEMMA4_NVFP4KV_MODEL", "jethachan/gemma-4-E2B-it-NVFP4KV-calib"
)
WORDS = [
    "the", "river", "runs", "quietly", "between", "old", "stones", "and",
    "the", "wind", "carries", "dust", "over", "the", "fields", "while",
    "travellers", "rest", "under", "tall", "trees",
]  # fmt: skip
# Word counts of neutral filler around the needle. 0 exercises the 34-token
# prompt that already failed under the cudagraph bug; the long ones cross
# several KV pages so prefill/decode both touch the shared cache.
FILLER_WORDS = (0, 50, 200, 800)


def _needle_prompt(n_words: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    needle = str(rng.randint(10000, 99999))
    filler = lambda n: " ".join(rng.choice(WORDS) for _ in range(n))  # noqa: E731
    head, tail = filler(n_words // 2), filler(n_words - n_words // 2)
    body = f"{head} The secret code is {needle}. {tail}"
    return f"{body}\n\nWhat is the secret code? Answer with just the number.", needle


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120),
    reason="NVFP4 KV on the FlashInfer FA2 path is sm120/sm121 only",
)
def test_gemma4_e2b_nvfp4_kv_needle_with_cudagraphs(vllm_runner):
    with vllm_runner(
        MODEL,
        kv_cache_dtype="nvfp4",
        max_model_len=4096,
        max_num_seqs=2,
        gpu_memory_utilization=0.6,
        enforce_eager=False,  # the regression only reproduces with graphs
        # Keep decode capture sizes so a FULL decode graph is attempted when the
        # backend advertises support for it (which it must not for VO-split).
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
        enable_prefix_caching=True,
        # VllmRunner turns chunked prefill off by default; the serving default
        # is on, and with it off the decode batches never reach the shape that
        # dispatches the FULL graph, which hid the regression entirely.
        enable_chunked_prefill=True,
    ) as vllm_model:
        tokenizer = vllm_model.llm.get_tokenizer()
        cases = [_needle_prompt(n, seed=n + 1) for n in FILLER_WORDS]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q, _ in cases
        ]
        # Read the completion only: VllmRunner.generate_greedy returns
        # prompt + completion, and the prompt contains the needle.
        req_outputs = vllm_model.llm.generate(
            prompts, SamplingParams(temperature=0.0, max_tokens=16)
        )
        answers = [out.outputs[0].text for out in req_outputs]

    misses = [
        (n, needle, text.strip()[:40])
        for n, (_, needle), text in zip(FILLER_WORDS, cases, answers)
        if needle not in text
    ]
    # bf16 KV retrieves all of these; NVFP4 KV must too. Allow no misses:
    # the failure modes here are all-or-nothing (garbage decodes, 0/N).
    assert not misses, f"NVFP4 KV needle misses (filler_words, needle, got): {misses}"
