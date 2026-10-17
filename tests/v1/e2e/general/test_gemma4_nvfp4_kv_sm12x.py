# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end NVFP4 KV correctness for Gemma 4 on consumer Blackwell (sm120/sm121).

Gemma 4's global-attention layers have head_dim 512, which the FA2 NVFP4 path
runs as a two-pass VO split through the prefill wrapper for every request.
Historical E2B and 12B tests on SM120/SM121 found stale plan replay when FULL
decode graphs captured that per-step-planned wrapper. This regression checks
the current implementation with compilation enabled; it does not establish
quality across other models or workloads.

This fixture requires a Gemma-4-E2B checkpoint with calibrated 4-bit KV
scales, including populated scales for KV-sharing reader layers, so the
independent KV-sharing propagation bug (#55559) does not affect this test.
Set ``VLLM_TEST_GEMMA4_NVFP4KV_MODEL`` to that checkpoint. With cudagraphs
enabled, needle retrieval checks the VO-split path at several filler lengths.
"""

import os
import random

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

MODEL = os.environ.get("VLLM_TEST_GEMMA4_NVFP4KV_MODEL")
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
    not MODEL,
    reason="Set VLLM_TEST_GEMMA4_NVFP4KV_MODEL to a calibrated shared-scale fixture",
)
@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120),
    reason="NVFP4 KV on the FlashInfer FA2 path is sm120/sm121 only",
)
def test_gemma4_e2b_nvfp4_kv_needle_with_cudagraphs(vllm_runner):
    with vllm_runner(
        MODEL,
        language_model_only=True,
        attention_config={"backend": "FLASHINFER"},
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
    # Require retrieval on this fixed regression fixture; inspect completions
    # only because each prompt itself contains the expected needle.
    assert not misses, f"NVFP4 KV needle misses (filler_words, needle, got): {misses}"
