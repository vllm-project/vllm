# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for hybrid GDN/Mamba models under sleep -> wake.

Hybrid Mamba / gated-delta-net (GDN) models (e.g. Qwen3-Next) keep a
persisted conv + recurrent state cache. With sleep mode (the RLHF reuse
pattern: ``sleep()`` -> weight update -> ``wake_up()``) the state-cache tag is
discarded on sleep and its device memory is re-created on wake. If a *new*
sequence's state slot is consumed before being reset, the gated-delta-rule
kernel faithfully propagates whatever is in that (now non-zeroed) memory; when
it contains NaN/inf the output becomes NaN and ``argmax`` collapses every token
to id 0 (which decodes to ``"!"``), giving ``reward=0`` / NaN log-probs in RL
training.

This test sleeps and wakes a small hybrid GDN model and asserts that
post-wake generation is neither degenerate (single repeated token) nor NaN.
"""

import pytest

from vllm import LLM, SamplingParams

# Small Qwen3-Next (GDN) model already used by the hybrid model test-suite.
MODEL = "tiny-random/qwen3-next-moe"

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "1, 2, 3, 4,",
    "Water is made of",
]


@pytest.mark.hybrid_model
def test_gdn_sleep_wake_no_stale_state():
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32, logprobs=1)

    llm = LLM(
        model=MODEL,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
    )

    # Warm generation before sleeping.
    llm.generate(PROMPTS, sampling_params)

    # Default sleep offloads weights and DISCARDS the kv / GDN state cache;
    # wake_up re-creates that memory (fresh, not guaranteed zeroed).
    llm.sleep()
    llm.wake_up()

    after = llm.generate(PROMPTS, sampling_params)

    for output in after:
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        assert token_ids, "empty generation after wake_up"
        # The bug collapses every token to a single id (e.g. 0 -> "!").
        assert len(set(token_ids)) > 1, (
            f"degenerate single-token output after wake_up: {token_ids[:16]}"
        )
        # NaN logits surface as NaN log-probs.
        for step_logprobs in completion.logprobs or []:
            for logprob in step_logprobs.values():
                assert logprob.logprob == logprob.logprob, (
                    "NaN log-prob after wake_up (stale GDN state)"
                )
