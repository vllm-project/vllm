# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with eagle-harness installed; selected by the EAGLE-4 GPU CI step."""

import pytest

from tests.models.utils import check_logprobs_close
from tests.v1.e2e.spec_decode.utils import compute_acceptance_len
from vllm import SamplingParams

TARGET = "Qwen/Qwen3.8-27B"
DRAFT = "EAGLE-Corp/Qwen3.8-27B-EAGLE4"
PROMPTS = [
    "Write a Python function that returns the Fibonacci numbers below 100.",
    "Explain why a hash table has amortized constant-time lookup.",
    "Solve step by step: a train travels 180 km in 3 hours. What is its speed?",
    "List the first ten prime numbers and explain what makes them prime.",
]


@pytest.mark.parametrize("top_k", [1, 4], ids=["chain", "tree"])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graphs"])
def test_eagle4_verification_and_request_slot_reuse(
    vllm_runner, monkeypatch, top_k, enforce_eager
):
    """Check target agreement, own-row logprobs and real multi-token acceptance."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    settings = dict(
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
    )
    params = [
        SamplingParams(temperature=t, seed=17, max_tokens=64, logprobs=5)
        for t in (0.0, 1.0)
    ]
    with vllm_runner(TARGET, **settings) as target:
        references = [target.generate_w_logprobs(PROMPTS, p) for p in params]

    with vllm_runner(
        TARGET,
        **settings,
        attention_config={"backend": "CUSTOM"} if top_k > 1 else {},
        speculative_config={
            "method": "eagle_harness",
            "model": DRAFT,
            "method_config": {
                "mode": "static",
                "node_top_k": top_k,
                "tree_depth": 5,
                "max_verify_width": 6 if top_k == 1 else 8,
            },
        },
    ) as draft:
        for p, reference in zip(params, references):
            outputs = draft.generate_w_logprobs(PROMPTS, p)
            check_logprobs_close(
                outputs_0_lst=reference,
                outputs_1_lst=outputs,
                name_0="target",
                name_1="eagle4",
            )
            for token_ids, _, logprobs in outputs:
                assert token_ids
                assert len(token_ids) == len(logprobs)
                for token_id, row in zip(token_ids, logprobs):
                    assert token_id in row
                    assert row[token_id].logprob <= 0
                    if p.temperature == 0:
                        assert row[token_id].rank == 1
        # Reusing a worker's request slots must not reuse an earlier tree's KV.
        repeated = draft.generate_w_logprobs(PROMPTS, params[0])
        check_logprobs_close(
            outputs_0_lst=references[0],
            outputs_1_lst=repeated,
            name_0="target",
            name_1="reused eagle4 slots",
        )
        assert compute_acceptance_len(draft.llm.get_metrics()) > 1
