# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ucx fault injection test for disaggregated prefill/decode

tests that vLLM can handle ucx failures gracefully when faults are injected
on the decode instances during kv transfer operations
"""

import os

import openai

BASE_URL = "http://localhost:8192/v1"

SIMPLE_PROMPT = (
    "The best part about working on vLLM is that I got to meet so many people across "
    "various different organizations like UCB, Google, and Meta which means",
)

# get model name from environment variable
MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")


def run_simple_prompt():
    """run a simple prompt to verify the system is working"""
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    completion = client.completions.create(
        model=MODEL_NAME, prompt=SIMPLE_PROMPT, max_tokens=100
    )

    print("-" * 50)
    print(f"Completion results for {MODEL_NAME}:")
    print(completion)
    print("-" * 50)

    return completion


def test_basic_completion_with_fault_injection():
    """
    basic test to verify that completions work with fault injection enabled

    the test verifies that vLLM can handle these faults gracefully using the
    default retry policy (recomputation on failure)
    """
    completion = run_simple_prompt()

    # basic validation
    assert completion.choices is not None
    assert len(completion.choices) > 0
    assert completion.choices[0].text is not None
    assert len(completion.choices[0].text) > 0

    print("✅ Basic completion test passed with active fault injection!")


# TODO: add more sophisticated tests that:
# - verify specific fault scenarios and recovery behavior
# - test different fault patterns
# - validate error handling and retry logic
# - measure impact on latency and throughput
