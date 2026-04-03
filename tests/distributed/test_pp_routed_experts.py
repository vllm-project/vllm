# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that routed expert capture works correctly with pipeline parallelism.

This test creates two LLM instances — one with PP=1 (baseline) and one with
PP=2 — and asserts that the routed experts returned are identical.
"""

import numpy as np

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams

MODEL_NAME = "TitanML/tiny-mixtral"

# tiny-mixtral config: 8 local experts, top-2 routing, 2 hidden layers
NUM_LOCAL_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2
NUM_HIDDEN_LAYERS = 2

PROMPT = "The quick brown fox jumps over the lazy dog"
MAX_TOKENS = 10


def _generate_with_routed_experts(pipeline_parallel_size: int):
    """Spin up an LLM with the given PP size and return routed_experts."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=256,
        max_num_seqs=32,
        enforce_eager=True,
        enable_return_routed_experts=True,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="mp",
    )
    outputs = llm.generate(PROMPT, SamplingParams(max_tokens=MAX_TOKENS, temperature=0))
    del llm

    assert len(outputs) == 1
    completion = outputs[0].outputs[0]
    assert completion.routed_experts is not None
    return completion.routed_experts


@multi_gpu_test(num_gpus=2)
def test_pp2_routed_experts_match_pp1():
    """PP=2 must return the same routing decisions as PP=1."""
    experts_pp1 = _generate_with_routed_experts(pipeline_parallel_size=1)
    experts_pp2 = _generate_with_routed_experts(pipeline_parallel_size=2)

    np.testing.assert_array_equal(
        experts_pp1,
        experts_pp2,
        err_msg=(
            "Routed experts differ between PP=1 and PP=2. "
            f"PP=1:\n{experts_pp1}\nPP=2:\n{experts_pp2}"
        ),
    )
