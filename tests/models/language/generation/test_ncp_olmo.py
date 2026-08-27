# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from ...utils import check_logprobs_close

MODEL = os.environ.get("NCP_OLMO_TEST_MODEL", "")
PROMPTS = [
    "A short request checks the model state.",
    "A longer request checks that batching preserves the request-local HLM state. " * 8,
    "Name the capital of France.",
]
MAX_NUM_SEQS = len(PROMPTS)

pytestmark = pytest.mark.skipif(
    not MODEL,
    reason="NCP_OLMO_TEST_MODEL must point to a pure-HF NCP-OLMo checkpoint",
)


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_greedy_logprobs(
    hf_runner,
    vllm_runner,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(
        MODEL,
        dtype="bfloat16",
        model_kwargs={"attn_implementation": "eager"},
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_batched_matches_sequential(
    vllm_runner,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        sequential_outputs = [
            vllm_model.generate_greedy_logprobs(
                [prompt],
                max_tokens,
                num_logprobs,
            )[0]
            for prompt in PROMPTS
        ]
        batched_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    check_logprobs_close(
        outputs_0_lst=sequential_outputs,
        outputs_1_lst=batched_outputs,
        name_0="sequential_vllm",
        name_1="batched_vllm",
    )
