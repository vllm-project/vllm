# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import pytest
import torch

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


def _generate_hf_greedy_logprobs(
    hf_model: Any,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int,
) -> list[tuple[list[int], str, list[dict[int, float]]]]:
    """Collect HF generation logprobs without relying on hidden states.

    NCP-OLMo's pure-HF generation path returns per-step scores but intentionally
    does not materialize generation hidden states. Consuming ``scores`` also
    compares the logits after the same generation-time processors that selected
    each greedy token.
    """
    outputs = []
    for inputs in hf_model.get_inputs(prompts):
        generated = hf_model.model.generate(
            **hf_model.wrap_device(inputs),
            tokenizer=hf_model.tokenizer,
            use_cache=True,
            do_sample=False,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

        scores = generated.scores
        output_ids = (
            generated.sequences[0, -len(scores) :].tolist() if scores else []
        )
        output_logprobs = []
        for score in scores:
            logprobs = torch.log_softmax(score[0].float(), dim=-1)
            topk = logprobs.topk(num_logprobs)
            output_logprobs.append(
                dict(
                    zip(
                        topk.indices.tolist(),
                        topk.values.tolist(),
                        strict=True,
                    )
                )
            )

        outputs.append(
            (
                output_ids,
                hf_model.tokenizer.decode(output_ids),
                output_logprobs,
            )
        )

    return outputs


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
        hf_outputs = _generate_hf_greedy_logprobs(
            hf_model,
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
        always_check_logprobs=True,
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
