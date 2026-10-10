# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end parity tests for auxiliary logprob replay."""

import math

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="auxiliary output replay requires a CUDA-like worker",
    ),
    pytest.mark.skip_global_cleanup,
]

MODEL = "facebook/opt-125m"
TOP_K = 5
MAX_TOKENS = 8
CHUNK_SIZE = 32

# The repeated prefix is deliberately longer than one scheduler chunk and
# several KV blocks. The second request therefore exercises both chunked
# prefill and a non-empty prefix-cache replay.
PREFIX = (
    "The quick brown fox jumps over the lazy dog. "
    "A prefix-cache replay must preserve every scored token. "
) * 12
PROMPT = PREFIX + "Finish this short sentence: auxiliary logprobs are"


def _request_tuple(request_output):
    sample = request_output.outputs[0]
    return (
        list(sample.token_ids),
        sample.text,
        sample.logprobs,
        request_output.prompt_logprobs,
    )


def _assert_logprob_values_match(reference, replay) -> None:
    assert reference is not None and replay is not None
    assert len(reference) == len(replay)
    for ref_row, replay_row in zip(reference, replay, strict=True):
        if ref_row is None:
            assert replay_row is None
            continue
        assert replay_row is not None
        assert ref_row.keys() == replay_row.keys()
        for token_id in ref_row:
            ref_value = ref_row[token_id]
            replay_value = replay_row[token_id]
            assert ref_value.rank == replay_value.rank
            assert math.isclose(
                ref_value.logprob,
                replay_value.logprob,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )


def _assert_top_k_width(logprobs, top_k: int) -> None:
    assert logprobs is not None
    for row in logprobs:
        if row is None:
            continue
        # The internal payload has selected-token + top-k columns. The public
        # dict merges the selected token when it is already among the top-k.
        assert len(row) in (top_k, top_k + 1)


def _make_llm(
    *, enable_replay: bool = True, chunk_size: int = CHUNK_SIZE, **overrides
):
    config = dict(
        model=MODEL,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=chunk_size,
        enable_prefix_caching=True,
    )
    if enable_replay:
        config["aux_output_config"] = {
            "enable_logprobs_replay": True,
            "enable_prompt_logprobs_replay": True,
        }
    config.update(overrides)
    return LLM(**config)


@pytest.mark.parametrize(
    ("logprobs", "prompt_logprobs"),
    [(TOP_K, None), (TOP_K, TOP_K)],
    ids=["generated", "generated-and-prompt"],
)
@pytest.mark.parametrize("chunk_size", [16, 32, 48])
def test_aux_output_logprobs_replay_matches_cold_reference(
    logprobs: int, prompt_logprobs: int | None, chunk_size: int
):
    reference_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
    )
    replay_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        extra_args={"aux_output_replay": True},
    )

    # Keep the reference path on the normal logprob data plane. This proves
    # replay is equivalent to the existing non-RL behavior.
    reference_llm = _make_llm(
        enable_replay=False,
        chunk_size=chunk_size,
        enable_prefix_caching=False,
    )
    reference = reference_llm.generate([PROMPT], reference_params)[0]
    del reference_llm

    replay_llm = _make_llm(chunk_size=chunk_size)
    replay_llm.generate([PREFIX], replay_params)  # Prime complete KV/logprob blocks.
    replay = replay_llm.generate([PROMPT], replay_params)[0]

    assert replay.num_cached_tokens > 0, (
        "expected a prefix-cache hit; the replay test would be vacuous"
    )
    reference_tuple = _request_tuple(reference)
    replay_tuple = _request_tuple(replay)
    check_logprobs_close(
        outputs_0_lst=[reference_tuple],
        outputs_1_lst=[replay_tuple],
        name_0="cold_reference",
        name_1="aux_output_replay",
        always_check_logprobs=True,
    )
    _assert_logprob_values_match(reference_tuple[2], replay_tuple[2])
    _assert_top_k_width(replay_tuple[2], logprobs)
    if prompt_logprobs is not None:
        _assert_logprob_values_match(reference_tuple[3], replay_tuple[3])
        _assert_top_k_width(replay_tuple[3], prompt_logprobs)
    else:
        assert replay_tuple[3] is None
    del replay_llm
