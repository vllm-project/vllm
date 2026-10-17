# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end parity tests for auxiliary logprobs."""

import math

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="auxiliary output requires a CUDA-like worker",
    ),
    pytest.mark.skip_global_cleanup,
]

MODEL = "facebook/opt-125m"
TOP_K = 5
MAX_TOKENS = 8
CHUNK_SIZE = 32

# The repeated prefix is deliberately longer than one scheduler chunk and
# several KV blocks. The second request therefore exercises both chunked
# prefill and a non-empty prefix-cache reuse.
PREFIX = (
    "The quick brown fox jumps over the lazy dog. "
    "A prefix-cache reuse must preserve every scored token. "
) * 12
PROMPT = PREFIX + "Finish this short sentence: auxiliary logprobs are"


def _request_tuple(request_output):
    sample = request_output.outputs[0]
    return (
        list(sample.token_ids),
        sample.text,
        sample.logprobs,
    )


def _assert_logprob_values_match(reference, aux) -> None:
    assert reference is not None and aux is not None
    assert len(reference) == len(aux)
    for ref_row, aux_row in zip(reference, aux, strict=True):
        if ref_row is None:
            assert aux_row is None
            continue
        assert aux_row is not None
        assert ref_row.keys() == aux_row.keys()
        for token_id in ref_row:
            ref_value = ref_row[token_id]
            aux_value = aux_row[token_id]
            assert ref_value.rank == aux_value.rank
            assert math.isclose(
                ref_value.logprob,
                aux_value.logprob,
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
    *, enable_aux: bool = True, chunk_size: int = CHUNK_SIZE, **overrides
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
    if enable_aux:
        config["aux_output_config"] = {
            "enable_return_logprobs": True,
        }
    config.update(overrides)
    return LLM(**config)


@pytest.mark.parametrize(
    "logprobs",
    [TOP_K],
    ids=["generated"],
)
@pytest.mark.parametrize("chunk_size", [16, 32, 48])
def test_aux_output_logprobs_matches_cold_reference(
    logprobs: int, chunk_size: int
):
    reference_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=logprobs,
    )
    aux_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        logprobs=logprobs,
        extra_args={"aux_output_enabled": True},
    )

    # Keep the reference path on the normal logprob data plane. This proves
    # auxiliary output is equivalent to the existing non-RL behavior.
    reference_llm = _make_llm(
        enable_aux=False,
        chunk_size=chunk_size,
        enable_prefix_caching=False,
    )
    reference = reference_llm.generate([PROMPT], reference_params)[0]
    del reference_llm

    aux_llm = _make_llm(chunk_size=chunk_size)
    aux_llm.generate([PREFIX], aux_params)  # Prime complete KV/logprob blocks.
    aux = aux_llm.generate([PROMPT], aux_params)[0]

    assert aux.num_cached_tokens > 0, (
        "expected a prefix-cache hit; the auxiliary test would be vacuous"
    )
    reference_tuple = _request_tuple(reference)
    aux_tuple = _request_tuple(aux)
    check_logprobs_close(
        outputs_0_lst=[reference_tuple],
        outputs_1_lst=[aux_tuple],
        name_0="cold_reference",
        name_1="aux_output_enabled",
        always_check_logprobs=True,
    )
    _assert_logprob_values_match(reference_tuple[2], aux_tuple[2])
    _assert_top_k_width(aux_tuple[2], logprobs)
    del aux_llm


def test_aux_output_logprobs_boundary_token_mismatch():
    """A changed token after a cached block must not cause an artifact miss."""
    aux_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=TOP_K,
        extra_args={"aux_output_enabled": True},
    )

    aux_llm = _make_llm(chunk_size=32)
    block_size = aux_llm.llm_engine.vllm_config.cache_config.block_size
    tokenizer = aux_llm.get_tokenizer()
    prefix = list(range(100, 100 + 2 * block_size))
    warm_prompt = TokensPrompt(prompt_token_ids=prefix)

    # The generated token becomes the boundary token in the warm artifact key.
    warm = aux_llm.generate([warm_prompt], aux_params)[0]
    generated_token = warm.outputs[0].token_ids[0]
    continuation = (generated_token + 1) % tokenizer.vocab_size
    aux_prompt = TokensPrompt(prompt_token_ids=prefix + [continuation])
    aux = aux_llm.generate([aux_prompt], aux_params)[0]

    assert 0 < aux.num_cached_tokens <= len(prefix)
    aux_tuple = _request_tuple(aux)
    del aux_llm

    reference_llm = _make_llm(
        enable_aux=False,
        enable_prefix_caching=False,
        chunk_size=32,
    )
    reference = reference_llm.generate(
        [aux_prompt],
        SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=TOP_K,
        ),
    )[0]

    reference_tuple = _request_tuple(reference)
    check_logprobs_close(
        outputs_0_lst=[reference_tuple],
        outputs_1_lst=[aux_tuple],
        name_0="cold_reference",
        name_1="aux_output_enabled_boundary_mismatch",
        always_check_logprobs=True,
    )
    del reference_llm


