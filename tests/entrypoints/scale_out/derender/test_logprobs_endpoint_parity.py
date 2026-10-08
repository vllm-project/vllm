# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""generate + derender must return the same `top_logprobs` as the coupled
`/v1/chat/completions` and `/v1/completions` endpoints (#59513).

Both sides start from the same per-position engine dicts, built with the
engine's own `append_logprobs_for_next_position` and detokenized the way the
engine does, and go through the real coupled serving objects on one side and
`/inference/v1/generate` + derender on the other. The cases include a sampled
token outside the top k, where chat keeps k entries and completions k + 1; the
greedy end-to-end parity tests never produce that case.
"""

import pytest

from tests.entrypoints.openai.chat_completion.test_serving_chat import (
    _build_mock_engine,
    _build_serving_chat,
)
from tests.entrypoints.openai.completion.test_completion_error import (
    _build_serving_completion,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateLogProbs
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import append_logprobs_for_next_position
from vllm.renderers.online_derenderer import (
    _chat_top_logprobs_limit,
    _completion_top_logprobs_limit,
    _convert_chat_logprobs_to_completion_logprobs,
    _resolve_logprobs,
)
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import convert_ids_list_to_tokens

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


@pytest.fixture(scope="module")
def serving_chat():
    return _build_serving_chat(_build_mock_engine())


@pytest.fixture(scope="module")
def serving_completion():
    return _build_serving_completion(_build_mock_engine())


@pytest.fixture(scope="module")
def vocab_ids(tokenizer):
    """Ordinary word pieces (no byte-fallback tokens, which need the U+FFFD
    context repair that is compared elsewhere)."""
    ids = tokenizer.encode(
        "the quick brown fox jumps over a lazy dog while seven cats watch "
        "from an old wooden fence near the river bank",
        add_special_tokens=False,
    )
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return list(dict.fromkeys(i for i, p in zip(ids, pieces) if "<0x" not in p))


def _positions(tokenizer, vocab_ids, k, sampled_ranks):
    """Engine output for one completion: per position the sampled id and its
    dict (sampled first, then ranks 1..k), with decoded tokens as the engine
    fills them. k = -1 returns every candidate, as for the whole vocabulary."""
    token_ids, dicts = [], []
    for pos, sampled_rank in enumerate(sampled_ranks):
        n_top = max(k, sampled_rank) if k != -1 else len(vocab_ids) - 1
        ranked = [vocab_ids[(pos + j) % len(vocab_ids)] for j in range(n_top + 1)]
        sampled = ranked[sampled_rank - 1]
        top = ranked[: len(ranked) - 1] if k == -1 else ranked[:k]
        ids = [sampled, *top]
        logprobs = [-0.25 * sampled_rank] + [-0.25 * (r + 1) for r in range(len(top))]
        per_position: list = []
        append_logprobs_for_next_position(
            per_position,
            ids,
            logprobs,
            convert_ids_list_to_tokens(tokenizer, ids),
            sampled_rank,
            len(top),
        )
        token_ids.append(sampled)
        dicts.append(per_position[0])
    return token_ids, dicts


def _generate(token_ids, dicts) -> GenerateLogProbs:
    out = ServingTokens._create_tokens_logprobs(
        None, token_ids=token_ids, top_logprobs=dicts
    )
    # Over the wire, as a derender client would send it back.
    return GenerateLogProbs.model_validate_json(out.model_dump_json())


# Sampled token rank per position: top-1, inside the top k, outside it.
RANKS = [1, 2, 1, 7, 3, 9, 1]


@pytest.mark.parametrize("k", [0, 1, 2, 5, -1])
def test_chat_derender_matches_coupled_chat(tokenizer, vocab_ids, serving_chat, k):
    token_ids, dicts = _positions(tokenizer, vocab_ids, k, RANKS)
    coupled = serving_chat._create_chat_logprobs(
        token_ids=token_ids,
        top_logprobs=dicts,
        tokenizer=tokenizer,
        num_output_top_logprobs=k,
    )
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        logprobs=True,
        top_logprobs=k,
    )
    derendered = _resolve_logprobs(
        _generate(token_ids, dicts),
        tokenizer,
        top_limit=_chat_top_logprobs_limit(request),
    )
    assert derendered.model_dump() == coupled.model_dump()


@pytest.mark.parametrize("k", [0, 1, 2, 5, -1])
def test_completion_derender_matches_coupled_completion(
    tokenizer, vocab_ids, serving_completion, k
):
    token_ids, dicts = _positions(tokenizer, vocab_ids, k, RANKS)
    coupled = serving_completion._create_completion_logprobs(
        token_ids=token_ids,
        top_logprobs=dicts,
        num_output_top_logprobs=k,
        tokenizer=tokenizer,
    )
    request = CompletionRequest(model=MODEL_NAME, prompt="hi", logprobs=k)
    generated = _generate(token_ids, dicts)
    derendered = _convert_chat_logprobs_to_completion_logprobs(
        _resolve_logprobs(
            generated,
            tokenizer,
            top_limit=_completion_top_logprobs_limit(request),
        ),
        generated,
    )
    assert derendered.model_dump() == coupled.model_dump()
    if k > 0:
        # The case #59513 is about: completions keeps k + 1 entries when the
        # sampled token is outside the top k.
        outside = [i for i, r in enumerate(RANKS) if r > k]
        assert all(len(derendered.top_logprobs[i]) == k + 1 for i in outside)
    if k == -1:
        # /v1/completions keeps entry i when -1 >= i, so none at all.
        assert all(top == {} for top in derendered.top_logprobs)


def test_without_the_request_derender_keeps_every_candidate(tokenizer, vocab_ids):
    token_ids, dicts = _positions(tokenizer, vocab_ids, 2, RANKS)
    derendered = _resolve_logprobs(
        _generate(token_ids, dicts),
        tokenizer,
        top_limit=_chat_top_logprobs_limit(None),
    )
    assert [len(e.top_logprobs) for e in derendered.content] == [len(d) for d in dicts]


@pytest.mark.parametrize("k", [-1, 0, 2])
def test_completion_position_without_logprobs_stays_none(
    tokenizer, vocab_ids, serving_completion, k
):
    """A position the engine returned no logprobs for is `None` in
    `/v1/completions`, while a position whose candidates were all cut
    (`logprobs=-1`) is `{}`; derender keeps the two apart."""
    token_ids, dicts = _positions(tokenizer, vocab_ids, k, RANKS)
    dicts[3] = None
    coupled = serving_completion._create_completion_logprobs(
        token_ids=token_ids,
        top_logprobs=dicts,
        num_output_top_logprobs=k,
        tokenizer=tokenizer,
    )
    request = CompletionRequest(model=MODEL_NAME, prompt="hi", logprobs=k)
    generated = _generate(token_ids, dicts)
    derendered = _convert_chat_logprobs_to_completion_logprobs(
        _resolve_logprobs(
            generated,
            tokenizer,
            top_limit=_completion_top_logprobs_limit(request),
        ),
        generated,
    )
    assert derendered.top_logprobs == coupled.top_logprobs
    assert derendered.top_logprobs[3] is None
