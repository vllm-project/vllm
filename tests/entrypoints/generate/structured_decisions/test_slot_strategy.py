# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SlotLMStrategy prompts and reads, against golden prompts rendered by qev."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
)
from vllm.entrypoints.generate.structured_decisions.serving import parse_questions
from vllm.entrypoints.openai.decisions import slot_prompt
from vllm.entrypoints.openai.decisions.adapters import make_read_question
from vllm.entrypoints.openai.decisions.protocol import ScoreQuestion
from vllm.entrypoints.openai.decisions.question_types import StructuredDecisionError
from vllm.entrypoints.openai.decisions.strategies import (
    ReadContext,
    SlotLMStrategy,
    select_read_strategy,
)
from vllm.logprobs import Logprob
from vllm.tokenizers import get_tokenizer
from vllm.v1.core.sched.prefix_lm_scheduler import prefix_lm_salt

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

#: The state and questions qev's DecisionEngine probes at load.
PROBE_STATE = {
    "ticket": "My card was charged twice for the same order and support has "
    "not replied in 3 days.",
    "customer": {"tenure_months": 14, "vip": False},
}
PROBE_QUESTIONS = {
    "a": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "b": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "charges and refunds",
            "technical": "bugs",
            "sales": "pricing",
        },
    },
    "c": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "irritated", "angry"],
    },
}

_PREFIX = (
    "<|im_start|>system\nApply the question to the state. Choose exactly one of "
    "the listed options. Respond with only its uppercase letter, with no "
    "explanation or reasoning.<|im_end|>\n<|im_start|>user\n<state>\n{\n"
    '  "ticket": "My card was charged twice for the same order and support has '
    'not replied in 3 days.",\n  "customer": {\n    "tenure_months": 14,\n'
    '    "vip": false\n  }\n}\n</state>\n\n'
)
_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
#: qev's prompts for PROBE_QUESTIONS.
QEV_PROMPTS = {
    "a": _PREFIX + "Question: Is the customer asking for money back?\n\nOptions:\n"
    "A. yes: the statement is true / the answer is yes\n"
    "B. no: the statement is false / the answer is no\n\n"
    "Answer with one letter: A, B." + _TAIL,
    "b": _PREFIX + "Question: Which team should handle this?\n\nOptions:\n"
    "A. billing: charges and refunds\nB. technical: bugs\nC. sales: pricing"
    "\n\nAnswer with one letter: A, B, C." + _TAIL,
    "c": _PREFIX + "Rate the state: How frustrated is the customer?\n\nOptions:\n"
    "A. level 0: calm\nB. level 1: irritated\nC. level 2: angry\n\n"
    "Answer with one letter: A, B, C." + _TAIL,
}
#: qev's shared prefix for PROBE_STATE, ending in `</`, `state`, `>`, `ĊĊ`.
QEV_PREFIX_LEN = 99


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("baselabs/elephant-1")


def make_strategy(tokenizer, *, head_dtype=torch.float32, prefix_caching=False):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(modeling_overrides={}),
        head_dtype=head_dtype,
        max_model_len=32768,
        is_hybrid=True,
    )
    engine = MagicMock(spec=EngineClient)
    engine.model_config = model_config
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching)
    )

    async def generate(prompt, params, request_id, **kwargs):
        ids = params.logprob_token_ids or [0]
        # The first slot gets twice the weight of each other slot.
        logprobs = {t: Logprob(-2.0 + math.log(2) * (t == ids[0])) for t in ids}
        out = SimpleNamespace(token_ids=ids[:1], logprobs=[logprobs], finish_reason="")
        yield SimpleNamespace(
            outputs=[out],
            prompt_token_ids=prompt["prompt_token_ids"],
            num_cached_tokens=0,
            num_cache_creation_tokens=0,
        )

    engine.generate.side_effect = generate
    renderer = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    context = ReadContext(engine, SimpleNamespace(renderer=renderer), None, "auto", {})
    return SlotLMStrategy(context), engine


async def read(strategy, request: StructuredDecisionRequest):
    questions = parse_questions(request, strategy.limits())
    reads = await strategy.read(
        questions,
        request.instructions,
        strategy.render_state(request.state),
        request_id="r",
        chat_template_kwargs=request.chat_template_kwargs,
        lora_request=None,
        priority=0,
        cache_salt=request.cache_salt,
    )
    return questions, reads


def test_strategy_is_registered():
    model = SimpleNamespace(
        architecture="Qwen3_5ForSlotDecision", logprobs_mode="raw_logprobs"
    )
    assert select_read_strategy(model, ("generate",)) is SlotLMStrategy


@pytest.mark.asyncio
async def test_reads_match_qev_byte_for_byte(tokenizer):
    strategy, engine = make_strategy(tokenizer)
    request = StructuredDecisionRequest(
        state=PROBE_STATE, questions=PROBE_QUESTIONS, cache_salt="salt"
    )
    questions, reads = await read(strategy, request)

    for q, call in zip(questions, engine.generate.call_args_list, strict=True):
        prompt, params = call.args[0], call.args[1]
        ids = tokenizer.encode(QEV_PROMPTS[q.id], add_special_tokens=False)
        assert prompt["prompt_token_ids"] == ids
        # The prefix's digest, then the caller's salt.
        assert prompt["cache_salt"] == prefix_lm_salt(ids[:QEV_PREFIX_LEN], "salt")
        assert params.extra_args == {"prefix_lm_len": QEV_PREFIX_LEN}
        assert params.logprob_token_ids == strategy.slot_ids[: len(q.options)]

    # Slot A, the first option, has twice the weight of each other slot.
    assert reads[2].probs == pytest.approx([0.5, 0.25, 0.25])


@pytest.mark.asyncio
async def test_prefix_ends_at_the_state_block(tokenizer):
    strategy, engine = make_strategy(tokenizer, prefix_caching=True)
    # Both questions start with "Question:", so the token LCP runs past the
    # state block. A long state makes the strategy warm the prefix first.
    questions = {k: PROBE_QUESTIONS[k] for k in "ab"}
    request = StructuredDecisionRequest(state="word " * 200, questions=questions)
    await read(strategy, request)
    warm_up, *calls = engine.generate.call_args_list
    prefix = warm_up.args[0]["prompt_token_ids"]
    assert tokenizer.decode(prefix).endswith("</state>\n\n")
    salt = prefix_lm_salt(prefix, None)
    for call in (warm_up, *calls):
        assert call.args[0]["prompt_token_ids"][: len(prefix)] == prefix
        assert call.args[1].extra_args == {"prefix_lm_len": len(prefix)}
        assert call.args[0]["cache_salt"] == salt


@pytest.mark.asyncio
async def test_many_options_use_two_letter_slots(tokenizer):
    strategy, engine = make_strategy(tokenizer)
    assert strategy.limits().max_options == 255
    criteria = {f"option {i}": {"id": i} if i % 2 else None for i in range(255)}
    questions = {"many": {"type": "choice", "criteria": criteria}}
    request = StructuredDecisionRequest(state="plain text", questions=questions)
    await read(strategy, request)
    prompt, params = engine.generate.call_args.args[:2]
    text = tokenizer.decode(prompt["prompt_token_ids"])
    assert "<state>\nplain text\n</state>\n\nQuestion: Which option applies" in text
    assert '\nAA. option 26\nAB. option 27: {\n  "id": 27\n}\n' in text
    assert "Answer with one label: A, B, C," in text
    assert len(params.logprob_token_ids) == 255
    assert len(set(strategy.slot_ids)) == 256
    assert slot_prompt.SLOT_LABELS[-1] == "JU"


def test_refuses_to_start_without_fp32_head(tokenizer):
    with pytest.raises(ValueError, match="fp32 LM head"):
        make_strategy(tokenizer, head_dtype=torch.bfloat16)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"instructions": "Be strict."},
        {"chat_template_kwargs": {"enable_thinking": True}},
        # A score has at most 10 levels, as in training.
        {"questions": {"s": {"type": "score", "criteria": list("0123456789X")}}},
    ],
)
async def test_refuses_untrained_prompt_options(tokenizer, extra):
    strategy, engine = make_strategy(tokenizer)
    request = StructuredDecisionRequest(
        **{"state": PROBE_STATE, "questions": PROBE_QUESTIONS, **extra}
    )
    with pytest.raises(StructuredDecisionError):
        await read(strategy, request)
    engine.generate.assert_not_called()


def test_decisions_api_refuses_duplicate_labels(tokenizer):
    strategy, _ = make_strategy(tokenizer)
    levels = [{"label": "low"}, {"label": "high"}, {"label": "low"}]
    question = ScoreQuestion(type="score", instructions="How urgent?", levels=levels)
    with pytest.raises(StructuredDecisionError, match="distinct"):
        strategy._suffix(make_read_question(0, question))
