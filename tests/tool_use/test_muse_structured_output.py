# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where structured outputs start in Muse output.

Muse keeps the response parser in its reasoning phase through a `to=user`
answer, so the grammar takes its start from the channel header instead of
`is_reasoning_end`.

Requires real tokenizers (`MUSE_CKPT`, `MUSE_GLIMMER_CKPT`); a test is skipped if
its checkpoint is unavailable.
"""

import os
import time

import pytest

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

CKPT = os.environ.get("MUSE_CKPT", "")
GLIMMER_CKPT = os.environ.get("MUSE_GLIMMER_CKPT", "")

PROMPT = "<|start|>user<|message|>hi<|eot|><|start|>assistant"
REASONING = " to=self<|message|>think<|eom|><|start|>assistant"
TOOL_CALL = (
    " to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n</atem:invoke>\n</atem:function_calls>'
)
NUM_SPEC_TOKENS = 3


def _tokenizer(ckpt: str):
    if not os.path.isdir(ckpt):
        pytest.skip(f"checkpoint not found at {ckpt!r}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)


@pytest.fixture(scope="module")
def tok():
    return _tokenizer(CKPT)


@pytest.fixture
def parser(tok):
    return ReasoningParserManager.get_reasoning_parser("muse")(tok)


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        pytest.param("", False, id="generation_prompt"),
        pytest.param(" to=user<|message|>", True, id="user_header"),
        pytest.param(' to=user<|message|>{"a', True, id="user_body"),
        pytest.param(" to=commentary<|message|>", True, id="commentary_header"),
        pytest.param(" to=self<|message|>think", False, id="reasoning"),
        pytest.param("<|message|>think", False, id="no_recipient"),
        pytest.param(TOOL_CALL, False, id="tool_channel"),
        pytest.param(
            " to=user<|message|>ok<|eot|><|start|>user<|message|>more<|eot|>"
            "<|start|>assistant",
            False,
            id="earlier_answer",
        ),
    ],
)
def test_structured_output_start_in_prompt(tok, parser, suffix, expected):
    assert parser.is_structured_output_start(_ids(tok, PROMPT + suffix)) is expected


@pytest.mark.parametrize("chunk", [1, 2, NUM_SPEC_TOKENS + 1])
@pytest.mark.parametrize(
    ("generated", "content_header"),
    [
        pytest.param(
            REASONING + ' to=user<|message|>{"a": 1}<|eot|>',
            " to=user<|message|>",
            id="reasoning_then_answer",
        ),
        pytest.param(' to=user<|message|>{"a": 1}<|eot|>', None, id="answer_only"),
        pytest.param(
            REASONING
            + " to=commentary<|message|>note<|eom|><|start|>assistant"
            + TOOL_CALL,
            " to=commentary<|message|>",
            id="commentary_then_tool",
        ),
        pytest.param(REASONING + TOOL_CALL + "<|eot|>", "", id="tool_call"),
        pytest.param(" to=self<|message|>think<|eom|>", "", id="reasoning_only"),
        pytest.param("<|message|>think<|eom|>", "", id="no_recipient"),
    ],
)
def test_structured_output_start_streaming(
    tok, parser, generated, content_header, chunk
):
    """The hook fires on the step that completes the first content header.

    `content_header` is the text the header starts with, None for a header at
    the start of the output and "" when the output has no content channel.
    """
    prompt_ids = _ids(tok, PROMPT)
    gen_ids = _ids(tok, generated)
    if content_header == "":
        expected = None
    else:
        before = generated.split(content_header)[0] if content_header else ""
        header = content_header or " to=user<|message|>"
        expected = len(_ids(tok, before + header)) - 1

    fired = None
    for i in range(0, len(gen_ids), chunk):
        delta_ids = gen_ids[i : i + chunk]
        input_ids = prompt_ids + gen_ids[: i + chunk]
        if parser.is_structured_output_start_streaming(input_ids, delta_ids):
            # The response parser stays in its reasoning phase.
            assert not parser.is_reasoning_end(input_ids)
            fired = i
            break

    if expected is None:
        assert fired is None
    else:
        assert fired == expected - expected % chunk


@pytest.mark.parametrize(
    ("generated", "fires"),
    [
        pytest.param(
            REASONING + ' to=user<|message|>{"a": 1}<|eot|>', True, id="answer"
        ),
        pytest.param(
            REASONING + " to=commentary<|message|>note<|eom|>", False, id="commentary"
        ),
        pytest.param(REASONING + TOOL_CALL + "<|eot|>", False, id="tool_call"),
    ],
)
def test_glimmer_structured_output_start(generated, fires):
    """The `muse_glimmer` profile constrains the `to=user` answer only."""
    tok = _tokenizer(GLIMMER_CKPT)
    parser = ReasoningParserManager.get_reasoning_parser("muse_glimmer")(tok)
    prompt_ids = _ids(tok, PROMPT)
    gen_ids = _ids(tok, generated)

    fired = [
        i
        for i in range(len(gen_ids))
        if parser.is_structured_output_start_streaming(
            prompt_ids + gen_ids[: i + 1], gen_ids[i : i + 1]
        )
    ]
    header_end = len(_ids(tok, REASONING + " to=user<|message|>")) - 1
    assert fired == ([header_end] if fires else [])
    in_answer = prompt_ids + gen_ids[: header_end + 2]
    assert parser.is_structured_output_start(in_answer) is fires


def _row_kinds(bitmask) -> str:
    """U: unconstrained row; C: row constrained by the grammar."""
    return "".join("U" if (row == -1).all() else "C" for row in bitmask)


def _allows(row, token_id: int) -> bool:
    return bool((int(row[token_id >> 5]) >> (token_id & 31)) & 1)


@pytest.mark.parametrize("chunk", [1, 2, NUM_SPEC_TOKENS + 1])
@pytest.mark.parametrize("answer_only", [False, True])
def test_grammar_starts_after_content_header(tok, chunk, answer_only):
    """Reasoning and the header are unconstrained; the answer is JSON."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=CKPT, tokenizer=CKPT),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar", reasoning_parser="muse"
        ),
        speculative_config=SpeculativeConfig(
            model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS
        ),
    )
    manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json_object=True)
    )
    sampling_params.structured_outputs._backend = "xgrammar"  # type: ignore[union-attr]
    eot = tok.convert_tokens_to_ids("<|eot|>")
    sampling_params.update_from_generation_config({}, eot)
    request = Request(
        request_id="muse-json",
        prompt_token_ids=_ids(tok, PROMPT),
        sampling_params=sampling_params,
        pooling_params=None,
        reasoning_ended=False,
    )
    manager.grammar_init(request)
    structured_req = request.structured_output_request
    assert structured_req is not None
    deadline = time.time() + 30
    while not structured_req._check_grammar_completion():
        assert time.time() < deadline, "Grammar compilation timed out"
        time.sleep(0.01)
    grammar = structured_req.grammar
    assert grammar is not None

    preamble = ("" if answer_only else REASONING) + " to=user<|message|>"
    answer = '{"a": 1}'
    gen_ids = _ids(tok, preamble + answer) + [eot]
    num_free = len(_ids(tok, preamble))
    prose = _ids(tok, "Hello")[0]

    kinds = ""
    for i in range(0, len(gen_ids), chunk):
        step = gen_ids[i : i + chunk]
        # All but the last token of a step are drafts, the last is sampled
        # from the row after them.
        bitmask = manager.grammar_bitmask(
            requests={request.request_id: request},
            structured_output_request_ids=[request.request_id],
            scheduled_spec_decode_tokens={request.request_id: step[:-1]},
        )
        assert bitmask is not None
        for j, token_id in enumerate(step):
            assert _allows(bitmask[j], token_id), tok.decode([token_id])
            if i + j == num_free:
                assert not _allows(bitmask[j], prose)
        kinds += _row_kinds(bitmask[: len(step)])
        request.append_output_token_ids(step)
        assert manager.accept_tokens(request, step)

    assert kinds == "U" * num_free + "C" * (len(gen_ids) - num_free)
    assert structured_req.reasoning_ended is True
    assert not isinstance(grammar, Exception)
    assert grammar.is_terminated()
