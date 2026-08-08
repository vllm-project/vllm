# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

TOKENIZER = "gpt2"
THINK_END = "\n"  # reasoning-end marker (single GPT-2 token)
EOS = "<|eos|>"  # resolved to tokenizer.eos_token_id
JSON_SCHEMA = '{"type": "object"}'
BACKENDS = ("xgrammar", "guidance")
MAX_WAIT_SECONDS = 5
NUM_SPEC_TOKENS = 8


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@dataclass(frozen=True)
class FlowCase:
    """One validate -> bitmask -> accept -> post-accept bitmask scenario."""

    raw_drafts: tuple[str, ...]
    expected_row_pattern: str
    expected_reasoning: bool | None
    expect_terminated: bool
    prefix: str = ""
    expected_validated: tuple[str, ...] | None = None
    reasoning_ended: bool | None = False
    xfail_guidance: str | None = None


class MockReasoner:
    def __init__(self, tokenizer, marker: int | None = None):
        self.marker = marker

    def is_reasoning_end(self, input_ids):
        if self.marker is None:
            return True
        return self.marker in list(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self.is_reasoning_end(delta_ids)


def _single_token(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text)
    assert len(token_ids) == 1, (text, token_ids)
    return token_ids[0]


def _to_token_ids(tokenizer, texts: tuple[str, ...]) -> list[int]:
    """Convert literal text (or EOS sentinel) to token IDs."""
    return [
        tokenizer.eos_token_id if t == EOS else _single_token(tokenizer, t)
        for t in texts
    ]


def _wait_for_grammar(request: Request) -> None:
    structured_req = request.structured_output_request
    assert structured_req is not None

    deadline = time.time() + MAX_WAIT_SECONDS
    while not structured_req._check_grammar_completion():
        if time.time() > deadline:
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)


def _build_harness(
    tokenizer,
    backend: str,
    prefix: str = "",
    use_reasoner: bool = True,
    reasoning_ended: bool | None = None,
    enable_in_reasoning: bool = False,
    reasoning_parser_kwargs: dict | None = None,
) -> tuple[StructuredOutputManager, Request]:
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(
            backend=backend,
            enable_in_reasoning=enable_in_reasoning,
        ),
        speculative_config=SpeculativeConfig(
            model="[ngram]",
            num_speculative_tokens=NUM_SPEC_TOKENS,
        ),
    )
    manager = StructuredOutputManager(vllm_config)
    if use_reasoner:
        manager.reasoner_cls = MockReasoner

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
    )
    sampling_params.structured_outputs._backend = backend
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    prompt_ids = tokenizer.encode(prefix) if prefix else []
    request = Request(
        request_id=f"{backend}-flow",
        prompt_token_ids=prompt_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        reasoning_ended=reasoning_ended,
        reasoning_parser_kwargs=reasoning_parser_kwargs,
    )
    manager.grammar_init(request)
    _wait_for_grammar(request)

    structured_req = request.structured_output_request
    assert request.prompt_token_ids is not None
    assert structured_req is not None
    assert structured_req.grammar is not None
    if prompt_ids:
        assert structured_req.grammar.accept_tokens(
            request.request_id,
            prompt_ids,
        )
    return manager, request


def _row_pattern(bitmask) -> str:
    return "".join("U" if (row == -1).all() else "C" for row in bitmask)


def _run_real_flow(
    manager: StructuredOutputManager,
    request: Request,
    raw_drafts: list[int],
    expected_validated: list[int],
    expected_row_pattern: str,
    expected_reasoning: bool | None,
    expect_terminated: bool,
) -> None:
    assert len(expected_validated) <= len(raw_drafts)

    validated = manager.validate_tokens(request, list(raw_drafts))
    assert validated == expected_validated

    padded = validated + [-1] * (len(raw_drafts) - len(validated))
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(raw_drafts) + 1
    assert _row_pattern(bitmask) == expected_row_pattern

    # grammar_bitmask() must rollback any speculative state it advanced.
    assert manager.validate_tokens(request, list(raw_drafts)) == expected_validated

    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert grammar is not None
    assert not grammar.is_terminated()
    # `bitmask[i]` is the grammar state after the first `i` scheduled tokens.
    # These tests commit the validated prefix, so the matching post-accept
    # state is already present in the scheduled bitmask.
    expected_post_accept_row = bitmask[len(expected_validated)].copy()

    # Mirror the scheduler flow: sampled tokens are appended before
    # StructuredOutputManager.accept_tokens() trims them to the
    # grammar-constrained suffix.
    request.append_output_token_ids(list(expected_validated))
    assert manager.accept_tokens(request, list(expected_validated))

    post_accept_bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={},
    )
    assert post_accept_bitmask is not None
    assert post_accept_bitmask.shape[0] == 1
    assert (post_accept_bitmask[0] == expected_post_accept_row).all()
    assert structured_req.reasoning_ended == expected_reasoning
    assert grammar.is_terminated() is expect_terminated


FLOW_CASES = [
    pytest.param(
        FlowCase(
            raw_drafts=(" ", "z", " "),
            expected_row_pattern="UUUU",
            expected_reasoning=False,
            expect_terminated=False,
        ),
        id="inactive",
    ),
    pytest.param(
        FlowCase(
            prefix='{"a"',
            raw_drafts=(":", ' "', "b", '"}'),
            expected_row_pattern="CCCCC",
            expected_reasoning=True,
            expect_terminated=False,
            reasoning_ended=True,
        ),
        id="active_all_valid",
    ),
    pytest.param(
        FlowCase(
            prefix='{"a"',
            raw_drafts=("z", ":", ' "', "b"),
            expected_validated=(),
            expected_row_pattern="CCCCC",
            expected_reasoning=True,
            expect_terminated=False,
            reasoning_ended=True,
        ),
        id="active_first_token_invalid",
    ),
    pytest.param(
        FlowCase(
            prefix='{"a"',
            raw_drafts=(":", "z", ' "', "b"),
            expected_validated=(":",),
            expected_row_pattern="CCCCC",
            expected_reasoning=True,
            expect_terminated=False,
            reasoning_ended=True,
        ),
        id="active_later_token_invalid",
    ),
    pytest.param(
        FlowCase(
            prefix='{"a": "b"}',
            raw_drafts=(EOS, " ", " "),
            expected_validated=(EOS,),
            expected_row_pattern="CUUU",
            expected_reasoning=True,
            expect_terminated=True,
            reasoning_ended=True,
            xfail_guidance=(
                "guidance validate_tokens rejects EOS on an already-complete "
                "object even though accept_tokens would accept it"
            ),
        ),
        id="active_first_token_terminates",
    ),
    pytest.param(
        FlowCase(
            prefix='{"a"',
            raw_drafts=(":", ' "', "b", '"}', EOS, " "),
            expected_validated=(":", ' "', "b", '"}', EOS),
            expected_row_pattern="CCCCCUU",
            expected_reasoning=True,
            expect_terminated=True,
            reasoning_ended=True,
        ),
        id="active_later_token_terminates",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(THINK_END, "{", ' "', "b"),
            expected_row_pattern="UCCCC",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_on_first_token",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", THINK_END, "{", ' "'),
            expected_row_pattern="UUCCC",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_on_middle_token",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", " ", THINK_END),
            expected_row_pattern="UUUC",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_on_last_token",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", THINK_END, "z", "{"),
            expected_validated=(" ", THINK_END),
            expected_row_pattern="UUCCC",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_first_token_invalid",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", THINK_END, "{", "z"),
            expected_validated=(" ", THINK_END, "{"),
            expected_row_pattern="UUCCC",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_later_token_invalid",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", THINK_END, "{", "}", EOS, " "),
            expected_validated=(" ", THINK_END, "{", "}", EOS),
            expected_row_pattern="UUCCCUU",
            expected_reasoning=True,
            expect_terminated=True,
        ),
        id="becomes_active_terminates",
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", FLOW_CASES)
def test_real_flow(
    tokenizer,
    backend: str,
    case: FlowCase,
):
    if backend == "guidance" and case.xfail_guidance:
        pytest.xfail(case.xfail_guidance)

    reasoner_kwargs = {"marker": _single_token(tokenizer, THINK_END)}
    manager, request = _build_harness(
        tokenizer,
        backend,
        prefix=case.prefix,
        reasoning_ended=case.reasoning_ended,
        reasoning_parser_kwargs=reasoner_kwargs,
    )

    raw_drafts = _to_token_ids(tokenizer, case.raw_drafts)
    expected_texts = (
        case.raw_drafts if case.expected_validated is None else case.expected_validated
    )
    expected_validated = _to_token_ids(tokenizer, expected_texts)

    _run_real_flow(
        manager,
        request,
        raw_drafts=raw_drafts,
        expected_validated=expected_validated,
        expected_row_pattern=case.expected_row_pattern,
        expected_reasoning=case.expected_reasoning,
        expect_terminated=case.expect_terminated,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("use_reasoner", "reasoning_ended", "enable_in_reasoning"),
    [
        pytest.param(False, False, True, id="enable_in_reasoning"),
        pytest.param(True, True, False, id="reasoning_ended"),
        pytest.param(False, False, False, id="no_reasoner"),
        pytest.param(True, None, False, id="inferred"),
    ],
)
def test_initial_constraint_activation(
    tokenizer,
    backend: str,
    use_reasoner: bool,
    reasoning_ended: bool | None,
    enable_in_reasoning: bool,
):
    manager, request = _build_harness(
        tokenizer,
        backend,
        use_reasoner=use_reasoner,
        reasoning_ended=reasoning_ended,
        enable_in_reasoning=enable_in_reasoning,
        reasoning_parser_kwargs={
            "marker": _single_token(tokenizer, THINK_END)
            if reasoning_ended is not None
            else None
        },
    )

    # "{" is valid JSON start; "z" is not. Truncation proves grammar is active.
    open_brace = _single_token(tokenizer, "{")
    z = _single_token(tokenizer, "z")
    assert manager.validate_tokens(request, [open_brace, z]) == [open_brace]
