# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import permutations
from threading import Event

import numpy as np
import pytest
from transformers import AutoTokenizer
from xgrammar import get_model_structural_tag

from vllm.config import (
    DeviceConfig,
    SchedulerConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.structured_outputs import StructuredOutputsBackend
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.parser_engine import ReasoningEnd
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceGrammar
from vllm.v1.structured_output.backend_outlines import OutlinesBackend, OutlinesGrammar
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from vllm.v1.structured_output.request import get_structured_output_key

TOKENIZER = "gpt2"
THINK_END = "\n"  # reasoning-end marker (single GPT-2 token)
IMPLICIT_THINK_END = "z"  # accepted in strings, rejected at the object start
EOS = "<|eos|>"  # resolved to tokenizer.eos_token_id
JSON_SCHEMA = '{"type": "object"}'
BACKENDS = ("xgrammar", "guidance")
REGEX_BACKENDS = (*BACKENDS, "outlines")
DIGITS_REGEX = "[0-9]+"
MAX_WAIT_SECONDS = 5
NUM_SPEC_TOKENS = 8


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.mark.parametrize("order", list(permutations(("json", "named", "required"))))
@pytest.mark.parametrize("async_grammar", [False, True])
@pytest.mark.cpu_test
def test_auto_backend_selection_ignores_request_order(tokenizer, order, async_grammar):
    """Mixed auto requests retain their validated compiler and native masks."""
    config = VllmConfig(
        model_config=ModelConfig(model=TOKENIZER, tokenizer=TOKENIZER),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(
            backend="auto", enable_in_reasoning=True
        ),
    )
    manager = StructuredOutputManager(config)
    manager._use_async_grammar_compilation = async_grammar
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    named = {"type": "function", "function": {"name": "lookup"}}
    call = (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
    )
    try:
        for index, kind in enumerate((*order, *reversed(order), *order)):
            if kind == "json":
                outputs = StructuredOutputsParams(
                    json={"type": "integer", "multipleOf": 2}
                )
                text, expected_backend = "6", "guidance"
            else:
                tag = get_model_structural_tag(
                    "qwen_3_coder",
                    tools,
                    named if kind == "named" else "required",
                    reasoning=False,
                )
                outputs = StructuredOutputsParams(structural_tag=tag.model_dump_json())
                text, expected_backend = call, "xgrammar"
            sampling = SamplingParams(structured_outputs=outputs)
            sampling._validate_structured_outputs(
                config.model_config, config.structured_outputs_config, tokenizer
            )
            assert outputs._backend == expected_backend
            sampling.update_from_generation_config({}, tokenizer.eos_token_id)
            request = Request(
                request_id=f"mixed-{index}",
                prompt_token_ids=[],
                sampling_params=sampling,
                pooling_params=None,
                reasoning_ended=True,
            )
            manager.grammar_init(request)
            _wait_for_grammar(request)
            assert request.structured_output_request is not None
            grammar = request.structured_output_request.grammar
            assert grammar is not None and not isinstance(grammar, Exception)
            expected_grammar = (
                GuidanceGrammar if expected_backend == "guidance" else XgrammarGrammar
            )
            assert isinstance(grammar, expected_grammar)
            mask = manager.grammar_bitmask(
                {request.request_id: request}, [request.request_id], {}
            )
            assert mask is not None and mask.dtype == np.int32
            assert mask.shape == (1, (config.model_config.get_vocab_size() + 31) // 32)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            assert grammar.validate_tokens(token_ids) == token_ids
            first = token_ids[0]
            assert mask[0].view(np.uint32)[first >> 5] & np.uint32(1 << (first & 31))
            if kind == "json":
                bad = tokenizer.encode("5", add_special_tokens=False) + [
                    tokenizer.eos_token_id
                ]
                assert grammar.validate_tokens(bad) != bad
            else:
                wrong_name = tokenizer.encode(
                    call.replace("lookup", "other"), add_special_tokens=False
                )
                assert grammar.validate_tokens(wrong_name) != wrong_name
            assert grammar.accept_tokens(request.request_id, token_ids)
            assert grammar.accept_tokens(request.request_id, [tokenizer.eos_token_id])
            assert grammar.is_terminated()
    finally:
        manager.clear_backend()
    assert manager.backend is None


@pytest.fixture(scope="module")
def added_tokenizer_path(tmp_path_factory):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    assert tokenizer.add_tokens([f"<extra_{i}>" for i in range(64)]) == 64
    path = tmp_path_factory.mktemp("mixed-added-tokenizer")
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def larger_native_tokenizer_path(tmp_path_factory):
    """Real byte-level BPE tokens, not HF-added/control tokens suppressed by llg."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    path = tmp_path_factory.mktemp("mixed-native-tokenizer")
    tokenizer.save_pretrained(path)
    tokenizer_file = path / "tokenizer.json"
    spec = json.loads(tokenizer_file.read_text())
    vocab = spec["model"]["vocab"]
    assert len(vocab) == 50257
    for index in range(64):
        token = f"nativeextra{index}_"
        assert token not in vocab
        vocab[token] = len(vocab)
    tokenizer_file.write_text(json.dumps(spec))
    vocab_file = path / "vocab.json"
    if vocab_file.exists():
        vocab_file.write_text(json.dumps(vocab))
    return str(path)


@pytest.mark.parametrize("guidance_first", [False, True])
@pytest.mark.parametrize("async_grammar", [False, True])
@pytest.mark.parametrize("parallel_fill", [False, True])
@pytest.mark.cpu_test
def test_mixed_batch_with_tokenizer_larger_than_model(
    added_tokenizer_path, guidance_first, async_grammar, parallel_fill
):
    """Native work shapes differ; both batch rows still mask model logits."""
    config = VllmConfig(
        model_config=ModelConfig(model=TOKENIZER, tokenizer=added_tokenizer_path),
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=256, max_model_len=1024, is_encoder_decoder=False
        ),
        structured_outputs_config=StructuredOutputsConfig(
            backend="auto", enable_in_reasoning=True
        ),
    )
    manager = StructuredOutputManager(config)
    manager._use_async_grammar_compilation = async_grammar
    if parallel_fill:
        manager.fill_bitmask_parallel_threshold = 1
        manager.fill_bitmask_parallel_batch_size = 1
    tokenizer = manager.tokenizer
    model_vocab = config.model_config.get_vocab_size()
    assert model_vocab == 50257 and len(tokenizer) == 50321
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tag = get_model_structural_tag(
        "qwen_3_coder",
        tools,
        {"type": "function", "function": {"name": "lookup"}},
        reasoning=False,
    )
    cases = [
        (
            "guidance",
            StructuredOutputsParams(json={"type": "integer", "multipleOf": 2}),
            "24680",
        ),
        (
            "xgrammar",
            StructuredOutputsParams(structural_tag=tag.model_dump_json()),
            "<tool_call>\n<function=lookup>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>",
        ),
    ]
    if not guidance_first:
        cases.reverse()
    batch_cases = [
        (backend, backend, outputs, text) for backend, outputs, text in cases
    ]
    batch_cases.append(
        (
            "guidance-second",
            "guidance",
            StructuredOutputsParams(json={"type": "integer", "multipleOf": 2}),
            "13568",
        )
    )
    resolved_backends = {req_id: backend for req_id, backend, _, _ in batch_cases}
    requests = {}
    tokens = {}
    try:
        # Submit all requests before waiting; two native rows share one backend.
        for request_id, backend, outputs, text in batch_cases:
            sampling = SamplingParams(structured_outputs=outputs)
            sampling._validate_structured_outputs(
                config.model_config, config.structured_outputs_config, tokenizer
            )
            assert outputs._backend == backend
            sampling.update_from_generation_config({}, tokenizer.eos_token_id)
            request = Request(
                request_id=request_id,
                prompt_token_ids=[],
                sampling_params=sampling,
                pooling_params=None,
                reasoning_ended=True,
            )
            requests[request_id] = request
            text_tokens = (
                [_single_token(tokenizer, digit) for digit in text]
                if backend == "guidance"
                else tokenizer.encode(text, add_special_tokens=False)
            )
            tokens[request_id] = text_tokens + [tokenizer.eos_token_id]
            manager.grammar_init(request)
        grammars = {}
        for request_id, request in requests.items():
            _wait_for_grammar(request)
            structured_request = request.structured_output_request
            assert structured_request is not None
            grammar = structured_request.grammar
            assert isinstance(grammar, (GuidanceGrammar, XgrammarGrammar))
            grammars[request_id] = grammar
        native_masks = {
            name: backend.allocate_token_bitmask(1)
            for name, backend in manager._backends.items()
        }
        widths = {
            name: backend.allocate_token_bitmask(2).shape[1]
            for name, backend in manager._backends.items()
        }
        assert widths == {"guidance": 1573, "xgrammar": 1571}
        for request_id, backend, outputs, text in batch_cases:
            grammar = grammars[request_id]
            assert (
                grammar.validate_tokens(tokens[request_id][:-1])
                == tokens[request_id][:-1]
            )
            invalid = "5" if backend == "guidance" else text.replace("lookup", "other")
            invalid_ids = tokenizer.encode(invalid, add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            assert grammar.validate_tokens(invalid_ids) != invalid_ids
        steps = min(map(len, tokens.values()))
        assert steps >= 3
        independent_guidance_rows = False
        for step in range(max(map(len, tokens.values()))):
            mask = manager.grammar_bitmask(requests, list(requests), {})
            assert mask is not None and mask.shape == (3, 1571)
            if step < len(tokens["guidance"]):
                ids = list(requests)
                independent_guidance_rows |= not np.array_equal(
                    mask[ids.index("guidance")], mask[ids.index("guidance-second")]
                )
            for row, (request_id, request) in enumerate(requests.items()):
                backend = resolved_backends[request_id]
                grammar = grammars[request_id]
                if step >= len(tokens[request_id]):
                    continue
                grammar.fill_bitmask(native_masks[backend], 0)
                expected = np.unpackbits(
                    native_masks[backend][0].numpy().view(np.uint8),
                    bitorder="little",
                )[:model_vocab]
                actual = np.unpackbits(mask[row].view(np.uint8), bitorder="little")
                np.testing.assert_array_equal(actual[:model_vocab], expected)
                # Invalid padding token IDs must never become model outputs.
                assert mask[row].view(np.uint32)[-1] & np.uint32(0xFFFE0000) == 0
                token = tokens[request_id][step]
                assert mask[row].view(np.uint32)[token >> 5] & np.uint32(
                    1 << (token & 31)
                )
                assert grammar.accept_tokens(request_id, [token])
        assert independent_guidance_rows
        for grammar in grammars.values():
            assert grammar.is_terminated()
    finally:
        manager.clear_backend()
    assert not manager._backend_bitmasks
    assert manager._grammar_bitmask is None


@pytest.mark.cpu_test
def test_guidance_masks_tokenizer_only_ids_from_model_output(
    larger_native_tokenizer_path,
):
    """A native-allowed added token must not leak into padded model-logit bits."""
    config = VllmConfig(
        model_config=ModelConfig(
            model=TOKENIZER, tokenizer=larger_native_tokenizer_path
        ),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(
            backend="guidance", enable_in_reasoning=True
        ),
    )
    manager = StructuredOutputManager(config)
    tokenizer = manager.tokenizer
    assert tokenizer.vocab_size == len(tokenizer) == 50321
    assert tokenizer.decode([50257]) == "nativeextra0_"
    sampling = SamplingParams(
        structured_outputs=StructuredOutputsParams(json={"type": "string"})
    )
    sampling._validate_structured_outputs(
        config.model_config, config.structured_outputs_config, tokenizer
    )
    sampling.update_from_generation_config({}, tokenizer.eos_token_id)
    request = Request(
        request_id="native-tail",
        prompt_token_ids=[],
        sampling_params=sampling,
        pooling_params=None,
        reasoning_ended=True,
    )
    try:
        manager.grammar_init(request)
        _wait_for_grammar(request)
        structured_request = request.structured_output_request
        assert structured_request is not None
        grammar = structured_request.grammar
        assert isinstance(grammar, GuidanceGrammar)
        assert grammar.accept_tokens(
            "native-tail", tokenizer.encode('"', add_special_tokens=False)
        )
        native = manager._backends["guidance"].allocate_token_bitmask(1)
        grammar.fill_bitmask(native, 0)
        native_bits = np.unpackbits(native[0].numpy().view(np.uint8), bitorder="little")
        model_vocab = config.model_config.get_vocab_size()
        assert native_bits[model_vocab : 1571 * 32].any()
        mask = manager.grammar_bitmask(
            {request.request_id: request}, [request.request_id], {}
        )
        assert mask is not None and mask.shape == (1, 1571)
        model_bits = np.unpackbits(mask[0].view(np.uint8), bitorder="little")
        np.testing.assert_array_equal(
            model_bits[:model_vocab], native_bits[:model_vocab]
        )
        assert not model_bits[model_vocab:].any()
        assert grammar.accept_tokens(
            "native-tail", tokenizer.encode('Paris"', add_special_tokens=False)
        )
        assert grammar.accept_tokens("native-tail", [tokenizer.eos_token_id])
        assert grammar.is_terminated()
    finally:
        manager.clear_backend()


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
    engine_reasoner: bool = False
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


class MockEngineReasoner(ParserEngineReasoningAdapter):
    """Ends reasoning implicitly on ``marker``, which is also content, and
    explicitly on ``end_marker`` when one is given."""

    def __init__(self, tokenizer, marker: int, end_marker: int | None = None):
        self.marker = marker
        self.end_marker = end_marker

    @property
    def reasoning_end_token_ids(self):
        return frozenset({self.marker, self.end_marker} - {None})

    @property
    def implicit_reasoning_end_token_ids(self):
        return frozenset({self.marker})

    def find_reasoning_end(self, token_ids):
        for offset, token in enumerate(token_ids):
            if token in self.reasoning_end_token_ids:
                return ReasoningEnd(offset, token == self.marker)
        return ReasoningEnd(len(token_ids), False)


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
    backend: StructuredOutputsBackend,
    prefix: str = "",
    use_reasoner: bool = True,
    reasoning_ended: bool | None = None,
    enable_in_reasoning: bool = False,
    reasoning_parser_kwargs: dict | None = None,
    structured_outputs: StructuredOutputsParams | None = None,
    num_spec_tokens: int = NUM_SPEC_TOKENS,
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
            num_speculative_tokens=num_spec_tokens,
        )
        if num_spec_tokens
        else None,
    )
    manager = StructuredOutputManager(vllm_config)
    if use_reasoner:
        manager.reasoner_cls = MockReasoner

    sampling_params = SamplingParams(
        structured_outputs=structured_outputs
        or StructuredOutputsParams(json=JSON_SCHEMA)
    )
    sampling_params.structured_outputs._backend = backend  # type: ignore[union-attr]
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
    grammar = structured_req.grammar
    assert grammar is not None and not isinstance(grammar, Exception)
    if prompt_ids:
        assert grammar.accept_tokens(
            request.request_id,
            prompt_ids,
        )
    return manager, request


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.cpu_test
def test_reasoning_end_outside_model_vocab_is_not_a_mask_index(tokenizer, backend):
    """A tokenizer-only reasoning marker cannot be sampled by model logits."""
    manager, request = _build_harness(tokenizer, backend, use_reasoner=False)
    try:
        model_vocab = manager.vllm_config.model_config.get_vocab_size()
        manager.reasoner_cls = MockEngineReasoner
        structured_request = request.structured_output_request
        assert structured_request is not None
        structured_request.reasoner = MockEngineReasoner(
            tokenizer, marker=model_vocab + 63
        )
        assert manager._masked_reasoning_ends(request) == ()
    finally:
        manager.clear_backend()


def _row_pattern(bitmask, masked_ids=()) -> str:
    """U: unconstrained; R: all but `masked_ids` allowed; C: grammar rows."""
    reasoning_row = np.full(bitmask.shape[1], -1, dtype=np.int32)
    for token_id in masked_ids:
        reasoning_row.view(np.uint32)[token_id >> 5] &= ~np.uint32(1 << (token_id & 31))

    def kind(row) -> str:
        if (row == -1).all():
            return "U"
        if masked_ids and (row == reasoning_row).all():
            return "R"
        return "C"

    return "".join(kind(row) for row in bitmask)


def _run_real_flow(
    manager: StructuredOutputManager,
    request: Request,
    raw_drafts: list[int],
    expected_validated: list[int],
    expected_row_pattern: str,
    expected_reasoning: bool | None,
    expect_terminated: bool,
    masked_ids: list[int] | None = None,
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
    assert bitmask.shape[0] == NUM_SPEC_TOKENS + 1
    row_pattern = _row_pattern(bitmask[: len(raw_drafts) + 1], masked_ids or ())
    assert row_pattern == expected_row_pattern

    # grammar_bitmask() must rollback any speculative state it advanced.
    assert manager.validate_tokens(request, list(raw_drafts)) == expected_validated

    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert grammar is not None and not isinstance(grammar, Exception)
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
    assert post_accept_bitmask.shape[0] == NUM_SPEC_TOKENS + 1
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
            expected_row_pattern="CUUUU",
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
            expected_row_pattern="CCUUU",
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
            expected_row_pattern="UUCUU",
            expected_reasoning=True,
            expect_terminated=False,
        ),
        id="becomes_active_first_token_invalid",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", THINK_END, "{", "z"),
            expected_validated=(" ", THINK_END, "{"),
            expected_row_pattern="UUCCU",
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
    pytest.param(
        FlowCase(
            prefix='{"a": "',
            raw_drafts=(IMPLICIT_THINK_END,),
            expected_row_pattern="UC",
            expected_reasoning=True,
            expect_terminated=False,
            engine_reasoner=True,
        ),
        id="implicit_end_accepted",
    ),
    pytest.param(
        FlowCase(
            raw_drafts=(" ", IMPLICIT_THINK_END, "{", "}"),
            expected_validated=(" ",),
            # The grammar cannot start with the marker, so reasoning rows mask it.
            expected_row_pattern="RRUUU",
            expected_reasoning=False,
            expect_terminated=False,
            engine_reasoner=True,
        ),
        id="implicit_end_rejected",
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", FLOW_CASES)
def test_real_flow(
    tokenizer,
    backend: StructuredOutputsBackend,
    case: FlowCase,
):
    if backend == "guidance" and case.xfail_guidance:
        pytest.xfail(case.xfail_guidance)

    marker = IMPLICIT_THINK_END if case.engine_reasoner else THINK_END
    reasoner_kwargs = {"marker": _single_token(tokenizer, marker)}
    manager, request = _build_harness(
        tokenizer,
        backend,
        prefix=case.prefix,
        reasoning_ended=case.reasoning_ended,
        reasoning_parser_kwargs=reasoner_kwargs,
    )
    if case.engine_reasoner:
        manager.reasoner_cls = MockEngineReasoner

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
        masked_ids=[reasoner_kwargs["marker"]] if case.engine_reasoner else None,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("structured_outputs", "raw_drafts", "expected_row_pattern"),
    [
        # JSON cannot start with the marker: reasoning rows mask it and keep
        # every other token, including the explicit end, allowed.
        pytest.param(
            StructuredOutputsParams(json=JSON_SCHEMA), (" ", " "), "RRR", id="json"
        ),
        # The grammar can start with the marker: it stays allowed and, once
        # sampled, is the first token the grammar consumes.
        pytest.param(
            StructuredOutputsParams(regex="z[0-9]+"),
            (" ", IMPLICIT_THINK_END, "1"),
            "UUCC",
            id="regex_starts_with_marker",
        ),
    ],
)
def test_reasoning_rows_mask_grammar_invalid_implicit_end(
    tokenizer,
    backend: StructuredOutputsBackend,
    structured_outputs: StructuredOutputsParams,
    raw_drafts: tuple[str, ...],
    expected_row_pattern: str,
):
    marker = _single_token(tokenizer, IMPLICIT_THINK_END)
    manager, request = _build_harness(
        tokenizer,
        backend,
        reasoning_ended=False,
        reasoning_parser_kwargs={
            "marker": marker,
            "end_marker": _single_token(tokenizer, THINK_END),
        },
        structured_outputs=structured_outputs,
    )
    manager.reasoner_cls = MockEngineReasoner
    drafts = _to_token_ids(tokenizer, raw_drafts)
    assert manager.validate_tokens(request, drafts) == drafts

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert _row_pattern(bitmask[: len(drafts) + 1], [marker]) == expected_row_pattern


@pytest.mark.parametrize("backend", BACKENDS)
def test_parallel_fill_masks_grammar_invalid_implicit_end(
    tokenizer, backend: StructuredOutputsBackend
):
    marker, end = _to_token_ids(tokenizer, (IMPLICIT_THINK_END, THINK_END))
    manager, request = _build_harness(
        tokenizer,
        backend,
        reasoning_ended=False,
        reasoning_parser_kwargs={"marker": marker, "end_marker": end},
        num_spec_tokens=0,
    )
    manager.reasoner_cls = MockEngineReasoner

    def row_pattern() -> str:
        bitmask = manager.grammar_bitmask(
            requests={request.request_id: request},
            structured_output_request_ids=[request.request_id],
            scheduled_spec_decode_tokens={},
        )
        assert bitmask is not None
        return _row_pattern(bitmask, [marker])

    manager.fill_bitmask_parallel_threshold = 0
    manager.fill_bitmask_parallel_batch_size = 16
    with ThreadPoolExecutor(max_workers=1) as executor:
        manager.executor_for_fillmask = executor
        assert row_pattern() == "R"
        request.append_output_token_ids([end])
        assert manager.accept_tokens(request, [end])
        assert row_pattern() == "C"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("use_reasoner", "reasoning_ended", "enable_in_reasoning"),
    [
        pytest.param(True, False, True, id="enable_in_reasoning"),
        pytest.param(True, True, False, id="reasoning_ended"),
        pytest.param(False, False, False, id="no_reasoner"),
        pytest.param(True, None, False, id="inferred"),
    ],
)
def test_initial_constraint_activation(
    tokenizer,
    backend: StructuredOutputsBackend,
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

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={},
    )
    assert bitmask is not None
    request.append_output_token_ids([open_brace])
    assert manager.accept_tokens(request, [open_brace])

    # Short-circuit paths must not construct a request-local reasoner (#52573).
    if enable_in_reasoning or reasoning_ended is True:
        structured_req = request.structured_output_request
        assert structured_req is not None
        assert structured_req.reasoner is None


@pytest.mark.parametrize("backend", REGEX_BACKENDS)
@pytest.mark.parametrize(
    ("raw_drafts", "expected_validated", "expected_row_pattern", "terminated"),
    [
        pytest.param(("1", "2"), None, "CCC", False, id="all_valid"),
        # EOS is only valid once the regex has matched; rows after EOS are
        # unconstrained since the request stops there.
        pytest.param(("1", EOS, "2"), ("1", EOS), "CCUU", True, id="terminates"),
        pytest.param((EOS, "1"), (), "CUU", False, id="early_eos"),
    ],
)
def test_regex_flow(
    tokenizer,
    backend: StructuredOutputsBackend,
    raw_drafts: tuple[str, ...],
    expected_validated: tuple[str, ...] | None,
    expected_row_pattern: str,
    terminated: bool,
):
    manager, request = _build_harness(
        tokenizer,
        backend,
        use_reasoner=False,
        structured_outputs=StructuredOutputsParams(regex=DIGITS_REGEX),
    )
    validated = _to_token_ids(
        tokenizer, raw_drafts if expected_validated is None else expected_validated
    )
    _run_real_flow(
        manager,
        request,
        raw_drafts=_to_token_ids(tokenizer, raw_drafts),
        expected_validated=validated,
        expected_row_pattern=expected_row_pattern,
        # accept_tokens() latches reasoning_ended once it accepts any token.
        expected_reasoning=True if validated else None,
        expect_terminated=terminated,
    )


@pytest.mark.parametrize("backend", REGEX_BACKENDS)
def test_rejected_draft_keeps_later_rows_constrained(
    tokenizer, backend: StructuredOutputsBackend
):
    """A draft the grammar rejects during bitmask fill must not unconstrain
    the remaining rows or the bonus row."""
    manager, request = _build_harness(
        tokenizer,
        backend,
        use_reasoner=False,
        structured_outputs=StructuredOutputsParams(regex=DIGITS_REGEX),
    )
    drafts = _to_token_ids(tokenizer, ("1", "z", "2"))
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert _row_pattern(bitmask[: len(drafts) + 1]) == "CCCC"
    assert (bitmask[2] == bitmask[1]).all()
    assert (bitmask[3] == bitmask[1]).all()


def test_outlines_termination(tokenizer):
    """outlines_core never advances on EOS; termination must still be
    tracked exactly once per accepted EOS and undone by rollback."""
    manager, request = _build_harness(
        tokenizer,
        "outlines",
        use_reasoner=False,
        structured_outputs=StructuredOutputsParams(regex=DIGITS_REGEX),
    )
    structured_req = request.structured_output_request
    assert structured_req is not None
    grammar = structured_req.grammar
    assert isinstance(grammar, OutlinesGrammar)
    one, eos = _to_token_ids(tokenizer, ("1", EOS))

    assert not grammar.accept_tokens(request.request_id, [eos])
    assert grammar.accept_tokens(request.request_id, [one])
    # Polling must not change the result.
    assert not any(grammar.is_terminated() for _ in range(3))
    assert grammar.accept_tokens(request.request_id, [eos])
    assert all(grammar.is_terminated() for _ in range(3))
    assert grammar.num_processed_tokens == 2

    grammar.rollback(2)
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0
    assert grammar.validate_tokens([eos]) == []
    assert grammar.accept_tokens(request.request_id, [one, eos, one])
    assert grammar.is_terminated()


def test_outlines_choice_with_non_bmp_characters(tokenizer):
    """The choice spec is JSON, which encodes emoji as surrogate pairs."""
    choices = ["😀 yes", "no"]
    backend = OutlinesBackend(
        VllmConfig(), tokenizer=tokenizer, vocab_size=len(tokenizer)
    )
    request_type, grammar_spec = get_structured_output_key(
        StructuredOutputsParams(choice=choices)
    )

    grammar = backend.compile_grammar(request_type, grammar_spec)

    assert grammar.accept_tokens("", tokenizer.encode(choices[0]))


def test_clear_backend_waits_for_running_grammar_and_cancels_pending(
    tokenizer, monkeypatch
):
    """Keep the compiler alive until running work finishes during shutdown."""
    manager, request = _build_harness(tokenizer, "xgrammar", use_reasoner=False)
    manager.executor.shutdown(wait=True)
    manager.executor = ThreadPoolExecutor(max_workers=1)
    started, release = Event(), Event()
    assert manager.backend is not None
    compile_grammar = manager.backend.compile_grammar

    def delayed_compile(*args, **kwargs):
        started.set()
        assert release.wait(MAX_WAIT_SECONDS)
        return compile_grammar(*args, **kwargs)

    monkeypatch.setattr(manager.backend, "compile_grammar", delayed_compile)
    try:
        manager.grammar_init(request)
        assert request.structured_output_request is not None
        running = request.structured_output_request._grammar
        assert isinstance(running, Future)
        assert started.wait(MAX_WAIT_SECONDS)
        pending_request = Request(
            request_id="pending-at-shutdown",
            prompt_token_ids=[],
            sampling_params=request.sampling_params,
            pooling_params=None,
        )
        manager.grammar_init(pending_request)
        assert pending_request.structured_output_request is not None
        pending = pending_request.structured_output_request._grammar
        assert isinstance(pending, Future)
        # Cancellation releases the running task without timing-based sleeps.
        pending.add_done_callback(lambda _: release.set())
        manager.clear_backend()
        release.set()
        assert running.result(timeout=MAX_WAIT_SECONDS) is not None
        assert pending.cancelled()
        assert manager.backend is None
        manager.clear_backend()
    finally:
        release.set()
        manager.executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("skip_tokenizer_init", [False, True])
@pytest.mark.parametrize("max_num_seqs", [1, 256])
def test_clear_backend_without_grammar(skip_tokenizer_init, max_num_seqs):
    """Stop initialized thread pools even if no request needed a backend."""
    manager = StructuredOutputManager(
        VllmConfig(
            model_config=ModelConfig(
                tokenizer=TOKENIZER, skip_tokenizer_init=skip_tokenizer_init
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_model_len=128,
                is_encoder_decoder=False,
            ),
            device_config=DeviceConfig(device="cpu"),
        )
    )
    manager.clear_backend()
    manager.clear_backend()
    assert manager.backend is None
    executors = []
    if not skip_tokenizer_init:
        executors.append(manager.executor)
    if max_num_seqs > manager.fill_bitmask_parallel_threshold:
        executors.append(manager.executor_for_fillmask)
    for executor in executors:
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            executor.submit(int, 0)
