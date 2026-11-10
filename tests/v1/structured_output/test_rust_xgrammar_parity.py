# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential tests: the Rust xgrammar structured output manager
(`vllm._rust_core.RustStructuredOutputManager`) against Python's
`StructuredOutputManager` + `XgrammarBackend`.

Both run xgrammar 0.2.8's C++ engine, so bitmasks, accept/validate/rollback
results, and termination must match bit for bit on random token walks with
real tokenizers.
"""

import json
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import regex as re

rust_core = pytest.importorskip("vllm._rust_core")

import xgrammar as xgr  # noqa: E402

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig  # noqa: E402
from vllm.config.model import ModelConfig  # noqa: E402
from vllm.config.speculative import SpeculativeConfig  # noqa: E402
from vllm.parser.engine.parser_engine import ReasoningEnd  # noqa: E402
from vllm.reasoning import ReasoningParser  # noqa: E402
from vllm.reasoning.deepseek_r1_reasoning_parser import (  # noqa: E402
    DeepSeekR1ReasoningParser,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from vllm.v1.structured_output.backend_types import (  # noqa: E402
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend  # noqa: E402
from vllm.v1.structured_output.request import (  # noqa: E402
    get_structured_output_key,
)

RustStructuredOutputManager = rust_core.RustStructuredOutputManager

# (model, lm_head vocab size): byte-level BPE (two vocab sizes) and
# byte-fallback BPE tokenizers.
MODELS = [
    ("Qwen/Qwen3-0.6B", 151936),
    ("meta-llama/Llama-3.2-1B-Instruct", 128256),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 32000),
]
MANAGER_MODEL = "Qwen/Qwen3-0.6B"

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["name", "age"],
}
NESTED_SCHEMA = {
    "$defs": {
        "point": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {
        "kind": {"enum": ["circle", "polygon", 3, None]},
        "points": {"type": "array", "items": {"$ref": "#/$defs/point"}},
        "meta": {
            "anyOf": [
                {"type": "string", "pattern": "^[a-z]{2,4}$"},
                {"type": "boolean"},
                {"type": "object", "additionalProperties": {"type": "integer"}},
            ]
        },
        "when": {"type": "string", "format": "date-time"},
    },
    "required": ["kind", "points"],
}
ARRAY_SCHEMA = {
    "type": "array",
    "items": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["a", "b"]}]},
    "minItems": 1,
    "maxItems": 4,
}
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
    "required": ["city"],
}

GRAMMARS: list[tuple[str, str]] = [
    ("JSON", json.dumps(PERSON_SCHEMA)),
    ("JSON", json.dumps(NESTED_SCHEMA)),
    ("JSON", json.dumps(ARRAY_SCHEMA)),
    ("JSON", '{"type": "string", "minLength": 2, "maxLength": 8}'),
    ("JSON_OBJECT", ""),
    ("REGEX", r"\d{3}-\d{4}"),
    ("REGEX", r"(yes|no|maybe)\.?"),
    ("REGEX", r"[a-z]+@[a-z]+\.(com|org)"),
    ("REGEX", r"\w+( \w+){0,4}[.!?]"),
    (
        "GRAMMAR",
        (
            'root ::= expr\nexpr ::= term (("+" | "-") term)*\n'
            'term ::= factor (("*" | "/") factor)*\n'
            'factor ::= [0-9]+ | "(" expr ")"'
        ),
    ),
    ("GRAMMAR", 'start: item ("," item)*\nitem: /[a-c]+/ | "zz"\n'),
    ("CHOICE", json.dumps(["alpha", "beta gamma", "δέλτα", 'x"y'])),
    (
        "STRUCTURAL_TAG",
        json.dumps(
            {
                "structures": [
                    {
                        "begin": "<function=get_weather>",
                        "schema": WEATHER_SCHEMA,
                        "end": "</function>",
                    }
                ],
                "triggers": ["<function="],
            }
        ),
    ),
    (
        "STRUCTURAL_TAG",
        json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<tool>"],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": "<tool>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": WEATHER_SCHEMA,
                            },
                            "end": "</tool>",
                        }
                    ],
                    "at_least_one": True,
                },
            }
        ),
    ),
]

INVALID_GRAMMARS: list[tuple[str, str]] = [
    ("JSON", '{"type": "object", "properties": '),
    ("REGEX", "(unclosed"),
    ("GRAMMAR", 'root ::= "a" undefined_rule'),
    ("STRUCTURAL_TAG", '{"type": "structural_tag", "format": {"type": "nope"}}'),
]


def _load_tokenizer(model: str):
    from vllm.tokenizers import get_tokenizer

    try:
        return get_tokenizer(model)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {model} unavailable: {e}")


@pytest.fixture(scope="module", params=MODELS, ids=[m for m, _ in MODELS])
def model_and_tokenizer(request):
    model, vocab_size = request.param
    return model, vocab_size, _load_tokenizer(model)


def _python_backend(tokenizer, vocab_size, *, disable_any_whitespace=False, spec=0):
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(
            disable_any_whitespace=disable_any_whitespace
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test")),
        speculative_config=SimpleNamespace(num_speculative_tokens=spec)
        if spec
        else None,
    )
    return XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=vocab_size)


def _allowed(row: np.ndarray, vocab_size: int) -> np.ndarray:
    bits = np.unpackbits(row.view(np.uint8), bitorder="little")[:vocab_size]
    return np.flatnonzero(bits)


def test_tokenizer_info_parity(model_and_tokenizer):
    """The Rust `from_huggingface` + `_filter_special_tokens` port builds the
    same tokenizer infos (decoded vocab, type, stop and special tokens)."""
    _, vocab_size, tokenizer = model_and_tokenizer
    full = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    filtered = XgrammarBackend._filter_special_tokens(tokenizer, full)
    rs_full, rs_filtered = rust_core.xgrammar_tokenizer_info_json(tokenizer, vocab_size)
    assert json.loads(rs_full) == json.loads(full.serialize_json())
    assert json.loads(rs_filtered) == json.loads(filtered.serialize_json())


class _Walk:
    """Applies the same random grammar operations to a Python
    `XgrammarGrammar` and the Rust one, comparing every observable."""

    def __init__(self, py_grammar, rs, rid, vocab_size, eos, rng):
        self.py = py_grammar
        self.rs = rs
        self.rid = rid
        self.vocab_size = vocab_size
        self.eos = eos
        self.rng = rng
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)

    def check_state(self) -> np.ndarray | None:
        assert self.py.is_terminated() == self.rs.grammar_is_terminated(self.rid)
        assert self.py.num_processed_tokens == self.rs.grammar_num_processed_tokens(
            self.rid
        )
        if self.py.is_terminated():
            return None
        self.py.fill_bitmask(self.bitmask, 0)
        py_row = self.bitmask[0].numpy()
        rs_row = np.frombuffer(self.rs.grammar_fill_bitmask(self.rid), dtype=np.int32)
        np.testing.assert_array_equal(py_row, rs_row)
        return py_row.copy()

    def valid_run(self, n: int) -> list[int]:
        """Up to `n` grammar-valid tokens, found by advancing and rolling
        back the Python grammar."""
        run: list[int] = []
        for _ in range(n):
            if self.py.is_terminated():
                break
            self.py.fill_bitmask(self.bitmask, 0)
            allowed = _allowed(self.bitmask[0].numpy(), self.vocab_size)
            if len(allowed) == 0:
                break
            if self.eos in allowed and self.rng.random() < 0.15:
                token = self.eos
            else:
                token = int(self.rng.choice(list(allowed)))
            assert self.py.accept_tokens(self.rid, [token])
            run.append(token)
        if run:
            self.py.rollback(len(run))
        return run

    def random_tokens(self, n: int) -> list[int]:
        return [self.rng.randrange(self.vocab_size) for _ in range(n)]

    def step(self) -> None:
        op = self.rng.random()
        if op < 0.5:
            tokens = self.valid_run(self.rng.randint(1, 4))
            if self.rng.random() < 0.2:
                tokens += self.random_tokens(1)
            assert self.py.accept_tokens(
                self.rid, tokens
            ) == self.rs.grammar_accept_tokens(self.rid, tokens), tokens
        elif op < 0.65:
            tokens = self.random_tokens(self.rng.randint(1, 2))
            assert self.py.accept_tokens(
                self.rid, tokens
            ) == self.rs.grammar_accept_tokens(self.rid, tokens), tokens
        elif op < 0.85:
            draft = self.valid_run(self.rng.randint(0, 5)) + self.random_tokens(
                self.rng.randint(0, 2)
            )
            assert self.py.validate_tokens(draft) == self.rs.grammar_validate_tokens(
                self.rid, draft
            ), draft
        elif self.py.num_processed_tokens > 0:
            n = self.rng.randint(1, min(3, self.py.num_processed_tokens))
            self.py.rollback(n)
            self.rs.grammar_rollback(self.rid, n)


@pytest.mark.parametrize("disable_any_whitespace", [False, True])
def test_grammar_walk_parity(model_and_tokenizer, disable_any_whitespace):
    """Random accept/validate/rollback walks give identical bitmasks, accept
    and validate results, processed-token counts, and termination."""
    model, vocab_size, tokenizer = model_and_tokenizer
    spec = 5
    py_backend = _python_backend(
        tokenizer, vocab_size, disable_any_whitespace=disable_any_whitespace, spec=spec
    )
    rs = RustStructuredOutputManager(
        tokenizer,
        vocab_size,
        max_num_seqs=8,
        num_speculative_tokens=spec,
        disable_any_whitespace=disable_any_whitespace,
        async_grammar_compilation=False,
    )
    eos = tokenizer.eos_token_id
    for gi, (request_type, spec_str) in enumerate(GRAMMARS):
        for stop_ids in (None, [eos, 13]):
            rid = f"{gi}-{stop_ids is not None}"
            py_grammar = py_backend.compile_grammar(
                StructuredOutputOptions[request_type],
                spec_str,
                stop_token_ids=set(stop_ids) if stop_ids else None,
            )
            rs.grammar_init(rid, request_type, spec_str, stop_ids)
            assert rs.grammar_status(rid) == "ready", rs.grammar_status(rid)
            rng = random.Random(f"{model}-{gi}-{disable_any_whitespace}-{rid}")
            walk = _Walk(py_grammar, rs, rid, vocab_size, eos, rng)
            for _ in range(60):
                walk.check_state()
                walk.step()
            walk.check_state()
            py_grammar.reset()
            rs.grammar_reset(rid)
            walk.check_state()
            rs.remove_request(rid)


def test_compile_failures_match():
    """Invalid specs fail in both implementations with the same message
    (`str(exception)`, which the scheduler reports), up to the timestamp."""
    _, vocab_size = MODELS[0]
    tokenizer = _load_tokenizer(MODELS[0][0])
    py_backend = _python_backend(tokenizer, vocab_size)
    rs = RustStructuredOutputManager(
        tokenizer, vocab_size, max_num_seqs=8, async_grammar_compilation=False
    )

    def strip_time(message: str) -> str:
        return re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", message)

    for i, (request_type, spec_str) in enumerate(INVALID_GRAMMARS):
        with pytest.raises(Exception) as exc_info:  # noqa: B017
            py_backend.compile_grammar(StructuredOutputOptions[request_type], spec_str)
        rs.grammar_init(str(i), request_type, spec_str)
        status = rs.grammar_status(str(i))
        assert status[0] == "failed", (request_type, spec_str, status)
        assert strip_time(status[1]) == strip_time(str(exc_info.value))


# ---- Manager-level parity ----


class MockEngineReasoner(ReasoningParser):
    """Ends reasoning implicitly on `marker`, which is also content, and
    explicitly on `end_marker`; reports both by token id."""

    def __init__(self, tokenizer, marker: int, end_marker: int):
        super().__init__(tokenizer)
        self.marker = marker
        self.end_marker = end_marker

    @property
    def reasoning_end_token_ids(self):
        return frozenset({self.marker, self.end_marker})

    @property
    def implicit_reasoning_end_token_ids(self):
        return frozenset({self.marker})

    def find_reasoning_end(self, token_ids):
        for offset, token in enumerate(token_ids):
            if token in self.reasoning_end_token_ids:
                return ReasoningEnd(offset, token == self.marker)
        return ReasoningEnd(len(token_ids), False)

    def is_reasoning_end(self, input_ids):
        return any(t in self.reasoning_end_token_ids for t in input_ids)

    def extract_content_ids(self, input_ids):
        raise NotImplementedError

    def extract_reasoning(self, *args, **kwargs):
        raise NotImplementedError

    def extract_reasoning_streaming(self, *args, **kwargs):
        raise NotImplementedError


def _python_manager(tokenizer, monkeypatch, *, num_spec, max_num_seqs, reasoner_cls):
    import vllm.v1.structured_output as so_module

    monkeypatch.setattr(
        so_module, "cached_tokenizer_from_config", lambda **_: tokenizer
    )
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=MANAGER_MODEL),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=SpeculativeConfig(
            model="[ngram]", num_speculative_tokens=num_spec
        )
        if num_spec
        else None,
    )
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    manager = StructuredOutputManager(vllm_config)
    manager.reasoner_cls = reasoner_cls
    return manager


def _grammar(request: Request) -> StructuredOutputGrammar:
    assert request.structured_output_request is not None
    grammar = request.structured_output_request.grammar
    assert isinstance(grammar, StructuredOutputGrammar)
    return grammar


def _wait_ready(request: Request) -> None:
    deadline = time.time() + 30
    so_request = request.structured_output_request
    assert so_request is not None
    while not so_request._check_grammar_completion():
        assert time.time() < deadline, "grammar compilation timed out"
        time.sleep(0.005)


def _wait_ready_rs(rs, rid: str) -> None:
    deadline = time.time() + 30
    while rs.grammar_status(rid) == "pending":
        assert time.time() < deadline, "grammar compilation timed out"
        time.sleep(0.005)


@pytest.mark.parametrize(
    ("reasoning", "num_spec", "num_requests"),
    [
        (None, 4, 6),
        ("think", 4, 6),
        ("engine", 3, 6),
        (None, 0, 6),
        ("think", 4, 40),  # Rust fills >= 32 requests in parallel
        ("think", 0, 140),  # Python's parallel fill path (> 128, no spec)
    ],
)
def test_manager_parity(monkeypatch, reasoning, num_spec, num_requests):
    """`grammar_bitmask` (incl. spec-decode rows, padding, reasoning rows,
    and stale rows), `trim_spec_tokens`, and `accept_tokens` match Python over
    random multi-step decoding."""
    tokenizer = _load_tokenizer(MANAGER_MODEL)
    vocab_size = 151936
    think_start = tokenizer.convert_tokens_to_ids("<think>")
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    brace = tokenizer.convert_tokens_to_ids("{")
    reasoner_cls = {
        None: None,
        "think": DeepSeekR1ReasoningParser,
        "engine": MockEngineReasoner,
    }[reasoning]
    max_num_seqs = max(num_requests, 8)
    py = _python_manager(
        tokenizer,
        monkeypatch,
        num_spec=num_spec,
        max_num_seqs=max_num_seqs,
        reasoner_cls=reasoner_cls,
    )
    rs = RustStructuredOutputManager(
        tokenizer,
        vocab_size,
        max_num_seqs=max_num_seqs,
        num_speculative_tokens=num_spec,
    )
    rng = random.Random(f"{reasoning}-{num_spec}-{num_requests}")
    eos = tokenizer.eos_token_id
    reasoner_kwargs = (
        {"marker": brace, "end_marker": think_end} if reasoning == "engine" else None
    )

    requests: dict[str, Request] = {}
    for i in range(num_requests):
        request_type, spec_str = GRAMMARS[i % len(GRAMMARS)]
        params = {
            "JSON": lambda s: StructuredOutputsParams(json=s),
            "JSON_OBJECT": lambda s: StructuredOutputsParams(json_object=True),
            "REGEX": lambda s: StructuredOutputsParams(regex=s),
            "GRAMMAR": lambda s: StructuredOutputsParams(grammar=s),
            "CHOICE": lambda s: StructuredOutputsParams(choice=json.loads(s)),
            "STRUCTURAL_TAG": lambda s: StructuredOutputsParams(structural_tag=s),
        }[request_type](spec_str)
        params._backend = "xgrammar"
        sampling_params = SamplingParams(structured_outputs=params)
        sampling_params.update_from_generation_config({}, eos)
        prompt = [rng.randrange(1000, 2000) for _ in range(rng.randint(1, 4))]
        if reasoning == "think":
            # Half the prompts end reasoning in the prompt already.
            prompt += [think_start] + ([think_end] if i % 2 else [])
        request = Request(
            request_id=f"req-{i}",
            prompt_token_ids=prompt,
            sampling_params=sampling_params,
            pooling_params=None,
            reasoning_parser_kwargs=reasoner_kwargs,
        )
        py.grammar_init(request)
        key_type, key_spec = get_structured_output_key(params)
        stop_ids = list(sampling_params.all_stop_token_ids) or None
        if reasoning == "think":
            rs.grammar_init(
                request.request_id,
                key_type.name,
                key_spec,
                stop_ids,
                think_tokens=(think_start, think_end),
            )
        elif reasoning == "engine":
            rs.grammar_init(
                request.request_id,
                key_type.name,
                key_spec,
                stop_ids,
                reasoner=MockEngineReasoner(tokenizer, **reasoner_kwargs),
            )
        else:
            rs.grammar_init(request.request_id, key_type.name, key_spec, stop_ids)
        requests[request.request_id] = request
    for request in requests.values():
        _wait_ready(request)
        _wait_ready_rs(rs, request.request_id)

    def rs_req(request: Request):
        return (
            request.request_id,
            request.prompt_token_ids,
            list(request.all_token_ids),
        )

    def draft_for(request: Request) -> list[int]:
        """Mostly grammar-valid drafts with reasoning ends, invalid tokens,
        and -1 padding mixed in."""
        grammar = _grammar(request)
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        draft: list[int] = []
        advanced = 0
        for _ in range(num_spec):
            r = rng.random()
            if r < 0.1:
                draft.append(-1)
                break
            if r < 0.2:
                draft.append(think_end if reasoning else rng.randrange(vocab_size))
                continue
            if r < 0.3 and reasoning == "engine":
                draft.append(brace)
                continue
            if r < 0.4 or grammar.is_terminated():
                draft.append(rng.randrange(vocab_size))
                continue
            grammar.fill_bitmask(bitmask, 0)
            allowed = _allowed(bitmask[0].numpy(), vocab_size)
            token = int(rng.choice(list(allowed)))
            if grammar.accept_tokens(request.request_id, [token]):
                advanced += 1
            draft.append(token)
        if advanced:
            grammar.rollback(advanced)
        # Pad like the scheduler does for rejected draft slots.
        while len(draft) < num_spec and draft and draft[-1] == -1:
            draft.append(-1)
        return draft

    active = list(requests)
    for step in range(30):
        if not active:
            break
        rng.shuffle(active)
        batch = active[: rng.randint(1, len(active))]
        spec_map = {rid: draft_for(requests[rid]) for rid in batch} if num_spec else {}

        py_mask = py.grammar_bitmask(requests, batch, spec_map)
        rows, words, data = rs.grammar_bitmask(
            [rs_req(requests[r]) for r in batch], spec_map
        )
        rs_mask = np.frombuffer(data, dtype=np.int32).reshape(rows, words)
        assert py_mask.shape == rs_mask.shape, step
        np.testing.assert_array_equal(py_mask, rs_mask, err_msg=f"step {step}")

        stride = num_spec + 1
        for idx, rid in enumerate(batch):
            request = requests[rid]
            drafts = spec_map.get(rid, [])
            trimmed = py.trim_spec_tokens(request, list(drafts))
            assert trimmed == rs.trim_spec_tokens(rs_req(request), list(drafts)), (
                step,
                rid,
            )
            # Like rejection sampling, accept only drafts their row allows,
            # then sample a bonus token from the next row; rarely, force an
            # invalid one to exercise rejection.
            max_accepted = 0
            while max_accepted < len(trimmed) and trimmed[max_accepted] in set(
                _allowed(py_mask[idx * stride + max_accepted], vocab_size).tolist()
            ):
                max_accepted += 1
            num_accepted = rng.randint(0, max_accepted)
            row = py_mask[idx * stride + num_accepted]
            allowed = _allowed(row, vocab_size)
            unconstrained = bool((row == -1).all())
            if rng.random() < 0.05:
                bonus = rng.randrange(vocab_size)
            elif reasoning and unconstrained and rng.random() < 0.3:
                bonus = think_end
            elif eos in allowed and rng.random() < 0.3:
                bonus = eos
            else:
                bonus = int(rng.choice(list(allowed)))
            new_ids = trimmed[:num_accepted] + [bonus]
            request.append_output_token_ids(new_ids)
            py_ok = py.accept_tokens(request, new_ids)
            rs_ok = rs.accept_tokens(rs_req(request), new_ids)
            assert py_ok == rs_ok, (step, rid, new_ids)
            if _grammar(request).is_terminated() or not py_ok:
                active.remove(rid)
                rs.remove_request(rid)
