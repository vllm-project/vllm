# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

import json

import pytest
from transformers import MistralCommonBackend, TokenizersBackend

from vllm.config import StructuredOutputsConfig
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.sampling_params import (
    MAX_STRUCTURED_OUTPUT_JSON_NESTING,
    STRUCTURAL_TAG_WRAPPER_NESTING,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tokenizers import mistral as mistral_tokenizers
from vllm.v1.structured_output import backend_guidance

pytestmark = pytest.mark.cpu_test

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "customer": {"type": "string"},
    },
    "required": ["invoice_id", "customer"],
    "additionalProperties": False,
}


class _StubModelConfig:
    def __init__(self, is_diffusion: bool):
        self.is_diffusion = is_diffusion


def test_structured_outputs_rejected_for_diffusion_models():
    """Diffusion LLMs denoise the canvas in parallel, which is incompatible
    with the token-by-token grammar FSM. The request must fail with a clear
    validation error instead of an FSM rejection mid-generation (#45436)."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
    )
    with pytest.raises(VLLMValidationError, match="not yet supported for diffusion"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=True),
            StructuredOutputsConfig(),
            tokenizer=None,
        )


def test_plain_request_allowed_for_diffusion_models():
    """Requests without structured outputs are unaffected by the guard."""
    params = SamplingParams()
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=True),
        StructuredOutputsConfig(),
        tokenizer=None,
    )


@pytest.mark.parametrize(
    "structured_outputs, match",
    [
        (StructuredOutputsParams(json_object=False), "json_object must be True"),
        (StructuredOutputsParams(json=""), "json cannot be an empty string"),
        (
            StructuredOutputsParams(structural_tag=""),
            "structural_tag cannot be an empty string",
        ),
    ],
)
def test_degenerate_structured_outputs_rejected(structured_outputs, match):
    """json_object=False and an empty json schema pass the `is not None`
    exclusivity check but resolve to no structured-output key, so they must be
    rejected at request validation (-> 400). Empty `structural_tag` is rejected
    for the same reason: `json.loads("")` in `compile_grammar` would otherwise
    raise JSONDecodeError and surface as a per-request engine error."""
    params = SamplingParams(structured_outputs=structured_outputs)
    with pytest.raises(VLLMValidationError, match=match):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )


@pytest.mark.parametrize(
    "regex",
    [
        "\x00",  # a lone leading NUL
        "\x00\x01\x02\x1f",  # a NUL followed by other control chars
        "[0-9]\x00",  # an embedded NUL
    ],
)
def test_regex_with_nul_byte_rejected(regex):
    """A NUL byte is never meaningful in a structured-outputs regex and is not
    handled by xgrammar's native regex converter. It must be rejected at request
    validation in every backend mode (a clean 400), instead of reaching that
    native code or silently falling back to another backend in the default
    'auto' mode."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(regex=regex))

    # Rejected before backend selection, so it is a 400 even in 'auto' mode
    # (which would otherwise catch the error and fall back to another backend).
    with pytest.raises(VLLMValidationError, match="NUL"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(),
            tokenizer=object(),
        )

    # The xgrammar backend also rejects it directly (defense in depth), before
    # the pattern reaches the native from_regex call.
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    with pytest.raises(ValueError, match="NUL"):
        validate_xgrammar_grammar(params)


def _nested_array_schema(levels: int) -> str:
    # Built as text: json.dumps on a deeply nested dict would itself recurse.
    wrappers = levels - 1
    return (
        '{"type": "array", "items": ' * wrappers + '{"type": "string"}' + "}" * wrappers
    )


def _structural_tag(schema: str) -> str:
    return (
        '{"type": "structural_tag", "format": {"type": "json_schema", '
        f'"json_schema": {schema}}}}}'
    )


@pytest.mark.parametrize(
    "structured_outputs",
    [
        pytest.param(
            {"json": _nested_array_schema(MAX_STRUCTURED_OUTPUT_JSON_NESTING + 1)},
            id="json-str-over-limit",
        ),
        pytest.param(
            {"json": json.loads(_nested_array_schema(2_000))},
            id="json-dict-past-recursion-limit",
        ),
        pytest.param(
            {"json": _nested_array_schema(200_000)},
            id="json-str-past-json-loads-recursion-limit",
        ),
        pytest.param(
            {
                "structural_tag": _structural_tag(
                    _nested_array_schema(
                        MAX_STRUCTURED_OUTPUT_JSON_NESTING
                        + STRUCTURAL_TAG_WRAPPER_NESTING
                    )
                )
            },
            id="structural-tag-over-limit",
        ),
    ],
)
@pytest.mark.parametrize(
    "backend", ["auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"]
)
def test_deeply_nested_schema_rejected(structured_outputs, backend):
    """Converting a schema to a grammar gets steeply more expensive with nesting
    and runs on the API server; past the recursion limit it used to escape as a
    RecursionError (HTTP 500 with an empty message). Deep nesting must be a
    validation error before any backend touches the schema."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(**structured_outputs)
    )
    with pytest.raises(VLLMValidationError, match="nested too deeply"):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(backend=backend),
            tokenizer=object(),
        )


@pytest.mark.parametrize("field", ["json", "structural_tag"])
def test_xgrammar_validation_rejects_deeply_nested_schema(field):
    """Request parsing and tool parsers call validate_xgrammar_grammar directly,
    before _validate_structured_outputs, so it must check nesting itself."""
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    schema = _nested_array_schema(2_000)
    value = _structural_tag(schema) if field == "structural_tag" else schema
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(**{field: value})
    )
    with pytest.raises(VLLMValidationError, match="nested too deeply"):
        validate_xgrammar_grammar(params)


@pytest.mark.parametrize("field", ["json", "structural_tag"])
def test_schema_at_nesting_limit_accepted(field):
    """A structural tag wraps its schema in a few more levels (more for the
    tags tool parsers generate), which must not count against the schema."""
    schema = _nested_array_schema(MAX_STRUCTURED_OUTPUT_JSON_NESTING)
    value = _structural_tag(schema) if field == "structural_tag" else schema
    structured_outputs = StructuredOutputsParams(**{field: value})
    SamplingParams(structured_outputs=structured_outputs)._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(),
        tokenizer=object(),
    )
    assert structured_outputs._backend == "xgrammar"


INVALID_JSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "str"}}}
STRUCTURAL_TAG = (
    '{"type": "structural_tag", "format": {"type": "const_string", "value": "hi"}}'
)


@pytest.mark.parametrize(
    "backend, structured_outputs",
    [
        ("xgrammar", StructuredOutputsParams(json=INVALID_JSON_SCHEMA)),
        ("outlines", StructuredOutputsParams(json=INVALID_JSON_SCHEMA)),
        ("auto", StructuredOutputsParams(json=INVALID_JSON_SCHEMA)),
        ("auto", StructuredOutputsParams(json='{"type": ')),
        ("xgrammar", StructuredOutputsParams(grammar="not a grammar")),
        ("guidance", StructuredOutputsParams(grammar="not a grammar")),
        ("lm-format-enforcer", StructuredOutputsParams(grammar="not a grammar")),
        ("outlines", StructuredOutputsParams(regex="(")),
        ("guidance", StructuredOutputsParams(structural_tag='{"nope": 1}')),
        ("outlines", StructuredOutputsParams(structural_tag=STRUCTURAL_TAG)),
        ("lm-format-enforcer", StructuredOutputsParams(structural_tag=STRUCTURAL_TAG)),
        ("outlines", StructuredOutputsParams(json_object=True)),
    ],
)
def test_unsupported_grammar_is_a_client_error(backend, structured_outputs):
    """Only `VLLMClientError` survives `AsyncLLM.generate` untouched; anything else
    is wrapped in `EngineGenerateError` and served as a 500 instead of a 400."""
    params = SamplingParams(structured_outputs=structured_outputs)
    with pytest.raises(VLLMClientError):
        params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            StructuredOutputsConfig(backend=backend),
            tokenizer=object(),
        )


@pytest.mark.parametrize(
    "schema, expected_backend",
    [
        # multipleOf is unsupported by xgrammar.
        (
            {
                "type": "object",
                "properties": {"n": {"type": "integer", "multipleOf": 2}},
            },
            "guidance",
        ),
        (
            {
                "type": "object",
                "properties": {"n": {"type": ["number", "null"], "multipleOf": 3}},
            },
            "guidance",
        ),
        (
            {
                "type": ["string", "null"],
                "pattern": "^a+$",
                "maxLength": 2,
            },
            "guidance",
        ),
        # patternProperties + properties is also unsupported by guidance.
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "patternProperties": {"^a$": {"type": "string"}},
            },
            "outlines",
        ),
    ],
)
def test_auto_backend_falls_back_on_unsupported_schema(schema, expected_backend):
    """`auto` falls back on rejection, so it must catch what the validators raise."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    # trick to create tokenizer that guidance backend accepts
    tokenizer = object.__new__(TokenizersBackend)
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=tokenizer,
    )
    assert params.structured_outputs is not None
    assert params.structured_outputs._backend == expected_backend


# ================================================
# Test validate_structured_outputs functionality
# ================================================


class TestValidateStructuredOutputs:
    class TestPR52298Regression:
        class _StubSlowTokenizer:
            is_fast = False

        class _StubVllmMistralTokenizer:
            IS_MISTRAL_TOKENIZER = True

            def __init__(self, is_tekken: bool):
                self.is_tekken = is_tekken
                self.llg_tokenizer = object()

        @staticmethod
        def _validate_guidance(tokenizer):
            params = SamplingParams(
                structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
            )
            params._validate_structured_outputs(
                _StubModelConfig(is_diffusion=False),
                StructuredOutputsConfig(backend="guidance"),
                tokenizer=tokenizer,
            )
            return params

        def _make_tokenizer(self, tokenizer_id: str):
            if tokenizer_id == "slow-hf":
                return self._StubSlowTokenizer()
            if tokenizer_id == "plain-object":
                return object()
            if tokenizer_id == "raw-mistral-common":
                return object.__new__(MistralCommonBackend)
            if tokenizer_id == "vllm-mistral-non-tekken":
                return self._StubVllmMistralTokenizer(is_tekken=False)
            if tokenizer_id == "fast-hf":
                return object.__new__(TokenizersBackend)
            if tokenizer_id == "vllm-mistral-tekken":
                return self._StubVllmMistralTokenizer(is_tekken=True)
            raise ValueError(tokenizer_id)

        @pytest.mark.parametrize(
            "tokenizer_id, patch_mistral",
            [
                pytest.param("slow-hf", False, id="slow-hf"),
                pytest.param("plain-object", False, id="plain-object"),
                pytest.param("raw-mistral-common", False, id="raw-mistral-common"),
                pytest.param(
                    "vllm-mistral-non-tekken",
                    True,
                    id="vllm-mistral-non-tekken",
                ),
            ],
        )
        def test_unsupported_tokenizer(self, monkeypatch, tokenizer_id, patch_mistral):
            if patch_mistral:
                monkeypatch.setattr(
                    mistral_tokenizers,
                    "MistralTokenizer",
                    self._StubVllmMistralTokenizer,
                )
            with pytest.raises(VLLMValidationError):
                self._validate_guidance(self._make_tokenizer(tokenizer_id))

        @pytest.mark.parametrize(
            "tokenizer_id, patch_mistral",
            [
                pytest.param("fast-hf", False, id="fast-hf"),
                pytest.param("vllm-mistral-tekken", True, id="vllm-mistral-tekken"),
            ],
        )
        def test_supported_tokenizer(self, monkeypatch, tokenizer_id, patch_mistral):
            if patch_mistral:
                monkeypatch.setattr(
                    mistral_tokenizers,
                    "MistralTokenizer",
                    self._StubVllmMistralTokenizer,
                )
                monkeypatch.setattr(
                    backend_guidance,
                    "validate_guidance_grammar",
                    lambda *_args, **_kwargs: None,
                )
            params = self._validate_guidance(self._make_tokenizer(tokenizer_id))
            assert params.structured_outputs._backend == "guidance"
