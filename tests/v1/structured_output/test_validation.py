# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

from unittest.mock import Mock

import pytest
from transformers import MistralCommonBackend, PreTrainedTokenizerFast

from vllm.config import StructuredOutputsConfig
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import mistral as mistral_tokenizers
from vllm.v1.structured_output import (
    backend_guidance,
    backend_outlines,
    backend_xgrammar,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

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


class _StubSlowTokenizer:
    is_fast = False


class _StubVllmMistralTokenizer:
    IS_MISTRAL_TOKENIZER = True

    def __init__(self, is_tekken: bool):
        self.is_tekken = is_tekken
        self.llg_tokenizer = None


def _validate(tokenizer, backend):
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
    )
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend=backend),
        tokenizer=tokenizer,
    )
    return params


def _guidance_must_not_run(*_args, **_kwargs):
    pytest.fail("grammar validation must not run for an unsupported tokenizer")


def _reject_xgrammar(*_args, **_kwargs):
    raise VLLMValidationError("unsupported schema")


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


INVALID_JSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "str"}}}


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
@pytest.mark.parametrize("slow_tokenizer", [False, True])
def test_auto_backend_falls_back_on_unsupported_schema(
    schema, expected_backend, slow_tokenizer
):
    """`auto` falls back on rejection, so it must catch what the validators raise."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    if slow_tokenizer and expected_backend == "guidance":
        with pytest.raises(VLLMValidationError, match="No compatible"):
            params._validate_structured_outputs(
                _StubModelConfig(is_diffusion=False),
                StructuredOutputsConfig(backend="auto"),
                tokenizer=_StubSlowTokenizer(),
            )
        return
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=_StubSlowTokenizer()
        if slow_tokenizer
        else object.__new__(PreTrainedTokenizerFast),
    )
    assert params.structured_outputs._backend == expected_backend


@pytest.mark.parametrize(
    "tokenizer_factory, raw_mistral",
    [
        pytest.param(_StubSlowTokenizer, False, id="slow-hf"),
        pytest.param(
            lambda: object.__new__(MistralCommonBackend),
            True,
            id="raw-mistral-spm",
        ),
        pytest.param(
            lambda: object.__new__(MistralCommonBackend),
            "tekken",
            id="raw-mistral-tekken",
        ),
    ],
)
def test_guidance_rejects_unsupported_tokenizers(
    monkeypatch, tokenizer_factory, raw_mistral
):
    """Unsupported tokenizers must fail before EngineCore initializes guidance."""
    if raw_mistral:
        monkeypatch.setattr(
            mistral_tokenizers,
            "mistral_common_tekkenizer",
            lambda tokenizer: object() if raw_mistral == "tekken" else None,
        )
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        _guidance_must_not_run,
    )

    with pytest.raises(VLLMValidationError, match="only supports fast"):
        _validate(tokenizer_factory(), backend="guidance")


@pytest.mark.parametrize(
    "tokenizer_factory, raw_mistral",
    [
        pytest.param(
            lambda: object.__new__(PreTrainedTokenizerFast),
            False,
            id="fast-hf",
        ),
        pytest.param(
            lambda: _StubVllmMistralTokenizer(is_tekken=True),
            True,
            id="vllm-mistral-tekken",
        ),
    ],
)
def test_guidance_allows_supported_tokenizers(
    monkeypatch, tokenizer_factory, raw_mistral
):
    """The early guard must preserve all supported tokenizer paths."""
    if raw_mistral:
        monkeypatch.setattr(
            mistral_tokenizers,
            "MistralTokenizer",
            _StubVllmMistralTokenizer,
        )
    validate_guidance = Mock()
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        validate_guidance,
    )

    params = _validate(tokenizer_factory(), backend="guidance")
    validate_guidance.assert_called_once()
    assert params.structured_outputs._backend == "guidance"


def test_auto_preserves_non_tekken_mistral_outlines_fallback(monkeypatch):
    """Preserve the existing auto fallback for vLLM non-Tekken Mistral."""
    monkeypatch.setattr(
        mistral_tokenizers,
        "MistralTokenizer",
        _StubVllmMistralTokenizer,
    )
    monkeypatch.setattr(
        backend_xgrammar,
        "validate_xgrammar_grammar",
        _reject_xgrammar,
    )
    validate_outlines = Mock()
    monkeypatch.setattr(
        backend_outlines,
        "validate_structured_output_request_outlines",
        validate_outlines,
    )

    params = _validate(
        _StubVllmMistralTokenizer(is_tekken=False),
        backend="auto",
    )
    validate_outlines.assert_called_once()
    assert params.structured_outputs._backend == "outlines"


def test_auto_excludes_guidance_for_slow_tokenizer(monkeypatch):
    """Auto must fail safely when no compatible backend can handle the request."""
    monkeypatch.setattr(
        backend_xgrammar,
        "validate_xgrammar_grammar",
        _reject_xgrammar,
    )
    monkeypatch.setattr(
        backend_guidance,
        "validate_guidance_grammar",
        _guidance_must_not_run,
    )

    with pytest.raises(VLLMValidationError, match="No compatible"):
        _validate(_StubSlowTokenizer(), backend="auto")
