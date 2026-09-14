# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

import pytest

from vllm.config import StructuredOutputsConfig
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

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
    ],
)
def test_degenerate_structured_outputs_rejected(structured_outputs, match):
    """json_object=False and an empty json schema pass the `is not None`
    exclusivity check but resolve to no structured-output key, so they must be
    rejected at request validation (-> 400) instead of reaching and crashing
    the engine."""
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
        # multipleOf is unsupported by xgrammar, patternProperties also by guidance.
        (
            {
                "type": "object",
                "properties": {"n": {"type": "integer", "multipleOf": 2}},
            },
            "guidance",
        ),
        (
            {"type": "object", "patternProperties": {"^a": {"type": "string"}}},
            "outlines",
        ),
    ],
)
def test_auto_backend_falls_back_on_unsupported_schema(schema, expected_backend):
    """`auto` falls back on rejection, so it must catch what the validators raise."""
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
    params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        StructuredOutputsConfig(backend="auto"),
        tokenizer=object(),
    )
    assert params.structured_outputs._backend == expected_backend
