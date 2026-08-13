# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-time validation of structured output requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.structured_output as structured_output
from vllm.config import StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

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

GUIDANCE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "multipleOf": 2},
    },
    "required": ["count"],
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


def test_auto_rejects_mixed_structured_output_backends():
    config = StructuredOutputsConfig(backend="auto")
    xgrammar_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA)
    )
    guidance_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json=GUIDANCE_JSON_SCHEMA)
    )

    xgrammar_params._validate_structured_outputs(
        _StubModelConfig(is_diffusion=False),
        config,
        tokenizer=object(),
    )

    with pytest.raises(VLLMValidationError, match="only supports one backend"):
        guidance_params._validate_structured_outputs(
            _StubModelConfig(is_diffusion=False),
            config,
            tokenizer=object(),
        )


def test_manager_rejects_mixed_structured_output_backends(monkeypatch):
    xgrammar_backend = Mock()
    xgrammar_backend.compile_grammar.return_value = object()
    xgrammar_factory = Mock(return_value=xgrammar_backend)
    guidance_factory = Mock()
    monkeypatch.setattr(structured_output, "XgrammarBackend", xgrammar_factory)
    monkeypatch.setattr(structured_output, "GuidanceBackend", guidance_factory)

    manager = object.__new__(StructuredOutputManager)
    manager.backend = None
    manager.backend_name = None
    manager.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 128)
    )
    manager.tokenizer = object()
    manager._use_async_grammar_compilation = False

    def make_request(backend):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(
                structured_outputs=SimpleNamespace(_backend=backend)
            ),
            structured_output_request=SimpleNamespace(
                structured_output_key=(StructuredOutputOptions.JSON, "{}"),
                grammar=None,
            ),
        )

    manager.grammar_init(make_request("xgrammar"))

    with pytest.raises(VLLMValidationError, match="already using 'xgrammar'"):
        manager.grammar_init(make_request("guidance"))

    xgrammar_factory.assert_called_once()
    xgrammar_backend.compile_grammar.assert_called_once()
    guidance_factory.assert_not_called()
