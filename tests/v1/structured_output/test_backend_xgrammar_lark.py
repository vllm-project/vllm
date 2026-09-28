# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lark grammars in the xgrammar backend.

`structured_outputs.grammar` accepts either EBNF or Lark. The xgrammar backend
parses Lark natively, both when validating a request and when compiling it, so
the request's grammar is never rewritten and frontends that skip Python
request validation get the same behavior.
"""

import pytest
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoiceEnum
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    validate_xgrammar_grammar,
)

pytestmark = pytest.mark.cpu_test

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257
MISTRAL_TOKENIZER = "mistralai/Ministral-3-3B-Instruct-2512"

# Uses Lark features beyond plain rules: a regex terminal, `%import common`,
# and a rule alias. The Codex `apply_patch` tool grammar has the same shape.
LARK_GRAMMAR = """
start: "id=" NUMBER tag? LF?
tag: " #" /[a-z]+/ -> label
NUMBER: /[0-9]+/
%import common.LF
"""

EBNF_GRAMMAR = 'root ::= "id=" [0-9]+'


def _validate(
    grammar: str, backend: str = "xgrammar", tokenizer: object | None = None
) -> StructuredOutputsParams:
    params = SamplingParams(structured_outputs=StructuredOutputsParams(grammar=grammar))
    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend=backend),
        tokenizer=tokenizer or object(),
    )
    assert params.structured_outputs is not None
    return params.structured_outputs


class _StubModelConfig:
    is_diffusion = False


GRAMMARS = [
    pytest.param(LARK_GRAMMAR, id="lark"),
    pytest.param(EBNF_GRAMMAR, id="ebnf"),
]


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_validation_accepts_grammar_without_rewriting_it(grammar: str):
    assert _validate(grammar).grammar == grammar


def test_auto_backend_selects_xgrammar_for_lark():
    assert _validate(LARK_GRAMMAR, backend="auto")._backend == "xgrammar"


@pytest.mark.parametrize("mode", [ToolChoiceEnum.required, ToolChoiceEnum.none])
def test_auto_backend_selects_guidance_for_mistral_lark(mode: ToolChoiceEnum):
    """The Mistral tool parser generates Lark grammars for llguidance, which
    keeps special tokens such as `[TOOL_CALLS]` out of regex matches."""
    tokenizer = MistralTokenizer.from_pretrained(MISTRAL_TOKENIZER)
    factory = tokenizer.grammar_factory
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    grammar = factory.get_lark_from_jinja(
        template=factory.select_jinja_template(),
        mode=mode,
        tools=[Tool.from_openai(tool)],
        json_schema=None,
        parallel_tool_calls=True,
        json_only=False,
    )
    assert _validate(grammar, "auto", tokenizer)._backend == "guidance"


@pytest.mark.parametrize(
    "grammar",
    [
        pytest.param('rule: "hello"', id="missing-start-rule"),
        pytest.param("start: 'hello'", id="single-quoted-literal"),
    ],
)
def test_validation_rejects_invalid_lark(grammar: str):
    params = SamplingParams(structured_outputs=StructuredOutputsParams(grammar=grammar))
    with pytest.raises(VLLMValidationError, match="Invalid grammar specification"):
        validate_xgrammar_grammar(params)


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
        vocab_size=VOCAB_SIZE,
    )


@pytest.mark.parametrize("grammar", GRAMMARS)
@pytest.mark.parametrize(
    "text, accepted",
    [("id=42", True), ("id=x", False)],
)
def test_compile_grammar_matches_lark_and_ebnf(
    backend: XgrammarBackend, grammar: str, text: str, accepted: bool
):
    compiled = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, grammar)
    token_ids = backend.tokenizer.encode(text, add_special_tokens=False)
    assert compiled.accept_tokens("req", token_ids) is accepted


def test_compile_grammar_uses_lark_semantics(backend: XgrammarBackend):
    compiled = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, LARK_GRAMMAR)
    token_ids = backend.tokenizer.encode("id=7 #ok\n", add_special_tokens=False)
    assert compiled.accept_tokens("req", token_ids)
    assert compiled.is_terminated() or compiled.accept_tokens(
        "req", [backend.tokenizer.eos_token_id]
    )
