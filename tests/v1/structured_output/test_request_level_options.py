# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vllm-project/vllm#48765.

The request-level `disable_any_whitespace` must reach grammar compilation
in every backend that supports it; `None` inherits the server setting.
"""

import pytest

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

TOKENIZER = "openai-community/gpt2"
SCHEMA = '{"type": "object", "properties": {"a": {"type": "integer"}}}'
SPACED = '{"a": 1}'
BACKENDS = ["xgrammar", "guidance"]


def _manager(backend: str, server: bool) -> StructuredOutputManager:
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(
            backend=backend, disable_any_whitespace=server
        ),
        # synchronous grammar compilation
        parallel_config=ParallelConfig(
            distributed_executor_backend="external_launcher"
        ),
    )
    return StructuredOutputManager(vllm_config)


def _accepts_spaced(
    manager: StructuredOutputManager,
    backend: str,
    request_value: bool | None,
    rid: str,
) -> bool:
    params = StructuredOutputsParams(json=SCHEMA, disable_any_whitespace=request_value)
    params._backend = backend
    request = Request(
        rid,
        prompt_token_ids=[0],
        sampling_params=SamplingParams(structured_outputs=params),
        pooling_params=None,
    )
    manager.grammar_init(request)
    assert request.structured_output_request is not None
    grammar = request.structured_output_request.grammar
    assert grammar is not None and not isinstance(grammar, Exception)
    return grammar.accept_tokens(rid, manager.tokenizer.encode(SPACED))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("server", "request_value", "whitespace_allowed"),
    [
        (False, None, True),
        (False, True, False),
        (True, None, False),
        (True, False, True),
    ],
)
def test_request_value_overrides_server(
    backend: str, server: bool, request_value: bool | None, whitespace_allowed: bool
):
    manager = _manager(backend, server)
    accepted = _accepts_spaced(manager, backend, request_value, "r")
    assert accepted == whitespace_allowed


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("first", [True, False])
def test_same_schema_different_flag_not_served_from_cache(backend: str, first: bool):
    manager = _manager(backend, server=False)
    second = not first
    assert _accepts_spaced(manager, backend, first, "r1") == (not first)
    assert _accepts_spaced(manager, backend, second, "r2") == (not second)


STRUCTURAL_TAG = (
    '{"type": "structural_tag", "format": {"type": "triggered_tags",'
    ' "triggers": ["<tool_call>"], "tags": [{"begin": "<tool_call>",'
    ' "content": {"type": "json_schema", "json_schema": ' + SCHEMA + "},"
    ' "end": "</tool_call>"}]}}'
)


@pytest.mark.parametrize(
    ("server", "request_value", "whitespace_allowed"),
    [
        (False, None, True),
        (False, True, False),
        (True, None, False),
        (True, False, True),
    ],
)
def test_structural_tag_follows_disable_any_whitespace(
    server: bool, request_value: bool | None, whitespace_allowed: bool
):
    manager = _manager("xgrammar", server)
    params = StructuredOutputsParams(
        structural_tag=STRUCTURAL_TAG, disable_any_whitespace=request_value
    )
    params._backend = "xgrammar"
    request = Request(
        "r",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(structured_outputs=params),
        pooling_params=None,
    )
    manager.grammar_init(request)
    assert request.structured_output_request is not None
    grammar = request.structured_output_request.grammar
    assert grammar is not None and not isinstance(grammar, Exception)
    spaced = manager.tokenizer.encode("<tool_call>" + SPACED + "</tool_call>")
    assert grammar.accept_tokens("r", spaced) == whitespace_allowed
