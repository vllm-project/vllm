# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import xgrammar as xgr

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.parser.plamo3 import (
    BEGIN_TOOL_ARGUMENTS,
    BEGIN_TOOL_NAME,
    BEGIN_TOOL_REQUEST,
    BEGIN_TOOL_REQUESTS,
    END_TOOL_ARGUMENTS,
    END_TOOL_NAME,
    END_TOOL_REQUEST,
    END_TOOL_REQUESTS,
)
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend


class Plamo3Tokenizer:
    """Minimal stand-in for PLaMo's remote-code PythonBackend tokenizer."""

    eos_token_id = 0

    def __init__(self, tokens: list[str]) -> None:
        self._vocab = {token: token_id for token_id, token in enumerate(tokens)}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def init_xgrammar(self):
        stop_token_ids = [self.eos_token_id]
        tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab=list(self._vocab),
            vocab_type=xgr.VocabType.BYTE_FALLBACK,
            vocab_size=len(self._vocab),
            stop_token_ids=stop_token_ids,
        )
        return tokenizer_info, stop_token_ids


def make_vllm_config() -> VllmConfig:
    config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
    )
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="plamo3")
    )
    return config


def test_plamo3_python_tokenizer_compiles_and_accepts_structural_tag():
    call_tokens = [
        BEGIN_TOOL_REQUESTS,
        BEGIN_TOOL_REQUEST,
        BEGIN_TOOL_NAME,
        "get_weather",
        END_TOOL_NAME,
        BEGIN_TOOL_ARGUMENTS,
        '{"city":"Tokyo"}',
        END_TOOL_ARGUMENTS,
        END_TOOL_REQUEST,
        END_TOOL_REQUESTS,
    ]
    tokenizer = Plamo3Tokenizer(["<eos>", *call_tokens])
    backend = XgrammarBackend(
        make_vllm_config(),
        tokenizer=tokenizer,
        vocab_size=len(tokenizer.get_vocab()),
    )
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            },
        )
    ]
    tag = get_model_structural_tag("plamo3", tools, "required", reasoning=False)
    assert tag is not None

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        tag.model_dump_json(),
    )
    token_ids = [tokenizer.get_vocab()[token] for token in call_tokens]
    assert grammar.accept_tokens("plamo3", token_ids)


@pytest.mark.parametrize("value", ['"', "あ"])
def test_plamo3_grammar_matches_decoded_bytes(value):
    pieces = [f"<0x{byte:02X}>" for byte in value.encode("utf-8")]
    tokenizer = Plamo3Tokenizer(list(dict.fromkeys(["<eos>", value, *pieces])))
    backend = XgrammarBackend(
        make_vllm_config(),
        tokenizer=tokenizer,
        vocab_size=len(tokenizer.get_vocab()),
    )
    for tokens in ([value], pieces):
        grammar = backend.compile_grammar(StructuredOutputOptions.REGEX, value)
        assert grammar.accept_tokens(
            "plamo3", [tokenizer.get_vocab()[token] for token in tokens]
        )
