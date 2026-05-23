# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.structured_output.backend_guidance as guidance_mod
import vllm.v1.structured_output.backend_lm_format_enforcer as lmfe_mod
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

pytestmark = pytest.mark.cpu_test


def test_guidance_compile_grammar_reuses_serialized_grammar(monkeypatch):
    guidance_mod._cached_serialize_guidance_grammar.cache_clear()
    serialize_calls: list[tuple[StructuredOutputOptions, str, bool, bool]] = []
    matcher_calls: list[str] = []

    def fake_serialize(
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        disable_any_whitespace: bool = False,
        disable_additional_properties: bool = False,
    ) -> str:
        serialize_calls.append(
            (
                request_type,
                grammar_spec,
                disable_any_whitespace,
                disable_additional_properties,
            )
        )
        return f"serialized:{request_type.name}:{grammar_spec}"

    class FakeLLMatcher:
        def __init__(self, ll_tokenizer, serialized_grammar, log_level):
            matcher_calls.append(serialized_grammar)

        def get_error(self):
            return None

        def is_stopped(self):
            return False

        def consume_tokens(self, tokens):
            return True

        def validate_tokens(self, tokens):
            return len(tokens)

        def rollback(self, num_tokens):
            return None

        def reset(self):
            return None

    monkeypatch.setattr(guidance_mod, "serialize_guidance_grammar", fake_serialize)
    monkeypatch.setattr(
        guidance_mod,
        "llguidance",
        SimpleNamespace(LLMatcher=FakeLLMatcher),
    )

    backend = GuidanceBackend.__new__(GuidanceBackend)
    backend.vllm_config = SimpleNamespace(speculative_config=None)
    backend.tokenizer = object()
    backend.vocab_size = 128
    backend.disable_any_whitespace = False
    backend.disable_additional_properties = False
    backend.ll_tokenizer = object()

    grammar1 = GuidanceBackend.compile_grammar(
        backend,
        StructuredOutputOptions.JSON,
        '{"type": "object"}',
    )
    grammar2 = GuidanceBackend.compile_grammar(
        backend,
        StructuredOutputOptions.JSON,
        '{"type": "object"}',
    )

    assert len(serialize_calls) == 1
    assert matcher_calls == [
        'serialized:JSON:{"type": "object"}',
        'serialized:JSON:{"type": "object"}',
    ]
    assert grammar1 is not grammar2


def test_lmfe_compile_grammar_reuses_cached_parser(monkeypatch):
    lmfe_mod._cached_compile_character_level_parser.cache_clear()
    parser_builds: list[tuple[str, object]] = []

    class FakeParser:
        def __init__(self, kind: str, payload: object):
            self.kind = kind
            self.payload = payload

    token_enforcer_parsers: list[FakeParser] = []

    class FakeTokenEnforcer:
        def __init__(self, tokenizer_data, parser):
            self.tokenizer_data = tokenizer_data
            self.parser = parser
            self.eos_token_id = -1
            token_enforcer_parsers.append(parser)

    def build_json_schema_parser(spec_dict):
        parser_builds.append(("json", spec_dict))
        return FakeParser("json", spec_dict)

    monkeypatch.setattr(
        lmfe_mod,
        "lmformatenforcer",
        SimpleNamespace(
            JsonSchemaParser=build_json_schema_parser,
            RegexParser=lambda grammar_spec: FakeParser("regex", grammar_spec),
            UnionParser=lambda choices: FakeParser("choice", tuple(choices)),
            StringParser=lambda choice: ("string", choice),
            TokenEnforcer=FakeTokenEnforcer,
        ),
    )

    backend = LMFormatEnforcerBackend.__new__(LMFormatEnforcerBackend)
    backend.vllm_config = SimpleNamespace(speculative_config=None)
    backend.tokenizer = object()
    backend.vocab_size = 128
    backend.tokenizer_data = object()

    grammar1 = LMFormatEnforcerBackend.compile_grammar(
        backend,
        StructuredOutputOptions.JSON,
        '{"type": "object"}',
    )
    grammar2 = LMFormatEnforcerBackend.compile_grammar(
        backend,
        StructuredOutputOptions.JSON,
        '{"type": "object"}',
    )

    assert len(parser_builds) == 1
    assert len(token_enforcer_parsers) == 2
    assert grammar1.token_enforcer is not grammar2.token_enforcer
    assert token_enforcer_parsers[0] is token_enforcer_parsers[1]
    assert token_enforcer_parsers[0].payload == {"type": "object"}


def test_lmfe_post_init_falls_back_to_transformers_builder(monkeypatch):
    sentinel = object()
    vllm_builder_calls: list[tuple[object, int]] = []
    transformers_builder_calls: list[tuple[object, int]] = []
    warning_messages: list[str] = []

    def fail_vllm_builder(tokenizer, vocab_size):
        vllm_builder_calls.append((tokenizer, vocab_size))
        raise ImportError("broken vllm integration")

    def succeed_transformers_builder(tokenizer, vocab_size):
        transformers_builder_calls.append((tokenizer, vocab_size))
        return sentinel

    monkeypatch.setattr(
        lmfe_mod,
        "_cached_build_vllm_token_enforcer_tokenizer_data",
        fail_vllm_builder,
    )
    monkeypatch.setattr(
        lmfe_mod,
        "_cached_build_transformers_token_enforcer_tokenizer_data",
        succeed_transformers_builder,
    )
    monkeypatch.setattr(
        lmfe_mod,
        "logger",
        SimpleNamespace(
            warning_once=lambda msg, *args: warning_messages.append(msg % args)
        ),
    )

    backend = LMFormatEnforcerBackend.__new__(LMFormatEnforcerBackend)
    backend.tokenizer = object()
    backend.vocab_size = 128

    LMFormatEnforcerBackend.__post_init__(backend)

    assert backend.tokenizer_data is sentinel
    assert len(vllm_builder_calls) == 1
    assert len(transformers_builder_calls) == 1
    assert (
        "falling back to the generic transformers tokenizer-data builder"
        in (warning_messages[0])
    )
