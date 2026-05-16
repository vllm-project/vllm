# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.utils.cache import CacheInfo
from vllm.v1.metrics.stats import StructuredOutputCacheStats
from vllm.v1.structured_output import StructuredOutputManager

from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.skip_global_cleanup


class _FakeCompiledGrammar:
    def __init__(self, name: str):
        self.name = name


class _FakeGrammarMatcher:
    def __init__(self, ctx, max_rollback_tokens: int):
        self.ctx = ctx
        self.max_rollback_tokens = max_rollback_tokens

    def accept_token(self, token: int) -> bool:
        return True

    def is_terminated(self) -> bool:
        return False

    def rollback(self, num_tokens: int) -> None:
        return None

    def fill_next_token_bitmask(self, bitmask, idx: int) -> None:
        return None

    def reset(self) -> None:
        return None


class _FakeGrammarCompiler:
    def __init__(self, *args, **kwargs):
        self.compile_json_schema_calls: list[tuple[str, bool]] = []

    def compile_json_schema(self, grammar_spec: str, any_whitespace: bool):
        self.compile_json_schema_calls.append((grammar_spec, any_whitespace))
        return _FakeCompiledGrammar(grammar_spec)


class _FakeTokenizerInfo:
    @staticmethod
    def from_huggingface(tokenizer, vocab_size: int):
        return {"tokenizer": tokenizer, "vocab_size": vocab_size}


def test_xgrammar_backend_reuses_compiled_context(monkeypatch):
    fake_xgr = SimpleNamespace(
        TokenizerInfo=_FakeTokenizerInfo,
        GrammarCompiler=_FakeGrammarCompiler,
        GrammarMatcher=_FakeGrammarMatcher,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.xgr",
        fake_xgr,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.is_mistral_tokenizer",
        lambda tokenizer: False,
    )

    structured_outputs_config = Mock()
    structured_outputs_config.disable_any_whitespace = False
    vllm_config = Mock()
    vllm_config.structured_outputs_config = structured_outputs_config
    vllm_config.speculative_config = None

    backend = XgrammarBackend(
        vllm_config=vllm_config,
        tokenizer=object(),
        vocab_size=4096,
    )

    grammar_spec = '{"type":"object","properties":{"value":{"type":"string"}}}'
    grammar1 = backend.compile_grammar(StructuredOutputOptions.JSON, grammar_spec)
    grammar2 = backend.compile_grammar(StructuredOutputOptions.JSON, grammar_spec)

    assert len(backend.compiler.compile_json_schema_calls) == 1
    assert grammar1.ctx is grammar2.ctx
    assert grammar1.matcher is not grammar2.matcher
    assert grammar1.matcher.ctx is grammar1.ctx
    assert grammar2.matcher.ctx is grammar2.ctx


def test_xgrammar_backend_reports_cache_stats(monkeypatch):
    fake_xgr = SimpleNamespace(
        TokenizerInfo=_FakeTokenizerInfo,
        GrammarCompiler=_FakeGrammarCompiler,
        GrammarMatcher=_FakeGrammarMatcher,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.xgr",
        fake_xgr,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.is_mistral_tokenizer",
        lambda tokenizer: False,
    )

    structured_outputs_config = Mock()
    structured_outputs_config.disable_any_whitespace = False
    vllm_config = Mock()
    vllm_config.structured_outputs_config = structured_outputs_config
    vllm_config.speculative_config = None

    backend = XgrammarBackend(
        vllm_config=vllm_config,
        tokenizer=object(),
        vocab_size=4096,
    )

    grammar_spec = '{"type":"object","properties":{"value":{"type":"string"}}}'
    backend.compile_grammar(StructuredOutputOptions.JSON, grammar_spec)
    backend.compile_grammar(StructuredOutputOptions.JSON, grammar_spec)
    backend.compile_grammar(StructuredOutputOptions.JSON, '{"type":"object"}')

    assert backend.compiled_grammar_cache_stats() == CacheInfo(hits=1, total=3)
    assert backend.compiled_grammar_cache_stats(delta=True) == CacheInfo(
        hits=1,
        total=3,
    )

    backend.clear_compiled_grammar_cache()

    assert backend.compiled_grammar_cache_stats() == CacheInfo(hits=0, total=0)


def test_xgrammar_backend_destroy_clears_cache(monkeypatch):
    fake_xgr = SimpleNamespace(
        TokenizerInfo=_FakeTokenizerInfo,
        GrammarCompiler=_FakeGrammarCompiler,
        GrammarMatcher=_FakeGrammarMatcher,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.xgr",
        fake_xgr,
    )
    monkeypatch.setattr(
        "vllm.v1.structured_output.backend_xgrammar.is_mistral_tokenizer",
        lambda tokenizer: False,
    )

    structured_outputs_config = Mock()
    structured_outputs_config.disable_any_whitespace = False
    vllm_config = Mock()
    vllm_config.structured_outputs_config = structured_outputs_config
    vllm_config.speculative_config = None

    backend = XgrammarBackend(
        vllm_config=vllm_config,
        tokenizer=object(),
        vocab_size=4096,
    )

    backend.compile_grammar(StructuredOutputOptions.JSON, '{"type":"object"}')
    assert len(backend._compiled_grammars) == 1

    backend.destroy()

    assert len(backend._compiled_grammars) == 0


def test_structured_output_manager_exposes_compiled_cache_stats():
    vllm_config = Mock()
    vllm_config.parallel_config.distributed_executor_backend = None
    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.structured_outputs_config.enable_in_reasoning = False

    manager = StructuredOutputManager(vllm_config)
    backend = Mock()
    backend.compiled_grammar_cache_stats.return_value = CacheInfo(hits=3, total=5)
    manager.backend = backend

    assert manager.compiled_grammar_cache_stats() == CacheInfo(hits=3, total=5)
    backend.compiled_grammar_cache_stats.assert_called_once_with(delta=False)


def test_structured_output_manager_clear_backend_logs_cache_stats(monkeypatch):
    vllm_config = Mock()
    vllm_config.parallel_config.distributed_executor_backend = None
    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.structured_outputs_config.enable_in_reasoning = False

    manager = StructuredOutputManager(vllm_config)
    backend = Mock()
    backend.compiled_grammar_cache_stats.return_value = CacheInfo(hits=2, total=4)
    manager.backend = backend
    debug_log = Mock()
    monkeypatch.setattr("vllm.v1.structured_output.logger.debug", debug_log)

    manager.clear_backend()

    backend.destroy.assert_called_once_with()
    debug_log.assert_called_once()


def test_structured_output_manager_logs_periodic_cache_stats(monkeypatch):
    vllm_config = Mock()
    vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.structured_outputs_config.enable_in_reasoning = False

    manager = StructuredOutputManager(vllm_config)
    manager._compiled_grammar_cache_log_interval = 2
    manager._next_compiled_grammar_cache_log_total = 2

    backend = Mock()
    backend.compile_grammar.return_value = Mock()
    backend.compiled_grammar_cache_stats.side_effect = [
        CacheInfo(hits=1, total=1),
        CacheInfo(hits=1, total=2),
        CacheInfo(hits=1, total=2),
    ]
    manager.backend = backend

    info_log = Mock()
    monkeypatch.setattr("vllm.v1.structured_output.logger.info", info_log)

    request = Mock()
    request.structured_output_request.structured_output_key = (
        StructuredOutputOptions.JSON,
        '{"type":"object"}',
    )

    manager._create_grammar(request)
    info_log.assert_not_called()

    manager._create_grammar(request)
    info_log.assert_called_once()


def test_structured_output_manager_make_cache_stats():
    vllm_config = Mock()
    vllm_config.parallel_config.distributed_executor_backend = None
    vllm_config.scheduler_config.max_num_seqs = 8
    vllm_config.model_config.skip_tokenizer_init = True
    vllm_config.structured_outputs_config.enable_in_reasoning = False

    manager = StructuredOutputManager(vllm_config)
    backend = Mock()
    backend.compiled_grammar_cache_stats.return_value = CacheInfo(hits=2, total=5)
    manager.backend = backend

    assert manager.make_cache_stats() == StructuredOutputCacheStats(
        requests=5,
        queries=5,
        hits=2,
    )
    backend.compiled_grammar_cache_stats.assert_called_once_with(delta=True)
