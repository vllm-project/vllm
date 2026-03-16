# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Cohere embed protocol: build_typed_embeddings and its
underlying packing helpers, plus Cohere-specific serving helpers."""

import base64
import struct

import numpy as np
import pytest

from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    build_typed_embeddings,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    return [
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        [-0.05, 0.15, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75],
    ]


class TestBuildTypedEmbeddingsFloat:
    def test_float_passthrough(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(sample_embeddings, ["float"])
        assert result.float == sample_embeddings
        assert result.binary is None

    def test_empty_input(self):
        result = build_typed_embeddings([], ["float"])
        assert result.float == []


class TestBuildTypedEmbeddingsBinary:
    def test_binary_packing(self):
        # 8 values: positive->1, negative->0 => bits: 10101010 = 0xAA = 170
        # signed: 170 - 128 = 42
        embs = [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]
        result = build_typed_embeddings(embs, ["binary"])
        assert result.binary is not None
        assert result.binary[0] == [42]

    def test_ubinary_packing(self):
        embs = [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]
        result = build_typed_embeddings(embs, ["ubinary"])
        assert result.ubinary is not None
        assert result.ubinary[0] == [170]  # 0b10101010

    def test_binary_all_positive(self):
        embs = [[0.1] * 8]
        result = build_typed_embeddings(embs, ["binary"])
        assert result.binary is not None
        # all bits = 1 => 0xFF = 255, signed: 255 - 128 = 127
        assert result.binary[0] == [127]

    def test_binary_all_negative(self):
        embs = [[-0.1] * 8]
        result = build_typed_embeddings(embs, ["binary"])
        assert result.binary is not None
        # all bits = 0, signed: 0 - 128 = -128
        assert result.binary[0] == [-128]

    def test_binary_dimension_is_eighth(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(sample_embeddings, ["binary"])
        assert result.binary is not None
        for orig, packed in zip(sample_embeddings, result.binary):
            assert len(packed) == len(orig) // 8

    def test_zero_treated_as_positive(self):
        embs = [[0.0] * 8]
        result = build_typed_embeddings(embs, ["binary"])
        assert result.binary is not None
        # 0.0 >= 0 is True, so bit=1 for all => 127 (signed)
        assert result.binary[0] == [127]

    def test_non_multiple_of_8_raises(self):
        embs = [[0.1] * 7]
        with pytest.raises(ValueError, match="multiple of 8"):
            build_typed_embeddings(embs, ["binary"])

    def test_ubinary_non_multiple_of_8_raises(self):
        embs = [[0.1] * 10]
        with pytest.raises(ValueError, match="multiple of 8"):
            build_typed_embeddings(embs, ["ubinary"])


class TestBuildTypedEmbeddingsBase64:
    def test_base64_roundtrip(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(sample_embeddings, ["base64"])
        assert result.base64 is not None
        assert len(result.base64) == 2

        for orig, b64_str in zip(sample_embeddings, result.base64):
            decoded = base64.b64decode(b64_str)
            n = len(orig)
            values = struct.unpack(f"<{n}f", decoded)
            np.testing.assert_allclose(orig, values, rtol=1e-5)

    def test_base64_byte_length(self):
        embs = [[0.1, 0.2, 0.3]]
        result = build_typed_embeddings(embs, ["base64"])
        assert result.base64 is not None
        raw = base64.b64decode(result.base64[0])
        assert len(raw) == 3 * 4  # 3 floats * 4 bytes each


class TestBuildTypedEmbeddingsMultiple:
    def test_all_types_at_once(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(
            sample_embeddings,
            ["float", "binary", "ubinary", "base64"],
        )
        assert result.float is not None
        assert result.binary is not None
        assert result.ubinary is not None
        assert result.base64 is not None

    def test_subset_types(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(sample_embeddings, ["float", "binary"])
        assert result.float is not None
        assert result.binary is not None
        assert result.ubinary is None
        assert result.base64 is None

    def test_unknown_type_ignored(self, sample_embeddings: list[list[float]]):
        result = build_typed_embeddings(sample_embeddings, ["float", "unknown_type"])
        assert result.float is not None


class TestResolveTruncation:
    """Unit tests for ServingEmbedding._resolve_cohere_truncation."""

    @staticmethod
    def _make_request(**kwargs) -> CohereEmbedRequest:
        defaults = {
            "model": "test",
            "input_type": "search_document",
            "texts": ["hello"],
        }
        return CohereEmbedRequest(**(defaults | kwargs))

    def test_truncate_end_default(self):
        req = self._make_request()
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side is None

    def test_truncate_end_explicit(self):
        req = self._make_request(truncate="END")
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side is None

    def test_truncate_end_with_max_tokens(self):
        req = self._make_request(truncate="END", max_tokens=128)
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens == 128
        assert side is None

    def test_truncate_none(self):
        req = self._make_request(truncate="NONE")
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens is None
        assert side is None

    def test_truncate_none_with_max_tokens(self):
        """truncate=NONE should NOT set truncate_prompt_tokens; the
        max_tokens limit is enforced separately via _check_max_tokens."""
        req = self._make_request(truncate="NONE", max_tokens=10)
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens is None
        assert side is None

    def test_truncate_start(self):
        req = self._make_request(truncate="START")
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side == "left"

    def test_truncate_start_with_max_tokens(self):
        req = self._make_request(truncate="START", max_tokens=64)
        tokens, side = ServingEmbedding._resolve_cohere_truncation(req)
        assert tokens == 64
        assert side == "left"


class TestApplyStPrompt:
    """Unit tests for ServingEmbedding._apply_task_instruction."""

    @staticmethod
    def _make_handler(task_instructions: dict[str, str] | None):
        handler = object.__new__(ServingEmbedding)
        io = object.__new__(EmbedIOProcessor)
        io.task_instructions = task_instructions
        handler.io_processor = io
        return handler

    def test_no_prompts_configured(self):
        handler = self._make_handler(None)
        texts = ["hello", "world"]
        assert handler._apply_task_instruction(texts, "query") is texts

    def test_matching_input_type(self):
        handler = self._make_handler({"query": "search_query: "})
        result = handler._apply_task_instruction(["hello"], "query")
        assert result == ["search_query: hello"]

    def test_non_matching_input_type(self):
        handler = self._make_handler({"query": "search_query: "})
        texts = ["hello"]
        assert handler._apply_task_instruction(texts, "document") is texts

    def test_multiple_texts(self):
        handler = self._make_handler(
            {"query": "Represent this sentence for searching: "}
        )
        result = handler._apply_task_instruction(["a", "b", "c"], "query")
        assert result == [
            "Represent this sentence for searching: a",
            "Represent this sentence for searching: b",
            "Represent this sentence for searching: c",
        ]

    def test_empty_prefix_returns_unchanged(self):
        handler = self._make_handler({"passage": ""})
        texts = ["hello"]
        assert handler._apply_task_instruction(texts, "passage") is texts


class TestLoadTaskInstructions:
    """Unit tests for EmbedIOProcessor._load_task_instructions."""

    def test_no_attribute(self):
        class FakeConfig:
            pass

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None

    def test_with_task_instructions(self):
        class FakeConfig:
            task_instructions = {
                "retrieval.query": "Represent the query: ",
                "retrieval.passage": "",
            }

        result = EmbedIOProcessor._load_task_instructions(FakeConfig())
        assert result == {
            "retrieval.query": "Represent the query: ",
            "retrieval.passage": "",
        }

    def test_empty_dict(self):
        class FakeConfig:
            task_instructions = {}

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None

    def test_non_dict(self):
        class FakeConfig:
            task_instructions = "not a dict"

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None


class TestCheckMaxTokens:
    """Unit tests for ServingEmbedding._check_cohere_max_tokens."""

    @staticmethod
    def _fake_output(n_tokens: int):
        class _Out:
            def __init__(self, n: int):
                self.prompt_token_ids = list(range(n))

        return _Out(n_tokens)

    def test_none_check_is_noop(self):
        outs = [self._fake_output(100)]
        ServingEmbedding._check_cohere_max_tokens(outs, None)

    def test_within_limit(self):
        outs = [self._fake_output(5), self._fake_output(3)]
        ServingEmbedding._check_cohere_max_tokens(outs, 5)

    def test_exceeds_limit(self):
        outs = [self._fake_output(3), self._fake_output(10)]
        with pytest.raises(ValueError, match="exceeds max_tokens=5"):
            ServingEmbedding._check_cohere_max_tokens(outs, 5)

    def test_exact_limit(self):
        outs = [self._fake_output(5)]
        ServingEmbedding._check_cohere_max_tokens(outs, 5)


class TestValidateInputType:
    """Unit tests for ServingEmbedding._validate_input_type."""

    @staticmethod
    def _make_handler(task_instructions: dict[str, str] | None):
        handler = object.__new__(ServingEmbedding)
        io = object.__new__(EmbedIOProcessor)
        io.task_instructions = task_instructions
        handler.io_processor = io
        return handler

    def test_none_input_type_always_accepted(self):
        handler = self._make_handler(None)
        handler._validate_input_type(None)
        handler_with = self._make_handler({"query": "q: "})
        handler_with._validate_input_type(None)

    def test_no_prompts_rejects(self):
        handler = self._make_handler(None)
        with pytest.raises(ValueError, match="does not define any input_type"):
            handler._validate_input_type("anything")

    def test_known_type_accepted(self):
        handler = self._make_handler({"query": "q: ", "document": "d: "})
        handler._validate_input_type("query")
        handler._validate_input_type("document")

    def test_unknown_type_rejected(self):
        handler = self._make_handler({"query": "q: ", "document": "d: "})
        with pytest.raises(ValueError, match="Unsupported input_type 'other'"):
            handler._validate_input_type("other")

    def test_error_lists_supported(self):
        handler = self._make_handler({"a": "", "b": ""})
        with pytest.raises(ValueError, match="Supported values: a, b"):
            handler._validate_input_type("z")
