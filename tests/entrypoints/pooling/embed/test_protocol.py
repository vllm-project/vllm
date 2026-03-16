# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Cohere embed protocol: build_typed_embeddings and its
underlying packing helpers, plus Cohere-specific serving helpers."""

import base64
import struct

import numpy as np
import pytest

from vllm.entrypoints.pooling.embed.protocol import (
    build_typed_embeddings,
)


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
