# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 templates the decode schedule metadata on the batch rounded up to 32."""

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder


def _builder(*, max_num_seqs, mnbt, num_spec, flattening, varlen):
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=mnbt
        )
    )
    builder.num_speculative_tokens = num_spec
    builder.use_flattening = flattening
    builder.supports_varlen = varlen
    return builder


def _buckets(shapes, cols):
    return {(rows + 31) // 32 * 32 for rows, c in shapes if c == cols}


@pytest.mark.parametrize("num_spec", [0, 3, 5])
@pytest.mark.parametrize("max_num_seqs, mnbt", [(1024, 8192), (100, 16384), (256, 512)])
def test_flattened_decode_covers_every_bucket(num_spec, max_num_seqs, mnbt):
    builder = _builder(
        max_num_seqs=max_num_seqs,
        mnbt=mnbt,
        num_spec=num_spec,
        flattening=True,
        varlen=False,
    )
    shapes = builder.paged_mqa_logits_metadata_shapes()
    max_tokens = min(max_num_seqs * (1 + num_spec), mnbt)
    # Every runtime decode token count maps to a warmed bucket.
    assert {(t + 31) // 32 * 32 for t in range(1, max_tokens + 1)} == _buckets(
        shapes, 1
    )
    assert max(rows for rows, _ in shapes) == max_tokens


def test_native_decode_adds_per_next_n_shapes(monkeypatch):
    monkeypatch.setattr(indexer, "_supports_native_decode", lambda n: n in (1, 2, 4))
    builder = _builder(
        max_num_seqs=100, mnbt=8192, num_spec=3, flattening=False, varlen=False
    )
    shapes = builder.paged_mqa_logits_metadata_shapes()
    assert {c for _, c in shapes} == {1, 2, 4}
    for cols in (2, 4):
        assert _buckets(shapes, cols) == {32, 64, 96, 128}
        assert max(r for r, c in shapes if c == cols) == 100
