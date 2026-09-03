# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer sparse-MLA warmup capture-bucket selection."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    _sparse_mla_refine_tokens,
)


@pytest.mark.parametrize(
    ("capture_sizes", "expected"),
    [
        # Only buckets the decode kernel can serve (<= 64) qualify; the list
        # comes back sorted and deduplicated.
        ([128, 64, 1, 32, 32, 0, 65, -4], (1, 32, 64)),
        ([], ()),
        (None, ()),
    ],
)
def test_sparse_mla_refine_tokens(capture_sizes, expected):
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=capture_sizes)
        )
    )
    assert _sparse_mla_refine_tokens(worker) == expected
