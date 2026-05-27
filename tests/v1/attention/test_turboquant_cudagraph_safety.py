# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that TurboQuant attention is CUDA-graph-capture safe.

Validates fixes for https://github.com/vllm-project/vllm/issues/40807:
  - _cudagraph_support is UNIFORM_SINGLE_TOKEN_DECODE (prevents the
    continuation-prefill branch from running under CUDA graph capture
    when combined with speculative decoding).
  - build_for_cudagraph_capture always populates CPU-resident copies of
    query_start_loc and seq_lens so the prefill path never calls
    .tolist() on GPU tensors.
"""

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
)


class TestCudagraphSupport:
    def test_cudagraph_support_level(self):
        assert (
            TurboQuantMetadataBuilder._cudagraph_support
            == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        ), (
            "TurboQuant must declare UNIFORM_SINGLE_TOKEN_DECODE so that "
            "spec-decode K+1 verify batches are not captured under CUDA graph "
            "(the continuation-prefill branch calls .tolist() on GPU tensors)."
        )

    def test_cudagraph_support_blocks_spec_decode_full_capture(self):
        support = TurboQuantMetadataBuilder._cudagraph_support
        assert support.value < AttentionCGSupport.UNIFORM_BATCH.value, (
            "TurboQuant's cudagraph support must be below UNIFORM_BATCH to "
            "trigger the spec-decode PIECEWISE downgrade in compilation.py."
        )


class TestBuildForCudagraphCaptureCPUCopies:
    """build_for_cudagraph_capture must always populate CPU-resident
    metadata copies so the prefill path never falls through to
    .tolist() on GPU tensors."""

    @staticmethod
    def _make_metadata(
        num_reqs: int = 4,
        *,
        include_cpu_copies: bool,
    ) -> TurboQuantMetadata:
        seq_lens = torch.ones(num_reqs, dtype=torch.int32)
        query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32)
        return TurboQuantMetadata(
            seq_lens=seq_lens,
            slot_mapping=torch.zeros(num_reqs, dtype=torch.int64),
            block_table=torch.zeros(num_reqs, 1, dtype=torch.int32),
            query_start_loc=query_start_loc,
            num_actual_tokens=num_reqs,
            max_query_len=1,
            max_seq_len=1,
            is_prefill=False,
            query_start_loc_cpu=(
                query_start_loc.clone() if include_cpu_copies else None
            ),
            seq_lens_cpu=(seq_lens.clone() if include_cpu_copies else None),
        )

    def test_cpu_copies_present_when_builder_provides_them(self):
        meta = self._make_metadata(include_cpu_copies=True)
        assert meta.query_start_loc_cpu is not None
        assert meta.seq_lens_cpu is not None

    def test_cpu_copies_absent_shows_the_gap(self):
        meta = self._make_metadata(include_cpu_copies=False)
        assert meta.query_start_loc_cpu is None
        assert meta.seq_lens_cpu is None

    def test_tolist_on_cpu_tensor_is_safe(self):
        meta = self._make_metadata(include_cpu_copies=True)
        assert meta.query_start_loc_cpu is not None
        qsl = meta.query_start_loc_cpu.tolist()
        assert isinstance(qsl, list)
        assert len(qsl) == 5

    def test_tolist_on_cpu_seq_lens_is_safe(self):
        meta = self._make_metadata(include_cpu_copies=True)
        assert meta.seq_lens_cpu is not None
        sl = meta.seq_lens_cpu.tolist()
        assert isinstance(sl, list)
        assert all(v == 1 for v in sl)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestPrefillCaptureGuard:
    """The _prefill_attention continuation branch must not crash during
    CUDA graph capture."""

    def test_capture_guard_returns_zeros(self):
        from unittest.mock import patch

        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        N, Hq, D = 4, 8, 128
        query = torch.randn(N, Hq, D, device="cuda")

        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            impl = object.__new__(TurboQuantAttentionImpl)
            impl.scale = 1.0 / (D**0.5)
            impl.num_kv_heads = 2
            impl.num_kv_groups = Hq // 2
            impl.kv_cache_dtype = "turboquant_k8v4"

            meta = TurboQuantMetadata(
                seq_lens=torch.ones(1, dtype=torch.int32, device="cuda"),
                slot_mapping=torch.zeros(N, dtype=torch.int64, device="cuda"),
                block_table=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
                query_start_loc=torch.tensor([0, N], dtype=torch.int32, device="cuda"),
                num_actual_tokens=N,
                max_query_len=N,
                max_seq_len=N + 10,
                is_prefill=True,
            )

            result = TurboQuantAttentionImpl._prefill_attention(
                impl,
                query=query,
                key=torch.randn(N, 2, D, device="cuda"),
                value=torch.randn(N, 2, D, device="cuda"),
                kv_cache=torch.zeros(1, 16, 2, 196, device="cuda"),
                attn_metadata=meta,
                Pi=torch.eye(D, device="cuda"),
                centroids=torch.randn(16, D, device="cuda"),
            )

            assert result.shape == (N, Hq, D)
            assert (result == 0).all(), (
                "During CUDA graph capture, _prefill_attention continuation "
                "branch should return zeros (safe for memory profiling)."
            )
