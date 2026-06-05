# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for F3: fused RoPE + MLA KV-cache write dispatch in AiterMLAImpl.

PR3 will add two methods to AiterMLAImpl (and AiterTritonMLAImpl):
  - fused_rope_kvcache_supported() -> bool
      Returns True when VLLM_ROCM_USE_AITER_TRITON_ROPE=1 AND
      VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=1.
  - do_rope_and_kv_cache_update(layer, query, key, value, positions,
                                 cos_sin_cache, is_neox, kv_cache,
                                 layer_slot_mapping)
      Calls ops.concat_and_cache_mla_rope_fused() instead of the unfused
      ops.concat_and_cache_mla() + separate rope path.

These tests are ROCm-only and are skipped when the PR3 methods are not yet
implemented in AiterMLAImpl (i.e. when running against this PR only).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# DeepSeek-V3/R1 MLA dimensions
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_TOKENS = 4
NUM_Q_HEADS = 128


def _make_mock_impl(kv_cache_dtype: str = "auto") -> MagicMock:
    """Return a MagicMock that mimics AiterMLAImpl attributes needed by F3."""
    impl = MagicMock()
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.kv_cache_dtype = kv_cache_dtype
    return impl


def _make_tensors(device: str = "cpu"):
    """Build minimal tensors for do_rope_and_kv_cache_update."""
    query = torch.randn(NUM_TOKENS, NUM_Q_HEADS, QK_ROPE_HEAD_DIM)
    # MLA key: [seq_len, 1, qk_rope_head_dim + kv_lora_rank]
    key = torch.randn(NUM_TOKENS, 1, QK_ROPE_HEAD_DIM + KV_LORA_RANK)
    value = torch.empty(0)  # unused in MLA path
    positions = torch.randint(0, 8192, (NUM_TOKENS,))
    cos_sin_cache = torch.randn(8192, 2 * QK_ROPE_HEAD_DIM)
    slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.long)
    # kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    kv_cache = torch.zeros(16, 16, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    return query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache


def _make_mock_layer(k_scale_value: float = 1.0) -> MagicMock:
    layer = MagicMock()
    layer._k_scale = torch.tensor([k_scale_value])
    return layer


# ---------------------------------------------------------------------------
# Tests: fused_rope_kvcache_supported()
# ---------------------------------------------------------------------------


class TestFusedRopeKVCacheSupported:
    """fused_rope_kvcache_supported() must respect both env-var gates."""

    @pytest.fixture(autouse=True)
    def _import_impl(self):
        """Import here so the test is skipped if the module is absent."""
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
            AiterMLAImpl,  # noqa: F401
        )

        self.ImplClass = AiterMLAImpl
        if not hasattr(AiterMLAImpl, "fused_rope_kvcache_supported"):
            pytest.skip("fused_rope_kvcache_supported not implemented (requires PR3)")

    def _call_supported(self, impl_instance) -> bool:
        return impl_instance.fused_rope_kvcache_supported()

    def test_returns_true_when_both_env_vars_set(self, monkeypatch):
        """Feature is enabled only when both gate vars are 1."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE", "1")
        impl = MagicMock(spec=self.ImplClass)
        # Call the real method via unbound call on the class
        result = self.ImplClass.fused_rope_kvcache_supported(impl)
        assert result is True

    def test_returns_false_when_f3_var_unset(self, monkeypatch):
        """F3 disabled when VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=0."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE", "0")
        impl = MagicMock(spec=self.ImplClass)
        result = self.ImplClass.fused_rope_kvcache_supported(impl)
        assert result is False

    def test_returns_false_when_rope_var_unset(self, monkeypatch):
        """F3 disabled when base aiter-rope gate is off."""
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE", "1")
        impl = MagicMock(spec=self.ImplClass)
        result = self.ImplClass.fused_rope_kvcache_supported(impl)
        assert result is False

    def test_returns_false_when_both_unset(self, monkeypatch):
        """F3 disabled when neither gate is set."""
        monkeypatch.delenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", raising=False)
        monkeypatch.delenv(
            "VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE", raising=False
        )
        impl = MagicMock(spec=self.ImplClass)
        result = self.ImplClass.fused_rope_kvcache_supported(impl)
        assert result is False

    def test_aiter_triton_impl_inherits_support(self, monkeypatch):
        """AiterTritonMLAImpl must also expose fused_rope_kvcache_supported."""
        from vllm.v1.attention.backends.mla.aiter_triton_mla import AiterTritonMLAImpl

        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE", "1")
        impl = MagicMock(spec=AiterTritonMLAImpl)
        result = AiterTritonMLAImpl.fused_rope_kvcache_supported(impl)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: do_rope_and_kv_cache_update() dispatch
# ---------------------------------------------------------------------------


class TestDoRopeAndKVCacheUpdate:
    """do_rope_and_kv_cache_update() must call concat_and_cache_mla_rope_fused."""

    @pytest.fixture(autouse=True)
    def _import_impl(self):
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

        self.ImplClass = AiterMLAImpl
        if not hasattr(AiterMLAImpl, "do_rope_and_kv_cache_update"):
            pytest.skip("do_rope_and_kv_cache_update not implemented (requires PR3)")

    def _run_update(self, impl_instance, layer, tensors):
        query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = tensors
        self.ImplClass.do_rope_and_kv_cache_update(
            impl_instance,
            layer,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox=True,
            kv_cache=kv_cache,
            layer_slot_mapping=slot_mapping,
        )

    def test_fused_op_is_called(self):
        """concat_and_cache_mla_rope_fused must be invoked once."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        tensors = _make_tensors()

        with patch("vllm._custom_ops.concat_and_cache_mla_rope_fused") as mock_fused:
            self._run_update(impl, layer, tensors)
            assert mock_fused.call_count == 1

    def test_unfused_op_is_not_called(self):
        """concat_and_cache_mla must NOT be called on the fused path."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        tensors = _make_tensors()

        with (
            patch("vllm._custom_ops.concat_and_cache_mla") as mock_unfused,
            patch("vllm._custom_ops.concat_and_cache_mla_rope_fused"),
        ):
            self._run_update(impl, layer, tensors)
            mock_unfused.assert_not_called()

    def test_positions_passed_correctly(self):
        """positions tensor must be forwarded to the fused op."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = (
            _make_tensors()
        )

        with patch("vllm._custom_ops.concat_and_cache_mla_rope_fused") as mock_fused:
            self.ImplClass.do_rope_and_kv_cache_update(
                impl,
                layer,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox=True,
                kv_cache=kv_cache,
                layer_slot_mapping=slot_mapping,
            )
            call_args = mock_fused.call_args
            # positions is the first positional arg
            passed_positions = (
                call_args.args[0]
                if call_args.args
                else call_args.kwargs.get("positions")
            )
            assert passed_positions is positions

    def test_kv_cache_passed_correctly(self):
        """kv_cache tensor must be forwarded to the fused op."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = (
            _make_tensors()
        )

        with patch("vllm._custom_ops.concat_and_cache_mla_rope_fused") as mock_fused:
            self.ImplClass.do_rope_and_kv_cache_update(
                impl,
                layer,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox=True,
                kv_cache=kv_cache,
                layer_slot_mapping=slot_mapping,
            )
            call_args = mock_fused.call_args
            all_args = list(call_args.args) + list(call_args.kwargs.values())
            assert any(arg is kv_cache for arg in all_args), (
                "kv_cache tensor was not passed to concat_and_cache_mla_rope_fused"
            )

    def test_k_scale_from_layer_used(self):
        """The k_scale must come from layer._k_scale."""
        impl = _make_mock_impl()
        expected_scale = torch.tensor([0.5])
        layer = _make_mock_layer(k_scale_value=0.5)
        layer._k_scale = expected_scale
        query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = (
            _make_tensors()
        )

        with patch("vllm._custom_ops.concat_and_cache_mla_rope_fused") as mock_fused:
            self.ImplClass.do_rope_and_kv_cache_update(
                impl,
                layer,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox=True,
                kv_cache=kv_cache,
                layer_slot_mapping=slot_mapping,
            )
            call_args = mock_fused.call_args
            all_args = list(call_args.args) + list(call_args.kwargs.values())
            assert any(
                isinstance(a, torch.Tensor) and torch.equal(a, expected_scale)
                for a in all_args
            ), "layer._k_scale was not passed to concat_and_cache_mla_rope_fused"

    def test_kv_cache_dtype_forwarded(self):
        """kv_cache_dtype string must be forwarded to the fused op."""
        for dtype in ("auto", "fp8"):
            impl = _make_mock_impl(kv_cache_dtype=dtype)
            layer = _make_mock_layer()
            tensors = _make_tensors()

            with patch(
                "vllm._custom_ops.concat_and_cache_mla_rope_fused"
            ) as mock_fused:
                self._run_update(impl, layer, tensors)
                call_args = mock_fused.call_args
                all_args = list(call_args.args) + list(call_args.kwargs.values())
                assert dtype in all_args, (
                    f"kv_cache_dtype='{dtype}' was not forwarded to the fused op"
                )

    def test_key_split_into_k_pe_and_kv_c(self):
        """k_pe and kv_c must be sliced from key using qk_rope_head_dim."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = (
            _make_tensors()
        )

        # key shape: [NUM_TOKENS, 1, QK_ROPE_HEAD_DIM + KV_LORA_RANK]
        # expected k_pe = key[..., :QK_ROPE_HEAD_DIM],
        # kv_c = key[..., QK_ROPE_HEAD_DIM:]
        expected_k_pe = key[..., :QK_ROPE_HEAD_DIM]
        expected_kv_c = key[..., QK_ROPE_HEAD_DIM:]

        captured: dict[str, Any] = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        with patch(
            "vllm._custom_ops.concat_and_cache_mla_rope_fused", side_effect=capture
        ):
            self.ImplClass.do_rope_and_kv_cache_update(
                impl,
                layer,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox=True,
                kv_cache=kv_cache,
                layer_slot_mapping=slot_mapping,
            )

        all_args = list(captured.get("args", [])) + list(
            captured.get("kwargs", {}).values()
        )
        k_pe_found = any(
            isinstance(a, torch.Tensor) and a.shape == expected_k_pe.squeeze(1).shape
            for a in all_args
        )
        kv_c_found = any(
            isinstance(a, torch.Tensor) and a.shape == expected_kv_c.squeeze(1).shape
            for a in all_args
        )
        assert k_pe_found, "k_pe (shape {}) not found in fused op args".format(
            expected_k_pe.squeeze(1).shape
        )
        assert kv_c_found, "kv_c (shape {}) not found in fused op args".format(
            expected_kv_c.squeeze(1).shape
        )

    @pytest.mark.parametrize("is_neox", [True, False])
    def test_is_neox_forwarded(self, is_neox: bool):
        """is_neox bool must be passed through to the fused op unchanged."""
        impl = _make_mock_impl()
        layer = _make_mock_layer()
        tensors = _make_tensors()

        with patch("vllm._custom_ops.concat_and_cache_mla_rope_fused") as mock_fused:
            query, key, value, positions, cos_sin_cache, slot_mapping, kv_cache = (
                tensors
            )
            self.ImplClass.do_rope_and_kv_cache_update(
                impl,
                layer,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox=is_neox,
                kv_cache=kv_cache,
                layer_slot_mapping=slot_mapping,
            )
            call_args = mock_fused.call_args
            all_args = list(call_args.args) + list(call_args.kwargs.values())
            assert is_neox in all_args, (
                f"is_neox={is_neox} was not forwarded to "
                "concat_and_cache_mla_rope_fused"
            )
