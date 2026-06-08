# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for F3: fused RoPE + MLA KV-cache write dispatch in AiterMLAImpl.

F3 auto-enables when rocm_aiter_ops.has_fused_rope_mla_kv_cache() returns True
(i.e. aiter.fused_qk_rope_concat_and_cache_mla is importable). No env var is
required — follows the same pattern as has_fused_rmsnorm_mxfp4_quant() for F2.
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
# Tests: has_fused_rope_mla_kv_cache() probe
# ---------------------------------------------------------------------------


class TestHasFusedRopeMlaKvCache:
    """has_fused_rope_mla_kv_cache() must return bool without raising."""

    def test_probe_returns_bool(self):
        """Probe must always return bool, never raise."""
        from vllm._aiter_ops import rocm_aiter_ops

        result = rocm_aiter_ops.has_fused_rope_mla_kv_cache()
        assert isinstance(result, bool), (
            f"Expected bool, got {type(result).__name__}"
        )

    def test_probe_false_when_kernel_absent(self, monkeypatch):
        """When the aiter import is mocked to fail, probe must return False."""
        from vllm._aiter_ops import rocm_aiter_ops

        monkeypatch.setattr(
            rocm_aiter_ops,
            "has_fused_rope_mla_kv_cache",
            classmethod(lambda cls: False),
        )
        assert rocm_aiter_ops.has_fused_rope_mla_kv_cache() is False

    def test_f3_disabled_when_mla_disabled(self, monkeypatch):
        """F3 must not fire when is_mla_enabled() returns None/False."""
        from vllm._aiter_ops import rocm_aiter_ops

        monkeypatch.setattr(
            rocm_aiter_ops,
            "is_mla_enabled",
            classmethod(lambda cls: False),
        )
        f3_enabled = bool(
            rocm_aiter_ops.is_mla_enabled()
            and rocm_aiter_ops.has_fused_rope_mla_kv_cache()
        )
        assert not f3_enabled


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
