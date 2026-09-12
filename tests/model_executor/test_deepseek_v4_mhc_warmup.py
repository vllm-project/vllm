# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-pure tests for DeepSeek V4 MHC warmup selection and gating logic.

The TileLang JIT kernels require CUDA; these tests verify CPU-side selection,
layer-finding, and gating without a GPU.  ``_select_mhc_split_key_token_sizes``
tests monkeypatch ``compute_num_split`` (CUDA-backed) with a deterministic fake
so the real function runs under CPU-pure coverage.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _find_deepseek_v4_model,
    _find_first_mhc_layer,
    _normalize_token_sizes,
    _select_mhc_split_key_token_sizes,
    _select_mhc_warmup_token_sizes,
)
from vllm.utils.math_utils import cdiv

# ── Helpers ───────────────────────────────────────────────────────────────


def _fake_compute_num_split(n_sms: int):
    """Deterministic ``compute_num_split`` with explicit SM count instead of
    ``torch.cuda``."""

    def _inner(block_k: int, k: int | None, grid_size: int) -> int:
        split_k = n_sms // grid_size
        if k is not None:
            split_k = min(split_k, cdiv(k, block_k) // 4)
        return max(split_k, 1)

    return _inner


def _patch_compute_split(*, n_sms: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``tilelang_kernels`` in ``sys.modules`` via ``monkeypatch.setitem``
    so the local import inside ``_select_mhc_split_key_token_sizes`` resolves
    without triggering the real module (which requires TileLang + CUDA).  The
    stub is automatically restored after each test."""

    MODULE_PATH = "vllm.model_executor.kernels.mhc.tilelang_kernels"

    # Ensure parent packages exist in sys.modules so the dotted-path import
    # resolves through parent lookups.
    for parent_path in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.mhc",
    ):
        if parent_path not in sys.modules:
            monkeypatch.setitem(sys.modules, parent_path, types.ModuleType(parent_path))

    stub = types.ModuleType(MODULE_PATH)
    stub.compute_num_split = _fake_compute_num_split(n_sms)
    monkeypatch.setitem(sys.modules, MODULE_PATH, stub)


# ── _normalize_token_sizes ───────────────────────────────────────────────


class TestNormalizeTokenSizes:
    def test_empty_when_no_sizes(self) -> None:
        assert _normalize_token_sizes((), max_tokens=100) == []

    def test_removes_out_of_range(self) -> None:
        assert _normalize_token_sizes([0, 1, 50, 100, 200], max_tokens=100) == [
            1,
            50,
            100,
        ]

    def test_deduplicates_and_sorts(self) -> None:
        assert _normalize_token_sizes([4, 1, 4, 8, 2], max_tokens=100) == [1, 2, 4, 8]

    def test_accepts_iterator(self) -> None:
        assert _normalize_token_sizes(iter({1, 2, 3}), max_tokens=10) == [1, 2, 3]


# ── _select_mhc_warmup_token_sizes ────────────────────────────────────────


class TestSelectMhcWarmupTokenSizes:
    def test_empty_on_zero_max_tokens(self) -> None:
        assert (
            _select_mhc_warmup_token_sizes(max_tokens=0, cudagraph_capture_sizes=[])
            == []
        )

    def test_contains_1(self) -> None:
        assert 1 in _select_mhc_warmup_token_sizes(
            max_tokens=10, cudagraph_capture_sizes=[]
        )

    def test_bounded_by_max_tokens(self) -> None:
        sizes = _select_mhc_warmup_token_sizes(max_tokens=5, cudagraph_capture_sizes=[])
        assert all(1 <= s <= 5 for s in sizes)

    def test_includes_cudagraph_capture_sizes(self) -> None:
        sizes = _select_mhc_warmup_token_sizes(
            max_tokens=100, cudagraph_capture_sizes=[7, 33]
        )
        assert 7 in sizes and 33 in sizes

    def test_includes_max_auto_tokens(self) -> None:
        sizes = _select_mhc_warmup_token_sizes(
            max_tokens=100, cudagraph_capture_sizes=[]
        )
        assert 100 in sizes

    def test_respects_auto_warmup_cap(self) -> None:
        sizes = _select_mhc_warmup_token_sizes(
            max_tokens=20000, cudagraph_capture_sizes=[]
        )
        assert max(sizes) == 16384

    def test_does_not_include_zero(self) -> None:
        assert _select_mhc_warmup_token_sizes(
            max_tokens=1, cudagraph_capture_sizes=[]
        ) == [1]


# ── _select_mhc_split_key_token_sizes ─────────────────────────────────────


class TestSelectMhcSplitKeyTokenSizes:
    """Real ``_select_mhc_split_key_token_sizes`` invoked under a deterministic
    ``compute_num_split`` monkeypatch.  No CUDA required outside the skip-guard
    cross-check below."""

    # ── Broadcast-variant key counts (k_size = hidden_size) ──────────────

    @pytest.mark.parametrize(
        "max_tokens,k_size,n_sms,expected_keys",
        [
            (8192, 4096, 188, 16),  # RTX PRO 6000 Blackwell, broadcast K
            (8192, 4096, 132, 15),  # H100 SXM
            (8192, 4096, 80, 12),
            (8192, 2048, 188, 8),
            (8192, 1024, 188, 4),
            (1024, 4096, 188, 6),
            (1, 4096, 188, 1),
        ],
    )
    def test_broadcast_key_count(
        self, monkeypatch, max_tokens, k_size, n_sms, expected_keys
    ) -> None:
        _patch_compute_split(n_sms=n_sms, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=max_tokens, k_size=k_size)
        assert len(reps) == expected_keys, (
            f"Expected {expected_keys} keys for max_tokens={max_tokens}, "
            f"k_size={k_size}, n_sms={n_sms}, got {len(reps)}"
        )

    # ── Non-broadcast-variant key counts (k_size = hc_mult * hidden_size) ──

    @pytest.mark.parametrize(
        "max_tokens,k_size,n_sms,expected_keys",
        [
            (8192, 65536, 188, 26),  # DSv4 default: hc_mult=16, hidden=4096
            (8192, 65536, 132, 22),  # H100 SXM
            (8192, 65536, 80, 16),
            (8192, 32768, 188, 26),  # hc_mult=8, hidden=4096; same as 188 SM full
            (8192, 8192, 188, 22),
            (1024, 65536, 188, 16),
            (1, 65536, 188, 1),
        ],
    )
    def test_non_broadcast_key_count(
        self, monkeypatch, max_tokens, k_size, n_sms, expected_keys
    ) -> None:
        _patch_compute_split(n_sms=n_sms, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=max_tokens, k_size=k_size)
        assert len(reps) == expected_keys, (
            f"Expected {expected_keys} keys for max_tokens={max_tokens}, "
            f"k_size={k_size}, n_sms={n_sms}, got {len(reps)}"
        )

    # ── Semantic invariants ─────────────────────────────────────────────

    def test_all_keys_distinct(self, monkeypatch) -> None:
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=4096)
        fake_cns = _fake_compute_num_split(188)
        ns_values = [fake_cns(64, 4096, cdiv(t, 64)) for t in reps]
        assert len(set(ns_values)) == len(ns_values), (
            f"Duplicate n_splits: {dict(zip(reps, ns_values))}"
        )

    def test_last_key_n_splits_is_one(self, monkeypatch) -> None:
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=4096)
        last_ns = _fake_compute_num_split(188)(64, 4096, cdiv(reps[-1], 64))
        assert last_ns == 1, f"Last key n_splits={last_ns}, expected 1"

    # ── Exact broadcast sequence (k_size=hidden_size=4096, 188 SMs) ──────

    def test_exact_broadcast_sequence_188_sms(self, monkeypatch) -> None:
        # fmt: off
        expected = [1, 705, 769, 833, 897, 961, 1089, 1153,
                    1281, 1473, 1665, 1985, 2369, 3009, 3969, 6017]
        # fmt: on
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=4096)
        assert reps == expected, (
            f"188-SM broadcast (K=4096) sequence mismatch\n"
            f"  Expected ({len(expected)}): {expected}\n"
            f"  Got      ({len(reps)}): {reps}"
        )

    # ── Exact non-broadcast sequence (k_size=65536, 188 SMs) ─────────────

    def test_exact_non_broadcast_sequence_188_sms(self, monkeypatch) -> None:
        # fmt: off
        expected = [1, 65, 129, 193, 257, 321, 385, 449, 513,
                    577, 641, 705, 769, 833, 897, 961, 1089,
                    1153, 1281, 1473, 1665, 1985, 2369, 3009,
                    3969, 6017]
        # fmt: on
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=65536)
        assert reps == expected, (
            f"188-SM non-broadcast (K=65536) sequence mismatch\n"
            f"  Expected ({len(expected)}): {expected}\n"
            f"  Got      ({len(reps)}): {reps}"
        )

    # ── CUDA real cross-check (skipped when GPU or TileLang unavailable) ─

    def test_selector_matches_compute_num_split(self, monkeypatch) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for production cross-check")

        MODULE_PATH = "vllm.model_executor.kernels.mhc.tilelang_kernels"
        monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
        importlib.invalidate_caches()

        try:
            from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                compute_num_split,
            )
        except Exception as e:
            pytest.skip(f"TileLang real module unavailable: {e}")

        real_module = sys.modules[MODULE_PATH]
        assert real_module.__file__ is not None
        assert "vllm" in real_module.__file__

        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=4096)
        assert len(reps) >= 1
        ns_values = [compute_num_split(64, 4096, cdiv(t, 64)) for t in reps]
        assert len(set(ns_values)) == len(ns_values), (
            f"Duplicate n_splits: {dict(zip(reps, ns_values))}"
        )
        assert all(ns >= 1 for ns in ns_values), "Some n_splits are zero"


# ── _find_first_mhc_layer ───────────────────────────────────────────────


class TestFindFirstMhcLayer:
    def test_finds_layer_with_all_required_attrs(self) -> None:
        class MockLayer:
            hc_pre = hc_post = hc_attn_fn = hc_attn_scale = None
            hc_attn_base = hc_ffn_fn = hc_ffn_scale = hc_ffn_base = None

        MockLayer.__name__ = MockLayer.__qualname__ = "DeepseekV4DecoderLayer"

        class MockModel:
            def modules(self):
                yield self
                yield MockLayer()

        result = _find_first_mhc_layer(MockModel())
        assert result is not None
        assert result.__class__.__name__ == "DeepseekV4DecoderLayer"

    def test_skips_layer_missing_required_attr(self) -> None:
        class IncompleteLayer:
            hc_post = hc_attn_fn = hc_attn_scale = hc_attn_base = None
            hc_ffn_fn = hc_ffn_scale = hc_ffn_base = None

        IncompleteLayer.__name__ = IncompleteLayer.__qualname__ = (
            "DeepseekV4DecoderLayer"
        )

        class MockModel:
            def modules(self):
                yield self
                yield IncompleteLayer()

        assert _find_first_mhc_layer(MockModel()) is None


# ── _find_deepseek_v4_model ──────────────────────────────────────────────


class TestFindDeepseekV4Model:
    def test_finds_model_with_all_required_attrs(self) -> None:
        class MockDsModel:
            hc_head_fn = hc_head_scale = hc_head_base = None

        MockDsModel.__name__ = MockDsModel.__qualname__ = "DeepseekV4Model"

        class MockModel:
            def modules(self):
                yield self
                yield MockDsModel()

        result = _find_deepseek_v4_model(MockModel())
        assert result is not None
        assert result.__class__.__name__ == "DeepseekV4Model"

    def test_skips_model_missing_required_attr(self) -> None:
        class IncompleteModel:
            hc_head_fn = hc_head_scale = None

        IncompleteModel.__name__ = IncompleteModel.__qualname__ = "DeepseekV4Model"

        class MockModel:
            def modules(self):
                yield self
                yield IncompleteModel()

        assert _find_deepseek_v4_model(MockModel()) is None

    def test_skips_wrong_class_name(self) -> None:
        class OtherModel:
            hc_head_fn = hc_head_scale = hc_head_base = None

        OtherModel.__name__ = OtherModel.__qualname__ = "OtherModel"

        class MockModel:
            def modules(self):
                yield self
                yield OtherModel()

        assert _find_deepseek_v4_model(MockModel()) is None


# ── Broadcast no-op gates ─────────────────────────────────────────────────


class TestWarmupBroadcastNoOpConditions:
    @staticmethod
    def _install_broadcast_spy(monkeypatch) -> list[str]:
        calls: list[str] = []
        module_path = "vllm.model_executor.kernels.mhc.tilelang"
        stub = types.ModuleType(module_path)
        stub.mhc_pre_broadcast_tilelang = lambda *args, **kwargs: calls.append(
            "mhc_pre_broadcast_tilelang"
        )
        monkeypatch.setitem(sys.modules, module_path, stub)
        return calls

    @staticmethod
    def _model_with_layer(layer):
        layer.__name__ = layer.__qualname__ = "DeepseekV4DecoderLayer"

        class MockModel:
            def modules(self):
                yield self
                yield layer()

        return MockModel()

    def test_noop_when_no_broadcast_layer(self) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_broadcast_mhc,
        )

        class NoBroadcastLayer:
            hc_attn_fn_broadcast = None

        _warmup_broadcast_mhc(
            self._model_with_layer(NoBroadcastLayer), token_sizes=[1, 2, 4]
        )

    def test_noop_when_device_not_cuda(self, monkeypatch) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_broadcast_mhc,
        )

        calls = self._install_broadcast_spy(monkeypatch)

        class CpuLayer:
            hc_attn_fn_broadcast = torch.empty(0)
            hc_attn_fn = torch.empty(0, device="cpu")

        _warmup_broadcast_mhc(self._model_with_layer(CpuLayer), token_sizes=[1, 2, 4])
        assert not calls

    def test_noop_when_broadcast_not_a_tensor(self, monkeypatch) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_broadcast_mhc,
        )

        calls = self._install_broadcast_spy(monkeypatch)

        class BoolBroadcastLayer:
            hc_attn_fn_broadcast = True
            hc_attn_fn = SimpleNamespace(device=torch.device("cuda"))

        _warmup_broadcast_mhc(
            self._model_with_layer(BoolBroadcastLayer), token_sizes=[1, 2, 4]
        )
        assert not calls

    def test_noop_when_broadcast_device_mismatch(self, monkeypatch) -> None:
        from unittest import mock

        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_broadcast_mhc,
        )

        calls = self._install_broadcast_spy(monkeypatch)
        broadcast = mock.MagicMock(spec=torch.Tensor)
        broadcast.device = torch.device("cuda:1")

        class MismatchedLayer:
            hc_attn_fn_broadcast = broadcast
            hc_attn_fn = SimpleNamespace(device=torch.device("cuda:0"))

        _warmup_broadcast_mhc(
            self._model_with_layer(MismatchedLayer), token_sizes=[1, 2, 4]
        )
        assert not calls


# ── _warmup_layer_mhc union behavior ─────────────────────────────────────


class TestWarmupLayerMhcUnion:
    """Verify ``_warmup_layer_mhc`` unions general ``token_sizes`` with
    split-key reps computed from ``hc_mult * hidden_size``, and calls
    ``hc_pre``/``hc_post`` for each token size in the union."""

    def test_unions_general_sizes_with_split_key_reps(self, monkeypatch) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_layer_mhc,
        )

        # Patch compute_num_split so _select_mhc_split_key_token_sizes works
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)

        called_sizes: list[int] = []

        class MockFn:
            device = torch.device("cpu")

        class MockLayer:
            hidden_size = 4096
            hc_mult = 16
            hc_attn_fn = hc_attn_scale = hc_attn_base = MockFn()
            hc_ffn_fn = hc_ffn_scale = hc_ffn_base = MockFn()

            def hc_pre(self, residual_slice, fn, scale, base):
                called_sizes.append(residual_slice.shape[0])
                return (None, None, None)

            def hc_post(self, layer_input, residual_slice, post_mix, comb_mix):
                pass

        MockLayer.__name__ = MockLayer.__qualname__ = "DeepseekV4DecoderLayer"

        general_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        _warmup_layer_mhc(MockLayer(), general_sizes)

        # Compute expected union: general | split-key (K=65536, 188 SMs)
        split_key_sizes = _select_mhc_split_key_token_sizes(
            max_tokens=max(general_sizes), k_size=65536
        )
        expected_union = sorted(set(general_sizes) | set(split_key_sizes))

        # hc_pre is called once per size per (attn + ffn) = 2x per size.
        # Deduplicate to check unique sizes covered.
        unique_called = sorted(set(called_sizes))
        assert unique_called == expected_union, (
            f"_warmup_layer_mhc called sizes: {unique_called}\n"
            f"Expected union: {expected_union}\n"
            f"General: {general_sizes}\n"
            f"Split-key: {split_key_sizes}"
        )

    def test_calls_both_attn_and_ffn_for_each_size(self, monkeypatch) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            _warmup_layer_mhc,
        )

        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)

        call_log: list[str] = []

        class MockFn:
            device = torch.device("cpu")

        class MockLayer:
            hidden_size = 4096
            hc_mult = 16
            hc_attn_fn = hc_attn_scale = hc_attn_base = MockFn()
            hc_ffn_fn = hc_ffn_scale = hc_ffn_base = MockFn()

            def hc_pre(self, residual_slice, fn, scale, base):
                size = residual_slice.shape[0]
                call_log.append(f"hc_pre(size={size})")
                return (None, None, None)

            def hc_post(self, layer_input, residual_slice, post_mix, comb_mix):
                call_log.append("hc_post")

        MockLayer.__name__ = MockLayer.__qualname__ = "DeepseekV4DecoderLayer"

        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        _warmup_layer_mhc(MockLayer(), [1, 2])

        # For each size: 2 calls (attn + ffn) * hc_pre+post = 4 log entries
        # For 2 general sizes + any split-key reps bounded by max_tokens=2
        # (which should be zero split-key reps since max_tokens=2 < 65)
        # So 2 sizes * (hc_pre_attn, hc_post, hc_pre_ffn, hc_post) = 8 entries
        assert len(call_log) == 8, (
            f"Expected 8 log entries for 2 general sizes, got {len(call_log)}: "
            f"{call_log}"
        )


# ── Model-type gate ───────────────────────────────────────────────────────


class TestDeepseekV4ModelGate:
    def test_returns_early_for_non_dsv4_model_type(self) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            deepseek_v4_mhc_warmup,
        )

        class OtherModel:
            config = SimpleNamespace(model_type="llama")

            def modules(self):
                return iter([])

        deepseek_v4_mhc_warmup(OtherModel(), max_tokens=1024)


# ── Orchestration ─────────────────────────────────────────────────────────


class TestDeepseekV4MhcWarmupOrchestration:
    """All three internal stages called in order.  TileLang dependencies
    monkeypatched; mock modules use CUDA-like device attributes."""

    def test_all_three_stages_called_in_order(self, monkeypatch) -> None:
        from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
            deepseek_v4_mhc_warmup,
        )

        calls: list[tuple] = []
        _fake_cuda = SimpleNamespace(device=torch.device("cuda"))

        def _record_layer(layer, token_sizes):
            calls.append(("_warmup_layer_mhc", token_sizes))

        def _record_broadcast(model_arg, token_sizes):
            calls.append(("_warmup_broadcast_mhc", model_arg, token_sizes))

        def _record_head(model_arg, token_sizes):
            calls.append(("_warmup_hc_head", token_sizes))

        class MockLayer:
            hc_pre = hc_post = lambda *a: None
            hc_attn_fn = hc_attn_scale = hc_attn_base = _fake_cuda
            hc_ffn_fn = hc_ffn_scale = hc_ffn_base = _fake_cuda
            hidden_size = 4096
            hc_mult = 16

        MockLayer.__name__ = MockLayer.__qualname__ = "DeepseekV4DecoderLayer"

        class MockDsModel:
            hc_head_fn = hc_head_scale = hc_head_base = _fake_cuda
            config = SimpleNamespace(hidden_size=4096)
            hc_mult = 16
            hc_eps = rms_norm_eps = 1e-6

        MockDsModel.__name__ = MockDsModel.__qualname__ = "DeepseekV4Model"

        class MockModel:
            config = SimpleNamespace(model_type="deepseek_v4")

            def modules(self):
                yield self
                yield MockDsModel()
                yield MockLayer()

        monkeypatch.setattr(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._warmup_layer_mhc",
            _record_layer,
        )
        monkeypatch.setattr(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._warmup_broadcast_mhc",
            _record_broadcast,
        )
        monkeypatch.setattr(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._warmup_hc_head",
            _record_head,
        )
        monkeypatch.setattr(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup.torch.accelerator.synchronize",
            lambda: calls.append(("synchronize",)),
        )
        monkeypatch.setattr(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup.logger.info",
            lambda *a, **kw: None,
        )

        deepseek_v4_mhc_warmup(MockModel(), max_tokens=1024)

        assert len(calls) == 4, f"Expected 4 calls, got {len(calls)}: {calls}"

        s1_name, s1_sizes = calls[0]
        assert s1_name == "_warmup_layer_mhc"
        assert isinstance(s1_sizes, list) and len(s1_sizes) > 0
        assert 1 in s1_sizes

        s2_name, s2_model, s2_sizes = calls[1]
        assert s2_name == "_warmup_broadcast_mhc"
        assert s2_sizes == s1_sizes
        assert s2_model.__class__.__name__ == "MockModel"

        s3_name, s3_sizes = calls[2]
        assert s3_name == "_warmup_hc_head"
        assert s3_sizes == s1_sizes

        assert calls[3] == ("synchronize",)


# ── No sys.modules leakage ───────────────────────────────────────────────


class TestNoSysModulesLeakage:
    """Verify that ``_select_mhc_split_key_token_sizes`` does not leave
    ``tilelang_kernels`` permanently cached in ``sys.modules`` after the
    stub is restored by monkeypatch cleanup."""

    MODULE_PATH = "vllm.model_executor.kernels.mhc.tilelang_kernels"

    def test_clean_modules_after_monkeypatch_cleanup(self, monkeypatch) -> None:
        # Ensure module is not already in sys.modules
        monkeypatch.delitem(sys.modules, self.MODULE_PATH, raising=False)

        # Patch and invoke
        _patch_compute_split(n_sms=188, monkeypatch=monkeypatch)
        reps = _select_mhc_split_key_token_sizes(max_tokens=8192, k_size=4096)
        assert len(reps) == 16

        # Monkeypatch cleanup restores original state: module removed if absent
        monkeypatch.undo()
        # After undo, the module should not be present (it wasn't before)
        assert self.MODULE_PATH not in sys.modules, (
            f"{self.MODULE_PATH} leaked into sys.modules"
        )
