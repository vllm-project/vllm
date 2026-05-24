# SPDX-License-Identifier: Apache-2.0
"""TDD tests for Patch 38 — TurboQuant continuation-prefill persistent
workspace (K_full/V_full) and 4-D dequant prealloc.

Covers:
- 4-D dequant buffer allocation shape correctness
- K_full/V_full allocation shape + dtype
- Idempotency of get_or_create (same pointer on repeat)
- clear_for_tests() removes ALL buffer classes
- get_registry_info() aggregates P38 correctly
- Copy-assembly correctness: prefix + chunk → concatenated K_full matches
  reference `torch.cat(...)` output byte-for-byte
- Copy-assembly handles non-contiguous source (matching actual dequant
  `k_cached[0, :, :cached_len, :].transpose(0, 1)` layout)
- Platform guard: should_apply() on non-NVIDIA

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _clean_manager_state():
    """Each test starts with an empty manager. Fixture auto-applied via
    `@pytest.fixture(autouse=True)`."""
    from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
    TurboQuantBufferManager.clear_for_tests()
    yield
    TurboQuantBufferManager.clear_for_tests()


class TestP38DequantBuffer4D:
    """Group 1: P38 4-D K/V dequant buffer allocation."""

    def test_shape_is_4d_matches_engine_expectation(self, monkeypatch):
        """Dev134 engine does `k_buf[:, :, :alloc_len, :]` — shape must be 4-D."""
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        # Force should_apply True for CPU-run of this test
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k, v = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=8192,
            device="cpu", dtype=torch.float16,
        )
        assert k.dim() == 4
        assert v.dim() == 4
        # Engine's slice pattern: k_buf[:, :, :alloc_len, :] — needs
        # shape[2] >= alloc_len. We expose max_alloc_len on that axis.
        assert k.shape == (1, 2, 8192, 128)
        assert v.shape == (1, 2, 8192, 128)
        assert k.dtype == torch.float16
        assert v.dtype == torch.float16

    def test_pointer_stable_on_repeat(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        a_k, a_v = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        b_k, b_v = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        assert a_k.data_ptr() == b_k.data_ptr()
        assert a_v.data_ptr() == b_v.data_ptr()

    def test_different_keys_get_different_buffers(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        a_k, _ = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        b_k, _ = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=4, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        assert a_k.data_ptr() != b_k.data_ptr()
        assert a_k.shape[1] != b_k.shape[1]


class TestP38FullBuffer:
    """Group 2: P38 K_full/V_full workspace buffer."""

    def test_shape_matches_flash_attn_layout(self, monkeypatch):
        """K_full is consumed by flash_attn_varlen as
        (total_seq_len, Hk, D) — verify layout exactly."""
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k, v = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=10240,
            device="cpu", dtype=torch.float16,
        )
        assert k.shape == (10240, 2, 128)
        assert v.shape == (10240, 2, 128)
        assert k.dtype == torch.float16

    def test_is_contiguous(self, monkeypatch):
        """flash_attn_varlen needs contiguous K/V."""
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k, _ = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=1024,
            device="cpu", dtype=torch.float16,
        )
        assert k.is_contiguous()

    def test_idempotent(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        a_k, a_v = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=1024,
            device="cpu", dtype=torch.float16,
        )
        b_k, b_v = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=1024,
            device="cpu", dtype=torch.float16,
        )
        assert a_k.data_ptr() == b_k.data_ptr()
        assert a_v.data_ptr() == b_v.data_ptr()


class TestP38CopyAssembly:
    """Group 3: Correctness — prefix+chunk copy-assembly equals torch.cat."""

    @pytest.fixture
    def guards_cuda(self, monkeypatch):
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        return True

    def test_prefix_chunk_equals_torch_cat(self, guards_cuda):
        """Assemble (cached_len + q_len, Hk, D) via copy; compare byte-exact
        to `torch.cat([cached_trim, chunk])`."""
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager

        # Mimic upstream layout: k_cached is (1, Hk, alloc_len, D) FP16.
        torch.manual_seed(42)
        Hk, D, cached_len, q_len = 2, 16, 100, 20
        alloc_len = 128
        k_cached = torch.randn(
            (1, Hk, alloc_len, D), dtype=torch.float16,
        )
        key_chunk = torch.randn((q_len, Hk, D), dtype=torch.float16)

        # Reference: upstream `_continuation_prefill` path
        k_cached_trim = (
            k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
        )
        reference = torch.cat([k_cached_trim, key_chunk], dim=0)

        # P38 path: in-place copy into persistent K_full
        k_full_buf, _ = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=Hk, head_size=D, max_seq_cap=alloc_len + q_len + 64,
            device="cpu", dtype=torch.float16,
        )
        seq_len = cached_len + q_len
        k_full = k_full_buf[:seq_len]
        # Non-contiguous source (transpose) → contiguous destination via copy_
        k_full[:cached_len].copy_(
            k_cached[0, :, :cached_len, :].transpose(0, 1)
        )
        k_full[cached_len:seq_len].copy_(key_chunk)

        assert k_full.shape == reference.shape
        assert torch.equal(k_full, reference)

    def test_copy_source_can_be_noncontiguous(self, guards_cuda):
        """transpose(0, 1) result is non-contiguous; verify copy_ handles it
        and the destination stays contiguous."""
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager

        Hk, D, cached_len = 2, 8, 40
        alloc_len = 64
        k_cached = torch.randn((1, Hk, alloc_len, D), dtype=torch.float16)
        src = k_cached[0, :, :cached_len, :].transpose(0, 1)
        assert not src.is_contiguous()

        k_full_buf, _ = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=Hk, head_size=D, max_seq_cap=256,
            device="cpu", dtype=torch.float16,
        )
        k_full = k_full_buf[:cached_len]
        assert k_full.is_contiguous()
        k_full.copy_(src)
        assert torch.equal(k_full, src.contiguous())


class TestP38RegistryIntegration:
    """Group 4: get_registry_info + clear_for_tests integration."""

    def test_registry_includes_p38_bytes(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        _ = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        _ = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=2048,
            device="cpu", dtype=torch.float16,
        )

        info = TurboQuantBufferManager.get_registry_info()
        assert "total_bytes_p38_continuation" in info
        assert info["total_bytes_p38_continuation"] > 0
        assert "p38_continuation_entries" in info
        # 4 pools: k_dequant_4d, v_dequant_4d, k_full, v_full
        assert len(info["p38_continuation_entries"]) == 4
        pool_types = {e["pool"] for e in info["p38_continuation_entries"]}
        assert pool_types == {"k_dequant_4d", "v_dequant_4d", "k_full", "v_full"}

    def test_clear_for_tests_removes_p38(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=1024,
            device="cpu", dtype=torch.float16,
        )

        TurboQuantBufferManager.clear_for_tests()

        # All four P38 dicts must be empty
        assert len(TurboQuantBufferManager._P38_K_DEQUANT_4D_BUFFERS) == 0
        assert len(TurboQuantBufferManager._P38_V_DEQUANT_4D_BUFFERS) == 0
        assert len(TurboQuantBufferManager._P38_K_FULL_BUFFERS) == 0
        assert len(TurboQuantBufferManager._P38_V_FULL_BUFFERS) == 0


class TestP38PlatformGuard:
    """Group 5: should_apply() and platform-skip behaviour."""

    def test_returns_none_tuple_when_platform_skipped(self, monkeypatch):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        k, v = TurboQuantBufferManager.get_or_create_p38_dequant_4d(
            num_kv_heads=2, head_size=128, max_alloc_len=1024,
            device="cpu", dtype=torch.float16,
        )
        assert k is None
        assert v is None

        k, v = TurboQuantBufferManager.get_or_create_p38_full(
            num_kv_heads=2, head_size=128, max_seq_cap=1024,
            device="cpu", dtype=torch.float16,
        )
        assert k is None
        assert v is None


class TestP38WiringPatch:
    """Group 6: wiring/patch_38_tq_continuation_memory module surface."""

    def test_should_apply_guard(self, monkeypatch):
        # patch_38 imports `is_nvidia_cuda` at module level via
        # `from ... import is_nvidia_cuda`, so the local name binding
        # is what should_apply() actually calls. Monkey-patching the
        # source `vllm._genesis.guards.is_nvidia_cuda` does NOT
        # propagate to patch_38's local name (already captured at
        # import time). We must patch the local binding directly.
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        monkeypatch.setattr(p38, "is_nvidia_cuda", lambda: False)
        assert p38.should_apply() is False

    def test_apply_skips_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        monkeypatch.setattr(p38, "is_nvidia_cuda", lambda: False)
        status, reason = p38.apply()
        assert status == "skipped"
        assert "NVIDIA" in reason

    def test_wiring_has_expected_public_surface(self):
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        assert callable(p38.apply)
        assert callable(p38.is_applied)
        assert callable(p38.revert)
        assert callable(p38.should_apply)
        assert callable(p38._genesis_continuation_prefill)


class TestP38Idempotency:
    """Regression tests for the double-apply `_genesis_p38_original`
    overwrite bug caught in the v7.3 code audit. Protects against a
    class of state-corruption that would have `revert()` restore our own
    wrapper instead of the real upstream method.

    Note on isolation: `_genesis_continuation_prefill` is a MODULE-LEVEL
    function inside `patch_38_tq_continuation_memory`, so `apply()`
    stamps `_genesis_p38_original` on it ONCE per process. We wipe that
    attr (and the marker) before each test to keep the guard logic
    observable per test.
    """

    @pytest.fixture(autouse=True)
    def _reset_p38_module_state(self):
        """Clear the stamp from any prior apply() so each test sees a
        fresh guard state."""
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        fn = p38._genesis_continuation_prefill
        for attr in ("_genesis_p38_original", p38._GENESIS_P38_MARKER_ATTR):
            if hasattr(fn, attr):
                delattr(fn, attr)
        yield
        for attr in ("_genesis_p38_original", p38._GENESIS_P38_MARKER_ATTR):
            if hasattr(fn, attr):
                delattr(fn, attr)

    def _install_fake_impl(self, monkeypatch):
        """Helper: inject a fake turboquant_attn module with a
        TurboQuantAttentionImpl class whose _continuation_prefill is a
        distinct sentinel function we can identify later.

        Patches `is_nvidia_cuda` / `is_sm_at_least` on the WIRING MODULE
        directly (not on `guards`) because the wiring does
        `from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least`
        at import time → the symbols are bound in the wiring module's
        namespace, not looked up dynamically.
        """
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        monkeypatch.setattr(p38, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(
            p38, "is_sm_at_least", lambda major, minor=0: True,
        )

        import sys, types
        fake_mod = types.ModuleType(
            "vllm.v1.attention.backends.turboquant_attn"
        )

        def _fake_original(self, *args, **kwargs):
            return "ORIGINAL_SENTINEL"

        def _fake_flash_attn_varlen(self, *args, **kwargs):
            return None

        class FakeImpl:
            # Minimum interface required by P38's v7.8 P49 pre-flight
            # guard. If our guard's required_methods list grows, this
            # FakeImpl must gain the new methods too — that's the
            # correctness contract we're testing.
            _continuation_prefill = _fake_original
            _flash_attn_varlen = _fake_flash_attn_varlen

        fake_mod.TurboQuantAttentionImpl = FakeImpl
        monkeypatch.setitem(
            sys.modules,
            "vllm.v1.attention.backends.turboquant_attn",
            fake_mod,
        )
        return FakeImpl, _fake_original

    def test_double_apply_preserves_true_original(self, monkeypatch):
        """If apply() is called twice, the saved `_genesis_p38_original`
        MUST still point to the true upstream method — never our wrapper.
        """
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        FakeImpl, fake_original = self._install_fake_impl(monkeypatch)

        # First apply
        status1, reason1 = p38.apply()
        assert status1 == "applied", (
            f"first apply should succeed but returned ({status1}, {reason1})"
        )
        assert getattr(
            FakeImpl._continuation_prefill,
            "_genesis_p38_original", None,
        ) is fake_original

        # Second apply — must NOT overwrite _genesis_p38_original with
        # our own wrapper.
        status2, reason2 = p38.apply()
        assert status2 == "applied"
        assert "already wrapped" in reason2 or "idempotent" in reason2
        assert getattr(
            FakeImpl._continuation_prefill,
            "_genesis_p38_original", None,
        ) is fake_original, (
            "double-apply overwrote _genesis_p38_original with wrapper "
            "— revert() would fail to restore real upstream method"
        )

        # revert() must restore the true original.
        assert p38.revert() is True
        assert FakeImpl._continuation_prefill is fake_original

    def test_is_applied_tracks_state(self, monkeypatch):
        """is_applied() returns True only while our wrapper is live."""
        from vllm._genesis.wiring.legacy import patch_38_tq_continuation_memory as p38
        _FakeImpl, _fake_original = self._install_fake_impl(monkeypatch)

        assert p38.is_applied() is False
        status, reason = p38.apply()
        assert status == "applied", f"apply returned ({status}, {reason})"
        assert p38.is_applied() is True
        assert p38.revert() is True
        assert p38.is_applied() is False
