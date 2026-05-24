# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.dequant_buffer.TurboQuantBufferManager.

Patch 22 migration target: pre-allocate K/V dequant buffers during warmup
(profiler-visible) rather than lazy allocation during forward (profiler-invisible).

Root cause: vllm-project/vllm#40420 — CUDA OOM at ~234k context when lazy
torch.empty() inside _continuation_prefill wasn't accounted for by KV cache
sizing during profile_run.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════
#                       PLATFORM GUARD BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

class TestPlatformGuard:
    """Group 1: should_apply() correctly gates on platform."""

    def test_should_apply_returns_bool(self):
        from vllm._genesis.kernels.dequant_buffer import TurboQuantBufferManager
        result = TurboQuantBufferManager.should_apply()
        assert isinstance(result, bool)

    def test_should_apply_false_on_non_nvidia(self, monkeypatch):
        """Non-NVIDIA platforms → should_apply returns False."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        # Mock is_nvidia_cuda to return False
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        assert db.TurboQuantBufferManager.should_apply() is False

    def test_should_apply_false_on_ancient_sm(self, monkeypatch):
        """SM < 8.0 → should_apply returns False (TurboQuant unsupported)."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)

        assert db.TurboQuantBufferManager.should_apply() is False

    def test_should_apply_true_on_ampere(self, monkeypatch):
        """SM 8.0+ with NVIDIA → should_apply returns True."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        assert db.TurboQuantBufferManager.should_apply() is True


# ═══════════════════════════════════════════════════════════════════════════
#                      KV BUFFER ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════

class TestKVBufferAllocation:
    """Group 2: K/V buffer get_or_create correctness."""

    def test_returns_none_when_platform_incompatible(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """When should_apply False → return (None, None) — never crash."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        k, v = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            num_kv_heads=4, head_size=128, max_alloc_len=4096,
            device="cpu", dtype=torch.bfloat16,
        )
        assert k is None
        assert v is None

    def test_returns_tensors_when_platform_compatible(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """When should_apply True → return two tensors with requested shape."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        # Mock to allow path on CPU for testing
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k, v = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            num_kv_heads=4, head_size=128, max_alloc_len=4096,
            device="cpu", dtype=torch.bfloat16,
        )
        assert isinstance(k, torch.Tensor)
        assert isinstance(v, torch.Tensor)
        assert k.shape == (4, 128, 4096)
        assert v.shape == (4, 128, 4096)
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16

    def test_same_key_returns_same_tensors(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """Same (num_kv_heads, head_size, max_alloc_len, device) → same buffers.

        CRITICAL: shared buffers across layers (10 attn layers in Qwen3.6
        share 1 buffer pair since forward is sequential per layer).
        """
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k1, v1 = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            2, 64, 4096, "cpu", torch.bfloat16
        )
        k2, v2 = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            2, 64, 4096, "cpu", torch.bfloat16
        )

        assert k1 is k2
        assert v1 is v2

    def test_different_shape_different_tensors(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """Different shape params → distinct buffers."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        k1, _ = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            2, 64, 4096, "cpu", torch.bfloat16
        )
        k2, _ = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            4, 128, 8192, "cpu", torch.bfloat16
        )

        assert k1 is not k2
        assert k1.shape != k2.shape

    def test_pointer_stability_across_many_calls(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """CRITICAL for CUDA graph: pointer MUST be stable across 100+ calls."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        pointers = set()
        for _ in range(100):
            k, v = db.TurboQuantBufferManager.get_or_create_kv_buffers(
                2, 64, 4096, "cpu", torch.bfloat16
            )
            pointers.add(k.data_ptr())
            pointers.add(v.data_ptr())

        # Exactly 2 distinct pointers (one for K, one for V)
        assert len(pointers) == 2, (
            f"CUDA graph safety broken: pointer changed across calls. "
            f"Got {len(pointers)} distinct pointers: {pointers}")


# ═══════════════════════════════════════════════════════════════════════════
#                    CU_SEQLENS SCRATCH TENSORS
# ═══════════════════════════════════════════════════════════════════════════

class TestCuSeqlensScratch:
    """Group 3: cu_seqlens pre-allocation (Patch 23 bundled)."""

    def test_returns_none_on_incompatible_platform(
        self, reset_genesis_prealloc, monkeypatch
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        cu_q, cu_k = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        assert cu_q is None
        assert cu_k is None

    def test_returns_int32_tensors_shape_2(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """cu_seqlens format: int32 tensor of shape (2,) — [0, q_len]."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        cu_q, cu_k = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        assert cu_q.dtype == torch.int32
        assert cu_k.dtype == torch.int32
        assert cu_q.shape == (2,)
        assert cu_k.shape == (2,)

    def test_zero_initialized(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """cu_seqlens start with zero values (ready for in-place fill)."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        cu_q, cu_k = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        assert (cu_q == 0).all()
        assert (cu_k == 0).all()

    def test_same_device_returns_same_tensors(
        self, reset_genesis_prealloc, monkeypatch
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        cu_q1, cu_k1 = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        cu_q2, cu_k2 = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")

        assert cu_q1 is cu_q2
        assert cu_k1 is cu_k2

    def test_inplace_modification_persists(
        self, reset_genesis_prealloc, monkeypatch
    ):
        """Writing to cu_seqlens updates the shared tensor (for hot-path use)."""
        from vllm._genesis.kernels import dequant_buffer as db
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)

        cu_q, cu_k = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        cu_q[1] = 1234
        cu_k[1] = 5678

        # Retrieve again — same tensor, modifications visible
        cu_q2, cu_k2 = db.TurboQuantBufferManager.get_or_create_cu_seqlens("cpu")
        assert cu_q2[1].item() == 1234
        assert cu_k2[1].item() == 5678


# ═══════════════════════════════════════════════════════════════════════════
#                    MEMORY FOOTPRINT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryFootprint:
    """Group 4: Memory footprint estimation (helper for warnings)."""

    def test_estimate_buffer_bytes(self):
        """estimate_buffer_bytes correctly computes footprint."""
        from vllm._genesis.kernels.dequant_buffer import estimate_buffer_bytes

        # Qwen3.6-35B-A3B typical: num_kv_heads=4 (TP=2 → 2 per rank),
        #                          head_size=128, max_model_len=262144
        # K+V = 2 × num_kv_heads × head_size × max_alloc_len × 2 bytes (bf16)
        #     = 2 × 2 × 128 × 262144 × 2 = 268435456 bytes = 256 MiB
        estimated = estimate_buffer_bytes(
            num_kv_heads=2, head_size=128, max_alloc_len=262144,
            dtype=torch.bfloat16,
        )
        # K + V combined
        expected = 2 * (2 * 128 * 262144 * 2)
        assert estimated == expected

    def test_estimate_scales_with_tp(self):
        """TP=1 (num_kv_heads=4) = 2× TP=2 (num_kv_heads=2) footprint."""
        from vllm._genesis.kernels.dequant_buffer import estimate_buffer_bytes

        tp1 = estimate_buffer_bytes(4, 128, 262144, torch.bfloat16)
        tp2 = estimate_buffer_bytes(2, 128, 262144, torch.bfloat16)

        assert tp1 == 2 * tp2


# ═══════════════════════════════════════════════════════════════════════════
#                   CUDA-SPECIFIC (optional)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#               P32/P33 — cu_2 + synth_seq_lens preallocs
# ═══════════════════════════════════════════════════════════════════════════

class TestPatch32Cu2:
    """P32: second-hop cu_seqlens scratch (shape (2,), int32)."""

    def test_returns_none_on_incompatible_platform(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: False),
        )
        assert db.TurboQuantBufferManager.get_or_create_cu_2("cpu") is None

    def test_shape_dtype_on_compatible(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_cu_2("cpu")
        assert t is not None
        assert t.shape == (2,)
        assert t.dtype == torch.int32

    def test_zero_initialized(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_cu_2("cpu")
        assert t is not None
        assert t.sum().item() == 0

    def test_pointer_stable_across_calls(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_cu_2("cpu")
        b = db.TurboQuantBufferManager.get_or_create_cu_2("cpu")
        assert a is b, "cu_2 should be pointer-stable for CUDA-graph safety"


class TestPatch33SynthSeqLens:
    """P33: synthetic seq_lens device mirror (shape (max_batch,), int32)."""

    def test_returns_none_on_incompatible_platform(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: False),
        )
        assert db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=16, device="cpu",
        ) is None

    def test_shape_dtype_on_compatible(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=16, device="cpu",
        )
        assert t is not None
        # rounded up to multiple of 8 (16 → 16)
        assert t.shape == (16,)
        assert t.dtype == torch.int32

    def test_rounds_max_batch_up_to_8(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=13, device="cpu",
        )
        assert t is not None
        assert t.shape == (16,)  # 13 rounded up to 16

    def test_same_key_returns_same_buffer(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=32, device="cpu",
        )
        b = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=32, device="cpu",
        )
        assert a is b

    def test_different_batch_caps_distinct_buffers(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=8, device="cpu",
        )
        b = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=64, device="cpu",
        )
        assert a is not b
        assert a.shape[0] != b.shape[0]

    def test_zero_initialized(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_synth_seq_lens(
            max_batch=16, device="cpu",
        )
        assert t is not None
        assert t.sum().item() == 0


class TestPatch26PrefillOutput:
    """P26: prefill output scratch buffer (N_max, Hq, D)."""

    def test_returns_none_on_incompatible_platform(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: False),
        )
        assert db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.bfloat16,
        ) is None

    def test_shape_dtype_on_compatible(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        t = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.bfloat16,
        )
        assert t is not None
        assert t.shape == (4096, 32, 128)
        assert t.dtype == torch.bfloat16

    def test_distinct_shape_keys(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.bfloat16,
        )
        b = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=2048, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.bfloat16,
        )
        assert a is not b
        assert a.shape != b.shape

    def test_same_key_returns_same_buffer(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=16, head_size=64,
            device="cpu", dtype=torch.bfloat16,
        )
        b = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=16, head_size=64,
            device="cpu", dtype=torch.bfloat16,
        )
        assert a is b, "Prefill output buffer must be pointer-stable"

    def test_dtype_separates_buffers(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )
        a = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.bfloat16,
        )
        b = db.TurboQuantBufferManager.get_or_create_prefill_output(
            max_batched_tokens=4096, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.float16,
        )
        assert a is not b
        assert a.dtype != b.dtype


class TestPatch36SharedDecodeBuffers:
    """P36: shared TQ decode mid_o / output / lse across all TQ layers.

    Mirrors upstream PR #40655. Correctness invariant: multiple `init`
    calls with identical (B, Hq, S, D, device, dtype) return the SAME
    tensor (shared pool), while different keys yield different tensors.
    """

    def _mark_compatible(self, monkeypatch):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply",
            classmethod(lambda cls: True),
        )

    def test_mid_o_returns_none_on_incompat(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply",
            classmethod(lambda cls: False),
        )
        assert db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        ) is None

    def test_mid_o_shape_dtype(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)
        t = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        assert t is not None
        assert t.shape == (2, 32, 32, 129)  # D+1
        assert t.dtype == torch.float32

    def test_mid_o_shared_across_same_key(self, monkeypatch, reset_genesis_prealloc):
        """Multiple TQ-layer inits with identical config share ONE tensor."""
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)
        a = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        b = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        assert a is b, "shared pool must return same tensor"
        assert a.data_ptr() == b.data_ptr()

    def test_mid_o_distinct_config_distinct_buffers(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)
        a = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=2, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        b = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            max_num_seqs=4, num_q_heads=32, tq_max_kv_splits=32,
            head_size=128, device="cpu", dtype=torch.float32,
        )
        assert a is not b
        assert a.shape != b.shape

    def test_output_and_lse_shared(self, monkeypatch, reset_genesis_prealloc):
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)
        o1 = db.TurboQuantBufferManager.get_shared_decode_output(
            max_num_seqs=2, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.float32,
        )
        o2 = db.TurboQuantBufferManager.get_shared_decode_output(
            max_num_seqs=2, num_q_heads=32, head_size=128,
            device="cpu", dtype=torch.float32,
        )
        assert o1 is o2
        assert o1.shape == (2, 32, 128)

        l1 = db.TurboQuantBufferManager.get_shared_decode_lse(
            max_num_seqs=2, num_q_heads=32, device="cpu", dtype=torch.float32,
        )
        l2 = db.TurboQuantBufferManager.get_shared_decode_lse(
            max_num_seqs=2, num_q_heads=32, device="cpu", dtype=torch.float32,
        )
        assert l1 is l2
        assert l1.shape == (2, 32)

    def test_pool_memory_math(self, monkeypatch, reset_genesis_prealloc):
        """Qwen3.6-35B-A3B prod config: B=2, Hq=32, S=32, D=128.

        Direct savings with 10 TQ attention layers:
          per-layer mid_o: 2*32*32*129*4 = 1 056 768 B = 1.008 MiB
          per-layer output: 2*32*128*4   = 32 768 B   = 32 KiB
          per-layer lse: 2*32*4          = 256 B
          per-layer total ~= 1.04 MiB
          across 10 layers: 10.40 MiB
          shared: 1.04 MiB
          savings: 9.36 MiB direct (plus allocator slab overhead saved).
        """
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)

        # Allocate once — mimic 10 layers hitting the same key
        t_mid = db.TurboQuantBufferManager.get_shared_decode_mid_o(
            2, 32, 32, 128, device="cpu", dtype=torch.float32,
        )
        t_out = db.TurboQuantBufferManager.get_shared_decode_output(
            2, 32, 128, device="cpu", dtype=torch.float32,
        )
        t_lse = db.TurboQuantBufferManager.get_shared_decode_lse(
            2, 32, device="cpu", dtype=torch.float32,
        )
        shared_bytes = (
            t_mid.element_size() * t_mid.numel()
            + t_out.element_size() * t_out.numel()
            + t_lse.element_size() * t_lse.numel()
        )
        # 1_056_768 + 32_768 + 256 = 1_089_792 B
        assert shared_bytes == 1_056_768 + 32_768 + 256
        # 10 layers × old-pattern would be 10 × shared_bytes;
        # savings = 9 × shared_bytes
        savings_10_layers = 9 * shared_bytes
        assert savings_10_layers > 9 * 1024 * 1024  # > 9 MiB
        assert savings_10_layers < 10 * 1024 * 1024  # < 10 MiB

    def test_clear_for_tests_clears_decode_pools(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        self._mark_compatible(monkeypatch)
        db.TurboQuantBufferManager.get_shared_decode_mid_o(
            2, 32, 32, 128, device="cpu", dtype=torch.float32,
        )
        db.TurboQuantBufferManager.get_shared_decode_output(
            2, 32, 128, device="cpu", dtype=torch.float32,
        )
        assert len(db.TurboQuantBufferManager._DECODE_MID_O_BUFFERS) == 1
        assert len(db.TurboQuantBufferManager._DECODE_OUTPUT_BUFFERS) == 1
        db.TurboQuantBufferManager.clear_for_tests()
        assert len(db.TurboQuantBufferManager._DECODE_MID_O_BUFFERS) == 0
        assert len(db.TurboQuantBufferManager._DECODE_OUTPUT_BUFFERS) == 0


class TestEnsureBuffersBundledP32P33:
    """ensure_turboquant_buffers attaches _tq_cu_2 and _tq_synth_seq_lens."""

    def test_layer_gets_cu_2_and_synth_seq_lens(
        self, monkeypatch, reset_genesis_prealloc,
    ):
        from vllm._genesis.kernels import dequant_buffer as db
        monkeypatch.setattr(
            db.TurboQuantBufferManager, "should_apply", classmethod(lambda cls: True),
        )

        class FakeImpl:
            num_kv_heads = 2
            head_size = 64
            _max_model_len = 2048
            _max_num_seqs = 32

        class FakeLayer:
            pass

        impl = FakeImpl()
        layer = FakeLayer()

        db.ensure_turboquant_buffers(impl, layer, torch.device("cpu"))

        assert hasattr(layer, "_tq_cu_2"), "P32 scratch not attached"
        assert hasattr(layer, "_tq_synth_seq_lens"), "P33 scratch not attached"
        assert layer._tq_cu_2.shape == (2,)
        assert layer._tq_synth_seq_lens.shape == (32,)


@pytest.mark.cuda_required
class TestCUDAIntegration:
    """Group 5: Real CUDA device allocation (GPU-required)."""

    def test_cuda_allocation_succeeds(
        self, reset_genesis_prealloc, cuda_available
    ):
        if not cuda_available:
            pytest.skip("CUDA not available")

        from vllm._genesis.kernels import dequant_buffer as db

        # Use small size to avoid OOM during test
        k, v = db.TurboQuantBufferManager.get_or_create_kv_buffers(
            num_kv_heads=2, head_size=64, max_alloc_len=1024,
            device="cuda", dtype=torch.bfloat16,
        )

        # If should_apply() returns True on this CUDA device, tensors should exist
        if k is not None:
            assert k.device.type == "cuda"
            assert v.device.type == "cuda"
