# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FLA Triton kernel warmup fix (issue #34954).

Validates that:
1. The _warmup_prefill_kernels method has correct flag management,
   tensor shapes, and one-shot semantics via double-checked locking.
2. The _forward_core method calls warmup when attn_metadata is None
   (profile_run path).
3. On GPU, FLA kernels can be pre-warmed so that subsequent calls
   under memory pressure succeed (the core bug scenario).
"""

import threading
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform

_cuda_mod = torch.cuda
_empty_cache_fn = getattr(torch.accelerator, "empty_cache", _cuda_mod.empty_cache)


try:
    from vllm.model_executor.models.qwen3_next import (
        Qwen3NextGatedDeltaNet,
    )

    _CAN_IMPORT_GDN = True
except (RuntimeError, ImportError):
    _CAN_IMPORT_GDN = False

_requires_gdn = pytest.mark.skipif(
    not _CAN_IMPORT_GDN,
    reason="Cannot import Qwen3NextGatedDeltaNet "
    "(likely _custom_ops registration conflict in dev setup)",
)


def _make_mock_instance(
    *,
    num_k_heads=4,
    num_v_heads=4,
    tp_size=1,
    head_k_dim=64,
    head_v_dim=64,
    dtype=torch.bfloat16,
    device="cpu",
):
    """Create a mock that satisfies _warmup_prefill_kernels's interface."""
    instance = mock.MagicMock()
    instance.num_k_heads = num_k_heads
    instance.num_v_heads = num_v_heads
    instance.tp_size = tp_size
    instance.head_k_dim = head_k_dim
    instance.head_v_dim = head_v_dim
    instance.prefix = "test_layer"
    instance.get_state_dtype.return_value = (dtype, dtype)

    mixed_qkv = torch.zeros(1, dtype=dtype, device=device)
    return instance, mixed_qkv


# ---------------------------------------------------------------------------
# Unit tests (CPU, no GPU required) — mock out chunk_gated_delta_rule
# ---------------------------------------------------------------------------


@_requires_gdn
@pytest.mark.skip_global_cleanup
class TestWarmupFlagManagement:
    """Verify the one-shot flag logic without executing real kernels."""

    def setup_method(self):
        Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up = False

    def teardown_method(self):
        Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up = False

    def test_flag_starts_false(self):
        assert Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up is False

    def test_warmup_sets_flag(self):
        """After one call the flag must be True."""
        instance, mixed_qkv = _make_mock_instance()
        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(instance, mixed_qkv)
        assert Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up is True

    def test_warmup_runs_only_once(self):
        """chunk_gated_delta_rule must be called exactly 3 times (T=16,32,64)
        on the first invocation, then zero times on subsequent calls."""
        inst1, qkv1 = _make_mock_instance()
        inst2, qkv2 = _make_mock_instance()
        inst3, qkv3 = _make_mock_instance()

        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(inst1, qkv1)
        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(inst2, qkv2)
        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(inst3, qkv3)

        assert inst1.chunk_gated_delta_rule.call_count == 3
        assert inst2.chunk_gated_delta_rule.call_count == 0
        assert inst3.chunk_gated_delta_rule.call_count == 0

    def test_warmup_thread_safe(self):
        """Concurrent calls from multiple threads must still invoke
        chunk_gated_delta_rule on exactly one instance."""
        barrier = threading.Barrier(8)
        instances = []

        def _call():
            inst, qkv = _make_mock_instance()
            instances.append(inst)
            barrier.wait()
            Qwen3NextGatedDeltaNet._warmup_prefill_kernels(inst, qkv)

        threads = [threading.Thread(target=_call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_calls = sum(inst.chunk_gated_delta_rule.call_count for inst in instances)
        assert total_calls == 3

    def test_warmup_passes_correct_shapes(self):
        """Verify dummy tensors have shapes derived from instance attributes."""
        instance, mixed_qkv = _make_mock_instance(
            num_k_heads=16,
            num_v_heads=32,
            tp_size=1,
            head_k_dim=128,
            head_v_dim=128,
        )
        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(instance, mixed_qkv)

        # 3 calls for T=16, T=32, T=64
        assert instance.chunk_gated_delta_rule.call_count == 3
        for i, T in enumerate((16, 32, 64)):
            kw = instance.chunk_gated_delta_rule.call_args_list[i].kwargs
            assert kw["q"].shape == (1, T, 16, 128)
            assert kw["v"].shape == (1, T, 32, 128)
            assert kw["g"].shape == (1, T, 32)
            assert kw["beta"].shape == (1, T, 32)
            assert kw["cu_seqlens"].tolist() == [0, T]
            assert kw["output_final_state"] is False
            assert kw["use_qk_l2norm_in_kernel"] is True

    def test_warmup_respects_tp_size(self):
        """With tp_size > 1 the head counts must be divided."""
        instance, mixed_qkv = _make_mock_instance(
            num_k_heads=16,
            num_v_heads=32,
            tp_size=2,
            head_k_dim=128,
            head_v_dim=128,
        )
        Qwen3NextGatedDeltaNet._warmup_prefill_kernels(instance, mixed_qkv)

        kw = instance.chunk_gated_delta_rule.call_args_list[0].kwargs
        assert kw["q"].shape[2] == 8  # 16 / 2
        assert kw["v"].shape[2] == 16  # 32 / 2


@_requires_gdn
@pytest.mark.skip_global_cleanup
class TestForwardCoreProfilePath:
    """Verify _forward_core calls warmup when attn_metadata is None."""

    def setup_method(self):
        Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up = False

    def teardown_method(self):
        Qwen3NextGatedDeltaNet._prefill_kernels_warmed_up = False

    def test_forward_core_calls_warmup_when_attn_metadata_none(self):
        """During profile_run, attn_metadata is None and _warmup should
        be invoked."""
        mock_forward_ctx = mock.MagicMock()
        mock_forward_ctx.attn_metadata = None

        with (
            mock.patch(
                "vllm.model_executor.models.qwen3_next.get_forward_context",
                return_value=mock_forward_ctx,
            ),
            mock.patch.object(
                Qwen3NextGatedDeltaNet,
                "_warmup_prefill_kernels",
            ) as mock_warmup,
        ):
            instance = mock.MagicMock(spec=Qwen3NextGatedDeltaNet)
            mixed_qkv = torch.empty(0)
            Qwen3NextGatedDeltaNet._forward_core(
                instance,
                mixed_qkv=mixed_qkv,
                b=torch.empty(0),
                a=torch.empty(0),
                core_attn_out=torch.empty(0),
            )
            mock_warmup.assert_called_once_with(mixed_qkv)

    def test_forward_core_returns_early_during_profile(self):
        """_forward_core should return immediately during profile_run
        without processing attn_metadata further."""
        mock_forward_ctx = mock.MagicMock()
        mock_forward_ctx.attn_metadata = None

        with (
            mock.patch(
                "vllm.model_executor.models.qwen3_next.get_forward_context",
                return_value=mock_forward_ctx,
            ),
            mock.patch.object(
                Qwen3NextGatedDeltaNet,
                "_warmup_prefill_kernels",
            ),
        ):
            instance = mock.MagicMock(spec=Qwen3NextGatedDeltaNet)
            result = Qwen3NextGatedDeltaNet._forward_core(
                instance,
                mixed_qkv=torch.empty(0),
                b=torch.empty(0),
                a=torch.empty(0),
                core_attn_out=torch.empty(0),
            )
            assert result is None


# ---------------------------------------------------------------------------
# GPU integration tests — use fla_chunk_gated_delta_rule directly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA GPU")
@pytest.mark.skip_global_cleanup
class TestFLAWarmupGPU:
    """Run actual FLA Triton kernels on GPU to verify the warmup pattern
    prevents OOM under memory pressure."""

    @pytest.fixture(autouse=True)
    def _setup_triton_allocator(self):
        """Triton 3.4+ requires an explicit allocator."""
        try:
            import triton

            device = torch.device("cuda:0")
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device=device, dtype=torch.int8
                )
            )
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def _run_fla_kernel(H_K, H_V, K, V, T=128):
        """Run chunk_gated_delta_rule with the given dimensions."""
        from vllm.model_executor.layers.fla.ops import (
            chunk_gated_delta_rule,
        )

        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        q = torch.randn(1, T, H_K, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H_K, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H_V, V, dtype=dtype, device=device)
        g = torch.randn(1, T, H_V, dtype=dtype, device=device)
        beta = torch.rand(1, T, H_V, dtype=dtype, device=device).sigmoid()
        initial_state = torch.zeros(1, H_V, V, K, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

        o, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        torch.accelerator.synchronize()
        return o

    def test_fla_kernel_basic(self):
        """FLA kernel executes successfully with small dimensions."""
        o = self._run_fla_kernel(H_K=4, H_V=4, K=64, V=64)
        assert o.shape == (1, 128, 4, 64)

    def test_fla_kernel_qwen3next_dims(self):
        """FLA kernel executes with actual Qwen3-Next dimensions."""
        o = self._run_fla_kernel(H_K=16, H_V=32, K=128, V=128)
        assert o.shape == (1, 128, 32, 128)

    def test_warmup_then_memory_pressure(self):
        """Core scenario: after warmup, FLA kernels succeed even under
        memory pressure because Triton autotune results are cached."""
        device = torch.device("cuda:0")
        H_K, H_V, K, V = 4, 4, 64, 64

        self._run_fla_kernel(H_K, H_V, K, V)

        _empty_cache_fn()
        free = torch.cuda.mem_get_info()[0]
        target_free = 512 * 1024 * 1024
        alloc_bytes = max(0, free - target_free)
        hog = torch.empty(alloc_bytes, dtype=torch.uint8, device=device)

        try:
            o = self._run_fla_kernel(H_K, H_V, K, V)
            assert o.shape == (1, 128, H_V, V)
        finally:
            del hog
            _empty_cache_fn()
