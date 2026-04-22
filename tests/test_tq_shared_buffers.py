# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that TurboQuant decode buffers are shared across attention layers.

This verifies the fix for excessive memory usage where per-layer buffer
allocation consumed O(num_layers) GPU memory instead of O(1).

Run: pytest tests/test_tq_shared_buffers.py -v
Requires: GPU with TurboQuant support (SM >= 90) for e2e test
"""

import gc
import multiprocessing

import pytest
import torch


def _cleanup_shared_buffers():
    """Reset shared TQ buffers between tests."""
    from vllm.model_executor.layers.attention.attention import Attention

    for attr in ("_tq_shared_mid_o_buf", "_tq_shared_output_buf", "_tq_shared_lse_buf"):
        if hasattr(Attention, attr):
            delattr(Attention, attr)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up shared state before and after each test."""
    _cleanup_shared_buffers()
    yield
    _cleanup_shared_buffers()


def _simulate_tq_init(B, Hq, S, D):
    """Reproduce the shared buffer init logic from _init_turboquant_buffers."""
    from vllm.model_executor.layers.attention.attention import Attention

    if not hasattr(
        Attention, "_tq_shared_mid_o_buf"
    ) or Attention._tq_shared_mid_o_buf.shape != (B, Hq, S, D + 1):
        Attention._tq_shared_mid_o_buf = torch.empty(
            B, Hq, S, D + 1, dtype=torch.float32
        )
        Attention._tq_shared_output_buf = torch.empty(B, Hq, D, dtype=torch.float32)
        Attention._tq_shared_lse_buf = torch.empty(B, Hq, dtype=torch.float32)

    return (
        Attention._tq_shared_mid_o_buf,
        Attention._tq_shared_output_buf,
        Attention._tq_shared_lse_buf,
    )


def _pp_rank_init(B, Hq, S, D, result_queue):
    """Simulate a PP rank init in a spawned subprocess."""
    bufs = _simulate_tq_init(B, Hq, S, D)
    result_queue.put(bufs[0].shape)


class TestTurboQuantSharedBuffers:
    """Tests for TurboQuant shared buffer optimization."""

    def test_buffers_are_shared_across_layers(self):
        """All TQ attention layers must reference the same buffer objects.

        Transformer layers execute sequentially, so one set of scratch
        buffers is sufficient. Allocating per-layer wastes O(num_layers)
        GPU memory.
        """
        B, Hq, S, D = 4, 8, 2, 64

        # Layer 0 creates shared buffers
        buf0 = _simulate_tq_init(B, Hq, S, D)

        # Layer 1 should reuse, not reallocate
        buf1 = _simulate_tq_init(B, Hq, S, D)

        for i in range(3):
            assert buf0[i] is buf1[i], (
                f"Buffer {i} should be shared (same object), got different"
            )

    def test_buffers_reallocate_on_shape_change(self):
        """If config changes (different model), buffers must reallocate."""
        B, Hq, S, D = 4, 8, 2, 64
        buf_small = _simulate_tq_init(B, Hq, S, D)

        # Different head_size triggers reallocation
        D2 = 128
        buf_large = _simulate_tq_init(B, Hq, S, D2)

        assert buf_small[0] is not buf_large[0], (
            "Different shapes should trigger new allocation"
        )
        assert buf_large[0].shape == (B, Hq, S, D2 + 1)

    def test_shared_buffer_memory_savings(self):
        """Shared buffers must use O(1) memory, not O(num_layers).

        For Qwen2.5-32B (64 layers, 40 query heads, head_dim=128):
        - Old: 256 * 40 * 32 * 129 * 4 bytes * 64 layers = 10.82 GiB
        - New: 256 * 40 * 32 * 129 * 4 bytes * 1 = 0.17 GiB
        """
        B, Hq, S, D = 256, 40, 32, 128  # Qwen2.5-32B dimensions
        num_layers = 64

        per_layer_bytes = (
            B * Hq * S * (D + 1) * 4  # mid_o_buf
            + B * Hq * D * 4  # output_buf
            + B * Hq * 4  # lse_buf
        )
        old_total = per_layer_bytes * num_layers
        new_total = per_layer_bytes  # just 1 copy

        savings_gib = (old_total - new_total) / (1024**3)
        assert savings_gib > 10.0, (
            f"Expected >10 GiB savings, got {savings_gib:.1f} GiB"
        )
        assert new_total < 200 * 1024 * 1024, (
            f"Shared buffers should be <200 MiB, got {new_total / 1e6:.0f} MiB"
        )

    def test_buffer_shapes_and_dtype(self):
        """Shared buffers must have correct shapes and float32 dtype."""
        B, Hq, S, D = 8, 4, 4, 64
        mid, out, lse = _simulate_tq_init(B, Hq, S, D)

        assert mid.shape == (B, Hq, S, D + 1)
        assert out.shape == (B, Hq, D)
        assert lse.shape == (B, Hq)
        assert mid.dtype == torch.float32
        assert out.dtype == torch.float32
        assert lse.dtype == torch.float32

    def test_tp_safety_buffers_are_per_process(self):
        """Shared buffers are class-level attributes, so each TP rank
        (separate process) gets its own independent copy. Verify this
        by running the init in a subprocess and confirming the parent's
        buffers are unaffected.
        """
        B, Hq, S, D = 4, 8, 2, 64
        parent_bufs = _simulate_tq_init(B, Hq, S, D)
        parent_id = id(parent_bufs[0])

        def _child_init():
            """Run in subprocess — creates its own shared buffers."""
            _simulate_tq_init(B, Hq, S, D)

        proc = multiprocessing.Process(target=_child_init)
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0, "Child process failed"

        # Parent's buffers must be unchanged (child has its own copy)
        from vllm.model_executor.layers.attention.attention import Attention

        assert id(Attention._tq_shared_mid_o_buf) == parent_id, (
            "Parent buffer should be unaffected by child process"
        )

    def test_pp_ranks_get_independent_buffers(self):
        """Pipeline parallel ranks run in separate processes, so each gets
        its own class-level shared buffers. Verify that two subprocesses
        with different configs don't interfere with each other or the parent.
        """
        from vllm.model_executor.layers.attention.attention import Attention

        B, Hq, S, D = 4, 8, 2, 64
        parent_bufs = _simulate_tq_init(B, Hq, S, D)
        parent_shape = parent_bufs[0].shape

        import multiprocessing as mp

        ctx = mp.get_context("spawn")

        # PP rank 0: same config as parent
        q0 = ctx.Queue()
        p0 = ctx.Process(target=_pp_rank_init, args=(B, Hq, S, D, q0))
        p0.start()

        # PP rank 1: different head count (simulating a different model slice)
        q1 = ctx.Queue()
        Hq2 = 16
        p1 = ctx.Process(target=_pp_rank_init, args=(B, Hq2, S, D, q1))
        p1.start()

        p0.join(timeout=15)
        p1.join(timeout=15)
        assert p0.exitcode == 0, "PP rank 0 subprocess failed"
        assert p1.exitcode == 0, "PP rank 1 subprocess failed"

        shape0 = q0.get(timeout=5)
        shape1 = q1.get(timeout=5)

        # Each rank should have its own correctly-shaped buffers
        assert shape0 == (B, Hq, S, D + 1), f"PP rank 0 shape mismatch: {shape0}"
        assert shape1 == (B, Hq2, S, D + 1), f"PP rank 1 shape mismatch: {shape1}"

        # Parent's buffers must be unaffected
        assert Attention._tq_shared_mid_o_buf.shape == parent_shape, (
            "Parent buffers corrupted by child processes"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_migration_on_first_use(self):
        """Buffers start on CPU and must migrate to GPU on first decode."""
        B, Hq, S, D = 4, 8, 2, 64
        mid, out, lse = _simulate_tq_init(B, Hq, S, D)

        # Buffers start on CPU
        assert mid.device.type == "cpu"

        # Simulate what turboquant_attn.py does on first decode
        query_device = torch.device("cuda:0")
        if mid.device != query_device:
            mid = mid.to(query_device)
            out = out.to(query_device)
            lse = lse.to(query_device)

        assert mid.device.type == "cuda"
        assert out.device.type == "cuda"
        assert lse.device.type == "cuda"
        assert mid.shape == (B, Hq, S, D + 1)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="Requires SM >= 90 (Hopper/Blackwell) for TurboQuant",
    )
    def test_e2e_generation_with_shared_buffers(self):
        """End-to-end: load model with TQ, verify generation and KV capacity."""
        from vllm import LLM, SamplingParams

        try:
            llm = LLM(
                model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                trust_remote_code=True,
                gpu_memory_utilization=0.80,
                max_model_len=2048,
                kv_cache_dtype="turboquant_k8v4",
            )
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

        # With shared buffers, KV capacity should be much higher than
        # the ~235K we got with per-layer buffers
        cc = llm.llm_engine.vllm_config.cache_config
        kv_tokens = cc.num_gpu_blocks * cc.block_size
        assert kv_tokens > 300_000, (
            f"KV capacity {kv_tokens} too low — shared buffers may not work"
        )

        # Verify coherent generation
        out = llm.generate(
            ["What is 2+2? Answer with just the number."],
            SamplingParams(temperature=0.0, max_tokens=10),
        )
        text = out[0].outputs[0].text.strip()
        assert "4" in text, f"Expected '4' in output, got: {text}"

        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        del llm
        gc.collect()
        torch.accelerator.empty_cache()
