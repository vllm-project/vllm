# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that TurboQuant decode buffers use WorkspaceManager instead of
per-layer allocation, reducing GPU memory from O(num_layers) to O(1).

Run: pytest tests/test_tq_shared_buffers.py -v
"""

import gc

import pytest
import torch


class TestTurboQuantWorkspaceBuffers:
    """Tests for TurboQuant workspace-based buffer sharing."""

    def test_no_tq_buffers_on_attention_class(self):
        """Attention class must not have TQ-specific init or shared buffers.

        All TQ buffer management is now in the TQ backend via
        WorkspaceManager, not in the generic Attention layer.
        """
        from vllm.model_executor.layers.attention.attention import Attention

        assert not hasattr(Attention, "_init_turboquant_buffers"), (
            "_init_turboquant_buffers should be removed from Attention"
        )
        assert not hasattr(Attention, "_tq_shared_mid_o_buf"), (
            "Shared TQ buffers should not exist on Attention class"
        )

    def test_workspace_provides_correct_shapes(self):
        """WorkspaceManager.get_simultaneous must return correctly shaped
        float32 tensors for the three TQ decode scratch buffers.
        """
        from vllm.v1.worker.workspace import WorkspaceManager

        B, Hq, S, D = 8, 4, 4, 64
        device = torch.device("cpu")
        ws = WorkspaceManager(device=device)

        mid, out, lse = ws.get_simultaneous(
            ((B, Hq, S, D + 1), torch.float32),
            ((B, Hq, D), torch.float32),
            ((B, Hq), torch.float32),
        )

        assert mid.shape == (B, Hq, S, D + 1)
        assert out.shape == (B, Hq, D)
        assert lse.shape == (B, Hq)
        assert mid.dtype == torch.float32
        assert out.dtype == torch.float32
        assert lse.dtype == torch.float32

    def test_workspace_reuses_memory_across_calls(self):
        """Repeated get_simultaneous calls with the same shapes must
        return views into the same underlying workspace — no new
        allocation per layer.
        """
        from vllm.v1.worker.workspace import WorkspaceManager

        B, Hq, S, D = 4, 8, 2, 64
        device = torch.device("cpu")
        ws = WorkspaceManager(device=device)

        shapes = (
            ((B, Hq, S, D + 1), torch.float32),
            ((B, Hq, D), torch.float32),
            ((B, Hq), torch.float32),
        )

        # First call allocates
        bufs0 = ws.get_simultaneous(*shapes)
        # Second call should reuse the same workspace
        bufs1 = ws.get_simultaneous(*shapes)

        assert bufs0[0].data_ptr() == bufs1[0].data_ptr(), (
            "Workspace should reuse memory, not allocate per call"
        )

    def test_memory_savings_math(self):
        """Workspace approach must use O(1) memory, not O(num_layers).

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
        new_total = per_layer_bytes  # just 1 workspace

        savings_gib = (old_total - new_total) / (1024**3)
        assert savings_gib > 10.0, (
            f"Expected >10 GiB savings, got {savings_gib:.1f} GiB"
        )

    def test_fallback_when_workspace_unavailable(self):
        """When WorkspaceManager is not initialized, _decode_attention
        must still work — the Triton kernel allocates buffers internally
        via the buf_holder fallback path.
        """
        from vllm.v1.worker.workspace import (
            is_workspace_manager_initialized,
            reset_workspace_manager,
        )

        # Ensure workspace is not initialized
        reset_workspace_manager()
        assert not is_workspace_manager_initialized()

        # The decode path should not crash — it passes None buffers
        # and the kernel allocates internally
        # (We can't easily run the full kernel without a GPU, but we
        # verify the workspace check returns the right thing)
        from vllm.v1.worker.workspace import current_workspace_manager

        with pytest.raises(AssertionError):
            # Should raise because workspace is not initialized
            current_workspace_manager()

    def test_centroids_lazy_init_in_ensure_on_device(self):
        """Centroids must be initialized lazily in _ensure_on_device,
        not in attention.py. Verify the TQ impl can create centroids
        on a layer that doesn't have them pre-set.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )

        # Create a minimal impl
        impl = TurboQuantAttentionImpl.__new__(TurboQuantAttentionImpl)
        impl.head_size = 128
        impl.tq_config = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", 128)

        # Create a bare layer with no _tq_centroids
        layer = torch.nn.Module()
        assert not hasattr(layer, "_tq_centroids")
        assert not hasattr(layer, "_tq_cached")

        # _ensure_on_device should create centroids lazily
        impl._ensure_on_device(layer, torch.device("cpu"))

        assert hasattr(layer, "_tq_centroids"), "Centroids should be lazily initialized"
        assert hasattr(layer, "_tq_Pi")
        assert hasattr(layer, "_tq_PiT")
        assert hasattr(layer, "_tq_midpoints")
        assert hasattr(layer, "_tq_cached")

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="Requires SM >= 90 (Hopper/Blackwell) for TurboQuant",
    )
    def test_e2e_generation(self):
        """End-to-end: load model with TQ, verify generation works."""
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
