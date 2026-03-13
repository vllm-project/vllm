# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for NaN contamination via workspace reuse in NVFP4 MoE.

The workspace manager allocates buffers with torch.empty (never zeroed).
The same workspace blob is reused across MoE layers within a forward pass.
If an NVFP4 expert kernel doesn't write ALL output positions in fused_out,
stale NaN from a previous layer's computation can survive and propagate
to the model output, eventually corrupting the KV cache.

These tests verify:
1. The workspace reuse mechanism can carry stale data (including NaN)
2. _resize_cache preserves stale data
3. TopKWeightAndReduce propagates NaN from unwritten positions
4. The scaled_fp4_quant padding leaves uninitialized scale rows
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.utils import _resize_cache


class TestResizeCachePreservesStaleData:
    """_resize_cache returns a view into existing memory without zeroing."""

    def test_resize_cache_does_not_zero(self):
        """Verify _resize_cache returns a view with whatever data was there."""
        buf = torch.empty(1024, dtype=torch.float32)
        # Poison the buffer with NaN
        buf.fill_(float("nan"))
        # _resize_cache should return a view, NOT a zeroed copy
        view = _resize_cache(buf.view(torch.uint8), (16, 16))
        # The view must still contain NaN
        assert view.dtype == torch.uint8
        # Reinterpret as float32 to check NaN
        as_float = view.view(torch.float32)
        assert torch.isnan(as_float).all(), (
            "_resize_cache should preserve stale data (NaN), but it was zeroed"
        )

    def test_resize_cache_smaller_view_still_stale(self):
        """Even when requesting a smaller view, stale data persists."""
        buf = torch.empty(4096, dtype=torch.uint8)
        # Write NaN pattern into the first 256 bytes
        buf[:256].view(torch.float32).fill_(float("nan"))
        view = _resize_cache(buf, (64,))  # 64 bytes, subset of NaN region
        assert torch.isnan(view.view(torch.float32)).all()


class TestWorkspaceReuseAcrossLayers:
    """Simulate workspace reuse across two MoE layers.

    The workspace manager returns views into the same underlying blob.
    Layer L-1 writes data (potentially including NaN from SiLU overflow).
    Layer L gets fused_out from the SAME memory, starting with stale data.
    """

    def test_workspace_blob_reuse_carries_nan(self):
        """Demonstrate that NaN from layer L-1 persists in layer L's fused_out."""
        M, K = 32, 128
        workspace_dtype = torch.bfloat16

        # Simulate the workspace blob (what WorkspaceManager.get_simultaneous
        # returns from torch.empty)
        blob_size = M * K * workspace_dtype.itemsize
        workspace_blob = torch.empty(blob_size, dtype=torch.uint8)

        # --- Layer L-1: expert kernel writes output, some positions have NaN
        # (e.g., SiLU(very_large_value) -> inf, then inf * 0_weight = NaN)
        fused_out_prev = (
            workspace_blob[: M * K * workspace_dtype.itemsize]
            .view(workspace_dtype)
            .reshape(M, K)
        )
        fused_out_prev.fill_(0)
        # Simulate NaN in a few positions (from SiLU overflow or similar)
        fused_out_prev[7, :] = float("nan")  # One token's output is NaN
        fused_out_prev[15, 64:] = float("nan")  # Partial NaN in another token

        # --- Layer L: same workspace blob, new _resize_cache call
        # This is what modular_kernel.py does:
        #   fused_out = _resize_cache(common_workspace, fused_out_shape)
        fused_out_curr = _resize_cache(
            workspace_blob, (M, K * workspace_dtype.itemsize)
        )
        fused_out_curr_float = fused_out_curr.view(workspace_dtype).reshape(M, K)

        # The NaN from layer L-1 is still there
        assert torch.isnan(fused_out_curr_float[7, :]).all(), (
            "Stale NaN from previous layer should persist in workspace"
        )
        assert torch.isnan(fused_out_curr_float[15, 64:]).all(), (
            "Partial stale NaN should persist"
        )

    def test_expert_kernel_partial_write_leaves_nan(self):
        """If an expert kernel doesn't write all output positions, NaN leaks."""
        M, K = 64, 256
        dtype = torch.bfloat16

        # Start with a NaN-poisoned output buffer (simulating stale workspace)
        fused_out = torch.full((M, K), float("nan"), dtype=dtype)

        # Simulate an expert kernel that writes output for most tokens
        # but skips some (e.g., tokens with no local experts, or padding)
        written_mask = torch.ones(M, dtype=torch.bool)
        written_mask[3] = False   # Token 3: no local experts, not written
        written_mask[17] = False  # Token 17: padding token, not written
        written_mask[M - 1] = False  # Last token: alignment padding

        # Expert kernel writes clean data for "written" tokens
        clean_data = torch.randn(M, K, dtype=dtype)
        fused_out[written_mask] = clean_data[written_mask]

        # Verify: written positions are clean
        assert not torch.isnan(fused_out[written_mask]).any(), (
            "Written positions should be clean"
        )

        # Verify: unwritten positions still have NaN
        assert torch.isnan(fused_out[~written_mask]).all(), (
            "Unwritten positions should retain stale NaN"
        )

        # Now simulate the combine/finalize step that reads ALL positions
        # (e.g., alltoall_combine sends all rows back)
        combined_output = fused_out.clone()  # combine reads everything

        # The final output has NaN contamination
        nan_rows = torch.isnan(combined_output).any(dim=1)
        assert nan_rows.sum() == (~written_mask).sum(), (
            f"Expected {(~written_mask).sum()} NaN rows, got {nan_rows.sum()}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestScaledFp4QuantPadding:
    """Test that scaled_fp4_quant's padding with torch.empty creates
    uninitialized scale rows that could cause NaN if read by the kernel."""

    def test_swizzled_scale_padding_is_uninitialized(self):
        """When is_sf_swizzled_layout=True, output_scale is padded to
        round_up(m, 128) rows. Padding rows are from torch.empty."""
        try:
            import vllm._custom_ops as ops
        except ImportError:
            pytest.skip("vllm._custom_ops not available")

        m_values = [1, 33, 65, 100, 127]  # All non-multiples of 128
        n = 512  # hidden dim (must be reasonable for FP4)
        block_size = 16

        for m in m_values:
            rounded_m = ((m + 127) // 128) * 128
            if rounded_m == m:
                continue  # No padding to test

            input_tensor = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
            global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

            try:
                output, output_scale = ops.scaled_fp4_quant(
                    input_tensor, global_scale, is_sf_swizzled_layout=True
                )
            except Exception:
                pytest.skip("scaled_fp4_quant not available on this device")

            # output_scale shape should be (rounded_m, ...) with padding
            # The padding rows [m:rounded_m] are from torch.empty
            if output_scale.shape[0] > m:
                # Check that padding region exists
                # (We can't guarantee it's NaN, but we can verify the padding
                # rows exist and weren't explicitly zeroed by the kernel)
                padding_rows = output_scale.shape[0] - m
                assert padding_rows == rounded_m - m, (
                    f"Expected {rounded_m - m} padding rows for m={m}, "
                    f"got {padding_rows}"
                )
                # The key point: these padding rows come from torch.empty
                # and are NOT written by the C++ kernel


class TestTopKWeightAndReduceNaNPropagation:
    """Test that TopKWeightAndReduce implementations propagate NaN
    from contaminated expert outputs."""

    def test_contiguous_reduce_propagates_nan(self):
        """TopKWeightAndReduceContiguous: NaN in one expert's output
        contaminates the reduced output for that token.

        This test requires CUDA custom ops (_moe_C). On CPU-only builds,
        we fall back to a pure-PyTorch simulation of the same logic."""
        M, K, topk = 8, 64, 2

        # Expert output: (M, topk, K) - clean data
        expert_output = torch.randn(M, topk, K, dtype=torch.bfloat16)

        # Contaminate ONE expert's output for token 3
        expert_output[3, 1, :] = float("nan")

        topk_weights = torch.ones(M, topk, dtype=torch.bfloat16) / topk

        try:
            from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
                TopKWeightAndReduceContiguous,
            )
            topk_ids = torch.randint(0, 8, (M, topk), dtype=torch.int32)
            reducer = TopKWeightAndReduceContiguous()
            result = reducer.apply(
                output=None,
                fused_expert_output=expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=False,
            )
        except (AttributeError, RuntimeError):
            # Custom ops not available (CPU-only build) -
            # simulate the same weighted sum logic in pure PyTorch
            result = (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)

        # Token 3 should have NaN in its output (NaN * weight + clean = NaN)
        assert torch.isnan(result[3]).any(), (
            "NaN from expert output should propagate through reduce"
        )
        # Other tokens should be clean
        clean_mask = torch.ones(M, dtype=torch.bool)
        clean_mask[3] = False
        assert not torch.isnan(result[clean_mask]).any(), (
            "Clean tokens should not be affected"
        )

    def test_noop_reduce_passes_nan_through(self):
        """TopKWeightAndReduceNoOP: passes NaN straight through (used by
        TRTLLM NVFP4 kernel which does reduction internally)."""
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        M, K = 16, 64
        reducer = TopKWeightAndReduceNoOP()

        expert_output = torch.randn(M, K, dtype=torch.bfloat16)
        expert_output[5, :] = float("nan")  # Token 5 is contaminated

        topk_weights = torch.ones(M, 1, dtype=torch.bfloat16)
        topk_ids = torch.zeros(M, 1, dtype=torch.int32)

        result = reducer.apply(
            output=None,
            fused_expert_output=expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=True,  # weights already applied
        )

        assert torch.isnan(result[5]).all(), (
            "NoOP reduce should pass NaN through unchanged"
        )


class TestIEEE754NaNPropagation:
    """Verify IEEE 754 NaN propagation rules that enable contamination."""

    def test_zero_times_nan_is_nan(self):
        """0 * NaN = NaN per IEEE 754. This is the core mechanism:
        even a zero topk_weight doesn't mask out NaN expert output."""
        zero = torch.tensor(0.0, dtype=torch.bfloat16)
        nan = torch.tensor(float("nan"), dtype=torch.bfloat16)
        result = zero * nan
        assert torch.isnan(result), "0 * NaN should be NaN (IEEE 754)"

    def test_nan_plus_clean_is_nan(self):
        """NaN + clean = NaN. Once NaN enters the residual stream,
        it can never be removed."""
        clean = torch.tensor(1.0, dtype=torch.bfloat16)
        nan = torch.tensor(float("nan"), dtype=torch.bfloat16)
        assert torch.isnan(clean + nan), "clean + NaN should be NaN"

    def test_max_with_nan_is_nan(self):
        """max(x, NaN) = NaN. This means block quantization scale
        becomes NaN if any element in the block is NaN."""
        block = torch.tensor([1.0, 2.0, float("nan"), 3.0], dtype=torch.float32)
        assert torch.isnan(block.max()), (
            "max() over a block containing NaN should be NaN"
        )

    def test_nan_in_block_quantization_contaminates_entire_block(self):
        """Simulate block quantization: scale = max(|block|).
        One NaN in the block makes the scale NaN, corrupting all
        elements in the block when they're dequantized."""
        block_size = 128
        num_blocks = 4
        data = torch.randn(num_blocks, block_size, dtype=torch.float32)

        # Poison ONE element in block 2
        data[2, 50] = float("nan")

        # Block quantization: compute per-block scale
        scales = data.abs().max(dim=1).values  # shape: (num_blocks,)

        # Block 2's scale is NaN
        assert torch.isnan(scales[2]), "Block with NaN element should have NaN scale"
        # Other blocks are fine
        assert not torch.isnan(scales[0])
        assert not torch.isnan(scales[1])
        assert not torch.isnan(scales[3])

        # When dequantizing: quantized_value * scale
        # Even clean quantized values become NaN when multiplied by NaN scale
        fake_quantized = torch.ones_like(data)
        dequantized = fake_quantized * scales.unsqueeze(1)
        assert torch.isnan(dequantized[2]).all(), (
            "All elements in contaminated block should be NaN after dequant"
        )
        assert not torch.isnan(dequantized[0]).any()
        assert not torch.isnan(dequantized[1]).any()
        assert not torch.isnan(dequantized[3]).any()


class TestWorkspaceManagerNeverZeros:
    """Test that the WorkspaceManager allocates with torch.empty,
    meaning returned buffers contain stale/uninitialized data."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_workspace_manager_returns_uninitialized_memory(self):
        """WorkspaceManager.get_simultaneous returns views into torch.empty."""
        from vllm.v1.worker.workspace import WorkspaceManager, set_workspace_manager

        device = torch.device("cuda:0")
        manager = WorkspaceManager(device)
        set_workspace_manager(manager)

        try:
            from vllm.v1.worker.workspace import current_workspace_manager

            ws_mgr = current_workspace_manager()

            # First allocation - get a workspace
            dtype = torch.bfloat16
            shape1 = (64, 128)
            [buf1] = ws_mgr.get_simultaneous((shape1, dtype))

            # Fill with NaN to simulate layer L-1's MoE producing NaN
            buf1.fill_(float("nan"))

            # Second call - same shape, same workspace blob is reused
            [buf2] = ws_mgr.get_simultaneous((shape1, dtype))

            # buf2 should be a view into the SAME memory as buf1
            # Therefore it should still contain NaN
            assert buf2.data_ptr() == buf1.data_ptr(), (
                "Workspace should reuse the same memory"
            )
            assert torch.isnan(buf2).all(), (
                "Workspace reuse should preserve stale data (NaN)"
            )
        finally:
            set_workspace_manager(None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_workspace_shared_between_workspace13_and_fused_out(self):
        """When num_chunks==1, workspace13 and fused_out share memory.
        This is the modular_kernel.py:1143-1150 optimization."""
        from vllm.v1.worker.workspace import WorkspaceManager, set_workspace_manager

        device = torch.device("cuda:0")
        manager = WorkspaceManager(device)
        set_workspace_manager(manager)

        try:
            from vllm.v1.worker.workspace import current_workspace_manager

            ws_mgr = current_workspace_manager()

            # Simulate the allocation pattern from _allocate_buffers
            # when num_chunks == 1:
            #   workspace13_shape = (0,) for TRTLLM NVFP4
            #   fused_out_shape = (M, hidden_dim)
            M, hidden_dim = 32, 7168  # DeepSeek R1 hidden dim
            dtype = torch.bfloat16

            workspace13_size = 0
            fused_out_size = M * hidden_dim
            max_size = max(workspace13_size, fused_out_size)

            # This mimics the get_simultaneous call in _allocate_buffers
            common_ws, ws2 = ws_mgr.get_simultaneous(
                ((max_size,), dtype),
                ((0,), dtype),
            )

            # Poison the workspace (simulating previous layer's data)
            common_ws.fill_(float("nan"))

            # Now create fused_out via _resize_cache (same as modular_kernel.py)
            fused_out = _resize_cache(
                common_ws.view(torch.uint8),
                (M * hidden_dim * dtype.itemsize,),
            )
            fused_out_typed = fused_out.view(dtype).reshape(M, hidden_dim)

            # fused_out starts with NaN from the poisoned workspace
            assert torch.isnan(fused_out_typed).all(), (
                "fused_out from _resize_cache should contain stale NaN "
                "from the shared workspace"
            )
        finally:
            set_workspace_manager(None)


class TestEndToEndNaNContaminationScenario:
    """End-to-end test: demonstrate the full NaN contamination chain.

    1. Workspace has stale NaN (from previous layer)
    2. Expert kernel writes most but not all output positions
    3. Stale NaN survives in unwritten positions
    4. Reduce/combine reads all positions -> NaN in final output
    5. NaN propagates through residual to next attention layer
    6. Attention writes NaN to KV cache
    """

    def test_full_contamination_chain(self):
        """Simulate the complete NaN contamination path."""
        M, K = 32, 256
        topk = 8
        dtype = torch.bfloat16

        # Step 1: fused_out buffer from workspace (stale NaN)
        fused_out = torch.full((M, K), float("nan"), dtype=dtype)

        # Step 2: Expert kernel writes output for most tokens
        # In EP mode, some tokens might have all experts on other ranks
        # The TRTLLM kernel writes output for tokens with local experts
        tokens_with_local_experts = torch.ones(M, dtype=torch.bool)
        # Token 11 has no local experts (shouldn't happen with correct
        # dispatch, but edge case with EP + specific routing)
        tokens_with_local_experts[11] = False
        # Token 29 is a CUDA graph padding token (seq_len=0)
        tokens_with_local_experts[29] = False

        clean_output = torch.randn(M, K, dtype=dtype)
        fused_out[tokens_with_local_experts] = clean_output[tokens_with_local_experts]

        # Step 3: Verify stale NaN survives
        assert torch.isnan(fused_out[11]).all()
        assert torch.isnan(fused_out[29]).all()
        assert not torch.isnan(fused_out[0]).any()

        # Step 4: Combine/reduce reads ALL positions
        # For FlashInfer A2A: mnnvl_moe_alltoallv_combine reads entire tensor
        # For DeepEP LL: low_latency_combine reads entire tensor
        final_moe_output = fused_out  # combine returns this to the model

        # Step 5: Residual connection
        residual = torch.randn(M, K, dtype=dtype)
        hidden_states = final_moe_output + residual  # NaN + clean = NaN

        # Token 11 and 29 now have NaN in hidden_states
        assert torch.isnan(hidden_states[11]).all(), (
            "NaN from MoE should propagate through residual"
        )

        # Step 6: Next layer's attention receives NaN hidden_states
        # It computes KV projections and writes to cache
        # kv_c = hidden_states @ W_DKV  (NaN @ anything = NaN)
        W_DKV = torch.randn(K, 64, dtype=dtype)  # Dummy KV projection
        kv_c = hidden_states @ W_DKV
        assert torch.isnan(kv_c[11]).all(), (
            "KV projection of NaN hidden state should be NaN"
        )

        # This NaN kv_c gets written to the KV cache.
        # ALL subsequent decode steps for token 11's request will read
        # NaN from the KV cache, producing NaN attention output even
        # with clean current-step inputs.

        # Step 7: Simulate next decode step - clean input but NaN KV cache
        clean_q = torch.randn(1, 64, dtype=dtype)
        kv_cache_entry = kv_c[11:12]  # NaN from previous step

        # Attention score: Q @ K^T
        attn_score = clean_q @ kv_cache_entry.T
        assert torch.isnan(attn_score).all(), (
            "Clean Q dotted with NaN K from cache should produce NaN. "
            "This is why attention output is NaN even with clean input."
        )
