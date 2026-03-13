# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test whether uninitialized padding rows in NVFP4 swizzled block scales
can contaminate real rows' GEMM output.

Background:
- scaled_fp4_quant with is_sf_swizzled_layout=True pads output_scale
  to round_up(m, 128) rows using torch.empty (never written for padding rows)
- The CUTLASS/flashinfer mm_fp4 kernel operates on 128-row tiles
- If padding scale rows contain NaN, does it corrupt real rows in the same tile?

This is the suspected root cause for NaN corruption in NVFP4 models
with CUDA graphs, where most capture sizes (1, 2, 4, 8, 16, 32, ...)
are not multiples of 128.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)


def round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def has_flashinfer():
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


def has_blackwell():
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


class TestScaledFp4QuantPaddingShape:
    """Verify that scaled_fp4_quant produces padded scale tensors."""

    @pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 255, 256])
    def test_scale_padding_shape(self, m):
        """Check that swizzled scales are padded to multiple of 128."""
        from vllm._custom_ops import scaled_fp4_quant

        hidden_dim = 512  # must be multiple of 16
        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        output, output_scale = scaled_fp4_quant(
            x, global_scale, is_sf_swizzled_layout=True
        )

        # output should have m rows
        assert output.shape[0] == m

        # output_scale should have rounded_m rows (viewed as float8)
        rounded_m = round_up(m, 128)
        scale_n = hidden_dim // 16
        rounded_n = round_up(scale_n, 4)
        expected_scale_elements = rounded_m * rounded_n
        assert output_scale.numel() == expected_scale_elements, (
            f"m={m}: expected {expected_scale_elements} scale elements "
            f"(rounded_m={rounded_m}), got {output_scale.numel()}"
        )

        if m % 128 != 0:
            # There ARE padding rows that the kernel never writes
            padding_elements = (rounded_m - m) * rounded_n
            assert padding_elements > 0, (
                f"m={m}: expected padding rows but got none"
            )


class TestScalePaddingContamination:
    """Test whether poisoning padding rows in block scales affects
    real rows' GEMM output via flashinfer mm_fp4."""

    @pytest.mark.skipif(not has_flashinfer(), reason="Requires flashinfer")
    @pytest.mark.skipif(not has_blackwell(), reason="Requires Blackwell GPU")
    @pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("backend", ["cutlass", "trtllm"])
    def test_poison_padding_scales_nan(self, m, backend):
        """Poison padding scale rows with NaN and check if real output
        rows are contaminated.

        This simulates what happens with CUDA graphs: the padding rows
        of the scale tensor are allocated once with torch.empty and never
        written by the quantization kernel. If previous computation left
        NaN in that memory, the GEMM kernel's 128-row tiles would read
        NaN scales for padding rows.
        """
        from vllm._custom_ops import scaled_fp4_quant
        from flashinfer import mm_fp4 as flashinfer_mm_fp4

        hidden_dim = 512
        output_dim = 256

        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        # Quantize input
        x_fp4, x_blockscale = scaled_fp4_quant(
            x, global_scale, is_sf_swizzled_layout=True
        )

        # Create fake quantized weights (FP4 packed as uint8)
        w_fp4 = torch.randint(
            0, 255, (output_dim, hidden_dim // 2),
            dtype=torch.uint8, device="cuda"
        )
        w_scale = torch.ones(
            output_dim, hidden_dim // 16,
            dtype=torch.float8_e4m3fn, device="cuda"
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        use_8x4 = (backend == "trtllm" and m <= 32)

        # --- Baseline: run GEMM with clean (original) scales ---
        clean_scale = x_blockscale.clone()
        result_clean = flashinfer_mm_fp4(
            x_fp4,
            w_fp4.t(),
            clean_scale,
            w_scale.t(),
            alpha,
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
        )

        # --- Poison: fill padding rows of scale tensor with NaN ---
        # The scale tensor has rounded_m rows. We poison rows [m:rounded_m].
        rounded_m = round_up(m, 128)
        if rounded_m == m:
            pytest.skip(f"m={m} is already multiple of 128, no padding")

        scale_n = hidden_dim // 16
        rounded_n = round_up(scale_n, 4)

        # View the scale as its raw int32 swizzled layout to poison padding
        poisoned_scale = x_blockscale.clone()
        # The scale tensor is float8_e4m3fn with shape (rounded_m * rounded_n,)
        # Reshape to (rounded_m, rounded_n) to identify padding rows
        scale_2d = poisoned_scale.view(torch.uint8).reshape(rounded_m, rounded_n)
        # Set padding rows to NaN pattern (0x7F is NaN in float8_e4m3fn)
        scale_2d[m:, :] = 0x7F
        poisoned_scale = scale_2d.reshape(-1).view(torch.float8_e4m3fn)

        result_poisoned = flashinfer_mm_fp4(
            x_fp4,
            w_fp4.t(),
            poisoned_scale,
            w_scale.t(),
            alpha,
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
        )

        # Check: do real rows (0..m-1) have NaN in the poisoned result?
        has_nan_real = torch.isnan(result_poisoned[:m]).any().item()
        has_nan_clean = torch.isnan(result_clean[:m]).any().item()

        # The clean result should never have NaN
        assert not has_nan_clean, (
            f"m={m}, backend={backend}: clean GEMM produced NaN "
            f"(this is a kernel bug, not a padding issue)"
        )

        if has_nan_real:
            # Count how many real rows are contaminated
            nan_rows = torch.isnan(result_poisoned[:m]).any(dim=-1)
            num_nan_rows = nan_rows.sum().item()
            print(
                f"CONTAMINATION CONFIRMED: m={m}, backend={backend}: "
                f"{num_nan_rows}/{m} real rows have NaN from poisoned "
                f"padding scales"
            )

        # This is the key assertion: if this fails, padding NaN
        # DOES contaminate real rows
        assert not has_nan_real, (
            f"m={m}, backend={backend}: NaN in padding scale rows "
            f"contaminated {torch.isnan(result_poisoned[:m]).any(dim=-1).sum().item()}/{m} "
            f"real output rows! This confirms the NVFP4 CUDA graph NaN bug."
        )

    @pytest.mark.skipif(not has_flashinfer(), reason="Requires flashinfer")
    @pytest.mark.skipif(not has_blackwell(), reason="Requires Blackwell GPU")
    @pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("backend", ["cutlass", "trtllm"])
    def test_poison_padding_scales_large_values(self, m, backend):
        """Poison padding scale rows with large (but finite) values.

        Even if NaN doesn't leak directly, large scale values in padding
        rows could cause intermediate overflow -> NaN in the kernel.
        """
        from vllm._custom_ops import scaled_fp4_quant
        from flashinfer import mm_fp4 as flashinfer_mm_fp4

        hidden_dim = 512
        output_dim = 256

        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        x_fp4, x_blockscale = scaled_fp4_quant(
            x, global_scale, is_sf_swizzled_layout=True
        )

        w_fp4 = torch.randint(
            0, 255, (output_dim, hidden_dim // 2),
            dtype=torch.uint8, device="cuda"
        )
        w_scale = torch.ones(
            output_dim, hidden_dim // 16,
            dtype=torch.float8_e4m3fn, device="cuda"
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        rounded_m = round_up(m, 128)
        if rounded_m == m:
            pytest.skip(f"m={m} is already multiple of 128, no padding")

        scale_n = hidden_dim // 16
        rounded_n = round_up(scale_n, 4)

        use_8x4 = (backend == "trtllm" and m <= 32)

        # Poison padding with max float8 value (0x7E = 448.0 in e4m3fn)
        poisoned_scale = x_blockscale.clone()
        scale_2d = poisoned_scale.view(torch.uint8).reshape(rounded_m, rounded_n)
        scale_2d[m:, :] = 0x7E  # max finite value in float8_e4m3fn
        poisoned_scale = scale_2d.reshape(-1).view(torch.float8_e4m3fn)

        result_poisoned = flashinfer_mm_fp4(
            x_fp4,
            w_fp4.t(),
            poisoned_scale,
            w_scale.t(),
            alpha,
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
        )

        has_nan_real = torch.isnan(result_poisoned[:m]).any().item()
        if has_nan_real:
            nan_rows = torch.isnan(result_poisoned[:m]).any(dim=-1)
            print(
                f"OVERFLOW CONTAMINATION: m={m}, backend={backend}: "
                f"{nan_rows.sum().item()}/{m} real rows have NaN from "
                f"large padding scales causing overflow"
            )

        assert not has_nan_real, (
            f"m={m}, backend={backend}: large padding scale values caused "
            f"NaN in real output rows via overflow"
        )

    @pytest.mark.skipif(not has_flashinfer(), reason="Requires flashinfer")
    @pytest.mark.skipif(not has_blackwell(), reason="Requires Blackwell GPU")
    @pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("backend", ["cutlass", "trtllm"])
    def test_poison_padding_activations_nan(self, m, backend):
        """Poison padding rows of the quantized activation tensor itself.

        With CUDA graphs, x_fp4 has exactly m rows, but what if the
        kernel reads beyond m due to the 128-row tile size?
        We extend x_fp4 to rounded_m rows with NaN-like patterns and
        check if real rows are affected.
        """
        from vllm._custom_ops import scaled_fp4_quant
        from flashinfer import mm_fp4 as flashinfer_mm_fp4

        hidden_dim = 512
        output_dim = 256

        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        x_fp4, x_blockscale = scaled_fp4_quant(
            x, global_scale, is_sf_swizzled_layout=True
        )

        rounded_m = round_up(m, 128)
        if rounded_m == m:
            pytest.skip(f"m={m} is already multiple of 128, no padding")

        # Extend x_fp4 to rounded_m rows, padding with 0xFF pattern
        x_fp4_padded = torch.zeros(
            rounded_m, hidden_dim // 2, dtype=torch.uint8, device="cuda"
        )
        x_fp4_padded[:m] = x_fp4
        x_fp4_padded[m:] = 0xFF  # all-ones pattern in FP4

        w_fp4 = torch.randint(
            0, 255, (output_dim, hidden_dim // 2),
            dtype=torch.uint8, device="cuda"
        )
        w_scale = torch.ones(
            output_dim, hidden_dim // 16,
            dtype=torch.float8_e4m3fn, device="cuda"
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        use_8x4 = (backend == "trtllm" and m <= 32)

        # Run with padded activations (rounded_m rows) + full scale tensor
        result = flashinfer_mm_fp4(
            x_fp4_padded,
            w_fp4.t(),
            x_blockscale,  # already has rounded_m rows of scales
            w_scale.t(),
            alpha,
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
        )

        # Check real rows only
        has_nan_real = torch.isnan(result[:m]).any().item()

        # Also run with just m rows as baseline
        result_clean = flashinfer_mm_fp4(
            x_fp4,
            w_fp4.t(),
            x_blockscale,
            w_scale.t(),
            alpha,
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=use_8x4,
            backend=backend,
        )

        # Check if results for real rows differ
        if not has_nan_real:
            max_diff = (result[:m].float() - result_clean[:m].float()).abs().max().item()
            if max_diff > 1e-3:
                print(
                    f"WARNING: m={m}, backend={backend}: padding activations "
                    f"changed real rows' output by max_diff={max_diff}"
                )

        assert not has_nan_real, (
            f"m={m}, backend={backend}: garbage padding activations "
            f"caused NaN in real output rows"
        )


class TestCudaGraphScaleReuse:
    """Simulate the CUDA graph scenario: scale tensor allocated once,
    reused across replays without reinitializing padding rows."""

    @pytest.mark.skipif(not has_flashinfer(), reason="Requires flashinfer")
    @pytest.mark.skipif(not has_blackwell(), reason="Requires Blackwell GPU")
    @pytest.mark.parametrize("m", [1, 8, 32])
    @pytest.mark.parametrize("backend", ["cutlass"])
    def test_scale_reuse_accumulates_contamination(self, m, backend):
        """Simulate multiple CUDA graph replays.

        1. Allocate scale tensor once (like CUDA graph capture)
        2. Run quantization + GEMM multiple times (like replays)
        3. Between replays, write NaN to padding rows
           (simulating another kernel leaving NaN in that memory)
        4. Check if NaN accumulates and eventually hits real rows
        """
        from vllm._custom_ops import scaled_fp4_quant
        from flashinfer import mm_fp4 as flashinfer_mm_fp4

        hidden_dim = 512
        output_dim = 256
        rounded_m = round_up(m, 128)
        if rounded_m == m:
            pytest.skip(f"m={m} is already multiple of 128, no padding")

        scale_n = hidden_dim // 16
        rounded_n = round_up(scale_n, 4)

        w_fp4 = torch.randint(
            0, 255, (output_dim, hidden_dim // 2),
            dtype=torch.uint8, device="cuda"
        )
        w_scale = torch.ones(
            output_dim, hidden_dim // 16,
            dtype=torch.float8_e4m3fn, device="cuda"
        )
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        use_8x4 = (backend == "trtllm" and m <= 32)

        # Pre-allocate the scale tensor (simulating CUDA graph capture)
        # This is what torch.empty does inside scaled_fp4_quant
        persistent_scale = torch.empty(
            (rounded_m, rounded_n // 4), dtype=torch.int32, device="cuda"
        )

        contaminated = False
        for replay in range(10):
            # Generate new input each replay
            x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device="cuda")

            # Quantize - but write scales into our persistent buffer
            # (In real CUDA graphs, scaled_fp4_quant's torch.empty
            #  returns the same memory each replay)
            x_fp4, fresh_scale = scaled_fp4_quant(
                x, global_scale, is_sf_swizzled_layout=True
            )

            # Copy the real scale rows into our persistent buffer
            # (simulating the C++ kernel writing only m rows)
            fresh_scale_raw = fresh_scale.view(torch.uint8)
            persistent_raw = persistent_scale.view(torch.uint8)
            # Only copy the real rows' worth of data
            real_bytes = m * rounded_n  # bytes for real rows
            persistent_raw[:real_bytes] = fresh_scale_raw[:real_bytes]
            # Padding rows are NOT touched - they retain previous content

            # Poison padding rows (simulating stale NaN from other computation)
            scale_2d = persistent_raw.reshape(rounded_m, rounded_n)
            scale_2d[m:, :] = 0x7F  # NaN in float8_e4m3fn

            reuse_scale = persistent_raw.view(torch.float8_e4m3fn)

            result = flashinfer_mm_fp4(
                x_fp4,
                w_fp4.t(),
                reuse_scale,
                w_scale.t(),
                alpha,
                torch.bfloat16,
                block_size=16,
                use_8x4_sf_layout=use_8x4,
                backend=backend,
            )

            if torch.isnan(result[:m]).any():
                nan_rows = torch.isnan(result[:m]).any(dim=-1).sum().item()
                print(
                    f"CUDA GRAPH CONTAMINATION at replay {replay}: "
                    f"m={m}, backend={backend}: "
                    f"{nan_rows}/{m} real rows have NaN"
                )
                contaminated = True
                break

        assert not contaminated, (
            f"m={m}, backend={backend}: CUDA graph scale reuse with "
            f"NaN padding caused contamination of real output rows"
        )
