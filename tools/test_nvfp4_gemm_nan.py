"""Test whether the NVFP4 GEMM path contaminates clean tokens
when other tokens in the batch have NaN input.

Replicates the actual inference path:
  bf16 input -> scaled_fp4_quant -> flashinfer/cutlass FP4 GEMM -> bf16 output

Run on a GPU with SM >= 100 (Blackwell) for flashinfer-cutlass,
or SM >= 90 (Hopper) for vllm-cutlass fallback:
    python tools/test_nvfp4_gemm_nan.py
"""
import torch

# Load vllm ops
import vllm._C  # noqa: F401
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    prepare_weights_for_nvfp4_cutlass,
    select_nvfp4_linear_backend,
    slice_nvfp4_output,
    swizzle_blockscale,
)

# Try to import flashinfer backend
try:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer
    HAS_FLASHINFER = has_flashinfer()
except ImportError:
    HAS_FLASHINFER = False


def make_fake_fp4_weight(output_size, input_size, device):
    """Create a fake NVFP4 weight with random data and valid scales."""
    # FP4 weights: 2 values packed per uint8 byte, so shape is [N, K//2]
    k_packed = input_size // 2
    weight_fp4 = torch.randint(0, 256, (output_size, k_packed),
                               dtype=torch.uint8, device=device)

    # Block scales: one fp8 scale per 16 elements along K dim
    # Shape: [N, K//16]
    num_blocks_k = input_size // 16
    weight_scale = torch.ones(output_size, num_blocks_k,
                              dtype=torch.float8_e4m3fn, device=device)

    # Global scale (scalar)
    weight_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Alpha = weight_global_scale (used as scaling in the GEMM)
    alpha = weight_global_scale.clone()

    # Input global scale inverse
    input_global_scale_inv = torch.tensor(1.0, dtype=torch.float32,
                                          device=device)

    return weight_fp4, weight_scale, weight_global_scale, alpha, input_global_scale_inv


def run_nvfp4_gemm(x_bf16, weight_fp4, weight_scale_swizzled, alpha,
                   input_global_scale_inv, output_size, backend_name,
                   weights_padding_cols=0):
    """Run the full NVFP4 GEMM path: quantize input + matmul."""
    # Step 1: Quantize bf16 input to FP4
    x_fp4, x_blockscale = scaled_fp4_quant(
        x_bf16, input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=backend_name,
    )

    # Step 2: Pad activations if needed
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    # Step 3: Run the GEMM
    mm_args = (x_fp4, weight_fp4, x_blockscale, weight_scale_swizzled,
               alpha, torch.bfloat16)

    if backend_name.startswith("flashinfer-"):
        backend_short = backend_name[len("flashinfer-"):]
        out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_short)
    else:
        from vllm._custom_ops import cutlass_scaled_fp4_mm
        out = cutlass_scaled_fp4_mm(*mm_args)

    # Step 4: Slice output
    out = slice_nvfp4_output(out, output_size)
    return out


def test_nan_contamination():
    torch.manual_seed(42)
    device = "cuda"

    # DeepSeek V3 fused A projection dimensions
    hidden_size = 7168
    output_size = 2112  # q_lora_rank(1536) + kv_lora_rank(512) + rope(64)

    # Select backend (same logic as production)
    backend = select_nvfp4_linear_backend()
    backend_name = backend.value
    print(f"Using NVFP4 backend: {backend_name}")

    # Create fake FP4 weights
    (weight_fp4, weight_scale, weight_global_scale,
     alpha, input_global_scale_inv) = make_fake_fp4_weight(
        output_size, hidden_size, device)

    # Prepare weights for the backend
    weight_prepared, weight_scale_swizzled, weights_padding_cols = \
        prepare_weights_for_nvfp4_cutlass(weight_fp4, weight_scale)

    # Build test patterns: CUDA graph scenario
    patterns = []
    for batch_size in [8, 16]:
        for num_real in [1, 2, 3]:
            patterns.append((batch_size, f"first_{num_real}_real"))
        patterns.append((batch_size, "none"))

    for batch_size in [1, 2, 4, 7, 8, 16]:
        for p in ["none", "all_but_first", "all_but_last", "even"]:
            patterns.append((batch_size, p))

    for batch_size, nan_pattern in patterns:
        # Clean bf16 input
        hidden = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16,
                             device=device)

        # Reference output from clean input
        ref_output = run_nvfp4_gemm(
            hidden, weight_prepared, weight_scale_swizzled, alpha,
            input_global_scale_inv, output_size, backend_name,
            weights_padding_cols)

        # Now poison some tokens with NaN
        poisoned = hidden.clone()
        if nan_pattern == "none":
            pass
        elif nan_pattern.startswith("first_"):
            num_real = int(nan_pattern.split("_")[1])
            poisoned[num_real:] = float("nan")
        elif nan_pattern == "all_but_first":
            if batch_size > 1:
                poisoned[1:] = float("nan")
        elif nan_pattern == "all_but_last":
            if batch_size > 1:
                poisoned[:-1] = float("nan")
        elif nan_pattern == "even":
            poisoned[::2] = float("nan")

        # Run GEMM with poisoned input
        test_output = run_nvfp4_gemm(
            poisoned, weight_prepared, weight_scale_swizzled, alpha,
            input_global_scale_inv, output_size, backend_name,
            weights_padding_cols)

        # Check: clean tokens should produce the same output
        for tok in range(batch_size):
            is_poisoned = poisoned[tok].isnan().any().item()
            out_has_nan = test_output[tok].isnan().any().item()

            if is_poisoned:
                assert out_has_nan, (
                    f"batch={batch_size} pattern={nan_pattern} tok={tok}: "
                    f"NaN input but clean output!"
                )
            else:
                if out_has_nan:
                    print(f"FAIL batch={batch_size} pattern={nan_pattern} "
                          f"tok={tok}: clean input but NaN output! "
                          f"CROSS-CONTAMINATION DETECTED")
                    max_diff = (test_output[tok].float()
                                - ref_output[tok].float()).abs().max()
                    print(f"  max_diff from ref: {max_diff}")
                    return False
                else:
                    match = torch.allclose(test_output[tok], ref_output[tok])
                    if not match:
                        max_diff = (test_output[tok].float()
                                    - ref_output[tok].float()).abs().max()
                        print(f"WARN batch={batch_size} "
                              f"pattern={nan_pattern} tok={tok}: "
                              f"output differs from ref, "
                              f"max_diff={max_diff}")

        print(f"OK batch={batch_size} pattern={nan_pattern}")

    print("\nAll tests passed — no cross-token NaN contamination")
    return True


if __name__ == "__main__":
    test_nan_contamination()
