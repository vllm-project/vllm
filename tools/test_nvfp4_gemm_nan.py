"""Test whether the fused_qkv_a_proj NVFP4 GEMM contaminates clean tokens
when other tokens in the batch have NaN input.

Targets the EXACT decode path for DeepSeek-R1-0528-NVFP4-v2:
  DeepSeekV2FusedQkvAProjLinear.forward()
    -> _use_min_latency_gemm = False (weights are uint8 FP4, not bf16)
    -> MergedColumnParallelLinear.forward()
      -> ModelOptNvFp4LinearMethod.apply()
        -> apply_nvfp4_linear(flashinfer-cutlass on GB200)
          -> scaled_fp4_quant(bf16 -> fp4)
          -> flashinfer_scaled_fp4_mm(fp4 x fp4 -> bf16)

Loads real weights from the model checkpoint (layer 0).

SELF-CONTAINED: does NOT import from vllm._custom_ops (blocked by
cache_nan_ext JIT extension). All needed functions are inlined.

Run on GB200:
    python tools/test_nvfp4_gemm_nan.py
"""
import glob
import os

import torch
from torch.nn import Parameter

# Load the _C extension directly (avoids _custom_ops.py import chain)
import vllm._C  # noqa: F401

try:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
except ImportError:
    flashinfer_scaled_fp4_mm = None


# ── Helpers inlined from vllm (avoids cache_nan_ext import) ──────────────


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def create_fp4_scale_tensor(m, n, device, is_sf_swizzled_layout):
    block_size = 16
    if is_sf_swizzled_layout:
        rounded_m = round_up(m, 128)
        scale_n = n // block_size
        rounded_n = round_up(scale_n, 4)
        return torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        return torch.zeros((m, n // block_size), device=device, dtype=torch.uint8)


def create_fp4_output_tensors(m, n, device, is_sf_swizzled_layout):
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    output_scale = create_fp4_scale_tensor(m, n, device, is_sf_swizzled_layout)
    return output, output_scale


def scaled_fp4_quant(input, input_global_scale, is_sf_swizzled_layout=True):
    assert input.ndim >= 1
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    output, output_scale = create_fp4_output_tensors(
        m, n, input.device, is_sf_swizzled_layout
    )
    torch.ops._C.scaled_fp4_quant.out(
        input,
        input_global_scale,
        is_sf_swizzled_layout,
        output=output,
        output_scale=output_scale,
    )
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def swizzle_blockscale(scale):
    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)
    B, M, K = scale.shape
    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)
    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()
    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def pad_nvfp4_weight_for_cutlass(weight, alignment=32):
    weight_current_rows = weight.shape[0]
    if weight_current_rows % alignment != 0:
        total_rows = round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()
    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2
    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes
    return weight, weights_padding_bytes


def pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_bytes):
    if weights_padding_bytes > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_bytes)).contiguous()
    return x_fp4


def slice_nvfp4_output(out, output_size):
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


def apply_nvfp4_linear(layer, x):
    """Inlined FLASHINFER_CUTLASS path of apply_nvfp4_linear."""
    weight = layer.weight
    weight_scale = layer.weight_scale
    input_global_scale_inv = layer.input_global_scale_inv
    alpha = layer.alpha
    output_size = layer.output_size_per_partition

    output_dtype = x.dtype
    output_shape = [*x.shape[:-1], output_size]

    x_fp4, x_blockscale = scaled_fp4_quant(
        x, input_global_scale_inv, is_sf_swizzled_layout=True
    )

    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    out = flashinfer_scaled_fp4_mm(
        x_fp4, weight, x_blockscale, weight_scale, alpha,
        output_dtype, backend="cutlass",
    )

    out = slice_nvfp4_output(out, output_size)
    return out.view(*output_shape)


# ── Weight loading / synthetic weights ───────────────────────────────────

# DeepSeek V3/R1 MLA dimensions
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
OUTPUT_SIZE = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 2112
GROUP_SIZE = 16  # NVFP4 block size


def load_layer0_weights(device):
    """Load fused_qkv_a_proj weights from the model checkpoint (layer 0)."""
    model_name = "DeepSeek-R1-0528-NVFP4-v2"
    search_paths = [
        "/mnt/local/hf_cache",
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/mnt/lustre",
    ]

    model_dir = None
    for base in search_paths:
        candidates = glob.glob(f"{base}/**/models--nvidia--{model_name}",
                               recursive=True)
        if candidates:
            snapshots = glob.glob(f"{candidates[0]}/snapshots/*/")
            if snapshots:
                model_dir = snapshots[0]
                break

    if model_dir is None:
        for base in search_paths:
            candidates = glob.glob(f"{base}/**/{model_name}", recursive=True)
            if candidates:
                model_dir = candidates[0]
                break

    if model_dir is None:
        print("Model checkpoint not found, using synthetic weights")
        return None

    print(f"Loading weights from: {model_dir}")

    from safetensors import safe_open

    prefix = "model.layers.0.self_attn.fused_qkv_a_proj"
    needed = {
        f"{prefix}.weight": "weight",
        f"{prefix}.input_scale": "input_scale",
        f"{prefix}.weight_scale": "weight_scale",
        f"{prefix}.weight_scale_2": "weight_scale_2",
    }
    found = {}

    st_files = sorted(glob.glob(f"{model_dir}/*.safetensors"))
    for sf_path in st_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed:
                    found[needed[key]] = f.get_tensor(key)
        if len(found) == len(needed):
            break

    if len(found) != len(needed):
        missing = set(needed.values()) - set(found.keys())
        print(f"Missing weights: {missing}, using synthetic weights")
        return None

    return {k: v.to(device) for k, v in found.items()}


def make_synthetic_weights(device):
    """Create synthetic NVFP4 weights matching fused_qkv_a_proj dimensions."""
    k_packed = HIDDEN_SIZE // 2
    num_blocks_k = HIDDEN_SIZE // GROUP_SIZE

    weight = torch.randint(0, 256, (OUTPUT_SIZE, k_packed),
                           dtype=torch.uint8, device=device)
    weight_scale = torch.ones(OUTPUT_SIZE, num_blocks_k,
                              dtype=torch.float8_e4m3fn, device=device)
    input_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    weight_scale_2 = torch.tensor([1.0], dtype=torch.float32, device=device)

    return {
        "weight": weight,
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "weight_scale_2": weight_scale_2,
    }


def build_layer(weights):
    """Build a fake layer module with the right attributes for apply_nvfp4_linear."""

    class FakeLayer(torch.nn.Module):
        pass

    layer = FakeLayer()

    layer.weight = Parameter(weights["weight"], requires_grad=False)
    layer.weight_scale = Parameter(weights["weight_scale"], requires_grad=False)

    input_global_scale = weights["input_scale"].max().to(torch.float32)
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
    weight_global_scale = weights["weight_scale_2"].max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    layer.alpha = Parameter(
        input_global_scale * weight_global_scale, requires_grad=False)
    layer.input_global_scale_inv = Parameter(
        (1.0 / input_global_scale).to(torch.float32), requires_grad=False)
    layer.output_size_per_partition = OUTPUT_SIZE
    layer.input_size_per_partition = HIDDEN_SIZE

    # Convert to kernel format (swizzle scales, pad weights)
    swizzled_ws = swizzle_blockscale(layer.weight_scale.data)
    padded_w, padding_cols = pad_nvfp4_weight_for_cutlass(layer.weight.data)
    layer.weight = Parameter(padded_w, requires_grad=False)
    layer.weight_scale = Parameter(swizzled_ws, requires_grad=False)
    layer.weights_padding_cols = padding_cols

    return layer


def test_nan_contamination():
    assert flashinfer_scaled_fp4_mm is not None, (
        "flashinfer not available — cannot test NVFP4 path"
    )

    torch.manual_seed(42)
    device = "cuda"

    print("NVFP4 backend: flashinfer-cutlass (hardcoded)")

    # Try loading real weights, fall back to synthetic
    weights = load_layer0_weights(device)
    if weights is None:
        weights = make_synthetic_weights(device)
        print("Using synthetic NVFP4 weights")
    else:
        print(f"Using real model weights — "
              f"weight={list(weights['weight'].shape)} "
              f"weight_scale={list(weights['weight_scale'].shape)}")

    layer = build_layer(weights)

    # CUDA graph decode patterns: first N tokens real, rest NaN padding
    patterns = []
    for batch_size in [8, 16, 32, 64, 128, 256, 512, 1024]:
        for num_real in [1, 2, 3]:
            patterns.append((batch_size, f"first_{num_real}_real"))
        patterns.append((batch_size, "none"))

    for batch_size in [1, 2, 4, 7, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for p in ["none", "all_but_first", "all_but_last", "even"]:
            patterns.append((batch_size, p))

    for batch_size, nan_pattern in patterns:
        hidden = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.bfloat16,
                             device=device)

        # Reference: clean input through the real NVFP4 path
        ref_output = apply_nvfp4_linear(layer, hidden)

        # Poison some tokens
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

        # Run with poisoned input
        test_output = apply_nvfp4_linear(layer, poisoned)

        # Check clean tokens
        for tok in range(batch_size):
            is_poisoned = poisoned[tok].isnan().any().item()
            out_has_nan = test_output[tok].isnan().any().item()

            if is_poisoned:
                assert out_has_nan, (
                    f"batch={batch_size} pattern={nan_pattern} tok={tok}: "
                    f"NaN input but clean output!")
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

    print("\nAll tests passed — no cross-token NaN contamination "
          "in fused_qkv_a_proj NVFP4 path")
    return True


if __name__ == "__main__":
    test_nan_contamination()
