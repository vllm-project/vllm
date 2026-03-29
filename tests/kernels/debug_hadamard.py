"""Test Hadamard round-trip WITHOUT quantization to isolate the error source."""
import torch
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    hadamard_turboquant_encode, hadamard_turboquant_decode,
)
from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig, TurboQuantState,
)

device = torch.device("cuda")
cfg = TurboQuantConfig(bit_width=4, outlier_fraction=0.0)  # No outliers
state = TurboQuantState(cfg, head_size=109, layer_idx=0, device=device)
# state.normal_size = 109, BLOCK_D = 128

N = 32
x = torch.randn(N, 1, 109, device=device, dtype=torch.float32)

# Encode→decode (includes quantization)
indices, norms = hadamard_turboquant_encode(
    x, state.sign_flips, state.codebook, state.boundaries)
decoded = hadamard_turboquant_decode(
    indices, norms, state.sign_flips, state.codebook,
    output_dtype=torch.float32)

cos_with_quant = torch.nn.functional.cosine_similarity(
    x.reshape(N, -1), decoded.reshape(N, -1), dim=1).mean().item()
print(f"With quantization:    cosine = {cos_with_quant:.6f}")

# Now test Hadamard ONLY (bypass quantization by using exact values)
# Encode: normalize → sign_flip → hadamard → get values
# Then decode with those EXACT values (no quantization loss)
indices2, norms2 = hadamard_turboquant_encode(
    x, state.sign_flips, state.codebook, state.boundaries)
# Get the exact rotated values (codebook lookup of indices)
exact_values = state.codebook[indices2.long()]
# Decode using exact values (same as decode kernel but with perfect indices)
decoded2 = hadamard_turboquant_decode(
    indices2, norms2, state.sign_flips, state.codebook,
    output_dtype=torch.float32)

# The above is the SAME as with quantization. Let me test differently:
# Manually do normalize → sign_flip → Hadamard → inverse Hadamard → sign_flip → scale
# WITHOUT any quantization
import math

BLOCK_D = state.sign_flips.shape[0]  # 128
x_flat = x.reshape(N, 109)
norm_vals = x_flat.norm(dim=-1, keepdim=True)
x_norm = x_flat / (norm_vals + 1e-16)

# Pad to 128
x_padded = torch.nn.functional.pad(x_norm, (0, BLOCK_D - 109))
# Sign flip
x_flipped = x_padded * state.sign_flips.unsqueeze(0)

# Triton Hadamard encode (as if for 128-dim input, 1 head)
x_for_encode = x_flipped.unsqueeze(1)  # [N, 1, 128]

# Use the encode kernel with head_size=128 (no trimming)
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    _fused_hadamard_encode_kernel,
)

# Full 128-dim encode
indices_full = torch.empty(N, 1, 128, dtype=torch.uint8, device=device)
norms_full = torch.empty(N, 1, dtype=torch.float16, device=device)
scratch = torch.empty(N, BLOCK_D, dtype=torch.float32, device=device)
sign_flips_128 = state.sign_flips  # already 128

_fused_hadamard_encode_kernel[(N, 1)](
    x_ptr=x_for_encode,
    signs_ptr=sign_flips_128,
    boundaries_ptr=state.boundaries,
    scratch_ptr=scratch,
    indices_ptr=indices_full,
    norms_ptr=norms_full,
    head_size=128,  # Full size, no trimming!
    num_kv_heads=1,
    num_boundaries=state.boundaries.shape[0],
    LOG2_D=int(math.log2(BLOCK_D)),
    x_stride_token=x_for_encode.stride(0),
    x_stride_head=x_for_encode.stride(1),
    idx_stride_token=indices_full.stride(0),
    idx_stride_head=indices_full.stride(1),
    norm_stride_token=norms_full.stride(0),
    BLOCK_D=BLOCK_D,
    num_warps=1,
    num_stages=1,
)

# Decode with ALL 128 indices
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    _fused_hadamard_decode_kernel,
)
decoded_full = torch.empty(N, 1, 128, dtype=torch.float32, device=device)
scratch2 = torch.empty(N, BLOCK_D, dtype=torch.float32, device=device)

_fused_hadamard_decode_kernel[(N, 1)](
    indices_ptr=indices_full,
    norms_ptr=norms_full,
    signs_ptr=sign_flips_128,
    codebook_ptr=state.codebook,
    scratch_ptr=scratch2,
    out_ptr=decoded_full,
    head_size=128,
    num_kv_heads=1,
    LOG2_D=int(math.log2(BLOCK_D)),
    idx_stride_token=indices_full.stride(0),
    idx_stride_head=indices_full.stride(1),
    norm_stride_token=norms_full.stride(0),
    out_stride_token=decoded_full.stride(0),
    out_stride_head=decoded_full.stride(1),
    BLOCK_D=BLOCK_D,
    OUTPUT_BF16=False,
    num_warps=1,
    num_stages=1,
)

# Trim to 109 and compare
decoded_trimmed = decoded_full[:, :, :109]
cos_full = torch.nn.functional.cosine_similarity(
    x.reshape(N, -1), decoded_trimmed.reshape(N, -1), dim=1).mean().item()
print(f"Full 128 encode/decode: cosine = {cos_full:.6f}")
print(f"Improvement from full:  {cos_full - cos_with_quant:.6f}")
