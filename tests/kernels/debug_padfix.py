"""Test: does storing ALL 128 Hadamard coefficients improve quality?"""
import math
import torch
from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig, TurboQuantState,
)
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    hadamard_turboquant_encode, hadamard_turboquant_decode,
)

device = torch.device("cuda")
HEAD_SIZE = 128
N = 32  # 8 tokens × 4 heads

cfg = TurboQuantConfig(bit_width=4, outlier_fraction=0.15)
state = TurboQuantState(cfg, HEAD_SIZE, layer_idx=0, device=device)

# Random data (like values, no RoPE inflation)
torch.manual_seed(42)
x = torch.randn(N, 1, state.normal_size, device=device, dtype=torch.bfloat16)

# === Current approach: store normal_size=109 indices ===
indices_109, norms = hadamard_turboquant_encode(
    x, state.sign_flips, state.codebook, state.boundaries)
decoded_109 = hadamard_turboquant_decode(
    indices_109, norms, state.sign_flips, state.codebook,
    output_dtype=torch.bfloat16)

cos_109 = torch.nn.functional.cosine_similarity(
    x.reshape(N, -1).float(), decoded_109.reshape(N, -1).float(), dim=1
).mean().item()

# === Fixed approach: store ALL BLOCK_D=128 indices ===
# Manually do encode with all 128 indices
BLOCK_D = state.sign_flips.shape[0]  # 128
x_padded = torch.nn.functional.pad(
    x.reshape(N, state.normal_size).float(), (0, BLOCK_D - state.normal_size))
# Normalize
norms_manual = x_padded.norm(dim=-1, keepdim=True)
x_norm = x_padded / (norms_manual + 1e-16)
# Sign flip
x_flipped = x_norm * state.sign_flips.unsqueeze(0)
# Hadamard (PyTorch)
from vllm.model_executor.layers.quantization.turboquant import _hadamard_transform
x_had = _hadamard_transform(x_flipped.clone())
# Quantize ALL 128
indices_128 = torch.bucketize(x_had.contiguous(), state.boundaries).to(torch.uint8)

# Decode ALL 128
reconstructed_128 = state.codebook[indices_128.long()]
# Inverse Hadamard
x_inv = _hadamard_transform(reconstructed_128.clone())
# Sign flip
x_inv = x_inv * state.sign_flips.unsqueeze(0)
# Scale by norm
x_inv = x_inv * norms_manual
# Trim to normal_size
decoded_128 = x_inv[:, :state.normal_size].to(torch.bfloat16)

cos_128 = torch.nn.functional.cosine_similarity(
    x.reshape(N, -1).float(), decoded_128.reshape(N, -1).float(), dim=1
).mean().item()

print(f"Normal size: {state.normal_size}, BLOCK_D: {BLOCK_D}")
print(f"Cosine (109 indices): {cos_109:.6f}")
print(f"Cosine (128 indices): {cos_128:.6f}")
print(f"Improvement: {cos_128 - cos_109:.6f}")
print(f"\nPacked bytes (109): {math.ceil(109 * 4 / 8)} bytes")
print(f"Packed bytes (128): {math.ceil(128 * 4 / 8)} bytes")
