"""Compare TurboQuant emulation (working) vs packed cache (broken).

Tests whether the encode→pack→unpack→decode pipeline matches
the encode→decode emulation pipeline.
"""
import math
import torch
from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig, TurboQuantState, _triton_pre_dequant_one,
)
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    hadamard_turboquant_encode, hadamard_turboquant_decode,
)

device = torch.device("cuda")
HEAD_SIZE = 128
NUM_TOKENS = 8
NUM_KV_HEADS = 4

# Create realistic K vectors (simulate RoPE-inflated channels)
torch.manual_seed(42)
k = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
# Inflate certain channels to simulate RoPE
k[..., 0:10] *= 20.0  # High-variance channels
k[..., 64:74] *= 15.0

# Create TurboQuant state
cfg = TurboQuantConfig(bit_width=4, outlier_fraction=0.15)
state = TurboQuantState(cfg, HEAD_SIZE, layer_idx=0, device=device)

# Calibrate on real data
k_flat = k.reshape(-1, HEAD_SIZE)
state.calibrate_outliers(k_flat)
print(f"Outlier channels: {state.outlier_idx.tolist()}")
print(f"Normal size: {state.normal_size}")

# === Path A: Emulation (encode→decode in one shot) ===
k_emulated = _triton_pre_dequant_one(k, state, k.dtype)

# === Path B: Packed cache (encode→pack→unpack→decode) ===
normal_x = k[..., state.normal_idx].contiguous()
outlier_x = k[..., state.outlier_idx]

# Encode
indices, norms = hadamard_turboquant_encode(
    normal_x, state.sign_flips, state.codebook, state.boundaries,
)

# 4-bit pack
normal_size = state.normal_size
flat_idx = indices.reshape(-1, normal_size)
N = flat_idx.shape[0]
if normal_size % 2 != 0:
    flat_idx = torch.nn.functional.pad(flat_idx, (0, 1), value=0)
packed = flat_idx[:, 0::2] | (flat_idx[:, 1::2] << 4)
packed_bytes = math.ceil(normal_size * 4 / 8)
packed = packed[:, :packed_bytes]

# 4-bit unpack
low = packed & 0x0F
high = (packed >> 4) & 0x0F
unpacked = torch.stack([low, high], dim=-1).reshape(N, -1)[:, :normal_size]

# Check indices roundtrip
idx_match = torch.equal(indices.reshape(N, normal_size), unpacked)
print(f"\nIndices roundtrip exact match: {idx_match}")

# Decode
decoded_normal = hadamard_turboquant_decode(
    unpacked.reshape(N, 1, normal_size).to(torch.uint8),
    norms.reshape(N, 1),
    state.sign_flips,
    state.codebook,
    output_dtype=torch.bfloat16,
).reshape(N, normal_size)

# Reassemble
k_packed = torch.empty_like(k, dtype=torch.bfloat16)
k_packed_flat = k_packed.reshape(N, HEAD_SIZE)
k_packed_flat[:, state.normal_idx] = decoded_normal
k_packed_flat[:, state.outlier_idx] = outlier_x.reshape(N, -1).to(torch.bfloat16)
k_packed = k_packed_flat.reshape(k.shape)

# === Compare ===
k_orig = k.float()
k_emu = k_emulated.float()
k_pck = k_packed.float()

cos_emu = torch.nn.functional.cosine_similarity(
    k_orig.reshape(N, -1), k_emu.reshape(N, -1), dim=1
).mean().item()
cos_pck = torch.nn.functional.cosine_similarity(
    k_orig.reshape(N, -1), k_pck.reshape(N, -1), dim=1
).mean().item()
cos_emu_vs_pck = torch.nn.functional.cosine_similarity(
    k_emu.reshape(N, -1), k_pck.reshape(N, -1), dim=1
).mean().item()

print(f"\nCosine similarity:")
print(f"  Original vs Emulation:  {cos_emu:.6f}")
print(f"  Original vs Packed:     {cos_pck:.6f}")
print(f"  Emulation vs Packed:    {cos_emu_vs_pck:.6f}")

max_diff = (k_emu - k_pck).abs().max().item()
print(f"\nMax diff (emulation vs packed): {max_diff:.6f}")

# Check if emulation itself is good
print(f"\nEmulation max error: {(k_orig - k_emu).abs().max().item():.4f}")
print(f"Packed max error:    {(k_orig - k_pck).abs().max().item():.4f}")
