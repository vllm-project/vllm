"""Minimal test: is the Triton Hadamard encode→decode lossless (no quantization)?"""
import math
import torch
from vllm.v1.attention.ops.triton_hadamard_turboquant import (
    _fused_hadamard_encode_kernel, _fused_hadamard_decode_kernel,
)

device = torch.device("cuda")
D = 128  # power of 2, no padding needed
N = 16
LOG2_D = int(math.log2(D))

torch.manual_seed(42)
x = torch.randn(N, 1, D, device=device, dtype=torch.float32)
signs = torch.where(torch.rand(D, device=device) > 0.5,
                     torch.ones(D, device=device),
                     -torch.ones(D, device=device))

from vllm.model_executor.layers.quantization.turboquant import CODEBOOKS_NORMALIZED

# 4-bit with CORRECT scale (1/sqrt(128) matching Hadamard output)
scale128 = 1.0 / math.sqrt(D)
codebook128 = torch.tensor([c * scale128 for c in CODEBOOKS_NORMALIZED[4]],
                            device=device, dtype=torch.float32)
boundaries128 = (codebook128[:-1] + codebook128[1:]) / 2.0

indices = torch.empty(N, 1, D, dtype=torch.uint8, device=device)
norms_out = torch.empty(N, 1, dtype=torch.float16, device=device)
scratch = torch.empty(N, D, dtype=torch.float32, device=device)

_fused_hadamard_encode_kernel[(N, 1)](
    x_ptr=x, signs_ptr=signs, boundaries_ptr=boundaries128,
    scratch_ptr=scratch, indices_ptr=indices, norms_ptr=norms_out,
    head_size=D, num_kv_heads=1,
    num_boundaries=boundaries128.shape[0], LOG2_D=LOG2_D,
    x_stride_token=x.stride(0), x_stride_head=x.stride(1),
    idx_stride_token=indices.stride(0), idx_stride_head=indices.stride(1),
    norm_stride_token=norms_out.stride(0), BLOCK_D=D,
    num_warps=1, num_stages=1,
)

decoded = torch.empty(N, 1, D, dtype=torch.float32, device=device)
scratch2 = torch.empty(N, D, dtype=torch.float32, device=device)

_fused_hadamard_decode_kernel[(N, 1)](
    indices_ptr=indices, norms_ptr=norms_out, signs_ptr=signs,
    codebook_ptr=codebook128, scratch_ptr=scratch2, out_ptr=decoded,
    head_size=D, num_kv_heads=1, LOG2_D=LOG2_D,
    idx_stride_token=indices.stride(0), idx_stride_head=indices.stride(1),
    norm_stride_token=norms_out.stride(0),
    out_stride_token=decoded.stride(0), out_stride_head=decoded.stride(1),
    BLOCK_D=D, OUTPUT_BF16=False,
    num_warps=1, num_stages=1,
)

cos128 = torch.nn.functional.cosine_similarity(
    x.reshape(N, -1), decoded.reshape(N, -1), dim=1).mean().item()
print(f"4-bit D=128 (correct scale 1/sqrt(128)):")
print(f"  Cosine: {cos128:.6f}")
print(f"  Expected (Lloyd-Max): ~0.990")

# Test with D=109 (padded to 128) - the actual use case
x109 = torch.randn(N, 1, 109, device=device, dtype=torch.float32)
from vllm.model_executor.layers.quantization.turboquant import (
    TurboQuantConfig, TurboQuantState, _get_codebook,
)
cfg = TurboQuantConfig(bit_width=4, outlier_fraction=0.0)
state = TurboQuantState(cfg, head_size=109, layer_idx=0, device=device)

indices109 = torch.empty(N, 1, 109, dtype=torch.uint8, device=device)
norms109 = torch.empty(N, 1, dtype=torch.float16, device=device)
scratch3 = torch.empty(N, 128, dtype=torch.float32, device=device)

_fused_hadamard_encode_kernel[(N, 1)](
    x_ptr=x109, signs_ptr=state.sign_flips,
    boundaries_ptr=state.boundaries, scratch_ptr=scratch3,
    indices_ptr=indices109, norms_ptr=norms109,
    head_size=109, num_kv_heads=1,
    num_boundaries=state.boundaries.shape[0],
    LOG2_D=int(math.log2(128)),
    x_stride_token=x109.stride(0), x_stride_head=x109.stride(1),
    idx_stride_token=indices109.stride(0),
    idx_stride_head=indices109.stride(1),
    norm_stride_token=norms109.stride(0), BLOCK_D=128,
    num_warps=1, num_stages=1,
)

decoded109 = torch.empty(N, 1, 109, dtype=torch.float32, device=device)
scratch4 = torch.empty(N, 128, dtype=torch.float32, device=device)
_fused_hadamard_decode_kernel[(N, 1)](
    indices_ptr=indices109, norms_ptr=norms109,
    signs_ptr=state.sign_flips, codebook_ptr=state.codebook,
    scratch_ptr=scratch4, out_ptr=decoded109,
    head_size=109, num_kv_heads=1,
    LOG2_D=int(math.log2(128)),
    idx_stride_token=indices109.stride(0),
    idx_stride_head=indices109.stride(1),
    norm_stride_token=norms109.stride(0),
    out_stride_token=decoded109.stride(0),
    out_stride_head=decoded109.stride(1),
    BLOCK_D=128, OUTPUT_BF16=False,
    num_warps=1, num_stages=1,
)

cos109 = torch.nn.functional.cosine_similarity(
    x109.reshape(N, -1), decoded109.reshape(N, -1), dim=1).mean().item()
print(f"\n4-bit round-trip (D=109, padded to 128):")
print(f"  Cosine: {cos109:.6f}")
print(f"  Codebook scale: 1/sqrt({state.normal_size}) = {1/math.sqrt(state.normal_size):.6f}")
print(f"  Hadamard scale: 1/sqrt(128) = {1/math.sqrt(128):.6f}")
print(f"  Scale mismatch: {math.sqrt(128/state.normal_size):.4f}x")
