#!/usr/bin/env python3
"""
NCU profiling script for MHC fusion kernels.
Measures actual HBM traffic reduction.
"""

import torch
import sys
import os

# Add vllm to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vllm.model_executor.kernels.mhc.optimized_wrappers import (
    mhc_post_hc_head_fused,
    mhc_post_hc_head_norm_fused,
    mhc_post_mean_fused
)


def profile_mhc_post_hc_head():
    """Profile MHC Post + HC Head fusion."""
    print("=" * 80)
    print("Profiling: MHC Post + HC Head Fusion")
    print("=" * 80)

    num_tokens = 256
    hc_mult = 4
    hidden_size = 2048

    # Input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    post_layer_mix = torch.randn(num_tokens, hc_mult, dtype=torch.float32, device='cuda')
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device='cuda')
    hc_head_fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32, device='cuda')
    hc_head_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    hc_head_base = torch.zeros(hc_mult, dtype=torch.float32, device='cuda')
    rms_norm_eps = 1e-6
    hc_eps = 1e-6

    # Warm-up
    for _ in range(10):
        _ = mhc_post_hc_head_fused(x, residual, post_layer_mix, comb_res_mix,
                                    hc_head_fn, hc_head_scale, hc_head_base,
                                    rms_norm_eps, hc_eps)

    torch.cuda.synchronize()

    # Run for profiling
    print("\nRunning fused kernel (profiling iteration)...")
    result = mhc_post_hc_head_fused(x, residual, post_layer_mix, comb_res_mix,
                                     hc_head_fn, hc_head_scale, hc_head_base,
                                     rms_norm_eps, hc_eps)
    torch.cuda.synchronize()

    print(f"Input shapes: x={x.shape}, residual={residual.shape}")
    print(f"Output shape: {result.shape}")
    print("\nExpected HBM traffic reduction: ~50%")


def profile_mhc_post_hc_head_norm():
    """Profile MHC Post + HC Head + RMSNorm fusion."""
    print("\n" + "=" * 80)
    print("Profiling: MHC Post + HC Head + RMSNorm Fusion")
    print("=" * 80)

    num_tokens = 256
    hc_mult = 4
    hidden_size = 2048

    # Input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    post_layer_mix = torch.randn(num_tokens, hc_mult, dtype=torch.float32, device='cuda')
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device='cuda')
    hc_head_fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32, device='cuda')
    hc_head_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    hc_head_base = torch.zeros(hc_mult, dtype=torch.float32, device='cuda')
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda')
    rms_norm_eps = 1e-6
    hc_eps = 1e-6
    norm_eps = 1e-6

    # Warm-up
    for _ in range(10):
        _ = mhc_post_hc_head_norm_fused(x, residual, post_layer_mix, comb_res_mix,
                                         hc_head_fn, hc_head_scale, hc_head_base,
                                         norm_weight, rms_norm_eps, hc_eps, norm_eps)

    torch.cuda.synchronize()

    # Run for profiling
    print("\nRunning fused kernel (profiling iteration)...")
    result = mhc_post_hc_head_norm_fused(x, residual, post_layer_mix, comb_res_mix,
                                          hc_head_fn, hc_head_scale, hc_head_base,
                                          norm_weight, rms_norm_eps, hc_eps, norm_eps)
    torch.cuda.synchronize()

    print(f"Input shapes: x={x.shape}, residual={residual.shape}, norm_weight={norm_weight.shape}")
    print(f"Output shape: {result.shape}")
    print("\nExpected HBM traffic reduction: ~80%")


def profile_mhc_post_mean():
    """Profile MHC Post + Mean fusion."""
    print("\n" + "=" * 80)
    print("Profiling: MHC Post + Mean Fusion")
    print("=" * 80)

    num_tokens = 256
    hc_mult = 4
    hidden_size = 2048

    # Input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cuda')
    post_layer_mix = torch.randn(num_tokens, hc_mult, dtype=torch.bfloat16, device='cuda')
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.bfloat16, device='cuda')

    # Warm-up
    for _ in range(10):
        _ = mhc_post_mean_fused(x, residual, post_layer_mix, comb_res_mix)

    torch.cuda.synchronize()

    # Run for profiling
    print("\nRunning fused kernel (profiling iteration)...")
    residual_out, mean_out = mhc_post_mean_fused(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()

    print(f"Input shapes: x={x.shape}, residual={residual.shape}")
    print(f"Output shapes: residual_out={residual_out.shape}, mean_out={mean_out.shape}")
    print("\nExpected HBM traffic reduction: ~37.5%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Profile MHC fusion kernels')
    parser.add_argument('--kernel', choices=['all', 'post_hc_head', 'post_hc_head_norm', 'post_mean'],
                        default='all', help='Which kernel to profile')
    args = parser.parse_args()

    if args.kernel in ['all', 'post_hc_head']:
        profile_mhc_post_hc_head()

    if args.kernel in ['all', 'post_hc_head_norm']:
        profile_mhc_post_hc_head_norm()

    if args.kernel in ['all', 'post_mean']:
        profile_mhc_post_mean()

    print("\n" + "=" * 80)
    print("Profiling complete. Use ncu to analyze HBM traffic.")
    print("=" * 80)
