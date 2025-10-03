# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton
import triton.language as tl
import torch
import functools
from typing import Tuple, Dict, Optional
from triton.tools.tensor_descriptor import TensorDescriptor


class TensorPool:
    """Pool of pre-allocated tensors for CUDA graph capture."""
    
    def __init__(self):
        self.pool: Dict[Tuple, torch.Tensor] = {}
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get or create a tensor with given specs."""
        key = (shape, dtype, str(device))
        if key not in self.pool:
            self.pool[key] = torch.empty(shape, dtype=dtype, device=device)
        return self.pool[key]
    
    def clear(self):
        """Clear the pool to free memory."""
        self.pool.clear()


class TMADescriptorCache:
    """Cache for TensorDescriptors to enable CUDA graph capture."""
    
    def __init__(self):
        self.cache: Dict[Tuple, Tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor, TensorDescriptor]] = {}
        
    def get_descriptors(self, a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor, c: torch.Tensor,
                       M: int, N: int, K: int, 
                       BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
                       EPILOGUE_SUBTILE: bool) -> Tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor, TensorDescriptor]:
        """Get cached descriptors and update them with actual tensors."""
        key = (M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPILOGUE_SUBTILE, 
               str(a.device), str(b.device), str(c.device), str(b_scale.device),
               a.dtype, b.dtype, c.dtype, b_scale.dtype)  # Include scale tensor ID to ensure different scales use different cache entries
        
        if key not in self.cache:
            # Create descriptors with the actual tensors for proper initialization
            a_shape, a_stride = a.shape, a.stride()
            b_shape, b_stride = b.shape, b.stride()
            c_shape, c_stride = c.shape, c.stride()
            b_scale_shape, b_scale_stride = b_scale.shape, b_scale.stride()

            a_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K]
            b_block_shape = [BLOCK_SIZE_N, BLOCK_SIZE_K]
            c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2] if EPILOGUE_SUBTILE else [BLOCK_SIZE_M, BLOCK_SIZE_N]
            # Scale descriptor doesn't need block shape since we access it linearly
            b_scale_block_shape = [BLOCK_SIZE_N]
            
            # Create and cache the descriptors
            a_desc = TensorDescriptor(a, a_shape, a_stride, a_block_shape)
            b_desc = TensorDescriptor(b, b_shape, b_stride, b_block_shape)
            c_desc = TensorDescriptor(c, c_shape, c_stride, c_block_shape)
            b_scale_desc = TensorDescriptor(b_scale, b_scale_shape, b_scale_stride, b_scale_block_shape)
            
            self.cache[key] = (a_desc, b_desc, c_desc, b_scale_desc)
        else:
            # Reuse cached descriptors but update with current tensors
            a_desc, b_desc, c_desc, b_scale_desc = self.cache[key]
            a_desc.tensor = a
            b_desc.tensor = b
            c_desc.tensor = c
            b_scale_desc.tensor = b_scale
            
        return self.cache[key]


# Global instances for reuse
_tensor_pool = TensorPool()
_descriptor_cache = TMADescriptorCache()
_NUM_SMS = {}  # Cache for device SMS count

## TODO: using tuner
def get_num_sms(device):
    """Get cached SMS count for device."""
    device_idx = device.index if hasattr(device, "index") else torch.cuda.current_device()
    if device_idx not in _NUM_SMS:
        _NUM_SMS[device_idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS[device_idx]


@functools.lru_cache(maxsize=256)
def _get_static_grid(M: int, N: int, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, NUM_SMS: int) -> Tuple[int]:
    """Static grid function with caching for CUDA graph capture."""
    return (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM, "EPILOGUE_SUBTILE":
                SUBTILE
            }, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [16, 32, 64]  #
        for BN in [64, 128, 256, 512]  #
        for BK in [32, 64, 128]  #
        for s in ([2, 3])  #
        for w in [4, 8]  #
        for GM in [1, 2, 4, 8]  #
        for SUBTILE in [True, False]  #
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _get_config(M):
    configuration = {
        1: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "EPILOGUE_SUBTILE": False,
            "num_stages": 4,
            "num_warps": 4
        },
        2 : {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 2,
            "EPILOGUE_SUBTILE": True,
            "num_stages": 4,
            "num_warps": 4
        },
        4 : {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 512,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "EPILOGUE_SUBTILE": True,
            "num_stages": 4,
            'num_warps': 4
        },
        8 : {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 512,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "EPILOGUE_SUBTILE": True,
            "num_stages": 4,
            "num_warps": 4
        }
    }
    return configuration.get(M, {})

@triton.jit
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc, b_scale_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 num_stages: tl.constexpr,  #
                                 num_warps: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k]).to(tl.float8e4nv)  # Changed here to load as FP8
            # a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        # Apply per-channel scaling - always enabled since scale is guaranteed to be present
        # Load the scales for this block of columns using TMA descriptor
        # Use the same offs_bn that was used for weight loading to ensure consistency
        scales = b_scale_desc.load([offs_bn])
        # Broadcast scales to match accumulator shape (M, N)
        scales_bc = tl.broadcast_to(scales[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        accumulator = accumulator * scales_bc

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am, offs_bn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am, offs_bn], accumulator)


def matmul_tma_persistent(x, weight, weight_scale, input_scale=None, bias=None, warp_specialize=False):
    """Optimized TMA persistent matmul with CUDA graph capture support.
    Supports FP8 weights with per-channel scaling.

    Args:
        x: Input tensor (M, K) in bf16 format
        weight: Weight tensor (N, K) in FP8E4M3FN format
        weight_scale: Per-channel weight scales (N,) in float32 - always required
        input_scale: Optional input scale (not used currently)
        bias: Optional bias tensor (N,) to add to output
        warp_specialize: Enable warp specialization

    Returns:
        Output tensor (M, N) in BF16 format
    """
    # Rename for clarity
    a, b = x, weight
    
    # Check constraints - inputs should already be BF16
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == torch.bfloat16, f"Expected BF16 input, got {a.dtype}"
    assert b.dtype == torch.float8_e4m3fn, f"Expected FP8 weight, got {b.dtype}"
    
    # Check weight scale - always required
    assert weight_scale is not None, "weight_scale is required"
    assert weight_scale.shape == (b.shape[0],), f"Weight scale shape {weight_scale.shape} doesn't match weight output dim {b.shape[0]}"
    assert weight_scale.dtype == torch.bfloat16, f"Expected bfloat16 weight scale, got {weight_scale.dtype}"

    M, K = a.shape
    N, K_b = b.shape

    optimal_config = _get_config(M)
    BLOCK_SIZE_M = optimal_config.get("BLOCK_SIZE_M", 16)
    BLOCK_SIZE_N = optimal_config.get("BLOCK_SIZE_N", 512)
    BLOCK_SIZE_K = optimal_config.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = optimal_config.get("GROUP_SIZE_M", 1)
    EPILOGUE_SUBTILE = optimal_config.get("EPILOGUE_SUBTILE", True)
    num_stages = optimal_config.get("num_stages", 4) 
    num_warps = optimal_config.get("num_warps", 4)

    # Use tensor pool for output allocation (enables graph capture)
    c = _tensor_pool.get_tensor((M, N), torch.bfloat16, a.device)

    NUM_SMS = get_num_sms(a.device)

    # Get cached descriptors (enables graph capture) - includes scale descriptor
    a_desc, b_desc, c_desc, b_scale_desc = _descriptor_cache.get_descriptors(
        a, b, weight_scale, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPILOGUE_SUBTILE
    )

    # Use static grid function (enables graph capture)
    grid_size = _get_static_grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_SMS)

    matmul_kernel_tma_persistent[grid_size](
        a_desc, b_desc, c_desc, b_scale_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        FP8_OUTPUT=False,  # Keep BF16 output as requested
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=warp_specialize,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    # Add bias if provided - use in-place operation for graph capture efficiency
    if bias is not None:
        c.add_(bias.unsqueeze(0))

    return c


def matmul_tma_persistent_original(a, b, warp_specialize=False):
    """Original implementation for comparison."""
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    # assert a.dtype == b.dtype, "Incompatible dtypes"

    if a.dtype != torch.float8_e4m3fn:
        a = a.to(torch.float8_e4m3fn)

    if b.dtype != torch.float8_e4m3fn:
        b = b.to(torch.float8_e4m3fn)

    M, K = a.shape
    N, K = b.shape

    optimal_config = _get_config(M)
    BLOCK_SIZE_M = optimal_config.get("BLOCK_SIZE_M", 16)
    BLOCK_SIZE_N = optimal_config.get("BLOCK_SIZE_N", 512)
    BLOCK_SIZE_K = optimal_config.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = optimal_config.get("GROUP_SIZE_M", 1)
    EPILOGUE_SUBTILE = optimal_config.get("EPILOGUE_SUBTILE", True)
    num_stages = optimal_config.get("num_stages", 4) 
    num_warps = optimal_config.get("num_warps", 4)

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Pre-compute shapes, strides, and block shapes to minimize TensorDescriptor overhead
    a_shape, a_stride = a.shape, a.stride()
    b_shape, b_stride = b.shape, b.stride()
    c_shape, c_stride = c.shape, c.stride()
    
    # Pre-allocate block shape lists
    a_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K]
    b_block_shape = [BLOCK_SIZE_N, BLOCK_SIZE_K]
    c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2] if EPILOGUE_SUBTILE else [BLOCK_SIZE_M, BLOCK_SIZE_N]
    
    # Create TensorDescriptors with pre-computed values
    a_desc = TensorDescriptor(a, a_shape, a_stride, a_block_shape)
    b_desc = TensorDescriptor(b, b_shape, b_stride, b_block_shape)
    c_desc = TensorDescriptor(c, c_shape, c_stride, c_block_shape)

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,  #
        BLOCK_SIZE_N=BLOCK_SIZE_N,  #
        BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        FP8_OUTPUT=False,  #
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c

def clear_tma_caches():
    """Clear all caches to free memory."""
    global _tensor_pool, _descriptor_cache, _NUM_SMS
    _tensor_pool.clear()
    _descriptor_cache.cache.clear()
    _NUM_SMS.clear()
    _get_static_grid.cache_clear()


def get_cache_info():
    """Get information about cache usage."""
    return {
        "tensor_pool_size": len(_tensor_pool.pool),
        "descriptor_cache_size": len(_descriptor_cache.cache),
        "sms_cache_size": len(_NUM_SMS),
        "grid_cache_info": _get_static_grid.cache_info()
    }


if __name__ == "__main__":
    import time

    M, N, K = 8, 201088, 2880
    dtype = torch.bfloat16

    # load the quantized data
    SAVED_DIR = "/home/ubuntu/quantization/saved_model/quantized_lm_head_scaled.pt"
    state_dict = torch.load(SAVED_DIR)
    weight = state_dict['lm_head.weight'].to(torch.float8_e4m3fn)
    weight_scale = state_dict['lm_head.weight_scale'].to(torch.bfloat16).squeeze().cuda()

    print(f"Weight scale data is {weight_scale}")
    print(f"Weight scale device: {weight_scale.device}")
    
    # Create test tensors
    x = torch.randn((M, K), device="cuda", dtype=dtype)

    print("Testing FP8 GEMM with per-channel scaling...")

    # Test with per-channel scaling
    print("\n=== With per-channel scaling ===")
    c_with_scale = matmul_tma_persistent(x, weight, weight_scale=weight_scale, warp_specialize=False)
    print(f"Output (with scaling) sample: {c_with_scale[0, :5]}")
    torch.cuda.synchronize()

    # Compute reference solutions
    print("\n=== Computing original as reference ===")
    x_fp32 = x.to(torch.float32)
    original_weight = torch.load("/home/ubuntu/quantization/saved_model/original_lm_head_only.pt")['lm_head.weight'].to(torch.float32).to(x.device)
    
    # Reference without scaling
    c_original = torch.matmul(x_fp32, original_weight.t()).to(torch.bfloat16)
    print(f"Output (no scaling) sample: {c_original[0, :5]}")
    torch.cuda.synchronize()

    print("\n=== Computing manual scaling ===")
    # Reference with scaling (manual application)
    weight_scaled = weight.to(torch.float32).to(x.device) * weight_scale.to(x.device).unsqueeze(1)  # broadcast scale to (N, K)
    c_ref_with_scale = torch.matmul(x_fp32, weight_scaled.t())
    c_ref_with_scale = c_ref_with_scale.to(torch.bfloat16)
    print(f"Output reference sample: {c_ref_with_scale[0, :5]}")
    torch.cuda.synchronize()

    # Compare results
    print("\n=== Detailed Results Analysis ===")
    
    # Error between our implementation and reference (manual scaling)
    error_vs_ref = (c_ref_with_scale - c_with_scale).abs()
    print(f"Our Kernel vs Manual Scaling Reference:")
    print(f"  Max abs error:     {error_vs_ref.max().item():.6f}")
    print(f"  Mean abs error:    {error_vs_ref.mean().item():.6f}")
    print(f"  Median abs error:  {error_vs_ref.median().item():.6f}")
    print(f"  Std abs error:     {error_vs_ref.std().item():.6f}")
    print(f"  95th percentile:   {torch.quantile(error_vs_ref.float(), 0.95).item():.6f}")
    
    # Error between our implementation and original (no scaling)
    error_our_vs_original = (c_original - c_with_scale).abs()
    print(f"\nOur Kernel vs Original (no scaling):")
    print(f"  Max abs error:     {error_our_vs_original.max().item():.6f}")
    print(f"  Mean abs error:    {error_our_vs_original.mean().item():.6f}")
    print(f"  Median abs error:  {error_our_vs_original.median().item():.6f}")
    print(f"  Std abs error:     {error_our_vs_original.std().item():.6f}")
    print(f"  95th percentile:   {torch.quantile(error_our_vs_original.float(), 0.95).item():.6f}")
    
    # Error between manual scaling and original (no scaling)
    error_manual_vs_original = (c_original - c_ref_with_scale).abs()
    print(f"\nManual Scaling vs Original (no scaling):")
    print(f"  Max abs error:     {error_manual_vs_original.max().item():.6f}")
    print(f"  Mean abs error:    {error_manual_vs_original.mean().item():.6f}")
    print(f"  Median abs error:  {error_manual_vs_original.median().item():.6f}")
    print(f"  Std abs error:     {error_manual_vs_original.std().item():.6f}")
    print(f"  95th percentile:   {torch.quantile(error_manual_vs_original.float(), 0.95).item():.6f}")
    
    # Relative error analysis
    ref_magnitude = c_ref_with_scale.abs()
    orig_magnitude = c_original.abs()
    
    relative_error_vs_ref = error_vs_ref / (ref_magnitude + 1e-8)
    relative_error_our_vs_orig = error_our_vs_original / (orig_magnitude + 1e-8)
    relative_error_manual_vs_orig = error_manual_vs_original / (orig_magnitude + 1e-8)
    
    print(f"\nRelative Error Analysis:")
    print(f"Our Kernel vs Manual Reference:")
    print(f"  Max relative error:    {relative_error_vs_ref.max().item():.6f}")
    print(f"  Mean relative error:   {relative_error_vs_ref.mean().item():.6f}")
    print(f"  Median relative error: {relative_error_vs_ref.median().item():.6f}")
    
    print(f"Our Kernel vs Original:")
    print(f"  Max relative error:    {relative_error_our_vs_orig.max().item():.6f}")
    print(f"  Mean relative error:   {relative_error_our_vs_orig.mean().item():.6f}")
    print(f"  Median relative error: {relative_error_our_vs_orig.median().item():.6f}")
    
    print(f"Manual Scaling vs Original:")
    print(f"  Max relative error:    {relative_error_manual_vs_orig.max().item():.6f}")
    print(f"  Mean relative error:   {relative_error_manual_vs_orig.mean().item():.6f}")
    print(f"  Median relative error: {relative_error_manual_vs_orig.median().item():.6f}")
    
    # Distribution analysis
    print(f"\nOutput Magnitude Analysis:")
    print(f"  Our output range:      [{c_with_scale.min().item():.6f}, {c_with_scale.max().item():.6f}]")
    print(f"  Reference range:       [{c_ref_with_scale.min().item():.6f}, {c_ref_with_scale.max().item():.6f}]")
    print(f"  Original range:        [{c_original.min().item():.6f}, {c_original.max().item():.6f}]")
    
    print(f"\nOutput Statistics:")
    print(f"  Our output mean:       {c_with_scale.mean().item():.6f}")
    print(f"  Reference mean:        {c_ref_with_scale.mean().item():.6f}")
    print(f"  Original mean:         {c_original.mean().item():.6f}")
    
    print(f"  Our output std:        {c_with_scale.std().item():.6f}")
    print(f"  Reference std:         {c_ref_with_scale.std().item():.6f}")
    print(f"  Original std:          {c_original.std().item():.6f}")
    
    # Correlation analysis
    correlation_our_with_ref = torch.corrcoef(torch.stack([c_with_scale.flatten(), c_ref_with_scale.flatten()]))[0, 1]
    correlation_our_with_orig = torch.corrcoef(torch.stack([c_with_scale.flatten(), c_original.flatten()]))[0, 1]
    correlation_manual_with_orig = torch.corrcoef(torch.stack([c_ref_with_scale.flatten(), c_original.flatten()]))[0, 1]
    
    print(f"\nCorrelation Analysis:")
    print(f"  Our Kernel vs Manual Reference: {correlation_our_with_ref.item():.6f}")
    print(f"  Our Kernel vs Original:         {correlation_our_with_orig.item():.6f}")
    print(f"  Manual Scaling vs Original:     {correlation_manual_with_orig.item():.6f}")
    
    # Per-channel error analysis (sample first few channels)
    print(f"\nPer-channel Error Sample (first 10 channels):")
    for i in range(min(10, c_with_scale.shape[1])):
        channel_error = error_vs_ref[:, i].mean().item()
        channel_scale = weight_scale[i].item()
        print(f"  Channel {i:2d}: error={channel_error:.6f}, scale={channel_scale:.8f}")
    
    # Compare how much each method changes from the original
    our_change_from_orig = (c_with_scale - c_original).abs().max().item()
    manual_change_from_orig = (c_ref_with_scale - c_original).abs().max().item()
    
    print(f"\nChange from Original (Scaling Effects):")
    print(f"  Our Kernel max change:    {our_change_from_orig:.6f}")
    print(f"  Manual Scaling max change: {manual_change_from_orig:.6f}")
    print(f"  Difference in changes:     {abs(our_change_from_orig - manual_change_from_orig):.6f}")
    
    # Summary assessment
    if abs(our_change_from_orig - manual_change_from_orig) < 0.01:
        print(f"\n✅ SUCCESS: Both methods produce very similar changes from original!")
    else:
        print(f"\n⚠️  WARNING: Methods produce different amounts of change from original")
    
    # Performance benchmark
    print("\n=== Performance Benchmark ===")
    
    # Warmup
    for _ in range(10):
        _ = matmul_tma_persistent(x=x, weight=weight, weight_scale=weight_scale, warp_specialize=False)
    torch.cuda.synchronize()
    
    # Benchmark with scaling
    start = time.time()
    for _ in range(100):
        _ = matmul_tma_persistent(x=x, weight=weight, weight_scale=weight_scale, warp_specialize=False)
    torch.cuda.synchronize()
    end = time.time()
    print(f"With per-channel scaling (100 iters): {(end - start) * 1000:.3f}ms")
    
    print("\nAll cache info:", get_cache_info())