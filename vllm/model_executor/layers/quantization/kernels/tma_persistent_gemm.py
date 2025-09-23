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
        self.cache: Dict[Tuple, Tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]] = {}
        
    def get_descriptors(self, M: int, N: int, K: int, 
                       BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int,
                       EPILOGUE_SUBTILE: bool,
                       a_device: torch.device, b_device: torch.device, c_device: torch.device) -> Tuple[TensorDescriptor, TensorDescriptor, TensorDescriptor]:
        """Get cached descriptors or create new ones."""
        key = (M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, EPILOGUE_SUBTILE, str(a_device), str(b_device), str(c_device))
        
        if key not in self.cache:
            # Create template tensors for descriptor creation
            a_template = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=a_device)
            b_template = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=b_device)
            c_template = torch.empty((M, N), dtype=torch.bfloat16, device=c_device)
            
            # Pre-compute shapes, strides, and block shapes
            a_shape, a_stride = a_template.shape, a_template.stride()
            b_shape, b_stride = b_template.shape, b_template.stride()
            c_shape, c_stride = c_template.shape, c_template.stride()
            
            a_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K]
            b_block_shape = [BLOCK_SIZE_N, BLOCK_SIZE_K]
            c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2] if EPILOGUE_SUBTILE else [BLOCK_SIZE_M, BLOCK_SIZE_N]
            
            # Create TensorDescriptors
            a_desc = TensorDescriptor(a_template, a_shape, a_stride, a_block_shape)
            b_desc = TensorDescriptor(b_template, b_shape, b_stride, b_block_shape)
            c_desc = TensorDescriptor(c_template, c_shape, c_stride, c_block_shape)
            
            self.cache[key] = (a_desc, b_desc, c_desc)
            
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



# @triton.autotune(
#     configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
#     key=["M", "N", "K", "WARP_SPECIALIZE"],
# )
# @triton.jit(launch_metadata=_matmul_launch_metadata)
@triton.jit(do_not_specialize=['M','N','K'])
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
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

    tile_id_c = start_pid - NUM_SMS
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
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


def matmul_tma_persistent(a, b, bias=None, warp_specialize=False):
    """Optimized TMA persistent matmul with CUDA graph capture support.

    Args:
        a: Input tensor (M, K) in bf16 format
        b: Weight tensor (N, K) in FP8E4M3FN format
        bias: Optional bias tensor (N,) to add to output
        warp_specialize: Enable warp specialization

    Returns:
        Output tensor (M, N) in BF16 format
    """
    # Check constraints - inputs should already be BF16
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == torch.bfloat16, f"Expected BF16 input, got {a.dtype}"
    assert b.dtype == torch.float8_e4m3fn, f"Expected FP8 weight, got {b.dtype}"

    M, K = a.shape
    N, K_b = b.shape

    # # Static configuration for optimal performance
    # BLOCK_SIZE_M = 16
    # BLOCK_SIZE_N = 512
    # BLOCK_SIZE_K = 64
    # GROUP_SIZE_M = 1
    # EPILOGUE_SUBTILE = True
    # num_stages = 4
    # num_warps = 4

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

    # Get cached descriptors (enables graph capture)
    a_desc, b_desc, c_desc = _descriptor_cache.get_descriptors(
        M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        EPILOGUE_SUBTILE, a.device, b.device, a.device
    )

    # Update descriptors with actual tensors (minimal overhead)
    a_desc.tensor = a
    b_desc.tensor = b
    c_desc.tensor = c

    # Use static grid function (enables graph capture)
    grid_size = _get_static_grid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_SMS)

    matmul_kernel_tma_persistent[grid_size](
        a_desc, b_desc, c_desc,
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
    assert a.dtype == b.dtype, "Incompatible dtypes"

    if a.dtype != torch.float8_e4m3fn:
        a = a.to(torch.float8_e4m3fn)

    if b.dtype != torch.float8_e4m3fn:
        b = b.to(torch.float8_e4m3fn)

    M, K = a.shape
    N, K = b.shape

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    GROUP_SIZE_M = 4
    EPILOGUE_SUBTILE = True
    num_stages = 4
    num_warps = 8

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
    a = torch.randn((M, K), device="cuda", dtype=dtype).to(torch.float8_e4m3fn)
    b = torch.randn((N, K), device="cuda", dtype=dtype).to(torch.float8_e4m3fn)

    # warmup
    c = matmul_tma_persistent(a, b, warp_specialize=False)
    torch.cuda.synchronize()

    # benchmark
    start = time.time()
    c = matmul_tma_persistent(a, b, warp_specialize=False)
    torch.cuda.synchronize()
    end = time.time()
    print(f"TMA persistent matmul time: {end - start:.6f}s")