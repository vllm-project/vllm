import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from functools import lru_cache
import vllm
import torch
import cuda.tile as ct
from cuda.tile import kernel, ByTarget
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

ConstInt = ct.Constant[int]
DEVICE_NAME = current_platform.get_device_name().lower().replace(" ", "_")


@lru_cache(maxsize=1)
def _load_master_json():
    vllm_root = os.path.dirname(vllm.__file__)
    config_path = os.path.join(
        vllm_root, "model_executor", "layers", "quantization", "utils", "configs", "cutile_w8a8.json"
    )
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


_CONFIG_INDEX = {}

def initialize_registry():
    global _CONFIG_INDEX
    raw_data = _load_master_json() # Your existing JSON loader
    
    for device, configs in raw_data.items():
        dev_key = device.lower().replace(" ", "_")
        for key, data in configs.items():
            # Index by the exact key: (device, "mperrank_1_n_...")
            _CONFIG_INDEX[(dev_key, key)] = data
            
            # Index by the shape for fallback: (device, N, K)
            parts = key.split("_")
            if len(parts) >= 6:
                try:
                    n_val, k_val = int(parts[3]), int(parts[5])
                    _CONFIG_INDEX[(dev_key, n_val, k_val)] = data
                except: continue

# Initialize immediately
initialize_registry()

# This function,(adapted from triton/cutile) maps a linear Block ID (bid) 
# to a 2D tile coordinate (bid_m, bid_n).
# We group tiles along the M dimension to optimize memory access patterns:
#   - In Matrix A: Each block handles a specific row-tile strip [bid_m, :].
#   - In Matrix B: Multiple blocks in a group share the same column-tile [:, bid_n].
# So instead of loading the whole matrix B to compute a single row of matrix A, 
# we process a group of M-rows together to achieve the same N elements 
# with better data reuse from the L2 cache.
def map_block_to_tile_grouped(M, N, tm, tn, GROUP_SIZE_M):
    bid = ct.bid(0) # block id
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    
    num_tiles_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_tiles_in_group
    
    first_tile_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_tile_m, GROUP_SIZE_M)
    
    bid_m = first_tile_m + ((bid % num_tiles_in_group) % group_size_m)
    bid_n = (bid % num_tiles_in_group) // group_size_m
    
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_121=4))
def matmul_kernel(A, B, As, Bs, C,
                                 M: ConstInt, N: ConstInt, K: ConstInt,
                                 TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
                                 GROUP_SIZE_M:ConstInt = 1 ):

    bid_m, bid_n = map_block_to_tile_grouped(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.cdiv(K, TILE_K)

    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k_idx in range(num_tiles_k):
        a_tile = ct.load(A, index=(bid_m, k_idx), shape=(TILE_M, TILE_K), padding_mode=zero_pad)
        b_tile = ct.load(B, index=(k_idx, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad)

        a_scale = ct.load(As, index=(bid_m, k_idx), shape=(TILE_M, 1))
        b_scale = ct.load(Bs, index=(k_idx, bid_n), shape=(1, 1))

        a_fp32 = ct.astype(a_tile, ct.float32) * a_scale
        b_fp32 = ct.astype(b_tile, ct.float32) * b_scale
        
        acc = ct.mma(a_fp32, b_fp32, acc)

    ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))

def cutile_blockwise_mm(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, out_dtype: torch.dtype)-> torch.Tensor:
    """
    A: (M, K) in fp8, row-major (stride: (K, 1))
    B: (K, N) in fp8, col-major (stride: (1, K))
    As(A_scale): (M, k_tiles) , col-major (stride: (1, M)) 
    Bs: (k_tiles, n_tiles), col-major (stride: (1, k_tiles))
    Out: (M, N) in out_dtype, row-major (stride: (N, 1))

    """
    M, K = A.size()
    K_, N = B.size()
    exact_key = f"mperrank_{M}_n_{N}_k_{K}"
    config = _CONFIG_INDEX.get((DEVICE_NAME, exact_key))

    # if config is None:
    #     config = _CONFIG_INDEX.get((DEVICE_NAME, int(N), int(K)))
    
    if config:
        # Load tuned parameters
        TILE_M, TILE_N, TILE_K, GROUP_SIZE_M = config["block_sizes"]
        #print(f"Using tuned config: device={DEVICE_NAME}, M={M}, N={N}, K={K}, TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}, GROUP_SIZE_M={GROUP_SIZE_M}")
    else:
        # Fallback to safe defaults if not tuned for this exact M
        TILE_M, TILE_N, TILE_K, GROUP_SIZE_M = 128, 128, 128, 1
        #print ("cuTile Default: No config found for this shape, using default TILE_M=128, TILE_N=128, TILE_K=128, GROUP_SIZE_M=1")
    #assert B.is_contiguous(), "B must be contiguous"
    #assert Bs.T.is_contiguous(), "Bs must be contiguous"

    assert As.dtype == torch.float32, "As must be float32"
    assert Bs.dtype == torch.float32, "Bs must be float32"

    M, K = A.shape
    K_check, N = B.shape 
    
    assert K == K_check, f"Inner dimension mismatch: A_K={K}, B_K={K_check}"

    C = torch.empty((M, N), dtype=out_dtype, device=A.device)
    grid_m = ct.cdiv(M, TILE_M)
    grid_n = ct.cdiv(N, TILE_N)
    grid_1d = (grid_m * grid_n, 1, 1)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    
    ct.launch(stream_ptr, grid_1d, matmul_kernel, 
              (A, B, As, Bs, C, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M))
    return C

def cutile_scaled_mm_fake(self, A, B, As, Bs, out_dtype) -> torch.Tensor:
    print("In forward_native of CuTileBlockwiseMM")
    M = A.shape[0]
    N = B.shape[1]
    return torch.empty((M, N), device=A.device, dtype=out_dtype)

direct_register_custom_op(
    op_name="cutile_scaled_mm",
    op_func=cutile_blockwise_mm,
    fake_impl=cutile_scaled_mm_fake,
)
