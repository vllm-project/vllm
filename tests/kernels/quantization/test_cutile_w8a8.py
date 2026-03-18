import torch
import cuda.tile as ct
from cuda.tile import kernel, ByTarget
from vllm.platforms import current_platform

from vllm.model_executor.custom_op import CustomOp
TILE_M, TILE_N, TILE_K = 128, 128, 128
ConstInt = ct.Constant[int]

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

    M = A.shape[0]
    N = B.shape[1]
    bid_m, bid_n = map_block_to_tile_grouped(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))

    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k_idx in range(num_tiles_k):
        a_tile = ct.load(A, index=(bid_m, k_idx), shape=(TILE_M, TILE_K), padding_mode=zero_pad).astype(dtype)
        b_tile = ct.load(B, index=(k_idx, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad).astype(dtype)

        a_scale = ct.load(As, index=(bid_m, k_idx), shape=(TILE_M, 1))
        b_scale = ct.load(Bs, index=(bid_n, k_idx), shape=(1, 1))

        a_fp32 = ct.astype(a_tile, ct.float32) * a_scale
        b_fp32 = ct.astype(b_tile, ct.float32) * b_scale
        
        acc = ct.mma(a_fp32, b_fp32, acc)

    ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))

def cutile_blockwise_mm(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, out_dtype: torch.dtype):
    """
    A: (M, K) in fp8, row-major (stride: (K, 1))
    B: (K, N) in fp8, col-major (stride: (1, K))
    As(A_scale): (M, k_tiles) , col-major (stride: (1, M)) 
    Bs: (k_tiles, n_tiles), col-major (stride: (1, k_tiles))
    Out: (M, N) in out_dtype, row-major (stride: (N, 1))

    """
    
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
              (A, B, As, Bs, C, M, N, K, TILE_M, TILE_N, TILE_K, 1))
    return C

@CustomOp.register("cutile_scaled_mm")
class CuTileBlockwiseMM(CustomOp):
    def __init__(self,):
        super().__init__()
        if current_platform.is_cuda() and torch.cuda.get_device_capability()==(12,1):
            print("Registering cutile_blockwise_mm kernel for sm_121")
            self._forward_method = cutile_blockwise_mm
        else:
            self._forward_method = forward_native
    
    def forward_native(self, A, B, As, Bs, out_dtype) -> torch.Tensor:
        print("In forward_native of CuTileBlockwiseMM")
        M = A.shape[0]
        N = B.shape[1]
        return torch.empty((M, N), device=A.device, dtype=out_dtype)

    def forward_cuda(self, A, B, As, Bs, out_dtype) -> torch.Tensor:
        print("In forward_cuda of CuTileBlockwiseMM")
        return self.ops(A=A, B=B, As=As, Bs=Bs, out_dtype=out_dtype)



# -------------------
# Test Implementation
# -------------------
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.benchmarks.lib.utils import default_vllm_config
def test_cutile_blockwise_fp8_kernel(default_vllm_config):
    cutile_mm = CuTileBlockwiseMM()
    torch.set_default_device("cuda")
    M, N, K = 128,512,7168

    block_size = [128, 128]
    out_dtype = torch.bfloat16
    seed = 0

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp8_t = B_fp8.T.contiguous()
    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale
    
    A_fp8, As = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=False
    )
    # CUTLASS uses column-major format for scales
    A_fp8_cutlass, As_cutlass = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True
    )
    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)   
    out = cutile_mm(A_fp8_cutlass, B_fp8_t, As_cutlass, Bs, out_dtype)
    print("Reference output:", ref_out)
    rel_diff = torch.mean(torch.abs(out.float() - ref_out.float())) / torch.mean(torch.abs(ref_out.float()))
    assert rel_diff < 0.001

if __name__ == "__main__":
    test_cutile_blockwise_fp8_kernel()