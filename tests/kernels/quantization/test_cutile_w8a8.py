import torch
import cuda.tile as ct
from cuda.tile import kernel, ByTarget
from vllm.platforms import current_platform
import vllm.kernels.cutile.cutile_w8a8
# -------------------
# Test Implementation
# -------------------
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.benchmarks.lib.utils import default_vllm_config
def test_cutile_blockwise_fp8_kernel(default_vllm_config):
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
    out = torch.ops.vllm.cutile_scaled_mm(A_fp8_cutlass, B_fp8_t, As_cutlass, Bs, out_dtype)
    rel_diff = torch.mean(torch.abs(out.float() - ref_out.float())) / torch.mean(torch.abs(ref_out.float()))
    assert rel_diff < 0.001

if __name__ == "__main__":
    test_cutile_blockwise_fp8_kernel()