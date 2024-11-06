import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    decompress_torch_sparse_semi_structured_mat, dense_matmul,
    generate_pruned_semi_structured_mat, get_random_mat,
    is_semi_structured_supported, semi_structured_dense_sparse_T_gemm,
    semi_structured_dense_sparse_T_gemm_scaled,
    semi_structured_sparse_dense_gemm,
    semi_structured_sparse_dense_gemm_scaled,
    clear_cache)

DTYPES = [torch.float16, torch.bfloat16, torch.int8]
SIZES = [(128, 128), (1024, 8192)]
SIZES_FP8 = [(32, 64), (1024, 1024)]
MNK = [(128, 128, 128), (128, 512, 1024), (512, 512, 512), (1024, 2048, 4096)]


# From pytorch test
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_semi_structured_compress(size, dtype):
    input_pruned = generate_pruned_semi_structured_mat(*size, dtype)
    output_pruned = decompress_torch_sparse_semi_structured_mat(
        compress_to_torch_sparse_semi_structured_mat(input_pruned))
    torch.testing.assert_close(input_pruned, output_pruned)


# TODO modelopt config has to be replaced with corresponding fp8_24 config
@pytest.mark.skipif(
    not is_semi_structured_supported()
    or not is_quant_method_supported("modelopt"),
    reason="Semi structured fp8 matmul is not supported on this GPU type.")
@pytest.mark.parametrize("size", SIZES_FP8)
def test_semi_structured_fp8_compress(size):
    dtype = torch.float8_e4m3fn
    input_pruned = generate_pruned_semi_structured_mat(*size, dtype)
    output_pruned = decompress_torch_sparse_semi_structured_mat(
        compress_to_torch_sparse_semi_structured_mat(input_pruned))
    torch.testing.assert_close(input_pruned.to(torch.float32),
                               output_pruned.to(torch.float32),
                               rtol=1e-1,
                               atol=1e-1)


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_semi_structured_sparse_dense_matmul(mnk, dtype):
    M, N, K = mnk
    A_pruned = generate_pruned_semi_structured_mat(M, K, dtype)
    A = compress_to_torch_sparse_semi_structured_mat(A_pruned)
    B = get_random_mat(K, N, dtype)
    if dtype is torch.int8:
        with pytest.raises(ValueError):
            C_sparse = semi_structured_sparse_dense_gemm(A, B)
    else:
        C_sparse = semi_structured_sparse_dense_gemm(A, B)
        C = dense_matmul(A_pruned, B, dtype)
        torch.testing.assert_close(C, C_sparse)

        # Verify cache
        B = get_random_mat(K, N, dtype)
        C = dense_matmul(A_pruned, B, dtype)
        C_sparse = semi_structured_sparse_dense_gemm(A, B)
        torch.testing.assert_close(C, C_sparse)

        C_sparse = semi_structured_sparse_dense_gemm(A, B, cached=False)
        torch.testing.assert_close(C, C_sparse)
        clear_cache()


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_semi_structured_sparse_dense_T_matmul(mnk, dtype):
    M, N, K = mnk
    A_pruned = generate_pruned_semi_structured_mat(M, K, dtype)
    A = compress_to_torch_sparse_semi_structured_mat(A_pruned)
    B = get_random_mat(N, K, dtype)

    C_sparse = semi_structured_sparse_dense_gemm(A, B.t())
    C = dense_matmul(A_pruned, B.t(), dtype)
    torch.testing.assert_close(C, C_sparse)

    # Verify cache
    B = get_random_mat(N, K, dtype)
    C = dense_matmul(A_pruned, B.t(), dtype)
    C_sparse = semi_structured_sparse_dense_gemm(A, B.t())
    torch.testing.assert_close(C, C_sparse)

    C_sparse = semi_structured_sparse_dense_gemm(A, B.t(), cached=False)
    torch.testing.assert_close(C, C_sparse)
    clear_cache()


# TODO modelopt config has to be replaced with corresponding fp8_24 config
@pytest.mark.skipif(
    not is_semi_structured_supported()
    or not is_quant_method_supported("modelopt"),
    reason="Semi structured fp8 matmul is not supported on this GPU type.")
def test_torch_semi_structured_sparse_dense_T_fp8_matmul():
    M, N, K = (32, 64, 32)
    dtype = torch.float8_e4m3fn
    A_pruned = generate_pruned_semi_structured_mat(M, N, dtype=dtype)
    A = compress_to_torch_sparse_semi_structured_mat(A_pruned)
    B = torch.full((K, N), .25, device='cuda', dtype=dtype).t()

    C = dense_matmul(A_pruned, B, dtype=dtype).to(torch.float32)
    C_sparse = semi_structured_sparse_dense_gemm(A, B).to(torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)

    # Cached version
    B = torch.full((K, N), .25, device='cuda', dtype=dtype).t()
    C = dense_matmul(A_pruned, B, dtype=dtype).to(torch.float32)
    C_sparse = semi_structured_sparse_dense_gemm(A, B).to(torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)

    # Noncached version
    C_sparse = semi_structured_sparse_dense_gemm(A, B, cached=False).to(
        torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)
    clear_cache()


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_semi_structured_dense_sparse_T_matmul(mnk, dtype):
    M, N, K = mnk
    B_T_pruned = generate_pruned_semi_structured_mat(N, K, dtype)
    B_T = compress_to_torch_sparse_semi_structured_mat(B_T_pruned)
    A = get_random_mat(M, K, dtype)

    C_sparse = semi_structured_dense_sparse_T_gemm(A, B_T)
    C = dense_matmul(A, B_T_pruned.t(), dtype)
    torch.testing.assert_close(C, C_sparse)

    C_sparse = semi_structured_dense_sparse_T_gemm(A, B_T, cached=False)
    C = dense_matmul(A, B_T_pruned.t(), dtype)
    torch.testing.assert_close(C, C_sparse)
    clear_cache()


# TODO modelopt config has to be replaced with corresponding fp8_24 config
@pytest.mark.skipif(
    not is_semi_structured_supported()
    or not is_quant_method_supported("modelopt"),
    reason="Semi structured fp8 matmul is not supported on this GPU type.")
def test_torch_semi_structured_dense_sparse_T_fp8_matmul():
    M, N, K = (32, 64, 32)
    dtype = torch.float8_e4m3fn
    B_T_pruned = generate_pruned_semi_structured_mat(N, K, dtype=dtype)
    B_T = compress_to_torch_sparse_semi_structured_mat(B_T_pruned)
    A = torch.full((M, K), .25, device='cuda', dtype=dtype)

    C_sparse = semi_structured_dense_sparse_T_gemm(A, B_T).to(torch.float32)
    C = dense_matmul(A, B_T_pruned.t(), dtype=dtype).to(torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)

    C_sparse = semi_structured_dense_sparse_T_gemm(A, B_T).to(torch.float32)
    C = dense_matmul(A, B_T_pruned.t(), dtype=dtype).to(torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)
    clear_cache()


@pytest.mark.skipif(
    not is_semi_structured_supported()
    or not is_quant_method_supported("modelopt"),
    reason="Semi structured fp8 matmul is not supported on this GPU type.")
def test_torch_semi_structured_sparse_dense_T_fp8_scaled_matmul():
    M, N, K = (32, 64, 32)
    A_pruned = generate_pruned_semi_structured_mat(M, N, dtype=torch.float16)
    A_pruned_fp8, scale_A = to_float8(A_pruned)
    B = torch.rand((K, N), device='cuda').to(torch.float16).t()
    B_fp8, scale_B = to_float8(B)

    A_fp8_sparse = compress_to_torch_sparse_semi_structured_mat(A_pruned_fp8)

    C = torch._scaled_mm(A_pruned_fp8,
                         B_fp8,
                         scale_a=scale_A,
                         scale_b=scale_B,
                         out_dtype=torch.float32)
    C_sparse = semi_structured_sparse_dense_gemm_scaled(A_fp8_sparse,
                                                        B_fp8,
                                                        scale_a=scale_A,
                                                        scale_b=scale_B).to(
                                                            torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=7e-2, atol=7e-2)

    # cached
    B = torch.rand((K, N), device='cuda').to(torch.float16).t()
    B_fp8, scale_B = to_float8(B)

    C = torch._scaled_mm(A_pruned_fp8,
                         B_fp8,
                         scale_a=scale_A,
                         scale_b=scale_B,
                         out_dtype=torch.float32)
    C_sparse = semi_structured_sparse_dense_gemm_scaled(A_fp8_sparse,
                                                        B_fp8,
                                                        scale_a=scale_A,
                                                        scale_b=scale_B).to(
                                                            torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=7e-2, atol=7e-2)

    # noncached
    C_sparse = semi_structured_sparse_dense_gemm_scaled(A_fp8_sparse,
                                                        B_fp8,
                                                        scale_a=scale_A,
                                                        scale_b=scale_B,
                                                        cached=False).to(
                                                            torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=7e-2, atol=7e-2)
    clear_cache()


@pytest.mark.skipif(
    not is_semi_structured_supported()
    or not is_quant_method_supported("modelopt"),
    reason="Semi structured fp8 matmul is not supported on this GPU type.")
def test_torch_semi_structured_dense_sparse_T_fp8_scaled_matmul():
    M, N, K = (32, 64, 32)
    A = torch.rand((M, K), device='cuda', dtype=torch.float16)
    A_fp8, scale_a = to_float8(A)
    B_T_pruned = generate_pruned_semi_structured_mat(N, K, dtype=torch.float16)
    B_T_pruned_fp8, scale_b = to_float8(B_T_pruned)
    B_T_packed = compress_to_torch_sparse_semi_structured_mat(B_T_pruned_fp8)

    C_sparse = semi_structured_dense_sparse_T_gemm_scaled(A_fp8,
                                                          B_T_packed,
                                                          scale_a=scale_a,
                                                          scale_b=scale_b).to(
                                                              torch.float32)
    C = torch._scaled_mm(B_T_pruned_fp8,
                         A_fp8.t(),
                         scale_a=scale_b,
                         scale_b=scale_a,
                         out_dtype=torch.float32).t()
    torch.testing.assert_close(C, C_sparse, rtol=7e-2, atol=7e-2)
    clear_cache()


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
def test_torch_semi_structured_sparse_dense_t_int8_scaled_matmul():
    dtype = torch.int8
    M, N, K = (32, 64, 32)
    A_pruned = generate_pruned_semi_structured_mat(M, K, dtype)
    A = compress_to_torch_sparse_semi_structured_mat(A_pruned)
    B = get_random_mat(N, K, dtype)

    scale_a = torch.tensor(2.0, dtype=torch.float32, device='cuda')
    scale_b = torch.tensor(2.0, dtype=torch.float32, device='cuda')

    C = dense_matmul(A_pruned,
                     B.t(),
                     dtype=dtype,
                     scale_a=scale_a,
                     scale_b=scale_b).to(torch.float32)
    C_sparse = semi_structured_sparse_dense_gemm_scaled(A,
                                                        B.t(),
                                                        scale_a=scale_a,
                                                        scale_b=scale_b).to(
                                                            torch.float32)
    torch.testing.assert_close(C, C_sparse, rtol=1e-1, atol=1e-1)
    clear_cache()
