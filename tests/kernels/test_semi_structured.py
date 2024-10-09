import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.sparsity.utils.cusparse_2_4_utils import (
    compress_to_torch_sparse_semi_structured_mat,
    decompress_torch_sparse_semi_structured_mat,
    generate_pruned_semi_structured_mat, get_random_mat,
    is_semi_structured_supported, semi_structured_dense_sparse_T_gemm,
    semi_structured_sparse_dense_gemm,
    dense_matmul)

# DTYPES = [torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn]
DTYPES = [torch.float8_e4m3fn]
SIZES = [(128, 128), (1024, 8192)]
MNK = [(128, 128, 128), (128, 512, 1024), (512, 512, 512), (1024, 2048, 4096)]


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_semi_structured_compress(size, dtype):
    if dtype == torch.float8_e4m3fn and not is_quant_method_supported("fp8"):
        pytest.skip("fp8 is not supported on this device")
    input_pruned = generate_pruned_semi_structured_mat(*size, dtype)
    output_pruned = decompress_torch_sparse_semi_structured_mat(
        compress_to_torch_sparse_semi_structured_mat(input_pruned))
    torch.testing.assert_close(input_pruned, output_pruned)


# @pytest.mark.skipif(
#  not is_semi_structured_supported() or not is_quant_method_supported("fp8"),
#     reason="Semi structured matmul is not supported on this GPU type.")
# @pytest.mark.parametrize("size", SIZES)
# def test_torch_fp8_compress(size):
#     x = generate_pruned_semi_structured_mat(*size, torch.float8_e4m3fn)


@pytest.mark.skipif(
    not is_semi_structured_supported(),
    reason="Semi structured matmul is not supported on this GPU type.")
@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_semi_structured_sparse_dense_matmul(mnk, dtype):
    if dtype is torch.int8:
        pytest.skip("cusparse does not support sparse x non transposed dense")
    M, N, K = mnk
    A_pruned = generate_pruned_semi_structured_mat(M, K, dtype)
    A = compress_to_torch_sparse_semi_structured_mat(A_pruned)
    B = get_random_mat(K, N, dtype)
    C_sparse = semi_structured_sparse_dense_gemm(A, B)
    C = dense_matmul(A_pruned, B, dtype)
    torch.testing.assert_close(C, C_sparse)


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
