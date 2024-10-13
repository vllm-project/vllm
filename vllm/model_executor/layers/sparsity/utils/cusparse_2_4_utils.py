import torch
from packaging.version import Version
from torch.sparse import (SparseSemiStructuredTensor,
                          SparseSemiStructuredTensorCUSPARSELT,
                          to_sparse_semi_structured)

from vllm import _custom_ops as ops
from vllm._custom_ops import (semi_structured_fp8_compress,
                              semi_structured_fp8_mm)
from vllm.platforms import current_platform


def compress_to_torch_sparse_semi_structured_mat(original_tensor):
    if original_tensor.dtype == torch.float8_e4m3fn:
        packed = semi_structured_fp8_compress(original_tensor)
        return SparseSemiStructuredTensorCUSPARSELT(
            shape=original_tensor.shape,
            packed=packed,
            meta=None,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            fuse_transpose_cusparselt=SparseSemiStructuredTensor.
            _FUSE_TRANSPOSE,
            alg_id_cusparselt=SparseSemiStructuredTensor._DEFAULT_ALG_ID,
            requires_grad=original_tensor.requires_grad,
        )
    else:
        return to_sparse_semi_structured(original_tensor)


def decompress_torch_sparse_semi_structured_mat(sp_mat):
    if sp_mat.dtype == torch.float8_e4m3fn:
        return semi_structured_fp8_mm(sp_mat.packed,
                                      torch.eye(sp_mat.shape[-1],
                                                dtype=sp_mat.dtype,
                                                device=sp_mat.device).t(),
                                      transpose_result=False)
    else:
        # Fix of to_dense() function supporting int8
        # cuSparseLT for int8 requires dense matrix to be non-contiguous
        return torch.mm(
            sp_mat,
            torch.eye(sp_mat.shape[-1],
                      dtype=sp_mat.dtype,
                      device=sp_mat.device).t())


def semi_structured_sparse_dense_gemm(a_sparse: torch.Tensor,
                                      b_dense: torch.Tensor):
    assert a_sparse.dtype in [
        torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn
    ], f"Semi structured sparse-dense matmul does not support {a_sparse.dtype}"
    if a_sparse.dtype == torch.float8_e4m3fn:
        return semi_structured_fp8_mm(a_sparse.packed,
                                      b_dense,
                                      transpose_result=False)
    else:
        return torch.mm(a_sparse, b_dense)


def semi_structured_dense_sparse_T_gemm(a: torch.Tensor, b_T: torch.Tensor):
    return (semi_structured_sparse_dense_gemm(b_T, a.t())).t()


# test utils
def dense_matmul(A, B, dtype):
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        return ops.cutlass_scaled_mm(A, B, scale_a, scale_b,
                                     torch.bfloat16).to(dtype)
    else:
        return A @ B


def is_semi_structured_supported() -> bool:
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    base_torch_version = Version(Version(torch.__version__).base_version)

    capability = current_platform.get_device_capability()
    assert capability is not None
    capability = capability.to_int()
    min_capability = 80

    return capability == min_capability or (
        capability > min_capability and base_torch_version >= Version("2.5.0"))


def get_random_mat(M, K, dtype):
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda()
    mat = mat.masked_fill_(mat == 0, 1)
    return mat.to(dtype)


def generate_pruned_semi_structured_mat(M, K, dtype):
    mask = torch.Tensor([0, 0, 1, 1]).tile((M, K // 4)).cuda().bool()
    rand_tensor_dtype = dtype
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda()
    mat = mat.masked_fill_(mat == 0, 1)
    mat = mat * mask
    return mat.to(dtype)
