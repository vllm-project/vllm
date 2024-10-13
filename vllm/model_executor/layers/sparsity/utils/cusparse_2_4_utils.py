import torch
from packaging.version import Version
from torch.sparse import (SparseSemiStructuredTensor,
                          SparseSemiStructuredTensorCUSPARSELT,
                          to_sparse_semi_structured)
from torch.sparse import (SparseSemiStructuredTensor,
                          SparseSemiStructuredTensorCUSPARSELT,
                          to_sparse_semi_structured)

from vllm import _custom_ops as ops
from vllm._custom_ops import (semi_structured_fp8_compress,
                              semi_structured_fp8_mm)
from vllm.platforms import current_platform

SparseSemiStructuredTensor._FORCE_CUTLASS = False

def compress_to_torch_sparse_semi_structured_mat(pruned_tensor: torch.Tensor):
    '''
    Compresses original pruned (with zeros) tensor into packed version
    Args:
        pruned_tensor(torch.Tensor) - pruned but not packed tensor
    Returns: 
        torch.SparseSemiStructuredTensorCUSPARSELT: torch wrapped cusparseLt-packed tensor. 
    '''
    
    if pruned_tensor.dtype == torch.float8_e4m3fn:
        packed = semi_structured_fp8_compress(pruned_tensor)
        return SparseSemiStructuredTensorCUSPARSELT(
            shape=pruned_tensor.shape,
            packed=packed,
            meta=None,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            fuse_transpose_cusparselt=SparseSemiStructuredTensor.
            _FUSE_TRANSPOSE,
            alg_id_cusparselt=SparseSemiStructuredTensor._DEFAULT_ALG_ID,
            requires_grad=pruned_tensor.requires_grad,
        )
    else:
        return to_sparse_semi_structured(pruned_tensor)


def decompress_torch_sparse_semi_structured_mat(packed_tensor: torch.Tensor):
    '''
    Unpacks the cusparseLt packed tensor into pruned tensor
    Args:
        packed_tensor - torch wrapped cusparseLt-packed tensor. Result of compress_to_torch_sparse_semi_structured_mat.
    Returns:
        pruned (torch.Tensor) - pruned torch.tensor
    '''
    if packed_tensor.dtype == torch.float8_e4m3fn:
        return semi_structured_fp8_mm(packed_tensor.packed,
                                      torch.eye(packed_tensor.shape[-1],
                                                dtype=packed_tensor.dtype,
                                                device=packed_tensor.device).t(),
                                      transpose_result=False)
    else:
        # Fix of to_dense() function supporting int8
        # cuSparseLT for int8 requires dense matrix to be non-contiguous
        return torch.mm(
            packed_tensor,
            torch.eye(packed_tensor.shape[-1],
                      dtype=packed_tensor.dtype,
                      device=packed_tensor.device).t())


def semi_structured_sparse_dense_gemm(a_packed: torch.Tensor,
                                      b_dense: torch.Tensor,
                                      bias: torch.Tensor = None):
    '''
    Performs matrix multiplication (A @ B) of semi-structured sparse (A) and dense (B) matrices.
    In case of int8 and fp8 types, dense matrix B has to be non-contiguous.
    Args:
        a_packed (torch.Tensor) - torch wrapped cusparseLt-packed tensor. Result of compress_to_torch_sparse_semi_structured_mat.
        b_dense (torch.Tensor) - dense matrix tensor.
        bias (torch.Tensor) - bias to fuse in matrix multiplication. default : None. 
    Result:
        torch.Tensor - Result of matrix multiplication.
    '''
    assert a_packed.dtype in [
        torch.float16, torch.bfloat16, torch.int8, torch.float8_e4m3fn
    ], f"Semi structured sparse-dense matmul does not support {a_packed.dtype}"
    if b_dense.is_contiguous() and a_packed.dtype in [torch.int8, torch.float8_e4m3fn]:
        raise ValueError("cuSparseLt does not support contiguous dense matrix for int8 and fp8 types")
    if a_packed.dtype == torch.float8_e4m3fn:
        return semi_structured_fp8_mm(a_packed.packed,
                                      b_dense,
                                      bias=bias,
                                      transpose_result=False)
    else:
        return torch.mm(a_packed, b_dense)


def semi_structured_dense_sparse_T_gemm(a_dense: torch.Tensor,
                                        b_T_packed: torch.Tensor,
                                        bias: torch.Tensor = None):
    '''
    Performs matrix multiplication (a @ b_T) of transposed semi-structured sparse and dense matrices
    Args:
        a_dense (torch.Tensor) - dense matrix tensor.
        b_T_packed (torch.Tensor) - torch wrapped cusparseLt-packed tensor. Result of compress_to_torch_sparse_semi_structured_mat
        bias (torch.Tensor) - bias to fuse in matrix multiplication. default : None.
    
    Returns:
        torch.Tensor - Result of matrix multiplication.
    '''
    return (semi_structured_sparse_dense_gemm(b_T_packed, a_dense.t(), bias)).t()


def semi_structured_sparse_dense_gemm_scaled(a_packed: torch.Tensor,
                                             b_dense: torch.Tensor,
                                             scale_a: torch.Tensor,
                                             scale_b: torch.Tensor,
                                             bias: torch.Tensor = None):
    '''
    Performs scaled matrix multiplication (a @ b) of transposed semi-structured sparse and dense fp8 matrices
    Args:
        a_packed (torch.Tensor) - torch wrapped cusparseLt-packed tensor. Result of compress_to_torch_sparse_semi_structured_mat.
        b_dense (torch.Tensor) - dense matrix tensor.
        scale_a (torch.Tensor) - scaling factor for sparse matrix, must be in float32.
        scale_b (torch.Tensor) - scaling factor for dense matrix, must be in float32.
        bias (torch.Tensor) - bias to fuse in matrix multiplication. default : None.

    Returns:
        torch.Tensor - Result of matrix multiplication.
    '''

    assert (a_packed.dtype == torch.float8_e4m3fn
            and b_dense.dtype == torch.float8_e4m3fn)
    assert not b_dense.is_contiguous(
    ), "cusparseLt requires dense matrix be non-contiguous"
    # cusparseLt requires alpha to be float
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    return semi_structured_fp8_mm(a_packed.packed,
                                  b_dense,
                                  alpha=scale_a * scale_b,
                                  bias=bias,
                                  transpose_result=False)


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
