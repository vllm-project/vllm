import torch
from torch.sparse import to_sparse_semi_structured
from vllm.platforms import current_platform
from packaging.version import Version

def compress_to_torch_sparse_semi_structured_mat(mat):
    return to_sparse_semi_structured(mat)

def decompress_torch_sparse_semi_structured_mat(sp_mat):
    # Fix of to_dense() function supporting int8
    # cuSparseLT for int8 requires dense matrix to be non-contiguous
    return torch.mm(sp_mat, torch.eye(sp_mat.shape[-1], dtype=sp_mat.dtype, device=sp_mat.device).t())

def semi_structured_sparse_dense_gemm(
    a_sparse: torch.Tensor, b_dense: torch.Tensor
):
    return torch.mm(a_sparse, b_dense)

def semi_structured_dense_sparse_T_gemm(
    a: torch.Tensor, b_T: torch.Tensor
):
    return (semi_structured_sparse_dense_gemm(b_T, a.t())).t()

def is_semi_structured_supported() -> bool:
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    base_torch_version = Version(Version(torch.__version__).base_version)
    
    capability = current_platform.get_device_capability()
    assert capability is not None
    capability = capability.to_int()
    min_capability = 80

    return capability == min_capability or (capability > min_capability and base_torch_version >= Version("2.5.0"))

def get_random_mat(M, K, dtype):
    rand_tensor_dtype = dtype
    if dtype is torch.int8:
        rand_tensor_dtype = torch.float16
    mat = torch.rand(M, K, dtype=rand_tensor_dtype).cuda().to(dtype)
    return mat

def generate_pruned_semi_structured_mat(M, K, dtype):

    mask = torch.Tensor([0, 0, 1, 1]).tile((M, K // 4)).cuda().bool()
    mat = get_random_mat(M, K, dtype)
    mat = mat.masked_fill_(mat == 0, 1)
    return mat * mask
