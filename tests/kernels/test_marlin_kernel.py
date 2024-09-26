import vllm._custom_ops
import torch
import pytest

@pytest.mark.parametrize("size_m", list(range(64, 1024, 32)))
def test_marlin_gemm_opcheck(size_m):
    
    a  = torch.rand((size_m, 4096), device='cuda', dtype=torch.float16)
    w = torch.ones((256, 8192), device='cuda', dtype=torch.int32)
    s = torch.rand((32, 4096), device='cuda', dtype=torch.float16)
    wk = torch.zeros((8192*16,), device='cuda', dtype=torch.int32)

    size_n = 4096
    size_k = 4096
    x = torch.ops._C.marlin_gemm(a, w, s, wk, size_m, size_n, size_k)
    torch.cuda.synchronize()
    wk = torch.zeros((8192*16,), device='cuda', dtype=torch.int32)
    y = torch.ops._C.marlin_gemm(a, w, s, wk, size_m, size_n, size_k)
    torch.cuda.synchronize()
                                                              
    torch.testing.assert_close(x,y, atol=1, rtol=1e-1, equal_nan=True)
    #opcheck(torch.ops._C.marlin_gemm, (a, w, s, wk, size_m, size_n, size_k))