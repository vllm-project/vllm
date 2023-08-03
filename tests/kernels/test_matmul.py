import torch

from vllm.ops.matmul import matmul
from vllm.ops.selective_matmul import selective_matmul


def test_mat_mul() -> None:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)


def test_selective_mat_mul() -> None:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    c = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    index = torch.arange(0, 512, device='cuda', dtype=torch.int32)
    triton_output = selective_matmul(a, b, index, c)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)