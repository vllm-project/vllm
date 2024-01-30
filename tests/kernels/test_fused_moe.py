import pytest
import torch

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul


def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[1],
                      dtype=a.dtype,
                      device=a.device)
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1)).sum(dim=1)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)

    triton_output = fused_moe(a, w1, w2, topk_weight, topk_ids, False)
    torch_output = torch_moe(a, w1, w2, topk_weight, topk_ids)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
