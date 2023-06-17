import torch
import torch.nn.functional as F

from vllm import activation_ops


def ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(chunks=2, dim=1)
    return F.silu(x1) * x2


@torch.inference_mode()
def run_silu_and_mul(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
) -> None:
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device='cuda')
    out = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
    activation_ops.silu_and_mul(out, x)
    ref_out = ref_silu_and_mul(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


def test_silu_and_mul() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for num_tokens in [7, 83, 2048]:
            for d in [512, 4096, 5120, 13824]:
                print(f'Testing dtype={dtype}, num_tokens={num_tokens}, d={d}')
                run_silu_and_mul(num_tokens, d, dtype)
