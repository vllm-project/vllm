import itertools
import random
import pytest
import torch

from vllm.model_executor.layers.moe import grouped_matmul


def ref_grouped_matmul(
    fused_input: torch.Tensor,
    cum_group_range: torch.Tensor,
    fused_group_b: torch.Tensor,
    activation: str = "",
) -> torch.Tensor:
    num_groups = cum_group_range.shape[0] - 1
    output = torch.zeros(fused_input.shape[0],
                         fused_group_b.shape[2],
                         device=fused_input.device,
                         dtype=fused_input.dtype)
    for i in range(num_groups):
        group_i = fused_input[cum_group_range[i]:cum_group_range[i + 1]]
        group_i_b = fused_group_b[i]
        output[cum_group_range[i]:cum_group_range[i + 1]] = group_i @ group_i_b

    if activation == "silu":
        output = torch.nn.functional.silu(output)
    return output


@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("m", [1, 5, 33, 81])
@pytest.mark.parametrize("n", [128, 1024, 2000])
@pytest.mark.parametrize("k", [128, 1024, 2000])
@pytest.mark.parametrize("activation", ["", "silu"])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.float32, torch.bfloat16])
def test_moe_grouped_matmul(
    group_size: int,
    m: int,
    n: int,
    k: int,
    activation: str,
    dtype: torch.dtype,
):
    groups = [random.randint(1, m) for _ in range(group_size)]
    batch_size = sum(groups)
    fused_input = torch.randn(batch_size, k, dtype=dtype, device="cuda")
    cum_group_range = torch.tensor([0] + list(itertools.accumulate(groups)),
                                   dtype=torch.int32,
                                   device="cuda")
    fused_group_b = torch.randn(group_size, k, n, dtype=dtype, device="cuda")

    ref_output = ref_grouped_matmul(fused_input, cum_group_range,
                                    fused_group_b, activation)

    output = grouped_matmul(fused_input, cum_group_range, fused_group_b,
                            activation)
    assert torch.allclose(output, ref_output, atol=1e-2, rtol=0)
