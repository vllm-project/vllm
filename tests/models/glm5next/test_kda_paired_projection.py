# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA paired f_b/g_b gate projections.

``f_b_proj`` and ``g_b_proj`` are two same-shape BF16 projections applied to
adjacent ``head_dim``-wide shards of the merged ``q|k|v|b|f_a|g_a`` GEMM
output. On the decode hot path they run as one batched GEMM whose weight is
built in ``process_weights_after_loading``. The batched path must be
bit-identical to the two separate Linears, rebuild on every post-load call
(so weight refit cannot leave it stale), and stay disabled for non-BF16
weights.
"""

import pytest
import torch

from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only BF16 GEMM"
)

HEAD_DIM = 128
LOCAL_NUM_HEADS = 8
LOCAL_PROJ_SIZE = HEAD_DIM * LOCAL_NUM_HEADS
# Row layout of the merged in_proj_qkvbfg_a output: q|k|v, b, f_a, g_a.
F_A_OFFSET = 3 * LOCAL_PROJ_SIZE + LOCAL_NUM_HEADS
ROW_WIDTH = F_A_OFFSET + 2 * HEAD_DIM


def make_layer(dtype: torch.dtype, device: torch.device):
    """A Glm5NextLinearAttention with only the paired-projection state set up;
    the full __init__ needs a VllmConfig, which these tests don't exercise.
    """
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = HEAD_DIM
    layer.local_num_heads = LOCAL_NUM_HEADS
    layer.local_projection_size = LOCAL_PROJ_SIZE
    layer.f_b_proj = torch.nn.Linear(
        HEAD_DIM, LOCAL_PROJ_SIZE, bias=False, dtype=dtype, device=device
    )
    layer.g_b_proj = torch.nn.Linear(
        HEAD_DIM, LOCAL_PROJ_SIZE, bias=False, dtype=dtype, device=device
    )
    layer._paired_fg_weight = None
    return layer


@pytest.mark.parametrize("num_tokens", [1, 5, 64, 256])
def test_paired_fg_projection_bit_identical(num_tokens: int):
    device = torch.device("cuda")
    torch.manual_seed(0)
    layer = make_layer(torch.bfloat16, device)
    layer.process_weights_after_loading(torch.bfloat16)
    assert layer._paired_fg_weight is not None

    projected = torch.randn(num_tokens, ROW_WIDTH, dtype=torch.bfloat16, device=device)
    f_a = projected[:, F_A_OFFSET : F_A_OFFSET + HEAD_DIM]
    g_a = projected[:, F_A_OFFSET + HEAD_DIM : F_A_OFFSET + 2 * HEAD_DIM]

    g1, g2 = layer._paired_fg_projections(projected)
    # The two Linears are the reference; bf16 GEMM with fp32 accumulation over
    # K=head_dim batches without changing the reduction order.
    assert torch.equal(g1, layer.f_b_proj(f_a))
    assert torch.equal(g2, layer.g_b_proj(g_a))


def test_paired_fg_weight_rebuilt_after_reload():
    device = torch.device("cuda")
    torch.manual_seed(0)
    layer = make_layer(torch.bfloat16, device)
    layer.process_weights_after_loading(torch.bfloat16)
    first = layer._paired_fg_weight

    # Simulate a weight refit: same parameter objects, new values.
    layer.f_b_proj.weight.data.mul_(0.5)
    layer.process_weights_after_loading(torch.bfloat16)
    second = layer._paired_fg_weight

    assert second is not first
    assert torch.equal(second[0], layer.f_b_proj.weight.T)
    assert torch.equal(second[1], layer.g_b_proj.weight.T)


def test_paired_fg_weight_disabled_for_non_bf16():
    device = torch.device("cuda")
    layer = make_layer(torch.float32, device)
    layer.process_weights_after_loading(torch.float32)
    assert layer._paired_fg_weight is None
