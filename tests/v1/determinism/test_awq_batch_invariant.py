# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from utils import skip_if_not_cuda

from vllm.model_executor.determinism.batch_invariant_configs import (
    resolve_tuned_matmul_configs,
)
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQConfig,
    AutoAWQLinearMethod,
)


@skip_if_not_cuda
@pytest.mark.parametrize("num_tokens", [7, 64, 300])
def test_awq_linear_is_batch_invariant_under_compile(num_tokens: int):
    """A row's output must not depend on the other rows once compiled.

    Inductor lowers torch.matmul itself, bypassing the aten overrides.
    """
    torch.manual_seed(0)
    in_features, out_features, group_size = 4096, 4096, 128
    method = AutoAWQLinearMethod(
        AutoAWQConfig(
            weight_bits=4,
            group_size=group_size,
            zero_point=True,
            lm_head_quantized=False,
        )
    )
    pack_factor = method.quant_config.pack_factor
    int32 = torch.iinfo(torch.int32)
    layer = torch.nn.Module()
    layer.qweight = torch.randint(
        int32.min,
        int32.max,
        (in_features, out_features // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    layer.qzeros = torch.randint(
        int32.min,
        int32.max,
        (in_features // group_size, out_features // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    layer.scales = (
        torch.rand(
            in_features // group_size,
            out_features,
            dtype=torch.float16,
            device="cuda",
        )
        * 1e-2
    )

    # vLLM resolves these at startup, before torch.compile traces the model.
    resolve_tuned_matmul_configs()
    compiled = torch.compile(
        lambda x: method.apply(layer, x), dynamic=True, fullgraph=True
    )
    x = torch.randn(num_tokens, in_features, dtype=torch.float16, device="cuda")

    assert torch.equal(compiled(x)[:1], compiled(x[:1]))
