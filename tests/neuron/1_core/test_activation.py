# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.activation import FastGELU, SiluAndMul
from vllm.platforms import current_platform


@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_fast"])
@pytest.mark.parametrize("num_tokens,d,dtype", [
    (7, 512, torch.half),
    (7, 512, torch.float),
    (83, 512, torch.half),
])
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
) -> None:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    current_platform.seed_everything(0)
    torch.set_default_device("cpu")
    x = torch.randn(num_tokens, 2 * d, dtype=dtype).to(device=device)
    if activation == "silu_and_mul":
        layer = SiluAndMul()
        fn = layer.forward_native
    elif activation == "gelu_fast":
        layer = FastGELU()
        fn = F.gelu
    else:
        raise NotImplementedError(
            f"activation {activation} is not implemented.")
    assert x.is_xla, "input tensor under testing is expected to be XLA tensor."
    out = layer.to(device=device).forward_neuron(x)
    ref_out = fn(x.cpu())
    torch.testing.assert_close(out.cpu(), ref_out, atol=0.01, rtol=0.0)
