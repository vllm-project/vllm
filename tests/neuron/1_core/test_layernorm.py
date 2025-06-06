# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_tokens,hidden_size,add_residual,dtype", [
    (7, 8, False, torch.half),
    (83, 768, False, torch.half),
    (83, 768, True, torch.half),
    (83, 768, True, torch.bfloat16),
    (83, 768, True, torch.float32),
])
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
) -> None:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    current_platform.seed_everything(0)
    torch.set_default_device("cpu")
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype).to(device=device)
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    residual_cpu = residual.cpu() if add_residual else None
    ref_out = layer.to(device="cpu").forward_native(x.cpu(), residual_cpu)
    assert x.is_xla, "input tensor under testing is expected to be XLA tensor."
    out = layer.to(device=device)(x, residual)

    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        assert out[0].is_xla, "output tensor is expected to be XLA tensor"
        torch.testing.assert_close(out[0].cpu(),
                                   ref_out[0],
                                   atol=1e-2,
                                   rtol=1e-2)
        torch.testing.assert_close(out[1].cpu(),
                                   ref_out[1],
                                   atol=1e-2,
                                   rtol=1e-2)
    else:
        assert out.is_xla, "output tensor is expected to be XLA tensor"
        torch.testing.assert_close(out.cpu(), ref_out, atol=1e-2, rtol=1e-2)
