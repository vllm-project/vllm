# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.model_executor.hw_agnostic.custom_op import CustomOp


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


# Activation-and-mul ops keyed by HF activation name.
_ACTIVATION_AND_MUL_REGISTRY = {
    "silu": SiluAndMul,
    "swish": SiluAndMul,
}


def get_act_and_mul_fn(act_fn_name: str) -> CustomOp:
    """Build the hw-agnostic activation-and-mul op named `act_fn_name`."""
    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name.lower()]()
