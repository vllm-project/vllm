# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, ClassVar

import torch

from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

_HC = 4
_HIDDEN_SIZE = 2560
_HYPER_HIDDEN_SIZE = _HC * _HIDDEN_SIZE
_LORA_RANK = 320


class HCSiluUpGateMixOp:
    """Process-local compiled fused HyperConnection decode operation."""

    _instance: ClassVar[HCSiluUpGateMixOp | None] = None

    @classmethod
    def initialize(cls) -> HCSiluUpGateMixOp:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def is_supported(dtype: torch.dtype) -> bool:
        if dtype != torch.bfloat16 or not current_platform.is_device_capability((9, 0)):
            return False
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self) -> None:
        self._compiled: Any | None = None
        register_cutedsl_warmup_provider(self)

    def _compile(self) -> None:
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from quack.compile_utils import make_fake_tensor

        from ._hc_silu_up_gate_mix import HCSiluUpGateMixKernel

        lora = make_fake_tensor(
            cutlass.BFloat16,
            (1, _LORA_RANK),
            divisibility=2,
        )
        weight = make_fake_tensor(
            cutlass.BFloat16,
            (_HYPER_HIDDEN_SIZE, _LORA_RANK),
            divisibility=2,
        )
        x = make_fake_tensor(
            cutlass.BFloat16,
            (1, _HYPER_HIDDEN_SIZE),
            divisibility=1,
        )
        out = make_fake_tensor(
            cutlass.BFloat16,
            (1, _HIDDEN_SIZE),
            divisibility=1,
        )
        self._compiled = cute.compile(
            HCSiluUpGateMixKernel(),
            lora,
            weight,
            x,
            out,
            CUstream(current_stream().cuda_stream),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        return (
            CuTeDSLCompileUnit(
                name="Qwen4Exp fused HC SiLU/up-projection/gate-mix",
                key=("qwen4-exp-hc-silu-up-gate-mix", torch.bfloat16),
                compile=self._compile,
            ),
        )

    def __call__(
        self,
        lora: torch.Tensor,
        weight: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(lora, weight, x)
        if self._compiled is None:
            self._compile()
        compiled = self._compiled
        assert compiled is not None
        out = torch.empty(
            (1, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=lora.device,
        )
        from cuda.bindings.driver import CUstream

        compiled(
            lora,
            weight,
            x,
            out,
            CUstream(current_stream().cuda_stream),
        )
        return out

    @staticmethod
    def _validate_inputs(
        lora: torch.Tensor,
        weight: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
        if lora.shape != (1, _LORA_RANK):
            raise ValueError(f"lora must have shape [1, {_LORA_RANK}].")
        if weight.shape != (_HYPER_HIDDEN_SIZE, _LORA_RANK):
            raise ValueError(
                f"weight must have shape [{_HYPER_HIDDEN_SIZE}, {_LORA_RANK}]."
            )
        if x.shape != (1, _HYPER_HIDDEN_SIZE):
            raise ValueError(f"x must have shape [1, {_HYPER_HIDDEN_SIZE}].")

        tensors = (lora, weight, x)
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("All inputs must use torch.bfloat16.")
        if any(tensor.device != lora.device for tensor in tensors):
            raise ValueError("All inputs must be on the same device.")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("All inputs must be contiguous.")


def _hc_silu_up_gate_mix(
    lora: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    if lora.shape[0] == 1:
        return HCSiluUpGateMixOp.initialize()(lora, weight, x)

    from .hc import hc_gate_mix, hc_silu

    gate = torch.nn.functional.linear(hc_silu(lora, _HC), weight)
    return hc_gate_mix(x, gate, _HC)


def _hc_silu_up_gate_mix_fake(
    lora: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    del weight, x
    return lora.new_empty((lora.shape[0], _HIDDEN_SIZE))


direct_register_custom_op(
    op_name="qwen4_exp_hc_silu_up_gate_mix",
    op_func=_hc_silu_up_gate_mix,
    fake_impl=_hc_silu_up_gate_mix_fake,
)


def hc_silu_up_gate_mix(
    lora: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_silu_up_gate_mix(lora, weight, x)


__all__ = ["HCSiluUpGateMixOp", "hc_silu_up_gate_mix"]
