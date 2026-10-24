# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded BF16 projection dispatch derived from Pro6000d-community.

Reuses vLLM's CuTe dot-product kernel; this module adds device selection,
preparation and graph-safe runtime dispatch, not a new arithmetic kernel.
"""

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op

_SHAPES = {(4096, 1536), (4096, 4096), (4096, 2048), (6288, 4096), (4096, 8192)}
_PREFIXES = (".q_b_proj", ".o_proj", ".in_proj_qkvgfab", ".eh_proj")
_PREPARED: set[tuple[torch.device, int, int]] = set()


def prepare_rtx6000d_bf16(layer: torch.nn.Module) -> bool:
    weight = layer.weight
    if (
        weight.device.type != "cuda"
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) not in _SHAPES
        or not getattr(layer, "prefix", "").endswith(_PREFIXES)
    ):
        return False
    props = torch.cuda.get_device_properties(weight.device)
    if props.name != "NVIDIA RTX 6000D" or (
        props.major,
        props.minor,
        props.multi_processor_count,
    ) != (12, 0, 156):
        return False

    from .ll_bf16 import is_available, ll_bf16_gemm_kernel

    if not is_available():
        return False
    n, k = weight.shape
    with torch.accelerator.device_index(weight.device.index):
        # Use the same non-PDL instance for preparation and execution, including
        # M1. The historical measurements did not qualify the newer M1 PDL path.
        ll_bf16_gemm_kernel.warmup(shapes=[(k, n)], m_values=(1, 2, 3, 4))
    _PREPARED.add((weight.device, n, k))
    return True


def rtx6000d_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= 4
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.device != weight.device
        or not weight.is_contiguous()
        or (weight.device, *weight.shape) not in _PREPARED
    ):
        return F.linear(x, weight)

    from .ll_bf16 import ll_bf16_gemm_kernel

    n, k = weight.shape
    key = ll_bf16_gemm_kernel.dispatch(M=x.shape[0], K=k, N=n)
    if not ll_bf16_gemm_kernel.is_compiled(key):
        raise RuntimeError("RTX 6000D BF16 projection was not precompiled")
    return ll_bf16_gemm_kernel(x.contiguous(), weight).to(torch.bfloat16)


def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="rtx6000d_bf16_linear",
    op_func=rtx6000d_bf16_linear,
    fake_impl=_fake,
)
