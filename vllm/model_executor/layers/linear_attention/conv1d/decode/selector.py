# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.v1.kv_cache_interface import MambaSpec

from .base import Conv1dDecodeKernel
from .cpu import CPUConv1dDecodeKernel
from .triton import TritonConv1dDecodeKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Priority order for automatic selection.
CONV1D_DECODE_KERNELS: tuple[type[Conv1dDecodeKernel], ...] = (
    TritonConv1dDecodeKernel,
    CPUConv1dDecodeKernel,
)


def select_conv1d_decode_kernel(
    vllm_config: "VllmConfig", kv_cache_spec: MambaSpec | None
) -> type[Conv1dDecodeKernel]:
    """Select the causal conv1d decode kernel for a layer."""
    kernel = select_linear_attn_kernel(
        "conv1d decode",
        CONV1D_DECODE_KERNELS,
        LinearAttnSelectorConfig.from_current_platform(vllm_config, kv_cache_spec),
    )
    assert issubclass(kernel, Conv1dDecodeKernel)
    return kernel
