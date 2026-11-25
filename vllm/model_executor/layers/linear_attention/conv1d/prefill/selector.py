# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.v1.kv_cache_interface import MambaSpec

from .base import Conv1dPrefillKernel
from .cpu import CPUConv1dPrefillKernel
from .flashinfer import FlashInferConv1dPrefillKernel
from .triton import TritonConv1dPrefillKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Priority order for automatic selection.
CONV1D_PREFILL_KERNELS: tuple[type[Conv1dPrefillKernel], ...] = (
    TritonConv1dPrefillKernel,
    CPUConv1dPrefillKernel,
    FlashInferConv1dPrefillKernel,
)


def select_conv1d_prefill_kernel(
    vllm_config: "VllmConfig",
    kv_cache_spec: MambaSpec | None,
    *,
    width: int,
    activation: str | None,
) -> type[Conv1dPrefillKernel]:
    """Select the causal conv1d prefill kernel for a layer whose conv has this
    width and activation."""
    cfg = LinearAttnSelectorConfig.from_current_platform(vllm_config, kv_cache_spec)
    kernel = select_linear_attn_kernel(
        "conv1d prefill",
        CONV1D_PREFILL_KERNELS,
        replace(cfg, conv_width=width, conv_activation=activation),
        vllm_config.mamba_config.conv1d_prefill_backend,
    )
    assert issubclass(kernel, Conv1dPrefillKernel)
    return kernel
