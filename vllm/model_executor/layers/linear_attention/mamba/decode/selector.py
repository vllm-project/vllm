# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.v1.kv_cache_interface import MambaSpec

from .base import MambaSSUKernel
from .cpu import CPUSSUKernel
from .flashinfer import FlashInferSSUKernel
from .triton import TritonSSUKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Priority order for automatic selection.
SSU_KERNELS: tuple[type[MambaSSUKernel], ...] = (
    TritonSSUKernel,
    CPUSSUKernel,
    FlashInferSSUKernel,
)


def select_mamba_ssu_kernel(
    vllm_config: "VllmConfig", kv_cache_spec: MambaSpec | None
) -> type[MambaSSUKernel]:
    """Select the SSU kernel for a layer from `--mamba-backend`, or
    automatically when it is unset."""
    kernel = select_linear_attn_kernel(
        "Mamba SSU",
        SSU_KERNELS,
        LinearAttnSelectorConfig.from_current_platform(vllm_config, kv_cache_spec),
        vllm_config.mamba_config.backend,
    )
    assert issubclass(kernel, MambaSSUKernel)
    return kernel
