# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.v1.kv_cache_interface import MambaSpec

from .base import MambaSSDKernel
from .cpu import CPUSSDKernel
from .flashinfer import FlashInferSSDKernel
from .triton import TritonSSDKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Priority order for automatic selection.
SSD_KERNELS: tuple[type[MambaSSDKernel], ...] = (
    TritonSSDKernel,
    CPUSSDKernel,
    FlashInferSSDKernel,
)


def select_mamba2_ssd_kernel(
    vllm_config: "VllmConfig", kv_cache_spec: MambaSpec | None
) -> type[MambaSSDKernel]:
    """Select the SSD kernel for a layer from `--mamba-prefill-backend`, or
    automatically when it is unset."""
    kernel = select_linear_attn_kernel(
        "Mamba2 SSD",
        SSD_KERNELS,
        LinearAttnSelectorConfig.from_current_platform(vllm_config, kv_cache_spec),
        vllm_config.mamba_config.prefill_backend,
    )
    assert issubclass(kernel, MambaSSDKernel)
    return kernel
