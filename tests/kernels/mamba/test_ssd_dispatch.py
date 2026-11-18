# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.model_executor.layers.linear_attention.mamba2.prefill.cpu import CPUSSDKernel
from vllm.model_executor.layers.linear_attention.mamba2.prefill.selector import (
    SSD_KERNELS,
)
from vllm.model_executor.layers.linear_attention.mamba2.prefill.triton import (
    TritonSSDKernel,
)
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec


def _cfg(platform, capability, head_dim=64, state_size=128, dtype=torch.bfloat16):
    """A layer with Nemotron's geometry by default, which FlashInfer's SSD scan
    is built for."""
    spec = MambaSpec(
        block_size=256,
        shapes=((3, 1024), (8, head_dim, state_size)),
        dtypes=(dtype, torch.float32),
    )
    return LinearAttnSelectorConfig(platform, capability, spec, dtype)


SM100 = (PlatformEnum.CUDA, DeviceCapability(10, 0))
HOPPER = _cfg(PlatformEnum.CUDA, DeviceCapability(9, 0))
BLACKWELL = _cfg(*SM100)
CPU = _cfg(PlatformEnum.CPU, None)


def _select(cfg, requested=None):
    return select_linear_attn_kernel("Mamba2 SSD", SSD_KERNELS, cfg, requested)


@pytest.mark.parametrize(
    ("backend", "cfg", "reason"),
    [
        (MambaBackendEnum.FLASHINFER, HOPPER, "compute capability 9.0"),
        (MambaBackendEnum.FLASHINFER, _cfg(*SM100, head_dim=128), "head_dim 64"),
        (
            MambaBackendEnum.FLASHINFER,
            _cfg(*SM100, dtype=torch.float16),
            "dtype torch.float16",
        ),
        (MambaBackendEnum.TRITON, CPU, "platform CPU"),
        (MambaBackendEnum.CPU, BLACKWELL, "platform CUDA"),
    ],
)
def test_prefill_backend_that_cannot_run_fails_at_selection(backend, cfg, reason):
    """A requested backend fails at startup with its reason, not at the first
    prefill."""
    with pytest.raises(ValueError, match=reason):
        _select(cfg, backend)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [(BLACKWELL, TritonSSDKernel), (CPU, CPUSSDKernel)],
)
def test_unset_backend_selects_the_platform_default(cfg, expected):
    assert _select(cfg) is expected
