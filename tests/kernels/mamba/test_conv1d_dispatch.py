# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.linear_attention.conv1d.decode.cpu import (
    CPUConv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.selector import (
    CONV1D_DECODE_KERNELS,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.triton import (
    TritonConv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.cpu import (
    CPUConv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.selector import (
    CONV1D_PREFILL_KERNELS,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.triton import (
    TritonConv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test

SPEC = MambaSpec(
    block_size=256,
    shapes=((3, 1024), (8, 64, 128)),
    dtypes=(torch.bfloat16, torch.float32),
)


def _cfg(platform, capability=None, num_speculative_tokens=0):
    return LinearAttnSelectorConfig(
        platform,
        capability,
        SPEC,
        torch.bfloat16,
        num_speculative_tokens=num_speculative_tokens,
    )


CUDA = _cfg(PlatformEnum.CUDA, DeviceCapability(10, 0))
CPU = _cfg(PlatformEnum.CPU)


@pytest.mark.parametrize(
    ("kernels", "cfg", "expected"),
    [
        (CONV1D_PREFILL_KERNELS, CUDA, TritonConv1dPrefillKernel),
        (CONV1D_PREFILL_KERNELS, CPU, CPUConv1dPrefillKernel),
        (CONV1D_DECODE_KERNELS, CUDA, TritonConv1dDecodeKernel),
        (CONV1D_DECODE_KERNELS, CPU, CPUConv1dDecodeKernel),
    ],
)
def test_conv1d_selects_the_platform_default(kernels, cfg, expected):
    assert select_linear_attn_kernel("conv1d", kernels, cfg) is expected


def test_cpu_conv1d_decode_rejects_speculative_decoding():
    """The CPU decode kernel ignores the accepted-token counts, so with
    speculative decoding it would roll the conv state back wrongly. It fails
    at startup instead."""
    with pytest.raises(ValueError, match="speculative decoding not supported"):
        select_linear_attn_kernel(
            "conv1d",
            CONV1D_DECODE_KERNELS,
            _cfg(PlatformEnum.CPU, num_speculative_tokens=3),
        )
