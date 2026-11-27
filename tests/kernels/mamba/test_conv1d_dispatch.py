# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.linear_attention.conv1d.decode.cpu import (
    CPUConv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.flashinfer import (
    FlashInferConv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.selector import (
    CONV1D_DECODE_KERNELS,
    select_conv1d_decode_kernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.decode.triton import (
    TritonConv1dDecodeKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.cpu import (
    CPUConv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.flashinfer import (
    FlashInferConv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.selector import (
    CONV1D_PREFILL_KERNELS,
    select_conv1d_prefill_kernel,
)
from vllm.model_executor.layers.linear_attention.conv1d.prefill.triton import (
    TritonConv1dPrefillKernel,
)
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnSelectorConfig,
    select_linear_attn_kernel,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability, PlatformEnum
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test


def _spec(width=4, num_speculative_tokens=0, state_dtype=torch.bfloat16):
    return MambaSpec(
        block_size=256,
        shapes=((width - 1 + num_speculative_tokens, 1024), (8, 64, 128)),
        dtypes=(state_dtype, torch.float32),
    )


SPEC = _spec()


def _cfg(
    platform,
    capability=None,
    num_speculative_tokens=0,
    *,
    dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
    width=4,
    activation="silu",
):
    return LinearAttnSelectorConfig(
        platform,
        capability,
        _spec(width, num_speculative_tokens, state_dtype),
        dtype,
        num_speculative_tokens=num_speculative_tokens,
        conv_width=width,
        conv_activation=activation,
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


FLASHINFER_KERNELS = [
    (CONV1D_PREFILL_KERNELS, FlashInferConv1dPrefillKernel),
    (CONV1D_DECODE_KERNELS, FlashInferConv1dDecodeKernel),
]


@pytest.mark.parametrize(("kernels", "expected"), FLASHINFER_KERNELS)
@pytest.mark.parametrize(
    "capability", [DeviceCapability(10, 0), DeviceCapability(10, 3)]
)
@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
def test_flashinfer_conv1d_runs_on_request(
    kernels, expected, capability, num_speculative_tokens
):
    """Automatic selection keeps Triton; FlashInfer runs when requested."""
    if not expected.is_available():
        pytest.skip("FlashInfer is not installed")
    cfg = _cfg(PlatformEnum.CUDA, capability, num_speculative_tokens)
    assert select_linear_attn_kernel("conv1d", kernels, cfg) is not expected
    selected = select_linear_attn_kernel(
        "conv1d", kernels, cfg, MambaBackendEnum.FLASHINFER
    )
    assert selected is expected


@pytest.mark.parametrize(("kernels", "expected"), FLASHINFER_KERNELS)
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"capability": DeviceCapability(9, 0)}, "compute capability 9.0"),
        ({"width": 3}, "width 3"),  # ShortConv (LFM2)
        ({"activation": None}, "activation None"),  # ShortConv (LFM2)
        ({"dtype": torch.float16}, "dtype torch.float16"),
        ({"state_dtype": torch.float32}, "conv state dtype torch.float32"),
    ],
)
def test_requested_flashinfer_conv1d_fails_with_what_it_cannot_run(
    kernels, expected, overrides, reason
):
    """FlashInfer's conv1d runs only width-4 SiLU convolutions on BF16 inputs
    and conv states, on compute capability 10.0 and 10.3. A request outside
    that fails at startup with the reason."""
    overrides = {"capability": DeviceCapability(10, 0), **overrides}
    cfg = _cfg(PlatformEnum.CUDA, overrides.pop("capability"), **overrides)
    with pytest.raises(ValueError, match=f"flashinfer can't run.*{reason}"):
        select_linear_attn_kernel("conv1d", kernels, cfg, MambaBackendEnum.FLASHINFER)


@pytest.mark.parametrize(
    ("select", "field"),
    [
        (select_conv1d_prefill_kernel, "conv1d_prefill_backend"),
        (select_conv1d_decode_kernel, "conv1d_decode_backend"),
    ],
)
def test_conv1d_backend_flags_request_a_kernel(select, field):
    """`--conv1d-prefill-backend` and `--conv1d-decode-backend` request a
    kernel: one this platform can't run fails at startup."""
    unavailable = (
        MambaBackendEnum.FLASHINFER
        if current_platform.is_cpu()
        else MambaBackendEnum.CPU
    )
    vllm_config = VllmConfig(mamba_config=MambaConfig(**{field: unavailable}))
    with pytest.raises(ValueError, match=f"Backend {unavailable} can't run conv1d"):
        select(vllm_config, SPEC, width=4, activation="silu")
