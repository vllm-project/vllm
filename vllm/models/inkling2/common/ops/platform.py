# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side platform predicates of the Inkling2 kernels.

Each names one capability, so that the shared code takes the CUDA fast
paths of `nvidia/` exactly where they ran before and the portable Triton
paths everywhere else (e.g. XPU).
"""

import functools
from collections.abc import Mapping
from types import MappingProxyType


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def use_pdl() -> bool:
    """Whether the Triton kernels use programmatic dependent launch
    (griddepcontrol) and TMA descriptors, both SM90+ (named and defined as in
    vllm-project/vllm#69165). XPU reports no device capability: there the
    kernels run stream ordered. ROCm reports its gfx capability (gfx942 is
    9.4), so this is True there, as in #69165; ROCm does not reach these
    kernels (it rejects the inkling2 quantization; vllm-project/vllm#67685
    routes it to its own `amd/` package)."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@functools.cache
def pdl_launch_kwargs() -> Mapping[str, bool]:
    """Launch options of the PDL kernels: `launch_pdl=use_pdl()` on CUDA, as
    in vllm-project/vllm#69165, and none elsewhere. `launch_pdl` is an
    option of Triton's NVIDIA backend only: the XPU and the AMD (HIP)
    backends reject it as an unrecognised keyword even when it is False."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return MappingProxyType({})
    return MappingProxyType({"launch_pdl": use_pdl()})


@functools.cache
def use_native_conv_mix() -> bool:
    """Whether the csrc/inkling2 conv/mix kernels run here: they are built
    for Hopper and SM10x only."""
    from vllm.platforms import current_platform

    return any(current_platform.is_device_capability_family(f) for f in (90, 100))


@functools.cache
def use_native_e4m3() -> bool:
    """Whether the kernels let Triton convert e4m3 (fp8e4nv), SM90+. Below,
    they take e4m3 tensors as uint8 and convert in software (`ops/e4m3.py`):
    on SM89, Triton 3.8 converts fp32 through fp16, rounding twice."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)
