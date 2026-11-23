# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools


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
    (griddepcontrol) and TMA descriptors, both SM90+. Ampere and Ada (SM8x)
    run the same kernels without them, and Humming experts like Hopper."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


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
