# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classify allocation failures once, independently of server lifecycle stages."""

import errno

import pytest
import torch

from vllm.v1.spec_decode.remote.errors import (
    CompletionUnknownError,
    CudaRuntimeError,
    cuda_context_fatal,
    host_allocation_failure,
    runtime_unusable,
)


@pytest.mark.parametrize(
    "error,fatal,unusable,host_oom",
    [
        (MemoryError("allocation failed"), False, False, True),
        (OSError(errno.ENOMEM, "allocation failed"), False, False, True),
        (
            RuntimeError(
                "[enforce fail at alloc_cpu.cpp:127] err == 0. "
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
                "67108864 bytes. Error code 12 (Cannot allocate memory)"
            ),
            False,
            False,
            True,
        ),
        (torch.OutOfMemoryError("CUDA out of memory"), False, False, False),
        (RuntimeError("ordinary adapter failure"), False, False, False),
        (OSError(errno.EINVAL, "invalid allocation argument"), False, False, False),
        (
            RuntimeError(
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
                "64 bytes. Error code 22 (Invalid argument)"
            ),
            False,
            False,
            False,
        ),
        (CudaRuntimeError("allocation failed", cuda_error_code=2), False, False, False),
        (CompletionUnknownError("upload completion unknown"), False, True, False),
        *[
            (RuntimeError(detail), True, True, False)
            for detail in (
                "CUDA error: an illegal memory access was encountered",
                "CUDA error: device-side assert triggered",
                "CUDA_ERROR_ILLEGAL_ADDRESS",
                "cudaEventQuery failed: hardware stack error (714)",
                "cudaErrorHardwareStackError",
                "CUDA error: tensor memory not completely freed",
                "CUDA_ERROR_TENSOR_MEMORY_LEAK",
                "cudaErrorContained",
                "cudaErrorMpsClientTerminated",
            )
        ],
        *[
            (
                CudaRuntimeError("opaque failure", cuda_error_code=code),
                True,
                True,
                False,
            )
            for code in (226, 700, 702, 710, 714, 715, 716, 717, 718, 719, 810)
        ],
    ],
)
def test_error_classification(error, fatal, unusable, host_oom):
    assert cuda_context_fatal(error) is fatal
    assert runtime_unusable(error) is unusable
    assert host_allocation_failure(error) is host_oom


@pytest.mark.parametrize("wrapper", [MemoryError, torch.OutOfMemoryError, RuntimeError])
@pytest.mark.parametrize("link", ["cause", "context"])
def test_fatal_chain_takes_precedence_over_allocation_error(wrapper, link):
    error = wrapper("allocation after CUDA failure")
    fatal = RuntimeError("CUDA error: an illegal memory access was encountered")
    if link == "cause":
        error.__cause__ = fatal
    else:
        error.__context__ = fatal
        error.__cause__ = ValueError("nonfatal explicit cause")
    assert cuda_context_fatal(error)
    assert runtime_unusable(error)
    assert not host_allocation_failure(error)


def test_prior_host_allocation_does_not_mask_a_new_unrelated_error():
    error = ValueError("invalid shape")
    error.__cause__ = MemoryError("prior allocation failure")
    assert not host_allocation_failure(error)
    assert not runtime_unusable(error)


def test_completion_unknown_survives_wrapping_and_exception_cycles():
    uncertain = CompletionUnknownError("copy did not complete")
    wrapper = RuntimeError("feature import failed")
    wrapper.__context__ = uncertain
    uncertain.__cause__ = wrapper
    assert runtime_unusable(wrapper)
    assert not cuda_context_fatal(wrapper)
    assert not host_allocation_failure(wrapper)
