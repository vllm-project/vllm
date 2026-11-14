# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation and CUDA failure classification for remote drafting."""

import errno
from collections.abc import Iterator

import regex as re


class CudaRuntimeError(RuntimeError):
    """CUDA runtime failure retaining its numeric cudaError_t value."""

    def __init__(self, message: str, *, cuda_error_code: int | None = None):
        super().__init__(message)
        self.cuda_error_code = cuda_error_code


class CompletionUnknownError(RuntimeError):
    """GPU work may still reference storage; retire the owning process."""


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending = [error]
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        yield error
        if error.__cause__ is not None:
            pending.append(error.__cause__)
        if error.__context__ is not None:
            pending.append(error.__context__)


def cuda_context_fatal(error: BaseException) -> bool:
    """Recognize CUDA errors documented to require a process restart."""
    for cause in _exception_chain(error):
        if isinstance(cause, CudaRuntimeError) and cause.cuda_error_code in (
            226,  # cudaErrorContained
            700,  # cudaErrorIllegalAddress
            702,  # cudaErrorLaunchTimeout
            710,  # cudaErrorAssert
            714,  # cudaErrorHardwareStackError
            715,  # cudaErrorIllegalInstruction
            716,  # cudaErrorMisalignedAddress
            717,  # cudaErrorInvalidAddressSpace
            718,  # cudaErrorInvalidPc
            719,  # cudaErrorLaunchFailure
            810,  # cudaErrorMpsClientTerminated
        ):
            return True
        message = str(cause).lower().replace("_", "")
        if "cuda" in message and any(
            detail in message
            for detail in (
                "illegal memory access",
                "device-side assert triggered",
                "unspecified launch failure",
                "the launch timed out and was terminated",
                "hardware stack error",
                "illegal instruction",
                "misaligned address",
                "invalid address space",
                "invalid program counter",
                "tensor memory not completely freed",
                "cudaerrorillegaladdress",
                "cudaerrorassert",
                "cudaerrorlaunchfailure",
                "cudaerrorlaunchtimeout",
                "cudaerrorhardwarestackerror",
                "cudaerrorillegalinstruction",
                "cudaerrormisalignedaddress",
                "cudaerrorinvalidaddressspace",
                "cudaerrorinvalidpc",
                "cudaerrortensormemoryleak",
                "cudaerrorcontained",
                "cudaerrormpsclientterminated",
                "contained error",
                "mps client has been terminated",
            )
        ):
            return True
    return False


def runtime_unusable(error: BaseException) -> bool:
    """Reject new work after context damage or uncertain GPU completion."""
    return cuda_context_fatal(error) or any(
        isinstance(item, CompletionUnknownError) for item in _exception_chain(error)
    )


def host_allocation_failure(error: BaseException) -> bool:
    """Recognize direct host ENOMEM without masking unrelated runtime errors."""
    if cuda_context_fatal(error):
        return False
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, OSError):
        return error.errno == errno.ENOMEM
    return (
        isinstance(error, RuntimeError)
        and re.search(
            r"DefaultCPUAllocator: can't allocate memory: you tried to allocate "
            r"[0-9]+ bytes\. Error code 12(?:\s|$)",
            str(error),
        )
        is not None
    )
