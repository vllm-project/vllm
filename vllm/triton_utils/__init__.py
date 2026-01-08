# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    next_power_of_2 = triton.next_power_of_2  # type: ignore
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()

    # we define a dummy function here to avoid import errors
    def next_power_of_2(n: int):
        raise ImportError("Triton is not installed.")


__all__ = ["HAS_TRITON", "triton", "tl", "tldevice"]
