# SPDX-License-Identifier: Apache-2.0

from vllm.triton_utils.importing import (HAS_TRITON, TritonLanguagePlaceholder,
                                         TritonPlaceholder)

if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl"]
