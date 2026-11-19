# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The logits of the Qwen4Exp specialized decode: the target's vocab-parallel
logits gathered over NVLink (`gather.py`), the MTP draft passes' greedy token
chosen on the vocabulary shard (`choice.py`), and the logits processor that
serves both (`processor.py`). Only `processor` is imported here: the other
modules hold CuTe DSL kernels (CUDA builds only)."""

from .processor import (
    SpecializedDecodeLogitsProcessor,
    bind_logits_gather,
    bind_token_choice,
)

__all__ = [
    "SpecializedDecodeLogitsProcessor",
    "bind_logits_gather",
    "bind_token_choice",
]
