# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

warnings.warn(
    "vllm.lora.layers.vocal_parallel_embedding is deprecated; "
    "use vllm.lora.layers.vocab_parallel_embedding instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .vocab_parallel_embedding import *  # noqa: F401,F403
