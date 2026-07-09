# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.lora.ops.xpu_ops.lora_ops import (
    lora_expand,
    lora_shrink,
)

__all__ = [
    "lora_expand",
    "lora_shrink",
]
