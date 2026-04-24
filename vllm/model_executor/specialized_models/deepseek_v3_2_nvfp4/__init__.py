# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 model optimized for SM100 (Blackwell)."""

from .model import DeepseekV32ForCausalLM
from .mtp import DeepSeekMTP

__all__ = ["DeepseekV32ForCausalLM", "DeepSeekMTP"]
