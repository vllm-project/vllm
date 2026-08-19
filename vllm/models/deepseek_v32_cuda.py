# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA DSA models with generic fallbacks on other platforms."""

from vllm.platforms import current_platform

if current_platform.is_cuda():
    from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM
    from vllm.models.deepseek_v32.nvidia.mtp import DeepseekV32MTP
else:
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP as DeepseekV32MTP
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV3ForCausalLM as DeepseekV32ForCausalLM,
    )

__all__ = ["DeepseekV32ForCausalLM", "DeepseekV32MTP"]
