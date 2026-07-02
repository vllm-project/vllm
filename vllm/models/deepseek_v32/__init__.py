# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 (``deepseek_v32``) model — hardware-isolated entry point.

DeepSeek V3.2 introduced the DeepSeek Sparse Attention (DSA) architecture:
MLA + a "lightning indexer" that selects the top-k tokens for a sparse MLA
attend. The same model code serves any DSA checkpoint, including GLM-5.2
(``glm_moe_dsa``), which reuses this architecture.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from .amd.model import DeepseekV32ROCMAiterForCausalLM as DeepseekV32ForCausalLM
    from .amd.mtp import DeepseekV32ROCMAiterMTP as DeepseekV32MTP
elif current_platform.is_xpu():
    raise NotImplementedError("deepseek_v32 does not yet support XPU.")
else:
    from .nvidia.model import DeepseekV32ForCausalLM
    from .nvidia.mtp import DeepseekV32MTP

__all__ = [
    "DeepseekV32ForCausalLM",
    "DeepseekV32MTP",
]
