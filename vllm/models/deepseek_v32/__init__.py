# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 (``deepseek_v32``) model — hardware-isolated entry point.

DeepSeek V3.2 introduced the DeepSeek Sparse Attention (DSA) architecture:
MLA + a "lightning indexer" that selects the top-k tokens for a sparse MLA
attend. The same model code serves any DSA checkpoint, including GLM-5.2
(``glm_moe_dsa``), which reuses this architecture.

The kernels under ``nvidia/`` target the Blackwell (SM100) family. Pre-SM100
CUDA (e.g. H100) falls back to the generic ``deepseek_v2`` implementation, which
already handles the DSA (index_topk) architecture and is ``torch.compile``
-friendly there, so those devices keep working rather than hard-failing.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    # GLM-5.2 keeps the generic implementation here, as it has on main; only
    # DeepSeek V3.2 has an AMD DSA port.
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

    from .amd.model import DeepseekV32ForCausalLM
    from .amd.mtp import DeepseekV32MTP
elif current_platform.is_device_capability_family(100):
    # GLM-5.2 (glm_moe_dsa) reuses the same optimized DSA module on SM100.
    from .nvidia.model import DeepseekV32ForCausalLM
    from .nvidia.model import DeepseekV32ForCausalLM as GlmMoeDsaForCausalLM
    from .nvidia.mtp import DeepseekV32MTP
else:
    # Pre-SM100 CUDA, XPU and CPU. The generic implementation already handles
    # the DSA architecture, so these keep serving instead of hard-failing --
    # which is what GLM-5.2 does on main, where its registry entry points
    # straight at deepseek_v2 and never reaches this package.
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP as DeepseekV32MTP
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV3ForCausalLM as DeepseekV32ForCausalLM,
    )
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

__all__ = [
    "DeepseekV32ForCausalLM",
    "DeepseekV32MTP",
    "GlmMoeDsaForCausalLM",
]
