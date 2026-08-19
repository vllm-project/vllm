# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 (``deepseek_v32``) model — hardware-isolated entry point.

DeepSeek V3.2 introduced the DeepSeek Sparse Attention (DSA) architecture:
MLA + a "lightning indexer" that selects the top-k tokens for a sparse MLA
attend. The same model code serves any DSA checkpoint, including GLM-5.2
(``glm_moe_dsa``), which reuses this architecture.

The CUDA implementation selects capability-specific kernels internally and
falls back when an optimization is unavailable. GLM-5.2 uses the non-compiled
MRV2 path, so the same implementation can serve all NVIDIA GPUs.
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    # GLM-5.2 keeps the generic implementation here, as it has on main; only
    # DeepSeek V3.2 has an AMD DSA port.
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM

    from .amd.model import DeepseekV32ForCausalLM
    from .amd.mtp import DeepseekV32MTP
elif current_platform.is_cuda():
    # GLM-5.2 (glm_moe_dsa) reuses the CUDA DSA module. Individual optimized
    # kernels remain gated on the device capabilities they support.
    from .nvidia.model import DeepseekV32ForCausalLM
    from .nvidia.model import DeepseekV32ForCausalLM as GlmMoeDsaForCausalLM
    from .nvidia.mtp import DeepseekV32MTP
else:
    # XPU and CPU keep the generic implementation.
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
