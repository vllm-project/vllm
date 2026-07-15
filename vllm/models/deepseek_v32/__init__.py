# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 (``deepseek_v32``) model — hardware-isolated entry point.

DeepSeek V3.2 introduced the DeepSeek Sparse Attention (DSA) architecture:
MLA + a "lightning indexer" that selects the top-k tokens for a sparse MLA
attend. The same model code serves any DSA checkpoint, including GLM-5.2
(``glm_moe_dsa``), which reuses this architecture.

The optimized kernels under ``nvidia/`` target the Blackwell (SM100) family.
Every other platform — ROCm, XPU, pre-SM100 CUDA (e.g. H100), CPU — falls back
to the generic ``deepseek_v2`` implementation, which already handles the DSA
(index_topk) architecture and is ``torch.compile``-friendly there. This matches
main's behavior on those platforms (no hard failure).
"""

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from .amd.model import DeepseekV32ForCausalLM
    from .amd.mtp import DeepseekV32MTP
elif current_platform.is_xpu():
    raise NotImplementedError("deepseek_v32 does not yet support XPU.")
elif current_platform.is_cuda() and current_platform.is_device_capability_family(100):
    from .nvidia.model import DeepseekV32ForCausalLM
    from .nvidia.mtp import DeepseekV32MTP

    # GLM-5.2 (glm_moe_dsa) reuses the same optimized DSA module on SM100.
    GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM
else:
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
