# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 off SM10x: DeepSeek-V3.2's class, refusing the latency-specialized
decode instead of ignoring it.
"""

from vllm.config import VllmConfig
from vllm.models import deepseek_v32
from vllm.models.glm53.nvidia import switch


class GlmMoeDsaForCausalLM(deepseek_v32.GlmMoeDsaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if switch.requested(vllm_config):
            raise ValueError(
                "GLM-5.3 decode_backend=latency_specialized needs an SM100 or SM103 "
                "GPU (B200 or B300); use decode_backend=auto on this platform."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
