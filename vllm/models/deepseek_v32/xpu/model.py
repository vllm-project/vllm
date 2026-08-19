# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "deepseek_v32 does not yet support XPU. "
            "A dedicated XPU implementation is pending."
        )
