# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM
from vllm.model_executor.models.interfaces import supports_pp


def test_deepseek_v4_declares_pipeline_parallel_support():
    assert supports_pp(DeepseekV4ForCausalLM)
