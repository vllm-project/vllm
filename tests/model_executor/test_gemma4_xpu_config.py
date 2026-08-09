# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig
from vllm.model_executor.models.config import Gemma4Config


def test_gemma4_config_xpu():
    # 1. Create a mock configuration with heterogeneous head dimensions
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config.hf_text_config.head_dim = 256
    vllm_config.model_config.hf_text_config.global_head_dim = 192
    vllm_config.attention_config.backend = None
    vllm_config.attention_config.flash_attn_version = None

    # 2. Mock current_platform.is_xpu to return True
    with patch("vllm.platforms.current_platform.is_xpu", return_value=True):
        Gemma4Config.verify_and_update_config(vllm_config)

    # 3. Assert that the attention backend remained untouched (None)
    # so the platform's default attention backend is used instead of forcing TRITON_ATTN
    assert vllm_config.attention_config.backend is None
