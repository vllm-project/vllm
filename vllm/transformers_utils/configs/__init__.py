# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
"""

from vllm.transformers_utils.configs.chatglm import ChatGLMConfig
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.configs.eagle import EAGLEConfig
# RWConfig is for the original tiiuae/falcon-40b(-instruct) and
# tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
# `FalconConfig` class from the official HuggingFace transformers library.
from vllm.transformers_utils.configs.falcon import RWConfig
from vllm.transformers_utils.configs.jais import JAISConfig
from vllm.transformers_utils.configs.kimi_vl import KimiVLConfig
from vllm.transformers_utils.configs.medusa import MedusaConfig
from vllm.transformers_utils.configs.mllama import MllamaConfig
from vllm.transformers_utils.configs.mlp_speculator import MLPSpeculatorConfig
from vllm.transformers_utils.configs.moonvit import MoonViTConfig
from vllm.transformers_utils.configs.nemotron import NemotronConfig
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig
from vllm.transformers_utils.configs.nemotron_vl import Nemotron_Nano_VL_Config
from vllm.transformers_utils.configs.nvlm_d import NVLM_D_Config
from vllm.transformers_utils.configs.step3_vl import (Step3TextConfig,
                                                      Step3VisionEncoderConfig,
                                                      Step3VLConfig)
from vllm.transformers_utils.configs.ultravox import UltravoxConfig

__all__ = [
    "ChatGLMConfig",
    "DeepseekVLV2Config",
    "RWConfig",
    "JAISConfig",
    "MedusaConfig",
    "EAGLEConfig",
    "MllamaConfig",
    "MLPSpeculatorConfig",
    "MoonViTConfig",
    "KimiVLConfig",
    "NemotronConfig",
    "NemotronHConfig",
    "Nemotron_Nano_VL_Config",
    "NVLM_D_Config",
    "UltravoxConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
]
