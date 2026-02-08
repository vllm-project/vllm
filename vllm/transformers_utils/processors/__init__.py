# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

from vllm.transformers_utils.processors.bagel import BagelProcessor
from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
from vllm.transformers_utils.processors.funasr_processor import FunASRProcessor
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import HunYuanVLImageProcessor
from vllm.transformers_utils.processors.ovis import OvisProcessor
from vllm.transformers_utils.processors.ovis2_5 import Ovis2_5Processor

__all__ = [
    "BagelProcessor",
    "DeepseekVLV2Processor",
    "HunYuanVLProcessor",
    "HunYuanVLImageProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "FunASRProcessor",
]
