# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

from vllm.transformers_utils.processors.deepseek_vl2 import (
    DeepseekVLV2Processor)
from vllm.transformers_utils.processors.ovis import OvisProcessor

__all__ = ["DeepseekVLV2Processor", "OvisProcessor"]
