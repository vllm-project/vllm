# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    raise NotImplementedError("Step-5 currently supports ROCm only.")

from .amd.model import Step5ForCausalLM, Step5ForConditionalGeneration
from .amd.mtp import Step5MTP

__all__ = ["Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP"]
