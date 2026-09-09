# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stub for platforms where the DeepSeek-V4 vision variant is unsupported."""

from torch import nn


class DeepseekV4ForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        raise NotImplementedError(
            "DeepSeek-V4 vision (DeepseekV4ForConditionalGeneration) is only "
            "supported on NVIDIA GPUs for now."
        )
