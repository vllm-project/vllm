# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn


class HYV4VLForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        raise NotImplementedError("HYV4 VL requires NVIDIA CUDA or AMD ROCm.")
