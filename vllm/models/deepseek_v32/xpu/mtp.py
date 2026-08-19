# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.model_executor.models.deepseek_v2 import DeepseekV2MixtureOfExperts


class DeepseekV32MTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "deepseek_v32 does not yet support XPU. "
            "A dedicated XPU implementation is pending."
        )
