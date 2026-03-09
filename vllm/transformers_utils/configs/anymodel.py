# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig


class AnyModelConfig(PretrainedConfig):
    model_type = "anymodel"
