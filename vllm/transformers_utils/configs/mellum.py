# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import Qwen3MoeConfig


class MellumConfig(Qwen3MoeConfig):
    model_type = "mellum"
