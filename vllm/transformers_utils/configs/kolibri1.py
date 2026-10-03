# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class Kolibri1Config(Qwen3MoeConfig):
    model_type = "kolibri1"
