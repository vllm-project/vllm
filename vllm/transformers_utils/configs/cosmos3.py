# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class Cosmos3Config(Qwen3VLConfig):
    model_type = "cosmos3"
