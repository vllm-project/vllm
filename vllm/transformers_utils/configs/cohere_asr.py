# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class CohereASRConfig(PretrainedConfig):
    model_type = "cohere_asr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


AutoConfig.register("cohere_asr", CohereASRConfig)
