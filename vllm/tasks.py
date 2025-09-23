# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription"]
GENERATION_TASKS = get_args(GenerationTask)

PoolingTask = Literal["embed", "classify", "score", "token_embed", "token_classify"]
POOLING_TASKS = get_args(PoolingTask)

SupportedTask = Literal[GenerationTask, PoolingTask]

def encode2pooling_task(supported_tasks):
    if "token_embed" in supported_tasks:
        return "token_embed"
    elif "token_classify" in supported_tasks:
        return "token_classify"
    else:
        raise ValueError(
            f"pooling_task must be one of {supported_tasks}.")