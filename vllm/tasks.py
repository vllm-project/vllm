# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription"]
GENERATION_TASKS = get_args(GenerationTask)

PoolingTask = Literal["encode", "embed", "classify", "score"]
POOLING_TASKS = get_args(PoolingTask)

SupportedTask = Literal[GenerationTask, PoolingTask]
