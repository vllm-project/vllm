# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription", "realtime"]
GENERATION_TASKS: tuple[GenerationTask, ...] = get_args(GenerationTask)

PoolingTask = Literal[
    "embed", "classify", "score", "token_embed", "token_classify", "plugin"
]
POOLING_TASKS: tuple[PoolingTask, ...] = get_args(PoolingTask)

FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
