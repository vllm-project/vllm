# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription", "realtime"]
GENERATION_TASKS: tuple[GenerationTask, ...] = get_args(GenerationTask)

PoolingTask = Literal[
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
]
POOLING_TASKS: tuple[PoolingTask, ...] = get_args(PoolingTask)

GradientTask = Literal["gradient"]
GRADIENT_TASKS: tuple[GradientTask, ...] = get_args(GradientTask)

ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]

FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

SupportedTask = Literal[GenerationTask, PoolingTask, GradientTask, FrontendTask]
