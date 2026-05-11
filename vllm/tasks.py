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

ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]
SCORE_TYPE_MAP: dict[PoolingTask, ScoreType] = {
    "embed": "bi-encoder",
    "classify": "cross-encoder",
    "token_embed": "late-interaction",
}

FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
