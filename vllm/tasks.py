# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

from vllm.exceptions import VLLMValidationError

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

_REMOVED_POOLING_TASK_MESSAGES = {
    "score": "`score` task was removed; use `classify` instead.",
    "encode": (
        "`encode` task was removed; use `token_embed` or `token_classify` instead."
    ),
}


def check_removed_pooling_task(task: object) -> None:
    if isinstance(task, str) and (message := _REMOVED_POOLING_TASK_MESSAGES.get(task)):
        raise VLLMValidationError(message, parameter="task")


ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]
SCORE_TYPE_MAP: dict[PoolingTask, ScoreType] = {
    "embed": "bi-encoder",
    "classify": "cross-encoder",
    "token_embed": "late-interaction",
}

FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
