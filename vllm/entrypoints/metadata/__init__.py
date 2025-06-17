# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from .classify import ClassifyMetadata
from .draft import DraftMetadata
from .embed import EmbedMetadata
from .generate import GenerateMetadata
from .score import ScoreMetadata
from .transcription import TranscriptionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TASK2METADATA_CLASS = {
    "embed": EmbedMetadata,
    "score": ScoreMetadata,
    "classify": ClassifyMetadata,
    "generate": GenerateMetadata,
    "transcription": TranscriptionMetadata,
    "draft": DraftMetadata,
}


def get_metadata(vllm_config: "VllmConfig"):
    task = vllm_config.model_config.task
    metadata_class = TASK2METADATA_CLASS.get(task)
    assert metadata_class is not None, \
        (f"Task {task} is not supported yet. "
         f"Please add it to TASK2METADATA_CLASS in __init__.py.")

    return metadata_class.from_vllm_config(vllm_config)
