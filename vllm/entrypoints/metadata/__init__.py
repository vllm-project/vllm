# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from .base import Metadata
from .classify import ClassifyMetadata
from .embed import EmbedMetadata
from .generate import GenerateMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TASK2METADATA_CLASS: dict[str, type[Metadata]] = {
    "embed": EmbedMetadata,
    "classify": ClassifyMetadata,
    "generate": GenerateMetadata,
    "transcription": Metadata,
    "draft": Metadata,
}


def get_metadata(vllm_config: "VllmConfig"):
    task = vllm_config.model_config.task
    metadata_class = TASK2METADATA_CLASS.get(task)
    assert metadata_class is not None, \
        (f"Task {task} is not supported yet. "
         f"Please add it to TASK2METADATA_CLASS in __init__.py.")

    return metadata_class.from_vllm_config(vllm_config)
