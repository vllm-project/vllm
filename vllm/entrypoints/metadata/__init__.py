from typing import TYPE_CHECKING

from .embed import EmbedMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TASK2METADATA_CLASS = {
    "embed": EmbedMetadata,
}


def get_metadata(vllm_config: "VllmConfig"):
    task = vllm_config.model_config.task
    metadata_class = TASK2METADATA_CLASS[task]
    return metadata_class.from_vllm_config(vllm_config)
