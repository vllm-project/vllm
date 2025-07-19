# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingTask, VisionPooler
# yapf: disable
from vllm.model_executor.pooling_metadata import (
    PoolingMetadata as V0PoolingMetadata)
# yapf: enable
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.pooling_params import PoolingParams
from vllm.sequence import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata as V1PoolingMetadata

from .interfaces import SupportsCrossEncoding, SupportsMultiModal
from .qwen2_vl import (Qwen2VLDummyInputsBuilder,
                       Qwen2VLForConditionalGeneration,
                       Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo)
from .utils import maybe_prefix

logger = init_logger(__name__)

# Vision token IDs for Jina V4
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

PoolingMetadata = Union[V0PoolingMetadata, V1PoolingMetadata]


class JinaVLPooler(Pooler):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vision_pooler = VisionPooler(vllm_config.model_config)

    def get_pooling_params(self, task: PoolingTask) -> Optional[PoolingParams]:
        return self.vision_pooler.get_pooling_params(task)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        return self.vision_pooler.forward(hidden_states, pooling_metadata)


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class JinaVLForEmbedding(Qwen2VLForConditionalGeneration,
                         SupportsCrossEncoding, SupportsMultiModal):

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=maybe_prefix(prefix, "qwen2_vl"))

        self.pooler = JinaVLPooler(vllm_config)

        logger.info("Initialized JinaVLForEmbedding with vision-aware pooling")
