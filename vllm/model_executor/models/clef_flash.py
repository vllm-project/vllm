# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef-Flash: the Qwen3.5 backbone with a joint schema decision head."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY

from .clef_joint_schema import JOINT_HEAD_WEIGHTS, JointSchemaHead, JointSchemaPooler
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ClefFlashForJointSchema(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        self.pooler = JointSchemaPooler(
            JointSchemaHead(**model_config.hf_config.joint_head_config),
            self._embed_lexical,
        )
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="pooler.head.",
                allow_patterns_overrides=[JOINT_HEAD_WEIGHTS],
            )
        ]

    def _embed_lexical(self, token_ids: torch.Tensor) -> torch.Tensor:
        return VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)
