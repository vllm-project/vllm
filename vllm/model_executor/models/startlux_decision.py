# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StartLux decisions from the last prompt token, without text generation."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor


class StartLuxDecisionPooler(Pooler):
    """Return raw A–Z logits with the reference FP32 output projection.

    Each rank keeps the same small head, selected before vocabulary sharding.
    Candidate restriction and per-question temperature belong to the caller.
    """

    def __init__(self, symbol_ids: list[int], hidden_size: int):
        super().__init__()
        self.method = LastPool()
        self.symbol_ids = symbol_ids
        self.register_buffer(
            "letter_weight",
            torch.empty(len(symbol_ids), hidden_size, dtype=torch.float32),
            persistent=False,
        )

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    def load_head(self, weight: torch.Tensor) -> None:
        self.letter_weight.copy_(weight[self.symbol_ids].float())

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> torch.Tensor:
        selected = self.method(hidden_states, pooling_metadata)
        return F.linear(selected.float(), self.letter_weight)


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class StartLuxDecisionForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Dense Qwen3.5 StartLux checkpoints."""

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("StartLux decision pooling requires PP=1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        encoded = [
            self._tokenizer.encode(symbol, add_special_tokens=False)
            for symbol in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("StartLux answer letters must be single tokens")
        self.pooler = StartLuxDecisionPooler(
            [ids[0] for ids in encoded],
            vllm_config.model_config.get_hidden_size(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tied = self.config.get_text_config().tie_word_embeddings
        head_loaded = False

        def capture_head():
            nonlocal head_loaded
            for name, weight in weights:
                if name.endswith("lm_head.weight") or (
                    tied and name.endswith("embed_tokens.weight")
                ):
                    self.pooler.load_head(weight)
                    head_loaded = True
                yield name, weight

        loaded = super().load_weights(capture_head())
        if not head_loaded:
            raise ValueError("Checkpoint contains no StartLux output head")
        return loaded


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class StartLuxDecisionMoeForSequenceClassification(
    StartLuxDecisionForSequenceClassification, Qwen3_5MoeForConditionalGeneration
):
    """Qwen3.5 MoE StartLux checkpoints."""
