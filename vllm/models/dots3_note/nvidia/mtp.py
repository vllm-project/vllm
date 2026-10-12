# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA multi-token predictor for Dots3Note."""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_embed_norm import (
    fused_embed_eh_norm,
    has_full_vocab_on_rank,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v32.common.kernels import fused_eh_norm
from vllm.models.deepseek_v32.nvidia.mtp import (
    DeepseekV32MTP,
    DeepseekV32MultiTokenPredictor,
)

from .model import Dots3NoteDecoderLayer, _pad_dense_mlp_weight


class Dots3NoteMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = Dots3NoteDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            prefix=prefix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        embed_table: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        del spec_step_index
        if embed_table is not None:
            eh_input = fused_embed_eh_norm(
                positions,
                input_ids,
                embed_table,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        else:
            assert inputs_embeds is not None
            eh_input = fused_eh_norm(
                positions,
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        hidden_states = self.eh_proj(eh_input)[0]
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        is_sequence_parallel = self.mtp_block.use_sequence_parallel_moe
        if not is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        if is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[0]]
        return hidden_states


class Dots3NoteMultiTokenPredictor(DeepseekV32MultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): Dots3NoteMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.replicated_embed = has_full_vocab_on_rank(self.embed_tokens)
        self.register_buffer(
            "max_token_id",
            torch.tensor(config.vocab_size - 1, dtype=torch.int64),
            persistent=False,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = torch.minimum(input_ids, self.max_token_id)
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            input_ids = torch.minimum(input_ids, self.max_token_id)
        return super().forward(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        )


class Dots3NoteMTP(DeepseekV32MTP):
    has_own_embed_tokens = True
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Dots3NoteMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def _adapt_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        mtp_layer = self.config.num_hidden_layers
        for name, weight in weights:
            if name.startswith("model.mtp.embed_tokens."):
                name = name.replace(
                    "model.mtp.embed_tokens.",
                    f"model.layers.{mtp_layer}.embed_tokens.",
                    1,
                )
            yield (
                name,
                _pad_dense_mlp_weight(
                    name,
                    weight,
                    getattr(self.quant_config, "weight_block_size", None),
                ),
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._adapt_weights(weights))
