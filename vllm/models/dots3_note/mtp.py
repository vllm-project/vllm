# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-token predictor for Dots3 NOTE."""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
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

from .model import Dot3NoteDecoderLayer


class Dot3NoteMultiTokenPredictorLayer(nn.Module):
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
        self.mtp_block = Dot3NoteDecoderLayer(
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
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        del input_ids, spec_step_index
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


class Dot3NoteMultiTokenPredictor(DeepseekV32MultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): Dot3NoteMultiTokenPredictorLayer(
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
        self.logits_processor = LogitsProcessor(config.vocab_size)


class Dot3NoteMTP(DeepseekV32MTP):
    has_own_embed_tokens = True
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Dot3NoteMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()

    def _pad_dense_mlp_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        block_size = getattr(self.quant_config, "weight_block_size", None)
        if block_size is None or ".mlp.experts." in name:
            return loaded_weight
        if not any(
            proj_name in name
            for proj_name in (".gate_proj.", ".up_proj.", ".down_proj.")
        ):
            return loaded_weight
        dim = 1 if ".down_proj." in name else 0
        block_step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        multiple = get_tensor_model_parallel_world_size() * block_step
        pad = (-loaded_weight.shape[dim]) % multiple
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

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
            yield name, self._pad_dense_mlp_weight(name, weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._adapt_weights(weights))
