# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import (Qwen2DecoderLayer,
                                              Qwen2ForCausalLM)
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from .interfaces import MultiModalEmbeddings
from .utils import (AutoWeightsLoader, PPMissingLayer, maybe_prefix,
                    merge_multimodal_embeddings)

logger = init_logger(__name__)


class Qwen2_5DecodeLayer(Qwen2DecoderLayer):

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )
        # Add a normalization layer
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        # Reuse the target model's features
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class Qwen2_5Model(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        self.multimodal_config = (vllm_config.speculative_config.
                                  draft_model_config.multimodal_config)
        # embbeding
        if get_pp_group().is_first_rank or (self.config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # small language model initialization
        self.layers = nn.ModuleList([
            Qwen2_5DecodeLayer(
                config=self.config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
            ) for i in range(self.config.num_hidden_layers)
        ])
        # The EAGLE3 feature fusion layer needs to
        # fuse high, medium, and low-level features.
        # Therefore, the input size is hidden_size * 3
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(self.config.target_hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)
        else:
            self.fc = torch.nn.Linear(self.config.hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)
        self.norm = RMSNorm(self.config.hidden_size,
                            eps=self.config.rms_norm_eps)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        residual = None  # No residual on the first layer
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                inputs_embeds,
                hidden_states,
                residual,
            )
        # Normalized features (hidden_states)
        # original features (hidden_prenorm)
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Eagle3Qwen2_5_VLForCausalLM(Qwen2ForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.\
            draft_model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # The number of layers in the target model
        # start_layer_id for the draft model
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        # draft model quantization config may differ from target model
        quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config,
            vllm_config.load_config)
        # Initialize the EAGLE model of QWEN2.5
        self.model = Qwen2_5Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "draft_model"),
                                  start_layer_id=target_layer_num,
                                  quant_config=quant_config)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)
        # Establish a mapping relationship between
        # the draft model vocabulary and the target model vocabulary.
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.draft_vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE),
            prefix="")

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, \
                "Expected logits to have shape " \
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        # Mapping to the main model vocabulary space
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full((
            logits.shape[0],
            self.config.vocab_size,
        ), float('-inf'))
        logits_new[:, targets] = logits  # Only valid positions are filled
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)

    def load_weights(self, weights):
        """
        Load weights
        Not shared lm_head with target model
        Skip t2d
        """
        model_weights = {}
        include_draft_id_mapping = False
        include_embb_tokens_mapping = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                include_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                include_embb_tokens_mapping = True
            model_weights[name] = loaded_weight

        skip_substrs = []
        if not include_draft_id_mapping:
            skip_substrs.append("d2t")
        if not include_embb_tokens_mapping:
            skip_substrs.append("embed_tokens")
        # Not shared lm_head with target model
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_index)
        return inputs_embeds
