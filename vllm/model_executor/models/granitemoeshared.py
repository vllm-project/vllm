# SPDX-License-Identifier: Apache-2.0
"""Inference-only GraniteMoeShared model.

The architecture is the same as granitemoe but with the addition of shared
experts.
"""
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers.models.granitemoeshared import GraniteMoeSharedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from . import mixtral
from .granitemoe import GraniteMoeAttention, GraniteMoeMoE
from .interfaces import SupportsLoRA, SupportsPP
from .utils import make_layers, maybe_prefix


class GraniteMoeSharedMLP(nn.Module):

    def __init__(
        self,
        config: GraniteMoeSharedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.input_linear = MergedColumnParallelLinear(
            input_size=self.input_size,
            output_sizes=[self.hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_linear")
        self.output_linear = RowParallelLinear(
            self.hidden_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_linear")
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.input_linear(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.output_linear(hidden_states)
        return hidden_states


class GraniteMoeSharedDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GraniteMoeSharedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = GraniteMoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attention_multiplier=config.attention_multiplier)
        self.block_sparse_moe = GraniteMoeMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe")
        self.shared_mlp = None if \
            getattr(config, 'shared_intermediate_size', 0) == 0 \
            else GraniteMoeSharedMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp"
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            # create a copy since block_sparse_moe modifies in-place
            moe_hidden_states = hidden_states.clone()
            moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
            del moe_hidden_states
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


@support_torch_compile
class GraniteMoeSharedModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
        self.embedding_multiplier = config.embedding_multiplier

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: GraniteMoeSharedDecoderLayer(
                config, cache_config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            hidden_states *= self.embedding_multiplier
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GraniteMoeSharedForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = GraniteMoeSharedModel(vllm_config=vllm_config,
                                           prefix=maybe_prefix(
                                               prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                scale=1 /
                                                self.config.logits_scaling)

        self.sampler = get_sampler()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        new_weights = {}
        for n, p in weights:
            if n.endswith('.block_sparse_moe.input_linear.weight'):
                for e in range(p.size(0)):
                    w1_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w1.weight")
                    w3_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w3.weight")
                    w1_param, w3_param = p[e].chunk(2, dim=0)
                    assert w1_name not in new_weights
                    assert w3_name not in new_weights
                    new_weights[w1_name] = w1_param
                    new_weights[w3_name] = w3_param
            elif n.endswith('.block_sparse_moe.output_linear.weight'):
                for e in range(p.size(0)):
                    w2_name = n.replace(
                        '.block_sparse_moe.output_linear.weight',
                        f".block_sparse_moe.experts.{e}.w2.weight")
                    w2_param = p[e]
                    assert w2_name not in new_weights
                    new_weights[w2_name] = w2_param
            elif n.endswith('.block_sparse_moe.router.layer.weight'):
                gate_name = n.replace('.block_sparse_moe.router.layer.weight',
                                      ".block_sparse_moe.gate.weight")
                assert gate_name not in new_weights
                new_weights[gate_name] = p
            elif n == 'lm_head.weight' and self.config.tie_word_embeddings:
                pass
            else:
                new_weights[n] = p
        return mixtral.MixtralForCausalLM.load_weights(self,
                                                       new_weights.items())
