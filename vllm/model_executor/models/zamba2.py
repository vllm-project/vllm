# SPDX-License-Identifier: Apache-2.0
"""Inference-only Zamba2 model."""
# Added by the Zyphra Technologies, 2025
from itertools import cycle
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Zamba2Config

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2, extra_groups_for_head_shards)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid
from .utils import make_empty_intermediate_tensors_factory, maybe_prefix

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Zamba2Attention(nn.Module):

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer2block_map = layer2block_map
        self.num_fwd_mem_blocks = len(layer2block_map)
        self.rope_theta = config.rope_theta

        self.attention_hidden_size = config.attention_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.scale = (self.attention_head_dim / 2)**-0.5

        if (self.attention_head_dim *
                self.num_attention_heads) != self.attention_hidden_size:
            raise ValueError(
                f"attention_hidden_size must be divisible by"
                f" num_attention_heads"
                f" (got `attention_hidden_size`: {self.attention_hidden_size}"
                f" and `num_heads`: {self.num_attention_heads}).")

        self.qkv_proj = QKVParallelLinear(
            self.attention_hidden_size,
            self.attention_head_dim,
            self.num_attention_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.attention_hidden_size,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        # Need to define separate Attention objects, because in recent vLLM
        # KV cache tensors are tied to specific Attention objects.
        self.dpa_list = nn.ModuleList([])
        j = bare_block_idx * (self.num_fwd_mem_blocks + config.num_mem_blocks -
                              1) // config.num_mem_blocks
        for block_idx in range(self.num_fwd_mem_blocks):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                dpa = Attention(
                    self.num_attention_heads,
                    self.attention_head_dim,
                    self.scale,
                    cache_config=cache_config,
                    prefix=f"{prefix}.attn.{j}",
                )
                j += 1
            else:
                dpa = nn.Identity()
            self.dpa_list.append(dpa)

        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for block_idx in range(self.num_fwd_mem_blocks):
                if block_idx % config.num_mem_blocks == bare_block_idx:
                    linear_q_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                    linear_k_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                    linear_v_adapter = nn.ModuleList([
                        ColumnParallelLinear(self.attention_hidden_size,
                                             config.adapter_rank,
                                             bias=False,
                                             quant_config=quant_config),
                        ColumnParallelLinear(config.adapter_rank,
                                             self.attention_hidden_size,
                                             bias=False,
                                             quant_config=quant_config),
                    ])
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()
                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        if config.use_mem_rope:
            self.rotary_emb = get_rope(
                head_size=self.attention_head_dim,
                rotary_dim=self.attention_head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=None,
                is_neox_style=True,
            )

    def forward(
        self,
        hidden_states,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([
            self.attention_hidden_size, self.attention_hidden_size,
            self.attention_hidden_size
        ],
                                                           dim=-1)

        block_idx = self.layer2block_map[layer_idx]
        if self.config.use_shared_attention_adapter:
            q_lora_output = self.linear_q_adapter_list[block_idx][0](
                hidden_states)[0]
            q_lora_output = self.linear_q_adapter_list[block_idx][1](
                q_lora_output)[0]
            query_states = query_states + q_lora_output

            k_lora_output = self.linear_k_adapter_list[block_idx][0](
                hidden_states)[0]
            k_lora_output = self.linear_k_adapter_list[block_idx][1](
                k_lora_output)[0]
            key_states = key_states + k_lora_output

            v_lora_output = self.linear_v_adapter_list[block_idx][0](
                hidden_states)[0]
            v_lora_output = self.linear_v_adapter_list[block_idx][1](
                v_lora_output)[0]
            value_states = value_states + v_lora_output

        if self.config.use_mem_rope:
            query_states, key_states = self.rotary_emb(position_ids,
                                                       query_states,
                                                       key_states)

        # NOTE: No need anymore to pass specific kv_cache tensor,
        # but keeping it for API compatibility
        y = self.dpa_list[block_idx](query_states, key_states, value_states,
                                     kv_caches[block_idx], attn_metadata)
        y, _ = self.o_proj(y)
        return y


class Zamba2MLP(nn.Module):

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer2block_map = layer2block_map
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_fwd_mem_blocks = len(layer2block_map)

        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=self.config.add_bias_linear,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(self.intermediate_size,
                                           self.hidden_size,
                                           bias=self.config.add_bias_linear,
                                           quant_config=quant_config)
        if config.hidden_act != "gelu":
            raise ValueError(f"Only gelu activation is supported"
                             f" (got `hidden_act`: {config.hidden_act})")
        self.act_fn = F.gelu

        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for block_idx in range(self.num_fwd_mem_blocks):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                gate_up_proj_adapter = nn.ModuleList([
                    ColumnParallelLinear(config.hidden_size,
                                         self.config.adapter_rank,
                                         bias=False,
                                         quant_config=quant_config),
                    ColumnParallelLinear(config.adapter_rank,
                                         2 * self.intermediate_size,
                                         bias=False,
                                         quant_config=quant_config),
                ])
            else:
                gate_up_proj_adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

    def forward(self, hidden_states, layer_idx):
        gate_up_state, _ = self.gate_up_proj(hidden_states)
        block_idx = self.layer2block_map[layer_idx]
        lora_output = self.gate_up_proj_adapter_list[block_idx][0](
            hidden_states)[0]
        lora_output = self.gate_up_proj_adapter_list[block_idx][1](
            lora_output)[0]
        gate_up_state = gate_up_state + lora_output

        gate_up_state = torch.chunk(gate_up_state, 2, dim=-1)
        hidden_state = self.act_fn(gate_up_state[0]) * gate_up_state[1]
        output, _ = self.down_proj(hidden_state)
        return output


class Zamba2AttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        layer2block_map: Dict[int, int],
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Zamba2Attention(
            config,
            bare_block_idx=bare_block_idx,
            layer2block_map=layer2block_map,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.feed_forward = Zamba2MLP(
            config,
            bare_block_idx=bare_block_idx,
            layer2block_map=layer2block_map,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(2 * config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        # The argument original_hidden_states is concatenated with hidden_states
        # (which is the output of the previous (mamba) layer).
        # The concatenated tensor is then used as input of the pre-attention
        # RMSNorm (see fig. 2 in https://arxiv.org/pdf/2405.16712).
        hidden_states = torch.concatenate(
            [hidden_states, original_hidden_states], dim=-1)

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            position_ids=positions,
            layer_idx=layer_idx,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        # feed-forward (MLP)
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, layer_idx=layer_idx)

        return hidden_states


class Zamba2MambaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Zamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        intermediate_size = config.mamba_expand * config.hidden_size
        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=intermediate_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.add_bias_linear,
            n_groups=config.mamba_ngroups,
            num_heads=config.n_mamba_heads,
            head_dim=intermediate_size // config.n_mamba_heads,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            chunk_size=config.chunk_size,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        transformer_hidden_states: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        residual = hidden_states

        # `transformer_hidden_states` is the output from shared
        # transformer + linear layer (see fig. 2 in
        # https://arxiv.org/pdf/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba
        # layer below (as described in eq. (6) of
        # https://arxiv.org/pdf/2405.16712).
        if transformer_hidden_states is not None:
            hidden_states = hidden_states + transformer_hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states,
            attn_metadata=attn_metadata,
            mamba_cache_params=mamba_cache_params,
            sequence_idx=sequence_idx,
        )

        # residual connection after mamba
        hidden_states = residual + hidden_states

        return hidden_states


class Zamba2HybridLayer(nn.Module):

    def __init__(
        self,
        shared_transformer: Zamba2AttentionDecoderLayer,
        linear: ColumnParallelLinear,
        mamba: Zamba2MambaDecoderLayer,
    ):
        super().__init__()
        self.shared_transformer = shared_transformer
        self.linear = linear
        self.mamba_decoder = mamba

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        mamba_cache_params: Optional[MambaCacheParams] = None,
        sequence_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:

        transformer_hidden_states = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        transformer_hidden_states, _ = self.linear(transformer_hidden_states)

        layer_outputs = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attn_metadata=attn_metadata,
            mamba_cache_params=mamba_cache_params,
            sequence_idx=sequence_idx,
        )

        return layer_outputs


class Zamba2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Implement PP, need to use make_layers()

        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)
        assert not is_lora_enabled

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        layer2block_map = {
            layer_idx: block_idx
            for block_idx, layer_idx in enumerate(config.hybrid_layer_ids)
        }
        blocks = cycle([
            Zamba2AttentionDecoderLayer(config,
                                        bare_block_idx=idx,
                                        layer2block_map=layer2block_map,
                                        cache_config=cache_config,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}")
            for idx in range(config.num_mem_blocks)
        ])
        layers = []
        for layer_type in config.layers_block_type:
            mamba_layer = Zamba2MambaDecoderLayer(config,
                                                  quant_config=quant_config)
            if layer_type == "hybrid":
                block = next(blocks)
                linear_layer = ColumnParallelLinear(config.hidden_size,
                                                    config.hidden_size,
                                                    bias=False,
                                                    quant_config=quant_config)
                layers.append(
                    Zamba2HybridLayer(block, linear_layer, mamba_layer))
            else:
                layers.append(mamba_layer)
        self.layers = nn.ModuleList(layers)
        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        # TODO: decide whether we want to implement PP support
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        seq_idx = None
        if attn_metadata.num_prefills > 0:
            seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
            for i, (srt, end) in enumerate(
                    zip(
                        attn_metadata.query_start_loc,
                        attn_metadata.query_start_loc[1:],
                    )):
                seq_idx[srt:end] = i
            seq_idx.unsqueeze_(0)

        original_hidden_states = torch.clone(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                original_hidden_states=original_hidden_states,
                layer_idx=layer_idx,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                mamba_cache_params=mamba_cache_params.at_layer_idx(layer_idx),
                sequence_idx=seq_idx,
            )
            hidden_states = layer_outputs

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class Zamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.model = Zamba2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        if self.scheduler_config is not None and \
            not self.model_config.enforce_eager:
            if self.scheduler_config.max_num_seqs > \
                vllm_config.compilation_config.max_capture_size:
                self.max_batch_size = \
                    vllm_config.compilation_config.max_capture_size
            else:
                self.max_batch_size = vllm_config.pad_for_cudagraph(
                    self.scheduler_config.max_num_seqs)
        elif self.scheduler_config is not None:
            # For eager just take the scheduler_config if avail
            self.max_batch_size = self.scheduler_config.max_num_seqs
        else:
            self.max_batch_size = 8192 + 2

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):

        if self.mamba_cache is None:
            num_mamba_layers = self.config.num_hidden_layers
            self.mamba_cache = MambaCacheManager(
                self.lm_head.weight.dtype, num_mamba_layers,
                self.max_batch_size, *self._get_mamba_cache_shape())

        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            mamba_cache_params,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()

        intermediate_size = self.config.mamba_expand * self.config.hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba_ngroups + extra_groups_for_head_shards(
            self.config.mamba_ngroups, world_size))

        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size +
                    2 * n_groups * self.config.mamba_d_state)
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.config.mamba_d_conv - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(divide(intermediate_size, self.config.mamba_headdim),
                   world_size),
            self.config.mamba_headdim,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        weights_dict = {}
        for key, loaded_weight in weights:
            if "A_log" in key:
                key = key.replace("A_log", "A")
            weights_dict[key] = loaded_weight

        params_dict = dict(self.named_parameters())
        for chkpt_weight_name, loaded_weight in weights_dict.items():
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in chkpt_weight_name:
                    continue
                chkpt_weight_name = chkpt_weight_name.replace(
                    weight_name, param_name)
                param = params_dict[chkpt_weight_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if chkpt_weight_name not in params_dict:
                    continue
                param = params_dict[chkpt_weight_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
