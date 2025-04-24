# SPDX-License-Identifier: Apache-2.0
"""Inference-only Bamba model."""
# Added by the IBM Team, 2024
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import BambaConfig

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2, extra_groups_for_head_shards)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

from .interfaces import (HasInnerState, IsHybrid, SupportsLoRA, SupportsPP,
                         SupportsQuant, SupportsV0Only)
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class BambaMLP(nn.Module):

    def __init__(
        self,
        config: BambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
        )
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class BambaMixerDecoderLayer(nn.Module):

    def __init__(self,
                 config: BambaConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.mamba = MambaMixer2(hidden_size= config.hidden_size,
                                ssm_state_size = config.mamba_d_state,
                                conv_kernel_size = config.mamba_d_conv,
                                intermediate_size = config.mamba_expand *\
                                                    config.hidden_size,
                                use_conv_bias = config.mamba_conv_bias,
                                use_bias = config.mamba_proj_bias,
                                n_groups=config.mamba_n_groups,
                                num_heads=config.mamba_n_heads,
                                head_dim=config.mamba_d_head,
                                rms_norm_eps=config.rms_norm_eps,
                                activation=config.hidden_act,
                                quant_config=quant_config)

        self.feed_forward = BambaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.mamba(hidden_states, mamba_cache_params,
                                   mamba2_metadata)
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class BambaAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: BambaConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = self.head_dim * config.partial_rotary_factor
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = config.attn_rotary_emb  # for backward compatibility
        else:
            rotary_dim = self.head_dim  # default

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            rope_scaling=rope_scaling,
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

        self.feed_forward = BambaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": BambaAttentionDecoderLayer,
    "mamba": BambaMixerDecoderLayer
}


class BambaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: BambaConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

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

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.layers_block_type[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        attn_metadata = get_forward_context().attn_metadata

        mamba2_metadata = prepare_mamba2_metadata(
            chunk_size=self.config.mamba_chunk_size,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
        )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        residual = None
        num_attn = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, BambaAttentionDecoderLayer):
                num_attn += 1

            layer_mamba_cache_params = None
            if isinstance(layer, BambaMixerDecoderLayer):
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    i - num_attn)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BambaForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                       IsHybrid, SupportsV0Only, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Bamba currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = BambaModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
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
        )
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:

            num_mamba_layers = self.model_config.get_num_layers_by_block_type(
                self.vllm_config.parallel_config, LayerBlockType.mamba)

            self.mamba_cache = MambaCacheManager(
                self.vllm_config, self.lm_head.weight.dtype, num_mamba_layers,
                *self._get_mamba_cache_shape())
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)
        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size

        conv_state_shape, temporal_state_shape = None, None

        intermediate_size = self.config.mamba_expand * hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (self.config.mamba_n_groups + extra_groups_for_head_shards(
            self.config.mamba_n_groups, world_size))

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
            divide(self.config.mamba_n_heads, world_size),
            self.config.mamba_d_head,
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
