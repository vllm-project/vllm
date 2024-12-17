from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.dbrx import DbrxConfig

from .interfaces import SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DbrxConfig,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        self.layer = ReplicatedLinear(
            self.d_model,
            self.num_total_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits


class DbrxExperts(FusedMoE):

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            num_experts=config.ffn_config.moe_num_experts,
            top_k=config.ffn_config.moe_top_k,
            hidden_size=config.d_model,
            intermediate_size=config.ffn_config.ffn_hidden_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            tp_size=get_tensor_model_parallel_world_size(),
        )
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.d_model = config.d_model
        self.intermediate_size = (self.config.ffn_config.ffn_hidden_size //
                                  self.tp_size)

    # Define custom weight loader for dbrx model
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, param_name: str):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        # DBRX uses GLU for each experts.
        # GLU has 3 linear layers: w1, v1 and w2.
        if weight_name.endswith("w1"):
            if param_name.endswith("weight"):
                loaded_weight = torch.reshape(
                    loaded_weight,
                    [-1, self.intermediate_size * self.tp_size, self.d_model],
                )
                param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :]
            elif param_name.endswith("weight_scale"):
                param_data[:, 0] = loaded_weight
            else:
                param_data = loaded_weight
        if weight_name.endswith("v1"):
            if param_name.endswith("weight"):
                loaded_weight = torch.reshape(
                    loaded_weight,
                    [-1, self.intermediate_size * self.tp_size, self.d_model],
                )
                param_data[:, shard_size:2 *
                           shard_size, :] = loaded_weight[:, shard, :]
            elif param_name.endswith("weight_scale"):
                param_data[:, 1] = loaded_weight
            else:
                param_data[:] = loaded_weight
        if weight_name.endswith("w2"):
            if param_name.endswith("weight"):
                loaded_weight = torch.reshape(
                    loaded_weight,
                    [-1, self.intermediate_size * self.tp_size, self.d_model],
                ).transpose(1, 2)
                param_data[:] = loaded_weight[:, :, shard]
            else:
                param_data[:] = loaded_weight


class DbrxMoE(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DbrxRouter(config, self.params_dtype)

        self.experts = DbrxExperts(config=config,
                                   quant_config=quant_config,
                                   params_dtype=self.params_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class DbrxAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.rope_theta = config.attn_config.rope_theta
        self.max_position = config.max_seq_len

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        hidden_states, _ = self.out_proj(attn_output)
        return hidden_states


class DbrxFusedNormAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config,
                                  cache_config,
                                  quant_config,
                                  prefix=f"{prefix}.attn")
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + x
        residual = hidden_states
        hidden_states = self.norm_2(hidden_states)
        return hidden_states, residual


class DbrxBlock(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(
            config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.norm_attn_norm")
        self.ffn = DbrxMoE(config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states, residual = self.norm_attn_norm(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DbrxModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_layers,
            lambda prefix: DbrxBlock(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.blocks",
        )
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias,
                                                      nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.d_model))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DbrxForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        if config.tie_word_embeddings:
            raise ValueError(
                "tie_word_embeddings is not supported for Dbrx models.")
        self.quant_config = quant_config
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(
                                         prefix, "transformer"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors,
                                         inputs_embeds)
        return hidden_states

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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        expert_params_mapping = [(
            "w13" if weight_name in ["w1", "v1"] else "w2",
            f"mlp.{weight_name}",
        ) for weight_name in ["w1", "v1", "w2"]]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for quark and
                # compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0
                                    else loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            if name.endswith(("w1", "w2", "v1")):
                name = name + "_weight"
            for param_name, weight_name in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, weight_name, name)
                break

            else:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
