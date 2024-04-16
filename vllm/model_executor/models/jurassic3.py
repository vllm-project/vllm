# coding=utf-8

"""Inference-only Jurassic model."""
from typing import List, Optional, Tuple

import torch
from torch import nn
import os

from vllm.transformers_utils.configs.jurassic3 import Jurassic3Config
from vllm.config import LoRAConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Jurassic3MoE(nn.Module):
    """A tensor-parallel MoE implementation for Jurassic3 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            params_dtype: Optional[torch.dtype] = None,
            tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if self.num_total_experts > 1:
            #   init expert router iff this layer has multiple experts
            self.gate = ReplicatedLinear(self.hidden_size,
                                         self.num_total_experts,
                                         bias=False,
                                         params_dtype=self.params_dtype,
                                         linear_method=None)

        self.ws = nn.Parameter(
            torch.empty(self.num_total_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device="cuda",
                        dtype=self.params_dtype))
        self.w2s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda",
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
        })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id,
            shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        if self.num_total_experts > 1:
            router_logits, _ = self.gate(hidden_states)
        else:
            router_logits = torch.ones([hidden_states.shape[0], 1], device=hidden_states.device,
                                       dtype=hidden_states.dtype)

        final_hidden_states = fused_moe(hidden_states,
                                        self.ws,
                                        self.w2s,
                                        router_logits,
                                        self.top_k,
                                        renormalize=False,  # Mixtral normalize the expert probs to 1. We don't!
                                        inplace=True)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)


class Jurassic3Mamba(nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba(d_model=hidden_size, layer_idx=layer_idx)

    def forward(self, hidden_states: torch.Tensor, cache = None):
        max_seqlen = int(os.environ.get("MAMBA_MAX_SEQLEN", "2048"))
        inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=hidden_states.shape[0])
        if cache is not None:
            inference_params.key_value_memory_dict[self.layer_idx] = cache
        res = self.mamba(hidden_states, inference_params=inference_params)
        return res, inference_params.key_value_memory_dict

class Jurassic3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        use_positional_embeddings: bool = False,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        linear_method: Optional[LinearMethodBase] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_positional_embeddings = use_positional_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        if self.use_positional_embeddings:
            #   define positional embeddings conditioned on flag
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )

        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        #   TODO - add embedding flag
        if self.use_positional_embeddings:
            q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Jurassic3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Jurassic3Config,
        is_attn_layer: bool,
        is_expert_layer: bool,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.layer_idx = layer_idx

        self.is_attn_layer = is_attn_layer
        self.is_expert_layer = is_expert_layer

        if self.is_attn_layer:
            self.self_attn = Jurassic3Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                max_position=config.max_position_embeddings,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=rope_theta,
                sliding_window=config.sliding_window,
                linear_method=linear_method,
            )
        else:
            self.mamba = Jurassic3Mamba(hidden_size=self.hidden_size,layer_idx=layer_idx)

        actual_num_experts = config.num_experts if self.is_expert_layer else 1
        actual_num_experts_per_tok = config.num_experts_per_tok if self.is_expert_layer else 1

        self.block_sparse_moe = Jurassic3MoE(
            num_experts=actual_num_experts,
            top_k=actual_num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.is_attn_layer:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
            )
        else:
            cache = None
            if not input_metadata.is_prompt:
                for mamba_metadata in input_metadata.mamba_metadata:
                    # check if batch size of cache fits "n"
                    if mamba_metadata["cache"][self.layer_idx][0].shape[0] < mamba_metadata["n"]:
                        k_cache = mamba_metadata["cache"][self.layer_idx][0].repeat_interleave(mamba_metadata["n"],dim=0)
                        v_cache = mamba_metadata["cache"][self.layer_idx][1].repeat_interleave(mamba_metadata["n"],dim=0)
                        mamba_metadata["cache"][self.layer_idx] = (k_cache,v_cache)

                # mamba requires concatenated cache
                if len(input_metadata.mamba_metadata) > 1:
                    k_cache = torch.concat([req["cache"][self.layer_idx][0] for req in input_metadata.mamba_metadata],dim=0)
                    v_cache = torch.concat([req["cache"][self.layer_idx][1] for req in input_metadata.mamba_metadata],dim=0)
                    cache = (k_cache,v_cache)

            hidden_states ,cache = self.mamba(hidden_states, cache=cache)

            sample_id = 0
            # split cache back to individual requests
            for req_mamba_metadata in input_metadata.mamba_metadata:
                n = req_mamba_metadata["n"] if not input_metadata.is_prompt else 1
                req_mamba_metadata["cache"][self.layer_idx] = (cache[self.layer_idx][0][sample_id:sample_id+n]
                                                                ,cache[self.layer_idx][1][sample_id:sample_id+n])
                sample_id += n


        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class Jurassic3Model(nn.Module):
    def __init__(
            self,
            config: Jurassic3Config,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        #   init each model layer, decide if it's mamba/attention and has experts and pass it down

        module_list = []
        for i in range(config.num_hidden_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            module_list.append(
                Jurassic3DecoderLayer(
                    config,
                    is_attn_layer=is_attn,
                    is_expert_layer=is_expert,
                    layer_idx=i,
                    linear_method=linear_method,
                )
            )

        self.layers = nn.ModuleList(module_list)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], input_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Jurassic3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
            self,
            config: Jurassic3Config,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = Jurassic3Model(config,
                                    linear_method,
                                    lora_config=lora_config)
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
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
            self,
            hidden_states: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("ws" if weight_name in ["w1", "w3"] else "w2s",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=True):  # erez - might need to change later to False
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
