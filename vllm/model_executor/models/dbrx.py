# coding=utf-8
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.dbrx import DbrxConfig


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


class DbrxExperts(nn.Module):
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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.d_model = config.d_model
        self.intermediate_size = (config.ffn_config.ffn_hidden_size //
                                  self.tp_size)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DbrxRouter(config, self.params_dtype)
        self.ws = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                2 * self.intermediate_size,
                self.d_model,
                device="cuda",
                dtype=self.params_dtype,
            ))
        self.w2s = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.d_model,
                self.intermediate_size,
                device="cuda",
                dtype=self.params_dtype,
            ))

        set_weight_attrs(
            self.ws,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2s,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        # DBRX uses GLU for each experts.
        # GLU has 3 linear layers: w1, v1 and w2.
        if weight_name.endswith("w1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            )
            param_data[:, 0:shard_size, :] = loaded_weight[:, shard, :]
        if weight_name.endswith("v1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            )
            param_data[:,
                       shard_size:2 * shard_size, :] = loaded_weight[:,
                                                                     shard, :]
        if weight_name.endswith("w2"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.d_model],
            ).transpose(1, 2)
            param_data[:] = loaded_weight[:, :, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(hidden_states)
        final_hidden_states = fused_moe(
            hidden_states,
            self.ws,
            self.w2s,
            router_logits,
            self.top_k,
            renormalize=True,
            inplace=True,
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


class DbrxAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
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
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

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
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config, quant_config)
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
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.norm_attn_norm = DbrxFusedNormAttention(config, quant_config)
        self.ffn = DbrxExperts(config, quant_config)

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

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            [DbrxBlock(config, quant_config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias,
                                                      nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class DbrxForCausalLM(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(config, quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
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
        expert_params_mapping = [(
            "ws" if weight_name in ["w1", "v1"] else "w2s",
            f"experts.mlp.{weight_name}",
        ) for weight_name in ["w1", "v1", "w2"]]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            for param_name, weight_name in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, weight_name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
