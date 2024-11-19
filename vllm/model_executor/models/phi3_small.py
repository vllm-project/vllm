import math
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


def load_column_parallel_weight(param: torch.nn.Parameter,
                                loaded_weight: torch.Tensor):
    tp = get_tensor_model_parallel_world_size()
    rk = get_tensor_model_parallel_rank()
    assert param.size(0) * tp == loaded_weight.size(0)
    s = rk * param.size(0)
    e = (rk + 1) * param.size(0)
    loaded_weight = loaded_weight[s:e]
    assert param.shape == loaded_weight.shape
    param.data.copy_(loaded_weight)


class HeadMajorQKVParallelLinear(QKVParallelLinear):

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        return load_column_parallel_weight(param, loaded_weight)


class HeadMajorColumnParallelLinear(MergedColumnParallelLinear):

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor):
        return load_column_parallel_weight(param, loaded_weight)


@torch.compile(dynamic=True)
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.compile(dynamic=True)
def gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(torch.isinf(a_gelu), a_gelu,
                             a_gelu.clamp(min=None, max=limit))
        a_linear = torch.where(
            torch.isinf(a_linear),
            a_linear,
            a_linear.clamp(min=-limit, max=limit),
        )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)


class Phi3SmallMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        assert (self.config.hidden_act == "gegelu"
                ), "Only `gegelu` is supported for the 4.7 series of models .."
        self.hidden_size = config.hidden_size
        self.gegelu_limit = config.gegelu_limit
        self.intermediate_size = config.intermediate_size

        self.up_proj = HeadMajorColumnParallelLinear(
            self.hidden_size,
            2 * [self.intermediate_size],
            bias=True,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x):
        gate_up, _ = self.up_proj(x)
        x = gegelu(gate_up)
        x, _ = self.down_proj(x)
        return x


class Phi3SmallSelfAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.sparse_block_size = config.blocksparse_block_size
        self.homo_heads = config.blocksparse_homo_head_pattern
        self.local_blocks = config.blocksparse_num_local_blocks
        self.vert_stride = config.blocksparse_vert_stride

        assert (config.blocksparse_block_size ==
                config.blocksparse_triton_kernel_block_size)

        self.hidden_size = config.hidden_size
        # Number of Query Heads
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        # Number of total Key Value Heads before tensor parallel
        self.num_key_value_heads = config.num_key_value_heads
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads
        if self.tp_size > 1:
            assert self.num_key_value_heads % self.tp_size == 0
        self.num_kv_heads_per_partion = max(
            1, self.num_key_value_heads // self.tp_size)
        self.num_heads_per_partition = self.num_heads // self.tp_size

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_embedding_base = config.rope_embedding_base
        self.rope_position_scale = config.rope_position_scale
        self.is_causal = True

        norm_factor = None
        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)
        self.scale = 1 / norm_factor

        self.query_key_value = HeadMajorQKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
        )

        self.dense = RowParallelLinear(self.hidden_size,
                                       self.hidden_size,
                                       bias=True,
                                       quant_config=quant_config)

        if getattr(self.config, "rope_scaling", None) is not None:
            rope_scaling = self.config.rope_scaling
            for key in rope_scaling:
                if isinstance(rope_scaling[key], list):
                    rope_scaling[key] = tuple(rope_scaling[key])

            if "factor" not in rope_scaling:
                rope_scaling["factor"] = self.rope_position_scale
        else:
            rope_scaling = {
                "rope_type": "linear",
                "factor": self.rope_position_scale,
            }

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_embedding_base,
            rope_scaling=rope_scaling,
        )

        # blocksparse params
        self.blocksparse_block_size = config.blocksparse_block_size
        self.blocksparse_num_local_blocks = config.blocksparse_num_local_blocks
        self.blocksparse_vert_stride = config.blocksparse_vert_stride

        use_dense_attn = (getattr(self.config,
                                  "dense_attention_every_n_layers", None)
                          and (self.layer_idx + 1) %
                          self.config.dense_attention_every_n_layers == 0)

        bs_params = None
        if not use_dense_attn:
            bs_params = {
                'max_seqlen': self.max_position_embeddings,
                'num_heads': self.num_heads_per_partition,
                "num_kv_heads": self.num_kv_heads_per_partion,
                "block_size": self.sparse_block_size,
                "local_blocks": self.local_blocks,
                "vert_stride": self.vert_stride,
                "homo_head": self.homo_heads
            }

        self.attn = Attention(
            self.num_heads_per_partition,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads_per_partion,
            cache_config=cache_config,
            quant_config=quant_config,
            blocksparse_params=bs_params,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        qkv, _ = self.query_key_value(hidden_states)

        qkv = qkv.view(qkv.shape[:-1] +
                       (-1, (self.num_q_per_kv + 2), self.head_dim))
        q, k, v = qkv.split([self.num_q_per_kv, 1, 1], dim=-2)

        # NOTE: this is required by RotaryEmbed, which indeed does not have to
        # TODO: allow 3D QK for rotary forward
        q = q.reshape(-1, self.head_dim * self.num_heads_per_partition)
        k = k.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)
        v = v.reshape(-1, self.head_dim * self.num_kv_heads_per_partion)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata=attn_metadata)
        output, _ = self.dense(attn_output)

        return output


class Phi3SmallDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Phi3SmallSelfAttention(config,
                                                layer_idx,
                                                cache_config=cache_config,
                                                quant_config=quant_config)
        self.mlp = Phi3SmallMLP(config, quant_config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Phi3SmallModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.mup_embedding_multiplier = config.mup_embedding_multiplier
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Phi3SmallDecoderLayer(config,
                                                 int(prefix.split('.')[-1]),
                                                 cache_config, quant_config),
            prefix=f"{prefix}.layers")

        self.final_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            if (self.mup_embedding_multiplier is not None
                    and self.mup_embedding_multiplier > 0.0):
                hidden_states = hidden_states * self.mup_embedding_multiplier
        else:
            assert intermediate_tensors
            hidden_states = intermediate_tensors["hidden_states"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class Phi3SmallForCausalLM(nn.Module, SupportsPP):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Phi3SmallModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.vocab_size = config.vocab_size
        self.mup_width_multiplier = config.mup_width_multiplier
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # tokens in tiktoken but not used
        if hasattr(config, 'dummy_token_indices'):
            device = self.lm_head.weight.device
            self.register_buffer('dummy_token_indices',
                                 torch.LongTensor(
                                     config.dummy_token_indices).to(device),
                                 persistent=False)
        else:
            self.dummy_token_indices = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        if self.dummy_token_indices is not None and logits is not None:
            logits.index_fill_(-1, self.dummy_token_indices, -torch.inf)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        output_hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        output_hidden_states = output_hidden_states
        return output_hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits / self.mup_width_multiplier,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
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
