from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ACT2FN

from kaiju import KaijuTextConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import KaijuRMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig

# from vllm.attention import Attention
# from vllm.compilation.decorators import support_torch_compile

# from vllm.logger import init_logger
# from vllm.model_executor.layers.activation import GeluAndMul

# from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
#                                                QKVParallelLinear,
#                                                RowParallelLinear)
# from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.rotary_embedding import get_rope
# from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
# from vllm.model_executor.layers.vocab_parallel_embedding import (
#     VocabParallelEmbedding)
# from vllm.model_executor.model_loader.weight_utils import (
#     default_weight_loader, maybe_remap_kv_scale_name)
# from vllm.model_executor.sampling_metadata import SamplingMetadata
# from vllm.sequence import IntermediateTensors

# from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)

class KaijuMLP(nn.Module):
    def __init__(self, 
        hidden_size: int, 
        intermediate_size: int, 
        hidden_act: str,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.residual_scale = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.pre_ffn_norm = KaijuRMSNorm(self.hidden_size, eps=rms_norm_eps)

        # TODO: Megatron style TP (MergedColumnParallelLinear then RowParallelLinear)
        self.W_in = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_out = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # WARNING: In whippet checkpoints, there is an `args["quantize"]["ffn_clamp_middle_output"]`
        # It's only used in the backward pass in specific circumstances.
        hidden_states = x
        x = self.W_in(x)
        x = clamp(x, 4)
        x = self.act_fn(x)
        x = self.W_out(x)
        hidden_states *= self.residual_scale
        return x + hidden_states

@dataclass
class KaijuCache:
    key_states : Optional[torch.Tensor] = None
    value_states : Optional[torch.Tensor] = None

class KaijuAttention(nn.Module):
    def __init__(self, 
        config: KaijuTextConfig,
        max_position_embeddings: int,
        is_context_encoder: bool,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        attn_logits_soft_cap: Optional[float] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_context_encoder = is_context_encoder
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
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # TODO: Combine into single proj matrix and use QKVParallelLinear
        self.q_proj = nn.Linear(
            self.hidden_size, self.q_size, bias=False
        )
        if not self.is_context_encoder:
             self.k_proj = nn.Linear(
                self.hidden_size, self.kv_size, bias=False
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.kv_size, bias=False
            )
        
        # TODO: Use RowParallelLinear
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.pre_projection_norm = KaijuRMSNorm(self.config.hidden_size, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        self.is_sliding = layer_idx not in self.config.global_attention_layer_schedule
        if self.is_sliding:
            self.sliding_window = 1024
        else:
            self.sliding_window = None
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn"
        )

    def forward(
            self, 
            positions_embeddings: Tuple[torch.Tensor, torch.Tensor], 
            hidden_states: torch.Tensor, 
            kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> torch.Tensor:
        
        processed_hidden_states = self.pre_projection_norm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states = self.q_proj(processed_hidden_states).view(hidden_shape)

        if self.is_context_encoder:
            assert kv_cache is None
            key_states = kv_cache.key_states
            value_states = kv_cache.value_states
        else:
            key_states = self.k_proj(processed_hidden_states).view(hidden_shape)
            value_states = self.v_proj(processed_hidden_states).view(hidden_shape)
        
        if kv_cache is not None:
            key_states = kv_cache.key_states
            value_states = kv_cache.value_states
        

        # We should probably cache the clamped values.
        query_states = clamp(query_states, 4)
        key_states = clamp(key_states, 4)
        value_states = clamp(value_states, 4)

        # Should we cache post rope?
        query_states, key_states = apply_rotary_pos_emb_kaiju(query_states, key_states, cos, sin, unsqueeze_dim=2)
        
        # TODO: attention masking
        attn_output = self.attn(query_states, key_states, value_states)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        hidden_states *= self.residual_scale
        hidden_states += attn_output

        return hidden_states

class KaijuDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: KaijuTextConfig,
        is_context_encoder: bool,
        cache_config: Optional[CacheConfig] = None, 
        quant_config: Optional[QuantizationConfig] = None, 
        prefix: str = ""
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KaijuAttention(
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            is_context_encoder=is_context_encoder,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn"
        )

        self.mlp = KaijuMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(
        self, 
        positions_embeddings: Tuple[torch.Tensor, torch.Tensor], 
        hidden_states: torch.Tensor, 
        output_attentions: bool = False,
        kv_cache: Optional[KaijuCache] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention
        # attention module handles the residual stream update.
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
        )

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states,)
        # This isn't necessary for inference, we can consider writing a slow
        # attention implementation for debugging purposes.
        assert not output_attentions, "TODO: Support this"

        return outputs

@support_torch_compile
class KaijuModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.layer_to_kv_group = list(range(config.num_hidden_layers))
        for layers in config.share_kv_schedule:
            for layer_idx in layers:
                self.layer_to_kv_group[layer_idx] = min(layers)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Vocab parallel embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # TODO: Get rid of this scale by "compiling" it into the embedding weights, then 
        # when we convert the lm head/etc we can just adjust that scale.
        self.embedding_scale = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

        self.start_layer, self.end_layer, self.layers = make_layers_with_idx(
            config.num_hidden_layers,
            lambda prefix, idx: KaijuDecoderLayer(
                config, is_context_encoder=idx != self.layer_to_kv_group[idx], cache_config=cache_config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers"
        )












    