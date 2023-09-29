# coding=utf-8
"""Inference-only Refact model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.models.bloom import _get_alibi_slopes
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import VocabParallelEmbedding
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor, hf_model_weights_iterator,
    load_tensor_parallel_weights)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LayerNormWithoutBias(nn.LayerNorm):

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps, elementwise_affine=True, **factory_kwargs)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)


class RefactMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        mult: float,
        quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        multiple_of = 256
        intermediate_size = int(2 * (hidden_size * mult) / 3)
        self.intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.gate_up_proj = ParallelLinear.column(hidden_size,
                                                  2 * self.intermediate_size,
                                                  bias=False,
                                                  gather_output=False,
                                                  perform_initialization=False,
                                                  quant_config=quant_config)
        self.c_proj = ParallelLinear.row(self.intermediate_size,
                                         hidden_size,
                                         bias=False,
                                         input_is_parallel=True,
                                         perform_initialization=False,
                                         quant_config=quant_config)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class RefactAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        assert self.num_heads % tp_size == 0
        self.num_kv_heads = 1
        self.head_dim = hidden_size // self.total_num_heads
        self.kv_dim = self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.q = ParallelLinear.column(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            quant_config=quant_config,
        )
        self.kv = nn.Linear(
            self.hidden_size,
            2 * self.kv_dim,
            bias=False
        )
        self.c_proj = ParallelLinear.row(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            quant_config=quant_config,
        )
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()
        self.sa = PagedAttentionWithALiBi(
            self.num_heads,
            self.head_dim,
            self.scaling,
            slopes=alibi_slopes,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        del positions  # unused.
        q, _ = self.q(hidden_states)
        k, v = self.kv(hidden_states).split([self.kv_dim, self.kv_dim], dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.sa(q, k, v,
                              k_cache, v_cache,
                              input_metadata, cache_event)
        output, _ = self.c_proj(attn_output)
        return output


class RefactDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = RefactAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            quant_config=quant_config,
        )
        self.mlp = RefactMLP(
            hidden_size=self.hidden_size,
            mult=4.0,
            quant_config=quant_config,
        )
        self.ln_1 = LayerNormWithoutBias(
            self.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.ln_2 = LayerNormWithoutBias(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class RefactModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.h = nn.ModuleList([
            RefactDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        return hidden_states


class GPTRefactForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config
        self.transformer = RefactModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.ln_f = LayerNormWithoutBias(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = ParallelLinear.column(config.hidden_size,
                                             vocab_size,
                                             bias=False,
                                             gather_output=False,
                                             perform_initialization=False,
                                             quant_config=None)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        hidden_states = self.ln_f(hidden_states)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "wte.weight", "lm_head.weight",
        "q.weight", "gate_up_proj.weight"
    ]
    _row_parallel_weights = [
        "attn.c_proj.weight", "mlp.c_proj.weight"
    ]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        tp_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            if "wte.weight" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] * tp_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows,
                                         loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
