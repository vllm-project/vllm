from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
# from vllm.model_executor.layers.temp_sampler import TempSampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights,
                                              load_tensor_parallel_weights2)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs
from awq.quantize.qmodule import WQLinear
import awq_inference_engine
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

KVCache = Tuple[torch.Tensor, torch.Tensor]

class QuantLlamaQRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        
        # self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        # self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        # print(positions.shape, query.shape, key.shape, self.cos_sin_cache.shape)
        query = query.contiguous()
        key = key.contiguous()
        awq_inference_engine.rotary_embedding_neox(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache,
        )
        return query, key



class LlamaQMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.in_features = hidden_size
        self.intermediate_size = intermediate_size
        self.out_features = hidden_size
        self.w_bit = 4
        self.g_size = 128

        self.gate_up_proj = WQLinear(self.w_bit, self.g_size, self.in_features, 2 * self.intermediate_size, False, 'cuda')
        self.down_proj = WQLinear(self.w_bit, self.g_size, self.intermediate_size, self.out_features, False, 'cuda')
        self.act_fn = SiluAndMul()


    def forward(self, x):
        # return self.down_proj(self.custom_LlamaQ_mlp(x))
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
    
    def custom_LlamaQ_mlp(self, x):
        out_shape = x.shape[:-1] + (self.intermediate_size, )
        x = x.reshape(-1, x.shape[-1])

        gate_output = awq_inference_engine.gemm_forward_cuda(
            x, self.gate_proj.qweight, self.gate_proj.scales, self.gate_proj.qzeros, 8
        )
        gate_output = self.act_fn(gate_output)

        up_output = awq_inference_engine.gemm_forward_cuda(
            x, self.up_proj.qweight, self.up_proj.scales, self.up_proj.qzeros, 8
        )
        c = gate_output * up_output
        c = c.reshape(out_shape)
        return c


class LlamaQAttention2(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # tensor_model_parallel_world_size = (
        #     get_tensor_model_parallel_world_size())
        tensor_model_parallel_world_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.w_bit = 4
        self.g_size = 128

        self.qkv_proj = WQLinear(self.w_bit, self.g_size, hidden_size, 3 * self.total_num_heads * self.head_dim, False, 'cuda')
        self.o_proj = WQLinear(self.w_bit, self.g_size, self.total_num_heads * self.head_dim, hidden_size, False, 'cuda')

        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.head_dim,
                                           self.scaling,
                                           rotary_dim=self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # print(f"qkv proj size: {self.qkv_proj.shape}, hidden_states size: {hidden_states.shape} ")
        # 这里把qkv_proj和o_proj都变成WQLinear

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output = self.o_proj(attn_output)

        return output


class LlamaQAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # tensor_model_parallel_world_size = (
        #     get_tensor_model_parallel_world_size())
        tensor_model_parallel_world_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )
        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.head_dim,
                                           self.scaling,
                                           rotary_dim=self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # print(f"qkv proj size: {self.qkv_proj.shape}, hidden_states size: {hidden_states.shape} ")
        # 这里把qkv_proj和o_proj都变成WQLinear
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaQDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaQAttention2(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaQMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
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
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaQModel(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size, config.hidden_size, perform_initialization=False)
        self.layers = nn.ModuleList([
            LlamaQDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaQForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaQModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)
        #self.sampler = TempSampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        
        return next_tokens

    _column_parallel_weights = [
        "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight",
        "gate_proj.weight", "up_proj.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    _column_parallel_weights_fp16 = [
        "embed_tokens.weight", "lm_head.weight", "model.norm.weight"
    ]

    _row_parallel_weights_fp16 = []

    _column_parallel_weights_int4 = [
        "qkv_proj.qweight", "gate_proj.qweight", "up_proj.qweight",
        "qkv_proj.qzeros", "gate_proj.qzeros", "up_proj.qzeros",
        "qkv_proj.scales", "gate_proj.scales", "up_proj.scales",
        # "input_layernorm", "post_attention_layernorm"
    ]

    _row_parallel_weights_int4 = ["o_proj.qweight", "down_proj.qweight", 
                                  "o_proj.qzeros", "down_proj.qzeros",
                                  "o_proj.scales", "down_proj.scales"]


    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        # tensor_model_parallel_world_size = (
        #     get_tensor_model_parallel_world_size())
        tensor_model_parallel_world_size = 1
        # tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        tensor_model_parallel_rank = 0
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] *
                                     tensor_model_parallel_world_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows,
                                         loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
    
    def load_mix_weights(self,
                     model_name_or_path: str,
                     q_weight_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        # tensor_model_parallel_world_size = (
        #     get_tensor_model_parallel_world_size())
        tensor_model_parallel_world_size = 1
        # tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        tensor_model_parallel_rank = 0
        state_dict = self.state_dict()

        column_parallel_weights_fp16 = [
            # "embed_tokens.weight", "lm_head.weight", "model.norm.weight",
            # "input_layernorm", "post_attention_layernorm"
            "embed_tokens.weight", "lm_head.weight", "qkv_proj.weight"
        ]

        row_parallel_weights_fp16 = ["o_proj.weight"]

        column_parallel_weights_int4 = [
            "gate_proj.qweight", "up_proj.qweight",
            "gate_proj.qzeros", "up_proj.qzeros",
            "gate_proj.scales", "up_proj.scales"
        ]

        row_parallel_weights_int4 = ["down_proj.qweight", "down_proj.qzeros", "down_proj.scales"]        



        # load fp16
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue
            
            if "mlp" in name:
                continue

            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] *
                                     tensor_model_parallel_world_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows,
                                         loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[0] // 3
                loaded_weight = loaded_weight[
                    shard_size * tensor_model_parallel_rank:shard_size *
                    (tensor_model_parallel_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            param = state_dict[name]
            # print(f"fp16 layer name: {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights_fp16,
                                         row_parallel_weights_fp16,
                                         tensor_model_parallel_rank)
        print("****************** load int weight ***********************")
        # load int4
        for name, loaded_weight in hf_model_weights_iterator(
                q_weight_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens" in name or "lm_head" in name:
                continue
            
            if "self_attn" in name:
                continue

            if "input_layernorm" in name or "post_attention_layernorm" in name:
                continue

            if "model.norm.weight" in name:
                continue
            
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[1] // 2
                start = shard_size * stride_id
                end = shard_size * (stride_id + 1)
                param_slice = param.data[:, start:end]

                print(f"{name} param_slice.shape: {param_slice.shape}, loaded_weight.shape: {loaded_weight.shape}")
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            # print(f"int4 layer name: {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         column_parallel_weights_int4,
                                         row_parallel_weights_int4,
                                         tensor_model_parallel_rank)

    def load_int4_weights(self,
                     model_name_or_path: str,
                     q_weight_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        # tensor_model_parallel_world_size = (
        #     get_tensor_model_parallel_world_size())
        tensor_model_parallel_world_size = 1
        # tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        tensor_model_parallel_rank = 0
        state_dict = self.state_dict()

        q_proj_shard_size = (self.config.hidden_size // tensor_model_parallel_world_size)
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.num_key_value_heads // tensor_model_parallel_world_size)

        print(f"q_proj_shard_size: {q_proj_shard_size}, kv_proj_shard_size: {kv_proj_shard_size}")
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size,
             q_proj_shard_size + kv_proj_shard_size),
        ]
        # load int4
        for name, loaded_weight in hf_model_weights_iterator(
                q_weight_path, cache_dir, use_np_cache):
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] *           tensor_model_parallel_world_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows,
                                         loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)
            
            is_attention_weight = False

            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                # print(f"int4 layer name: {name}")
                # print(f"stride_id: {stride_id}, att_weight_name: {att_weight_name}")
                param_name = name.replace(att_weight_name, "qkv_proj")

                param = state_dict[param_name]
                shard_size = param.shape[1] // 3
                # loaded_weight = loaded_weight[
                #     shard_size * tensor_model_parallel_rank:shard_size *
                #     (tensor_model_parallel_rank + 1)]
                param_slice = param.data[:, shard_size * stride_id:shard_size *
                                         (stride_id + 1)]
                print(f"*** {param_name}***  param.shape: {param.shape}, param_slice.shape: {param_slice.shape}, loaded_weight.shape: {loaded_weight.shape}")
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[1] // 2
                start = shard_size * stride_id
                end = shard_size * (stride_id + 1)
                param_slice = param.data[:, start:end]

                print(f"{name} param_slice.shape: {param_slice.shape}, loaded_weight.shape: {loaded_weight.shape}")
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            print(f"int4 layer name: {name}")
            load_tensor_parallel_weights2(param, loaded_weight, name,
                                         tensor_model_parallel_rank)
