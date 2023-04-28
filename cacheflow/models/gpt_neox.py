"""1D GPT-NeoX model compatible with HuggingFace weights."""
import os
import glob
import filelock
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from huggingface_hub import snapshot_download

from cacheflow.models import InputMetadata
from cacheflow.models.attention import GPTNeoXCacheFlowAttention
from cacheflow.models.sample import Sampler
from cacheflow.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from cacheflow.parallel_utils.tensor_parallel import (VocabParallelEmbedding,
                                                      ColumnParallelLinear,
                                                      RowParallelLinear)
from cacheflow.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPTNeoXAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size

        self.query_key_value = ColumnParallelLinear(config.hidden_size,
                                                    3 * config.hidden_size,
                                                    gather_output=False,
                                                    perform_initialization=False)
        self.dense = RowParallelLinear(config.hidden_size, config.hidden_size,
                                       input_is_parallel=True,
                                       perform_initialization=False)

        scaling = self.head_size ** -0.5
        rotary_dim = int(self.head_size * config.rotary_pct)
        assert rotary_dim % 2 == 0
        self.attn = GPTNeoXCacheFlowAttention(scaling, rotary_dim)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)

        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            position_ids, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output, _ = self.dense(attn_output)
        return output


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(config.hidden_size,
                                                  config.intermediate_size,
                                                  gather_output=False,
                                                  perform_initialization=False)
        self.dense_4h_to_h = RowParallelLinear(config.intermediate_size, config.hidden_size,
                                               input_is_parallel=True,
                                               perform_initialization=False)
        if config.hidden_act != 'gelu':
            raise ValueError(f'Unsupported activation: {config.hidden_act}. '
                             'Only gelu is supported for now.')
        self.act = torch.nn.GELU()

    def forward(self, hidden_states):
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            position_ids=position_ids,
            hidden_states=attn_input,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output
        return hidden_states


class GPTNeoXModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_in = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                               perform_initialization=False)
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_in(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTNeoXForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = ColumnParallelLinear(config.hidden_size, config.vocab_size,
                                              bias=False, gather_output=False,
                                              perform_initialization=False)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.gpt_neox(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.embed_out.weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["embed_in.weight", "embed_out.weight", "dense_h_to_4h.weight", "dense_h_to_4h.bias"]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self, weights_path: str):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if "query_key_value" in name:
                # NOTE(woosuk): GPT-NeoX's fused QKV has the shape of
                # [num_heads * 3 * head_size, num_heads * head_size], while the
                # required shape is [3 * num_heads * head_size, num_heads * head_size].
                # Thus, we need weight conversion.
                loaded_weight = torch.from_numpy(
                    np.load(os.path.join(weights_path, name)))
                shard_size = param.shape[0]
                loaded_weight = loaded_weight[shard_size * tensor_model_parallel_rank
                                              :shard_size * (tensor_model_parallel_rank + 1)]

                num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // num_heads
                if 'query_key_value.weight' in name:
                    loaded_weight = loaded_weight.view(-1, 3, head_size, hidden_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, hidden_size).contiguous()
                elif 'query_key_value.bias' in name:
                    loaded_weight = loaded_weight.view(-1, 3, head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1).contiguous()
                else:
                    assert False
            else:
                loaded_weight = torch.from_numpy(
                    np.load(os.path.join(weights_path, name)))
                for p in self._column_parallel_weights:
                    if p in name:
                        shard_size = param.shape[0]
                        loaded_weight = loaded_weight[
                            shard_size * tensor_model_parallel_rank
                            :shard_size * (tensor_model_parallel_rank + 1)]
                        break
                for p in self._row_parallel_weights:
                    if p in name:
                        shard_size = param.shape[1]
                        loaded_weight = loaded_weight[
                            :,
                            shard_size * tensor_model_parallel_rank
                            :shard_size * (tensor_model_parallel_rank + 1)]
                        break

            assert param.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)

    @staticmethod
    def get_weights(model_name: str, path: str):
        path = os.path.join(path, f"{model_name}-np")
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        lock_path = os.path.join(path, "file_lock")
        lock = filelock.FileLock(lock_path)

        with lock:
            test_weight_path = os.path.join(
                path, "gpt_neox.embed_in.weight")
            if os.path.exists(test_weight_path):
                return path

            folder = snapshot_download(model_name, allow_patterns="*.bin",
                                       cache_dir=os.path.join(path, "cache"))
            bin_files = glob.glob(os.path.join(folder, "*.bin"))

            for bin_file in tqdm(bin_files, desc="Convert format"):
                state = torch.load(bin_file, map_location="cpu")
                for name, param in tqdm(state.items(), leave=False):
                    param_path = os.path.join(path, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())

            return path

    def initialize_dummy_weights(self) -> None:
        for param in self.state_dict().values():
            param.data.uniform_(-1e-3, 1e-3)
