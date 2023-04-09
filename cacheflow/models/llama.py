"""1D LLaMA model compatible with HuggingFace weights."""
import os
import glob
import filelock
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig

from cacheflow.models import InputMetadata
from cacheflow.models.activation import SiluAndMul
from cacheflow.models.attention import LlamaCacheFlowAttention
from cacheflow.models.layernorm import RMSNorm
from cacheflow.models.sample import Sampler
from cacheflow.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from cacheflow.parallel_utils.tensor_parallel import (VocabParallelEmbedding,
                                                      ColumnParallelLinear,
                                                      RowParallelLinear)
from cacheflow.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(hidden_size, 2 * intermediate_size,
                                                 bias=False, gather_output=False,
                                                 perform_initialization=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size,
                                           bias=False, input_is_parallel=True,
                                           perform_initialization=False)
        if hidden_act != 'silu':
            raise ValueError(f'Unsupported activation: {hidden_act}. '
                             'Only silu is supported for now.')
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5

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
        self.attn = LlamaCacheFlowAttention(self.scaling, self.head_dim)

    def forward(
        self,
        positions: torch.LongTensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            positions, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.LongTensor,
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


class LlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   perform_initialization=False)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
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


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            gather_output=False,
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
        hidden_states = self.model(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["embed_tokens.weight", "lm_head.weight",
                                "qkv_proj.weight", "gate_proj.weight",
                                "up_proj.weight"]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(self, weights_path: str):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if "qkv_proj" in name or "gate_up_proj" in name:
                if "qkv_proj" in name:
                    original_name = "qkv_proj"
                    weight_names = ["q_proj", "k_proj", "v_proj"]
                    shard_size = param.shape[0] // 3
                else:
                    original_name = "gate_up_proj"
                    weight_names = ["gate_proj", "up_proj"]
                    shard_size = param.shape[0] // 2
                weights_to_concat = []
                for weight_name in weight_names:
                    weight = np.load(os.path.join(
                        weights_path, name.replace(original_name, weight_name)))
                    weights_to_concat.append(weight[
                        shard_size * tensor_model_parallel_rank
                        :shard_size * (tensor_model_parallel_rank + 1)])
                loaded_weight = torch.from_numpy(
                    np.concatenate(weights_to_concat, axis=0))
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
        if not os.path.isfile(os.path.join(model_name, "config.json")):
            raise ValueError("LLaMA model's model_name has to be a path"
                             "to the huggingface model's directory.")
        path = os.path.join(model_name, f"np")
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        lock_path = os.path.join(path, "file_lock")
        lock = filelock.FileLock(lock_path)

        with lock:
            test_weight_path = os.path.join(path, "model.embed_tokens.weight")
            if os.path.exists(test_weight_path):
                return path

            bin_files = glob.glob(os.path.join(model_name, "*.bin"))

            for bin_file in tqdm(bin_files, desc="Convert format"):
                state = torch.load(bin_file, map_location="cpu")
                for name, param in tqdm(state.items(), leave=False):
                    param_path = os.path.join(path, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())

            return path

    def initialize_dummy_weights(self) -> None:
        for param in self.state_dict().values():
            param.data.uniform_(-0.1, 0.1)
