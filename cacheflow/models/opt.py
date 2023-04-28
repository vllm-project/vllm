"""1D OPT model compatible with HuggingFace weights."""
import os
import glob
import filelock
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import OPTConfig
from huggingface_hub import snapshot_download

from cacheflow.models import InputMetadata
from cacheflow.models.attention import OPTCacheFlowAttention
from cacheflow.models.sample import Sampler
from cacheflow.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from cacheflow.parallel_utils.tensor_parallel import (VocabParallelEmbedding,
                                                      ColumnParallelLinear,
                                                      RowParallelLinear)
from cacheflow.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.LongTensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = ColumnParallelLinear(embed_dim, 3 * embed_dim, bias=bias,
                                             gather_output=False,
                                             perform_initialization=False)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=bias,
                                          input_is_parallel=True,
                                          perform_initialization=False)
        self.attn = OPTCacheFlowAttention(scale=self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        key_cache, value_cache = kv_cache
        attn_output = self.attn(
            q, k, v, key_cache, value_cache, input_metadata, cache_event)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        assert config.activation_function == 'relu'
        self.activation_fn = nn.ReLU()

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(self.embed_dim, config.ffn_dim,
                                        bias=config.enable_bias,
                                        gather_output=False,
                                        perform_initialization=False)
        self.fc2 = RowParallelLinear(config.ffn_dim, self.embed_dim,
                                     bias=config.enable_bias,
                                     input_is_parallel=True,
                                     perform_initialization=False)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.word_embed_proj_dim,
                                                   perform_initialization=False)
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states, kv_caches[i], input_metadata, cache_event)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):

    def __init__(self, config: OPTConfig):
        super().__init__()
        self.decoder = OPTDecoder(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        return self.decoder(
            input_ids, positions, kv_caches, input_metadata, cache_events)


class OPTForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = OPTModel(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        self.lm_head_weight = self.model.decoder.embed_tokens.weight
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
            self.lm_head_weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["embed_tokens.weight", "fc1.weight", "fc1.bias"]
    _row_parallel_weights = ["out_proj.weight", "fc2.weight"]

    def load_weights(self, weights_path: str):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if "lm_head_weight" in name:
                continue
            if "qkv_proj" in name:
                shard_size = param.shape[0] // 3
                weights_to_concat = []
                for weight_name in ["q_proj", "k_proj", "v_proj"]:
                    weight = np.load(os.path.join(
                        weights_path, name.replace("qkv_proj", weight_name)))
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
        path = os.path.join(path, f"{model_name}-np")
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        lock_path = os.path.join(path, "file_lock")
        lock = filelock.FileLock(lock_path)

        with lock:
            test_weight_path = os.path.join(
                path, "model.decoder.embed_positions.weight")
            if os.path.exists(test_weight_path):
                return path

            folder = snapshot_download(model_name, allow_patterns="*.bin",
                                       cache_dir=os.path.join(path, "cache"))
            bin_files = glob.glob(os.path.join(folder, "*.bin"))

            for bin_file in tqdm(bin_files, desc="Convert format"):
                state = torch.load(bin_file, map_location="cpu")
                for name, param in tqdm(state.items(), leave=False):
                    if name.startswith("decoder."):
                        name = "model." + name
                    param_path = os.path.join(path, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())

            return path

    def initialize_dummy_weights(self) -> None:
        for param in self.state_dict().values():
            param.data.uniform_(-1e-3, 1e-3)
