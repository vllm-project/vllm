"""1D OPT model compatible with HuggingFace weights."""
import os
import glob
import shutil
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import OPTConfig
from transformers import PreTrainedModel
from huggingface_hub import snapshot_download

from cacheflow.models import InputMetadata
from cacheflow.models.attention import OPTCacheFlowAttention
from cacheflow.models.sample import Sampler
from cacheflow.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from cacheflow.parallel_utils.tensor_parallel import (VocabParallelEmbedding,
                                                      ColumnParallelLinear,
                                                      RowParallelLinear)

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
        self.scaling = self.head_dim**-0.5

        # TODO(woosuk): Fuse the three linear layers into one QKV linear layer.
        self.k_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=bias,
                                           gather_output=False,
                                           perform_initialization=False)
        self.v_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=bias,
                                           gather_output=False,
                                           perform_initialization=False)
        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=bias,
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
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
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

        assert config.word_embed_proj_dim == config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

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
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, Tuple[int, int]]:
        hidden_states = self.model(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["q_proj.weight", "k_proj.weight",
                                "v_proj.weight", "fc1.weight"]
    _column_parallel_biases = ["q_proj.bias", "k_proj.bias",
                                "v_proj.bias", "fc1.bias"]
    _row_parallel_weights = ["out_proj.weight", "fc2.weight"]

    def load_weights(self, weights_path: str):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            loaded_weight = torch.from_numpy(np.load(os.path.join(weights_path,
                                                                  name)))
            for p in (self._column_parallel_weights
                      + self._column_parallel_biases):
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

            assert param.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)

    @staticmethod
    def download_weights(model_name: str, path: str = "/tmp/transformers"):
        path = os.path.join(path, f"{model_name}-np")
        path = os.path.abspath(os.path.expanduser(path))
        test_weight_path = os.path.join(path,
                                        "model.decoder.embed_positions.weight")
        if os.path.exists(test_weight_path):
            return path

        folder = snapshot_download(model_name, allow_patterns="*.bin")
        bin_files = glob.glob(os.path.join(folder, "*.bin"))

        if "/" in model_name:
            model_name = model_name.split("/")[1].lower()
        os.makedirs(path, exist_ok=True)

        for bin_file in tqdm(bin_files, desc="Convert format"):
            state = torch.load(bin_file)
            for name, param in tqdm(state.items(), leave=False):
                param_path = os.path.join(path, name)
                with open(param_path, "wb") as f:
                    np.save(f, param.cpu().detach().numpy())

                # shared embedding
                if "model.decoder.embed_tokens.weight" in name:
                    shutil.copy(param_path, param_path.replace(
                        "model.decoder.embed_tokens.weight", "lm_head.weight"))

        return path