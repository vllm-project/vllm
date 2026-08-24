# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import torch
from itertools import islice
import torch.nn.functional as F

from typing import Iterable, Optional

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm, LayerNorm
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import make_layers, maybe_prefix

from vllm.transformers_utils.configs.hstu import (
    HSTUConfig,
    HSTUModelConfig,
    RankingConfig,
)


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=0.02, lower=-2.0, upper=2.0):
    size = tensor.size()
    tmp = tensor.new_empty(size).normal_()
    tmp = tmp.clamp(min=lower, max=upper)
    tensor.copy_(tmp)
    tensor.mul_(std).add_(mean)


class HSTUEmbedding(torch.nn.Module):
    def __init__(
        self,
        hf_config: HSTUConfig,
    ):
        super().__init__()

        embedding_configs = hf_config.task_config.embedding_configs
        self._embedding_dim = hf_config.hidden_size
        self._dropout_rate = hf_config.hstu_config.dropout_ratio
        self._feature_sum_dim = 0

        self._sum_vocab_size = 0
        self._feature_cnt = 0
        embedding_table_name = []
        for config in embedding_configs:
            if config.table_name not in embedding_table_name:
                self._sum_vocab_size += config.vocab_size
                self._feature_cnt += 1
                dim = config.dim
                embedding_table_name.append(config.table_name)
            if config.table_name == "item":
                self._feature_sum_dim += config.dim
        self._embedding_layer = torch.nn.Embedding(
            num_embeddings=self._sum_vocab_size,
            embedding_dim=dim,
        )

        # Multiple input feature embeddings are concatenated and
        # projected through a linear layer.
        self._emb_mlp = torch.nn.Linear(
            self._feature_sum_dim,
            self._embedding_dim
        )

        self._pos_emb = torch.nn.Embedding(
            num_embeddings=self._sum_vocab_size,
            embedding_dim=dim,
        )
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)

    def to_empty(self):

        @torch.no_grad()
        def init_embedding_weights(m):
            if isinstance(m, torch.nn.Embedding):
                truncated_normal_(m.weight, mean=0.0, std=0.02)

        self.apply(init_embedding_weights)

        torch.nn.init.normal_(self._emb_mlp.weight, mean=0.0, std=0.02)

    def process_embs(self, x, mlp, feature_cnt):
        B, D = x.shape
        assert B % feature_cnt == 0, \
            f"T must be divisible by {feature_cnt}, got {B}"

        N = B // feature_cnt
        x = x.view(N, feature_cnt, D)

        features = x[:, : feature_cnt - 1, :]  # (N, feature_cnt - 1, D)
        action = x[:, feature_cnt - 1:, :]  # (N, 1, D)

        features = features.view(
            N, (feature_cnt - 1) * D
        )  # (N, (feature_cnt - 1) * D))
        features = mlp(features)  # (N, D)
        features = features.unsqueeze(-2)  # (N, 1, D)

        result = torch.stack(
            [features, action],
            dim=1
        ).reshape(action.shape[0] * 2, D)

        return result

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        embs = self._embedding_layer(input_ids)
        # _emb_mlp is always created in __init__; removed hasattr check
        # (compile-unfriendly Python bool branch).
        embs = self.process_embs(embs, self._emb_mlp, self._feature_cnt)
        embs = embs * (self._embedding_dim**0.5)
        embs = embs + self._pos_emb(positions)
        embs = self._emb_dropout(embs)

        # Always compute mask — eliminates is_prefill branch.
        # During decode, action tokens are all non-zero, so mask is
        # all-ones and embs * mask == embs (semantically equivalent).
        action = input_ids[1::2]
        mask = (action != 0).unsqueeze(1).expand(-1, 2).reshape(-1, 1)
        embs = embs * mask

        return embs


class MLP(torch.nn.Module):  # type: ignore
    def __init__(
        self,
        in_size: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} not supported")

        self.feed_forward = torch.nn.Linear(
            in_features=in_size,
            out_features=in_size
        )
        self.out_layer = torch.nn.Linear(in_features=in_size, out_features=1)
        self.layer_norm_1 = torch.nn.LayerNorm([in_size], eps=1e-7)
        self.layer_norm_2 = torch.nn.LayerNorm([in_size], eps=1e-7)
        self.act = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        x = torch.sigmoid(x)
        return x


class HSTUFFNSwiglu(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, dtype):
        super().__init__()
        self.w13 = torch.nn.Linear(dim, 2* hidden_dim, bias=False, dtype=dtype)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)
        self.act_fn = SiluAndMul()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.w13(x)
        x = self.act_fn(x)
        x = self.w2(x)
        return self.dropout(x)


class HSTUAttention(torch.nn.Module):
    def __init__(
        self,
        config: HSTUModelConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self._layer_name = prefix

        self.attn = Attention(
            num_heads=config.num_heads,
            head_size=config.head_dim,
            scale=1.0 / math.sqrt(config.head_dim),
            num_kv_heads=config.num_heads,
            prefix=self._layer_name,
        )

    @torch.inference_mode()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        attn_output = self.attn(query, key, value)
        attn_output = F.silu(attn_output)
        return attn_output


class HSTUInferLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config: HSTUModelConfig = vllm_config.model_config.hf_config.hstu_config
        self._embedding_dim: int = config.hidden_size
        self._linear_dim_per_head: int = config.head_dim
        self._num_heads: int = config.num_heads
        self._eps = config.layernorm_epsilon
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
        ]

        dtype = vllm_config.model_config.dtype

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            self._linear_dim_per_head * 4 * self._num_heads,
            bias=False,
            dtype=dtype
        )

        # input norm
        self._input_layernorm = LayerNorm(self._embedding_dim, self._eps)

        # prefix from make_layers: "model.layers.{idx}"
        # Attention prefix follows standard model naming: *.self_attn
        self.self_attn = HSTUAttention(
            config=config,
            prefix=f"{prefix}.self_attn",
        )

        # output norm
        self._output_layernorm = LayerNorm(self._embedding_dim, self._eps)

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=True,
            dtype=dtype
        )

        # ffn
        self.has_ffn = config.has_ffn
        if config.has_ffn:
            self.norm_ffn = RMSNorm(self._embedding_dim, self._eps)
            ffn_expand = config.ffn_expand
            self.feed_forward = HSTUFFNSwiglu(
                dim=self._embedding_dim,
                hidden_dim=self._embedding_dim * ffn_expand,
                dropout=config.dropout_ratio,
                dtype=dtype,
            )

    @torch.inference_mode()
    def forward(
        self,
        layer_input: torch.Tensor,
    ) -> torch.Tensor:
        normed_input = self._input_layernorm(layer_input)
        mixed_uvqk = self._linear_uvqk(normed_input)
        user, value, query, key = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1
        )

        attn_output = self.self_attn(query, key, value)
        attn_output = attn_output.view(-1, self._embedding_dim)
        norm_output = user * self._output_layernorm(attn_output)
        layer_output = self._linear_proj(norm_output)

        if self.has_ffn:
            if self._residual:
                ffn_input, _ = self.norm_ffn(layer_output, layer_input)
            else:
                ffn_input = self.norm_ffn(layer_output)
            layer_output = self.feed_forward(ffn_input) + layer_output
        else:
            if self._residual:
                layer_output = layer_output + layer_input

        return layer_output


class HSTUModel(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config: HSTUConfig = vllm_config.model_config.hf_config
        hstu_config: HSTUModelConfig = hf_config.hstu_config
        task_config: RankingConfig = hf_config.task_config

        self.vllm_config = vllm_config
        self._embedding_dim = hstu_config.hidden_size

        # ── Embedding ──
        self._embedding_collection = HSTUEmbedding(hf_config)
        self._embedding_collection.to_empty()

        # ── Transformer layers (make_layers for PP / offloading support) ──
        self.start_layer, self.end_layer, self.layers = make_layers(
            hstu_config.num_layers,
            lambda prefix: HSTUInferLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # ── Dense head ──
        self._dense_module = MLP(
            self._embedding_dim,
            task_config.prediction_head_act_type,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── 1. Embedding lookup ──
        embeddings = self._embedding_collection(input_ids, positions)

        # ── 2. Layer loop ──
        hidden_states = embeddings
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        # ── 3. Dense head — always execute ──
        dense_output = self._dense_module(hidden_states)

        return hidden_states, dense_output


class HSTUForCausalLM(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: HSTUConfig = vllm_config.model_config.hf_config
        self.model_config = vllm_config.model_config
        self.model = HSTUModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model._embedding_collection(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:

        with torch.inference_mode():
            # ── Model forward — compiled region ──
            hidden_states, dense_output = self.model(
                input_ids,
                positions=positions,
            )
        return dense_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return hidden_states


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # use random model
        if self.config.use_random_model:
            return None
        params_dict = dict(self.named_parameters())
        loaded_weights = set()
        model_module_mapping = self.config.model_module_mapping
        embedding_idx = 0
        for name, loaded_weight in weights:
            mapped_name = model_module_mapping.get(name) \
                if model_module_mapping else name
            if mapped_name == "":
                continue
            if mapped_name not in params_dict and mapped_name.startswith("layers."):
                mapped_name = "model." + mapped_name
            if mapped_name in params_dict:
                if "_linear_uvqk" in mapped_name and (
                    tuple(reversed(params_dict[mapped_name].shape))
                    == loaded_weight.shape
                ):
                    params_dict[mapped_name].data.copy_(loaded_weight.T)
                elif "_embedding_layer" in mapped_name:
                    embedding_cnt = loaded_weight.shape[0]
                    with torch.inference_mode():
                        params_dict[mapped_name].data[
                            embedding_idx: embedding_idx + embedding_cnt
                        ].copy_(loaded_weight)
                    embedding_idx += embedding_cnt
                else:
                    params_dict[mapped_name].data.copy_(loaded_weight)
                loaded_weights.add(mapped_name)
            elif "self_attn" not in mapped_name:
                print(f"model_pth: {name}, mapped_name: {mapped_name}, no match")

        return loaded_weights
