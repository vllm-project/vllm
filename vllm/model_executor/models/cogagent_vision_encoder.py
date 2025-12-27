# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from einops import repeat

from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention
from vllm.config import MultiModalConfig
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    get_act_and_mul_fn,
    get_act_fn,
)
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.cogagent import (
    EVACLIPVisionConfig,
    EVALargeVisionConfig,
)

logger = init_logger(__name__)

HAS_APEX = importlib.util.find_spec("apex")


def sharded_weight_loader(
    params_dict: dict[str, nn.Parameter],
    weights: Iterable[tuple[str, torch.Tensor]],
    weights_mapper: dict[str, tuple[str, str, str | None]],
) -> set[str]:
    loaded_params = set()

    for name, loaded_weight in weights:
        shard_id = None
        if weights_mapper is not None and name in weights_mapper:
            (hf_name, vlm_name, shard_id) = weights_mapper[name]
            name = name.replace(hf_name, vlm_name)

        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)

        if shard_id is not None:
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
        else:
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    return loaded_params


def get_layernorm(norm_type: str) -> nn.Module:
    layernorm = None
    if norm_type is None or norm_type.upper() == "NONE":
        layernorm = nn.Identity
    elif norm_type.lower() == "APEX" and HAS_APEX:
        from apex.normalization import (
            FusedLayerNorm,  # pyright: ignore[reportMissingImports]
        )

        layernorm = FusedLayerNorm
    elif norm_type.lower() == "APEX" and not HAS_APEX:
        layernorm = LayerNorm
        logger.info(
            "Attempted to use apex FusedLayerNorm while apex was not installed. "
            "Please install apex (https://github.com/NVIDIA/apex). "
            "Falling Back to torch Layernorm"
        )

    elif norm_type.upper() == "BASE":
        layernorm = LayerNorm
    else:
        raise NotImplementedError(f"Layer Norm type {norm_type} not implemented")

    return layernorm


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
        "invalid dimensions for broadcastable concatentation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


class EVAVisionRotaryEmbeddingFast(nn.Module):
    # TODO: check if implemented with vllm
    def __init__(
        self, dim, pt_seq_len, ft_seq_len=None, custom_freqs=None, theta=10000
    ):
        super().__init__()

        if custom_freqs is not None:
            freqs = custom_freqs
        else:
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        logger.info_once(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t):
        return t * self.freqs_cos + rotate_gptj(t) * self.freqs_sin


class EVAPatchEmbedding(nn.Module):
    def __init__(
        self, config: EVACLIPVisionConfig | EVALargeVisionConfig, prefix: str = ""
    ):
        super().__init__()
        self.prefix = prefix
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.proj = Conv2dLayer(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.has_embedding = isinstance(config, EVACLIPVisionConfig)
        if self.has_embedding:
            self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
            self.position_embedding = nn.Embedding(
                self.num_patches + 1, config.hidden_size
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # B, C, H, W -> B, L, D
        x = self.proj(images).flatten(2).transpose(1, 2)

        if self.has_embedding:
            cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)

            x = torch.cat((cls_token, x), dim=1)
            x += self.position_embedding.weight.unsqueeze(0)
        return x


class EVAMLP(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig | EVALargeVisionConfig,
        hidden_features: int | None = None,
        layernorm_type: str | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.prefix = prefix
        self.activation_fn = get_act_fn(config.hidden_act)
        layernorm = get_layernorm(layernorm_type)

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            hidden_features,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )

        self.fc2 = RowParallelLinear(
            hidden_features,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.ffn_ln = layernorm(
            hidden_features or config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, bias = self.fc1(x)
        x = self.activation_fn(x)

        x = self.ffn_ln(x)

        x, bias = self.fc2(x)
        return x


class EVASwiGLU(nn.Module):
    def __init__(
        self,
        config: EVALargeVisionConfig,
        hidden_features: int = None,
        layernorm_type: str = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.prefix = prefix
        hidden_features = hidden_features or config.hidden_size
        layernorm = get_layernorm(layernorm_type)
        self.act_and_mul = get_act_and_mul_fn(config.mlp_hidden_act)

        self.w1_2 = MergedColumnParallelLinear(
            config.hidden_size,
            [hidden_features] * 2,
            quant_config=quant_config,
            prefix=f"{prefix}.w1_2",
        )

        self.w3 = RowParallelLinear(
            hidden_features,
            config.hidden_size,
            prefix=f"{prefix}.w3",
            quant_config=quant_config,
        )

        self.ffn_ln = layernorm(hidden_features, eps=config.layer_norm_eps)

    def forward(self, x):
        x, bias = self.w1_2(x)
        hidden = self.act_and_mul(x)

        x = self.ffn_ln(hidden)

        x, bias = self.w3(x)

        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        weights_mapper = {
            "w1.weight": ("w1", "w1_2", 0),
            "w2.weight": ("w2", "w1_2", 1),
            "w1.bias": ("w1", "w1_2", 0),
            "w2.bias": ("w2", "w1_2", 1),
        }

        loaded_params = sharded_weight_loader(
            params_dict=params_dict, weights=weights, weights_mapper=weights_mapper
        )

        return loaded_params


class EVAGLU(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """
        Credit to glm4_vision_encoder
        """
        super().__init__()
        self.linear_proj = ReplicatedLinear(
            config.hidden_size,
            config.outer_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_proj",
        )

        self.norm1 = LayerNorm(
            config.outer_hidden_size,
            eps=1e-5,
        )

        self.act1 = nn.GELU()
        self.act2 = SiluAndMul()

        self.merged_proj = MergedColumnParallelLinear(
            config.outer_hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.merged_proj",
        )

        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.outer_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
        )

    def forward(self, x):
        x, _ = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x, _ = self.merged_proj(x)
        x = self.act2(x)
        x, _ = self.dense_4h_to_h(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        weights_mapper = {
            "glu_gate_proj.weight": ("glu_gate_proj", "merged_proj", 0),
            "dense_h_to_4h.weight": ("dense_h_to_4h", "merged_proj", 1),
        }

        loaded_params = sharded_weight_loader(
            params_dict=params_dict, weights=weights, weights_mapper=weights_mapper
        )

        return loaded_params


# Represents the attention class for both vision and cross vision
class EVAAttention(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig | EVALargeVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        attn_name: str = "query_key_value",
        linear_name: str = "proj",
        layernorm_type: str | None = None,
        prefix: str = "",
    ):
        super().__init__()

        layernorm = get_layernorm(layernorm_type)

        self.prefix = prefix
        if quant_config is not None:
            quant_name = quant_config.get_name()
            assert quant_name == "bitsandbytes", (
                "Only bitsandbytes quantization currently tested"
            )

        self.attn_name = attn_name
        self.linear_name = linear_name
        self.attn_bias = config.attn_bias
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_width

        self.split_qkv = config.split_qkv

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.local_num_heads = divide(config.num_heads, self.tp_size)

        self.use_rope = config.use_rope
        self.scale = config.qk_scale

        query_key_value = QKVParallelLinear(
            config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_heads,
            bias=self.attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.{attn_name}",
        )

        if self.use_rope:
            half_head_dim = config.head_width // 2
            hw_seq_len = config.image_size // config.patch_size
            self.rope = EVAVisionRotaryEmbeddingFast(
                dim=half_head_dim,  # should be 32
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=hw_seq_len if config.intp_freq else None,
            )

        self.inner_attn_ln = layernorm(config.hidden_size, eps=config.layer_norm_eps)

        proj = RowParallelLinear(
            config.hidden_size,
            self.dim,
            quant_config=quant_config,
            prefix=f"{prefix}.{linear_name}",
        )

        self.attn = MMEncoderAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            scale=self.scale,
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

        # Register like this to allow different naming configs
        self.register_module(self.attn_name, query_key_value)
        self.register_module(self.linear_name, proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv, bias = getattr(self, self.attn_name)(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        if self.use_rope:
            q = tensor_model_parallel_all_gather(q, dim=-1)
            k = tensor_model_parallel_all_gather(k, dim=-1)

            # B, N, HD -> B, num_heads, N, C
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)
            q = q.permute(0, 2, 1, 3)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)
            k = k.permute(0, 2, 1, 3)

            if self.tp_size > 1:
                q = q.reshape(B, N, -1)  # B, N, C
                q = split_tensor_along_last_dim(q, self.tp_size)
                q = q[self.tp_rank]

                k = k.reshape(B, N, -1)
                k = split_tensor_along_last_dim(k, self.tp_size)
                k = k[self.tp_rank]

        x = self.attn(q, k, v)
        x = self.inner_attn_ln(x)
        x, bias = getattr(self, self.linear_name)(x)

        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        weights_mapper = None
        if self.split_qkv:
            weights_mapper = {
                "q_proj.weight": ("q_proj", self.attn_name, "q"),
                "k_proj.weight": ("k_proj", self.attn_name, "k"),
                "v_proj.weight": ("v_proj", self.attn_name, "v"),
                "q_bias": ("q_bias", f"{self.attn_name}.bias", "q"),
                "v_bias": ("v_bias", f"{self.attn_name}.bias", "v"),
            }

        loaded_params = sharded_weight_loader(
            params_dict=params_dict, weights=weights, weights_mapper=weights_mapper
        )
        if self.split_qkv:
            # Assumes Quantization is not applied to bias.
            # QKVLinear initialized with empty(). Not garunteed zero, so zero out.
            shard_offset = self.num_heads * self.head_dim
            shard_size = self.num_heads * self.head_dim
            param = params_dict[f"{self.attn_name}.bias"].data
            param[shard_offset : (shard_offset + shard_size)].zero_()

        return loaded_params


class EVALargeBlock(nn.Module):
    def __init__(
        self,
        config: EVALargeVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        dim = config.hidden_size
        self.init_values = config.ls_init_value
        layernorm = get_layernorm(config.layernorm_type)

        self.norm1 = layernorm(dim, eps=config.layer_norm_eps)
        self.attn = EVAAttention(
            config=config,
            layernorm_type=config.layernorm_type,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            attn_name="qkv_proj",
            linear_name="proj",
            prefix=f"{prefix}.attn",
        )

        self.norm2 = layernorm(dim, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(dim * config.mlp_ratio)

        if config.naiveswiglu:
            self.mlp = EVASwiGLU(
                config,
                hidden_features=mlp_hidden_dim,
                quant_config=quant_config,
                layernorm_type=config.layernorm_type,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = EVAMLP(
                config,
                hidden_features=mlp_hidden_dim,
                quant_config=quant_config,
                layernorm_type=config.layernorm_type,
                prefix=f"{prefix}.mlp",
            )

        self.gamma_1 = None
        self.gamma_2 = None
        if self.init_values is not None:
            self.gamma_1 = nn.Parameter(
                self.init_values * torch.ones(dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                self.init_values * torch.ones(dim), requires_grad=True
            )

        self.postnorm = config.postnorm

    def forward(self, x):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.norm1(self.attn(x))
                x = x + self.norm2(self.mlp(x))
            else:
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
        else:
            if self.postnorm:
                x = x + (self.gamma_1 * self.norm1(self.attn(x)))
                x = x + (self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + (self.gamma_1 * self.attn(self.norm1(x)))
                x = x + (self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EVATransformerLayer(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix

        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = EVAAttention(
            config,
            layernorm_type=None,
            multimodal_config=multimodal_config,
            quant_config=quant_config,
            attn_name="query_key_value",
            linear_name="dense",
            prefix=f"{prefix}.attention",
        )
        self.mlp = EVAMLP(
            config,
            hidden_features=config.mlp_intermediate_size,
            layernorm_type=None,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))

        mlp_input = attention_input + attention_output
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class EVALargeVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        config: EVALargeVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.prefix = prefix
        self.config = config
        self.quant_config = quant_config

        layernorm = get_layernorm(config.layernorm_type)

        embed_dim = config.hidden_size
        self.rel_pos_bias = None

        self.patch_embed = EVAPatchEmbedding(
            config=config, prefix=f"{prefix}.patch_embed"
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = None
        if config.use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                EVALargeBlock(
                    config,
                    multimodal_config=multimodal_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(config.layers)
            ]
        )

        if config.use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = layernorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.norm = layernorm(embed_dim, eps=config.layer_norm_eps)
            self.fc_norm = None

        self.head = (
            nn.Linear(embed_dim, config.final_embed_dim)
            if config.final_embed_dim > 0
            else nn.Identity()
        )

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_classifier(self):
        return self.head

    def forward_features(self, x: torch.FloatTensor, return_all_features=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                continue
            x = blk(x)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        x = self.forward_features(x, return_all_features=return_all_features)

        if return_all_features:
            return x

        return self.head(x)


class EVATransformer(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EVATransformerLayer(
                    config,
                    multimodal_config=multimodal_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.layers)
            ]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class Eva2LargeEncoder(nn.Module):
    def __init__(
        self,
        config: EVALargeVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        os.environ["delRoPE"] = (
            "1"  # to avoid error in rope params when changing image size
        )
        self.model = EVALargeVisionTransformer(
            self.config,
            multimodal_config=multimodal_config,
            quant_config=quant_config,
            prefix=f"{prefix}.model",
        )

    def forward(self, images: torch.Tensor):
        encode = self.model(images, return_all_features=True)[:, 1:, :]
        return encode


class EVA2CLIPModel(nn.Module):
    def __init__(
        self,
        config: EVACLIPVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.patch_embedding = EVAPatchEmbedding(
            config, prefix=f"{prefix}.patch_embedding"
        )
        self.transformer = EVATransformer(
            config,
            multimodal_config=multimodal_config,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer",
        )
        self.linear_proj = EVAGLU(
            config, quant_config=quant_config, prefix=f"{prefix}.linear_proj"
        )

        self.boi = nn.Parameter(torch.zeros(1, 1, config.outer_hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.outer_hidden_size))

        pos_embed = torch.zeros(
            (config.image_size // config.patch_size) ** 2, config.hidden_size
        )
        self.pos_embed = nn.Parameter(pos_embed)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # B, C, H, W -> B, L, D
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x + self.pos_embed.unsqueeze(0))
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x


class CrossVisionModel(nn.Module):
    def __init__(
        self,
        config: EVALargeVisionConfig,
        multimodal_config: MultiModalConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_tokens: int = (config.image_size // config.patch_size) ** 2

        self.vit = Eva2LargeEncoder(
            config=config,
            multimodal_config=multimodal_config,
            quant_config=quant_config,
            prefix=f"{prefix}.vit",
        )
        self.pos_embed = nn.Parameter(torch.zeros(self.num_tokens, config.hidden_size))

    def forward(self, images):
        enc = self.vit(images)
        return enc + self.pos_embed.unsqueeze(0)
