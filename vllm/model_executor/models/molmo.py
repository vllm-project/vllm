import math
import re
from array import array
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import (Any, Iterable, List, Mapping, Optional, Tuple, TypedDict,
                    Union)

import torch
from einops import rearrange
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.attention.selector import _Backend
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, InputContext,
                         token_inputs)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors,
                           SequenceData)
from vllm.transformers_utils.processor import get_processor

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (get_vit_attn_backend,
                    make_empty_intermediate_tensors_factory, make_layers)

# TODO: hard-coded for now. Consider making it configurable.
VIT_LAYERS = [-2, -9]
NUM_PREFIX_TOKENS = 1
ADDITIONAL_VOCAB_SIZE = 128


class MolmoImageInputs(TypedDict):
    images: torch.Tensor
    """Shape:
    `(batch_size, num_crops, num_patch, patch_dim)`
    """

    image_input_idx: torch.Tensor
    """Shape:
    `(batch_size, num_crops, num_patch)`
    """

    seq_len: torch.Tensor
    """Shape:
    `(batch_size, )`
    """

    image_masks: Optional[torch.Tensor]
    """Shape:
    `(batch_size, num_crops, num_patch)`
    """


@dataclass
class VisionBackboneConfig:
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 23
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "quick_gelu"
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5

    def __post_init__(self):
        self.image_default_input_size = tuple(
            self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class ViTMLP(nn.Module):
    """MLP used in Vision Transformer."""

    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.w1 = ColumnParallelLinear(
            config.image_emb_dim,
            config.image_mlp_dim,
            bias=True,
            quant_config=quant_config,
        )
        # Activation function.
        assert config.image_mlp_activations == "quick_gelu"
        self.act = QuickGELU()
        self.w2 = RowParallelLinear(
            config.image_mlp_dim,
            config.image_emb_dim,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head attention used in Vision Transformer."""

    def __init__(
        self,
        config: VisionBackboneConfig,
        use_bias: bool = True,
        nlayers: int = 1,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.hidden_size = config.image_emb_dim
        self.total_num_heads = config.image_num_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads

        self.total_num_kv_heads = config.image_num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.wq = ColumnParallelLinear(
            nlayers * self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wk = ColumnParallelLinear(
            nlayers * self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wv = ColumnParallelLinear(
            nlayers * self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=use_bias,
            quant_config=quant_config,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=use_bias,
            quant_config=quant_config,
        )

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend()
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"Molmo does not support {self.attn_backend} backend now.")

    def forward(self,
                inputs_q: torch.Tensor,
                inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, _ = self.wq(inputs_q)
        xk, _ = self.wk(inputs_k)
        xv, _ = self.wv(inputs_v)
        q_shape = xq.size()[:-1] + (self.num_heads, self.head_dim)
        kv_shape = xk.size()[:-1] + (self.num_kv_heads, self.head_dim)
        xq = xq.view(*q_shape)
        xk = xk.view(*kv_shape)
        xv = xv.view(*kv_shape)

        if self.attn_backend == _Backend.FLASH_ATTN:
            from flash_attn import flash_attn_func
            output = flash_attn_func(xq, xk, xv, dropout_p=0.0, causal=False)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            xq, xk, xv = (rearrange(x, "b s h d -> b h s d")
                          for x in (xq, xk, xv))
            output = F.scaled_dot_product_attention(xq, xk, xv)
            output = rearrange(output, "b h s d -> b s h d ")
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            output = xops.memory_efficient_attention_forward(xq, xk, xv, p=0)

        output = rearrange(output, "b s h d -> b s (h d)").contiguous()
        output, _ = self.wo(output)

        return output


class ResidualAttentionBlock(nn.Module):
    """Residual attention block used in Vision Transformer."""

    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.attention = MultiHeadDotProductAttention(
            config, quant_config=quant_config)
        self.feed_forward = ViTMLP(config, quant_config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
        )
        self.ffn_norm = nn.LayerNorm(
            config.image_emb_dim,
            eps=config.image_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class BlockCollection(nn.Module):
    """Collection of residual attention blocks used in Vision Transformer."""

    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config, quant_config)
            for _ in range(config.image_num_layers)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


def _expand_token(token: torch.Tensor, batch_size: int) -> torch.Tensor:
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    """Vision Transformer used in Vision Backbone."""

    def __init__(
        self,
        config: VisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        scale = config.image_emb_dim**-0.5
        self.patch_num = config.image_num_patch
        self.class_embedding = nn.Parameter(
            torch.randn(config.image_emb_dim) * scale)
        self.num_prefix_tokens: int = NUM_PREFIX_TOKENS
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.image_emb_dim) * scale)
        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.image_emb_dim,
            bias=False,
        )
        self.pre_ln = nn.LayerNorm(config.image_emb_dim,
                                   eps=config.image_norm_eps)
        self.transformer = BlockCollection(config, quant_config)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])),
             int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1]))

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]],
                          dim=1).to(x.dtype)
        return x

    def forward(self,
                x: torch.Tensor,
                patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.patch_num
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat(
            [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x],
            dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class MolmoAttention(nn.Module):
    """Molmo's LLM attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads \
            or self.total_num_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
        )

        self.tp_rank: Optional[int] = None
        self.k_norm: Optional[nn.Module] = None
        self.q_norm: Optional[nn.Module] = None
        if config.attention_layer_norm:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim,
                                  eps=config.layer_norm_eps)
            self.q_norm = RMSNorm(config.hidden_size,
                                  eps=config.layer_norm_eps)

        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def _apply_qk_norm(self, q: torch.Tensor,
                       k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MolmoMLP(nn.Module):
    """Molmo's LLM mlp."""

    def __init__(
        self,
        config: PretrainedConfig,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 2

        # Feed-forward input projection.
        self.gate_up_proj = MergedColumnParallelLinear(
            input_dim or self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )

        # Activation function.
        self.act_fn = SiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MolmoDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        # Attention block.
        self.self_attn = MolmoAttention(config, cache_config, quant_config)

        # MLP block.
        self.mlp = MolmoMLP(config, quant_config=quant_config)

        # LayerNorm
        assert config.layer_norm_type == "rms"
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.layer_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MolmoDecoderNormAfterLayer(MolmoDecoderLayer):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = None
        return hidden_states, residual


class MolmoVisionBackbone(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        vision_config: VisionBackboneConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.vit_layers = VIT_LAYERS
        self.image_num_patch = vision_config.image_num_patch
        self.llm_patches_per_crop = (
            (self.image_num_patch[0] + 1) // 2,
            (self.image_num_patch[1] + 1) // 2,
        )
        self.image_vit = VisionTransformer(vision_config,
                                           quant_config=quant_config)
        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {
            0, 1
        }, "Only 0 or 1 prefix tokens are supported"
        self.image_pooling_2d = MultiHeadDotProductAttention(
            vision_config,
            nlayers=len(self.vit_layers),
            quant_config=quant_config)
        self.image_projector = MolmoMLP(
            config,
            input_dim=vision_config.image_emb_dim,
            quant_config=quant_config,
        )

        image_dim = vision_config.image_emb_dim * len(self.vit_layers)
        self.pad_embed = nn.Parameter(torch.zeros((2, image_dim)))

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        B, T, N, D = images.shape

        mask = ~torch.all(
            images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)

        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        if self.vit_layers is not None:
            features = []
            for layer in self.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)
        else:
            image_features = image_features[-1]

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        return image_features

    def forward(
        self, images: torch.Tensor, image_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim) # noqa: E501
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.encode_image(images)

        og_dtype = image_features.dtype
        assert image_masks is not None
        pad_embed = self.pad_embed[:, None, None, None, :]
        all_pad = image_masks == 0
        partial_pad = torch.logical_and(
            image_masks < 1,
            torch.logical_not(all_pad)).to(dtype=torch.float32)
        all_pad = all_pad.to(dtype=torch.float32)
        image_features = image_features + pad_embed[0] * torch.unsqueeze(
            all_pad, -1)
        image_features = image_features + pad_embed[1] * torch.unsqueeze(
            partial_pad, -1)

        image_features = image_features.to(og_dtype)

        image_features = image_features.reshape(
            (batch_size, num_image) + self.image_num_patch + (-1, ), )

        if self.image_num_patch[0] % 2 == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )

        # image pooling
        image_features = rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=2,
            dw=2,
        )

        query = image_features.mean(-2, keepdim=True)
        image_features = self.image_pooling_2d(query, image_features)

        h, w = self.llm_patches_per_crop
        image_features = image_features.view(batch_size, num_image, h * w, -1)

        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        return image_features


@support_torch_compile
class MolmoModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding_size = config.embedding_size or config.vocab_size
        self.embedding_size += ADDITIONAL_VOCAB_SIZE
        self.embed_tokens = VocabParallelEmbedding(
            self.embedding_size,
            config.hidden_size,
            quant_config=quant_config,
        )

        decoder_layer = MolmoDecoderNormAfterLayer if config.norm_after \
            else MolmoDecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer(config, cache_config, quant_config),
            prefix=f"{prefix}.layers",
        )

        assert config.layer_norm_type == "rms"
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Apply blocks one-by-one.
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


cached_get_processor = lru_cache(get_processor)


def get_num_patches(num_tiles: int, crop_patches: int, left_margin: int,
                    right_margin: int, pooling_size: int) -> int:
    crop_window_patches = crop_patches - (left_margin + right_margin)
    if num_tiles > 1:
        left_crop_window_patches = (crop_window_patches + left_margin +
                                    pooling_size -
                                    1) // pooling_size * pooling_size
        middle_crop_window_patches = (crop_window_patches + pooling_size -
                                      1) // pooling_size * pooling_size
        right_crop_window_patches = (crop_window_patches + right_margin +
                                     pooling_size -
                                     1) // pooling_size * pooling_size
        return left_crop_window_patches + (
            num_tiles -
            2) * middle_crop_window_patches + right_crop_window_patches
    else:
        single_crop_window_patches = (crop_patches + pooling_size -
                                      1) // pooling_size * pooling_size
        return single_crop_window_patches


def get_tokens(tiling_h: int, tiling_w: int, crop_patches: int,
               left_margin: int, right_margin: int, pooling_size: int) -> int:
    h = get_num_patches(tiling_h, crop_patches, left_margin, right_margin,
                        pooling_size)
    w = get_num_patches(tiling_w, crop_patches, left_margin, right_margin,
                        pooling_size)
    per_row = w // pooling_size + 1
    joint = per_row * (h // pooling_size) + 2
    image_token_length = (crop_patches + pooling_size - 1) // pooling_size
    resize = (image_token_length + 1) * image_token_length + 2
    return resize + joint


def get_max_tokens(max_crops: int, crop_patches: int, left_margin: int,
                   right_margin: int, pooling_size: int) -> int:
    tilings = []
    for i in range(1, max_crops + 1):
        for j in range(1, max_crops + 1):
            if i * j <= max_crops:
                tilings.append((i, j))
    tokens = [
        get_tokens(tilings[i][0], tilings[i][1], crop_patches, left_margin,
                   right_margin, pooling_size) for i in range(len(tilings))
    ]
    return max(tokens)


def get_max_molmo_image_tokens(ctx: InputContext) -> int:
    processor = cached_get_processor(ctx.model_config.model,
                                     trust_remote_code=True,
                                     revision=ctx.model_config.code_revision)
    image_processor = processor.image_processor
    max_llm_image_tokens = get_max_tokens(
        image_processor.max_crops,
        image_processor.base_image_input_size[0] //
        image_processor.image_patch_size,
        image_processor.overlap_margins[0],
        image_processor.overlap_margins[1],
        2,
    )
    return max_llm_image_tokens


# NOTE: preprocessing for the image data has been included in the
# 'input_processor_for_molmo' function
def image_input_mapper_for_molmo(
    ctx: InputContext,
    data: object,
):
    return MultiModalInputs(data)


def dummy_data_for_molmo(ctx: InputContext, seq_len: int,
                         mm_counts: Mapping[str, int]):
    processor = cached_get_processor(ctx.model_config.model,
                                     trust_remote_code=True,
                                     revision=ctx.model_config.code_revision)
    image_processor = processor.image_processor

    base_image_input_d = image_processor.image_patch_size
    left_margin, right_margin = image_processor.overlap_margins
    max_crops = image_processor.max_crops

    # Assume: prompt_token_ids always starts with bos_token_id followed image tokens # noqa: E501
    max_llm_image_tokens = get_max_molmo_image_tokens(ctx)
    if seq_len - max_llm_image_tokens - 1 < 0:
        raise RuntimeError(
            f"Molmo cannot process {max_crops} crops in a prompt, "
            "please increase max_model_len or reduce number of crops")

    # The vertical image has the maximum number of image tokens due to column tokens. # noqa: E501
    tiling = (max_crops, 1)
    total_margin_pixels = base_image_input_d * (right_margin + left_margin)
    crop_patches = image_processor.base_image_input_size[
        0] // base_image_input_d
    crop_window_patches = crop_patches - (right_margin + left_margin)
    crop_window_size = crop_window_patches * base_image_input_d

    h = crop_window_size * tiling[0] + total_margin_pixels
    w = crop_window_size * tiling[1] + total_margin_pixels

    dummy_image = Image.new("RGB", (w, h), color="red")

    out = processor.process("dummy prompt", dummy_image)

    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      out["input_ids"][:1 + max_llm_image_tokens])
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - max_llm_image_tokens - 1)
    dummy_seqdata = SequenceData(token_ids)
    dummy_imgdata = {
        "images": out["images"],
        "image_input_idx": out["image_input_idx"],
    }
    if "image_masks" in out:
        dummy_imgdata["image_masks"] = out["image_masks"]
    dummy_imgdata["seq_len"] = torch.tensor(seq_len, dtype=torch.long)
    return dummy_seqdata, {"image": dummy_imgdata}


def pad_images(
    max_total_crops: int,
    images: torch.Tensor,
    image_input_idx: torch.Tensor,
    image_masks: Optional[torch.Tensor] = None,
):
    n = max_total_crops - images.shape[0]
    images = F.pad(images, (0, 0, 0, 0, 0, n), value=-1)
    image_input_idx = F.pad(image_input_idx, (0, 0, 0, n), value=-1)
    if image_masks is not None:
        image_masks = F.pad(image_masks, (0, 0, 0, n), value=-1)
    return images, image_input_idx, image_masks


def input_processor_for_molmo(ctx: InputContext, inputs: DecoderOnlyInputs):
    prompt = inputs.get("prompt")
    multi_modal_data = inputs.get("multi_modal_data")
    image = None if multi_modal_data is None else multi_modal_data.get("image")

    processor = cached_get_processor(ctx.model_config.model,
                                     trust_remote_code=True,
                                     revision=ctx.model_config.code_revision)

    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code)

    # NOTE: message formatting for raw text prompt is only applied for
    # offline inference; for online inference, the prompt is always in
    # instruction format and tokenized.
    if prompt is not None and re.match(r"^User:[\s\S]*?(Assistant:)*$",
                                       prompt):
        out = processor.process(prompt, image, message_format="none")
    elif prompt is not None:
        out = processor.process(prompt, image)
    else:
        out = processor.process(None, image, tokens=inputs["prompt_token_ids"])

    image_processor = processor.image_processor
    max_total_crops = 1 + image_processor.max_crops
    if image is not None:
        images, image_input_idx, image_masks = pad_images(
            max_total_crops,
            out["images"],
            out["image_input_idx"],
            out.get("image_masks"),
        )
    else:
        base_image_input_size = image_processor.base_image_input_size
        image_patch_size = image_processor.image_patch_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        image_length_w = image_processor.image_token_length_w
        image_length_h = image_processor.image_token_length_h
        tokens_per_image = image_length_w * image_length_h
        images = torch.full(
            (max_total_crops, n_patches, n_pixels),
            -1,
            dtype=torch.float32,
        )
        image_input_idx = torch.full(
            (max_total_crops, tokens_per_image),
            -1,
            dtype=torch.int32,
        )
        if image_processor.image_padding_mask:
            image_masks = torch.full(
                (max_total_crops, n_patches),
                -1,
                dtype=torch.float32,
            )

    image_data = dict(
        images=images,
        image_input_idx=image_input_idx,
    )
    if image_masks is not None:
        image_data["image_masks"] = image_masks

    image_data["seq_len"] = torch.tensor(len(out["input_ids"]),
                                         dtype=torch.long)

    multi_modal_data = dict(image=image_data)

    prompt = inputs.get("prompt")
    if prompt is None:
        prompt = tokenizer.decode(out["input_ids"])

    return token_inputs(
        prompt_token_ids=out["input_ids"],
        prompt=prompt,
        multi_modal_data=multi_modal_data,
    )


@MULTIMODAL_REGISTRY.register_image_input_mapper(image_input_mapper_for_molmo)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_molmo_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_molmo)
@INPUT_REGISTRY.register_input_processor(input_processor_for_molmo)
class MolmoForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: Optional[MultiModalConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        vision_config = VisionBackboneConfig()
        self.vision_backbone = MolmoVisionBackbone(config, vision_config,
                                                   quant_config)
        self.model = MolmoModel(config, cache_config, quant_config)

        if self.config.weight_tying:
            self.lm_head = self.model.transformer.wte
        else:
            self.lm_head = ParallelLMHead(
                config.embedding_size or config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )

        self.logits_processor = LogitsProcessor(config.embedding_size
                                                or config.vocab_size)
        self.sampler = Sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[MolmoImageInputs]:
        images = kwargs.pop("images", None)
        image_masks = kwargs.pop("image_masks", None)
        if images is None:
            return None

        image_input_idx = kwargs.pop("image_input_idx", None)
        seq_len = kwargs.pop("seq_len", None)
        if image_input_idx is None:
            raise ValueError("image_input_idx is required for Molmo model.")
        if seq_len is None:
            raise ValueError("seq_len is required for Molmo model.")
        if not isinstance(seq_len, torch.Tensor):
            seq_len = torch.tensor(seq_len)

        return MolmoImageInputs(
            images=images,
            image_input_idx=image_input_idx,
            seq_len=seq_len,
            image_masks=image_masks,
        )

    def _process_image_input(
        self,
        image_input: MolmoImageInputs,
    ) -> torch.Tensor:

        image_features = self.vision_backbone(
            images=image_input["images"],
            image_masks=image_input["image_masks"],
        )

        return image_features

    def _merge_multimodal_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
        image_input_idx: torch.Tensor,
        seq_len: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, num_image, num_patch = image_features.shape[:3]
        assert image_input_idx.shape == (batch_size, num_image, num_patch)

        image_features = image_features.to(inputs_embeds.device)
        seq_len = seq_len.to(inputs_embeds.device)

        # insert the image feature into the embedding.
        image_features = image_features.view(batch_size, num_image * num_patch,
                                             -1)
        image_input_idx = image_input_idx.view(batch_size,
                                               num_image * num_patch)

        valid = image_input_idx >= 0
        image_features = image_features * valid[:, :, None].to(
            image_features.dtype)
        image_features = image_features.view(
            batch_size * num_image * num_patch, -1).contiguous()

        image_input_idx = image_input_idx * valid.to(image_input_idx.dtype)
        offset = torch.cat(
            [seq_len.new_zeros(
                (1)), seq_len.cumsum(dim=0)[:-1]], dim=0)[:, None]
        image_input_idx = image_input_idx + offset.to(image_input_idx.dtype)
        image_input_idx = image_input_idx.flatten()[:, None]
        mat = image_input_idx == torch.arange(
            seq_len.sum().item(), device=inputs_embeds.device)[None, :]
        mat = mat.to(image_features.dtype)

        inputs_embeds = inputs_embeds + torch.einsum('nd,nm->md',
                                                     image_features, mat)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        if intermediate_tensors is not None:
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)

            if image_input is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_features = self._process_image_input(image_input)

                inputs_embeds = self._merge_multimodal_embeddings(
                    inputs_embeds,
                    image_features,
                    image_input["image_input_idx"],
                    image_input["seq_len"],
                )
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        params_mapping = [
            ("model.transformer.ln_f.weight", "model.norm.weight"),
            ("attn_out", "self_attn.o_proj"),
            ("att_proj", "self_attn.qkv_proj"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
            ("attn_norm", "input_layernorm"),
            ("ff_norm", "post_attention_layernorm"),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        embedding_weight = dict()
        projector_weight = dict()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if "wte.embedding" in name:
                embedding_weight["embedding"] = loaded_weight
                continue

            if "wte.new_embedding" in name:
                embedding_weight["new_embedding"] = loaded_weight
                continue

            if "vision_backbone" in name:
                if name.startswith("model"):
                    name = name[len("model."):]
                if 'image_projector' in name:
                    if 'w1' in name:
                        projector_weight['gate_proj'] = loaded_weight
                    elif 'w3' in name:
                        projector_weight['up_proj'] = loaded_weight
                    elif 'w2' in name:
                        projector_weight['down_proj'] = loaded_weight
                    else:
                        raise ValueError(
                            f"Unexpected projector weight: {name}")
                    continue
            else:
                if "transformer.blocks" in name:
                    name = name.replace("transformer.blocks", "layers")

                if "ff_proj" in name:
                    name = name.replace("ff_proj", "mlp.gate_up_proj")
                    assert 'weight' in name
                    up_weight, gate_weight = loaded_weight.chunk(2, dim=0)
                    loaded_weight = torch.cat([gate_weight, up_weight], dim=0)

                elif "ff_out" in name:
                    if "layers" in name:
                        name = name.replace("ff_out", "mlp.down_proj")
                    else:
                        # lm head
                        name = name.replace("model.transformer.ff_out",
                                            "lm_head")

                else:
                    for (param_name, weight_name) in params_mapping:
                        if param_name in name:
                            name = name.replace(param_name, weight_name)
                            break

            try:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
            except KeyError:
                raise ValueError(f"Unexpected weight: {name}") from None

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        gate_up_proj_weight = torch.cat(
            [projector_weight["gate_proj"], projector_weight["up_proj"]],
            dim=0)
        name = "vision_backbone.image_projector.gate_up_proj.weight"
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, gate_up_proj_weight)

        down_proj_weight = projector_weight["down_proj"]
        name = "vision_backbone.image_projector.down_proj.weight"
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, down_proj_weight)

        embedding_weight = torch.cat(
            [embedding_weight["embedding"], embedding_weight["new_embedding"]],
            dim=0)
        name = "model.embed_tokens.weight"
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, embedding_weight)
