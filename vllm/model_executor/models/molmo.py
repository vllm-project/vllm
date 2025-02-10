# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from functools import partial
from typing import (Iterable, List, Mapping, Optional, Set, Tuple, TypedDict,
                    Union)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (BatchFeature, PretrainedConfig, ProcessorMixin,
                          TensorType)
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention import Attention, AttentionMetadata
from vllm.attention.layer import MultiHeadAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import (MulAndSilu, QuickGELU,
                                                   SiluAndMul)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptReplacementDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, merge_multimodal_embeddings)

# TODO: hard-coded for now. Consider making it configurable.
VIT_LAYERS = [-2, -9]
NUM_PREFIX_TOKENS = 1
ADDITIONAL_VOCAB_SIZE = 128
DEFAULT_IMAGE_PATCH_TOKEN_ID = 152066
DEFAULT_IM_COL_TOKEN_ID = 152067
DEFAULT_IM_START_TOKEN_ID = 152064
DEFAULT_IM_END_TOKEN_ID = 152065
POOLING_SIZE = 2


class MolmoImageInputs(TypedDict):
    images: NestedTensors
    """Shape: `(batch_size, num_crops, num_patch, patch_dim)`"""

    image_input_idx: NestedTensors
    """Shape: `(batch_size, num_crops, num_patch)`"""

    image_masks: Optional[NestedTensors]
    """Shape: `(batch_size, num_crops, num_patch)`"""

    seq_len: torch.Tensor
    """Shape: `(batch_size, )`"""

    image_start_end: torch.Tensor
    """
    Starting and ending index of placeholder tokens.

    Shape: `(batch_size, 2)`
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

        self.scale = self.head_dim**-0.5
        self.attn = MultiHeadAttention(self.num_heads,
                                       self.head_dim,
                                       self.scale,
                                       num_kv_heads=self.num_kv_heads)

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

        output = self.attn(xq, xk, xv)
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
                patch_num: Optional[int] = None) -> List[torch.Tensor]:
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
        prefix: str = "",
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
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

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


class LanuageModelMLP(nn.Module):
    """Molmo's LLM mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 input_dim: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 2

        self.gate_up_proj = MergedColumnParallelLinear(
            input_dim or self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        # Activation function.
        self.act_fn = MulAndSilu()
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


class ImageProjectorMLP(nn.Module):
    """Molmo's image_projector mlp."""

    def __init__(
        self,
        config: PretrainedConfig,
        input_dim: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 2

        self.merged_linear = MergedColumnParallelLinear(
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
        gate_up, _ = self.merged_linear(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MolmoDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Attention block.
        self.self_attn = MolmoAttention(config,
                                        cache_config,
                                        quant_config,
                                        prefix=f"{prefix}.self_attn")

        # MLP block.
        self.mlp = LanuageModelMLP(config, quant_config=quant_config)

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
            (self.image_num_patch[0] + 1) // POOLING_SIZE,
            (self.image_num_patch[1] + 1) // POOLING_SIZE,
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
        self.image_projector = ImageProjectorMLP(
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

        missing_w = self.image_num_patch[0] % 2
        if missing_w:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, missing_w, 0, missing_w, 0, 0, 0, 0),
            )

        # image pooling
        image_features = rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=POOLING_SIZE,
            dw=POOLING_SIZE,
        )

        query = image_features.mean(-2, keepdim=True)
        image_features = self.image_pooling_2d(query, image_features)

        h, w = self.llm_patches_per_crop
        image_features = image_features.view(batch_size, num_image, h * w, -1)

        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        return image_features

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_linear", "gate_proj", 0),
            ("merged_linear", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@support_torch_compile
class MolmoModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

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
            lambda prefix: decoder_layer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        assert config.layer_norm_type == "rms"
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


def _lowest_multiple(x: int, k: int) -> int:
    return (x // k) * k


def get_num_patches(
    num_tiles: int,
    *,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> int:
    if num_tiles == 1:
        return _lowest_multiple(crop_patches + pooling_size - 1, pooling_size)

    crop_window_patches = crop_patches - (left_margin + right_margin)

    left_num = _lowest_multiple(
        crop_window_patches + left_margin + pooling_size - 1,
        pooling_size,
    )
    middle_num = _lowest_multiple(
        crop_window_patches + pooling_size - 1,
        pooling_size,
    )
    right_num = _lowest_multiple(
        crop_window_patches + right_margin + pooling_size - 1,
        pooling_size,
    )

    return left_num + (num_tiles - 2) * middle_num + right_num


def get_patches_grid_size(
    *,
    tiling_h: int,
    tiling_w: int,
    crop_patches: int,
    left_margin: int,
    right_margin: int,
    pooling_size: int,
) -> tuple[int, int]:
    nrows = get_num_patches(
        tiling_h,
        crop_patches=crop_patches,
        left_margin=left_margin,
        right_margin=right_margin,
        pooling_size=pooling_size,
    )
    ncols = get_num_patches(
        tiling_w,
        crop_patches=crop_patches,
        left_margin=left_margin,
        right_margin=right_margin,
        pooling_size=pooling_size,
    )

    return nrows, ncols


def get_candidate_tilings(max_num: int) -> list[tuple[int, int]]:
    tilings = [(i, j) for i in range(1, max_num + 1)
               for j in range(1, max_num + 1) if i * j <= max_num]
    return sorted(tilings, key=lambda x: x[0] * x[1])


def select_tiling(
    *,
    height: int,
    width: int,
    patch_size: int,
    max_num_patches: int,
):
    tilings = get_candidate_tilings(max_num_patches)
    candidate_tilings = np.array(tilings, dtype=np.int32)
    candidate_resolutions = candidate_tilings * patch_size

    original_size = np.array([height, width], dtype=np.float32)
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = required_scale_d.min(axis=-1, keepdims=True)

    if (required_scale < 1).all():
        ix = required_scale.argmax()
    else:
        ix = np.where(required_scale < 1.0, 10e9, required_scale).argmin()

    return candidate_tilings[ix]


class MolmoProcessorWrapper:
    """
    Wraps :class:`MolmoProcessor` so that it can be called directly.

    The original definition can be found here:
    https://huggingface.co/allenai/Molmo-7B-D-0924/blob/main/preprocessing_molmo.py
    """

    def __init__(self, processor: ProcessorMixin):
        super().__init__()

        self.processor = processor

    @property
    def max_crops(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        max_crops = image_processor.max_crops
        assert isinstance(max_crops, int)

        return max_crops

    @property
    def base_image_input_size(self) -> tuple[int, int]:
        image_processor = self.processor.image_processor  # type: ignore

        base_image_input_size = image_processor.base_image_input_size
        if isinstance(base_image_input_size, int):
            return base_image_input_size, base_image_input_size

        return tuple(base_image_input_size)

    @property
    def image_patch_size(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_patch_size = image_processor.image_patch_size
        assert isinstance(image_patch_size, int)

        return image_patch_size

    @property
    def overlap_margins(self) -> tuple[int, int]:
        image_processor = self.processor.image_processor  # type: ignore

        left_margin, right_margin = image_processor.overlap_margins
        assert isinstance(left_margin, int)
        assert isinstance(right_margin, int)

        return left_margin, right_margin

    @property
    def image_token_length_w(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_token_length_w = image_processor.image_token_length_w
        assert isinstance(image_token_length_w, int)

        return image_token_length_w

    @property
    def image_token_length_h(self) -> int:
        image_processor = self.processor.image_processor  # type: ignore

        image_token_length_h = image_processor.image_token_length_h
        assert isinstance(image_token_length_h, int)

        return image_token_length_h

    @property
    def message_format(self) -> Optional[str]:
        return "role"

    @property
    def always_start_with_space(self) -> bool:
        return True

    @property
    def pooling_size(self) -> int:
        return POOLING_SIZE

    def select_tiling(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_crops = self.max_crops
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size

        total_margin_pixels = base_image_input_d * (right_margin + left_margin)
        crop_patches = base_image_input_size[0] // base_image_input_d
        crop_window_patches = crop_patches - (right_margin + left_margin)
        crop_window_size = crop_window_patches * base_image_input_d
        tiling_h, tiling_w = select_tiling(
            height=image_height - total_margin_pixels,
            width=image_width - total_margin_pixels,
            patch_size=crop_window_size,
            max_num_patches=max_crops,
        )

        return tiling_w, tiling_h

    def get_patches_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        left_margin, right_margin = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        base_image_input_d = self.image_patch_size
        pooling_size = self.pooling_size

        crop_patches = base_image_input_size[0] // base_image_input_d
        tiling_w, tiling_h = self.select_tiling(
            image_height=image_height,
            image_width=image_width,
        )

        nrows, ncols = get_patches_grid_size(
            tiling_h=tiling_h,
            tiling_w=tiling_w,
            crop_patches=crop_patches,
            left_margin=left_margin,
            right_margin=right_margin,
            pooling_size=pooling_size,
        )

        return ncols, nrows

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        outputs = self.processor.process(  # type: ignore
            text, images, **kwargs)

        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        input_ids = outputs.pop("input_ids")
        outputs["input_ids"] = input_ids.unsqueeze(0)

        tilings = [
            self.select_tiling(
                image_width=image.size[0],
                image_height=image.size[1],
            ) for image in images
        ]
        if tilings:
            outputs["num_patches"] = torch.tensor(tilings).prod(-1) + 1

        outputs["seq_len"] = torch.tensor(len(input_ids))

        image_ids = [
            DEFAULT_IMAGE_PATCH_TOKEN_ID,
            DEFAULT_IM_COL_TOKEN_ID,
            DEFAULT_IM_START_TOKEN_ID,
            DEFAULT_IM_END_TOKEN_ID,
        ]
        is_image_ids, inv_idxs, counts = torch.unique_consecutive(
            torch.isin(input_ids, torch.tensor(image_ids), assume_unique=True),
            return_inverse=True,
            return_counts=True,
        )
        idxs = inv_idxs.diff(prepend=torch.tensor([-1])).nonzero().squeeze(1)
        assert len(is_image_ids) == len(idxs) == len(counts)

        image_start_end_lst = list[tuple[int, int]]()
        for is_image_id, idx, count in zip(is_image_ids, idxs, counts):
            if is_image_id:
                assert input_ids[idx] in image_ids
                image_start_end_lst.append((idx, idx + count))

        image_start_end = torch.tensor(image_start_end_lst)
        assert len(image_start_end) <= 1, "Multi-image input not supported yet"
        outputs["image_start_end"] = image_start_end.squeeze(0)

        return BatchFeature(outputs, tensor_type=return_tensors)


class MolmoProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(self) -> MolmoProcessorWrapper:
        processor = self.ctx.get_hf_processor()
        return MolmoProcessorWrapper(processor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[MolmoProcessorWrapper],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        ncols, nrows = processor.get_patches_grid_size(
            image_width=image_width,
            image_height=image_height,
        )
        pooling_size = processor.pooling_size

        base_image_input_size = processor.base_image_input_size
        base_image_input_d = processor.image_patch_size

        crop_patches = base_image_input_size[0] // base_image_input_d

        per_row = ncols // pooling_size + 1
        joint = per_row * (nrows // pooling_size) + 2
        image_token_length = (crop_patches + pooling_size - 1) // pooling_size
        resize = (image_token_length + 1) * image_token_length + 2

        return resize + joint

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=None,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        tilings = get_candidate_tilings(processor.max_crops)
        base_h, base_w = processor.base_image_input_size

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in tilings:
            width, height = base_w * wr, base_h * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


class MolmoDummyInputsBuilder(BaseDummyInputsBuilder[MolmoProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="",
            mm_data=mm_data,
        )


class MolmoMultiModalProcessor(BaseMultiModalProcessor[MolmoProcessingInfo]):

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        processor = self.info.get_hf_processor()

        # Apply the chat template to the tokens
        tokens = processor.processor.get_tokens_input(  # type: ignore
            self.info.get_tokenizer().decode(prompt_tokens),
            message_format=processor.message_format,
            always_start_with_space=processor.always_start_with_space,
        )

        processed_data = self.info.ctx.call_hf_processor(
            processor,  # type: ignore
            dict(tokens=tokens),
        )
        prompt_ids, = processed_data.pop("input_ids").tolist()

        return prompt_ids

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        num_images = len(num_patches)

        return dict(
            images=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            image_masks=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            image_input_idx=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            seq_len=MultiModalFieldConfig.shared("image", num_images),
            image_start_end=MultiModalFieldConfig.shared("image", num_images),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        image_token_length_w = processor.image_token_length_w
        image_token_length_h = processor.image_token_length_h
        pooling_size = processor.pooling_size

        user_str = "User:"
        if processor.always_start_with_space:
            user_str = " " + user_str

        user_tokens = tokenizer.encode(user_str, add_special_tokens=False)

        img_patch_id = DEFAULT_IMAGE_PATCH_TOKEN_ID
        img_col_id = DEFAULT_IM_COL_TOKEN_ID
        img_start_id = DEFAULT_IM_START_TOKEN_ID
        img_end_id = DEFAULT_IM_END_TOKEN_ID

        extra_row = [img_patch_id] * image_token_length_w + [img_col_id]
        extra_joint = ([img_start_id] + extra_row * image_token_length_h +
                       [img_end_id])

        def get_replacement_molmo(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = processor.get_patches_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            joint_row = ([img_patch_id] * ((ncols + 1) // pooling_size) +
                         [img_col_id])
            joint = ([img_start_id] + joint_row *
                     ((nrows + 1) // pooling_size) + [img_end_id])

            image_tokens = extra_joint + joint

            return PromptReplacementDetails(
                full=image_tokens + user_tokens,
                features=image_tokens,
            )

        return [
            PromptReplacement(
                modality="image",
                target=user_str,
                replacement=get_replacement_molmo,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(MolmoMultiModalProcessor,
                                        info=MolmoProcessingInfo,
                                        dummy_inputs=MolmoDummyInputsBuilder)
class MolmoForCausalLM(nn.Module, SupportsMultiModal, SupportsPP,
                       SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # vision backbone mapping
            "image_projector.w1.": "image_projector.gate_proj.",
            "image_projector.w3.": "image_projector.up_proj.",
            "image_projector.w2.": "image_projector.down_proj.",
            # language backbone mapping
            "att_proj": "self_attn.qkv_proj",
            "attn_out": "self_attn.o_proj",
            "q_norm": "self_attn.q_norm",
            "k_norm": "self_attn.k_norm",
            "ff_proj": "mlp.gate_up_proj",
            "ff_out": "mlp.down_proj",
            "attn_norm": "input_layernorm",
            "ff_norm": "post_attention_layernorm",
        },
        orig_to_new_prefix={
            # vision backbone mapping
            "model.vision_backbone.": "vision_backbone.",
            # language backbone mapping
            "model.transformer.blocks.": "model.layers.",
            "model.transformer.ln_f.": "model.norm.",
            # lm_head is renamed to model.transformer.mlp.down_proj firstly,
            # we need to run a second renaming for it
            "model.transformer.mlp.down_proj.": "lm_head.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_up_proj"],  # language model
        "merged_linear": ["gate_proj", "up_proj"]  # image_projector
    }

    # LoRA specific attributes
    supported_lora_modules = [
        # language model
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",  # same name with image_projector
        # vision tower
        "wq",
        "wk",
        "wv",
        "wo",
        "w1",
        "w2",
        # image_projector
        "merged_linear",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        vision_config = VisionBackboneConfig()
        self.vision_backbone = MolmoVisionBackbone(config, vision_config,
                                                   quant_config)
        self.model = MolmoModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

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
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Optional[MolmoImageInputs]:
        images = kwargs.pop("images", None)
        image_masks = kwargs.pop("image_masks", None)
        image_start_end = kwargs.pop("image_start_end", None)
        if images is None:
            return None

        image_input_idx = kwargs.pop("image_input_idx", None)
        seq_len = kwargs.pop("seq_len", None)
        if image_input_idx is None:
            raise ValueError("image_input_idx is required for Molmo model.")
        if seq_len is None:
            raise ValueError("seq_len is required for Molmo model.")
        if not isinstance(images, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. "
                             f"Got type: {type(images)}")
        if not isinstance(image_input_idx, (torch.Tensor, list)):
            raise ValueError("Incorrect type of image_input_idx. "
                             f"Got type: {type(image_input_idx)}")
        if not (image_masks is None or isinstance(image_masks,
                                                  (torch.Tensor, list))):
            raise ValueError("Incorrect type of image_masks. "
                             f"Got type: {type(image_masks)}")
        if not isinstance(seq_len, torch.Tensor):
            raise ValueError("Incorrect type of seq_len. "
                             f"Got type: {type(seq_len)}")
        if not isinstance(image_start_end, torch.Tensor):
            raise ValueError("Incorrect type of image_start_end. "
                             f"Got type: {type(image_start_end)}")

        return MolmoImageInputs(
            images=images,
            image_input_idx=image_input_idx,
            image_masks=image_masks,
            seq_len=seq_len,
            image_start_end=image_start_end,
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

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        image_features = self._process_image_input(image_input)
        image_input_idx = image_input["image_input_idx"]
        seq_len = image_input["seq_len"]
        batch_size, num_image, num_patch = image_features.shape[:3]
        assert image_input_idx.shape == (batch_size, num_image, num_patch)

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
        offset = torch.cat([seq_len.new_zeros(1),
                            seq_len.cumsum(dim=0)[:-1]],
                           dim=0)[:, None]
        image_input_idx = image_input_idx + offset.to(image_input_idx.dtype)
        image_input_idx = image_input_idx.flatten()[:, None]
        mat = image_input_idx == torch.arange(
            seq_len.sum().item(), device=image_features.device)[None, :]
        mat = mat.to(image_features.dtype)

        # Note: In this original implementation from AI2, the final
        # vision_embeddings will be always be the same length
        # of input embeddings.
        vision_embeddings = torch.einsum('nd,nm->md', image_features, mat)

        # Split by the sizes of the input sequences. For each full embedding,
        # extract the actual vision embeddings to be merged.
        vision_embeddings = list(vision_embeddings.split(seq_len.tolist()))
        for i in range(len(vision_embeddings)):
            start, end = image_input['image_start_end'][i]
            vision_embeddings[i] = vision_embeddings[i][start:end]

        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, [
                    DEFAULT_IMAGE_PATCH_TOKEN_ID, DEFAULT_IM_COL_TOKEN_ID,
                    DEFAULT_IM_START_TOKEN_ID, DEFAULT_IM_END_TOKEN_ID
                ])
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> SamplerOutput:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   kv_caches,
                                   attn_metadata,
                                   intermediate_tensors,
                                   inputs_embeds=inputs_embeds)

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

        loader = AutoWeightsLoader(self)
        weights = _get_weights_with_merged_embedding(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector="vision_backbone.image_projector",
            tower_model="vision_backbone",
        )


def _get_weights_with_merged_embedding(
    weights: Iterable[Tuple[str, torch.Tensor]]
) -> Iterable[Tuple[str, torch.Tensor]]:
    embedding_weights = {}
    for name, weight in weights:
        if "wte.embedding" in name:
            embedding_weights["embedding"] = weight
        elif "wte.new_embedding" in name:
            embedding_weights["new_embedding"] = weight
        else:
            yield (name, weight)
    # this is compatible with most of quantization,
    # because they won't quantize embed_tokens
    embedding_weights = torch.cat(
        [embedding_weights["embedding"], embedding_weights["new_embedding"]],
        dim=0,
    )
    yield ("model.embed_tokens.weight", embedding_weights)
