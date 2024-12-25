# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
from functools import cached_property, partial
from typing import (Any, Iterable, List, Literal, Mapping, Optional, Set,
                    Tuple, Type, TypedDict, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import (Qwen2VLImageProcessor,
                                          Qwen2VLProcessor)
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig, Qwen2VLVisionConfig)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.inputs import InputContext
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.platforms import _Backend
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import uses_mrope
from vllm.utils import is_list_of

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, get_vit_attn_backend,
                    init_vllm_registered_model, maybe_prefix)

logger = init_logger(__name__)

# === Vision Inputs === #


class Qwen2VLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """Shape:
    `(num_patches, num_channels * patch_size * patch_size)`
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2VLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor
    """Shape: `(num_images, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2VLImageInputs = Union[Qwen2VLImagePixelInputs,
                           Qwen2VLImageEmbeddingInputs]


class Qwen2VLVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """


class Qwen2VLVideoEmbeddingInputs(TypedDict):
    type: Literal["video_embeds"]
    video_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all videos' features.
        Each tensor holds an video's features.
    - `torch.Tensor`: A tensor holding all videos' features
      (concatenation of all videos' feature tensors).
    
    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on 
        the number and resolution of the videos.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`
    This should be in `(grid_t, grid_h, grid_w)` format.
    """


Qwen2VLVideoInputs = Union[Qwen2VLVideoPixelInputs,
                           Qwen2VLVideoEmbeddingInputs]

# === Vision Encoder === #


class Qwen2VisionMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features,
                                        hidden_features,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1")
        self.act = act_layer()
        self.fc2 = RowParallelLinear(hidden_features,
                                     in_features,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1),
                         "... d two -> ... (d two)",
                         two=2)


def apply_rotary_emb_torch(x: torch.Tensor,
                           cos: torch.Tensor,
                           sin: torch.Tensor,
                           interleaved: bool = False) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos +
            rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor,
                                freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = apply_rotary_emb_torch(t_, cos, sin).type_as(t)
    return output


class Qwen2VisionAttention(nn.Module):

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        projection_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, world_size)

        self.qkv = ColumnParallelLinear(input_size=embed_dim,
                                        output_size=3 * projection_size,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.qkv")
        self.proj = RowParallelLinear(input_size=projection_size,
                                      output_size=embed_dim,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.proj")

        # Detect attention implementation.
        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
        if self.attn_backend not in {
                _Backend.FLASH_ATTN, _Backend.TORCH_SDPA, _Backend.XFORMERS
        }:
            raise RuntimeError(
                f"Qwen2-VL does not support {self.attn_backend} backend now.")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, head * 3 * head_dim] --> [s, b, head, 3 * head_dim]
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        x = x.view(*new_x_shape)

        # [s, b, head, 3 * head_dim] --> 3 [s, b, head, head_dim]
        q, k, v = dist_utils.split_tensor_along_last_dim(x, 3)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if self.attn_backend == _Backend.FLASH_ATTN:
            # from vllm_flash_attn.flash_attn_interface import (
            #   flash_attn_varlen_func)
            from flash_attn import flash_attn_varlen_func

            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            output = flash_attn_varlen_func(q,
                                            k,
                                            v,
                                            cu_seqlens_q=cu_seqlens,
                                            cu_seqlens_k=cu_seqlens,
                                            max_seqlen_q=max_seqlen,
                                            max_seqlen_k=max_seqlen,
                                            dropout_p=0,
                                            causal=False)

            context_layer = rearrange(output,
                                      "(b s) ... -> b s ...",
                                      b=batch_size)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            seq_length = q.size(1)
            q, k, v = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
            attention_mask = torch.zeros([1, seq_length, seq_length],
                                         device=q.device,
                                         dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                               cu_seqlens[i - 1]:cu_seqlens[i]] = True
            output = F.scaled_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    attention_mask,
                                                    dropout_p=0.0)
            context_layer = rearrange(output, "b h s d -> b s h d ")
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops
            from xformers.ops.fmha.attn_bias import BlockDiagonalMask

            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                       kv_seqlen=None)

            context_layer = xops.memory_efficient_attention_forward(
                q, k, v, attn_bias=attn_bias, p=0, scale=None)
        context_layer = rearrange(context_layer,
                                  "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen2VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Type[nn.Module] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = Qwen2VisionAttention(embed_dim=dim,
                                         num_heads=num_heads,
                                         projection_size=dim,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.attn")
        self.mlp = Qwen2VisionMLP(dim,
                                  mlp_hidden_dim,
                                  act_layer=act_layer,
                                  quant_config=quant_config,
                                  prefix=f"{prefix}.mlp")

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionPatchEmbed(nn.Module):

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        x = self.proj(x).view(L, self.embed_dim)
        return x


class Qwen2VisionPatchMerger(nn.Module):

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Type[nn.Module] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(self.hidden_size,
                                 self.hidden_size,
                                 bias=True,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.mlp.0"),
            nn.GELU(),
            RowParallelLinear(self.hidden_size,
                              d_model,
                              bias=True,
                              quant_config=quant_config,
                              prefix=f"{prefix}.mlp.2"),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta**(torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        in_chans: int = vision_config.in_chans
        hidden_size: int = vision_config.hidden_size
        embed_dim: int = vision_config.embed_dim
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        mlp_ratio: float = vision_config.mlp_ratio

        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen2VisionBlock(dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             quant_config=quant_config,
                             prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(depth)
        ])
        self.merger = Qwen2VisionPatchMerger(
            d_model=hidden_size,
            context_dim=embed_dim,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        # adapter
        x = self.merger(x)
        return x

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith("qkv.weight"):
                    visual_num_heads = self.num_heads
                    visual_embed_dim = self.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads,
                                                       head_size,
                                                       visual_embed_dim)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1, visual_embed_dim)
                elif name.endswith("qkv.bias"):
                    visual_num_heads = self.num_heads
                    visual_embed_dim = self.embed_dim
                    head_size = visual_embed_dim // visual_num_heads
                    loaded_weight = loaded_weight.view(3, visual_num_heads,
                                                       head_size)
                    loaded_weight = loaded_weight.transpose(0, 1)
                    loaded_weight = loaded_weight.reshape(-1)

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# === Vision input helpers === #


def _get_vision_info(
    vision_config: Qwen2VLVisionConfig,
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    do_resize: bool = True,
    data_type_key: str = "image",
    mm_count: int = 1,
):
    """Get information (resized height / width and number of vision tokens)
    of input image / video frame."""
    patch_size = vision_config.patch_size
    merge_size = vision_config.spatial_merge_size
    temporal_patch_size = vision_config.temporal_patch_size

    if do_resize:
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        resized_height, resized_width = height, width

    if data_type_key == "image":
        grid_t = mm_count
    else:
        assert data_type_key == "video"
        grid_t = max(mm_count // temporal_patch_size, 1)

    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    vision_tokens = grid_t * grid_h * grid_w
    llm_num_vision_tokens = vision_tokens // (merge_size**2)

    return resized_height, resized_width, llm_num_vision_tokens


def _get_image_processor(hf_processor: Qwen2VLProcessor):
    image_processor = hf_processor.image_processor  # type: ignore
    assert isinstance(image_processor, Qwen2VLImageProcessor)
    return image_processor


def get_max_qwen2_vl_mm_tokens(ctx: InputContext,
                               data_type_key: str,
                               *,
                               min_pixels: Optional[int] = None,
                               max_pixels: Optional[int] = None) -> int:
    hf_config = ctx.get_hf_config(Qwen2VLConfig)
    vision_config = hf_config.vision_config

    hf_processor = ctx.get_hf_processor(Qwen2VLProcessor)
    image_processor = _get_image_processor(hf_processor)

    _, _, max_llm_image_tokens = _get_vision_info(
        vision_config,
        height=9999999,
        width=9999999,
        min_pixels=min_pixels or image_processor.min_pixels,
        max_pixels=max_pixels or image_processor.max_pixels,
        data_type_key=data_type_key,
    )
    return max_llm_image_tokens


get_max_qwen2_vl_image_tokens = partial(get_max_qwen2_vl_mm_tokens,
                                        data_type_key="image")
get_max_qwen2_vl_video_tokens = partial(get_max_qwen2_vl_mm_tokens,
                                        data_type_key="video")


class Qwen2VLMultiModalDataItems(MultiModalDataItems):

    @staticmethod
    def from_dict(data: MultiModalDataDict) -> "MultiModalDataItems":
        """
        Normalize :class:`MultiModalDataDict` to :class:`MultiModalDataItems`.
        """
        multi_data = Qwen2VLMultiModalDataItems()

        for k, v in data.items():
            # TODO: Make a separate modality for embedding inputs
            # to avoid confusion
            # yapf: disable
            if k == "video":
                # Special case since even a single item can be a list
                multi_data[k] = (  # type: ignore[index]
                    v if (isinstance(v, (dict, torch.Tensor))  # type: ignore[assignment]
                          or is_list_of(v, list)) else [v]
                )
            elif k in ("image", "audio"):
                multi_data[k] = (  # type: ignore[index]
                    v if isinstance(v, (dict, torch.Tensor, list)) else [v]
                )
            else:
                multi_data[k] = v if isinstance(v, list) else [v]  # type: ignore[index]
            # yapf: enable

        return multi_data

    def get_item_counts(self) -> Mapping[str, int]:
        return {
            m: (
                len(items[f"{m}_grid_thw"])  # type: ignore
                if isinstance(items, dict) else len(items))
            for m, items in self.items()
        }


class Qwen2VLMultiModalProcessor(BaseMultiModalProcessor):

    def _get_mm_items(
        self,
        mm_data: MultiModalDataDict,
    ) -> MultiModalDataItems:
        return Qwen2VLMultiModalDataItems.from_dict(mm_data)

    def _get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Qwen2VLProcessor:
        hf_processor = self.ctx.get_hf_processor(Qwen2VLProcessor)
        image_processor = _get_image_processor(hf_processor)

        if min_pixels:
            image_processor.min_pixels = min_pixels
        if max_pixels:
            image_processor.max_pixels = max_pixels
        if max_pixels or min_pixels:
            image_processor.size = {
                "min_pixels": image_processor.min_pixels,
                "max_pixels": image_processor.max_pixels,
            }

        return hf_processor

    def _get_processor_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processor_data = dict[str, Any]()
        passthrough_data = dict[str, Any]()

        for k, v in mm_items.items():
            # TODO: Make a separate modality for embedding inputs
            # to avoid confusion
            if k in ("image", "video", "audio"):
                if isinstance(v, dict):
                    # Pass through embedding inputs (dict)
                    passthrough_data.update(v)
                elif isinstance(v, torch.Tensor) and v.ndim == 3:
                    # Pass through embedding inputs (single)
                    passthrough_data[f"{k}_embeds"] = [v]
                elif (is_list_of(v, torch.Tensor) and len(v) > 0
                      and v[0].ndim == 2):
                    # Pass through embedding inputs (multi)
                    passthrough_data[f"{k}_embeds"] = v
                else:
                    # Map keys to plural form, e.g.: image -> images
                    processor_data[f"{k}s"] = v
            else:
                processor_data[k] = v

        return processor_data, passthrough_data

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        hf_processor = self._get_hf_processor()
        image_processor = _get_image_processor(hf_processor)

        # NOTE: Only Qwen2VLProcessor in transformers 4.47.0 has
        # image_token and video_token registered
        placeholder = {
            "image": hf_processor.image_token,
            "video": hf_processor.video_token,
        }
        merge_length = image_processor.merge_size**2

        def get_replacement_qwen2vl(item_idx: int, modality: str):
            grid_thw = hf_inputs[f"{modality}_grid_thw"][item_idx]
            num_tokens = grid_thw.prod() // merge_length
            return placeholder[modality] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=placeholder[modality],
                replacement=partial(get_replacement_qwen2vl,
                                    modality=modality),
            ) for modality in ("image", "video")
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts["image"]
        hf_processor = self._get_hf_processor()
        image_token: str = hf_processor.image_token
        image_processor = _get_image_processor(hf_processor)

        data = {}
        resized_height, resized_width = smart_resize(
            height=9999999,
            width=9999999,
            factor=image_processor.patch_size * image_processor.merge_size,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )

        dummy_image = Image.new("RGB", (resized_width, resized_height),
                                color=0)
        data["image"] = [dummy_image] * num_images

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=data,
            mm_processor_kwargs={},
        )


@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_qwen2_vl_image_tokens)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "video", get_max_qwen2_vl_video_tokens)
@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor)
class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    # TODO Support LoRA for the visual encoder in the future.
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2VLConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen2-VL currently does not support prefix caching"

        self.config = config
        self.multimodal_config = multimodal_config

        self.visual = Qwen2VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # GPTQ configs do not have a list of ignored modules, however AutoGPTQ
        # seems to avoid vision encoder sections for some models.
        # See: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
        if isinstance(quant_config, (GPTQConfig, GPTQMarlinConfig)):
            return None
        return quant_config

    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: Union[torch.Tensor,
                                                        List[torch.Tensor]],
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                                 f"Got ndim: {mm_input.ndim}")
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Qwen2VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2VLImagePixelInputs(type="pixel_values",
                                           pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(
                image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2VLImageEmbeddingInputs(type="image_embeds",
                                               image_embeds=image_embeds,
                                               image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input(
            self, **kwargs: object) -> Optional[Qwen2VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            return Qwen2VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(
                video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2VLVideoEmbeddingInputs(type="video_embeds",
                                               video_embeds=video_embeds,
                                               video_grid_thw=video_grid_thw)

    def _process_image_input(self,
                             image_input: Qwen2VLImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self.visual(pixel_values,
                                   grid_thw=image_input["image_grid_thw"])
        return image_embeds

    def _process_video_input(self,
                             video_input: Qwen2VLVideoInputs) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos,
                                   grid_thw=video_input["video_grid_thw"])
        return video_embeds

    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        mask = (input_ids == placeholder_token_id)
        inputs_embeds[mask, :] = multimodal_embeddings
        return inputs_embeds

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[List[Tuple[NestedTensors, str]]]:

        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)
        if image_input is None and video_input is None:
            return None

        # We make a tuple of each embedding with its modality string. This is a
        # temporary workaround for models to handle mixed modalities when
        # get_multimodal_embeddings and get_input_embeddings are called
        # separately.
        # TODO(ywang96): Add support for mixed-modality inference for v1.
        multimodal_embeddings: List[Tuple[NestedTensors, str]] = []

        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            multimodal_embeddings.append((image_embeds, "image"))
        if video_input is not None:
            video_embeds = self._process_video_input(video_input)
            multimodal_embeddings.append((video_embeds, "video"))

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[List[Tuple[NestedTensors,
                                                   str]]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            for embeddings, modality in multimodal_embeddings:
                if modality == "image":
                    inputs_embeds = self._merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        embeddings,
                        placeholder_token_id=self.config.image_token_id,
                    )
                if modality == "video":
                    inputs_embeds = self._merge_multimodal_embeddings(
                        input_ids,
                        inputs_embeds,
                        embeddings,
                        placeholder_token_id=self.config.video_token_id,
                    )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Qwen2-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            pixel_values: Pixel values to be fed to a model.
                `None` if no images are passed.
            image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in LLM.
                `None` if no images are passed.
            pixel_values_videos: Pixel values of videos to be fed to a model.
                `None` if no videos are passed.
            video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in LLM.
                `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)

            # We need to check for usage of mrope here in case there is
            # multimodal data.
            # TODO (ywang96): move this to model runner in V1.
            if multimodal_embeddings is not None and uses_mrope(self.config):
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}")

            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
