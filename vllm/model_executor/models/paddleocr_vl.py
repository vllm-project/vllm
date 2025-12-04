# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BatchFeature, PretrainedConfig
from transformers.activations import GELUActivation
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
)
from transformers.utils import torch_int

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import (
    maybe_get_vit_flash_attn_backend,
)
from vllm.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
)
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    dispatch_rotary_emb_function,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .ernie45 import Ernie4_5ForCausalLM
from .interfaces import MultiModalEmbeddings, SupportsMRoPE, SupportsMultiModal
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    maybe_prefix,
)
from .vision import get_vit_attn_backend


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """

    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rotary_emb_function = dispatch_rotary_emb_function(default=apply_rotary_emb_torch)
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = rotary_emb_function(t_, cos, sin).type_as(t)
    return output


class PaddleOCRVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor,
    ) -> int:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        resized_height, resized_width = smart_resize(
            height=image_height,
            width=image_width,
            factor=patch_size * merge_size,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )
        preprocessed_size = ImageSize(width=resized_width, height=resized_height)

        grid_t = 1
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_image_tokens = num_patches // (merge_size**2)

        return num_image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        # See `smart_resize` for the calculation of the image size.
        merge_size = hf_config.vision_config.spatial_merge_size
        patch_size = hf_config.vision_config.patch_size
        factor = merge_size * patch_size
        max_num_tokens = self.get_image_processor().max_pixels // (factor**2)
        # Find factors of max_num_tokens close to its square root
        # to create a dummy image with a reasonable aspect ratio.
        h_patches = int(math.sqrt(max_num_tokens))
        max_num_tokens -= max_num_tokens % h_patches
        w_patches = max_num_tokens // h_patches
        return ImageSize(height=h_patches * factor, width=w_patches * factor)


class PaddleOCRVLDummyInputsBuilder(BaseDummyInputsBuilder[PaddleOCRVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class PaddleOCRVLMultiModalProcessor(
    BaseMultiModalProcessor[PaddleOCRVLProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(text=prompt, **mm_data),
                dict(**mm_kwargs, **tok_kwargs),
            )
            num_patches_per_image = processed_outputs["image_grid_thw"].prod(-1)
            processed_outputs["pixel_values"] = processed_outputs["pixel_values"].split(
                num_patches_per_image.tolist()
            )
        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_id

        def get_replacement(item_idx: int, image_processor):
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
                image_processor=image_processor,
            )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=partial(get_replacement, image_processor=image_processor),
            ),
        ]


class Projector(nn.Module):
    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config
        self.merge_kernel_size = (2, 2)

        self.hidden_size = (
            self.vision_config.hidden_size
            * self.merge_kernel_size[0]
            * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(
            self.hidden_size, self.text_config.hidden_size, bias=True
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size
        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                image_feature = self.pre_norm(image_feature)
                t, h, w = image_grid

                image_feature = rearrange(
                    image_feature,
                    "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                    t=t,
                    h=h // m1,
                    p1=m1,
                    w=w // m2,
                    p2=m2,
                )
                hidden_states = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)

            return processed_features

        dims = image_features.shape[:-1]
        dim = image_features.shape[-1]
        image_features = image_features.view(np.prod(dims), dim)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states.view(*dims, -1)


class PaddleOCRImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("bn", "p", 3, "patch_size", "patch_size", dynamic_dims={"p"}),
    ]
    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("bn", 3),
    ]


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = Conv2dLayer(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def fetch_position_embedding_lfu_cache(
        self, embeddings: torch.Tensor, h: int, w: int, max_cache: int = 20
    ):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]

        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(
                self.cache_position_count,
                key=self.cache_position_count.get,
            )
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        interpolate_pos_encoding=False,
    ) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() == 5:
            if position_ids is None:
                raise ValueError(
                    "position_ids cannot be None when pixel_values.dim() is 5."
                )
            (
                batch_size,
                squence_len,
                channel,
                height,
                width,
            ) = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            embeddings = patch_embeds.flatten(-2).squeeze(-1)

            if interpolate_pos_encoding and image_grid_thw is not None:
                start = 0
                tmp_embeddings = list()
                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings[start:end, :]
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .repeat(t, 1)
                    )
                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end
                embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
            else:
                embeddings = embeddings + self.packing_position_embedding(position_ids)
            return embeddings
        else:
            raise ValueError(
                "Unsupported pixel_values dimension:"
                f" {pixel_values.dim()}. Expected 4 or 5."
            )


def all_gather_interleave(local_tensor: torch.Tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist

    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(
        gathered_tensors, local_tensor, group=parallel_state.get_tp_group().device_group
    )

    gathered_tensors_split = [
        torch.split(tensor, hidden_size // tp_size, -1) for tensor in gathered_tensors
    ]
    ordered_tensors = [
        tensor for pair in zip(*gathered_tensors_split) for tensor in pair
    ]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class SiglipAttention(nn.Module):
    """SigLIP vision attention adapted from Qwen2.5-VisionAttention."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()

        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.attn_backend = attn_backend
        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                attn_backend_override=attn_backend_override,
            )
        )
        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            qkv = all_gather_interleave(qkv, self.qkv_proj.hidden_size, self.tp_size)

        q, k, v = qkv.chunk(3, dim=2)

        if self.tp_size > 1:
            splitter = partial(
                dist_utils.split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape

        x = rearrange(hidden_states, "b s d -> s b d")
        x, _ = self.qkv_proj(x)
        q, k, v = self.split_qkv(x)
        q, k, v = (rearrange(t, "s b h d -> b s h d") for t in (q, k, v))

        if rotary_pos_emb is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        if self.is_flash_attn_backend:
            if max_seqlen is None:
                raise ValueError("Flash attention backend requires max_seqlen.")
            context_layer = vit_flash_attn_wrapper(
                q,
                k,
                v,
                cu_seqlens,
                max_seqlen,
                batch_size,
                self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA,
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (
                    rearrange(tensor, "b s h d -> b h s d")
                    for tensor in (q_i, k_i, v_i)
                )
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
            context_layer = rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()
        else:
            raise RuntimeError(
                f"PaddleOCR-VL does not support {self.attn_backend} backend now."
            )

        output, _ = self.out_proj(context_layer)
        output = rearrange(output, "s b d -> b s d")
        return output


class SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        # Special handling for BNB and torchao quantization
        if quant_config and quant_config.get_name() in ["bitsandbytes", "torchao"]:
            quantizable = True
        else:
            # For other quantization, we require the hidden size to be a
            # multiple of 64
            quantizable = (
                config.hidden_size % 64 == 0 and config.intermediate_size % 64 == 0
            )
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        attn_backend: AttentionBackendEnum = AttentionBackendEnum.TORCH_SDPA,
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_backend=attn_backend,
            attn_backend_override=attn_backend_override,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )
        if self.attn_backend not in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"PaddleOCR-VL does not support {self.attn_backend} backend now."
            )
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    attn_backend=self.attn_backend,
                    attn_backend_override=attn_backend_override,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = inputs_embeds.device
        hidden_states = inputs_embeds

        flatten_image_grid_thw = self.flatten_list(image_grid_thw)

        if width_position_ids is None or height_position_ids is None:
            split_hids = list()
            split_wids = list()
            for t, h, w in flatten_image_grid_thw:
                image_pids = torch.arange(t * h * w, device=device) % (h * w)
                sample_hids = image_pids // w
                sample_wids = image_pids % w
                split_hids.append(sample_hids)
                split_wids.append(sample_wids)
            width_position_ids = torch.concat(split_wids, dim=0)
            height_position_ids = torch.concat(split_hids, dim=0)

        pids = torch.stack(
            [height_position_ids, width_position_ids],
            dim=-1,
        )
        max_grid_size = pids.max() + 1
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rope_emb_max_grid[pids].flatten(1)

        if cu_seqlens is None:
            raise ValueError("cu_seqlens cannot be None for SiglipEncoder.")
        if not isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        else:
            cu_seqlens = cu_seqlens.to(device=device)

        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            attn_backend_override=attn_backend_override,
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool | None = False,
        position_ids: torch.Tensor | None = None,
        height_position_ids: torch.Tensor | None = None,
        width_position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend_override: AttentionBackendEnum | None = None,
    ):
        super().__init__()

        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_model",
            attn_backend_override=attn_backend_override,
        )
        self.quant_config = quant_config

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.embeddings.patch_embedding.weight.device

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: bool = False,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: list[tuple[int, int, int] | list[tuple[int, int, int]]]
        | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "head.attention" in name or "head.layernorm" in name:
                continue
            if "head.mlp" in name or "head.probe" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (
                param_name,
                weight_name,
                shard_id,
            ) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
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
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleOCRVLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsMRoPE):
    merge_by_field_config = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend
            if multimodal_config is not None
            else None
        )

        self.visual = SiglipVisionModel(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
            attn_backend_override=attn_backend_override,
        )
        self.mlp_AR = Projector(config, config.vision_config)

        self.language_model = Ernie4_5ForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        for layer in self.language_model.model.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.rotary_emb.is_neox_style = True

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
        second_per_grid_ts = kwargs.get("second_per_grid_ts", [])

        hf_config = self.config
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            else:
                ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    * video_second_per_grid_t
                    * tokens_per_second
                )
                .long()
                .flatten()
            )

            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return llm_positions, mrope_position_delta

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> PaddleOCRImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        return PaddleOCRImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        elif inputs_embeds is None:
            vision_embeddings = self.embed_multimodal(**kwargs)
            is_multimodal = kwargs.pop("is_multimodal", None)
            handle_oov_mm_token = kwargs.pop("handle_oov_mm_token", False)
            inputs_embeds = self.embed_input_ids(
                input_ids,
                vision_embeddings,
                is_multimodal=is_multimodal,
                handle_oov_mm_token=handle_oov_mm_token,
            )
            input_ids = None

        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"

        raise ValueError("Only image modality is supported")

    def encode_image(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.type(self.visual.dtype)
        siglip_position_ids = list()
        image_grid_hws = list()
        cu_seqlens = [0]

        thw_tuple = tuple(image_grid_thw.tolist())
        numel = np.prod(thw_tuple)
        image_grid_hws.append(thw_tuple)
        image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
        siglip_position_ids.append(image_position_ids)
        cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
            pixel_values.device
        )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)

        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        )
        return vision_outputs

    def _process_image_input(
        self, image_input: PaddleOCRImagePixelInputs
    ) -> MultiModalEmbeddings:
        pixel_values = image_input.pixel_values
        image_grid_thw = image_input.image_grid_thw
        vision_outputs = tuple(
            self.encode_image(pixel, grid).squeeze(0)
            for pixel, grid in zip(pixel_values, image_grid_thw)
        )
        image_embeds = self.mlp_AR(vision_outputs, image_grid_thw)
        return image_embeds

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return ()

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        image_embeds = self._process_image_input(image_input)
        multimodal_embeddings += tuple(image_embeds)

        return multimodal_embeddings

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights
