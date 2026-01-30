# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
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
"""Inference-only MiniCPM-V model compatible with HuggingFace weights."""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from itertools import chain
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.types
from torch import nn
from torch.nn.init import trunc_normal_
from transformers import BatchFeature, PretrainedConfig
from typing_extensions import TypeVar

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import (
    BaseResampler,
    Resampler2,
    get_2d_sincos_pos_embed,
)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageItem,
    ImageProcessorItems,
    ImageSize,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoItem,
    VideoProcessorItems,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    ResolvedPromptUpdate,
    _seq2text,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import flatten_2d_lists
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import set_default_torch_dtype

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, flatten_bn, maybe_prefix

# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class MiniCPMVImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bns: Batch size * number of images * number of slices
        - bn: Batch size * number of images
        - c: Number of channels
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"] = "pixel_values"

    # Note that the patch size may vary, so we pass it as a list instead of a
    # batched tensor.
    pixel_values: Annotated[
        list[torch.Tensor],
        TensorShape("bns", "c", "h", "w", dynamic_dims={"h", "w"}),
    ]
    tgt_sizes: Annotated[
        torch.Tensor,
        TensorShape("bns", 2),  # This should be in `(height, width)` format.
    ]
    num_slices: Annotated[
        torch.Tensor,
        TensorShape("bn"),
    ]


class MiniCPMVImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ns: Number of slices
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"]
    image_embeds: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "ns", "hs", dynamic_dims={"ns"}),
    ]


MiniCPMVImageInputs: TypeAlias = MiniCPMVImagePixelInputs | MiniCPMVImageEmbeddingInputs

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


class Resampler2_5(BaseResampler):
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: tuple[int, int] = (70, 70),
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(
        self, max_size: tuple[int, int], device: torch.types.Device = "cpu"
    ) -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(
            self.embed_dim, max_size, version=(2, 5)
        )
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(
        self, tgt_sizes: torch.Tensor, device: torch.types.Device
    ) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0
        ).permute(1, 0, 2)  # BLD => L * B * D
        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


class Resampler4_5(Resampler2_5):
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
        max_size: tuple[int, int] = (70, 70),
        max_temporal_size: int = 36000,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_queries,
            embed_dim,
            num_heads,
            kv_dim,
            norm_layer,
            max_size,
            quant_config=quant_config,
            prefix=prefix,
        )

        trunc_normal_(self.query, std=0.02)
        self.max_temporal_size = max_temporal_size
        self._set_temporal_pos_cache(self.max_temporal_size)
        self.apply(self._init_weights)

    def get_1d_sincos_pos_embed_from_temporal_size(
        self, embed_dim: int, pos: np.ndarray
    ):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def _set_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ) -> None:
        temporal_size = np.arange(max_temporal_size, dtype=np.float32)
        pos_embed = (
            torch.from_numpy(
                self.get_1d_sincos_pos_embed_from_temporal_size(
                    self.embed_dim, temporal_size
                )
            )
            .float()
            .to(device)
        )
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)

    def _adjust_temporal_pos_cache(
        self, max_temporal_size: int, device: torch.types.Device = "cpu"
    ):
        if max_temporal_size > self.max_temporal_size:
            self.max_temporal_size = max_temporal_size
            self._set_temporal_pos_cache(self.max_temporal_size, device)

    def _init_weights(self, m: nn.Linear | nn.LayerNorm):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        tgt_sizes: torch.Tensor,
        # temporal_ids for high refresh rate videos
        temporal_ids=None,
    ) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        temporal_pos_emb = False
        temporal_ids_flatten = None
        if temporal_ids is not None:
            # example: [[-1], [-1], [2, 6, 9]]
            temporal_ids_flatten = list(chain.from_iterable(temporal_ids))
            max_temporal_size = max(temporal_ids_flatten, default=0)
            if max_temporal_size > -1:
                temporal_pos_emb = True
            if max_temporal_size > self.max_temporal_size:
                self._adjust_temporal_pos_cache(max_temporal_size, device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros(
            (bs, max_patch_len), dtype=torch.bool, device=device
        )

        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D
        q = self.ln_q(self.query)  # Q * D

        pos_embed_2d = []
        pos_embed_temporal = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            if temporal_pos_emb:
                if temporal_ids_flatten[i] == -1:
                    pos_embed_temporal.append(
                        torch.zeros(self.embed_dim, dtype=dtype, device=device)
                    )
                else:
                    pos_embed_temporal.append(
                        self.temporal_pos_embed[temporal_ids_flatten[i]].to(dtype)
                    )  # D

            pos_embed_2d.append(
                self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            )  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed_2d = torch.nn.utils.rnn.pad_sequence(
            pos_embed_2d, batch_first=True, padding_value=0.0
        ).permute(1, 0, 2)  # BLD => L * B * D

        k = x
        v = x + pos_embed_2d
        if pos_embed_temporal:
            k += torch.stack(pos_embed_temporal, dim=0)
            bs = len(temporal_ids)
            merge_k = []
            merge_v = []
            merge_key_padding_mask = []

            start = 0
            for tp in temporal_ids:
                end = start + len(tp)
                # L * (end-start) * D -> (end-start) * L * D
                # -> 1 * L*(end-start) * D
                merge_k.append(
                    k[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_v.append(
                    v[:, start:end, :].permute(1, 0, 2).reshape(-1, self.embed_dim)
                )
                merge_key_padding_mask.append(
                    key_padding_mask[start:end, :].reshape(-1, 1)
                )

                start = end

            k = torch.nn.utils.rnn.pad_sequence(
                merge_k, batch_first=True, padding_value=0.0
            ).permute(1, 0, 2)  # L*(end-start)
            v = torch.nn.utils.rnn.pad_sequence(
                merge_v, batch_first=True, padding_value=0.0
            ).permute(1, 0, 2)  # L*(end-start)
            key_padding_mask = torch.nn.utils.rnn.pad_sequence(
                merge_key_padding_mask, batch_first=True, padding_value=True
            ).squeeze(-1)

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            k,  # L * B * D +  L * B * D
            v,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


def get_version_by_config(config: PretrainedConfig) -> tuple[int, ...]:
    version_float = getattr(config, "version", None)

    # The old configs do not include version number
    # TODO: Remove this after the HF repos are updated
    if version_float is None:
        if config.hidden_size == 2304 and config.query_num == 64:
            return (2, 0)
        return (2, 5)
    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


def _minicpmv_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
        image_sizes=MultiModalFieldConfig.batched("image"),
        tgt_sizes=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
        video_pixel_values=MultiModalFieldConfig.batched("video"),
        video_image_sizes=MultiModalFieldConfig.batched("video"),
        video_tgt_sizes=MultiModalFieldConfig.batched("video"),
        video_embeds=MultiModalFieldConfig.batched("video"),
    )


class MiniCPMVImageEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={"image_embeds", "image_sizes"},
            fields_factory=fields_factory,
        )

    def get_image_size(self, index: int) -> ImageSize:
        image_size = self.get(index)["image_sizes"].tolist()
        return ImageSize(width=image_size[0], height=image_size[1])


class MiniCPMVVideoEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="video",
            required_fields={"video_embeds", "video_image_sizes"},
            fields_factory=fields_factory,
        )

    def get_frame_size(self, index: int) -> ImageSize:
        frame_size = self.get(index)["video_image_sizes"].tolist()
        return ImageSize(width=frame_size[0], height=frame_size[1])

    def get_num_frames(self, index: int) -> int:
        return len(self.get(index)["video_image_sizes"])


class MiniCPMVMultiModalDataParser(MultiModalDataParser):
    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return MiniCPMVImageEmbeddingItems(
                data,
                fields_factory=_minicpmv_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return MiniCPMVVideoEmbeddingItems(
                data,
                fields_factory=_minicpmv_field_config,
            )

        return super()._parse_video_data(data)


class MiniCPMVProcessingInfo(BaseProcessingInfo):
    image_pattern = "(<image>./</image>)"
    video_pattern = "(<video>./</video>)"

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NumPy arrays are considered as Iterable but not Sequence in
        # https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L428
        image_processor = hf_processor.image_processor  # type: ignore
        for attr in ("mean", "std"):
            val = getattr(image_processor, attr)
            if isinstance(val, np.ndarray):
                setattr(image_processor, attr, val.tolist())

        return hf_processor

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_data_parser(self):
        return MiniCPMVMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_model_version(self):
        return get_version_by_config(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        mm_limits = {"image": None}
        if self.get_model_version() in {(2, 6), (4, 0), (4, 5)}:
            mm_limits["video"] = None

        return mm_limits

    def get_slice_image_placeholder(
        self,
        image_size: ImageSize,
        # For MiniCPM V/O 2.6
        image_idx: int = 0,
        max_slice_nums: int | None = None,
        use_image_id: bool = True,
    ) -> str:
        image_processor = self.get_image_processor()
        version = self.get_model_version()

        if version == (2, 0) or version == (2, 5):
            return image_processor.get_slice_image_placeholder(image_size)

        return image_processor.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
        )

    def get_sliced_grid(
        self,
        image_size: ImageSize,
        # For MiniCPM V/O 2.6
        max_slice_nums: int | None = None,
    ) -> tuple[int, int] | None:
        image_processor = self.get_image_processor()
        version = self.get_model_version()

        if version == (2, 0) or version == (2, 5):
            return image_processor.get_sliced_grid(image_size)

        if max_slice_nums is None:
            max_slice_nums = image_processor.max_slice_nums

        return image_processor.get_sliced_grid(
            image_size,
            max_slice_nums=max_slice_nums,
        )

    def get_num_image_tokens(
        self,
        image_size: ImageSize,
        max_slice_nums: int | None = None,
    ) -> int:
        image_processor = self.get_image_processor()

        grid = self.get_sliced_grid(
            image_size,
            max_slice_nums=max_slice_nums,
        )
        if grid is None:
            ncols = nrows = 0
        else:
            ncols, nrows = grid

        return (ncols * nrows + 1) * image_processor.image_feature_size

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_size)

    def get_image_max_slice_num(self) -> int:
        return getattr(self.get_hf_config(), "max_slice_num", 9)

    def get_image_size_with_most_features(self) -> ImageSize:
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        max_slice_num = self.get_image_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)

    def get_max_video_frame_tokens(self) -> int:
        frame_size = self.get_video_frame_size_with_most_features()

        return self.get_num_image_tokens(
            frame_size,
            max_slice_nums=self.get_video_max_slice_num(),
        )

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        num_frames = self.get_num_frames_with_most_features(seq_len, mm_counts)
        num_video_tokens_total = self.get_max_video_frame_tokens() * num_frames
        return num_video_tokens_total

    def get_video_max_slice_num(self) -> int:
        return 1

    def get_video_frame_size_with_most_features(self) -> ImageSize:
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        max_slice_num = self.get_video_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)

    def get_max_video_frames(self, max_tokens: int) -> int:
        num_frame_tokens = self.get_max_video_frame_tokens()
        num_frames = max_tokens // num_frame_tokens
        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self.get_max_video_frames(seq_len - max_image_tokens)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )

        return max(max_frames_per_video, 1)


_I = TypeVar("_I", bound=MiniCPMVProcessingInfo, default=MiniCPMVProcessingInfo)


class MiniCPMVDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_prompt_texts = self.info.image_pattern * num_images
        video_prompt_texts = self.info.video_pattern * num_videos

        return image_prompt_texts + video_prompt_texts

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_width, image_height = self.info.get_image_size_with_most_features()
        video_width, video_height = self.info.get_video_frame_size_with_most_features()
        num_video_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": [
                self._get_dummy_images(
                    width=video_width,
                    height=video_height,
                    num_images=num_video_frames,
                    overrides=video_overrides,
                )
            ]
            * num_videos,
        }


class MiniCPMVMultiModalProcessor(BaseMultiModalProcessor[_I]):
    def get_image_prompt_texts(self, image_size: ImageSize, image_idx: int = 0) -> str:
        return self.info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
        )

    def get_video_prompt_texts(self, image_size: ImageSize, num_frames: int) -> str:
        return (
            self.info.get_slice_image_placeholder(
                image_size=image_size,
                image_idx=0,
                max_slice_nums=self.info.get_video_max_slice_num(),
                use_image_id=False,
            )
            * num_frames
        )

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (images := mm_data.get("images")) is None:
            return {}

        parsed_images = self.data_parser.parse_mm_data({"image": images}).get_items(
            "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
        )

        if isinstance(parsed_images, MiniCPMVImageEmbeddingItems):
            image_inputs = {}
        else:
            image_inputs = self._base_call_hf_processor(
                prompts=[self.info.image_pattern] * len(parsed_images),
                mm_data={"images": [[image] for image in parsed_images]},
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        return image_inputs

    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (videos := mm_data.get("videos")) is None:
            return {}

        parsed_videos = self.data_parser.parse_mm_data({"video": videos}).get_items(
            "video", (MiniCPMVVideoEmbeddingItems, VideoProcessorItems)
        )

        if isinstance(parsed_videos, MiniCPMVVideoEmbeddingItems):
            video_inputs = {}
        else:
            video_inputs = self._base_call_hf_processor(
                prompts=[
                    self.info.image_pattern * len(video) for video in parsed_videos
                ],
                mm_data={"images": list(parsed_videos)},
                mm_kwargs={
                    **mm_kwargs,
                    "max_slice_nums": self.info.get_video_max_slice_num(),
                },
                tok_kwargs=tok_kwargs,
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        video_inputs = {f"video_{k}": v for k, v in video_inputs.items()}

        return video_inputs

    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        return {
            **self.process_images(mm_data, mm_kwargs, tok_kwargs),
            **self.process_videos(mm_data, mm_kwargs, tok_kwargs),
        }

    def _base_call_hf_processor(
        self,
        prompts: list[str],
        mm_data: Mapping[str, Sequence[object]],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
        *,
        out_keys: set[str],
    ) -> dict[str, NestedTensors]:
        # This processor supports zipping prompt and mm_data together
        if self.info.get_model_version() in {(2, 6), (4, 0), (4, 5)}:
            inputs = super()._call_hf_processor(
                prompt=prompts,  # type: ignore
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )
        else:
            inputs = defaultdict[str, list[torch.Tensor]](list)

            for i, prompt in enumerate(prompts):
                inputs_one = super()._call_hf_processor(
                    prompt=prompt,
                    mm_data={k: v[i] for k, v in mm_data.items()},
                    mm_kwargs=mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                for k, v in inputs_one.items():
                    assert len(v) == 1, (k, len(v))
                    inputs[k].append(v[0])

        return {k: inputs[k] for k in out_keys}

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        input_ids = torch.tensor([tokenizer.encode(prompt, **tok_kwargs)])
        mm_inputs = self.process_mm_inputs(mm_data, mm_kwargs, tok_kwargs)

        return BatchFeature(
            {
                "input_ids": input_ids,
                **mm_inputs,
            }
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        placeholders = [
            ("image", self.info.image_pattern),
            ("video", self.info.video_pattern),
        ]

        # hard code for inconsistency of encode-decode image_pattern
        additional_placeholders = []
        tokenizer = self.info.get_tokenizer()
        for modality, pattern in placeholders:
            sub_pattern = tokenizer.decode(
                tokenizer.encode(pattern, add_special_tokens=False)
            )
            if sub_pattern != pattern:
                additional_placeholders.append((modality, sub_pattern))
        placeholders += additional_placeholders

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems)
            )

            image_size = images.get_image_size(item_idx)

            return PromptUpdateDetails.select_text(
                self.get_image_prompt_texts(image_size, item_idx),
                "<unk>",
            )

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (MiniCPMVVideoEmbeddingItems, VideoProcessorItems)
            )

            frame_size = videos.get_frame_size(item_idx)
            num_frames = videos.get_num_frames(item_idx)

            return PromptUpdateDetails.select_text(
                self.get_video_prompt_texts(frame_size, num_frames),
                "<unk>",
            )

        get_replacement = {
            "image": get_image_replacement,
            "video": get_video_replacement,
        }

        return [
            PromptReplacement(
                modality=modality, target=pattern, replacement=get_replacement[modality]
            )
            for modality, pattern in placeholders
        ]

    def _recompute_cached_prompt_update(
        self,
        cached_update: ResolvedPromptUpdate,
        new_item_idx: int,
    ) -> ResolvedPromptUpdate:
        new_update = super()._recompute_cached_prompt_update(
            cached_update,
            new_item_idx,
        )

        if cached_update.modality == "image":
            tokenizer = self.info.get_tokenizer()
            image_processor = self.info.get_image_processor()
            version = self.info.get_model_version()

            text = _seq2text(tokenizer, cached_update.content.full)
            prev_item_idx = cached_update.item_idx

            if version == (2, 0) or version == (2, 5):
                im_start = image_processor.im_start_token
                im_end = image_processor.im_end_token
            else:
                im_start = image_processor.im_id_start
                im_end = image_processor.im_id_end

            new_update = new_update.with_content(
                PromptUpdateDetails.select_text(
                    text.replace(
                        f"{im_start}{prev_item_idx}{im_end}",
                        f"{im_start}{new_item_idx}{im_end}",
                        1,
                    ),
                    "<unk>",
                )
            )

        return new_update

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmv_field_config(hf_inputs)


class MiniCPMVBaseModel(nn.Module, SupportsMultiModal, SupportsPP):
    """
    The abstract class of MiniCPMV can only be inherited, but cannot be
    instantiated.
    """

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot
        # check `tie_word_embeddings` until vLLM integrate MiniCPM-V model
        # and config class
        self.config = config
        self.multimodal_config = multimodal_config

        self.version = get_version_by_config(self.config)

        with self._mark_language_model(vllm_config):
            self.llm = self.init_llm(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "llm")
            )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vpm = self.init_vision_module(
                config, quant_config, prefix=maybe_prefix(prefix, "vpm")
            )
            self.vision_dim = (
                self.vpm.embed_dim
                if self.version == (2, 0)
                else self.vpm.embeddings.embed_dim
            )
            self.embed_dim = self.config.hidden_size

            self.resampler = self.init_resampler(
                self.embed_dim,
                self.vision_dim,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "resampler"),
            )

        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

    def _parse_and_validate_vision_input(
        self,
        modality: str,
        **kwargs: object,
    ) -> MiniCPMVImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return MiniCPMVImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
            )

        tgt_sizes = kwargs.pop("tgt_sizes")

        num_slices_flat = torch.tensor([len(ps) for ps in pixel_values])
        pixel_values_flat = flatten_bn(pixel_values)
        tgt_sizes_flat = flatten_bn(tgt_sizes, concat=True)

        return MiniCPMVImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values_flat,
            tgt_sizes=tgt_sizes_flat,
            num_slices=num_slices_flat,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_vision_input(
                    "images", **kwargs
                )
            if (
                input_key in ("video_pixel_values", "video_embeds")
                and "videos" not in modalities
            ):
                modalities["videos"] = self._parse_and_validate_vision_input(
                    "videos", **{k.removeprefix("video_"): v for k, v in kwargs.items()}
                )

        return modalities

    def _process_vision_input(
        self,
        image_input: MiniCPMVImageInputs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        image_features_flat = self.get_vision_hidden_states(image_input)

        num_slices = image_input["num_slices"]
        return [e.flatten(0, 1) for e in image_features_flat.split(num_slices.tolist())]

    def _process_multimodal_inputs(self, modalities: dict):
        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_vision_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_vision_input(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        return self._process_multimodal_inputs(modalities)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.llm.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="llm", connector="resampler", tower_model="vpm"
        )

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        raise NotImplementedError


class MiniCPMV2_0(MiniCPMVBaseModel):
    supports_encoder_tp_data = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 0)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return MiniCPMForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module:
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm==0.9.10") from ImportError

        with set_default_torch_dtype(torch.float16):
            model = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )

        model = model.to(dtype=torch.get_default_dtype())

        if (
            isinstance(model, timm.models.VisionTransformer)
            and model.attn_pool is not None
        ):
            model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2(
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                grid_size=int(math.sqrt(self.config.query_num)),
                kv_dim=vision_dim,
                adaptive=False,
                do_post_projection=True,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(
            device=current_platform.device_type, dtype=torch.get_default_dtype()
        )

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]

        P_h, P_w = self.vpm.patch_embed.patch_size
        dtype: torch.dtype = self.vpm.pos_embed.data.dtype
        num_prefix_tokens = getattr(self.vpm, "num_prefix_tokens", 0)

        res = list[torch.Tensor]()
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (math.ceil(H / P_h), math.ceil(W / P_w))
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype)
            )

            if num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))

        return torch.vstack(res)


class MiniCPMV2_5(MiniCPMVBaseModel, SupportsLoRA):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 5)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return LlamaForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(
            device=current_platform.device_type, dtype=torch.get_default_dtype()
        )

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L), dtype=dtype, device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=None,
        )

        return self.resampler(vision_embedding, tgt_sizes)


class MiniCPMV2_6(MiniCPMVBaseModel, SupportsLoRA):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 6)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen2ForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(
            device=current_platform.device_type, dtype=torch.get_default_dtype()
        )

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L), dtype=dtype, device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["apm.", "audio", "tts"])
        return loader.load_weights(weights)


class MiniCPMV4_0(MiniCPMVBaseModel, SupportsLoRA):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (4, 0)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return LlamaForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 4.0 remains consistent with the one in 2.5/2.6.
            resampler = Resampler2_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(
            device=current_platform.device_type, dtype=torch.get_default_dtype()
        )

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L), dtype=dtype, device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["apm.", "audio", "tts"])
        return loader.load_weights(weights)


class MiniCPMV4_5(MiniCPMVBaseModel, SupportsLoRA):
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (4, 5)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen3ForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(
        self,
        embed_dim: int,
        vision_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 4.0 remains consistent with the one in 2.5/2.6.
            resampler = Resampler4_5(
                num_queries=self.config.query_num,
                embed_dim=embed_dim,
                num_heads=embed_dim // 128,
                kv_dim=vision_dim,
                quant_config=quant_config,
                prefix=prefix,
            )

        return resampler.to(
            device=current_platform.device_type, dtype=torch.get_default_dtype()
        )

    def get_vision_hidden_states(self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]
        temporal_ids = data.get("temporal_ids", None)

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L), dtype=dtype, device=device)
        all_temporal_ids = (
            None if temporal_ids is None else flatten_2d_lists(temporal_ids)
        )
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes, all_temporal_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["apm.", "audio", "tts"])
        return loader.load_weights(weights)


_SUPPORT_VERSION = {
    (2, 0): MiniCPMV2_0,
    (2, 5): MiniCPMV2_5,
    (2, 6): MiniCPMV2_6,
    (4, 0): MiniCPMV4_0,
    (4, 5): MiniCPMV4_5,
}


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMVMultiModalProcessor,
    info=MiniCPMVProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMV(MiniCPMVBaseModel, SupportsMultiModal, SupportsLoRA):
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in vLLM. Therefore, it is necessary to separate them.
    """

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if not hasattr(config, "version"):
            if config.hidden_size == 2304 and config.query_num == 64:
                version = (2, 0)
            else:
                version = (2, 5)
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version
        instance_cls = _SUPPORT_VERSION.get(version)
        if instance_cls is None:
            supported_versions = ", ".join(
                [f"{v[0]}.{v[1]}" for v in sorted(_SUPPORT_VERSION.keys())]
            )
            raise ValueError(
                f"Currently, MiniCPMV only supports versions "
                f"{supported_versions}. Got version: {version}"
            )

        # quant_config references base class members,
        # so update values before init is called
        cls.packed_modules_mapping.update(instance_cls.packed_modules_mapping)
        cls.embedding_modules.update(instance_cls.embedding_modules)
        return instance_cls(vllm_config=vllm_config, prefix=prefix)
