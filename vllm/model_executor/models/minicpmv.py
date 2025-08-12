# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property, partial
from typing import (Any, Callable, Literal, Optional, Set, Tuple, TypedDict,
                    Union)

import numpy as np
import torch
import torch.types
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from typing_extensions import TypeVar

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import (BaseResampler, Resampler2,
                                                  get_2d_sincos_pos_embed)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import (DictEmbeddingItems, ImageItem,
                                   ImageProcessorItems, ImageSize,
                                   ModalityData, ModalityDataItems,
                                   MultiModalDataItems, MultiModalDataParser,
                                   VideoItem, VideoProcessorItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import flatten_2d_lists

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)
from .vision import scatter_patch_features, select_patch_features

# For profile run
_MAX_FRAMES_PER_VIDEO = 16


class MiniCPMVImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: list[torch.Tensor]
    """
    Shape: `(batch_size * num_images * num_slices, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, 2)`

    This should be in `(height, width)` format.
    """

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.

    Shape: `(batch_size * num_images, num_embeds)`
    """

    num_slices: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class MiniCPMVImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_slices, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.

    Shape: `(batch_size * num_images, num_embeds)`
    """


MiniCPMVImageInputs = Union[MiniCPMVImagePixelInputs,
                            MiniCPMVImageEmbeddingInputs]

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


class Resampler2_5(BaseResampler):

    def __init__(self,
                 num_queries: int,
                 embed_dim: int,
                 num_heads: int,
                 kv_dim: Optional[int] = None,
                 norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
                 max_size: Tuple[int, int] = (70, 70),
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__(num_queries,
                         embed_dim,
                         num_heads,
                         kv_dim,
                         norm_layer,
                         quant_config=quant_config,
                         prefix=prefix)

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self,
                          max_size: Tuple[int, int],
                          device: torch.types.Device = "cpu") -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim,
                                                max_size,
                                                version=(2, 5))
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor,
                          device: torch.types.Device) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor,
                tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros((bs, max_patch_len),
                                       dtype=torch.bool,
                                       device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape(
                (tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed,
                                                    batch_first=True,
                                                    padding_value=0.0).permute(
                                                        1, 0,
                                                        2)  # BLD => L * B * D
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


def get_version_by_config(config: PretrainedConfig) -> Tuple[int, ...]:
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
    pixel_values = hf_inputs.get("pixel_values", torch.empty(0))
    num_images = len(pixel_values)

    video_pixel_values = hf_inputs.get("video_pixel_values", torch.empty(0))
    num_videos = len(video_pixel_values)

    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
        image_sizes=MultiModalFieldConfig.batched("image"),
        tgt_sizes=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
        embed_is_patch=MultiModalFieldConfig.batched("image"),
        video_pixel_values=MultiModalFieldConfig.batched("video"),
        video_image_sizes=MultiModalFieldConfig.batched("video"),
        video_tgt_sizes=MultiModalFieldConfig.batched("video"),
        video_embeds=MultiModalFieldConfig.batched("video"),
        video_embed_is_patch=MultiModalFieldConfig.batched("video"),
        image_token_id=MultiModalFieldConfig.shared("image", num_images),
        video_token_id=MultiModalFieldConfig.shared("video", num_videos),
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
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return MiniCPMVImageEmbeddingItems(
                data,
                fields_factory=_minicpmv_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> ModalityDataItems[Any, Any]:
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

    def get_image_processor(self):
        hf_processor = self.get_hf_processor()
        image_processor = hf_processor.image_processor  # type: ignore
        return image_processor

    def get_model_version(self):
        return get_version_by_config(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        mm_limits = {"image": None}
        if self.get_model_version() == (2, 6):
            mm_limits["video"] = None

        return mm_limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        mm_max_tokens = {"image": self.get_max_image_tokens()}
        if self.get_model_version() == (2, 6):
            mm_max_tokens["video"] = self.get_max_video_tokens(
                seq_len, mm_counts)

        return mm_max_tokens

    def get_slice_image_placeholder(
        self,
        image_size: ImageSize,
        # For MiniCPM V/O 2.6
        image_idx: int = 0,
        max_slice_nums: Optional[int] = None,
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

    def get_num_image_tokens(
        self,
        image_size: ImageSize,
        max_slice_nums: Optional[int] = None,
        use_image_id: bool = True,
    ) -> int:
        tokenizer = self.get_tokenizer()
        image_placeholders = self.get_slice_image_placeholder(
            image_size,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
        )
        image_token_ids = tokenizer.encode(image_placeholders,
                                           add_special_tokens=False)

        return len(image_token_ids)

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
            use_image_id=False,
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
        max_total_frames = self.get_max_video_frames(seq_len -
                                                     max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)


_I = TypeVar("_I",
             bound=MiniCPMVProcessingInfo,
             default=MiniCPMVProcessingInfo)


class MiniCPMVDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_width, image_height = \
            self.info.get_image_size_with_most_features()
        video_width, video_height = \
            self.info.get_video_frame_size_with_most_features()
        num_video_frames = \
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        mm_data = {
            "image":
            self._get_dummy_images(width=image_width,
                                   height=image_height,
                                   num_images=num_images),
            "video": [
                self._get_dummy_images(width=video_width,
                                       height=video_height,
                                       num_images=num_video_frames)
            ] * num_videos,
        }

        image_prompt_texts = self.info.image_pattern * num_images
        video_prompt_texts = self.info.video_pattern * num_videos

        return ProcessorInputs(prompt_text=image_prompt_texts +
                               video_prompt_texts,
                               mm_data=mm_data)


class MiniCPMVMultiModalProcessor(BaseMultiModalProcessor[_I]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMVMultiModalDataParser()

    def get_image_prompt_texts(self,
                               image_size: ImageSize,
                               image_idx: int = 0) -> str:
        return self.info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
        )

    def get_video_prompt_texts(self, image_size: ImageSize,
                               num_frames: int) -> str:
        return self.info.get_slice_image_placeholder(
            image_size=image_size,
            image_idx=0,
            max_slice_nums=self.info.get_video_max_slice_num(),
            use_image_id=False,
        ) * num_frames

    def get_embed_is_patch(
        self,
        input_ids: list[int],
    ) -> torch.Tensor:
        tokenizer = self.info.get_tokenizer()
        unk_token_id = tokenizer.get_vocab()["<unk>"]
        return torch.tensor(input_ids) == unk_token_id

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (images := mm_data.get("images")) is None:
            return {}

        parsed_images = (self._get_data_parser().parse_mm_data({
            "image": images
        }).get_items("image",
                     (MiniCPMVImageEmbeddingItems, ImageProcessorItems)))

        if isinstance(parsed_images, MiniCPMVImageEmbeddingItems):
            image_inputs = {}
        else:
            image_inputs = self._base_call_hf_processor(
                prompts=[self.info.image_pattern] * len(parsed_images),
                mm_data={"images": [[image] for image in parsed_images]},
                mm_kwargs=mm_kwargs,
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        image_sizes = [
            parsed_images.get_image_size(i) for i in range(len(parsed_images))
        ]
        image_repl_features = [
            self.get_image_prompt_texts(size, idx)
            for idx, size in enumerate(image_sizes)
        ]

        tokenizer = self.info.get_tokenizer()
        image_repls_feature_tokens = [
            tokenizer.encode(image_repl, add_special_tokens=False)
            for image_repl in image_repl_features
        ]

        embed_is_patch = [
            self.get_embed_is_patch(image_repl_tokens)
            for image_repl_tokens in image_repls_feature_tokens
        ]
        image_inputs["embed_is_patch"] = embed_is_patch

        unk_token_id = tokenizer.get_vocab()["<unk>"]
        image_inputs["image_token_id"] = torch.tensor(unk_token_id)

        return image_inputs

    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (videos := mm_data.get("videos")) is None:
            return {}

        parsed_videos = (self._get_data_parser().parse_mm_data({
            "video": videos
        }).get_items("video",
                     (MiniCPMVVideoEmbeddingItems, VideoProcessorItems)))

        if isinstance(parsed_videos, MiniCPMVVideoEmbeddingItems):
            video_inputs = {}
        else:
            video_inputs = self._base_call_hf_processor(
                prompts=[
                    self.info.image_pattern * len(video)
                    for video in parsed_videos
                ],
                mm_data={"images": list(parsed_videos)},
                mm_kwargs={
                    **mm_kwargs,
                    "max_slice_nums":
                    self.info.get_video_max_slice_num(),
                },
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        frame_sizes = [
            parsed_videos.get_frame_size(i) for i in range(len(parsed_videos))
        ]
        num_frames = [
            parsed_videos.get_num_frames(i) for i in range(len(parsed_videos))
        ]
        video_repl_features = [
            self.get_video_prompt_texts(size, nframes)
            for size, nframes in zip(frame_sizes, num_frames)
        ]

        tokenizer = self.info.get_tokenizer()
        video_repls_feature_tokens = [
            tokenizer.encode(video_repl, add_special_tokens=False)
            for video_repl in video_repl_features
        ]

        embed_is_patch = [
            self.get_embed_is_patch(video_repl_tokens)
            for video_repl_tokens in video_repls_feature_tokens
        ]
        video_inputs["embed_is_patch"] = embed_is_patch

        video_inputs = {f"video_{k}": v for k, v in video_inputs.items()}

        unk_token_id = tokenizer.get_vocab()["<unk>"]
        video_inputs["video_token_id"] = torch.tensor(unk_token_id)

        return video_inputs

    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        return {
            **self.process_images(mm_data, mm_kwargs),
            **self.process_videos(mm_data, mm_kwargs),
        }

    def _base_call_hf_processor(
        self,
        prompts: list[str],
        mm_data: Mapping[str, Sequence[object]],
        mm_kwargs: Mapping[str, object],
        *,
        out_keys: set[str],
    ) -> dict[str, NestedTensors]:
        # This processor supports zipping prompt and mm_data together
        if self.info.get_model_version() == (2, 6):
            inputs = super()._call_hf_processor(
                prompt=prompts,  # type: ignore
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
            )
        else:
            inputs = defaultdict[str, list[torch.Tensor]](list)

            for i, prompt in enumerate(prompts):
                inputs_one = super()._call_hf_processor(
                    prompt=prompt,
                    mm_data={
                        k: v[i]
                        for k, v in mm_data.items()
                    },
                    mm_kwargs=mm_kwargs,
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
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        input_ids = torch.tensor([tokenizer.encode(prompt)])
        mm_inputs = self.process_mm_inputs(mm_data, mm_kwargs)

        return BatchFeature({
            "input_ids": input_ids,
            **mm_inputs,
        })

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        placeholder = {
            "image": self.info.image_pattern,
            "video": self.info.video_pattern,
        }

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems))

            image_size = images.get_image_size(item_idx)

            return self.get_image_prompt_texts(image_size, item_idx)

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (MiniCPMVVideoEmbeddingItems, VideoProcessorItems))

            frame_size = videos.get_frame_size(item_idx)
            num_frames = videos.get_num_frames(item_idx)

            return self.get_video_prompt_texts(frame_size, num_frames)

        get_replacement = {
            "image": get_image_replacement,
            "video": get_video_replacement,
        }

        return [
            PromptReplacement(modality=modality,
                              target=placeholder[modality],
                              replacement=get_replacement[modality])
            for modality in ("image", "video")
        ]

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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot
        # check `tie_word_embeddings` until vLLM integrate MiniCPM-V model
        # and config class
        self.config = config
        self.multimodal_config = multimodal_config

        self.version = get_version_by_config(self.config)
        self.llm = self.init_llm(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "llm"))
        self.vpm = self.init_vision_module(config,
                                           quant_config,
                                           prefix=maybe_prefix(prefix, "vpm"))
        self.vision_dim = (self.vpm.embed_dim if self.version == (2, 0) else
                           self.vpm.embeddings.embed_dim)
        self.embed_dim = self.config.hidden_size

        self.resampler = self.init_resampler(self.embed_dim,
                                             self.vision_dim,
                                             quant_config=quant_config,
                                             prefix=maybe_prefix(
                                                 prefix, "resampler"))

        self.mm_token_ids = set[int]()
        self.make_empty_intermediate_tensors = (
            self.llm.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.llm, "sampler"):
            return self.llm.sampler

        return get_sampler()

    def _parse_and_validate_vision_input(
        self,
        modality: str,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        image_token_id = kwargs.pop("image_token_id")
        if image_token_id is not None:
            assert isinstance(image_token_id, torch.Tensor)
            self.mm_token_ids.add(image_token_id.flatten().unique().item())

        embed_is_patch = kwargs.pop("embed_is_patch")
        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of embed_is_patch for {modality=}. "
                f"Got type: {type(embed_is_patch)}")

        embed_is_patch = flatten_bn(embed_is_patch)

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of image_embeds for {modality=}. "
                    f"Got type: {type(image_embeds)}")

            image_embeds_flat = flatten_bn(image_embeds)

            return MiniCPMVImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds_flat,
                embed_is_patch=embed_is_patch,
            )

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of pixel_values for {modality=}. "
                f"Got type: {type(pixel_values)}")

        tgt_sizes = kwargs.pop("tgt_sizes")
        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of tgt_sizes for {modality=}. "
                             f"Got type: {type(tgt_sizes)}")

        num_slices = [[len(p) for p in ps] for ps in pixel_values]
        num_slices_flat = flatten_bn(torch.tensor(num_slices))

        pixel_values_flat = flatten_bn(flatten_2d_lists(pixel_values))
        tgt_sizes_flat = flatten_bn(flatten_2d_lists(tgt_sizes), concat=True)

        if len(pixel_values_flat) != len(tgt_sizes_flat):
            raise ValueError("Inconsistent flattened lengths, found: "
                             f"{len(pixel_values_flat)} vs. "
                             f"{len(tgt_sizes_flat)}")

        return MiniCPMVImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values_flat,
            tgt_sizes=tgt_sizes_flat,
            embed_is_patch=embed_is_patch,
            num_slices=num_slices_flat,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_vision_input(
                    "images", **kwargs)
            if input_key in ("video_pixel_values",
                             "video_embeds") and "videos" not in modalities:

                def _image_key(video_key: str):
                    if video_key == "video_token_id":
                        return "image_token_id"

                    return video_key.removeprefix("video_")

                modalities["videos"] = self._parse_and_validate_vision_input(
                    "videos", **{
                        _image_key(k): v
                        for k, v in kwargs.items()
                    })

        return modalities

    def _process_vision_input(
        self,
        image_input: MiniCPMVImageInputs,
    ) -> Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        image_features_flat = self.get_vision_hidden_states(image_input)

        num_slices = image_input["num_slices"]
        return [
            e.flatten(0, 1)
            for e in image_features_flat.split(num_slices.tolist())
        ]

    def _process_multimodal_inputs(self, modalities: dict):
        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_features = self._process_vision_input(image_input)
                multimodal_embeddings += tuple(
                    scatter_patch_features(
                        image_features,
                        image_input["embed_is_patch"],
                    ))
            if modality == "videos":
                video_input = modalities["videos"]
                video_features = self._process_vision_input(video_input)
                multimodal_embeddings += tuple(
                    scatter_patch_features(
                        video_features,
                        video_input["embed_is_patch"],
                    ))

        return multimodal_embeddings

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        return self._process_multimodal_inputs(modalities)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.llm.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            assert len(self.mm_token_ids) > 0
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                select_patch_features(multimodal_embeddings),
                list(self.mm_token_ids),
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

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
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.llm.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(language_model="llm",
                                                connector="resampler",
                                                tower_model="vpm")

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        raise NotImplementedError

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        raise NotImplementedError


class MiniCPMV2_0(MiniCPMVBaseModel):

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
        quant_config: Optional[QuantizationConfig],
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

        if (isinstance(model, timm.models.VisionTransformer)
                and model.attn_pool is not None):
            model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2(embed_dim=embed_dim,
                                   num_heads=embed_dim // 128,
                                   grid_size=int(
                                       math.sqrt(self.config.query_num)),
                                   kv_dim=vision_dim,
                                   adaptive=False,
                                   do_post_projection=True,
                                   quant_config=quant_config,
                                   prefix=prefix)

        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]

        P_h, P_w = self.vpm.patch_embed.patch_size
        dtype: torch.dtype = self.vpm.pos_embed.data.dtype
        num_prefix_tokens = getattr(self.vpm, "num_prefix_tokens", 0)

        res = list[torch.Tensor]()
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (math.ceil(H / P_h), math.ceil(W / P_w))
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))

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
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(config.vision_config,
                                          quant_config=quant_config,
                                          prefix=prefix)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2_5(num_queries=self.config.query_num,
                                     embed_dim=embed_dim,
                                     num_heads=embed_dim // 128,
                                     kv_dim=vision_dim,
                                     quant_config=quant_config,
                                     prefix=prefix)

        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L),
                                       dtype=dtype,
                                       device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches),
                                      dtype=torch.bool,
                                      device=device)
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
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(config.vision_config,
                                          quant_config=quant_config,
                                          prefix=prefix)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(num_queries=self.config.query_num,
                                     embed_dim=embed_dim,
                                     num_heads=embed_dim // 128,
                                     kv_dim=vision_dim,
                                     quant_config=quant_config,
                                     prefix=prefix)

        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L),
                                       dtype=dtype,
                                       device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)


_SUPPORT_VERSION = {
    (2, 0): MiniCPMV2_0,
    (2, 5): MiniCPMV2_5,
    (2, 6): MiniCPMV2_6,
}


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMVMultiModalProcessor,
    info=MiniCPMVProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder)
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
            raise ValueError(
                "Currently, MiniCPMV only supports versions 2.0, 2.5, and 2.6")

        # quant_config references base class members,
        # so update values before init is called
        cls.packed_modules_mapping.update(instance_cls.packed_modules_mapping)
        cls.embedding_modules.update(instance_cls.embedding_modules)
        cls.embedding_padding_modules += instance_cls.embedding_padding_modules
        return instance_cls(vllm_config=vllm_config, prefix=prefix)
