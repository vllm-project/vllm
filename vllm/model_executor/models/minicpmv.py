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
import re
from functools import cached_property, partial
from itertools import accumulate
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Set, Tuple, TypedDict, Union)

import numpy as np
import torch
import torch.types
from PIL import Image
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.whisper.modeling_whisper import (
    ACT2FN, WHISPER_ATTENTION_CLASSES, WhisperConfig, WhisperEncoder)

from vllm.attention import AttentionMetadata
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
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, PlaceholderRange)
from vllm.multimodal.parse import (ImageItem, ImageSize, ModalityData,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser, VideoItem)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix

CPU_DEVICE = torch.device("cpu")

RawImageType = Union[Image.Image, torch.Tensor]


class MiniCPMVImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: List[torch.Tensor]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """

    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class MiniCPMVImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(start, stop)` format.
    """


class MiniCPMVAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: List[torch.Tensor]
    """
    Shape: 
    """

    audio_feature_lens: torch.Tensor
    """
    Shape:
    """

    audio_bounds: torch.Tensor
    """
    Shape:
    """


class MiniCPMVAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: torch.Tensor
    """
    Shape:
    """
    audio_bounds: torch.Tensor
    """
    Shape:
    """


MiniCPMVImageInputs = Union[MiniCPMVImagePixelInputs,
                            MiniCPMVImageEmbeddingInputs]
MiniCPMVAudioInputs = Union[MiniCPMVAudioFeatureInputs,
                            MiniCPMVAudioEmbeddingInputs]


class MiniCPMVEmbeddingItems(ModalityDataItems[dict[str, torch.Tensor],
                                               dict[str, torch.Tensor]]):

    def __init__(self, data: Dict, modality: str) -> None:
        super().__init__(data, modality)

    def get_processor_data(self) -> Mapping[str, object]:
        return self.data

    def get_passthrough_data(self) -> Mapping[str, object]:
        return {}

    def get_count(self) -> int:
        return len(self.data[f"{self.modality}_embeds"])

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.data.items():
            out[k] = v[index]
        return out


class MiniCPMVImageEmbeddingItems(MiniCPMVEmbeddingItems):

    def __init__(self, data: Dict) -> None:
        super().__init__(data, "image")
        image_embeds = self.data.get("image_embeds", None)
        image_sizes = self.data.get("image_sizes", None)
        if image_embeds is None:
            raise ValueError("In correct type of image_embeds",
                             "Got type: None")
        if not isinstance(image_embeds[0], torch.Tensor):
            raise ValueError("In correct type of image_embeds",
                             f"Got type: {type(image_embeds[0])}")
        if image_sizes is None:
            raise ValueError(
                "In correct type of image_sizes", "Got type: None."
                "If you're using `image_size_list`, "
                "please rename it to `image_sizes`")
        if len(image_embeds[0].shape) == 2:
            image_embeds = [image_embeds]
            image_sizes = [image_sizes]
        self.data["image_embeds"] = image_embeds
        self.data["image_sizes"] = image_sizes

    def get_image_size(self, index: int) -> ImageSize:
        image_size = self.data["image_sizes"][index]
        return ImageSize(width=image_size[0], height=image_size[1])


class MiniCPMVVideoEmbeddingItems(MiniCPMVEmbeddingItems):

    def __init__(self, data: Dict) -> None:
        super().__init__(data, "video")
        video_embeds = self.data.get("video_embeds", None)
        image_sizes = self.data.get("image_sizes", None)
        num_frames = self.data.get("num_frames", None)
        if video_embeds is None:
            raise ValueError("In correct type of video_embeds",
                             "Got type: None")
        if not isinstance(video_embeds[0], torch.Tensor):
            raise ValueError("In correct type of video_embeds",
                             f"Got type: {type(video_embeds[0])}")
        if image_sizes is None:
            raise ValueError(
                "In correct type of image_sizes", "Got type: None."
                "If you're using `image_size_list`, "
                "please rename it to `image_sizes`")
        if num_frames is None:
            raise ValueError("In correct type of numframes", "Got type: None")
        if len(video_embeds[0].shape) == 2:
            video_embeds = [video_embeds]
            image_sizes = [image_sizes]
            num_frames = [num_frames]
        self.data["video_embeds"] = video_embeds
        self.data["image_sizes"] = image_sizes
        self.data["num_frames"] = num_frames

    def get_frame_size(self, index: int) -> ImageSize:
        frame_size = self.data["image_sizes"][index]
        return ImageSize(width=frame_size[0], height=frame_size[1])

    def get_num_frames(self, index: int) -> int:
        return self.data["num_frames"][index]


class MiniCPMVAudioEmbeddingItems(MiniCPMVEmbeddingItems):

    def __init__(self, data: Dict) -> None:
        super().__init__(data, "audio")
        audio_embeds = self.data.get("audio_embeds", None)
        if audio_embeds is None:
            raise ValueError("In correct type of video_embeds",
                             "Got type: None")
        self.data["audio_embeds"] = audio_embeds

    def get(self, index: int) -> object:
        return self.data["audio_embeds"][index]


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
    elif "MiniCPMO" in config.architectures:
        return (2, "6O")
    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


class MiniCPMVMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return MiniCPMVImageEmbeddingItems(data)
        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return MiniCPMVVideoEmbeddingItems(data)
        return super()._parse_video_data(data)

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return MiniCPMVAudioEmbeddingItems(data)
        return super()._parse_audio_data(data)


class MiniCPMVProcessingInfo(BaseProcessingInfo):
    image_pattern = "(<image>./</image>)"
    video_pattern = "(<video>./</video>)"
    audio_pattern = "(<audio>./</audio>)"

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(
        self,
        **kwargs: object,
    ):
        hf_processor = self.ctx.get_hf_processor()
        # image_processor = hf_processor.image_processor
        return hf_processor

    def get_image_processor(self, ):
        hf_processor = self.get_hf_processor()
        image_processor = hf_processor.image_processor  # type: ignore
        return image_processor

    def get_model_version(self):
        return get_version_by_config(self.get_hf_config())

    def get_supported_mm_modalities(self) -> List[str]:
        if self.get_model_version() == (2, "6O"):
            return ["image", "video", "audio"]
        elif self.get_model_version() == (2, 6):
            return ["image", "video"]
        else:
            return ["image"]

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        if self.get_model_version() == (2, "6O"):
            return {"image": None, "video": None, "audio": None}
        if self.get_model_version() == (2, 6):
            return {"image": None, "video": None}
        else:
            return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        mm_max_tokens = {"image": self.get_max_image_tokens()}
        if self.get_model_version() == (2, "6O"):
            mm_max_tokens["audio"] = self.get_max_audio_tokens()
        if self.get_model_version() in [(2, 6), (2, "6O")]:
            mm_max_tokens["video"] = self.get_max_video_tokens(seq_len)
        return mm_max_tokens

    def get_max_video_frame_tokens(self) -> int:
        frame_size = self.get_video_frame_size_with_most_features()
        return self.get_num_image_tokens(frame_size,
                                         self.get_video_max_slice_num())

    def get_max_video_tokens(self, seq_len: int) -> int:
        return self.get_max_video_frame_tokens(
        ) * self.get_num_frames_with_most_features(seq_len)

    def get_max_audio_tokens(self) -> int:
        return self.get_max_audio_tokens_per_chunk(
        ) * self.get_max_audio_chunks_with_most_features()

    def get_slice_query_num(self) -> int:
        hf_config = self.get_hf_config()
        query_num = getattr(hf_config, "query_num", 64)
        return query_num

    def get_max_slice_num(self) -> int:
        hf_config = self.get_hf_config()
        max_slice_num = getattr(hf_config, "max_slice_num", 9)
        return max_slice_num

    def get_sliced_grid(self, image_size, max_slice_num) -> Tuple[int, int]:
        if self.get_model_version() in [(2, 6), (2, "6O")]:
            slice_grid = self.get_image_processor().get_sliced_grid(
                image_size, max_slice_num)
        else:
            slice_grid = self.get_image_processor().get_sliced_grid(image_size)
        return slice_grid

    def get_num_image_tokens(self, image_size: ImageSize,
                             max_slice_num: int) -> int:
        slice_grid = self.get_sliced_grid(image_size, max_slice_num)
        num_tokens = self.get_slice_query_num(
        ) + 2  # <image>(<unk> * query_num)</image>
        if slice_grid is not None:
            if self.get_model_version() in [(2, 6), (2, "6O")]:
                num_additional_tokens = 0
            else:
                # <slice><image>(<unk> * query_num)</image></slice>
                num_additional_tokens = 2
            num_tokens += ((self.get_slice_query_num() + 2) \
                            * slice_grid[0] * slice_grid[1]) \
                            + slice_grid[1] - 1 + num_additional_tokens
        return num_tokens

    def get_image_slice_nums(self, image_size: torch.Tensor, max_slice_nums):
        grid = self.get_sliced_grid(image_size, max_slice_nums)
        return 1 if grid is None else grid[0] * grid[1] + 1

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_size, self.get_max_slice_num())

    def get_image_size_with_most_features(self) -> ImageSize:
        # Result in the max possible feature size (h:w = 9:1)
        return self.get_defaul_image_sizes(self.get_max_slice_num())

    def get_video_max_slice_num(self) -> int:
        return 1

    def get_video_frame_size_with_most_features(self) -> ImageSize:
        return self.get_defaul_image_sizes(self.get_video_max_slice_num())

    def get_max_video_frames(self, max_tokens: int) -> int:
        num_frame_tokens = self.get_max_video_frame_tokens()
        num_frames = max_tokens // num_frame_tokens
        return num_frames

    def get_num_frames_with_most_features(self, seq_len: int) -> int:
        mm_config = self.ctx.get_mm_config()
        max_images = mm_config.limit_per_prompt.get("image", 1)
        max_videos = mm_config.limit_per_prompt.get("video", 1)
        max_audios = mm_config.limit_per_prompt.get("audio", 1)

        # count <image_idx></image_idx> tokens
        # which are not in get_max_image_tokens
        max_image_tokens = self.get_max_image_tokens(
        ) * max_images + 4 * max_images
        seq_len = seq_len - max_image_tokens
        if "audio" in self.get_supported_mm_modalities():
            max_audio_tokens = self.get_max_audio_tokens(
            ) * max_audios + 2 * max_audios
            seq_len = seq_len - max_audio_tokens
        max_total_frames = self.get_max_video_frames(seq_len)

        num_frames = max(max_total_frames // max(max_videos, 1), 1)

        return num_frames

    def get_defaul_image_sizes(self, num_slices: int) -> ImageSize:
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        return ImageSize(width=image_size, height=image_size * num_slices)

    def get_default_audio_pool_step(self):
        return 2

    def get_default_audio_sampling_rate(self):
        return 16000

    def get_chunk_length(self) -> int:
        return self.get_hf_config().audio_chunk_length

    def get_max_audio_tokens_per_chunk(self) -> int:
        pool_step = self.get_default_audio_pool_step()
        fbank_feat_in_chunk = 100
        cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
        num_audio_tokens = (cnn_feat_in_chunk - pool_step) // pool_step + 1
        return num_audio_tokens + 2  # <audio>(<unk>*N)</audio>

    def get_max_audio_chunks_with_most_features(self) -> int:
        return 30

    def get_audio_len_by_num_chunks(self, num_chunks: int) -> int:
        sampling_rate = self.get_default_audio_sampling_rate()
        # exclude <audio> </audio>
        num_tokens_per_chunk = self.get_max_audio_tokens_per_chunk() - 2
        return int(num_chunks * sampling_rate / num_tokens_per_chunk) + 1


class MiniCPMVDummyInputsBuilder(BaseDummyInputsBuilder[MiniCPMVProcessingInfo]
                                 ):

    def get_dummy_processor_inputs(
            self, seq_len: int, mm_counts: Mapping[str,
                                                   int]) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        image_width, image_height = \
            self.info.get_image_size_with_most_features()
        video_width, video_height = \
            self.info.get_video_frame_size_with_most_features()
        num_video_frames = \
            self.info.get_num_frames_with_most_features(seq_len)

        audio_len = self.info.get_max_audio_chunks_with_most_features() * \
            self.info.get_default_audio_sampling_rate()

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
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }

        image_prompt_texts = self.info.image_pattern * num_images
        video_prompt_texts = self.info.video_pattern * num_videos
        audio_prompt_texts = self.info.audio_pattern * num_audios

        return ProcessorInputs(prompt_text=image_prompt_texts +
                               video_prompt_texts + audio_prompt_texts,
                               mm_data=mm_data)


class MiniCPMVMultiModalProcessor(
        BaseMultiModalProcessor[MiniCPMVProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMVMultiModalDataParser()

    def get_slice_image_placeholder(self, image_size: ImageSize,
                                    **kwargs) -> str:
        image_processor = self.info.get_image_processor()
        version = self.info.get_model_version()
        if version == (2, 0) or version == (2, 5):
            return image_processor.get_slice_image_placeholder(image_size)
        return image_processor.get_slice_image_placeholder(
            image_size, **kwargs)

    def get_image_prompt_texts(self,
                               image_size: ImageSize,
                               image_idx: int = 0) -> str:
        prompt_texts = self.get_slice_image_placeholder(image_size,
                                                        image_idx=image_idx)
        return prompt_texts

    def get_video_prompt_texts(self, image_size: ImageSize,
                               num_frames: int) -> str:
        prompt_texts = "".join([
            self.get_slice_image_placeholder(
                image_size=image_size,
                image_idx=0,
                max_slice_nums=self.info.get_video_max_slice_num(),
                use_image_id=False) for image_idx in range(num_frames)
        ])
        return prompt_texts

    def get_audio_prompt_texts(self,
                               audio_lens: int,
                               chunk_input: bool = True,
                               chunk_length: int = 1):
        return self.info.get_hf_processor().get_audio_placeholder(
            audio_lens, chunk_input, chunk_length)

    def get_special_tokens(self):
        tokenizer = self.info.get_tokenizer()
        special_tokens = {
            "im_start_id": torch.tensor(tokenizer.im_start_id),
            "im_end_id": torch.tensor(tokenizer.im_end_id)
        }
        if hasattr(tokenizer, "slice_start_id"):
            special_tokens["slice_start_id"] = torch.tensor(
                tokenizer.slice_start_id)
            special_tokens["slice_end_id"] = torch.tensor(
                tokenizer.slice_end_id)
        if hasattr(tokenizer, "audio_start_id"):
            special_tokens["audio_start_id"] = torch.tensor(
                tokenizer.audio_start_id)
            special_tokens["audio_end_id"] = torch.tensor(
                tokenizer.audio_end_id)
        return special_tokens

    @staticmethod
    def repack_processor_outputs(outputs: Any) -> BatchFeature:
        valid_keys = ["pixel_values", "image_sizes", "tgt_sizes"]
        outputs = {key: outputs[key][0] for key in valid_keys}
        return outputs

    def process_images(self, mm_data: Mapping[str, object],
                       mm_kwargs: Mapping[str, object]) -> Dict[str, object]:
        images = mm_data.pop("images", [])
        image_embeds = mm_data.pop("image_embeds", [])
        if isinstance(images, (list, torch.Tensor)) and len(images) > 0:
            image_outputs = super()._call_hf_processor(
                prompt=self.info.image_pattern * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs)
            image_outputs = MiniCPMVMultiModalProcessor.\
                repack_processor_outputs(image_outputs)
        elif len(image_embeds) > 0:
            image_sizes = mm_data.pop("image_sizes", None)
            image_outputs = {
                "image_embeds": torch.cat(image_embeds),
                "image_sizes": image_sizes
            }
        else:
            image_outputs = {}
        return image_outputs

    def process_videos(self, mm_data: Mapping[str, object],
                       mm_kwargs: Mapping[str, object]):
        videos = mm_data.pop("videos", [])
        video_embeds = mm_data.pop("video_embeds", [])
        if len(videos) > 0 and isinstance(videos[0], Image.Image):
            videos = [videos]
        if isinstance(videos, list) and len(videos) > 0:
            video_outputs = {
                "video_pixel_values": [],
                "video_image_sizes": [],
                "video_tgt_sizes": [],
                "num_frames": []
            }
            for video in videos:
                single_video_outputs = super()._call_hf_processor(
                    prompt=self.info.image_pattern * len(video),
                    mm_data={"images": video},
                    mm_kwargs={
                        **mm_kwargs, "max_slice_nums":
                        self.info.get_video_max_slice_num()
                    })
                video_outputs["num_frames"].append(len(video))
                for key in single_video_outputs:
                    if "video_" + key in video_outputs:
                        if key == "image_sizes":
                            video_outputs["video_" + key].append(
                                single_video_outputs[key][0][0])
                        else:
                            video_outputs["video_" +
                                          key] += single_video_outputs[key][0]
        elif len(video_embeds):
            image_sizes = mm_data.pop("image_sizes", None)
            num_frames = mm_data.pop("num_frames", None)
            video_outputs = {
                "video_embeds": torch.cat(video_embeds),
                "video_image_sizes": image_sizes,
                "num_frames": num_frames
            }
        else:
            video_outputs = {}
        return video_outputs

    def process_audios(self, mm_data: Mapping[str, object],
                       mm_kwargs: Mapping[str, object]):
        audios = mm_data.pop("audios", [])
        audio_embeds = mm_data.pop("audio_embeds", [])
        if isinstance(audios, (list, torch.Tensor)) and len(audios) > 0:
            audio_outputs = {
                "audio_lens": [],
                "audio_features": [],
                "audio_feature_lens": [],
                "audio_num_segments": []
            }
            for audio in audios:
                single_audio_outputs = super()._call_hf_processor(
                    prompt=self.info.audio_pattern,
                    mm_data={
                        "audios": audio,
                        "chunk_input": True
                    },
                    mm_kwargs=mm_kwargs)
                audio_outputs["audio_lens"].append(len(audio))
                audio_outputs["audio_features"].append(
                    single_audio_outputs["audio_features"])
                audio_outputs["audio_num_segments"].append(
                    len(single_audio_outputs["audio_feature_lens"][0]))
                audio_outputs["audio_feature_lens"] += \
                    single_audio_outputs["audio_feature_lens"]
            audio_outputs["audio_features"] = torch.cat(
                audio_outputs["audio_features"])
            audio_outputs["audio_feature_lens"] = torch.cat(
                audio_outputs["audio_feature_lens"])
        elif len(audio_embeds):
            audio_outputs = {
                "audio_lens": [
                    self.info.get_audio_len_by_num_chunks(
                        sum(chunk_embeds.shape[0]
                            for chunk_embeds in single_audio_embeds))
                    for single_audio_embeds in audio_embeds
                ],
                "audio_embeds": [
                    chunk_embeds for single_audio_embeds in audio_embeds
                    for chunk_embeds in single_audio_embeds
                ],
                "audio_num_segments": [
                    len(single_audio_embeds)
                    for single_audio_embeds in audio_embeds
                ]
            }
        else:
            audio_outputs = {}
        return audio_outputs

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Do not support combination inputs of images and videos for now
        # Try to handle interleaved multimodal data
        tokenizer = self.info.get_tokenizer()
        image_outputs = self.process_images(mm_data, mm_kwargs)
        video_outputs = self.process_videos(mm_data, mm_kwargs)
        audio_outputs = self.process_audios(mm_data, mm_kwargs)
        counts = {"image": 0, "video": 0, "audio": 0}
        num_image_slices = []
        num_video_slices = []
        num_audio_slices = []
        video_orders_in_mm_data = []
        image_orders_in_mm_data = []
        audio_orders_in_mm_data = []
        matches = re.findall(r"\(<(image|video|audio)>./</\1>\)", prompt)
        chunks = re.split(
            r"\(<(?:image|video|audio)>./</(?:image|video|audio)>\)", prompt)
        new_prompt = chunks[0]
        for idx, item in enumerate(matches):
            if item == "image":
                image_orders_in_mm_data.append(idx)
                num_image_slices.append(
                    self.info.get_image_slice_nums(
                        image_outputs["image_sizes"][counts[item]],
                        self.info.get_max_slice_num()))
                new_prompt += self.get_image_prompt_texts(
                    image_outputs["image_sizes"][counts[item]], counts[item])
            elif item == "video":
                video_orders_in_mm_data.append(idx)
                num_video_slices.append(
                    self.info.get_image_slice_nums(
                        video_outputs["video_image_sizes"][counts[item]],
                        self.info.get_video_max_slice_num()) *
                    video_outputs["num_frames"][counts[item]])
                new_prompt += self.get_video_prompt_texts(
                    video_outputs["video_image_sizes"][counts[item]],
                    video_outputs["num_frames"][counts[item]])
            else:  # audio
                audio_orders_in_mm_data.append(idx)
                num_audio_slices.append(
                    audio_outputs["audio_num_segments"][counts[item]])
                new_prompt += self.get_audio_prompt_texts(
                    audio_outputs["audio_lens"][counts[item]])

            counts[item] += 1
            new_prompt += chunks[idx + 1]

        input_ids = tokenizer.encode(new_prompt)

        def get_slices(num_slices: List[int]):
            slice_idices = [0] + list(accumulate(num_slices))
            slices = [(slice_idices[i], slice_idices[i + 1])
                      for i in range(len(num_slices))]
            return slices

        return {
            "input_ids": np.array([input_ids]),
            **image_outputs,
            **video_outputs,
            **audio_outputs,
            "image_orders_in_mm_data": image_orders_in_mm_data,
            "image_slices": get_slices(num_image_slices),
            "video_orders_in_mm_data": video_orders_in_mm_data,
            "video_slices": get_slices(num_video_slices),
            "audio_orders_in_mm_data": audio_orders_in_mm_data,
            "audio_slices": get_slices(num_audio_slices),
        }

    def _get_prompt_replacements(
            self, mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: MultiModalKwargs) -> List[PromptReplacement]:
        placeholder = {
            "image": self.info.image_pattern,
            "video": self.info.video_pattern,
            "audio": self.info.audio_pattern
        }

        def get_replacement_minicpmv(item_idx: int, modality: str):
            if modality == "image":
                return self.get_image_prompt_texts(
                    mm_items["image"].get_image_size(item_idx), item_idx)
            elif modality == "video":
                return self.get_video_prompt_texts(
                    mm_items["video"].get_frame_size(item_idx),
                    mm_items["video"].get_num_frames(item_idx))
            else:  # audio
                if isinstance(mm_items["audio"], MiniCPMVAudioEmbeddingItems):
                    single_audio_embeds = mm_items["audio"].get(item_idx)
                    audio_len = self.info.get_audio_len_by_num_chunks(
                        sum(chunk_embeds.shape[0]
                            for chunk_embeds in single_audio_embeds))
                    return self.get_audio_prompt_texts(audio_len)
                return self.get_audio_prompt_texts(
                    len(mm_items["audio"].get(item_idx)))

        return [
            PromptReplacement(modality=modality,
                              target=placeholder[modality],
                              replacement=partial(get_replacement_minicpmv,
                                                  modality=modality))
            for modality in ("image", "video", "audio")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:

        def get_slices(slices_indices: List[int]):
            return [slice(*slice_item) for slice_item in slices_indices]

        image_slices = get_slices(
            hf_inputs.get("image_slices", torch.empty(0, 2)))
        video_slices = get_slices(
            hf_inputs.get("video_slices", torch.empty(0, 2)))
        audio_slices = get_slices(
            hf_inputs.get("audio_slices", torch.empty(0, 2)))
        return dict(
            pixel_values=MultiModalFieldConfig.flat("image", image_slices),
            image_sizes=MultiModalFieldConfig.batched("image"),
            tgt_sizes=MultiModalFieldConfig.flat("image", image_slices),
            image_slices=MultiModalFieldConfig.batched("image"),
            image_orders_in_mm_data=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.flat("image", image_slices),
            video_pixel_values=MultiModalFieldConfig.flat(
                "video", video_slices),
            video_image_sizes=MultiModalFieldConfig.batched("video"),
            video_tgt_sizes=MultiModalFieldConfig.flat("video", video_slices),
            video_orders_in_mm_data=MultiModalFieldConfig.batched("video"),
            video_embeds=MultiModalFieldConfig.flat("video", video_slices),
            video_slices=MultiModalFieldConfig.batched("video"),
            audio_features=MultiModalFieldConfig.flat("audio", audio_slices),
            audio_feature_lens=MultiModalFieldConfig.flat(
                "audio", audio_slices),
            audio_slices=MultiModalFieldConfig.batched("audio"),
            audio_orders_in_mm_data=MultiModalFieldConfig.batched("audio"),
            audio_embeds=MultiModalFieldConfig.flat("audio", audio_slices))

    def apply(
        self,
        prompt: Union[str, List[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        result = super().apply(prompt, mm_data, hf_processor_mm_kwargs)

        # Exclude <image_id>x</image_id> from placeholders
        if "image" in result["mm_placeholders"] and \
            self.info.get_model_version() in [(2, 6), (2, "6O")]:
            result["mm_placeholders"]["image"] = [
                PlaceholderRange(offset=p["offset"] + 3 + idx // 10,
                                 length=p["length"] - 3 - idx // 10)
                for idx, p in enumerate(result["mm_placeholders"]["image"])
            ]
        result["mm_kwargs"].update(**self.get_special_tokens())
        return result


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

        self.make_empty_intermediate_tensors = (
            self.llm.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.llm, "sampler"):
            return self.llm.sampler

        return get_sampler()

    def get_embedding_with_vision(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[MiniCPMVImageInputs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vlm_embedding: torch.Tensor = self.llm.get_input_embeddings(input_ids)

        if image_inputs is None:  # No image
            vision_hidden_states = torch.tensor([], device=input_ids.device)
        else:
            if image_inputs["type"] == "image_embeds":
                vision_hidden_states = (image_inputs["data"].type(
                    vlm_embedding.dtype).to(vlm_embedding.device))
            else:
                vision_hidden_states = self.get_vision_hidden_states(
                    image_inputs)

            # See NOTE in _parse_and_validate_inputs
            image_bounds = image_inputs["image_bounds"]
            if len(image_bounds) > 0:
                image_indices = torch.stack([
                    torch.arange(start, end, dtype=torch.long)
                    for start, end in image_bounds.tolist()
                ]).to(vlm_embedding.device)
                vlm_embedding.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1,
                                                     vlm_embedding.shape[-1]),
                    vision_hidden_states.view(-1,
                                              vision_hidden_states.shape[-1]),
                )

        return vlm_embedding, vision_hidden_states

    def get_embedding_with_audios(self, vlm_embedding: torch.Tensor,
                                  audio_inputs: Optional[MiniCPMVAudioInputs],
                                  chunk_length: int) -> torch.Tensor:
        device, dtype = vlm_embedding.device, vlm_embedding.dtype
        if audio_inputs["type"] == "audio_embeds":
            audio_embeddings = audio_inputs["data"]
            audio_embeddings = [
                audio_embeddings[i].to(device=device, dtype=dtype)
                for i in range(len(audio_embeddings))
            ]
        else:
            audio_embeddings = self.get_audio_hidden_states(
                audio_inputs, chunk_length)[0]
        if audio_embeddings is None or len(audio_embeddings) == 0:
            return vlm_embedding
        audio_bounds = audio_inputs["audio_bounds"]
        if self.config.chunk_input:
            audio_embs = torch.cat(audio_embeddings, dim=0).to(device=device,
                                                               dtype=dtype)
            audio_start_pos = 0
            for bound in audio_bounds:
                audio_len = bound[1] - bound[0]
                vlm_embedding[bound[0]:bound[1]] = audio_embs[
                    audio_start_pos:audio_start_pos + audio_len, :]
                audio_start_pos += audio_len
        else:
            for embs, bound in zip(audio_embeddings, audio_bounds):
                audio_indices = torch.arange(bound[0],
                                             bound[1],
                                             dtype=torch.long).to(device)

                if embs.shape[0] != len(audio_indices):
                    raise ValueError(
                        "Shape mismatch: Trying to assign embeddings "
                        f"of shape {embs.shape} "
                        f"to input indices of length {len(audio_indices)}")
                vlm_embedding[audio_indices] = embs.to(dtype)
        return vlm_embedding

    def _get_image_bounds(
            self,
            input_ids: torch.Tensor,
            im_start_id: torch.Tensor,
            im_end_id: torch.Tensor,
            slice_start_id: Optional[torch.Tensor] = None,
            slice_end_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        # All the images in the batch should share the same special image
        # bound token ids.
        start_cond = input_ids == im_start_id[0]
        end_cond = input_ids == im_end_id[0]
        if slice_start_id is not None:
            start_cond |= (input_ids == slice_start_id[0])
            end_cond |= (input_ids == slice_end_id[0])

        image_start_tokens, = torch.where(start_cond)
        image_start_tokens += 1
        image_end_tokens, = torch.where(end_cond)
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        if valid_image_nums == 0:
            return torch.zeros((0, 2), device=input_ids.device)

        return torch.hstack([
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ])

    def _get_audio_bounds(self, input_ids: torch.Tensor,
                          audio_start_id: torch.Tensor,
                          audio_end_id: torch.Tensor) -> torch.Tensor:
        audio_start_tokens, = torch.where(input_ids == audio_start_id[0])
        audio_start_tokens += 1
        audio_end_tokens, = torch.where(input_ids == audio_end_id[0])
        valid_audio_nums = max(len(audio_start_tokens), len(audio_end_tokens))
        return torch.hstack([
            audio_start_tokens[:valid_audio_nums].unsqueeze(-1),
            audio_end_tokens[:valid_audio_nums].unsqueeze(-1)
        ])

    def _parse_and_validate_image_inputs(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        mm_data = {
            "image": {
                key: kwargs.pop(key, [])
                for key in ["pixel_values", "tgt_sizes", "image_slices"]
            },
            "video": {
                "pixel_values": kwargs.pop("video_pixel_values", []),
                "tgt_sizes": kwargs.pop("video_tgt_sizes", []),
                "video_slices": kwargs.pop("video_slices", [])
            }
        }
        im_start_id = kwargs.pop("im_start_id", None)
        im_end_id = kwargs.pop("im_end_id", None)
        slice_start_id = kwargs.pop("slice_start_id", None)
        slice_end_id = kwargs.pop("slice_end_id", None)
        orders_in_mm_data = {
            modality: kwargs.pop(f"{modality}_orders_in_mm_data", None)
            for modality in ["image", "video", "audio"]
        }
        batch_size = max(len(mm_data["image"]["pixel_values"]),
                         len(mm_data["video"]["pixel_values"]))
        image_embeds = kwargs.pop("image_embeds", None)
        video_embeds = kwargs.pop("video_embeds", None)
        if image_embeds is not None and video_embeds is not None:
            raise ValueError(
                "Incorrect inputs for vision embeddings. "
                "Image embeds and video embeds can not exist simultaneously.")
        if video_embeds is not None:
            image_embeds = video_embeds
        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")
            image_embeds = torch.concat(
                [image_embeds[i] for i in range(len(image_embeds))])

            return MiniCPMVImageEmbeddingInputs(
                image_bounds=self._get_image_bounds(input_ids, im_start_id,
                                                    im_end_id, slice_start_id,
                                                    slice_end_id),
                data=image_embeds,
                type="image_embeds",
            )

        for modality, modality_mm_data in mm_data.items():
            if not isinstance(modality_mm_data["pixel_values"],
                              (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. "
                    f"Got type: {type(modality_mm_data['pixel_values'])}")

            if not isinstance(modality_mm_data["tgt_sizes"],
                              (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of target sizes. "
                    f"Got type: {type(modality_mm_data['tgt_sizes'])}")

            if len(modality_mm_data["pixel_values"]) != len(
                    modality_mm_data["tgt_sizes"]):
                raise ValueError(
                    "Inconsistent batch lengths, found: "
                    f"{len(modality_mm_data['pixel_values'])} vs. "
                    f"{len(modality_mm_data['tgt_sizes'])}")

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for b in range(batch_size):
            orders_in_mm_data_b = {
                modality: orders[b] if orders is not None else []
                for modality, orders in orders_in_mm_data.items()
            }
            mm_data_indices = [
                (index, (pos, media_type))
                for media_type in ["image", "video", "audio"]
                for pos, index in enumerate(orders_in_mm_data_b[media_type])
            ]
            mm_data_indices = [(pos, modality) for index, (
                pos, modality) in sorted(mm_data_indices, key=lambda x: x[0])]
            for pos, modality in mm_data_indices:
                if modality == "image":
                    slice_index = mm_data[modality]["image_slices"][b][pos]
                    pixel_values_flat += mm_data[modality]["pixel_values"][b][
                        slice_index[0]:slice_index[1]]
                    tgt_sizes_flat += mm_data[modality]["tgt_sizes"][b][
                        slice_index[0]:slice_index[1]]
                elif modality == "video":
                    slice_index = mm_data[modality]["video_slices"][b][pos]
                    pixel_values_flat += mm_data[modality]["pixel_values"][b][
                        slice_index[0]:slice_index[1]]
                    tgt_sizes_flat += mm_data[modality]["tgt_sizes"][b][
                        slice_index[0]:slice_index[1]]

        # NOTE: Input IDs does not contain image tokens during memory profiling,
        # so we allow it to be empty
        if len(pixel_values_flat) != len(tgt_sizes_flat):
            raise ValueError("Inconsistent flattened lengths, found: "
                             f"{len(pixel_values_flat)} vs. "
                             f"{len(tgt_sizes_flat)}")

        if len(pixel_values_flat) == 0:
            return None

        if im_start_id is None:
            return None

        return MiniCPMVImagePixelInputs(
            image_bounds=self._get_image_bounds(input_ids, im_start_id,
                                                im_end_id, slice_start_id,
                                                slice_end_id),
            data=pixel_values_flat,
            tgt_sizes=torch.stack(tgt_sizes_flat),
            type="pixel_values",
        )

    def _parse_and_validate_audio_inputs(
            self, input_ids: torch.Tensor,
            **kwargs: object) -> Tuple[MiniCPMVImageInputs]:
        audio_features = kwargs.pop("audio_features", [])
        audio_feature_lens = kwargs.pop("audio_feature_lens", [])
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_start_id = kwargs.pop("audio_start_id", None)
        audio_end_id = kwargs.pop("audio_end_id", None)
        if audio_embeds is not None:
            audio_embeds = [
                audio_embeds[i][j] for i in range(len(audio_embeds))
                for j in range(len(audio_embeds[i]))
            ]
            return MiniCPMVAudioEmbeddingInputs(
                audio_bounds=self._get_audio_bounds(input_ids, audio_start_id,
                                                    audio_end_id),
                data=audio_embeds,
                type="audio_embeds")
        if len(audio_features) > 0:
            audio_features = torch.cat([item for item in audio_features])
            audio_feature_lens = torch.cat(
                [item for item in audio_feature_lens])

            return MiniCPMVAudioFeatureInputs(
                audio_bounds=self._get_audio_bounds(input_ids, audio_start_id,
                                                    audio_end_id),
                data=audio_features,
                audio_feature_lens=audio_feature_lens,
                type="audio_features")
        return None

    def _parse_and_validate_inputs(self, input_ids: torch.Tensor,
                                   **kwargs: object):
        image_inputs = self._parse_and_validate_image_inputs(
            input_ids, **kwargs)
        if not any("audio" in key for key in kwargs):
            return image_inputs, None
        audio_inputs = self._parse_and_validate_audio_inputs(
            input_ids, **kwargs)
        return image_inputs, audio_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            vlm_embeddings = None
        else:
            image_inputs, audio_inputs = \
                self._parse_and_validate_inputs(input_ids, **kwargs)
            vlm_embeddings, _ = self.get_embedding_with_vision(
                input_ids, image_inputs)

            if audio_inputs is not None:
                vlm_embeddings = self.get_embedding_with_audios(
                    vlm_embeddings, audio_inputs,
                    self.config.audio_chunk_length)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        output = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=vlm_embeddings,
        )
        return output

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

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        raise NotImplementedError

    def get_audio_hidden_states(self, data: MiniCPMVAudioInputs,
                                chunk_length: int) -> torch.Tensor:
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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

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

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (
                math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                math.ceil(W / self.vpm.patch_embed.patch_size[0]),
            )
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))
            if (hasattr(self.vpm, "num_prefix_tokens")
                    and self.vpm.num_prefix_tokens > 0):
                vision_embedding = vision_embedding[:, self.vpm.
                                                    num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))
        return torch.vstack(res)

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]

        return self.get_vision_embedding(pixel_values)


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
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []

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

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(pixel_values,
                                    patch_attention_mask=patch_attn_mask)
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        return self.get_vision_embedding(all_pixel_values.type(dtype),
                                         patch_attn_mask, tgt_sizes)


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
    # LoRA specific attributes
    supported_lora_modules = [
        # vision encoder
        "fc1",
        "fc2",
        "out_proj",
        # language model
        "qkv_proj",  # same name with vision encoder
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # resampler
        "kv_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version in [(2, 6), (2, "6O")]

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

        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)


class MultiModalProjector(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim,
                                 out_features=out_dim,
                                 bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim,
                                 out_features=out_dim,
                                 bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMWhisperEncoderLayer(nn.Module):

    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[
            config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.activation_dropout,
                                              training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any()
                or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        if use_cache:
            outputs += (past_key_values, )

        return outputs


class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            MiniCPMWhisperEncoderLayer(config, layer_idx=i)
            for i in range(config.encoder_layers)
        ])

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype,
                                           device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(),
                                                      DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(
                    DynamicCache.from_legacy_cache(past_key_values),
                    DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values,
                                                      DynamicCache())
            else:
                pass
            past_key_values_length = \
                past_key_values.self_attention_cache.get_usable_length(
                    inputs_embeds.shape[1]
                )
            if inputs_embeds.shape[
                    1] + past_key_values_length > embed_pos.shape[0]:
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat((
                    embed_pos_front,
                    torch.repeat_interleave(
                        embed_pos[-1, :].unsqueeze(0),
                        inputs_embeds.shape[1] - embed_pos.shape[0] +
                        past_key_values_length,
                        dim=0,
                    ),
                ))
            else:
                embed_pos = embed_pos[
                    past_key_values_length:inputs_embeds.shape[1] +
                    past_key_values_length, :]
        else:
            embed_pos = embed_pos[:inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), \
            f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}." # noqa

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx]
                                         if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_encoder_cache = layer_outputs[
                    2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )


class MiniCPMO2_6(MiniCPMV2_6):
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
        self.apm = self.init_audio_module(vllm_config=vllm_config,
                                          prefix=maybe_prefix(prefix, "apm"))

    def init_audio_module(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Do not use parameters temporarily
        audio_config = self.config.audio_config
        model = MiniCPMWhisperEncoder(audio_config)
        audio_output_dim = int(audio_config.encoder_ffn_dim // 4)
        self.audio_avg_pooler = \
            nn.AvgPool1d(self.config.audio_pool_step,
                         stride=self.config.audio_pool_step)
        self.audio_projection_layer = \
            MultiModalProjector(in_dim=audio_output_dim,out_dim=self.embed_dim)
        self.audio_encoder_layer = -1
        return model

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["tts"])
        return loader.load_weights(weights)

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = CPU_DEVICE,
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size,
                            0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead,
                         size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self,
                                         input_lengths: torch.LongTensor):
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn -
            self.config.audio_pool_step) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(
            dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    # Copied from HF repo of MiniCPM-o-2_6,
    # designed for batched inputs and outputs
    def get_audio_hidden_states(self, data: MiniCPMVAudioInputs,
                                chunk_length: int) -> torch.Tensor:
        wavforms = data.get(
            "data",
            [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = [data.get("audio_feature_lens",
                                           [])]  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (torch.arange(
                0,
                max_seq_len,
                dtype=audio_feature_lens.dtype,
                device=audio_feature_lens.device).unsqueeze(0).expand(
                    batch_size, max_seq_len))
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(
                batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(
                batch_size, 1, 1, max_seq_len).expand(batch_size, 1,
                                                      max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype,
                device=self.apm.conv1.weight.device)

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(
                    audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms,
                output_hidden_states=True,
                attention_mask=audio_attention_mask).hidden_states[
                    self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = \
                self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(
                        audio_embeds[idx, :num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []


_SUPPORT_VERSION = {
    (2, 0): MiniCPMV2_0,
    (2, 5): MiniCPMV2_5,
    (2, 6): MiniCPMV2_6,
    (2, "6O"): MiniCPMO2_6,
}


@MULTIMODAL_REGISTRY.register_processor(MiniCPMVMultiModalProcessor,
                                        info=MiniCPMVProcessingInfo,
                                        dummy_inputs=MiniCPMVDummyInputsBuilder
                                        )
class MiniCPMV(MiniCPMVBaseModel, SupportsMultiModal, SupportsLoRA):
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in vLLM. Therefore, it is necessary to separate them.
    """
    # Ensure that the LoRA support check passes when the class is not
    # initialized, but set all these attributes to empty.
    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if not hasattr(config, "version"):
            if config.hidden_size == 2304 and config.query_num == 64:
                version = (2, 0)
            else:
                version = (2, 5)
        elif "MiniCPMO" in config.architectures:
            version = (2, "6O")
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version
        instance_class = _SUPPORT_VERSION.get(version)
        if instance_class is None:
            raise ValueError(
                "Currently, MiniCPMV only supports versions 2.0, 2.5, and 2.6")
        return instance_class(vllm_config=vllm_config, prefix=prefix)
