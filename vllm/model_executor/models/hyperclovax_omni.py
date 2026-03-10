# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Literal, Optional, TypeAlias

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers import (
    AutoModel, AutoProcessor,
    BatchFeature, CLIPVisionConfig, PretrainedConfig, SiglipVisionConfig,
)
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.video_processing_utils import BaseVideoProcessor

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLVisionConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize, 
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


def get_compute_capability(
    device_index: int = 0,
):
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    cc_version = float(f'{major}.{minor}')
    return cc_version

class HyperCLOVAXOmniAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - nb: Number of samples
        - na: Number of audio
        - nc: Number of audio chunks
        - nm: Number of mel bins
        - ns: Number of max sequence length
        - nf: Number of max nb frames
        - lc: Length of code
    """

    type: Literal["audio_values"] = "audio_values"
    
    audio_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "nc", "nm", "nf", dynamic_dims={"nc"}),
    ]
    audio_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "nc", 1, "ns", "ns", dynamic_dims={"nc"}),
    ]
    audio_masks: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "nc", "nf", dynamic_dims={"nc"}),
    ]
    num_audio_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "na"),
    ]
    discrete_audio_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "lc", dynamic_dims={"lc"}),
    ]
    num_discrete_audio_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "na"),
    ]

class HyperCLOVAXOmniAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audio features
        - hs: Hidden size
        - nv: Number of videos

    Historical context:
        - audio_embeddings shape: (num_audio_features, hidden_size)
        - num_audio_features varies based on the number and length of audios.
        - hidden_size must match the hidden size of language model backbone.
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["audio_embeds"]

    audio_embeddings: Annotated[
        torch.Tensor,
        TensorShape("na", "hs"),
    ]

class HyperCLOVAXOmniImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - nb: Number of samples
        - ni: Number of images
        - np: Number of patches
        - nc: Number of channels
        - cps: Number of channels * patch_size * patch_size
        - ih: Image height
        - iw: Image width
    
    Historical context:
        - pixel_values shape: (num_patches, num_channels * patch_size *
          patch_size)
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format.
        - discrete_pixel_values shape: (num_images, 3, image_height, image_width)
        - discrete_image_ratios: (num_images, 2) in (ratio_width, ratio_height)
    """

    type: Literal["pixel_values"] = "pixel_values"
    
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "np", "cps", dynamic_dims={"np"}),
    ]
    image_grid_thw: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "ni", 3),
    ]
    num_image_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "ni")
    ]
    discrete_pixel_values: Annotated[
        list[torch.Tensor], 
        TensorShape("nb", "ni", 3, "ih", "iw", dynamic_dims={"ih", "iw"})
    ]
    discrete_image_ratios: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "ni", 2)
    ]
    num_discrete_image_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "ni")
    ]

class HyperCLOVAXOmniImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of image features
        - hs: Hidden size
        - ni: Number of images

    Historical context:
        - image_embeddings shape: (num_image_features, hidden_size)
        - num_image_features varies based on the number and resolution of the
          images.
        - hidden_size must match the hidden size of language model backbone.
        - image_grid_thw shape: (num_images, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["image_embeds"]

    image_embeddings: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

class HyperCLOVAXOmniVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - nb: Number of samples
        - nv: Number of videos
        - np: Number of patches
        - nc: Number of channels
        - cps: Number of channels * patch_size * patch_size
        - ih: Image height
        - iw: Image width
    
    Historical context:
        - pixel_values_videos shape: (num_patches, num_channels * patch_size *
          patch_size)
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format.
    """

    type: Literal["pixel_values_videos"] = "pixel_values_videos"
    
    pixel_values_videos: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "np", "cps", dynamic_dims={"np"}),
    ]
    video_grid_thw: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "nv", 3),
    ]
    num_video_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nb", "nv")
    ]

class HyperCLOVAXOmniVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - nf: Number of video features
        - hs: Hidden size
        - nv: Number of videos

    Historical context:
        - video_embeddings shape: (num_video_features, hidden_size)
        - num_video_features varies based on the number and resolution of the
          videos.
        - hidden_size must match the hidden size of language model backbone.
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
    """

    type: Literal["video_embeds"]

    video_embeddings: Annotated[
        torch.Tensor,
        TensorShape("nf", "hs"),
    ]

HyperCLOVAXOmniAudioInputs: TypeAlias = (
    HyperCLOVAXOmniAudioFeatureInputs | HyperCLOVAXOmniAudioEmbeddingInputs
) 
HyperCLOVAXOmniImageInputs: TypeAlias = (
    HyperCLOVAXOmniImagePixelInputs | HyperCLOVAXOmniImageEmbeddingInputs
)
HyperCLOVAXOmniVideoInputs: TypeAlias = (
    HyperCLOVAXOmniVideoPixelInputs | HyperCLOVAXOmniVideoEmbeddingInputs
)


class HyperCLOVAXOmniProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()
    
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_audio_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        audio_processor = hf_processor.audio_processor
        assert isinstance(audio_processor, SequenceFeatureExtractor)
        return audio_processor
    
    def get_image_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        image_processor = hf_processor.image_processor
        assert isinstance(image_processor, BaseImageProcessor)
        return image_processor
    
    def get_video_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        video_processor = hf_processor.audio_processor
        assert isinstance(video_processor, BaseVideoProcessor)
        return video_processor
        
    def get_data_parser(self):
        audio_processor = self.get_audio_processor()
        return MultiModalDataParser(
            target_sr=audio_processor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )
    
    def get_target_channels(self) -> int:
        """Return target audio channels for Audio models (mono)."""
        return 1
        
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        supported_mm = dict()
        if getattr(self.ctx.model_config.hf_config, "vision_config", None):
            supported_mm["image"] = None
            supported_mm["video"] = None
        if getattr(self.ctx.model_config.hf_config, "audio_config", None):
            supported_mm["audio"] = None
        return supported_mm
    
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens(seq_len, mm_counts)
        max_video_tokens = self.get_max_video_tokens(seq_len, mm_counts)
        return {"image": max_image_tokens, "video": max_video_tokens}
    
    def _get_max_image_pixels(self, max_tokens: int) -> int:
        """Find the largest max_pixels that stays within token budget."""
        image_processor = self.get_image_processor()
        base_max_pixels = image_processor.size["longest_edge"]
        unit = self.get_hf_config().vision_config.patch_size \
            * self.get_hf_config().vision_config.spatial_merge_size

        max_image_pixels = unit * unit  # minimum
        for candidate in range(unit * unit, base_max_pixels + 1, unit * unit):
            w, h = self.get_image_size_with_most_features(max_pixels=candidate)
            tokens = self.get_num_image_tokens(image_width=w, image_height=h)
            if tokens <= max_tokens:
                max_image_pixels = candidate
            else:
                break
        return max_image_pixels
    
    def get_image_size_with_most_features(
        self, 
        max_pixels: int | None = None,
    ) -> ImageSize:
        # NOTE: Simply processing a huge size with _get_vision_info might not give a
        # size that maximizes the number of featrues, i.e., the number of (merged)
        # patches. This is because the number of patches limits the allowed aspect
        # ratios. For example, suppose the maximum number of patches is 1280. A square
        # image cannot be broken down into 1280 patches, so feeding a giant square image
        # into _get_vision_info will not yield a size that maximizes the number of
        # patches. Therefore, we directly factorize the maximum number of patches into
        # height and width. The tricky part is to avoid extreme aspect ratios (>200 for
        # qwen2-vl). If we can't find a suitable aspect ratio, we decrease the number of
        # patches and retry. This is safe because the processor does not accept extreme
        # aspect ratios, so there is no valid post-resize image with the number of
        # patches that yields extreme aspect ratios.

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        if max_pixels is None:
            image_processor = self.get_image_processor()

            mm_kwargs = self.ctx.get_merged_mm_kwargs({})
            size = image_processor.size
            if override_size := mm_kwargs.get("size"):
                size = size | override_size
            if (override_min_pixels := mm_kwargs.get("min_pixels")) is not None:
                size = size | {"shortest_edge": override_min_pixels}
            if (override_max_pixels := mm_kwargs.get("max_pixels")) is not None:
                size = size | {"longest_edge": override_max_pixels}

            max_pixels = size["longest_edge"]

        unit = patch_size * merge_size
        max_seq_len = max_pixels // (unit * unit)

        def closest_factor_pair(n: int) -> tuple[int, int]:
            # left <= right
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        height_factor, width_factor = 1, max_seq_len
        for seq_len in range(max_seq_len, 0, -1):
            height_factor, width_factor = closest_factor_pair(seq_len)
            if width_factor / height_factor <= 200:
                break

        return ImageSize(width=unit * width_factor, height=unit * height_factor)
    
    def _get_max_video_frames(
        self, 
        max_tokens: int, 
        start_num_frames: int = 1,
    ) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        max_video_frames = start_num_frames
        while True:
            next_num_frames = max_video_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=image_processor,
                mm_kwargs={},
            )
            if next_max_tokens > max_tokens:
                break
            max_video_frames = next_num_frames
        return max_video_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        max_frames_per_video: int = 14,
    ) -> int:
        max_videos = mm_counts.get("video", 0)

        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), max_frames_per_video
        )
        return max(max_frames_per_video, 1)
    
    def get_num_audio_tokens(
        self,
        *,
        audio_masks: torch.Tensor,
        discrete_audio_values: torch.Tensor,
        **kwargs: object,
    ) -> int:
        _hf_processor = self.get_hf_processor(**kwargs)
        num_audio_tokens = _hf_processor.audio_processor.get_num_audio_tokens(
            audio_masks=audio_masks,
            discrete_audio_values=discrete_audio_values,
        )
        return num_audio_tokens
        
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        pixel_values: torch.Tensor | None = None,
        **kwargs: object,
    ) -> int:
        hf_config = self.get_hf_config()
        _hf_processor = self.get_hf_processor(**kwargs)
        num_image_tokens = _hf_processor.image_processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            pixel_values=pixel_values,
        )
        return num_image_tokens
    
    def get_max_image_tokens(
        self,
        seq_len: Optional[int] = None,
        mm_counts: Optional[Mapping[str, int]] = None,
    ) -> int:
        image_processor = self.get_image_processor()
        if seq_len and mm_counts:
            max_images = max(mm_counts.get("image", 1), 1)
            max_pixels_per_image = self._get_max_image_pixels(
                seq_len // max_images
            )
            target_width, target_height = self.get_image_size_with_most_features(
                max_pixels=max_pixels_per_image,
            )
        else:
            target_width, target_height = self.get_image_size_with_most_features()
        
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=image_processor,
            mm_kwargs={},
        )
    
    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        pixel_values_videos: torch.Tensor | None = None,
        **kwargs: object,
    ) -> int:
        hf_config = self.get_hf_config()
        _hf_processor = self.get_hf_processor(**kwargs)
        num_video_tokens = _hf_processor.video_processor.get_num_video_tokens(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            pixel_values_videos=pixel_values_videos,
        )
        return num_video_tokens

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=image_processor,
            mm_kwargs={},
        )


class HyperCLOVAXOmniDummyInputsBuilder(BaseDummyInputsBuilder[HyperCLOVAXOmniProcessingInfo]):
    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        
        dummy_text = ""
        hf_processor = self.info.get_hf_processor()
        
        if (
            num_audios
            and hf_processor.audio_processor is not None
        ):
            audio_placeholder = hf_processor.get_audio_placeholder()
            dummy_text += audio_placeholder * num_audios
        
        if (
            num_images
            and hf_processor.image_processor is not None
        ):
            image_placeholder = hf_processor.get_image_placeholder()
            dummy_text += image_placeholder * num_images
            
        if (
            num_videos
            and hf_processor.video_processor is not None
        ):
            video_placeholder = hf_processor.get_video_placeholder()
            dummy_text += video_placeholder * num_videos
        return dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        audio_processor = self.info.get_audio_processor()
        target_audio_length = min(audio_processor.chunk_length, 30,) * audio_processor.sampling_rate
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = 32

        audio_overrides = mm_options.get("audio") if mm_options else None
        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_width - 1,
                height=target_height - 1,
                num_frames=target_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }


class HyperCLOVAXOmniMultiModalProcessor(
    BaseMultiModalProcessor[HyperCLOVAXOmniProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        for video_idx, video_arr in enumerate(mm_data.get("videos", [])):
            if video_arr.dtype != np.uint8:
                mm_data["videos"][video_idx] = video_arr.astype(np.uint8)

        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        processed_outputs = self.info.ctx.call_hf_processor(
            hf_processor=hf_processor,
            data=dict(
                text=prompt,
                images=None,
                videos=None,
            ),
        ) # text-only

        # each mm_item should be processed seperately
        # since images with different patch_sizes are stacked in one single tensor
        if len(mm_data) > 0:
            audios = mm_data.get("audios")
            images = mm_data.get("images")
            videos = mm_data.get("videos")

            if audios:
                for _audio in audios:
                    _processed_outputs = self.info.ctx.call_hf_processor(
                        hf_processor=hf_processor,
                        data=dict(
                            text=None,
                            images=None,
                            videos=None,
                            audios=[_audio, ],
                        ),
                    )
                    for _k, _v in _processed_outputs.items():
                        if _k not in processed_outputs:
                            processed_outputs[_k] = list()
                        processed_outputs[_k] += [_v, ]

            if images:
                for _image in images:
                    _processed_outputs = self.info.ctx.call_hf_processor(
                        hf_processor=hf_processor,
                        data=dict(
                            text=None,
                            images=[_image, ],
                            videos=None,
                            audios=None,
                        ),
                    )
                    for _k, _v in _processed_outputs.items():
                        if _k not in processed_outputs:
                            processed_outputs[_k] = list()
                        processed_outputs[_k] += [_v, ]

            if videos:
                for _video in videos:
                    _processed_outputs = self.info.ctx.call_hf_processor(
                        hf_processor=hf_processor,
                        data=dict(
                            text=None,
                            images=None,
                            videos=[_video, ],
                            audios=None,
                        ),
                    )
                    for _k, _v in _processed_outputs.items():
                        if _k not in processed_outputs:
                            processed_outputs[_k] = list()
                        processed_outputs[_k] += [_v, ]
                    
        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor()
        audio_placeholder = hf_processor.get_audio_placeholder()
        image_placeholder = hf_processor.get_image_placeholder()
        video_placeholder = hf_processor.get_video_placeholder()
        placeholder = {
            "audio": audio_placeholder,
            "image": image_placeholder,
            "video": video_placeholder,
        }

        def get_replacement_hyperclovax(
            item_idx: int,
            modality: str,
            out_mm_kwargs: MultiModalKwargsItems,
            hf_processor: AutoProcessor, 
        ):
            out_item = out_mm_kwargs[modality][item_idx]
            replacement = list()
            embed_token_ids = list()
            if modality == "audio":
                replacement += hf_processor.get_audio_token_replacement(
                    num_audio_tokens=out_item["num_audio_tokens"].data,
                    num_discrete_audio_tokens=out_item["num_discrete_audio_tokens"].data,
                    include_boundary_tokens=True, # attach start_token & end_token
                    tokenize=True, # return token_ids
                )
                if hf_processor.audio_processor.use_discrete_token:
                    embed_token_ids.append(hf_processor.discrete_audio_token_id)
                embed_token_ids.append(hf_processor.audio_token_id)
                    
            elif modality == "image":
                replacement += hf_processor.get_image_token_replacement(
                    num_image_tokens=out_item["num_image_tokens"].data,
                    include_boundary_tokens=True, # attach start_token & end_token
                    tokenize=True, # return token_ids
                )
                if hf_processor.image_processor.use_discrete_token:
                    embed_token_ids.append(hf_processor.discrete_image_token_id)
                embed_token_ids.append(hf_processor.image_token_id)
                
            elif modality == "video":
                replacement = hf_processor.get_video_token_replacement(
                    num_video_tokens=out_item["num_video_tokens"].data,
                    include_boundary_tokens=False, # attach start_token & end_token
                    tokenize=True, # return token_ids
                )
                embed_token_ids.append(hf_processor.video_token_id)
                
            else:
                raise NotImplementedError(modality)

            return PromptUpdateDetails.select_token_ids(
                replacement,
                embed_token_ids=embed_token_ids,
            )

        prompt_updates = list()
        for modality in mm_items.keys():
            prompt_updates.append(PromptReplacement(
                modality=modality,
                target=placeholder[modality],
                replacement=partial(
                    get_replacement_hyperclovax,
                    modality=modality,
                    out_mm_kwargs=out_mm_kwargs,
                    hf_processor=hf_processor,
                ),
            ))
                
        return prompt_updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            # audio
            audio_values=MultiModalFieldConfig.batched("audio"),
            audio_attention_mask=MultiModalFieldConfig.batched("audio"),
            audio_masks=MultiModalFieldConfig.batched("audio"),
            num_audio_tokens=MultiModalFieldConfig.batched("audio"),
            discrete_audio_values=MultiModalFieldConfig.batched("audio"),
            num_discrete_audio_tokens=MultiModalFieldConfig.batched("audio"),
            # image
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            num_image_tokens=MultiModalFieldConfig.batched("image"),
            discrete_pixel_values=MultiModalFieldConfig.batched("image"),
            discrete_image_ratios=MultiModalFieldConfig.batched("image"),
            num_discrete_image_tokens=MultiModalFieldConfig.batched("image"),
            # video
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
            num_video_tokens=MultiModalFieldConfig.batched("video"),
        )

def initialize_continuous_vision_encoder(
    vision_config,
    quant_config: QuantizationConfig | None,
    multimodal_config: MultiModalConfig | None,
    *,
    norm_eps: float = 1e-5,
    vision_feature_layer: int | None = None,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel | Qwen2_5_VisionTransformer:
    num_hidden_layers = getattr(vision_config, "num_hidden_layers", None)
    if (
        not num_hidden_layers
        or not isinstance(vision_feature_layer, int)
    ):
        pass
    elif vision_feature_layer >= 0:
        num_hidden_layers = vision_feature_layer + 1
    else:
        num_hidden_layers = num_hidden_layers + vision_feature_layer + 1

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, Qwen2_5_VLVisionConfig):
        vision_model = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        return vision_model

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


class HyperCLOVAXOmniMlp(nn.Module):
    def __init__(
        self,
        mm_projector_type,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mm_projector_type = mm_projector_type
        if self.mm_projector_type == "mlp":
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        elif self.mm_projector_type == "inverted_mlp":
            self.fc1 = nn.Linear(in_features, 2 * hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(2 * hidden_features, out_features)
        else:
            raise NotImplementedError(
                "{} is not implemented".format(self.mm_projector_type)
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class HyperCLOVAXOmniCAbstractor(nn.Module):
    """
    This module is based on C-Abstractor, whose license is under apache-2.0.
    You can check the original code at
    https://github.com/khanrc/honeybee/blob/main/honeybee/projectors/projectors.py
    and we made necessary modifications.
    """

    def __init__(
        self,
        num_queries: int,
        num_input_tokens: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        pos_emb: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # Positional embedding
        if pos_emb:
            self.pos_emb = torch.nn.Parameter(
                torch.zeros(1, num_input_tokens, encoder_hidden_size)
            )
            self.pos_emb.data.normal_(mean=0.0, std=0.02)
        else:
            self.pos_emb = None

        # (Optional) Pre-normalization layer
        if prenorm:
            self.prenorm = LayerNorm(encoder_hidden_size)
        else:
            self.prenorm = None

        self.build_net(
            num_queries, encoder_hidden_size, hidden_size, output_hidden_size
        )
        self.dtype = next(self.parameters()).dtype

    def forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> torch.Tensor:
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(
            x,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_grids=num_grids,
        )  # (B, L, output_hidden_size)

        return x

    def _forward(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> torch.Tensor:
        # x: [B, L, dim]
        B, L, dim = x.shape
        hw = int(L**0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)

        if num_queries_vis_abstractors is not None:
            assert num_grids is not None
            return self._forward_adaptive_num_query(
                x, num_queries_vis_abstractors, num_grids
            )

        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def _forward_adaptive_num_query(
        self,
        x: torch.Tensor,
        num_queries_vis_abstractors: list[list[int]] | None = None,
        num_grids: list[int] | None = None,
    ) -> list[torch.Tensor]:
        # self.net is consisted by 3 layers (s1, sampler, s2)
        assert len(self.net) == 3

        x = self.net[0](x)  # s1
        new_x = []
        for i, num_queries in enumerate(num_queries_vis_abstractors):
            hw = int(num_queries**0.5)
            sampler = nn.AdaptiveAvgPool2d((hw, hw))
            out = sampler(x[num_grids[i] : num_grids[i + 1], :])
            out = self.net[2](out)  # s2

            out = rearrange(out, "b d h w -> b (h w) d")
            out = self.readout(out)

            new_x.append(out)
        return new_x

    def build_net(
        self,
        n_queries: int,
        encoder_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int,
        depth: int = 3,
        mlp_depth: int = 2,
    ):
        assert (n_queries**0.5).is_integer(), (
            f"n_queries must be square number. n_queries: {n_queries}"
        )
        hw = int(n_queries**0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = self.build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def build_mlp(
        self,
        depth: int,
        hidden_size: int,
        output_hidden_size: int,
    ):
        layers = [nn.Linear(hidden_size, output_hidden_size)]
        for _ in range(1, depth):
            layers.append(nn.SiLU())
            layers.append(nn.Linear(output_hidden_size, output_hidden_size))
        return nn.Sequential(*layers)


@MULTIMODAL_REGISTRY.register_processor(
    HyperCLOVAXOmniMultiModalProcessor,
    info=HyperCLOVAXOmniProcessingInfo,
    dummy_inputs=HyperCLOVAXOmniDummyInputsBuilder,
)
class HyperCLOVAXOmniForCausalLM(
    nn.Module, 
    SupportsMultiModal, 
    SupportsPP,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.mm_projector.": "model.vision_projector.",
            "model.": "",
        }
    )
    
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

    def _init_continuous_vision(self, config, quant_config,
                               multimodal_config, prefix):
        """Initialize continuous vision encoder and projector.

        Returns:
            (vision_model, vision_projector, is_qwen_visual, vision_config)
        """
        vision_config = None
        vision_model = None
        vision_projector = None
        is_qwen_visual = False
        if isinstance(getattr(config, "vision_config", None),
                       (dict, PretrainedConfig)):
            vision_config = config.vision_config
            vision_config.anyres = config.anyres
            vision_config.max_num_grids = config.max_num_grids
            vision_config.update({"torch_dtype": config.torch_dtype})
            if vision_config.model_type == "qwen2_5_vl_visual":
                is_qwen_visual = True
            if is_qwen_visual and get_compute_capability() >= 8.0:
                vision_config._attn_implementation = "flash_attention_2"
            # initialize continuous_vision_encoder
            vision_model = initialize_continuous_vision_encoder(
                vision_config=vision_config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                norm_eps=getattr(config.text_config, "rms_norm_eps", 1e-6),
                prefix=maybe_prefix(prefix, "visual"),
            )
            # initialize vision_projector
            _vision_projector_input_dim = vision_config.hidden_size
            if is_qwen_visual:
                _vision_projector_input_dim = vision_config.out_hidden_size
            _vision_projector_output_dim = config.text_config.hidden_size
            if config.mm_projector_type == "linear":
                vision_projector = nn.Linear(
                    in_features=_vision_projector_input_dim,
                    out_features=_vision_projector_output_dim,
                )
            elif config.mm_projector_type == "cabstractor":
                self.mm_projector = HyperCLOVAXOmniCAbstractor(
                    num_queries=self.num_queries_vis_abstractor,
                    num_input_tokens=(self.vision_config.image_size // self.vision_config.patch_size) ** 2,
                    encoder_hidden_size=_vision_projector_input_dim,
                    hidden_size=_vision_projector_input_dim,
                    output_hidden_size=_vision_projector_output_dim,
                    pos_emb=config.proj_pos_emb,
                    prenorm=config.proj_prenorm,
                )
                self.mm_projector.pos_emb.to(config.torch_dtype)
            elif config.mm_projector_type == "qwen_merger":
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger
                self.mm_projector = Qwen2_5_VLPatchMerger(
                    dim=_vision_projector_output_dim,
                    context_dim=_vision_projector_input_dim,
                )

                def new_forward(self, inputs) -> torch.Tensor:
                    x, window_index = inputs
                    x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
                    reverse_indices = torch.argsort(window_index)
                    x = x[reverse_indices, :]
                    return x

                self.mm_projector.forward = types.MethodType(
                    new_forward, self.mm_projector)
            else:
                self.mm_projector = HyperCLOVAXOmniMlp(
                    config.mm_projector_type,
                    _vision_projector_input_dim,
                    hidden_features=_vision_projector_input_dim,
                    out_features=_vision_projector_output_dim,
                )
            vision_projector.to(vision_model.dtype)
        return vision_model, vision_projector, is_qwen_visual, vision_config

    def _init_discrete_vision(self, config, vllm_config):
        """Initialize discrete vision encoder.

        Returns:
            (discrete_vision_model, discrete_vision_codebook_size,
             discrete_image_unit_0_id, discrete_vision_config)
        """
        discrete_vision_config = None
        discrete_vision_model = None
        discrete_vision_codebook_size = None
        discrete_image_unit_0_id = None
        if isinstance(getattr(config, "discrete_vision_config", None),
                       (dict, PretrainedConfig)):
            discrete_vision_config = config.discrete_vision_config
            discrete_vision_config.update({"torch_dtype": config.torch_dtype})
            discrete_vision_model = AutoModel.from_config(
                discrete_vision_config,
                trust_remote_code=True,
            )
            discrete_vision_codebook_size = 65536
            discrete_image_unit_0_id = (
                vllm_config.model_config.hf_config.discrete_image_unit_0_id)
            if (
                "regularizer" in discrete_vision_model.config.bottleneck["args"]
                and "codebook_size" in discrete_vision_model.config.bottleneck["args"]["regularizer"]["args"]
            ):
                discrete_vision_codebook_size = discrete_vision_model.config.bottleneck["args"]["regularizer"]["args"]["codebook_size"]
        return (discrete_vision_model, discrete_vision_codebook_size,
                discrete_image_unit_0_id, discrete_vision_config)

    def _init_continuous_audio(self, config, prefix):
        """Initialize continuous audio encoder and projector.

        Returns:
            (audio_model, audio_projector, video_audio_compressor_config,
             video_audio_compressor, audio_config)
        """
        audio_config = None
        audio_model = None
        audio_projector = None
        video_audio_compressor_config = None
        video_audio_compressor = None
        if isinstance(getattr(config, "audio_config", None),
                       (dict, PretrainedConfig)):
            # initialize audio_model & audio_projector
            audio_config = config.audio_config
            audio_config.update({"torch_dtype": config.torch_dtype})
            audio_model = AutoModel.from_config(
                audio_config,
                trust_remote_code=True,
            )
            if config.audio_projector_type == "linear":
                audio_projector = nn.Linear(
                    in_features=audio_config.d_model,
                    out_features=config.text_config.hidden_size,
                )
            else:
                audio_projector = HyperCLOVAXOmniMlp(
                    config.audio_projector_type,
                    audio_config.d_model,
                    hidden_features=audio_config.d_model,
                    out_features=config.text_config.hidden_size,
                )
            audio_projector.to(audio_model.dtype)

            # initialize video_audio_compressor
            video_audio_compressor = None
            if config.video_audio_compressor_type == "mambamia":
                video_audio_compressor_config = (
                    config.video_audio_compressor_config)
                video_audio_compressor = AutoModel.from_config(
                    video_audio_compressor_config,
                    trust_remote_code=True,
                )
                video_audio_compressor.to(audio_model.dtype)
        return (audio_model, audio_projector,
                video_audio_compressor_config, video_audio_compressor,
                audio_config)

    def _init_discrete_audio(self, config, vllm_config):
        """Initialize discrete audio encoder.

        Returns:
            (discrete_audio_model, discrete_audio_unit_0_id,
             discrete_audio_config)
        """
        discrete_audio_config = None
        discrete_audio_model = None
        discrete_audio_unit_0_id = None
        if isinstance(getattr(config, "discrete_audio_config", None),
                       (dict, PretrainedConfig)):
            discrete_audio_config = config.discrete_audio_config
            discrete_audio_config.update({"torch_dtype": config.torch_dtype})
            discrete_audio_model = AutoModel.from_config(
                discrete_audio_config,
                trust_remote_code=True,
            )
            discrete_audio_unit_0_id = (
                vllm_config.model_config.hf_config.discrete_audio_unit_0_id)
        return (discrete_audio_model, discrete_audio_unit_0_id,
                discrete_audio_config)

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # set text_config
        text_config = config.text_config
        if text_config.model_type in ["llama", "hyperclovax", "gpt2"]:
            text_config._attn_implementation = config._attn_implementation
        if text_config.model_type != "hyperclovax":
            text_config.logits_scaling = 1.0
        if getattr(text_config, "padded_vocab_size", None):
            text_config.vocab_size = text_config.padded_vocab_size
        text_config.update({"torch_dtype": config.torch_dtype})
        self.text_config = text_config

        # Language model
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # Vision tower (image + video)
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            (self.vision_model, self.vision_projector,
             self.is_qwen_visual,
             self.vision_config) = self._init_continuous_vision(
                config, quant_config, multimodal_config, prefix)
            (self.discrete_vision_model,
             self.discrete_vision_codebook_size,
             self.discrete_image_unit_0_id,
             self.discrete_vision_config) = self._init_discrete_vision(
                config, vllm_config)

        # Audio tower
        with self._mark_tower_model(vllm_config, "audio"):
            (self.audio_model, self.audio_projector,
             self.video_audio_compressor_config,
             self.video_audio_compressor,
             self.audio_config) = self._init_continuous_audio(config, prefix)
            (self.discrete_audio_model,
             self.discrete_audio_unit_0_id,
             self.discrete_audio_config) = self._init_discrete_audio(
                config, vllm_config)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _process_audio_input(
        self,
        audio_input: HyperCLOVAXOmniAudioInputs,
    ) -> tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            audio_embeddings = audio_input["audio_embeds"]
        else:
            audio_embeddings = list()
            for _idx, (_audio_values, _audio_attention_mask) in enumerate(zip(
                audio_input["audio_values"], audio_input["audio_attention_mask"],
            )):
                _audio_embeddings = self.audio_model(
                    _audio_values, 
                    attention_mask=_audio_attention_mask,
                ).last_hidden_state
                _audio_embeddings = _audio_embeddings.flatten(0, 1)
                _audio_embeddings = self.audio_projector(_audio_embeddings)
                
                if (
                    audio_input.get("discrete_audio_values", list()) is not None
                    and len(audio_input["discrete_audio_values"]) > _idx
                ):
                    _discrete_token_ids = self.discrete_audio_model.forward(
                        audio_input["discrete_audio_values"][0]
                    )
                    _discrete_token_ids = _discrete_token_ids + self.discrete_audio_unit_0_id
                    if (
                        (_discrete_token_ids < 0).any()
                        or (_discrete_token_ids >= self.language_model.config.vocab_size).any()
                    ):
                        _discrete_token_ids = torch.clamp(
                            input=_discrete_token_ids,
                            min=0,
                            max=self.language_model.config.vocab_size-1,
                        )
                    _discrete_audio_embeddings = self.embed_input_ids(
                        input_ids=_discrete_token_ids,
                    )[0]
                    _audio_embeddings = torch.cat([
                        _discrete_audio_embeddings,
                        _audio_embeddings,
                    ], dim=0)
                    
                audio_embeddings.append(_audio_embeddings)

        return audio_embeddings

    def _process_image_input(
        self,
        image_input: HyperCLOVAXOmniImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            image_embeddings = image_input["image_embeds"]
        else:
            image_embeddings = list()
            for _idx, (_pixel_values, _image_grid_thw) in enumerate(zip(
                image_input["pixel_values"], image_input["image_grid_thw"],
            )):
                _image_embeddings = self.vision_model(
                    _pixel_values,
                    grid_thw=_image_grid_thw,
                )
                _image_embeddings = self.vision_projector(_image_embeddings)

                if (
                    image_input.get("discrete_pixel_values", list()) is not None
                    and len(image_input["discrete_pixel_values"]) > _idx
                ):
                    _discrete_token_ids = self.discrete_vision_model(
                        image_input["discrete_pixel_values"][_idx]
                    )["encoded"]
                    if (
                        (_discrete_token_ids < 0).any()
                        or (_discrete_token_ids >= self.discrete_vision_codebook_size).any()
                    ):
                        _discrete_token_ids = torch.clamp(
                            input=_discrete_token_ids, 
                            min=0, 
                            max=self.discrete_vision_codebook_size,
                        )
                        
                    _discrete_token_ids += self.discrete_image_unit_0_id        
                    if (
                        (_discrete_token_ids < 0).any()
                        or (_discrete_token_ids >= self.language_model.config.vocab_size).any()
                    ):
                        _discrete_token_ids = torch.clamp(
                            input=_discrete_token_ids,
                            min=0,
                            max=self.language_model.config.vocab_size-1,
                        )
                    _discrete_image_embeddings = self.embed_input_ids(
                        input_ids=_discrete_token_ids,
                    )[0]
                    _image_embeddings = torch.cat([
                        _discrete_image_embeddings,
                        _image_embeddings,
                    ], dim=0)
                
                image_embeddings.append(_image_embeddings)

        return image_embeddings

    def _process_video_input(
        self,
        video_input: HyperCLOVAXOmniVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        if video_input["type"] == "video_embeds":
            video_embeddings = video_input["video_embeds"]
        else:
            video_embeddings = list()
            for _idx, (_pixel_values_videos, _video_grid_thw) in enumerate(zip(
                video_input["pixel_values_videos"], video_input["video_grid_thw"],
            )):
                _video_embeddings = self.vision_model(
                    _pixel_values_videos, 
                    grid_thw=_video_grid_thw,
                )
                _video_embeddings = self.vision_projector(_video_embeddings)
                video_embeddings.append(_video_embeddings)
        return video_embeddings

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key == "audio_values" and "audio" not in modalities:
                modalities["audio"] = self._parse_and_validate_audio_input(**kwargs)
            if input_key == "pixel_values" and "image" not in modalities:
                modalities["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key == "pixel_values_videos" and "video" not in modalities:
                modalities["video"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> HyperCLOVAXOmniAudioInputs | None:
        audio_values = kwargs.pop("audio_values", None)
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        audio_masks = kwargs.pop("audio_masks", None)
        num_audio_tokens = kwargs.pop("num_audio_tokens", None)
        discrete_audio_values = kwargs.pop("discrete_audio_values", None)
        num_discrete_audio_tokens = kwargs.pop("num_discrete_audio_tokens", None)
        audio_embeddings = kwargs.pop("audio_embeds", None)

        if (
            audio_values is None 
            and audio_embeddings is None
        ):
            return None
        
        if audio_values is not None:
            return HyperCLOVAXOmniAudioFeatureInputs(
                audio_values=audio_values,
                audio_attention_mask=audio_attention_mask,
                audio_masks=audio_masks,
                num_audio_tokens=num_audio_tokens,
                discrete_audio_values=discrete_audio_values,
                num_discrete_audio_tokens=num_discrete_audio_tokens,
            )
        
        if audio_embeddings is not None:
            return HyperCLOVAXOmniAudioEmbeddingInputs(
                audio_embeddings=audio_embeddings,
            )

        raise AssertionError("Validation failed: audio_input")
        
    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> HyperCLOVAXOmniImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        num_image_tokens = kwargs.pop("num_image_tokens", None)
        discrete_pixel_values = kwargs.pop("discrete_pixel_values", None)
        discrete_image_ratios = kwargs.pop("discrete_image_ratios", None)
        num_discrete_image_tokens = kwargs.pop("num_discrete_image_tokens", None)
        image_embeddings = kwargs.pop("image_embeds", None)

        if (
            pixel_values is None 
            and image_embeddings is None
        ):
            return None
        
        if pixel_values is not None:
            return HyperCLOVAXOmniImagePixelInputs(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_image_tokens=num_image_tokens,
                discrete_pixel_values=discrete_pixel_values,
                discrete_image_ratios=discrete_image_ratios,
                num_discrete_image_tokens=num_discrete_image_tokens,
            )
        
        if image_embeddings is not None:
            return HyperCLOVAXOmniImageEmbeddingInputs(
                image_embeddings=image_embeddings,
            )
        
        raise AssertionError("Validation failed: image_input")

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> HyperCLOVAXOmniVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        num_video_tokens = kwargs.pop("num_video_tokens", None)
        video_embeddings = kwargs.pop("video_embeds", None)

        if (
            pixel_values_videos is None 
            and video_embeddings is None
        ):
            return None
        
        if pixel_values_videos is not None:
            return HyperCLOVAXOmniVideoPixelInputs(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_video_tokens=num_video_tokens,
            )
        
        if video_embeddings is not None:
            return HyperCLOVAXOmniVideoEmbeddingInputs(
                video_embeddings=video_embeddings,
            )
        
        raise AssertionError("Validation failed: video_input")

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = list()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "audio":
                audio_input = modalities["audio"]
                _audio_embeddings = self._process_audio_input(audio_input)
                multimodal_embeddings += _audio_embeddings
            if modality == "image":
                image_input = modalities["image"]
                _image_embeddings = self._process_image_input(
                    image_input=image_input,
                )
                multimodal_embeddings += _image_embeddings
            if modality == "video":
                video_input = modalities["video"]
                _video_embeddings = self._process_video_input(
                    video_input=video_input,
                )
                multimodal_embeddings += _video_embeddings

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        skip_prefixes = list()
        if self.vision_model is None:
            skip_prefixes.extend(["vision_model."])
        if self.vision_projector is None:
            skip_prefixes.extend(["mm_projector."])
        if self.discrete_vision_model is None:
            skip_prefixes.extend(["discrete_vision_model."])
        if self.audio_model is None:
            skip_prefixes.extend(["audio_model."])
        if self.audio_projector is None:
            skip_prefixes.extend(["audio_projector."])
        if self.discrete_audio_model is None:
            skip_prefixes.extend(["discrete_audio_model."])

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )
        loaded_weights = loader.load_weights(
            weights, 
            mapper=self.hf_to_vllm_mapper,
        )
        return loaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="merger.",
            tower_model=["visual.", "audio_tower."],
        )