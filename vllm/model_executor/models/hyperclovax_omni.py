# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import types
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers import (
    AutoModel,
    BatchFeature,
    CLIPVisionConfig,
    PretrainedConfig,
    SiglipVisionConfig,
    WhisperFeatureExtractor,
)
from transformers.image_processing_utils import BaseImageProcessor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLVisionConfig,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
)
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

logger = init_logger(__name__)


def get_compute_capability(
    device_index: int = 0,
):
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    cc_version = float(f"{major}.{minor}")
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
        torch.Tensor,
        TensorShape("na", "nm", "nf"),
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
        torch.Tensor,
        TensorShape("np", "cps"),
    ]
    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
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
        torch.Tensor | list[torch.Tensor], TensorShape("nb", "nv")
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
        audio_processor = getattr(hf_processor, "audio_processor", None)
        if audio_processor is not None:
            return audio_processor
        # Fallback: create WhisperFeatureExtractor from audio_config
        # when HF processor does not include audio_processor
        hf_config = self.get_hf_config()
        audio_config = getattr(hf_config, "audio_config", None)
        if audio_config is not None:
            return WhisperFeatureExtractor(
                feature_size=audio_config.num_mel_bins,
                sampling_rate=16000,
                chunk_length=30,
            )
        return None

    def get_image_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        image_processor = hf_processor.image_processor
        assert isinstance(image_processor, BaseImageProcessor)
        return image_processor

    def get_video_processor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        video_processor = getattr(hf_processor, "video_processor", None)
        return video_processor

    def get_data_parser(self):
        audio_processor = self.get_audio_processor()
        kwargs = {}
        if audio_processor is not None:
            kwargs["target_sr"] = audio_processor.sampling_rate
            kwargs["target_channels"] = self.get_target_channels()
        return MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            **kwargs,
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
        unit = (
            self.get_hf_config().vision_config.patch_size
            * self.get_hf_config().vision_config.spatial_merge_size
        )

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
        # size that maximizes the number of features, i.e., the number of (merged)
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
        audio_masks: torch.Tensor | None = None,
        discrete_audio_values: torch.Tensor | None = None,
        **kwargs: object,
    ) -> int:
        _hf_processor = self.get_hf_processor(**kwargs)
        audio_processor = getattr(_hf_processor, "audio_processor", None)
        if audio_processor is not None and hasattr(
            audio_processor, "get_num_audio_tokens"
        ):
            return audio_processor.get_num_audio_tokens(
                audio_masks=audio_masks,
                discrete_audio_values=discrete_audio_values,
            )
        # Fallback: Whisper-style conv output length from config
        hf_config = self.get_hf_config()
        audio_config = getattr(hf_config, "audio_config", None)
        if audio_config is None:
            return 0
        max_pos = getattr(audio_config, "max_source_positions", 1500)
        feat_len = (max_pos - 1) // 2 + 1
        return (feat_len - 2) // 2 + 1

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        pixel_values: torch.Tensor | None = None,
        **kwargs: object,
    ) -> int:
        _hf_processor = self.get_hf_processor(**kwargs)
        image_processor = _hf_processor.image_processor
        if hasattr(image_processor, "get_num_image_tokens"):
            return image_processor.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
                pixel_values=pixel_values,
            )
        # Fallback: Qwen2VL-style patch count
        images_kwargs = {
            "min_pixels": getattr(image_processor, "min_pixels", 3136),
            "max_pixels": getattr(image_processor, "max_pixels", 2073600),
        }
        return image_processor.get_number_of_image_patches(
            image_height,
            image_width,
            images_kwargs=images_kwargs,
        )

    def get_max_image_tokens(
        self,
        seq_len: int | None = None,
        mm_counts: Mapping[str, int] | None = None,
    ) -> int:
        image_processor = self.get_image_processor()
        if seq_len and mm_counts:
            max_images = max(mm_counts.get("image", 1), 1)
            max_pixels_per_image = self._get_max_image_pixels(seq_len // max_images)
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
        _hf_processor = self.get_hf_processor(**kwargs)
        video_processor = _hf_processor.video_processor
        if hasattr(video_processor, "get_num_video_tokens"):
            return video_processor.get_num_video_tokens(
                image_width=image_width,
                image_height=image_height,
                num_frames=num_frames,
                pixel_values_videos=pixel_values_videos,
            )
        # Fallback: Qwen2VL-style patch count
        return video_processor.get_num_of_video_patches(
            num_frames,
            image_height,
            image_width,
            videos_kwargs={},
        )

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


class HyperCLOVAXOmniDummyInputsBuilder(
    BaseDummyInputsBuilder[HyperCLOVAXOmniProcessingInfo]
):
    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        added_vocab = tokenizer.get_added_vocab()

        # Resolve correct tokens from tokenizer (handles case mismatch)
        def _find_token(name, fallback):
            if name in added_vocab:
                return name
            # Try uppercase/lowercase variants
            for variant in [name.upper(), name.lower()]:
                if variant in added_vocab:
                    return variant
            return fallback

        image_token = _find_token(
            getattr(hf_processor, "image_token", "<|IMAGE_PAD|>"),
            "<|IMAGE_PAD|>",
        )
        video_token = _find_token(
            getattr(hf_processor, "video_token", "<|VIDEO_PAD|>"),
            "<|VIDEO_PAD|>",
        )

        dummy_text = ""
        if num_audios:
            audio_processor = getattr(hf_processor, "audio_processor", None)
            if audio_processor is not None:
                dummy_text += "<|AUDIO_PAD|>" * num_audios

        if num_images:
            dummy_text += image_token * num_images

        if num_videos:
            dummy_text += video_token * num_videos

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
        if audio_processor is not None:
            target_audio_length = (
                min(getattr(audio_processor, "chunk_length", 30), 30)
                * audio_processor.sampling_rate
            )
        else:
            target_audio_length = 30 * 16000
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

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> ProcessorInputs:
        """Build dummy processor inputs for memory profiling."""
        hf_processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        added_vocab = tokenizer.get_added_vocab()
        image_token = (
            "<|IMAGE_PAD|>"
            if "<|IMAGE_PAD|>" in added_vocab
            else getattr(hf_processor, "image_token", "<|IMAGE_PAD|>")
        )
        video_token = (
            "<|VIDEO_PAD|>"
            if "<|VIDEO_PAD|>" in added_vocab
            else getattr(hf_processor, "video_token", "<|VIDEO_PAD|>")
        )

        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        prompt_text = image_token * num_images + video_token * num_videos

        dummy_mm_data = self.get_dummy_mm_data(
            seq_len,
            mm_counts,
            mm_options,
        )

        data_parser = self.info.get_data_parser()
        dummy_mm_items = data_parser.parse_mm_data(dummy_mm_data)

        return ProcessorInputs(
            prompt=prompt_text,
            mm_data_items=dummy_mm_items,
            hf_processor_mm_kwargs=mm_processor_kwargs or {},
            tokenization_kwargs={"truncation": False},
        )


class HyperCLOVAXOmniMultiModalProcessor(
    BaseMultiModalProcessor[HyperCLOVAXOmniProcessingInfo]
):
    def _validate_mm_placeholders(self, mm_placeholders, mm_item_counts):
        # HyperCLOVAX-SEED-Omni uses discrete token streams for audio and
        # image generation (discrete_audio / discrete_image), which have no
        # continuous patch placeholders in the prompt.  Skip those modalities
        # in the base-class validation to avoid "0 placeholders found" errors.
        _SKIP = {"audio", "discrete_audio", "discrete_image"}
        filtered = {k: v for k, v in mm_item_counts.items() if k not in _SKIP}
        super()._validate_mm_placeholders(mm_placeholders, filtered)

    def _hf_processor_applies_updates(
        self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
    ):
        # HCXVisionV2Processor does NOT expand placeholder tokens.
        # Token expansion is handled by vLLM via _get_prompt_updates.
        return False

    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        # HCXVisionV2Processor requires text and images to be processed
        # together. The cache path separates them, causing failures.
        # Always use the non-cache path which handles this correctly.
        return self._apply_hf_processor(inputs, timing_ctx)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        added_vocab = tokenizer.get_added_vocab()
        image_token = (
            "<|IMAGE_PAD|>" if "<|IMAGE_PAD|>" in added_vocab else "<|image_pad|>"
        )
        video_token = (
            "<|VIDEO_PAD|>" if "<|VIDEO_PAD|>" in added_vocab else "<|video_pad|>"
        )
        audio_token = (
            "<|AUDIO_PAD|>" if "<|AUDIO_PAD|>" in added_vocab else "<|audio_pad|>"
        )

        # Separate audio from mm_data (HCXVisionV2Processor doesn't handle audio)
        mm_data = dict(mm_data)  # make mutable copy
        audios = mm_data.pop("audios", None)

        # Strip placeholders from prompt when corresponding mm data is absent
        if not mm_data.get("images") and image_token in prompt:
            prompt = prompt.replace(image_token, "")
        if not mm_data.get("videos") and video_token in prompt:
            prompt = prompt.replace(video_token, "")
        if not audios and audio_token in prompt:
            prompt = prompt.replace(audio_token, "")

        # Process text + images/videos via HF processor
        processed_outputs = self.info.ctx.call_hf_processor(
            hf_processor,
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

        # Process audio separately via WhisperFeatureExtractor
        if audios:
            audio_processor = self.info.get_audio_processor(**mm_kwargs)
            if audio_processor is not None:
                import torch

                all_features = []
                for audio in audios:
                    features = audio_processor(
                        audio,
                        sampling_rate=audio_processor.sampling_rate,
                        return_tensors="pt",
                    )
                    all_features.append(features["input_features"].squeeze(0))
                processed_outputs["audio_features"] = torch.stack(all_features)

        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        added_vocab = tokenizer.get_added_vocab()

        # Token IDs for placeholders
        placeholder = {}
        for mod, tok_str in [
            ("image", "<|IMAGE_PAD|>"),
            ("video", "<|VIDEO_PAD|>"),
            ("audio", "<|AUDIO_PAD|>"),
        ]:
            if tok_str in added_vocab:
                placeholder[mod] = added_vocab[tok_str]

        merge_size = hf_config.vision_config.spatial_merge_size

        def get_replacement(item_idx, modality, out_mm_kwargs):
            out_item = out_mm_kwargs[modality][item_idx]
            token_id = placeholder.get(modality, 0)

            if modality == "image":
                grid_thw_elem = out_item.get("image_grid_thw")
                if grid_thw_elem is not None:
                    grid_thw = grid_thw_elem.data
                    h, w = grid_thw[1].item(), grid_thw[2].item()
                    num_tokens = (h * w) // (merge_size**2)
                else:
                    num_tokens = 1
            elif modality == "video":
                grid_thw_elem = out_item.get("video_grid_thw")
                if grid_thw_elem is not None:
                    grid_thw = grid_thw_elem.data
                    t, h, w = grid_thw[0].item(), grid_thw[1].item(), grid_thw[2].item()
                    num_tokens = (t * h * w) // (merge_size**2)
                else:
                    num_tokens = 1
            elif modality == "audio":
                num_tokens = self.info.get_num_audio_tokens()
            else:
                num_tokens = 1

            return [token_id] * num_tokens

        prompt_updates = []
        for modality in mm_items:
            if modality in placeholder:
                prompt_updates.append(
                    PromptReplacement(
                        modality=modality,
                        target=[placeholder[modality]],
                        replacement=partial(
                            get_replacement,
                            modality=modality,
                            out_mm_kwargs=out_mm_kwargs,
                        ),
                    )
                )

        return prompt_updates

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Follow HCXVisionV2 pattern: use flat_from_sizes for pixel values
        hf_config = self.info.get_hf_config()
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)
        image_embed_grid_sizes = (
            image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_pixel_grid_sizes = video_grid_thw.prod(-1)
        video_embed_grid_sizes = (
            video_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
        )

        fields = dict(
            # image (Qwen2VL flat format)
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image",
                image_pixel_grid_sizes,
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image",
                image_embed_grid_sizes,
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            # video (Qwen2VL flat format)
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video",
                video_pixel_grid_sizes,
            ),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video",
                video_embed_grid_sizes,
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

        # Audio fields (only if present in inputs)
        if "audio_features" in hf_inputs:
            fields["audio_features"] = MultiModalFieldConfig.batched("audio")
        if "audio_values" in hf_inputs:
            fields.update(
                dict(
                    audio_values=MultiModalFieldConfig.batched("audio"),
                    audio_attention_mask=MultiModalFieldConfig.batched("audio"),
                    audio_masks=MultiModalFieldConfig.batched("audio"),
                    num_audio_tokens=MultiModalFieldConfig.batched("audio"),
                )
            )

        return fields


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
    if not num_hidden_layers or not isinstance(vision_feature_layer, int):
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
        assert (
            num_hidden_layers is None
            or num_hidden_layers == vision_config.num_hidden_layers
        ), (
            "Qwen2.5-VL does not support "
            f"num_hidden_layers override, got {num_hidden_layers}"
        )
        assert require_post_norm is None or require_post_norm is True, (
            "Qwen2.5-VL does not support "
            f"require_post_norm=False, got {require_post_norm}"
        )
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

    def _init_continuous_vision(self, config, quant_config, multimodal_config, prefix):
        """Initialize continuous vision encoder and projector.

        Returns:
            (vision_model, vision_projector, is_qwen_visual, vision_config)
        """
        vision_config = None
        vision_model = None
        vision_projector = None
        is_qwen_visual = False
        if isinstance(getattr(config, "vision_config", None), (dict, PretrainedConfig)):
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
                    num_input_tokens=(
                        self.vision_config.image_size // self.vision_config.patch_size
                    )
                    ** 2,
                    encoder_hidden_size=_vision_projector_input_dim,
                    hidden_size=_vision_projector_input_dim,
                    output_hidden_size=_vision_projector_output_dim,
                    pos_emb=config.proj_pos_emb,
                    prenorm=config.proj_prenorm,
                )
                self.mm_projector.pos_emb.to(config.torch_dtype)
            elif config.mm_projector_type == "qwen_merger":
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLPatchMerger,
                )

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
                    new_forward, self.mm_projector
                )
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
        raw_dvc = getattr(config, "discrete_vision_config", None)
        if isinstance(raw_dvc, (dict, PretrainedConfig)):
            # Skip if dict without a loadable model path
            can_load = not (
                isinstance(raw_dvc, dict) and not raw_dvc.get("model_name_or_path")
            )
            if can_load:
                discrete_vision_config = raw_dvc
                if isinstance(discrete_vision_config, dict):
                    discrete_vision_config.update({"torch_dtype": config.torch_dtype})
                discrete_vision_model = AutoModel.from_config(
                    discrete_vision_config,
                    trust_remote_code=True,
                )
                discrete_vision_codebook_size = 65536
                discrete_image_unit_0_id = (
                    vllm_config.model_config.hf_config.discrete_image_unit_0_id
                )
                if (
                    "regularizer" in discrete_vision_model.config.bottleneck["args"]
                    and "codebook_size"
                    in discrete_vision_model.config.bottleneck["args"]["regularizer"][
                        "args"
                    ]
                ):
                    discrete_vision_codebook_size = (
                        discrete_vision_model.config.bottleneck["args"]["regularizer"][
                            "args"
                        ]["codebook_size"]
                    )
        return (
            discrete_vision_model,
            discrete_vision_codebook_size,
            discrete_image_unit_0_id,
            discrete_vision_config,
        )

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
        if isinstance(getattr(config, "audio_config", None), (dict, PretrainedConfig)):
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
            vac_type = getattr(config, "video_audio_compressor_type", None)
            vac_config = getattr(config, "video_audio_compressor_config", None)
            if vac_type == "mambamia" and vac_config is not None:
                video_audio_compressor_config = vac_config
                video_audio_compressor = AutoModel.from_config(
                    video_audio_compressor_config,
                    trust_remote_code=True,
                )
                video_audio_compressor.to(audio_model.dtype)
        return (
            audio_model,
            audio_projector,
            video_audio_compressor_config,
            video_audio_compressor,
            audio_config,
        )

    def _init_discrete_audio(self, config, vllm_config):
        """Initialize discrete audio encoder.

        Returns:
            (discrete_audio_model, discrete_audio_unit_0_id,
             discrete_audio_config)
        """
        discrete_audio_config = None
        discrete_audio_model = None
        discrete_audio_unit_0_id = None
        raw_dac = getattr(config, "discrete_audio_config", None)
        if isinstance(raw_dac, (dict, PretrainedConfig)):
            can_load = not (
                isinstance(raw_dac, dict) and not raw_dac.get("model_name_or_path")
            )
            if can_load:
                discrete_audio_config = raw_dac
                if isinstance(discrete_audio_config, dict):
                    discrete_audio_config.update({"torch_dtype": config.torch_dtype})
                discrete_audio_model = AutoModel.from_config(
                    discrete_audio_config,
                    trust_remote_code=True,
                )
                discrete_audio_unit_0_id = (
                    vllm_config.model_config.hf_config.discrete_audio_unit_0_id
                )
        return (discrete_audio_model, discrete_audio_unit_0_id, discrete_audio_config)

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
            (
                self.vision_model,
                self.vision_projector,
                self.is_qwen_visual,
                self.vision_config,
            ) = self._init_continuous_vision(
                config, quant_config, multimodal_config, prefix
            )
            (
                self.discrete_vision_model,
                self.discrete_vision_codebook_size,
                self.discrete_image_unit_0_id,
                self.discrete_vision_config,
            ) = self._init_discrete_vision(config, vllm_config)

        # Audio tower
        with self._mark_tower_model(vllm_config, "audio"):
            (
                self.audio_model,
                self.audio_projector,
                self.video_audio_compressor_config,
                self.video_audio_compressor,
                self.audio_config,
            ) = self._init_continuous_audio(config, prefix)
            (
                self.discrete_audio_model,
                self.discrete_audio_unit_0_id,
                self.discrete_audio_config,
            ) = self._init_discrete_audio(config, vllm_config)

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
            audio_values = audio_input["audio_values"]
            audio_attention_mask = audio_input.get("audio_attention_mask")
            if audio_attention_mask is None:
                # WhisperFeatureExtractor output: process each audio individually
                for _idx in range(len(audio_values)):
                    _audio_values = audio_values[_idx : _idx + 1]
                    _audio_embeddings = self.audio_model(
                        _audio_values,
                    ).last_hidden_state
                    _audio_embeddings = _audio_embeddings.flatten(0, 1)
                    _audio_embeddings = self.audio_projector(_audio_embeddings)
                    audio_embeddings.append(_audio_embeddings)
                return audio_embeddings

            for _idx, (_audio_values, _audio_attention_mask) in enumerate(
                zip(audio_values, audio_attention_mask)
            ):
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
                    _discrete_token_ids = (
                        _discrete_token_ids + self.discrete_audio_unit_0_id
                    )
                    if (_discrete_token_ids < 0).any() or (
                        _discrete_token_ids >= self.language_model.config.vocab_size
                    ).any():
                        _discrete_token_ids = torch.clamp(
                            input=_discrete_token_ids,
                            min=0,
                            max=self.language_model.config.vocab_size - 1,
                        )
                    _discrete_audio_embeddings = self.embed_input_ids(
                        input_ids=_discrete_token_ids,
                    )[0]
                    _audio_embeddings = torch.cat(
                        [
                            _discrete_audio_embeddings,
                            _audio_embeddings,
                        ],
                        dim=0,
                    )

                audio_embeddings.append(_audio_embeddings)

        return audio_embeddings

    def _process_image_input(
        self,
        image_input: HyperCLOVAXOmniImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        # Qwen2VL flat format: process all patches at once
        pixel_values = image_input["pixel_values"]
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        image_embeds = self.vision_model(pixel_values, grid_thw=grid_thw_list)
        image_embeds = self.vision_projector(image_embeds)

        # Split by image using grid_thw sizes
        merge_size = getattr(
            self.vision_model,
            "spatial_merge_size",
            self.vision_config.spatial_merge_size,
        )
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self,
        video_input: HyperCLOVAXOmniVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"]

        # Qwen2VL flat format: process all patches at once
        pixel_values_videos = video_input["pixel_values_videos"]
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        video_embeds = self.vision_model(pixel_values_videos, grid_thw=grid_thw_list)
        video_embeds = self.vision_projector(video_embeds)

        # Split by video using grid_thw sizes
        merge_size = getattr(
            self.vision_model,
            "spatial_merge_size",
            self.vision_config.spatial_merge_size,
        )
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("audio_values", "audio_features")
                and "audio" not in modalities
            ):
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
        audio_features = kwargs.pop("audio_features", None)
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        audio_masks = kwargs.pop("audio_masks", None)
        num_audio_tokens = kwargs.pop("num_audio_tokens", None)
        discrete_audio_values = kwargs.pop("discrete_audio_values", None)
        num_discrete_audio_tokens = kwargs.pop("num_discrete_audio_tokens", None)
        audio_embeddings = kwargs.pop("audio_embeds", None)

        # audio_features from WhisperFeatureExtractor can be used as audio_values
        if audio_values is None and audio_features is not None:
            audio_values = audio_features

        if audio_values is None and audio_embeddings is None:
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
        kwargs.pop("num_image_tokens", None)
        kwargs.pop("discrete_pixel_values", None)
        kwargs.pop("discrete_image_ratios", None)
        kwargs.pop("num_discrete_image_tokens", None)
        image_embeddings = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeddings is None:
            return None

        if pixel_values is not None:
            return HyperCLOVAXOmniImagePixelInputs(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
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
        kwargs.pop("num_video_tokens", None)
        video_embeddings = kwargs.pop("video_embeds", None)

        if pixel_values_videos is None and video_embeddings is None:
            return None

        if pixel_values_videos is not None:
            return HyperCLOVAXOmniVideoPixelInputs(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
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
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
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
        if getattr(self, "video_audio_compressor", None) is None:
            skip_prefixes.extend(["video_audio_compressor."])

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
