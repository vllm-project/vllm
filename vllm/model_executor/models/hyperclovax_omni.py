# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HyperCLOVAX Omni (SEED-Omni-8B) Implementation.

This module extends the V2 architecture (Qwen2.5 Vision Transformer) with
audio support and discrete token streams for omni-modal generation.

Supports:
- HyperCLOVAX-SEED-Omni-8B: Text + Image + Video + Audio
"""

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

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, MultiModalConfig
from vllm.forward_context import set_forward_context
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
from vllm.transformers_utils import config
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
        - na: Number of audios (per-audio fields)
        - nc: Total number of audio chunks across audios (concat dim 0 of
          audio_values / audio_attention_mask; a single audio is split into
          one or more chunk_length-second chunks)
        - nam: audio_attention_mask broadcast dim (size 1)
        - nm: Number of mel bins
        - ns: Number of max sequence length
        - nf: Number of max nb frames
        - lc: Length of code
    """

    type: Literal["audio_values"] = "audio_values"

    audio_values: Annotated[
        torch.Tensor,
        TensorShape("nc", "nm", "nf"),
    ]
    audio_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("nc", "nam", "nq", "nk"),
    ]
    discrete_audio_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "ns", dynamic_dims={"ns"}),
    ]
    num_audio_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na"),
    ]
    num_discrete_audio_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na"),
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

    audio_embeds: Annotated[
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
    discrete_pixel_values: Annotated[
        torch.Tensor,
        TensorShape("ni", 3, "dh", "dw"),
    ]
    discrete_image_ratios: Annotated[
        torch.Tensor,
        TensorShape("ni", 2),
    ]
    num_image_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("ni"),
    ]
    num_discrete_image_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("ni"),
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

    image_embeds: Annotated[
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
        TensorShape("np", "cps", dynamic_dims={"np"}),
    ]
    video_grid_thw: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nv", 3),
    ]
    num_video_tokens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("nv"),
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

    video_embeds: Annotated[
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
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(),
        }

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_config = self.get_hf_config().vision_config
        # Qwen2.5-VL uses dynamic resolution; use a reasonable square default
        # for profiling (image_size may be absent/None in the config).
        size = getattr(vision_config, "image_size", None) or 448
        return ImageSize(width=size, height=size)

    def get_num_audio_tokens(
        self,
        *,
        audio_masks: torch.Tensor | None = None,
        discrete_audio_values: torch.Tensor | None = None,
        include_boundary_tokens: bool | None = True,
        return_tuple: bool | None = False,
        **kwargs: object,
    ) -> int:
        audio_processor = self.get_hf_processor().audio_processor
        num_continuous, num_discrete = audio_processor.get_num_audio_tokens(
            audio_masks=audio_masks,
            discrete_audio_values=discrete_audio_values,
            include_boundary_tokens=include_boundary_tokens,
            return_tuple=True,
        )
        if return_tuple:
            return num_continuous, num_discrete
        elif audio_processor.use_discrete_token:
            return num_continuous + num_discrete
        else:
            return num_continuous

    def get_num_image_tokens(
        self,
        *,
        image_width: int | None = None,
        image_height: int | None = None,
        pixel_values: torch.Tensor | None = None,
        include_boundary_tokens: bool | None = True,
        return_tuple: bool | None = False,
        **kwargs: object,
    ) -> int:
        image_processor = self.get_hf_processor().image_processor
        num_continuous, num_discrete = image_processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
            pixel_values=pixel_values,
            include_boundary_tokens=include_boundary_tokens,
            return_tuple=True,
        )
        if return_tuple:
            return num_continuous, num_discrete
        elif image_processor.use_discrete_token:
            return num_continuous + num_discrete
        else:
            return num_continuous

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )

    def get_num_video_tokens(
        self,
        *,
        video_width: int | None = None,
        video_height: int | None = None,
        num_frames: int | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        include_boundary_tokens: bool | None = True,
        return_tuple: bool | None = False,
        **kwargs: object,
    ) -> int:
        video_processor = self.get_hf_processor().video_processor
        num_continuous, num_discrete = video_processor.get_num_video_tokens(
            image_width=video_width,
            image_height=video_height,
            num_frames=num_frames,
            pixel_values_videos=pixel_values_videos,
            include_boundary_tokens=include_boundary_tokens,
            return_tuple=True,
        )
        if return_tuple:
            return num_continuous, num_discrete
        elif video_processor.use_discrete_token:
            return num_continuous + num_discrete
        else:
            return num_continuous

    def get_max_video_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_video_tokens(
            video_width=target_width,
            video_height=target_height,
            num_frames=32,
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
        # Reuse get_dummy_text so the prompt carries placeholders for every
        # modality (audio included); building it manually here previously
        # omitted the audio pad, breaking prompt-update application.
        prompt_text = self.get_dummy_text(mm_counts)

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
        audios = mm_data.get("audios")
        images = mm_data.get("images")
        videos = mm_data.get("videos")
        
        # Get the HF processor
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        
        # Build data dict for HF processor (images/videos only)
        # NOTE: We pass the prompt as-is without token normalization.
        # Token expansion is handled by vLLM via _get_prompt_updates since
        # _hf_processor_applies_updates returns False.
        data: dict[str, object] = dict(
            text=prompt,
            audios=audios,
            images=images,
            videos=videos,
        )
        
        processed_outputs = self.info.ctx.call_hf_processor(
            hf_processor=hf_processor,
            data=data,
            kwargs=dict(**mm_kwargs, **tok_kwargs),
        )

        # The audio processor concatenates each audio's chunk_length-second
        # chunks along dim 0 of audio_values/audio_masks/audio_attention_mask
        # (like image pixel patches). Record per-audio chunk counts so
        # _get_mm_fields_config can split those flat tensors back into per-audio
        # items (the audio analog of image_grid_thw). num_audio_tokens is a post
        # conv+pool token count, not a chunk count, so it cannot be used here.
        if audios and "audio_values" in processed_outputs:
            audio_processor = self.info.get_audio_processor(**mm_kwargs)
            chunk_samples = int(
                audio_processor.chunk_length * audio_processor.sampling_rate
            )
            lengths = [
                (a[0] if isinstance(a, tuple) else a).shape[-1] for a in audios
            ]
            processed_outputs["audio_chunks"] = torch.tensor(
                [max(1, -(-length // chunk_samples)) for length in lengths],
                dtype=torch.long,
            )
            # Split concatenated discrete_audio_values (Sum of per-audio samples)
            # into a per-audio list so it can be registered with batched("audio").
            dav = processed_outputs.get("discrete_audio_values")
            if dav is not None and not isinstance(dav, list):
                processed_outputs["discrete_audio_values"] = list(
                    torch.split(dav, lengths)
                )

        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()

        # Use token IDs directly from config.
        # This matches what get_dummy_processor_inputs uses, ensuring consistency.
        placeholder: dict[str, int] = {
            "audio": hf_config.audio_token_id,                    # 128071 for <|AUDIO_PAD|>
            "discrete_audio": hf_config.discrete_audio_token_id,  # 128074 for <|AUDIO_PAD|>
            "image": hf_config.image_token_id,                    # 128062 for <|IMAGE_PAD|>
            "discrete_image": hf_config.discrete_image_token_id,  # 128069 for <|IMAGE_PAD|>
            "video": hf_config.video_token_id,                    # 128063 for <|VIDEO_PAD|>
            "video_audio": hf_config.video_audio_token_id,        # 128070 for <|VIDEO_AUDIO_PAD|>
        }

        def get_replacement_omni(
            item_idx: int,
            modality: str,
            out_mm_kwargs: MultiModalKwargsItems,
        ):
            out_item = out_mm_kwargs[modality][item_idx]
            
            replacement = None
            if modality == "audio":
                num_continuous, num_discrete = self.info.get_num_audio_tokens(
                    audio_masks=out_item["audio_masks"].data,
                    discrete_audio_values=(
                        out_item["discrete_audio_values"].data
                        if "discrete_audio_values" in out_item
                        else None
                    ),
                    include_boundary_tokens=True,
                    return_tuple=True,
                )
                replacement = (num_discrete + num_continuous) * [placeholder.get(modality, 0), ]
                    
            elif modality == "image":
                num_continuous, num_discrete = self.info.get_num_image_tokens(
                    pixel_values=out_item["pixel_values"].data,
                    include_boundary_tokens=True,
                    return_tuple=True,
                )
                replacement = (num_discrete + num_continuous) * [placeholder.get(modality, 0), ]

            elif modality == "video":
                num_continuous, num_discrete = self.info.get_num_video_tokens(
                    pixel_values_videos=out_item["pixel_values_videos"].data,
                    include_boundary_tokens=True,
                    return_tuple=True,
                )
                replacement = (num_discrete + num_continuous) * [placeholder.get(modality, 0), ]

            return replacement

        prompt_updates = list()
        for modality in mm_items:
            if not placeholder.get(modality):
                continue
                
            prompt_updates.append(
                PromptReplacement(
                    modality=modality,
                    target=[placeholder[modality]],
                    replacement=partial(
                        get_replacement_omni,
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
        # Declare each modality's fields only when that modality is present in
        # hf_inputs, so we never compute sizes for / call flat_from_sizes on an
        # absent modality (which would feed empty/ill-shaped sizes).
        hf_config = self.info.get_hf_config()
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        fields: dict[str, MultiModalFieldConfig] = {}

        # audio: audio_values/audio_masks/audio_attention_mask are per-audio
        # chunks concatenated along dim 0 -> split by per-audio chunk counts.
        # audio_embeds (precomputed) splits by per-audio token counts.
        # num_audio_tokens/audio_chunks/discrete_* are (N,) -> batched.
        if "audio_values" in hf_inputs:
            audio_chunk_sizes = hf_inputs["audio_chunks"]
            audio_embed_sizes = hf_inputs.get("num_audio_tokens")
            if audio_embed_sizes is None:
                # Fallback (e.g. precomputed audio_embeds): mirror the audio
                # processor's continuous get_num_audio_tokens (conv stride 2 ->
                # pooling). Padded per-chunk frame length is uniform, so
                # per-audio tokens = per_chunk_tokens * chunk_count.
                audio_masks = hf_inputs.get("audio_masks")
                if audio_masks is not None and audio_masks.numel() > 0:
                    _pool_kernel_size = 2  # default
                    _pool_stride = 2  # default
                    conv_len = (audio_masks.shape[-1] - 1) // 2 + 1
                    per_chunk_tokens = (
                        conv_len - _pool_kernel_size
                    ) // _pool_stride + 1
                    audio_embed_sizes = per_chunk_tokens * audio_chunk_sizes
                else:
                    audio_embed_sizes = audio_chunk_sizes
            fields.update(
                audio_values=MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_chunk_sizes
                ),
                audio_masks=MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_chunk_sizes
                ),
                audio_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_chunk_sizes
                ),
                audio_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_embed_sizes
                ),
                num_audio_tokens=MultiModalFieldConfig.batched("audio"),
                audio_chunks=MultiModalFieldConfig.batched("audio"),
                discrete_audio_values=MultiModalFieldConfig.batched("audio"),
                num_discrete_audio_tokens=MultiModalFieldConfig.batched("audio"),
            )

        # image
        if "pixel_values" in hf_inputs or "image_embeds" in hf_inputs:
            image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
            image_pixel_grid_sizes = image_grid_thw.prod(-1)
            image_embed_sizes = hf_inputs.get(
                "num_image_tokens",
                image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size,
            )
            fields.update(
                pixel_values=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_pixel_grid_sizes
                ),
                image_grid_thw=MultiModalFieldConfig.batched(
                    "image", keep_on_cpu=True
                ),
                image_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_embed_sizes
                ),
                num_image_tokens=MultiModalFieldConfig.batched("image"),
                discrete_pixel_values=MultiModalFieldConfig.batched("image"),
                discrete_image_ratios=MultiModalFieldConfig.batched("image"),
                num_discrete_image_tokens=MultiModalFieldConfig.batched("image"),
            )

        # video
        if "pixel_values_videos" in hf_inputs or "video_embeds" in hf_inputs:
            video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
            video_pixel_grid_sizes = video_grid_thw.prod(-1)
            video_embed_sizes = hf_inputs.get(
                "num_video_tokens",
                video_pixel_grid_sizes // spatial_merge_size // spatial_merge_size,
            )
            fields.update(
                pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_pixel_grid_sizes
                ),
                video_grid_thw=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
                video_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_embed_sizes
                ),
                num_video_tokens=MultiModalFieldConfig.batched("video"),
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
        orig_to_new_substr={
            # Checkpoint stores the TA-Tok decoder as a wrapped Siglip2VisionModel
            # (`decoder.vision_model.*`), but the instantiated module is flattened
            # (`decoder.*`). Drop the extra wrapper level. (substr rules run before
            # the prefix rules below.)
            "discrete_vision_model.decoder.vision_model.": "discrete_vision_model.decoder.",  # noqa: E501
        },
        orig_to_new_prefix={
            "model.mm_projector.": "model.vision_projector.",
            "model.": "",
        },
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
        if isinstance(getattr(config, "vision_config", None), (dict, PretrainedConfig)):
            vision_config = config.vision_config
            vision_config.anyres = config.anyres
            vision_config.max_num_grids = config.max_num_grids
            vision_config.update({"torch_dtype": config.torch_dtype})
            vision_projector_input_dim = vision_config.hidden_size
            vision_projector_output_dim = config.text_config.hidden_size
            if vision_config.model_type == "qwen2_5_vl_visual":
                if get_compute_capability() >= 8.0:
                    vision_config._attn_implementation = "flash_attention_2"
                vision_projector_input_dim = vision_config.out_hidden_size
            # initialize continuous_vision_encoder
            vision_model = initialize_continuous_vision_encoder(
                vision_config=vision_config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                norm_eps=getattr(config.text_config, "rms_norm_eps", 1e-6),
                prefix=maybe_prefix(prefix, "visual"),
            )
            # initialize vision_projector            
            if config.vision_projector_type == "linear":
                vision_projector = nn.Linear(
                    in_features=vision_projector_input_dim,
                    out_features=vision_projector_output_dim,
                )
            elif config.vision_projector_type == "cabstractor":
                vision_projector = HyperCLOVAXOmniCAbstractor(
                    num_queries=config.num_queries_vis_abstractor,
                    num_input_tokens=(
                        vision_config.image_size // vision_config.patch_size
                    )
                    ** 2,
                    encoder_hidden_size=vision_projector_input_dim,
                    hidden_size=vision_projector_input_dim,
                    output_hidden_size=vision_projector_output_dim,
                    pos_emb=config.proj_pos_emb,
                    prenorm=config.proj_prenorm,
                )
                vision_projector.pos_emb.to(config.torch_dtype)
            elif config.vision_projector_type == "qwen_merger":
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLPatchMerger,
                )

                vision_projector = Qwen2_5_VLPatchMerger(
                    dim=vision_projector_output_dim,
                    context_dim=vision_projector_input_dim,
                )

                def new_forward(self, inputs) -> torch.Tensor:
                    x, window_index = inputs
                    x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
                    reverse_indices = torch.argsort(window_index)
                    x = x[reverse_indices, :]
                    return x

                vision_projector.forward = types.MethodType(
                    new_forward, vision_projector
                )
            else:
                vision_projector = HyperCLOVAXOmniMlp(
                    config.vision_projector_type,
                    vision_projector_input_dim,
                    hidden_features=vision_projector_input_dim,
                    out_features=vision_projector_output_dim,
                )
            vision_projector.to(vision_model.dtype)
        return (
            vision_model, 
            vision_projector, 
            vision_config,
        )

    def _init_discrete_vision(self, config, vllm_config):
        """Initialize discrete vision encoder.

        Returns:
            (discrete_vision_model, discrete_vision_codebook_size,
             discrete_image_unit_0_id, discrete_vision_config)
        """
        discrete_vision_config = getattr(config, "discrete_vision_config", None)
        discrete_vision_model = None
        discrete_vision_codebook_size = None
        discrete_image_unit_0_id = None
        discrete_image_ratio_token_ids = None
        discrete_image_eol_token_id = None
        discrete_image_eof_token_id = None
        if isinstance(discrete_vision_config, (dict, PretrainedConfig)):
            discrete_vision_config.update({"torch_dtype": torch.float32})
            discrete_vision_model = AutoModel.from_config(
                discrete_vision_config,
                trust_remote_code=True,
            )
            discrete_vision_codebook_size = vllm_config.model_config.hf_config.discrete_vision_config.codebook_size
            discrete_image_unit_0_id = (
                vllm_config.model_config.hf_config.discrete_image_unit_0_id
            )
            discrete_image_ratio_token_ids = vllm_config.model_config.hf_config.discrete_image_ratio_token_ids  
            discrete_image_eol_token_id = vllm_config.model_config.hf_config.discrete_image_eol_token_id
            discrete_image_eof_token_id = vllm_config.model_config.hf_config.discrete_image_eof_token_id
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
            discrete_image_ratio_token_ids,
            discrete_image_eol_token_id,
            discrete_image_eof_token_id,
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
            video_audio_compressor_config = getattr(config, "video_audio_compressor_config", None)
            video_audio_compressor_type = getattr(config, "video_audio_compressor_type", None)
            if (
                video_audio_compressor_config is not None
                and video_audio_compressor_type == "mambamia"
            ):
                video_audio_compressor = AutoModel.from_config(
                    video_audio_compressor_config,
                    trust_remote_code=True,
                )
                video_audio_compressor.to(audio_model.dtype)
        return (
            audio_model,
            audio_projector,
            audio_config,
            video_audio_compressor,
            video_audio_compressor_config,
        )

    def _init_discrete_audio(self, config, vllm_config):
        """Initialize discrete audio encoder.

        Returns:
            (discrete_audio_model, discrete_audio_unit_0_id,
             discrete_audio_config)
        """
        discrete_audio_config = getattr(config, "discrete_audio_config", None)
        discrete_audio_model = None
        discrete_audio_unit_0_id = None
        if isinstance(discrete_audio_config, (dict, PretrainedConfig)):
            discrete_audio_config.update({"torch_dtype": torch.float32})
            discrete_audio_model = AutoModel.from_config(
                discrete_audio_config,
                trust_remote_code=True,
            )
            discrete_audio_unit_0_id = (
                vllm_config.model_config.hf_config.discrete_audio_unit_0_id
            )
        return (
            discrete_audio_model, 
            discrete_audio_unit_0_id, 
            discrete_audio_config,
        )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # set text_config
        text_config = self.hf_config.text_config
        if text_config.model_type in ["llama", "hyperclovax", "gpt2"]:
            text_config._attn_implementation = self.hf_config._attn_implementation
        if text_config.model_type != "hyperclovax":
            text_config.logits_scaling = 1.0
        if getattr(text_config, "padded_vocab_size", None):
            text_config.vocab_size = text_config.padded_vocab_size
        text_config.update({"torch_dtype": self.hf_config.torch_dtype})
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
                self.vision_config,
            ) = self._init_continuous_vision(
                self.hf_config, quant_config, multimodal_config, prefix
            )
            (
                self.discrete_vision_model,
                self.discrete_vision_codebook_size,
                self.discrete_image_unit_0_id,
                self.discrete_image_ratio_token_ids,
                self.discrete_image_eol_token_id,
                self.discrete_image_eof_token_id,
                self.discrete_vision_config,
            ) = self._init_discrete_vision(self.hf_config, vllm_config)

        # Audio tower
        with self._mark_tower_model(vllm_config, "audio"):
            (
                self.audio_model,
                self.audio_projector,
                self.audio_config,
                self.video_audio_compressor,
                self.video_audio_compressor_config,
            ) = self._init_continuous_audio(self.hf_config, prefix)
            (
                self.discrete_audio_model,
                self.discrete_audio_unit_0_id,
                self.discrete_audio_config,
            ) = self._init_discrete_audio(self.hf_config, vllm_config)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image_start|><|IMAGE_PAD|><|image_end|>"
        if modality.startswith("video"):
            return "<|video_start|><|VIDEO_PAD|><|video_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|AUDIO_PAD|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def _process_audio_input(
        self,
        audio_input: HyperCLOVAXOmniAudioInputs,
    ) -> tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["audio_embeds"]
        
        # continuous audio_embeds
        audio_values = audio_input["audio_values"]
        audio_attention_mask = audio_input["audio_attention_mask"]
        num_audio_tokens = audio_input["num_audio_tokens"]
        # audio_masks = audio_input["audio_masks"]
        
        with set_forward_context(None, self.vllm_config):
            audio_embeds = self.audio_model(audio_values, attention_mask=audio_attention_mask).last_hidden_state
        audio_embeds = self.audio_projector(audio_embeds)
        audio_start_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.audio_start_token_id, ]).to(device=audio_embeds.device, dtype=torch.long))
        audio_end_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.audio_end_token_id, ]).to(device=audio_embeds.device, dtype=torch.long))
        
        # discrete audio_embeds
        discrete_audio_embeds = None
        discrete_audio_start_embed = None
        discrete_audio_end_embed = None
        if (
            self.discrete_audio_model is not None
            and audio_input.get("discrete_audio_values", None) is not None
        ):
            discrete_audio_values = audio_input["discrete_audio_values"]
            num_discrete_audio_tokens = audio_input["num_discrete_audio_tokens"]
            with set_forward_context(None, self.vllm_config):
                discrete_audio_token_ids = self.discrete_audio_model.forward(
                    discrete_audio_values.to(dtype=self.discrete_audio_model.dtype),
                )
            # index shift to backbone embedding matrix
            discrete_audio_token_ids += self.discrete_audio_unit_0_id
            discrete_audio_token_ids = torch.clamp(
                input=discrete_audio_token_ids,
                min=0,
                max=self.language_model.config.vocab_size - 1,
            )
            
            discrete_audio_embeds = self.embed_input_ids(input_ids=discrete_audio_token_ids)
            discrete_audio_start_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.discrete_audio_start_token_id, ]).to(device=discrete_audio_embeds.device, dtype=torch.long))
            discrete_audio_end_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.discrete_audio_end_token_id, ]).to(device=discrete_audio_embeds.device, dtype=torch.long))

        audio_embeds = list(audio_embeds)
        for item_idx in range(0, len(audio_embeds)):
            _audio_embeds = torch.cat([
                audio_start_embed,
                audio_embeds[item_idx],
                audio_end_embed,
            ], dim=0)
            if discrete_audio_embeds is not None:
                _audio_embeds = torch.cat([
                    discrete_audio_start_embed,
                    discrete_audio_embeds[item_idx],
                    discrete_audio_end_embed,
                    _audio_embeds,
                ], dim=0)
            audio_embeds[item_idx] = _audio_embeds

        return audio_embeds

    def _process_image_input(
        self,
        image_input: HyperCLOVAXOmniImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        # continuous image_embeds
        pixel_values = image_input["pixel_values"]
        image_grid_thw = image_input["image_grid_thw"]
        num_image_tokens = image_input["num_image_tokens"]
        assert image_grid_thw.ndim == 2
        image_grid_thw = image_grid_thw.tolist()

        with set_forward_context(None, self.vllm_config):
            image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        image_embeds = self.vision_projector(image_embeds)
        image_start_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.image_start_token_id, ]).to(device=image_embeds.device, dtype=torch.long))
        image_end_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.image_end_token_id, ]).to(device=image_embeds.device, dtype=torch.long))
        image_embeds = image_embeds.split(num_image_tokens)
        
        # discrete image_embeds
        discrete_image_embeds = None
        discrete_image_start_embed = None
        discrete_image_end_embed = None
        if (
            self.discrete_vision_model is not None
            and image_input.get("discrete_pixel_values", None) is not None
        ):
            discrete_pixel_values = image_input["discrete_pixel_values"]
            discrete_image_ratios = image_input["discrete_image_ratios"]
            with set_forward_context(None, self.vllm_config):
                discrete_image_token_ids = self.discrete_vision_model(
                    discrete_pixel_values.to(dtype=self.discrete_vision_model.dtype),
                )["encoded"]
            # prevent overflow according to discrete_vision_codebook_size
            discrete_image_token_ids = torch.clamp(discrete_image_token_ids, 0, self.discrete_vision_codebook_size - 1)
            # index shift to backbone embedding matrix
            discrete_image_token_ids += self.discrete_image_unit_0_id
            discrete_image_token_ids = torch.clamp(
                input=discrete_image_token_ids,
                min=0,
                max=self.language_model.config.vocab_size - 1,
            )
           
            # scatter eol_token every `grid` tokens (grid = sqrt(bottleneck_token_num) = 27)
            # reshape to rows, append an eol column, then flatten: [1, 729] -> [1, 756]
            grid = int(round(discrete_image_token_ids.shape[-1] ** 0.5))
            _batch = discrete_image_token_ids.shape[0]
            discrete_image_eol_token_ids = torch.full(
                (_batch, grid, 1),
                self.hf_config.discrete_image_eol_token_id,
                device=discrete_image_token_ids.device,
                dtype=torch.long,
            )
            discrete_image_token_ids = torch.cat([
                discrete_image_token_ids.reshape(_batch, grid, grid),
                discrete_image_eol_token_ids,
            ], dim=2).reshape(_batch, grid * (grid + 1))
            
            # concat ratio_token, eof_token 
            discrete_image_ratio_token_ids = torch.tensor([
                self.hf_config.discrete_image_ratio_token_ids[f'{_width}:{_height}'] 
                for _width, _height in discrete_image_ratios
            ]).to(device=discrete_image_token_ids.device, dtype=torch.long)
            discrete_image_eof_token_ids = torch.tensor([self.hf_config.discrete_image_eof_token_id, ]).to(device=discrete_image_token_ids.device, dtype=torch.long)
            discrete_image_token_ids = torch.concat([
                discrete_image_ratio_token_ids.unsqueeze(dim=0),
                discrete_image_token_ids,
                discrete_image_eof_token_ids.unsqueeze(dim=0),
            ], dim=1)

            discrete_image_embeds = self.embed_input_ids(input_ids=discrete_image_token_ids)
            discrete_image_start_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.discrete_image_start_token_id, ]).to(device=discrete_image_embeds.device, dtype=torch.long))
            discrete_image_end_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.discrete_image_end_token_id, ]).to(device=discrete_image_embeds.device, dtype=torch.long))
            
        image_embeds = list(image_embeds)
        for item_idx in range(0, len(image_embeds)):
            _image_embeds = torch.cat([
                image_start_embed,
                image_embeds[item_idx],
                image_end_embed,
            ], dim=0)
            if discrete_image_embeds is not None:
                _image_embeds = torch.cat([
                    discrete_image_start_embed,
                    discrete_image_embeds[item_idx],
                    discrete_image_end_embed,
                    _image_embeds,
                ], dim=0)
            image_embeds[item_idx] = _image_embeds

        return image_embeds

    def _process_video_input(
        self,
        video_input: HyperCLOVAXOmniVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"]

        # continuous video_embeds
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        num_video_tokens = video_input["num_video_tokens"]
        assert video_grid_thw.ndim == 2
        video_grid_thw = video_grid_thw.tolist()

        with set_forward_context(None, self.vllm_config):
            video_embeds = self.vision_model(pixel_values_videos, grid_thw=video_grid_thw)
        video_embeds = self.vision_projector(video_embeds)
        video_start_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.video_start_token_id, ]).to(device=video_embeds.device, dtype=torch.long))
        video_end_embed = self.embed_input_ids(input_ids=torch.tensor([self.hf_config.video_end_token_id, ]).to(device=video_embeds.device, dtype=torch.long))
        video_embeds = video_embeds.split(num_video_tokens)
        
        video_embeds = list(video_embeds)
        for item_idx in range(0, len(video_embeds)):
            _video_embeds = torch.cat([
                video_start_embed,
                video_embeds[item_idx],
                video_end_embed,
            ], dim=0)
            video_embeds[item_idx] = _video_embeds
            
        return video_embeds

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
                audio_embeds=audio_embeddings,
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

        if pixel_values is None and image_embeddings is None:
            return None

        if pixel_values is not None:
            return HyperCLOVAXOmniImagePixelInputs(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                discrete_pixel_values=discrete_pixel_values,
                discrete_image_ratios=discrete_image_ratios,
                num_image_tokens=num_image_tokens,
                num_discrete_image_tokens=num_discrete_image_tokens,
            )

        if image_embeddings is not None:
            return HyperCLOVAXOmniImageEmbeddingInputs(
                image_embeds=image_embeddings,
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

        if pixel_values_videos is None and video_embeddings is None:
            return None

        if pixel_values_videos is not None:
            return HyperCLOVAXOmniVideoPixelInputs(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                num_video_tokens=num_video_tokens,
            )

        if video_embeddings is not None:
            return HyperCLOVAXOmniVideoEmbeddingInputs(
                video_embeds=video_embeddings,
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
            skip_prefixes.extend(["vision_projector."])
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
