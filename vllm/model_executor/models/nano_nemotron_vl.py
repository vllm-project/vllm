# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py
# under Apache-2.0 License
#     LICENSE is in root directory.
# --------------------------------------------------------

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from io import BytesIO
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalInput
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsMultiModalPruning,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.parakeet import ParakeetExtractor, ProjectedParakeet
from vllm.model_executor.models.radio import RadioModel, calc_seq_lens
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_retained_tokens_count,
    compute_retention_mask,
)
from vllm.multimodal.inputs import (
    AudioItem,
    BatchedTensorInputs,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.media.audio import load_audio_pyav
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    ProcessorInputs,
    TimingContext,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.transformers_utils.processors.internvl import get_internvl_target_ratios
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    AUDIO_CONTEXT,
    IMG_CONTEXT,
    IMG_END,
    IMG_START,
    BaseNanoNemotronVLProcessor,
    DynamicResolutionImageTiler,
    NanoNemotronVLProcessor,
    get_video_target_size_and_feature_size,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import _merge_multimodal_embeddings

logger = init_logger(__name__)

MAX_AUDIO_LEN_S = 10 * 60  # 10 minutes


class NanoNemotronVLAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - c: Number of audio clips (possibly flattened across audio items)
        - b: Number of original audio items
        - t: Audio feature length
        - f: Feature size (mel bins)
    """

    type: Literal["audio_features"] = "audio_features"
    input_audio_features: Annotated[torch.Tensor, TensorShape("c", "t", "f")]
    feature_attention_mask: Annotated[torch.Tensor, TensorShape("c", "t")]
    audio_num_clips: list[int]


class NanoNemotronVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class NanoNemotronVLImagePixelInputsDynamic(TensorSchema):
    """
    Dynamic-resolution image inputs.

    imgs_sizes: per-image (height, width) in pixels.
    num_tokens_per_image: per-image number of embedding tokens (post downsample).
    """

    type: Literal["pixel_values_dynamic"] = "pixel_values_dynamic"
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bn", "h", "w")]
    imgs_sizes: list[tuple[int, int]]
    num_tokens_per_image: list[int]


class NanoNemotronVLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of images
        - f: Total image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


NanoNemotronVLImageInputs: TypeAlias = (
    NanoNemotronVLImagePixelInputs
    | NanoNemotronVLImagePixelInputsDynamic
    | NanoNemotronVLImageEmbeddingInputs
)


class NanoNemotronVLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - bvf: Batch size * number of videos * num_frames
        - bn: Batch size * number of videos
        - f: Number of frames
        - c: Number of channels (3)
        - h: Height of each video frame
        - w: Width of each video frame
    """

    type: Literal["pixel_values_videos"]
    pixel_values_flat: Annotated[torch.Tensor, TensorShape("bvf", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]
    frames_indices: Annotated[torch.Tensor, TensorShape("bvf")]
    frame_duration_ms: Annotated[torch.Tensor, TensorShape("bn")]


class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of videos
        - f: Total video feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["video_embeds"]
    data: Annotated[torch.Tensor | list[torch.Tensor], TensorShape("n", "f", "h")]


NanoNemotronVLVideoInputs: TypeAlias = (
    NanoNemotronVLVideoPixelInputs | NanoNemotronVLVideoEmbeddingInputs
)


class NanoNemotronVLProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor:
        return self.ctx.init_processor(
            NanoNemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            video_pruning_rate=self.get_video_pruning_rate(),
            max_model_len=self.ctx.model_config.max_model_len,
            **kwargs,
        )

    @cached_property
    def is_dynamic_tiler(self) -> bool:
        return self.get_hf_processor().dynamic_tiler is not None

    @cached_property
    def supports_video(self):
        return self.get_hf_processor().supports_video

    def get_video_token(self) -> str | None:
        return IMG_CONTEXT

    def get_video_pruning_rate(self) -> float | None:
        return self.ctx.get_mm_config().video_pruning_rate

    @property
    def audio_extractor(self) -> ParakeetExtractor | None:
        return self.get_hf_processor().audio_extractor

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        image_limit = {"image": None}
        video_limit = {"video": None} if self.supports_video else {}
        audio_limit = {"audio": None} if self.audio_extractor is not None else {}
        return {**image_limit, **video_limit, **audio_limit}

    def get_data_parser(self):
        target_sr = None
        target_channels = None
        if extractor := self.audio_extractor:
            target_sr = extractor.sampling_rate
            target_channels = 1

        return MultiModalDataParser(
            video_needs_metadata=True,
            target_sr=target_sr,
            target_channels=target_channels,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_image_size_with_most_features(self, max_num_tiles: int) -> ImageSize:
        processor = self.get_hf_processor()

        base_size = processor.image_size
        target_ratios = get_internvl_target_ratios(1, max_num_tiles)

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = processor.get_num_image_tokens(
                image_width=width, image_height=height, max_num_tiles=max_num_tiles
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width, height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        # Use default max_num_tiles for max tokens calculation
        max_num_tiles = processor.max_num_tiles
        target_width, target_height = self.get_image_size_with_most_features(
            max_num_tiles
        )

        return processor.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            max_num_tiles=max_num_tiles,
        )

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        processor = self.get_hf_processor()  # we get the CustomProcessor here
        T = processor.video_temporal_patch_size

        max_image_tokens = self.get_max_image_tokens() * max_images
        tokens_per_tubelet = processor.num_video_token
        max_total_tubelets = (seq_len - max_image_tokens) // tokens_per_tubelet
        max_tubelets_per_video = max_total_tubelets // max(max_videos, 1)
        max_frames_per_video = max_tubelets_per_video * T
        return max(max_frames_per_video, 1)


class NanoNemotronVLMultiModalProcessor(
    BaseMultiModalProcessor[NanoNemotronVLProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Bypass `call_hf_processor_mm_only` by no-op overriding`_call_hf_processor`,
        so it chooses this path:
        `type(self)._call_hf_processor != BaseMultiModalProcessor._call_hf_processor`
        """
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_image_fields_config(self, hf_inputs: BatchFeature):
        if self.info.is_dynamic_tiler:
            pixel_values_flat = MultiModalFieldConfig.batched("image")
        else:
            image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
            pixel_values_flat = MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches
            )

        return dict(
            pixel_values_flat=pixel_values_flat,
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_tokens_per_image=MultiModalFieldConfig.batched("image"),
            imgs_sizes=MultiModalFieldConfig.batched("image"),
        )

    def _get_video_fields_config(self, hf_inputs: BatchFeature):
        video_num_patches = hf_inputs.get("video_num_patches", torch.empty(0))

        return dict(
            pixel_values_flat_video=MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_patches
            ),
            video_num_patches=MultiModalFieldConfig.batched("video"),
            frames_indices=MultiModalFieldConfig.batched("video"),
            frame_duration_ms=MultiModalFieldConfig.batched("video"),
        )

    def _get_audio_fields_config(self, hf_inputs: BatchFeature):
        audio_num_clips = torch.as_tensor(hf_inputs["audio_num_clips"])

        return dict(
            input_audio_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_num_clips
            ),
            feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "audio", audio_num_clips
            ),
            audio_num_clips=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = self._get_image_fields_config(hf_inputs)
        if self.info.supports_video:
            fields |= self._get_video_fields_config(hf_inputs)
        if self.info.audio_extractor:
            fields |= self._get_audio_fields_config(hf_inputs)

        return fields

    def _get_prompt_repl_image(
        self,
        mm_items: MultiModalDataItems,
        hf_processor: NanoNemotronVLProcessor,
        out_mm_data: BatchedTensorInputs,
    ):
        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            elif tiler := hf_processor.dynamic_tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            else:
                image_size = images.get_image_size(item_idx)
                max_num_tiles = hf_processor.max_num_tiles
                feature_size = hf_processor.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    max_num_tiles=max_num_tiles,
                )

            num_patches = None
            local_image_num_patches = image_num_patches
            if isinstance(local_image_num_patches, torch.Tensor):
                local_image_num_patches = local_image_num_patches.tolist()
            if isinstance(local_image_num_patches, (list, tuple)) and item_idx < len(
                local_image_num_patches
            ):
                num_patches = int(local_image_num_patches[item_idx])

            return hf_processor.get_image_repl(feature_size, num_patches)

        return PromptReplacement(
            modality="image",
            target="<image>",
            replacement=get_image_replacement,
        )

    def _get_prompt_repl_video(
        self,
        mm_items: MultiModalDataItems,
        hf_processor: NanoNemotronVLProcessor,
        out_mm_data: BatchedTensorInputs,
    ):
        if "video_num_patches" in out_mm_data:
            video_num_patches = out_mm_data["video_num_patches"]
            assert isinstance(video_num_patches, torch.Tensor)
            video_num_patches = video_num_patches.tolist()
        else:
            video_num_patches = []

        def get_video_replacement(item_idx: int):
            video, metadata = mm_items["video"][item_idx]
            patch_size = hf_processor.config.patch_size
            downsample_ratio = hf_processor.config.downsample_ratio
            target_patches = hf_processor.video_target_num_patches

            if target_patches is not None and video is not None and video.shape[0] > 0:
                orig_h, orig_w = video.shape[1], video.shape[2]
                _, _, feature_size = get_video_target_size_and_feature_size(
                    orig_w=orig_w,
                    orig_h=orig_h,
                    target_patches=target_patches,
                    maintain_aspect_ratio=hf_processor.video_maintain_aspect_ratio,
                    patch_size=patch_size,
                    downsample_ratio=downsample_ratio,
                )
            else:
                feature_size = hf_processor.num_image_token
            num_patches = video_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

            T = hf_processor.video_temporal_patch_size
            if T > 1 and num_patches is not None:
                num_tubelets = math.ceil(num_patches / T)
            else:
                num_tubelets = num_patches

            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                # Start of EVS-specific code
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=feature_size,
                    num_frames=num_tubelets,
                    q=video_pruning_rate,
                )
                # Here we just need placeholders that won't actually be replaced -
                # we just need to make sure the total number of tokens is correct
                # assign all tokens to the first frame
                tokens_per_frame = [num_tokens] + [0] * (num_tubelets - 1)

                # End of EVS-specific code
            else:
                tokens_per_frame = [feature_size] * num_tubelets

            frame_duration_ms = int(1000 / metadata["fps"])
            return hf_processor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                frames_indices=metadata["frames_indices"],
                frame_duration_ms=frame_duration_ms,
                tokenizer=hf_processor.tokenizer,
                img_start_token_ids=hf_processor._img_start_token_ids,
                img_end_token_ids=hf_processor._img_end_token_ids,
                img_context_token_ids=hf_processor._img_context_token_ids,
                video_temporal_patch_size=T,
            )

        return PromptReplacement(
            modality="video",
            target="<video>",
            replacement=get_video_replacement,
        )

    def _get_prompt_repl_audio(
        self,
        mm_items: MultiModalDataItems,
        hf_processor: NanoNemotronVLProcessor,
        out_mm_data: BatchedTensorInputs,
    ):
        def get_audio_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            return hf_processor.get_audio_repl(audios.get(item_idx))

        return PromptReplacement(
            modality="audio",
            target=AUDIO_CONTEXT,
            replacement=get_audio_replacement,
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()

        prompt_repls = [
            self._get_prompt_repl_image(mm_items, hf_processor, out_mm_data),
        ]
        if self.info.supports_video:
            prompt_repls.append(
                self._get_prompt_repl_video(mm_items, hf_processor, out_mm_data)
            )
        if self.info.audio_extractor:
            prompt_repls.append(
                self._get_prompt_repl_audio(mm_items, hf_processor, out_mm_data)
            )

        return prompt_repls

    def _extract_audio_from_videos(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[MultiModalDataItems, list[AudioItem]]:
        """Extract audio tracks from video bytes in *mm_items*.

        Returns:
            The augmented *mm_items* (with audio added) and the list of
            extracted audio items.
        """
        videos = mm_items.get_items("video", VideoProcessorItems)
        assert isinstance(videos.metadata, list)
        metadata_list = videos.metadata

        audio_items: list[AudioItem] = []
        for metadata in metadata_list:
            video_bytes = metadata.get("original_video_bytes")
            if video_bytes is None or len(video_bytes) == 0:
                raise ValueError(
                    "Cannot extract audio from video: original_video_bytes is "
                    "missing or empty. When using use_audio_in_video=True, "
                    "video must be loaded with keep_video_bytes=True (e.g. via "
                    "the chat API with a model that sets use_audio_in_video)."
                )
            audio_items.append(load_audio_pyav(BytesIO(video_bytes)))

        # Create a new VideoProcessorItems with metadata that does not contain
        # the large video bytes, to avoid modifying the input `mm_items`.
        new_metadata_list = [
            {k: v for k, v in meta.items() if k != "original_video_bytes"}
            for meta in metadata_list
        ]
        new_videos = VideoProcessorItems(data=videos.data, metadata=new_metadata_list)

        audio_parsed = self.data_parser.parse_mm_data({"audio": audio_items})

        # Create a new MultiModalDataItems with the new video and audio items.
        new_mm_items_dict = {**mm_items, **audio_parsed, "video": new_videos}
        mm_items = MultiModalDataItems(new_mm_items_dict)

        return mm_items, audio_items

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        use_audio_in_video = bool(
            inputs.hf_processor_mm_kwargs.get("use_audio_in_video", False)
        )
        inputs.hf_processor_mm_kwargs = {
            k: v
            for k, v in inputs.hf_processor_mm_kwargs.items()
            if k != "use_audio_in_video"
        }

        if not (
            use_audio_in_video
            and "video" in inputs.mm_data_items
            and "audio" not in inputs.mm_data_items
        ):
            return super().apply(inputs, timing_ctx)

        mm_items, audio_items = self._extract_audio_from_videos(inputs.mm_data_items)
        inputs.mm_data_items = mm_items

        prompt = inputs.prompt
        tokenizer = self.info.get_tokenizer()
        if not isinstance(prompt, str):
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)

        for _ in audio_items:
            prompt = prompt.replace("<video>", "<video>" + AUDIO_CONTEXT, 1)

        inputs.prompt = tokenizer.encode(prompt, add_special_tokens=False)

        if inputs.tokenization_kwargs is None:
            inputs.tokenization_kwargs = {}

        # Bypass the cached path: the HF processor must receive the
        # prompt (with injected <so_embedding>) and the audio data
        # together so it can perform audio-token replacement natively.
        (
            prompt_ids,
            mm_info,
            is_update_applied,
        ) = self._apply_hf_processor(inputs, timing_ctx)

        with timing_ctx.record("apply_prompt_updates"):
            prompt_ids, mm_placeholders = self._maybe_apply_prompt_updates(
                mm_items=mm_items,
                prompt_ids=prompt_ids,
                mm_kwargs=mm_info.kwargs,
                mm_prompt_updates=mm_info.prompt_updates,
                is_update_applied=is_update_applied,
            )

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInput(
            type="multimodal",
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_info.kwargs,
            mm_hashes=mm_info.hashes,
            mm_placeholders=mm_placeholder_ranges,
        )


class NanoNemotronVLDummyInputsBuilder(
    BaseDummyInputsBuilder[NanoNemotronVLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)

        return (
            "<image>" * num_images + "<video>" * num_videos + AUDIO_CONTEXT * num_audios
        )

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
        overrides: VideoDummyOptions | None = None,
    ) -> list[VideoItem]:
        videos = super()._get_dummy_videos(
            width=width,
            height=height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        videos = [v.copy() for v in videos]

        video_items = []
        for video in videos:
            video_num_frames = video.shape[0]
            video_metadata = {
                "fps": 2,
                "duration": video_num_frames / 2.0,
                "total_num_frames": video_num_frames,
                "frames_indices": list(range(video_num_frames)),
                "video_backend": "opencv_dynamic",
                "do_sample_frames": False,
            }
            video_items.append((video, video_metadata))

        return video_items

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        if tiler := processor.dynamic_tiler:
            budget = tiler.max_num_tokens_available(text_prompt_length=num_images)
            target_width, target_height = (
                tiler.width_and_height_for_max_num_tokens_available(budget)
            )
        else:
            max_num_tiles = 12
            target_width, target_height = self.info.get_image_size_with_most_features(
                max_num_tiles
            )

        image_overrides = mm_options.get("image")

        dummy_image = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }

        if self.info.supports_video:
            config = self.info.get_hf_config()
            image_size: int = config.force_image_size

            # When video_target_num_patches is set the per-frame pixel
            # resolution can exceed image_size.  Use the actual target
            # dimensions so that profiling sees the correct upper bound.
            if processor.video_target_num_patches is not None:
                target_w, target_h, _ = get_video_target_size_and_feature_size(
                    orig_w=image_size,
                    orig_h=image_size,
                    target_patches=processor.video_target_num_patches,
                    maintain_aspect_ratio=processor.video_maintain_aspect_ratio,
                    patch_size=config.patch_size,
                    downsample_ratio=config.downsample_ratio,
                )
                video_width, video_height = target_w, target_h
            else:
                video_width, video_height = image_size, image_size

            target_num_frames = self.info.get_num_frames_with_most_features(
                seq_len, mm_counts
            )
            mm_config = self.info.ctx.get_mm_config()
            if num_frames := mm_config.media_io_kwargs.get("video", {}).get(
                "num_frames"
            ):
                assert num_frames > 0
                target_num_frames = num_frames
            num_videos = mm_counts.get("video", 0)
            video_overrides = mm_options.get("video")
            dummy_video = {
                "video": self._get_dummy_videos(
                    width=video_width,
                    height=video_height,
                    num_frames=target_num_frames,
                    num_videos=num_videos,
                    overrides=video_overrides,
                )
            }
        else:
            dummy_video = {}

        if extractor := self.info.audio_extractor:
            num_audios = mm_counts.get("audio", 0)
            audio_overrides = mm_options.get("audio") if mm_options else None
            tokens_per_audio = max(1, seq_len // max(num_audios, 1))
            max_audio_num_samples = MAX_AUDIO_LEN_S * extractor.sampling_rate
            calculated_max_audio_num_samples = extractor.audio_length(tokens_per_audio)
            audio_len = min(max_audio_num_samples, calculated_max_audio_num_samples)
            dummy_audio = {
                "audio": self._get_dummy_audios(
                    length=audio_len,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )
            }
        else:
            dummy_audio = {}

        return {**dummy_image, **dummy_video, **dummy_audio}


@MULTIMODAL_REGISTRY.register_processor(
    NanoNemotronVLMultiModalProcessor,
    info=NanoNemotronVLProcessingInfo,
    dummy_inputs=NanoNemotronVLDummyInputsBuilder,
)
class NemotronH_Nano_VL_V2(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsMultiModalPruning
):
    requires_sequential_video_encoding = True
    """Temporarily needed for dynamic res video w/ conv3d, doesn't support bs>1 yet"""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("video"):
            return "<video>"
        if modality.startswith("audio"):
            return AUDIO_CONTEXT
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        multimodal_config = model_config.multimodal_config
        image_size = config.force_image_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.image_tag_type = config.image_tag_type
        self.video_pruning_rate = multimodal_config.video_pruning_rate

        vision_config = getattr(config, "vision_config", config)
        self.video_temporal_patch_size: int = getattr(
            vision_config, "video_temporal_patch_size", 1
        )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        llm_dtype = self.language_model.config.dtype
        assert isinstance(llm_dtype, torch.dtype)
        self.llm_dtype = llm_dtype
        with self._mark_tower_model(vllm_config, {"image", "video", "audio"}):
            self.vision_model = self.get_vit_model_from_radio_config(config).to(
                llm_dtype
            )

            # Construct the vision projection.
            vit_hidden_size = config.vit_hidden_size
            vision_projection_hidden_size = config.projector_hidden_size
            llm_hidden_size = config.text_config.hidden_size

            mlp1 = nn.Sequential(
                RMSNorm(
                    hidden_size=vit_hidden_size
                    * int(round(1 / self.downsample_ratio)) ** 2,
                    eps=1e-5,
                ),
                nn.Linear(
                    vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2,
                    vision_projection_hidden_size,
                    bias=False,
                ),
                ReLUSquaredActivation(),
                nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),
            )
            self.mlp1 = mlp1.to(llm_dtype)
            self.sound_encoder: ProjectedParakeet | None = None
            if getattr(config, "sound_config", None) is not None:
                logger.info_once(
                    "Found sound config, initializing sound encoder for Nemotron AVLM",
                    scope="global",
                )
                self.sound_encoder = ProjectedParakeet(
                    config.sound_config,
                    dtype=llm_dtype,
                    llm_hidden_size=llm_hidden_size,
                    max_model_len=model_config.max_model_len,
                )

        self.config = config
        self.model_config = vllm_config.model_config

        # Pre-tokenize special tokens for video processing
        # to avoid repeated tokenization
        tokenizer = cached_tokenizer_from_config(model_config)
        self._img_start_token_ids = tokenizer.encode(
            IMG_START, add_special_tokens=False
        )
        self._img_end_token_ids = tokenizer.encode(IMG_END, add_special_tokens=False)
        self._img_context_token_ids = tokenizer.encode(
            IMG_CONTEXT, add_special_tokens=False
        )
        self.dynamic_resolution = BaseNanoNemotronVLProcessor.use_dynamic_resolution(
            config
        )
        if self.dynamic_resolution:
            logger.info_once(
                "Dynamic resolution is enabled for NanoNemotronVLProcessor",
                scope="global",
            )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(
            n,
            w,
            int(h * scale_factor),
            int(c / scale_factor),
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale -->
        # N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not "
                "been swapped back, which results in a transposed image.",
                stacklevel=2,
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def pixel_shuffle_dynamic_res(
        self, x: torch.Tensor, *, imgs_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        scale_factor = self.downsample_ratio
        patch_dim = self.patch_size
        seq_lens = calc_seq_lens(imgs_sizes, patch_dim)
        splits = torch.split(x, seq_lens, dim=-2)
        out = []
        for i, sv in enumerate(splits):
            h = imgs_sizes[i][0] // patch_dim
            w = imgs_sizes[i][1] // patch_dim
            sv = sv.reshape(sv.shape[0], h, w, -1)

            n, h, w, c = sv.size()

            sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
            sv = sv.permute(0, 2, 1, 3).contiguous()
            sv = sv.view(
                n,
                int(w * scale_factor),
                int(h * scale_factor),
                int(c / (scale_factor * scale_factor)),
            )

            if self.ps_version == "v2":
                sv = sv.permute(0, 2, 1, 3).contiguous()

            sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
            out.append(sv)

        x = torch.cat(out, dim=-2)

        return x

    def extract_feature_dynamic(
        self, pixel_values: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ):
        """Dynamic resolution extract_feature for images."""
        _, vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        vit_embeds = self.pixel_shuffle_dynamic_res(vit_embeds, imgs_sizes=imgs_sizes)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_feature(
        self,
        pixel_values: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        # Process images in a micro-batch of at most 128 frames per call
        #   This is done on purpose to ensure peak GPU ram usage of huge batch
        #   (namely for really long videos with EVS ON) won't cause any problems
        #   as we don't support chunked prefill for video media
        # When num_frames is provided and temporal_patch_size > 1, consecutive
        #   frames are grouped into tubelets — the batch size must be a multiple
        #   of T so chunk boundaries don't split a tubelet.
        N, _C, H, W = pixel_values.shape

        T = self.video_temporal_patch_size if num_frames is not None else 1
        micro_batch_size = 128 - (128 % T)
        patch_size = self.patch_size
        H_patches = H // patch_size
        W_patches = W // patch_size

        vit_embeds_list = []
        for i in range(0, N, micro_batch_size):
            chunk = pixel_values[i : i + micro_batch_size]
            if num_frames is not None and T > 1:
                _, vit_embeds = self.vision_model(chunk, num_frames=chunk.shape[0])
            else:
                _, vit_embeds = self.vision_model(chunk)
            vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], H_patches, W_patches, -1
            )
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio
            )
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds_list.append(vit_embeds)

        vit_embeds = torch.cat(vit_embeds_list, dim=0)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> NanoNemotronVLImageInputs | None:
        if image_embeds := kwargs.pop("image_embeds", None):
            return NanoNemotronVLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if self.dynamic_resolution:
            pixel_values_flat = DynamicResolutionImageTiler.stack(
                kwargs.pop("pixel_values_flat"), self.patch_size
            )
            return NanoNemotronVLImagePixelInputsDynamic(
                pixel_values_flat=pixel_values_flat, **kwargs
            )
        else:
            return NanoNemotronVLImagePixelInputs(
                num_patches=kwargs.pop("image_num_patches"), **kwargs
            )

    def _process_image_input_dynamic(
        self, image_input: NanoNemotronVLImagePixelInputsDynamic
    ) -> tuple[torch.Tensor, ...]:
        image_embeds = self.extract_feature_dynamic(
            image_input.pixel_values_flat, image_input.imgs_sizes
        )
        num_tokens_per_image = image_input.num_tokens_per_image

        if len(num_tokens_per_image) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        return image_embeds.split(num_tokens_per_image)

    def _process_image_input(
        self, image_input: NanoNemotronVLImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        image_embeds = self.extract_feature(image_input["pixel_values_flat"])
        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _process_video_input(
        self, video_input: NanoNemotronVLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:
        """Process video input and create final embeddings with video content
        and indicator tokens."""
        T = self.video_temporal_patch_size

        if T > 1:
            video_embeddings = self._extract_video_embeddings_temporal(video_input)
        else:
            video_embeddings = self._process_image_input(video_input)

        final_video_embeddings: tuple[torch.Tensor, ...] = ()

        downsample_ratio = self.config.downsample_ratio
        patch_size = self.config.patch_size
        pixel_values = video_input["pixel_values_flat"]
        frame_h, frame_w = pixel_values.shape[-2], pixel_values.shape[-1]
        rows = int(frame_h * downsample_ratio // patch_size)
        cols = int(frame_w * downsample_ratio // patch_size)
        video_pruning_rate = self.video_pruning_rate
        video_num_frames = video_input["num_patches"].tolist()
        video_frames_indices = video_input["frames_indices"].split(video_num_frames)
        # Calculate video feature dimensions (number of frames and
        # their feature size (AKA tokens per frame))
        # TODO: Maybe this can be optimized to avoid the loop?
        for i, single_video_embeddings in enumerate(video_embeddings):
            num_frames = video_num_frames[i]
            frames_indices = video_frames_indices[i].tolist()
            frame_duration_ms = video_input["frame_duration_ms"][i].item()
            num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames
            assert single_video_embeddings.shape[0] % num_tubelets == 0

            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                # Start of EVS-specific code
                retention_mask = compute_retention_mask(
                    single_video_embeddings,
                    video_size_thw=(num_tubelets, rows, cols),
                    spatial_merge_size=1,
                    q=video_pruning_rate,
                )

                # apply retention mask
                single_video_embeddings = single_video_embeddings[retention_mask]

                # calculate the actual number of retained tokens per frame
                retention_mask_thw = retention_mask.reshape(num_tubelets, rows, cols)
                num_tokens_per_frame = (
                    retention_mask_thw.sum(dim=(1, 2)).long().tolist()
                )
                # End of EVS-specific code
            else:
                feature_size = single_video_embeddings.shape[0] // num_tubelets
                num_tokens_per_frame = [feature_size] * num_tubelets

            final_video_embeddings += (
                self._create_final_video_embeddings(
                    single_video_embeddings,
                    num_tokens_per_frame,
                    frames_indices,
                    frame_duration_ms,
                    video_temporal_patch_size=T,
                ),
            )

        return final_video_embeddings

    def _extract_video_embeddings_temporal(
        self, video_input: NanoNemotronVLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:
        """Extract per-video embeddings with temporal compression.

        Each video is processed separately through extract_feature with
        num_frames, which uses the fixed-resolution temporal path in RADIO
        (no attention mask, flash attention).
        """
        pixel_values = video_input["pixel_values_flat"]
        num_frames_per_video = video_input["num_patches"].tolist()
        hidden_size = self.config.text_config.hidden_size

        results: list[torch.Tensor] = []
        frame_offset = 0
        for nf in num_frames_per_video:
            video_frames = pixel_values[frame_offset : frame_offset + nf]
            frame_offset += nf

            vit_embeds = self.extract_feature(video_frames, num_frames=nf)
            results.append(vit_embeds.view(-1, hidden_size))

        return tuple(results)

    def _process_audio_input(
        self, audio_input: NanoNemotronVLAudioFeatureInputs
    ) -> tuple[torch.Tensor, ...]:
        assert self.sound_encoder is not None
        input_audio_features = audio_input.input_audio_features
        feature_attention_mask = audio_input.feature_attention_mask
        audio_num_clips = audio_input.audio_num_clips
        target_device = next(self.sound_encoder.parameters()).device

        input_audio_features = input_audio_features.to(
            dtype=self.llm_dtype, device=target_device
        )
        feature_attention_mask = feature_attention_mask.to(device=target_device)
        sound_embeds = self.sound_encoder(input_audio_features, feature_attention_mask)

        valid_input_lens = feature_attention_mask.sum(dim=1)
        valid_output_lens = self.sound_encoder.encoder._get_subsampling_output_length(
            valid_input_lens
        ).tolist()
        grouped_embeds = []
        clip_offset = 0
        for num_clips in audio_num_clips:
            embeds = []
            for clip_idx in range(clip_offset, clip_offset + num_clips):
                valid_len = valid_output_lens[clip_idx]
                embeds.append(sound_embeds[clip_idx, :valid_len])
            grouped_embeds.append(torch.cat(embeds, dim=0))
            clip_offset += num_clips

        return tuple(grouped_embeds)

    def _create_final_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        num_tokens_per_frame: list[int],
        frames_indices: list[int],
        frame_duration_ms: int,
        video_temporal_patch_size: int = 1,
    ) -> torch.Tensor:
        """Create final embeddings that combine video embeddings with
        text embeddings of indicator tokens.

        These final embeddings contain:
        - Actual video embeddings in positions corresponding to video content
        - Text embeddings for indicator tokens (<img>, </img>, and
          frame separation text) in their respective positions

        These embeddings will replace the placeholder embeddings to create
        input_embeds for the LLM.
        """
        device = video_embeddings.device
        tokenizer = cached_tokenizer_from_config(self.model_config)

        # Generate video replacement token IDs using get_video_repl
        # This tokenizes each frame separator independently, then uses pre-tokenized
        # special tokens to ensure consistent tokenization regardless of
        # num_tokens_per_frame values.
        video_repl = NanoNemotronVLProcessor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            frames_indices=frames_indices,
            frame_duration_ms=frame_duration_ms,
            tokenizer=tokenizer,
            img_start_token_ids=self._img_start_token_ids,
            img_end_token_ids=self._img_end_token_ids,
            img_context_token_ids=self._img_context_token_ids,
            video_temporal_patch_size=video_temporal_patch_size,
        )

        # video_repl.full is a list of token IDs
        repl_token_ids = torch.tensor(video_repl.full, device=device)

        # Get embedding token IDs for image context (use pre-tokenized version)
        embed_token_ids = torch.tensor(self._img_context_token_ids, device=device)

        # Create mask for video embedding positions
        is_video_embed = torch.isin(repl_token_ids, embed_token_ids)

        # Create final video embeddings, merging text embeddings for indicator
        # tokens with video embeddings
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)
        final_video_embeddings = _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=video_embeddings,
            is_multimodal=is_video_embed,
        )

        return final_video_embeddings

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> NanoNemotronVLVideoPixelInputs | None:
        pixel_values_flat_video = kwargs.pop("pixel_values_flat_video", None)
        video_num_patches = kwargs.pop("video_num_patches", None)
        video_embeds = kwargs.pop("video_embeds", None)
        frames_indices = kwargs.pop("frames_indices", None)
        frame_duration_ms = kwargs.pop("frame_duration_ms", None)

        if pixel_values_flat_video is None and video_embeds is None:
            return None

        if video_embeds is not None:
            return NanoNemotronVLVideoEmbeddingInputs(
                type="video_embeds",
                data=video_embeds,
            )

        if pixel_values_flat_video is not None:
            if torch.is_tensor(frames_indices):
                frames_indices = frames_indices.flatten()
            else:
                frames_indices = torch.cat([f.flatten() for f in frames_indices], dim=0)

            if torch.is_tensor(frame_duration_ms):
                frame_duration_ms = frame_duration_ms.flatten()
            else:
                frame_duration_ms = torch.cat(
                    [f.flatten() for f in frame_duration_ms], dim=0
                )

            if (
                torch.is_tensor(pixel_values_flat_video)
                and pixel_values_flat_video.ndim == 5
            ):
                # batched._reduce_data stacked same-shape videos into
                # [num_videos, nf, 3, H, W]; unstack back to a list so the
                # same-H,W cat path below handles it uniformly.
                pixel_values_flat_video = list(pixel_values_flat_video)

            if not torch.is_tensor(pixel_values_flat_video):
                pixel_values_flat_video = torch.cat(pixel_values_flat_video, dim=0)

            expected_h = pixel_values_flat_video.shape[-2]
            expected_w = pixel_values_flat_video.shape[-1]
            num_frames = video_num_patches[0].item()
            resolve_bindings = {"h": expected_h, "w": expected_w, "f": num_frames}

            return NanoNemotronVLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_flat=pixel_values_flat_video,
                num_patches=video_num_patches,
                frames_indices=frames_indices,
                frame_duration_ms=frame_duration_ms,
                resolve_bindings=resolve_bindings,
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}
        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values_flat", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_flat_video",) and "videos" not in modalities:
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)
            if (
                input_key
                in (
                    "input_audio_features",
                    "feature_attention_mask",
                    "audio_num_clips",
                )
                and "audios" not in modalities
            ):
                modalities["audios"] = NanoNemotronVLAudioFeatureInputs(
                    **kwargs, validate=False
                )

        return modalities

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Validate the multimodal input keyword arguments
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if modalities is None:
            return []

        # # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                if image_input["type"] == "image_embeds":
                    image_embeddings = image_input["data"]
                elif self.dynamic_resolution:
                    assert image_input["type"] == "pixel_values_dynamic"
                    image_embeddings = self._process_image_input_dynamic(image_input)
                else:
                    image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_input(video_input)
                multimodal_embeddings += tuple(video_embeddings)
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(audio_input)
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return hidden_states

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["mlp1", "sound_encoder.projection"],
            tower_model=["vision_model", "sound_encoder.encoder"],
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        adapter_dict = dict(self.mlp1.named_parameters())

        def is_llm(name: str) -> bool:
            return name.startswith("language_model")

        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def is_vision_weights(name: str) -> bool:
            return name.startswith("vision_model.radio_model.")

        def is_sound_weights(name: str) -> bool:
            return name.startswith("sound")

        # Separate weights by component
        llm_weights = []
        vision_weights = []
        sound_weights = []

        for name, w in weights:
            if is_llm(name):
                # Strip 'language_model.' prefix for LLM weights
                llm_weights.append((".".join(name.split(".")[1:]), w))
            elif is_adapter_weights((name, w)):
                # Load vision-language adapter weights directly
                trimmed_name = ".".join(name.split(".")[1:])
                param = adapter_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            elif is_vision_weights(name):
                # Convert: vision_model.radio_model.* → radio_model.*
                hf_key = name[len("vision_model.") :]  # Remove "vision_model." prefix
                vision_weights.append((hf_key, w))
            elif is_sound_weights(name):
                assert self.sound_encoder is not None
                sound_weights.append((name, w))

        self.language_model.load_weights(llm_weights)
        self.vision_model.load_weights(vision_weights)
        if self.sound_encoder is not None and len(sound_weights) > 0:
            self.sound_encoder.load_weights(sound_weights)

    def get_vit_model_from_radio_config(self, hf_config):
        hf_config_vision = hf_config.vision_config
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f"Unsupported vit model type: {model_name}")

        preferred_resolution = getattr(hf_config_vision, "preferred_resolution", None)
        image_size = preferred_resolution[0] if preferred_resolution else 224
        patch_size = getattr(hf_config_vision, "patch_size", 16)

        # video_temporal_patch_size and separate_video_embedder are
        # top-level vision_config attributes, not inside args.
        video_temporal_patch_size = getattr(
            hf_config_vision, "video_temporal_patch_size", 1
        )
        separate_video_embedder = getattr(
            hf_config_vision, "separate_video_embedder", True
        )

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            norm_mean=hf_config.norm_mean,
            norm_std=hf_config.norm_std,
            video_temporal_patch_size=video_temporal_patch_size,
            separate_video_embedder=separate_video_embedder,
            **hf_config_vision.args,
        )

        return RadioModel(config=radio_config)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return NemotronHForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig"):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return NemotronHForCausalLM.get_mamba_state_copy_func()
