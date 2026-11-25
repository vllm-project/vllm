# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import BatchFeature

from vllm.config.multimodal import (
    ImageDummyOptions,
    MultiModalDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict, MultiModalInput
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import (
    KIMI_K3_VIDEO_PLACEHOLDER,
    KimiK3Processor,
    encode_k3_text_prompt,
    format_kimi_k3_timestamp,
    get_kimi_k3_max_video_frames,
    get_kimi_k3_max_video_tokens,
    plan_kimi_k3_video,
    supports_kimi_k3_video,
)
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


@VIDEO_LOADER_REGISTRY.register("kimi_k3")
class KimiK3VideoBackend(VideoBackend):
    """Decode only the frames `plan_kimi_k3_video` keeps."""

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        media_proc_cfg = kwargs.get("media_proc_cfg")
        frame_size = kwargs.get("frame_size")
        if (
            not isinstance(media_proc_cfg, dict)
            or frame_size is None
            or not supports_kimi_k3_video(media_proc_cfg)
        ):
            return super().compute_frames_index_to_sample(source, target, **kwargs)

        cfg = dict(media_proc_cfg)
        if target.fps > 0:
            cfg["sample_fps"] = min(cfg["sample_fps"], target.fps)
        if target.num_frames > 0:
            max_frames = cfg.get("max_num_frames_each_video")
            cfg["max_num_frames_each_video"] = (
                target.num_frames
                if max_frames is None
                else min(max_frames, target.num_frames)
            )
        width, height = frame_size
        return plan_kimi_k3_video(
            source.total_frames_num, source.original_fps, width, height, cfg
        )


def get_kimi_k3_media_proc_cfg(model_config: "ModelConfig") -> Mapping[str, Any]:
    return cached_get_image_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls_overrides=KimiK25FusedVisionProcessor,
    ).media_proc_cfg


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def _get_video_fields(inputs: Mapping[str, torch.Tensor]):
    grids = inputs.get("video_grid_thws", torch.empty((0, 3), dtype=torch.int64))
    num_chunks = inputs.get("video_num_chunks", torch.empty(0, dtype=torch.int64))
    chunk_patches = grids.prod(-1).split(num_chunks.tolist()) if len(num_chunks) else ()
    patches_per_video = torch.tensor(
        [int(patches.sum()) for patches in chunk_patches], dtype=torch.int64
    )
    return dict(
        video_pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "video", patches_per_video
        ),
        video_grid_thws=MultiModalFieldConfig.flat_from_sizes(
            "video", num_chunks, keep_on_cpu=True
        ),
        video_chunk_timestamps=MultiModalFieldConfig.flat_from_sizes(
            "video", num_chunks, keep_on_cpu=True
        ),
        video_sizes=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        video_num_chunks=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
    )


def _get_image_fields(inputs: Mapping[str, torch.Tensor]):
    grid = inputs.get("grid_thws", torch.empty((0, 3)))
    num_tokens = inputs.get("image_num_tokens", torch.empty(0)).flatten()
    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid.prod(-1)),
        grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        image_sizes=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        image_num_tokens=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        image_embeds=MultiModalFieldConfig.flat_from_sizes("image", num_tokens),
    )


class KimiK3MultiModalDataParser(MultiModalDataParser):
    embedding_fields = {
        "image": {
            "image_embeds": "values",
            "image_sizes": "metadata",
        },
    }

    def __init__(self, media_proc_cfg: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.media_proc_cfg = media_proc_cfg

    def _parse_image_data(self, data):
        if isinstance(data, dict):
            sizes = data.get("image_sizes")
            if (
                not isinstance(sizes, torch.Tensor)
                or sizes.dtype not in (torch.int32, torch.int64)
                or sizes.ndim != 2
                or sizes.shape[1] != 2
                or not torch.all(sizes > 0)
            ):
                raise ValueError(
                    "image_sizes must contain positive integer [width, height] pairs"
                )
            cfg = self.media_proc_cfg
            num_tokens = torch.tensor(
                [
                    [
                        navit_resize_image(
                            width,
                            height,
                            cfg["patch_size"],
                            cfg["merge_kernel_size"],
                            cfg["in_patch_limit"],
                            cfg["patch_limit_on_one_side"],
                            cfg["fixed_output_tokens"],
                        )["num_tokens"]
                    ]
                    for width, height in sizes.tolist()
                ],
                dtype=torch.int64,
            )
            required, optional = self.embedding_field_sets("image")
            return DictEmbeddingItems(
                {**data, "image_num_tokens": num_tokens},
                modality="image",
                required_fields=required,
                optional_fields=optional,
                fields_factory=_get_image_fields,
            )
        return super()._parse_image_data(data)


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for the Kimi-K3 model.

    K3 uses the standard `image` and `video` modalities (unlike K2.5's unified
    `vision_chunk`), so it builds its own wrapper around the checkpoint's
    image processor and resolves the `<|media_pad|>` token id the same way K2.5 does.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls_overrides=KimiK25FusedVisionProcessor,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    @property
    def supports_video(self) -> bool:
        return supports_kimi_k3_video(self.image_processor.media_proc_cfg)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        if self.supports_video:
            return {"image": None, "video": None}
        return {"image": None}

    def get_data_parser(self) -> KimiK3MultiModalDataParser:
        return KimiK3MultiModalDataParser(
            media_proc_cfg=self.image_processor.media_proc_cfg,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
            video_needs_metadata=True,
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        cfg = self.image_processor.media_proc_cfg
        max_size = self.get_max_image_size(
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )
        options = self.ctx.get_mm_config().limit_per_prompt.get("image")
        width = min(max_size.width, getattr(options, "width", None) or math.inf)
        height = min(max_size.height, getattr(options, "height", None) or math.inf)
        max_tokens = {
            "image": navit_resize_image(
                int(width),
                int(height),
                cfg["patch_size"],
                cfg["merge_kernel_size"],
                cfg["in_patch_limit"],
                cfg["patch_limit_on_one_side"],
                cfg["fixed_output_tokens"],
            )["num_tokens"]
        }
        if self.supports_video:
            max_tokens["video"] = get_kimi_k3_max_video_tokens(cfg)
        return max_tokens

    def get_video_loader_kwargs(self) -> dict[str, Any]:
        if not self.supports_video:
            return {}
        return {
            "video_backend": "kimi_k3",
            "num_frames": -1,
            "media_proc_cfg": dict(self.image_processor.media_proc_cfg),
        }

    def get_dummy_video_frame_size(self) -> ImageSize:
        """An unpadded square frame, so the dummy video fills the video budget."""
        cfg = self.image_processor.media_proc_cfg
        merge = cfg["merge_kernel_size"]
        side = min(
            math.isqrt(cfg["in_patch_limit_each_frame"]) // merge * merge,
            cfg["patch_limit_on_one_side"] // merge * merge,
        )
        side_pixels = max(side, merge) * cfg["patch_size"]
        return ImageSize(width=side_pixels, height=side_pixels)

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds image and video dummy inputs for K3 profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return (
            self.info.get_hf_config().image_placeholder * num_images
            + KIMI_K3_VIDEO_PLACEHOLDER * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        images = self._get_dummy_images(
            width=max_size.width,
            height=max_size.height,
            num_images=mm_counts.get("image", 0),
            overrides=image_overrides,
        )
        num_videos = mm_counts.get("video", 0)
        if not num_videos:
            return {"image": images}
        video_overrides = cast(
            VideoDummyOptions | None,
            mm_options.get("video") if mm_options else None,
        )
        return {
            "image": images,
            "video": self._get_dummy_k3_videos(num_videos, video_overrides),
        }

    def _get_dummy_k3_videos(
        self, num_videos: int, overrides: VideoDummyOptions | None
    ) -> list[VideoItem]:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        frame_size = self.info.get_dummy_video_frame_size()
        num_frames = get_kimi_k3_max_video_frames(
            frame_size.width, frame_size.height, media_proc_cfg
        )
        videos = self._get_dummy_videos(
            width=frame_size.width,
            height=frame_size.height,
            num_frames=num_frames,
            num_videos=num_videos,
            overrides=overrides,
        )
        # Sampling at `sample_fps` keeps every frame of these videos.
        fps = float(media_proc_cfg["sample_fps"])
        return [
            (
                video,
                {
                    "fps": fps,
                    "duration": len(video) / fps,
                    "total_num_frames": len(video),
                    "frames_indices": list(range(len(video))),
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            )
            for video in videos
        ]


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Image and video multi-modal processor for Kimi-K3."""

    def _encode_text_prompt(self, text: str) -> list[int]:
        return encode_k3_text_prompt(
            self.info.get_tokenizer(),
            text,
            self.info.get_hf_config().image_placeholder,
        )

    def __call__(
        self,
        prompt: str | list[int],
        *args: Any,
        **kwargs: Any,
    ) -> MultiModalInput:
        if isinstance(prompt, str):
            prompt = self._encode_text_prompt(prompt)
        return super().__call__(prompt, *args, **kwargs)

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        inputs = super().get_dummy_inputs(seq_len, mm_counts, mm_options)
        inputs.prompt = self._encode_text_prompt(
            self.dummy_inputs.get_dummy_text(mm_counts)
        )
        return inputs

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Apply the prompt updates at token level only.

        K3 token ids cannot be rebuilt from their decoded text (see
        `KimiK3Renderer`), so there is no text fallback: an image without a
        placeholder in the token ids is an error.
        """
        new_token_ids, match_result, placeholders = (
            self._apply_token_matches_with_placeholders(token_ids, mm_prompt_updates)
        )
        self._matched_updates_from_result(mm_prompt_updates, match_result)
        return new_token_ids, {
            modality: modality_placeholders
            for modality, modality_placeholders in placeholders.items()
            if modality_placeholders
        }

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensors back into per-item fields.

        `pixel_values` holds all patches from every image concatenated; each
        image's patch count is `prod(grid_thws[i])`. `grid_thws` is one
        `[N_t, N_h, N_w]` row per image.
        """
        return {**_get_image_fields(hf_inputs), **_get_video_fields(hf_inputs)}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 image placeholder into a resolution-aware update.

        K3's prompt carries a single `<|kimi_image_placeholder|>` token per
        image. This replaces that token with
        `<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>`,
        embedding the per-image resolution in the prompt and marking only the
        `<|media_pad|>` positions as embedding slots (the number of pads is
        the feature size returned by `media_tokens_calculator`).
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items["image"]
            if isinstance(images, DictEmbeddingItems):
                item = images.get(item_idx)
                width, height = item["image_sizes"].tolist()
                num_media_token = int(item["image_num_tokens"].item())
            else:
                assert isinstance(images, ImageProcessorItems)
                image = images.get(item_idx)
                if image is None:
                    raise ValueError(f"Missing image data at index {item_idx}")
                width, height = images.get_image_size(item_idx)
                num_media_token = self.info.media_tokens_calculator(
                    {"type": "image", "image": image}
                )
            pads = media_token * num_media_token

            # The prompt carries original dimensions, not the resized grid.
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, full, add_special_tokens=False),
                media_token_id,
            )

        merge = self.info.image_processor.media_proc_cfg["merge_kernel_size"]

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            # One block per video: `<|media_begin|>video {w}x{h}<|media_content|>`,
            # then each chunk's start time and pads, then `<|media_end|>`.
            item = out_mm_kwargs["video"][item_idx]
            sizes, grids, timestamps = (
                cast(torch.Tensor, item[key].data).tolist()
                for key in ("video_sizes", "video_grid_thws", "video_chunk_timestamps")
            )
            width, height = sizes
            token_ids = list(
                cached_encode(
                    tokenizer,
                    f"<|media_begin|>video {width}x{height}<|media_content|>",
                    add_special_tokens=False,
                )
            )
            for (_, grid_h, grid_w), timestamp in zip(grids, timestamps):
                token_ids += cached_encode(
                    tokenizer,
                    format_kimi_k3_timestamp(timestamp),
                    add_special_tokens=False,
                )
                token_ids += [media_token_id] * ((grid_h // merge) * (grid_w // merge))
            token_ids += cached_encode(
                tokenizer, "<|media_end|>", add_special_tokens=False
            )
            return PromptUpdateDetails.select_token_id(token_ids, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    tokenizer, image_placeholder, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=cached_encode(
                    tokenizer, KIMI_K3_VIDEO_PLACEHOLDER, add_special_tokens=False
                ),
                replacement=get_video_replacement,
            ),
        ]


def encode_kimi_k3_videos(
    encode_chunks: Callable[[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]],
    pixel_values: torch.Tensor | list[torch.Tensor],
    grid_thws: torch.Tensor | list[torch.Tensor],
    num_chunks: torch.Tensor | list[torch.Tensor],
    merge_kernel_size: Sequence[int],
    max_patches_per_forward: int,
) -> list[torch.Tensor]:
    """Encode chunks in groups of at most `max_patches_per_forward` patches.

    `encode_chunks` returns one feature tensor per chunk; this returns one per
    video.
    """
    if isinstance(pixel_values, list):
        pixel_values = torch.cat(pixel_values)
    if isinstance(grid_thws, list):
        grid_thws = torch.cat(grid_thws)
    grid_thws = grid_thws.reshape(-1, 3)
    if isinstance(num_chunks, list):
        chunk_counts = [int(count) for count in num_chunks]
    else:
        chunk_counts = num_chunks.flatten().tolist()

    merge_h, merge_w = merge_kernel_size
    grids = grid_thws.tolist()
    chunk_patches = [t * h * w for t, h, w in grids]
    chunk_tokens = [(h // merge_h) * (w // merge_w) for _, h, w in grids]
    if sum(chunk_patches) != pixel_values.shape[0] or sum(chunk_counts) != len(grids):
        raise ValueError("Kimi-K3 video pixel values do not match their chunks")

    out: torch.Tensor | None = None
    offset = patch_start = start = 0
    while start < len(grids):
        end, patches = start + 1, chunk_patches[start]
        while (
            end < len(grids) and patches + chunk_patches[end] <= max_patches_per_forward
        ):
            patches += chunk_patches[end]
            end += 1
        features = encode_chunks(
            pixel_values[patch_start : patch_start + patches], grid_thws[start:end]
        )
        for feature in features:
            if out is None:
                out = feature.new_empty((sum(chunk_tokens), feature.shape[-1]))
            out[offset : offset + feature.shape[0]] = feature
            offset += feature.shape[0]
        patch_start += patches
        start = end

    if out is None or offset != out.shape[0]:
        raise RuntimeError(
            f"Kimi-K3 vision tower returned {offset} video embeddings, "
            f"expected {sum(chunk_tokens)}"
        )
    video_tokens, chunk_start = [], 0
    for count in chunk_counts:
        video_tokens.append(sum(chunk_tokens[chunk_start : chunk_start + count]))
        chunk_start += count
    return list(out.split(video_tokens))
