# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM-OCR model.

Same vision/LLM stack as MiniCPM-V 4.6. ``MiniCPMOcrForConditionalGeneration``
is a plain subclass of ``MiniCPMV4_6ForConditionalGeneration`` in transformers
too; what differs is the image geometry:

* ``downsample_mode`` defaults to ``"4x"``, and in that mode
  ``MiniCPMOcrImageProcessor.find_best_resize`` aligns the patch grid to
  ``patch_size * 2`` instead of ``patch_size * 4``, because ``"4x"`` goes
  through one 2x2 merge (downsample MLP) rather than two (ViT insert merger as
  well).
* An image small enough not to be sliced may be resized to a target picked from
  its own pixels (``adaptive_small_image_upsize``) or kept at its native
  resolution (``disable_small_image_upsize``).

The second point is why the image placeholder here is measured from the
``target_sizes`` the image processor actually returned rather than recomputed
from the image dimensions: the adaptive tier is decided by an Otsu threshold
plus connected-component analysis over the pixels, which a size-only
reimplementation cannot reproduce. This mirrors
``MiniCPMOcrProcessor.replace_image_token``, which reads the same tensor.
"""

import copy
from collections.abc import Mapping
from dataclasses import replace as dataclass_replace

import torch

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageProcessorItems, ImageSize
from vllm.multimodal.processing.processor import (
    PromptUpdateDetails,
    cached_encode,
)

from .minicpmv import MiniCPMVDummyInputsBuilder, MiniCPMVImageEmbeddingItems
from .minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
    _scoped_mm_kwarg,
)

# Options declared by `MiniCPMOcrImageProcessorKwargs` on top of the 4.6 set.
# `MiniCPMOcrVideoProcessorKwargs` deliberately declares none of them: in
# transformers videos go through `MiniCPMOcrVideoProcessor`, which has no
# small-image upsize logic at all.
_OCR_IMAGE_UPSIZE_KWARGS = (
    "disable_small_image_upsize",
    "adaptive_small_image_upsize",
    "adaptive_upsize_font_px",
    "adaptive_upsize_dense_per_kpx",
    "adaptive_upsize_tiny_pixels",
    "adaptive_upsize_tiny_target_resolution",
    "adaptive_tile_upsize",
)

# Pinned for video frames: `_compute_visual_tokens` counts an unsliced frame as
# upscaled to `scale_resolution` with uniform tiles, but `process_videos` runs
# the image processor, whose own defaults are not visible to that count.
_VIDEO_PINNED_UPSIZE_KWARGS = {
    "disable_small_image_upsize": False,
    "adaptive_small_image_upsize": False,
    "adaptive_tile_upsize": False,
}


class MiniCPMOcrProcessingInfo(MiniCPMV4_6ProcessingInfo):
    def _pinned_image_processor(self, downsample_mode: str | None = None):
        """Image processor view pinned to the mode resolved for this request.

        ``MiniCPMOcrImageProcessor.find_best_resize`` reads ``downsample_mode``
        off the instance, so a per-request override only reaches it through the
        shallow copy that ``MiniCPMOcrImageProcessor.__call__`` makes. Measure
        the token count on a copy pinned the same way, otherwise the pixels are
        aligned to one grid and the placeholder is counted on another.
        """
        image_processor = self.get_image_processor()
        resolved = self._get_downsample_mode(downsample_mode)
        if getattr(image_processor, "downsample_mode", None) == resolved:
            return image_processor

        pinned = copy.copy(image_processor)
        pinned.downsample_mode = resolved
        return pinned

    def get_image_size_with_most_features(self) -> ImageSize:
        # OCR-only override: source the dummy-input dimension from the image
        # processor's `scale_resolution` (the slice resize target) rather than the
        # top-level `config.image_size`. The top-level field is dropped from
        # `MiniCPMOcrConfig`, and the existing default of 448 was always equal to
        # `scale_resolution` here. `vision_config.image_size` is the ViT patch grid
        # (980 for this checkpoint), not the preprocessed image dimension, so it is
        # not the right value for dummy-input sizing.
        image_size = self.get_image_processor().scale_resolution
        max_slice_num = self.get_image_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)

    def get_video_frame_size_with_most_features(self) -> ImageSize:
        # See `get_image_size_with_most_features` for why OCR sources from
        # `scale_resolution` instead of the (now-removed) top-level `config.image_size`.
        image_size = self.get_image_processor().scale_resolution
        max_slice_num = self.get_video_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)

    def _compute_visual_tokens(
        self,
        image_size: ImageSize,
        max_slice_nums: int | None = None,
        downsample_mode: str | None = None,
    ) -> tuple[list[int], int, int]:
        """Grid, source tokens and per-slice tokens for one image.

        Same contract as 4.6, re-derived on the pinned processor so that the
        ``"4x"`` patch-grid alignment is applied.

        The unsliced branch upscales to ``scale_resolution``, which is what
        ``MiniCPMOcrImageProcessor._preprocess`` does for the default options
        and what the video path does unconditionally. The remaining
        ``*_small_image_upsize`` combinations depend on the pixels, so for
        images the placeholder is taken from ``target_sizes`` instead; see
        ``MiniCPMOcrMultiModalProcessor._ocr_image_prompt_text``.
        """
        image_processor = self._pinned_image_processor(downsample_mode)
        if max_slice_nums is None:
            max_slice_nums = image_processor.max_slice_nums

        patch_size = image_processor.patch_size
        scale_res = image_processor.scale_resolution
        token_divisor = 4 if image_processor.downsample_mode == "4x" else 16
        tokens_per_patch = patch_size * patch_size * token_divisor

        # vLLM ImageSize is (width, height); transformers expects (height, width)
        hf_image_size = (image_size.height, image_size.width)

        grids = image_processor.get_sliced_grid(
            hf_image_size,
            max_slice_nums,
            scale_res,
        )

        if grids is None:
            best_size = image_processor.find_best_resize(
                hf_image_size,
                scale_res,
                patch_size,
                allow_upscale=True,
            )
            source_tokens = best_size[0] * best_size[1] // tokens_per_patch
            return [0, 0], source_tokens, 0

        best_resize = image_processor.find_best_resize(
            hf_image_size,
            scale_res,
            patch_size,
        )
        source_tokens = best_resize[0] * best_resize[1] // tokens_per_patch

        refine_size = image_processor.get_refine_size(
            hf_image_size,
            grids,
            scale_res,
            patch_size,
            allow_upscale=True,
        )
        patch_h = refine_size[0] // grids[0]
        patch_w = refine_size[1] // grids[1]
        patch_tokens = patch_h * patch_w // tokens_per_patch
        return grids, source_tokens, patch_tokens


class MiniCPMOcrMultiModalProcessor(MiniCPMV4_6MultiModalProcessor):
    def _promote_image_upsize_kwargs(
        self,
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Resolve the OCR-only upsize options into the flat form.

        ``process_images`` forwards flat keys only (``_flat_processor_kwargs``
        drops the nested scopes), and 4.6 recovers just ``downsample_mode`` /
        ``max_slice_nums`` from the merge. Resolve the OCR options the same way
        so that a server-level ``mm_processor_kwargs`` value, and the nested
        ``images_kwargs`` form, both reach the image processor instead of being
        silently dropped.
        """
        info = self.info
        assert isinstance(info, MiniCPMOcrProcessingInfo)
        merged = info._merge_and_resolve_mm_processor_kwargs(mm_kwargs)

        promoted = {}
        for key in _OCR_IMAGE_UPSIZE_KWARGS:
            value = _scoped_mm_kwarg(merged, "image", key)
            if value is not None:
                promoted[key] = value

        if not promoted:
            return mm_kwargs
        return {**mm_kwargs, **promoted}

    def _drop_image_upsize_kwargs(
        self,
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Remove the image-only upsize options from the video path.

        ``process_videos`` runs the *image* processor per frame, so a flat
        upsize option would reach frames as well. In transformers frames go
        through ``MiniCPMOcrVideoProcessor``, which never applies these, and
        the video placeholder is counted without them.
        """
        if not any(key in mm_kwargs for key in _OCR_IMAGE_UPSIZE_KWARGS):
            return mm_kwargs
        return {
            key: value
            for key, value in mm_kwargs.items()
            if key not in _OCR_IMAGE_UPSIZE_KWARGS
        }

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ):
        return super().process_images(
            mm_data,
            self._promote_image_upsize_kwargs(mm_kwargs),
        )

    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ):
        return super().process_videos(
            mm_data,
            {
                **self._drop_image_upsize_kwargs(mm_kwargs),
                **_VIDEO_PINNED_UPSIZE_KWARGS,
            },
        )

    def _ocr_image_prompt_text(
        self,
        tgt_sizes: torch.Tensor,
        image_size: ImageSize,
        image_idx: int,
        downsample_mode: str | None,
        max_slice_nums: int | None,
    ) -> str | None:
        """Placeholder text for one image, measured from its ``target_sizes``.

        ``MiniCPMOcrProcessor.replace_image_token`` counts
        ``target_sizes.prod(-1) // token_divisor`` per patch, row 0 being the
        source image and row 1 a slice. Doing the same here keeps the count
        correct for ``adaptive_small_image_upsize``, whose resize target is
        decided from the pixels and cannot be recovered from ``image_size``.

        ``target_sizes`` is already in *patch* units -- the image processor
        stores ``source_h // patch_size`` -- so the divisor here is the bare
        ``token_divisor``, unlike the pixel-side computation in
        ``_compute_visual_tokens`` which also divides by ``patch_size ** 2``.

        The slice grid is not pixel-dependent (``get_sliced_grid`` is a
        function of the image size, ``max_slice_nums`` and
        ``scale_resolution``), so it still comes from the geometry helper.
        Returns ``None`` when the two disagree, leaving the caller on the 4.6
        path rather than emitting a count we cannot justify.
        """
        info = self.info
        assert isinstance(info, MiniCPMOcrProcessingInfo)

        if tgt_sizes.ndim != 2 or tgt_sizes.shape[-1] != 2 or tgt_sizes.shape[0] < 1:
            return None

        image_processor = info._pinned_image_processor(downsample_mode)
        token_divisor = 4 if image_processor.downsample_mode == "4x" else 16

        grids, _, _ = info._compute_visual_tokens(
            image_size,
            max_slice_nums,
            downsample_mode=downsample_mode,
        )
        num_rows, num_cols = grids[0], grids[1]
        if num_rows * num_cols + 1 != tgt_sizes.shape[0]:
            return None

        patch_tokens_per_slice = (
            tgt_sizes.to(torch.long).prod(-1) // token_divisor
        ).tolist()
        source_tokens = int(patch_tokens_per_slice[0])
        if source_tokens <= 0:
            return None

        tokenizer = info.get_tokenizer()
        image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        image_start = getattr(tokenizer, "image_start_token", "<image>")
        image_end = getattr(tokenizer, "image_end_token", "</image>")
        slice_start = getattr(tokenizer, "slice_start_token", "<slice>")
        slice_end = getattr(tokenizer, "slice_end_token", "</slice>")
        id_start = getattr(tokenizer, "image_id_start_token", "<image_id>")
        id_end = getattr(tokenizer, "image_id_end_token", "</image_id>")

        placeholder = (
            f"{id_start}{image_idx}{id_end}"
            + image_start
            + image_token * source_tokens
            + image_end
        )

        if num_rows > 0 and num_cols > 0:
            # `adaptive_tile_upsize` classifies each tile on its own crop, so
            # slices in one grid can differ; emit one placeholder per tile.
            tile_tokens = [int(t) for t in patch_tokens_per_slice[1:]]
            if len(set(tile_tokens)) <= 1:
                slice_tokens = tile_tokens[0] if tile_tokens else 0
                if slice_tokens > 0:
                    slice_ph = slice_start + image_token * slice_tokens + slice_end
                    rows = [slice_ph * num_cols for _ in range(num_rows)]
                    placeholder += "\n".join(rows)
            else:
                if len(tile_tokens) != num_rows * num_cols:
                    return None
                rows = []
                for r in range(num_rows):
                    line = []
                    for c in range(num_cols):
                        n = tile_tokens[r * num_cols + c]
                        if n > 0:
                            line.append(slice_start + image_token * n + slice_end)
                    rows.append("".join(line))
                placeholder += "\n".join(rows)

        return placeholder

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ):
        updates = super()._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            out_mm_kwargs,
        )

        image_ds_mode = self._resolve_downsample_mode(
            hf_processor_mm_kwargs, modality="image"
        )
        image_max_slice = self._resolve_max_slice_nums(hf_processor_mm_kwargs)

        tokenizer = self.info.get_tokenizer()
        image_embed_text = getattr(tokenizer, "image_token", "<|image_pad|>")
        image_embed_ids = [tokenizer.get_vocab()[image_embed_text]]
        image_items = out_mm_kwargs.get("image")

        def ocr_image_replacement(item_idx: int, fallback):
            tgt_sizes = None
            if image_items is not None and item_idx < len(image_items):
                tgt_elem = image_items[item_idx].get("tgt_sizes")
                if tgt_elem is not None and isinstance(tgt_elem.data, torch.Tensor):
                    tgt_sizes = tgt_elem.data

            if tgt_sizes is None:
                # Image embeddings bypass the image processor, so there is no
                # `target_sizes` to measure and no pixel-dependent resize.
                return fallback(item_idx)

            images = mm_items.get_items(
                "image",
                (MiniCPMVImageEmbeddingItems, ImageProcessorItems),
            )
            text = self._ocr_image_prompt_text(
                tgt_sizes,
                images.get_image_size(item_idx),
                item_idx,
                downsample_mode=image_ds_mode,
                max_slice_nums=image_max_slice,
            )
            if text is None:
                return fallback(item_idx)

            return PromptUpdateDetails.select_token_ids(
                cached_encode(tokenizer, text, add_special_tokens=False),
                image_embed_ids,
            )

        def with_ocr_replacement(update):
            fallback = update.replacement
            return dataclass_replace(
                update,
                replacement=lambda item_idx: ocr_image_replacement(item_idx, fallback),
            )

        return [
            with_ocr_replacement(update) if update.modality == "image" else update
            for update in updates
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOcrMultiModalProcessor,
    info=MiniCPMOcrProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMOcrForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    def _drop_vision_last_layer(self) -> bool:
        # `MiniCPMOcrConfig` deletes the inherited field (the modular file marks
        # it removed) and OCR keeps every vision encoder layer.
        return False
