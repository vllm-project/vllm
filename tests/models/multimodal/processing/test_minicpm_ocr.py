# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-OCR semantics that differ from MiniCPM-V 4.6.

The model code is inherited wholesale, so what needs pinning is the image
geometry: the `"4x"` patch-grid alignment, and the fact that the image
placeholder is measured from the `target_sizes` the image processor returned
rather than recomputed from the image dimensions.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.minicpm_ocr import (
    _OCR_IMAGE_UPSIZE_KWARGS,
    MiniCPMOcrMultiModalProcessor,
    MiniCPMOcrProcessingInfo,
)
from vllm.model_executor.models.minicpmv4_6 import MiniCPMV4_6MultiModalProcessor
from vllm.multimodal.parse import ImageSize

pytestmark = pytest.mark.skip_global_cleanup

_PATCH_SIZE = 14
_SCALE_RESOLUTION = 448


def ensure_divide(length: int, divisor: int) -> int:
    """Copy of `transformers.models.minicpmv4_6...ensure_divide`."""
    return max(round(length / divisor) * divisor, divisor)


class _StubImageProcessor:
    """Stands in for `MiniCPMOcrImageProcessor`.

    `find_best_resize` and `get_sliced_grid`/`get_refine_size` are transcribed
    from `modular_minicpm_ocr.py` / `image_processing_minicpmv4_6.py` so the
    geometry under test is the real one, without needing a checkpoint or an
    installed transformers build that carries the model.
    """

    downsample_mode = "4x"
    max_slice_nums = 9
    scale_resolution = _SCALE_RESOLUTION
    patch_size = _PATCH_SIZE
    slice_mode = True

    def find_best_resize(
        self,
        image_size,
        scale_resolution,
        patch_size,
        downsample_mode,
        allow_upscale=False,
    ):
        height, width = image_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # The OCR delta: one 2x2 merge in "4x" mode, two in "16x".
        # transformers 5.21 takes the mode as an argument, not off the instance.
        merge_factor = 2 if downsample_mode == "4x" else 4
        best_width = ensure_divide(width, patch_size * merge_factor)
        best_height = ensure_divide(height, patch_size * merge_factor)
        return best_height, best_width

    def get_refine_size(
        self,
        image_size,
        grid,
        scale_resolution,
        patch_size,
        downsample_mode,
        allow_upscale=False,
    ):
        height, width = image_size
        grid_y, grid_x = grid
        refine_width = ensure_divide(width, grid_x)
        refine_height = ensure_divide(height, grid_y)
        best_height, best_width = self.find_best_resize(
            (refine_height / grid_y, refine_width / grid_x),
            scale_resolution,
            patch_size,
            downsample_mode,
            allow_upscale=allow_upscale,
        )
        return best_height * grid_y, best_width * grid_x

    def get_sliced_grid(self, image_size, max_slice_nums, scale_resolution):
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution**2)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error
        return best_grid


class _StubTokenizer:
    image_token = "<|image_pad|>"
    image_start_token = "<image>"
    image_end_token = "</image>"
    slice_start_token = "<slice>"
    slice_end_token = "</slice>"
    image_id_start_token = "<image_id>"
    image_id_end_token = "</image_id>"


def _make_info(**processor_attrs) -> MiniCPMOcrProcessingInfo:
    info = MiniCPMOcrProcessingInfo.__new__(MiniCPMOcrProcessingInfo)
    image_processor = _StubImageProcessor()
    for key, value in processor_attrs.items():
        setattr(image_processor, key, value)

    info.get_image_processor = lambda **kwargs: image_processor  # type: ignore[method-assign]
    info.get_tokenizer = lambda: _StubTokenizer()  # type: ignore[method-assign,return-value]
    return info


def _make_processor(info: MiniCPMOcrProcessingInfo):
    processor = MiniCPMOcrMultiModalProcessor.__new__(MiniCPMOcrMultiModalProcessor)
    processor.info = info
    return processor


# --------------------------------------------------------------------------
# Geometry: "4x" aligns to patch_size * 2, not patch_size * 4
# --------------------------------------------------------------------------


def test_4x_mode_aligns_the_patch_grid_to_two_patches():
    """The resize grid follows the number of 2x2 merges, not just the divisor.

    `MiniCPMOcrImageProcessor.find_best_resize` aligns to `patch_size * 2` in
    `"4x"` mode, so a 4.6-style alignment would round the grid to a different
    size and the vision tower and the placeholder would disagree.
    """
    info = _make_info()
    image_size = ImageSize(width=300, height=200)

    grids_4x, source_4x, patch_4x = info._compute_visual_tokens(
        image_size, downsample_mode="4x"
    )
    grids_16x, source_16x, patch_16x = info._compute_visual_tokens(
        image_size, downsample_mode="16x"
    )

    # Not sliced: 300*200 < 448^2.
    assert grids_4x == [0, 0]
    assert grids_16x == [0, 0]
    assert patch_4x == 0 and patch_16x == 0

    # Upscaled to ~448 on the long side, then aligned: "4x" rounds to 28px and
    # "16x" to 56px, which lands on a different grid for the same image.
    assert info._pinned_image_processor("4x").find_best_resize(
        (200, 300),
        _SCALE_RESOLUTION,
        _PATCH_SIZE,
        downsample_mode="4x",
        allow_upscale=True,
    ) == (364, 560)
    assert info._pinned_image_processor("16x").find_best_resize(
        (200, 300),
        _SCALE_RESOLUTION,
        _PATCH_SIZE,
        downsample_mode="16x",
        allow_upscale=True,
    ) == (392, 560)

    # 364*560 / (14*14*4) = 260 against 392*560 / (14*14*16) = 70: the ratio is
    # not a clean 4x precisely because the alignment differs, which is what a
    # 4.6-style `find_best_resize` would get wrong.
    assert source_4x == 260
    assert source_16x == 70


def test_pinned_processor_does_not_mutate_the_shared_instance():
    """A per-request mode must not leak into the shared image processor.

    `find_best_resize` reads `downsample_mode` off the instance, and the engine
    reuses one processor across requests.
    """
    info = _make_info()
    shared = info.get_image_processor()
    assert shared.downsample_mode == "4x"

    pinned = info._pinned_image_processor("16x")
    assert pinned is not shared
    assert pinned.downsample_mode == "16x"
    assert shared.downsample_mode == "4x"

    # The default mode needs no copy at all.
    assert info._pinned_image_processor("4x") is shared
    assert info._pinned_image_processor(None) is shared


def test_downsample_mode_defaults_to_the_processor_value():
    """OCR checkpoints ship `downsample_mode: "4x"`; 4.6's fallback is "16x"."""
    info = _make_info()
    assert info._get_downsample_mode() == "4x"
    assert info._get_downsample_mode("16x") == "16x"

    # The fallback only applies when the processor carries no value at all.
    bare = _make_info()
    del type(bare.get_image_processor()).downsample_mode
    try:
        assert bare._get_downsample_mode() == "16x"
    finally:
        _StubImageProcessor.downsample_mode = "4x"


def test_sliced_image_token_counts_match_the_transformers_divisor():
    """Sliced geometry: source + grid*patch tokens, counted with divisor 4."""
    info = _make_info()
    image_size = ImageSize(width=1680, height=1120)

    grids, source_tokens, patch_tokens = info._compute_visual_tokens(
        image_size, downsample_mode="4x"
    )

    assert grids != [0, 0]
    processor = info._pinned_image_processor("4x")
    refine = processor.get_refine_size(
        (1120, 1680),
        grids,
        _SCALE_RESOLUTION,
        _PATCH_SIZE,
        downsample_mode="4x",
        allow_upscale=True,
    )
    expected_patch = (
        (refine[0] // grids[0])
        * (refine[1] // grids[1])
        // (_PATCH_SIZE * _PATCH_SIZE * 4)
    )
    assert patch_tokens == expected_patch
    assert source_tokens > 0


# --------------------------------------------------------------------------
# Placeholder measured from target_sizes
# --------------------------------------------------------------------------


def _placeholder(info, tgt_sizes, image_size, *, image_idx=0, mode="4x"):
    processor = _make_processor(info)
    return processor._ocr_image_prompt_text(
        torch.tensor(tgt_sizes, dtype=torch.long),
        image_size,
        image_idx,
        downsample_mode=mode,
        max_slice_nums=None,
    )


def test_placeholder_is_counted_from_target_sizes():
    """Token count mirrors `MiniCPMOcrProcessor.replace_image_token`.

    That method computes `target_sizes.prod(-1) // token_divisor` per patch.
    Reproducing it here is what keeps the count right under
    `adaptive_small_image_upsize`, whose resize target is decided from the
    pixels and so is invisible to a size-only computation.
    """
    info = _make_info()
    # An unsliced image: one row only.
    text = _placeholder(info, [[8, 8]], ImageSize(width=112, height=112))

    # 8*8 = 64 patches, divisor 4 -> 16 tokens.
    assert text == ("<image_id>0</image_id><image>" + "<|image_pad|>" * 16 + "</image>")


def test_adaptive_upsize_changes_the_count_without_changing_the_image_size():
    """Two tiers of the same image must yield different placeholders.

    `adaptive_upsize_tier` can send a small image to `scale_resolution` or to
    `adaptive_upsize_tiny_target_resolution`, and `ImageSize` is identical in
    both cases. This is exactly the divergence that counting from `image_size`
    would miss.
    """
    info = _make_info()
    image_size = ImageSize(width=80, height=80)

    native = _placeholder(info, [[4, 4]], image_size)
    tier2 = _placeholder(info, [[6, 6]], image_size)
    tier1 = _placeholder(info, [[16, 16]], image_size)

    assert native.count("<|image_pad|>") == 4
    assert tier2.count("<|image_pad|>") == 9
    assert tier1.count("<|image_pad|>") == 64

    # And the size-only path cannot tell them apart: it always upscales to
    # `scale_resolution`, so all three tiers would be counted as 448*448 worth
    # of tokens regardless of what the processor actually produced.
    _, size_only_tokens, _ = info._compute_visual_tokens(
        image_size, downsample_mode="4x"
    )
    assert size_only_tokens == 256
    assert size_only_tokens not in (4, 9, 64)


def test_placeholder_emits_the_slice_grid_from_the_geometry():
    """Slice rows come from `get_sliced_grid`, which is not pixel-dependent."""
    info = _make_info()
    image_size = ImageSize(width=1680, height=1120)
    grids, _, _ = info._compute_visual_tokens(image_size, downsample_mode="4x")
    num_slices = grids[0] * grids[1]

    tgt_sizes = [[16, 16]] + [[8, 8]] * num_slices
    text = _placeholder(info, tgt_sizes, image_size)

    assert text is not None
    assert text.count("<slice>") == num_slices
    assert text.count("\n") == grids[0] - 1
    # 16*16/4 source tokens + one 8*8/4 block per slice.
    assert text.count("<|image_pad|>") == 64 + 16 * num_slices


def test_placeholder_emits_differing_tiles_one_by_one():
    """`adaptive_tile_upsize` can leave tiles at different resolutions.

    Each slice then gets its own count instead of row 1 repeated.
    """
    info = _make_info()
    image_size = ImageSize(width=1680, height=1120)
    grids, _, _ = info._compute_visual_tokens(image_size, downsample_mode="4x")
    num_rows, num_cols = grids
    num_slices = num_rows * num_cols

    tile_sizes = [[8 + (i % 2) * 4, 8 + (i % 2) * 4] for i in range(num_slices)]
    text = _placeholder(info, [[16, 16]] + tile_sizes, image_size)

    assert text is not None
    assert text.count("<slice>") == num_slices
    per_tile = sum(size[0] * size[1] // 4 for size in tile_sizes)
    assert text.count("<|image_pad|>") == 64 + per_tile


def test_video_path_pins_the_upsize_defaults(monkeypatch):
    """Frames must be counted the same way `_compute_visual_tokens` counts them.

    `_drop_image_upsize_kwargs` only removes request keys; the loaded processor
    still applies its own defaults per frame, so they are pinned too.
    """
    captured = {}

    def fake_process_videos(self, mm_data, mm_kwargs):
        captured.update(mm_kwargs)

    monkeypatch.setattr(
        MiniCPMV4_6MultiModalProcessor, "process_videos", fake_process_videos
    )
    processor = _processor_with_merge({})
    processor.process_videos(
        {},
        {"downsample_mode": "4x", "adaptive_small_image_upsize": True},
    )

    assert captured["downsample_mode"] == "4x"
    assert captured["adaptive_small_image_upsize"] is False
    assert captured["disable_small_image_upsize"] is False
    assert captured["adaptive_tile_upsize"] is False


def test_placeholder_falls_back_when_target_sizes_disagree_with_the_grid():
    """A mismatch must not silently emit a count we cannot justify.

    Returning `None` leaves the caller on the inherited 4.6 path instead of
    producing a placeholder whose length the vision tower will not match.
    """
    info = _make_info()
    image_size = ImageSize(width=1680, height=1120)

    # Only the thumbnail row, while the geometry says the image is sliced.
    assert _placeholder(info, [[16, 16]], image_size) is None
    # Malformed tables.
    assert _placeholder(info, [[16, 16, 16]], ImageSize(width=112, height=112)) is None
    # A zero-token source.
    assert _placeholder(info, [[1, 1]], ImageSize(width=112, height=112)) is None


def test_placeholder_uses_the_divisor_of_the_resolved_mode():
    """`token_divisor` follows the same resolved mode as the pixel path."""
    info = _make_info()
    image_size = ImageSize(width=112, height=112)

    text_4x = _placeholder(info, [[16, 16]], image_size, mode="4x")
    text_16x = _placeholder(info, [[16, 16]], image_size, mode="16x")

    assert text_4x.count("<|image_pad|>") == 64
    assert text_16x.count("<|image_pad|>") == 16


def test_placeholder_carries_the_local_image_id():
    info = _make_info()
    text = _placeholder(info, [[8, 8]], ImageSize(width=112, height=112), image_idx=3)
    assert text.startswith("<image_id>3</image_id><image>")


def _transformers_replace_image_token(
    target_sizes, grids, num_patches_per_image, image_idx, image_token_divisor
):
    """Transcription of `MiniCPMOcrProcessor.replace_image_token`.

    Kept verbatim (minus the tokenizer attribute lookups) so the count under
    test is compared against the transformers formula itself rather than
    against a number restated in the assertion. Note the divisor is applied to
    `target_sizes` directly: that tensor is already in patch units.
    """
    cum_patches = np.cumsum(num_patches_per_image)
    start_idx = cum_patches[image_idx - 1] if image_idx > 0 else 0
    end_idx = cum_patches[image_idx]
    img_target_sizes = target_sizes[start_idx:end_idx]
    num_tokens_per_patch = img_target_sizes.prod(-1) // image_token_divisor
    num_rows, num_cols = grids[image_idx]

    image_placeholder = (
        "<image>" + "<|image_pad|>" * int(num_tokens_per_patch[0]) + "</image>"
    )
    if num_rows > 0 and num_cols > 0:
        per_slice = int(num_tokens_per_patch[1]) if len(num_tokens_per_patch) > 1 else 0
        slice_placeholder = "<slice>" + "<|image_pad|>" * per_slice + "</slice>"
        slices = [slice_placeholder * num_cols for _ in range(num_rows)]
        image_placeholder += "\n".join(slices)
    return image_placeholder


@pytest.mark.parametrize(
    "width,height,mode",
    [
        (112, 112, "4x"),
        (112, 112, "16x"),
        (300, 200, "4x"),
        (1680, 1120, "4x"),
        (1680, 1120, "16x"),
        (2048, 512, "4x"),
    ],
)
def test_placeholder_matches_the_transformers_formula(width, height, mode):
    """End-to-end agreement with `replace_image_token` on real geometry.

    `target_sizes` is derived the way `_preprocess` does (`size // patch_size`
    per patch), then both implementations are asked for the placeholder. This
    is the check that pins the divisor: `target_sizes` is in patch units, so
    dividing by `patch_size ** 2 * token_divisor` here -- as the pixel-side
    `_compute_visual_tokens` correctly does -- would undercount by 196x.
    """
    info = _make_info()
    image_size = ImageSize(width=width, height=height)
    processor = info._pinned_image_processor(mode)
    grids, _, _ = info._compute_visual_tokens(image_size, downsample_mode=mode)

    # Rebuild `target_sizes` exactly as `_preprocess` records it.
    sliced = grids != [0, 0]
    source_h, source_w = processor.find_best_resize(
        (height, width),
        _SCALE_RESOLUTION,
        _PATCH_SIZE,
        downsample_mode=mode,
        allow_upscale=not sliced,
    )
    target_sizes = [[source_h // _PATCH_SIZE, source_w // _PATCH_SIZE]]
    if sliced:
        refine_h, refine_w = processor.get_refine_size(
            (height, width),
            grids,
            _SCALE_RESOLUTION,
            _PATCH_SIZE,
            downsample_mode=mode,
            allow_upscale=True,
        )
        patch_h, patch_w = refine_h // grids[0], refine_w // grids[1]
        target_sizes += [[patch_h // _PATCH_SIZE, patch_w // _PATCH_SIZE]] * (
            grids[0] * grids[1]
        )

    expected = _transformers_replace_image_token(
        np.array(target_sizes),
        [grids],
        [len(target_sizes)],
        image_idx=0,
        image_token_divisor=4 if mode == "4x" else 16,
    )
    actual = _placeholder(info, target_sizes, image_size, mode=mode)

    assert actual == "<image_id>0</image_id>" + expected
    # The count is non-trivial, so a silently empty placeholder cannot pass.
    assert actual.count("<|image_pad|>") > 0


def test_pixel_and_target_size_paths_agree_on_the_default_resize():
    """The two divisors must land on the same count for the plain path.

    `_compute_visual_tokens` divides pixels by `patch_size ** 2 * divisor`
    while `_ocr_image_prompt_text` divides patch counts by `divisor` alone.
    With no adaptive upsize in play they describe the same image, so they have
    to agree -- that equality is what makes the fallback in
    `_get_prompt_updates` safe.
    """
    info = _make_info()
    for width, height in [(112, 112), (300, 200), (1680, 1120)]:
        image_size = ImageSize(width=width, height=height)
        grids, source_tokens, patch_tokens = info._compute_visual_tokens(
            image_size, downsample_mode="4x"
        )
        processor = info._pinned_image_processor("4x")
        sliced = grids != [0, 0]
        source_h, source_w = processor.find_best_resize(
            (height, width),
            _SCALE_RESOLUTION,
            _PATCH_SIZE,
            downsample_mode="4x",
            allow_upscale=not sliced,
        )
        target_sizes = [[source_h // _PATCH_SIZE, source_w // _PATCH_SIZE]]
        if sliced:
            refine_h, refine_w = processor.get_refine_size(
                (height, width),
                grids,
                _SCALE_RESOLUTION,
                _PATCH_SIZE,
                downsample_mode="4x",
                allow_upscale=True,
            )
            patch_h, patch_w = refine_h // grids[0], refine_w // grids[1]
            target_sizes += [[patch_h // _PATCH_SIZE, patch_w // _PATCH_SIZE]] * (
                grids[0] * grids[1]
            )

        text = _placeholder(info, target_sizes, image_size)
        expected_total = source_tokens + patch_tokens * grids[0] * grids[1]
        assert text.count("<|image_pad|>") == expected_total, (
            f"{width}x{height}: {text.count('<|image_pad|>')} != {expected_total}"
        )


# --------------------------------------------------------------------------
# Processor kwargs routing
# --------------------------------------------------------------------------


def _processor_with_merge(merged):
    info = MiniCPMOcrProcessingInfo.__new__(MiniCPMOcrProcessingInfo)
    info._merge_and_resolve_mm_processor_kwargs = lambda mm_kwargs: merged  # type: ignore[method-assign]
    return _make_processor(info)


def test_scoped_upsize_kwargs_are_promoted_to_the_flat_form():
    """`images_kwargs`-scoped options must still reach the image processor.

    `_flat_processor_kwargs` drops the nested scopes before the processor call,
    so without this an `images_kwargs` (or server-level) upsize option would be
    silently ignored.
    """
    processor = _processor_with_merge(
        {"images_kwargs": {"adaptive_small_image_upsize": True}}
    )
    promoted = processor._promote_image_upsize_kwargs({"max_slice_nums": 4})

    assert promoted["adaptive_small_image_upsize"] is True
    assert promoted["max_slice_nums"] == 4


def test_upsize_kwargs_are_kept_out_of_the_video_path():
    """Frames go through the video processor, which has no upsize logic.

    `process_videos` runs the *image* processor per frame, so a flat upsize
    option would otherwise change frame geometry while the video placeholder is
    counted without it.
    """
    processor = _processor_with_merge({})
    mm_kwargs = {
        "downsample_mode": "4x",
        "adaptive_small_image_upsize": True,
        "disable_small_image_upsize": True,
    }
    stripped = processor._drop_image_upsize_kwargs(mm_kwargs)

    assert stripped == {"downsample_mode": "4x"}
    # Untouched input is returned as-is.
    assert processor._drop_image_upsize_kwargs({"downsample_mode": "4x"}) == {
        "downsample_mode": "4x"
    }


def test_declared_upsize_kwargs_match_the_transformers_schema():
    """The promoted/stripped set must track `MiniCPMOcrImageProcessorKwargs`.

    vLLM routes a flat kwarg into a scope only when the HF processor declares
    it, so this list is what decides whether a request option is honoured.
    """
    assert set(_OCR_IMAGE_UPSIZE_KWARGS) == {
        "disable_small_image_upsize",
        "adaptive_small_image_upsize",
        "adaptive_upsize_font_px",
        "adaptive_upsize_dense_per_kpx",
        "adaptive_upsize_tiny_pixels",
        "adaptive_upsize_tiny_target_resolution",
        "adaptive_tile_upsize",
    }
    # `downsample_mode` / `max_slice_nums` are inherited from 4.6 and already
    # handled there; re-promoting them here would double-apply the resolution.
    assert "downsample_mode" not in _OCR_IMAGE_UPSIZE_KWARGS
    assert "max_slice_nums" not in _OCR_IMAGE_UPSIZE_KWARGS


# --------------------------------------------------------------------------
# Model wiring
# --------------------------------------------------------------------------


def test_vision_last_layer_is_kept():
    """`MiniCPMOcrConfig` deletes `drop_vision_last_layer`.

    Reading it off the config would raise, and OCR keeps every encoder layer.
    """
    from vllm.model_executor.models.minicpm_ocr import (
        MiniCPMOcrForConditionalGeneration,
    )

    model = MiniCPMOcrForConditionalGeneration.__new__(
        MiniCPMOcrForConditionalGeneration  # type: ignore[type-abstract]
    )
    # No `drop_vision_last_layer` on the config at all.
    model.config = SimpleNamespace(downsample_mode="4x", insert_layer_id=6)

    assert model._drop_vision_last_layer() is False


def test_weight_mapper_and_loader_are_inherited_from_4_6():
    """The converter emits 4.6-shaped keys, so the mapper must be reused."""
    from vllm.model_executor.models.minicpm_ocr import (
        MiniCPMOcrForConditionalGeneration,
    )
    from vllm.model_executor.models.minicpmv4_6 import (
        MiniCPMV4_6ForConditionalGeneration,
    )

    assert (
        MiniCPMOcrForConditionalGeneration.hf_to_vllm_mapper
        is MiniCPMV4_6ForConditionalGeneration.hf_to_vllm_mapper
    )
    assert (
        MiniCPMOcrForConditionalGeneration.load_weights
        is MiniCPMV4_6ForConditionalGeneration.load_weights
    )
