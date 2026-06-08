# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor

module = types.ModuleType("vllm.vllm_flash_attn")
module.flash_attn_varlen_func = lambda *args, **kwargs: None
module.get_scheduler_metadata = lambda *args, **kwargs: None
sys.modules.setdefault("vllm.vllm_flash_attn", module)

from vllm.model_executor.models.qwen3_vl import (  # noqa: E402
    Qwen3_VisionPatchEmbed,
    Qwen3VLMultiModalProcessor,
    _qwen3_vl_fused_compact_preprocess_uint8_bchw,
)
from vllm.multimodal.parse import MultiModalDataParser  # noqa: E402
from vllm.multimodal.processing import BaseMultiModalProcessor  # noqa: E402
from vllm.multimodal.processing.context import TimingContext  # noqa: E402
from vllm.multimodal.processing.inputs import ProcessorInputs  # noqa: E402


def _make_tokenizer() -> PreTrainedTokenizerFast:
    special_tokens = [
        "<unk>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    tokenizer_impl = Tokenizer(
        WordLevel(
            vocab={token: index for index, token in enumerate(special_tokens)},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens[1:]}
    )
    tokenizer.image_token = "<|image_pad|>"
    tokenizer.video_token = "<|video_pad|>"
    tokenizer.vision_start_token = "<|vision_start|>"
    tokenizer.vision_end_token = "<|vision_end|>"
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    tokenizer.video_token_id = tokenizer.convert_tokens_to_ids(tokenizer.video_token)
    tokenizer.vision_start_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.vision_start_token
    )
    tokenizer.vision_end_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.vision_end_token
    )
    return tokenizer


class _FakeProcessingContext:
    def call_hf_processor(
        self,
        processor: Any,
        data: dict[str, object],
        kwargs: dict[str, object],
    ) -> Any:
        kwargs = dict(kwargs)
        kwargs.setdefault("return_tensors", "pt")
        return processor(**data, **kwargs)

    def get_mm_config(self) -> Any:
        return SimpleNamespace(video_pruning_rate=None)


class _FakeProcessingInfo:
    model_id = "qwen3-fast-path-test"

    def __init__(self, hf_processor: Qwen3VLProcessor) -> None:
        self._hf_processor = hf_processor
        self.ctx = _FakeProcessingContext()
        self._hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                spatial_merge_size=hf_processor.image_processor.merge_size,
            ),
            image_token_id=hf_processor.image_token_id,
            video_token_id=hf_processor.video_token_id,
            vision_start_token_id=hf_processor.vision_start_token_id,
            vision_end_token_id=hf_processor.vision_end_token_id,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self._hf_processor

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
        return self._hf_processor.image_processor

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        return self._hf_processor.tokenizer

    def get_hf_config(self) -> Any:
        return self._hf_config


class _FakeDummyInputs:
    def get_dummy_text(self, mm_counts: dict[str, int]) -> str:
        return "".join(
            "<|vision_start|><|image_pad|><|vision_end|>"
            for _ in range(mm_counts.get("image", 0))
        )


def _make_processor() -> Qwen3VLMultiModalProcessor:
    tokenizer = _make_tokenizer()
    hf_processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    info = _FakeProcessingInfo(hf_processor)
    return Qwen3VLMultiModalProcessor(info, _FakeDummyInputs(), cache=None)


def _make_images() -> list[Image.Image]:
    images = []
    for index in range(2):
        array = np.full((224, 224, 3), index * 40, dtype=np.uint8)
        images.append(Image.fromarray(array, "RGB"))
    return images


def _compact_reference_pixel_values(
    images: torch.Tensor,
    image_processor: Qwen2VLImageProcessor,
) -> torch.Tensor:
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    patches = image_processor.rescale_and_normalize(
        images,
        image_processor.do_rescale,
        image_processor.rescale_factor,
        image_processor.do_normalize,
        image_processor.image_mean,
        image_processor.image_std,
    )
    batch_size, channel, height, width = patches.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = patches.reshape(
        batch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
    return patches.reshape(batch_size * grid_h * grid_w, channel * patch_size**2)


def test_qwen3_fused_compact_preprocess_matches_reference() -> None:
    processor = _make_processor()
    image_processor = processor.info.get_image_processor()
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    height = patch_size * merge_size * 2
    width = patch_size * merge_size * 3
    images = torch.arange(
        2 * 3 * height * width,
        dtype=torch.int64,
    ).remainder(256).to(torch.uint8)
    images = images.reshape(2, 3, height, width)

    fused = _qwen3_vl_fused_compact_preprocess_uint8_bchw(
        images,
        image_processor.image_mean,
        image_processor.image_std,
        image_processor.rescale_factor,
        patch_size,
        merge_size,
    )
    reference = _compact_reference_pixel_values(images, image_processor)

    torch.testing.assert_close(fused, reference)


def test_qwen3_fast_preprocess_requires_merge_aligned_size() -> None:
    processor = _make_processor()
    image_processor = processor.info.get_image_processor()
    image_processor.do_resize = False
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size

    patch_aligned = torch.zeros(
        (1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    merge_aligned = torch.zeros(
        (1, 3, patch_size * merge_size, patch_size * merge_size),
        dtype=torch.uint8,
    )

    assert not Qwen3VLMultiModalProcessor._can_fast_preprocess_batched_images(
        image_processor,
        patch_aligned,
        {},
    )
    assert Qwen3VLMultiModalProcessor._can_fast_preprocess_batched_images(
        image_processor,
        merge_aligned,
        {},
    )


def test_qwen3_image_only_mm_only_matches_dummy_text_path() -> None:
    processor = _make_processor()
    mm_items = processor.data_parser.parse_mm_data({"image": _make_images()})
    hf_kwargs = {"return_mm_token_type_ids": False}

    baseline = BaseMultiModalProcessor._apply_hf_processor_mm_only(
        processor,
        mm_items,
        hf_kwargs,
        {},
    )
    optimized = processor._apply_hf_processor_mm_only(mm_items, hf_kwargs, {})

    assert "attention_mask" not in optimized
    assert torch.equal(optimized["image_grid_thw"], baseline["image_grid_thw"])

    patch_size = processor.info.get_image_processor().patch_size
    temporal_patch_size = processor.info.get_image_processor().temporal_patch_size
    baseline_pixel_values = baseline["pixel_values"]
    compact_pixel_values = optimized["pixel_values"]
    assert compact_pixel_values.shape[0] == baseline_pixel_values.shape[0]
    assert compact_pixel_values.shape[1] * temporal_patch_size == (
        baseline_pixel_values.shape[1]
    )
    expanded_compact = (
        compact_pixel_values.view(
            compact_pixel_values.shape[0],
            3,
            patch_size,
            patch_size,
        )
        .unsqueeze(2)
        .expand(-1, -1, temporal_patch_size, -1, -1)
        .reshape_as(baseline_pixel_values)
    )
    assert torch.equal(expanded_compact, baseline_pixel_values)


def test_qwen3_mm_only_fallback_ignores_tokenization_kwargs() -> None:
    processor = _make_processor()
    images = [
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB"),
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), "RGB"),
    ]
    mm_items = processor.data_parser.parse_mm_data({"image": images})

    optimized = processor._apply_hf_processor_mm_only(
        mm_items,
        {"return_mm_token_type_ids": False},
        {"truncation": False},
    )

    assert optimized["image_grid_thw"].shape[0] == 2


def test_qwen3_patch_embed_compact_image_matches_duplicated_temporal(
    default_vllm_config,
) -> None:
    patch_embed = Qwen3_VisionPatchEmbed(
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=5,
    )
    with torch.no_grad():
        patch_embed.proj.weight.copy_(
            torch.arange(
                patch_embed.proj.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(patch_embed.proj.weight)
        )
        patch_embed.proj.bias.copy_(
            torch.arange(
                patch_embed.proj.bias.numel(),
                dtype=torch.float32,
            )
        )
    compact_pixel_values = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    duplicated_pixel_values = (
        compact_pixel_values.view(3, 3, 2, 2)
        .unsqueeze(2)
        .expand(-1, -1, 2, -1, -1)
        .reshape(3, 24)
    )

    compact_embeds = patch_embed(compact_pixel_values)
    duplicated_embeds = patch_embed(duplicated_pixel_values)

    torch.testing.assert_close(compact_embeds, duplicated_embeds)


def test_qwen3_tokenized_text_and_images_apply_keeps_placeholder_ranges() -> None:
    processor = _make_processor()
    tokenizer = processor.info.get_tokenizer()
    mm_items = processor.data_parser.parse_mm_data({"image": _make_images()})
    prompt_token_ids = [
        tokenizer.unk_token_id,
        tokenizer.unk_token_id,
        tokenizer.vision_start_token_id,
        tokenizer.image_token_id,
        tokenizer.vision_end_token_id,
        tokenizer.unk_token_id,
        tokenizer.vision_start_token_id,
        tokenizer.image_token_id,
        tokenizer.vision_end_token_id,
        tokenizer.unk_token_id,
        tokenizer.unk_token_id,
    ]

    output = processor.apply(
        ProcessorInputs(
            prompt_token_ids,
            mm_items,
            hf_processor_mm_kwargs={"return_mm_token_type_ids": False},
        ),
        TimingContext(enabled=True),
    )

    assert len(output["mm_placeholders"]["image"]) == 2
    assert len(output["prompt_token_ids"]) > len(prompt_token_ids)
