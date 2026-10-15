# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 VL processing helpers."""

from PIL import Image

from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    build_image_block_pad_free,
    image_llm_grid_from_size,
    load_image,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

from ...utils import build_model_context


class _ConfigCtx:
    def __init__(self, config: DeepseekV4Config) -> None:
        self._config = config

    def get_hf_config(self, *args, **kwargs):
        return self._config


def _feature_counts(
    processor: DeepseekV4VLProcessor, width: int, height: int
) -> tuple[int, int]:
    out = processor(images=[Image.new("RGB", (width, height))])
    n_llm_h, n_llm_w = out["llm_grid"][0].tolist()
    types, _ = build_image_block_pad_free(n_llm_h, n_llm_w)
    return len(types), out["patches"].shape[0]


def test_get_image_size_with_most_features_beats_wide_real_image():
    """Dummy size must maximize tokens/patches within the budget (#59271).

    A square under-counts because each aligner row pays an IMAGE_NEW_LINE.
    """
    config = DeepseekV4Config(vision_n_layers=24)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    processor = DeepseekV4VLProcessor(config)

    dummy = info.get_image_size_with_most_features()
    dummy_features = _feature_counts(processor, dummy.width, dummy.height)
    wide_features = _feature_counts(processor, 1932, 336)

    assert dummy == (1932, 336)
    assert wide_features <= dummy_features


def test_get_image_size_with_most_features_respects_max_wh_ratio():
    config = DeepseekV4Config(vision_n_layers=24)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    dummy = info.get_image_size_with_most_features()
    assert dummy.width / dummy.height <= config.vision_max_wh_ratio


def test_image_llm_grid_matches_pixel_path():
    config = DeepseekV4Config(vision_n_layers=24)
    for width, height in [(1280, 720), (640, 480), (1932, 336), (64, 64)]:
        image = Image.new("RGB", (width, height), color=(12, 40, 90))
        _, _, _, n_llm_h, n_llm_w = load_image(
            image,
            patch_size=config.vision_patch_size,
            downsample_ratio=config.vision_downsample_ratio,
            max_n_token=config.vision_max_n_token,
            min_pixels=config.vision_min_pixels,
            max_wh_ratio=config.vision_max_wh_ratio,
        )
        assert image_llm_grid_from_size(
            width,
            height,
            patch_size=config.vision_patch_size,
            downsample_ratio=config.vision_downsample_ratio,
            max_n_token=config.vision_max_n_token,
            min_pixels=config.vision_min_pixels,
            max_wh_ratio=config.vision_max_wh_ratio,
        ) == (n_llm_h, n_llm_w)


def test_routing_metadata_fast_path_requires_exact_processor_opt_in() -> None:
    capability = "_supports_routing_metadata_image_fast_path"
    assert DeepseekV4VLMultiModalProcessor.__dict__[capability]
    assert DeepseekV4VLProcessingInfo.__dict__[capability]
    assert (
        DeepseekV4VLMultiModalProcessor.__dict__[
            "_routing_metadata_processing_info_type"
        ]
        is DeepseekV4VLProcessingInfo
    )


def test_routing_metadata_matches_full_image_processing():
    ctx = build_model_context(
        "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        limit_mm_per_prompt={"image": 2, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )
    image_token_id = processor.info.get_image_placeholder_token_id()
    images = [
        Image.new("RGB", (1280, 720), color=(12, 40, 90)),
        Image.new("RGB", (640, 480), color=(80, 20, 40)),
    ]

    def make_inputs() -> ProcessorInputs:
        return ProcessorInputs(
            prompt=[image_token_id, image_token_id],
            mm_data_items=processor.info.parse_mm_data({"image": images}),
        )

    full = processor.apply(make_inputs(), TimingContext(enabled=False))
    routing = processor.apply_routing_metadata(
        make_inputs(),
        TimingContext(enabled=False),
    )

    assert routing["prompt_token_ids"] == full["prompt_token_ids"]
    assert routing["mm_hashes"] == full["mm_hashes"]
    assert routing["mm_placeholders"] == full["mm_placeholders"]
    assert routing["mm_kwargs"] == {}
