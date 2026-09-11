# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from packaging.version import Version
from PIL import Image
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.model_executor.models.vision import FusedInputNorm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal

from ....conftest import ImageTestAssets
from ...utils import build_model_context


def test_jina_vl_processing_order() -> None:
    """Jina's document-first prompt keeps cached features and hashes aligned."""
    ctx = build_model_context(
        "jinaai/jina-reranker-m0",
        runner="pooling",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=1,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
        cache=cache,
    )

    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    query_image = Image.new("RGB", (128, 160), color=(255, 0, 0))
    document_image = Image.new("RGB", (192, 128), color=(0, 255, 0))

    def process(images: list[Image.Image]):
        return processor(
            placeholder * len(images),
            mm_items=processor.info.parse_mm_data({"image": images}),
        )

    query = process([query_image])
    document = process([document_image])
    pair = process([query_image, document_image])

    pair_items = pair["mm_kwargs"]["image"]
    assert pair["mm_hashes"]["image"] == [
        document["mm_hashes"]["image"][0],
        query["mm_hashes"]["image"][0],
    ]
    assert batched_tensors_equal(
        pair_items[0].get_data(),
        document["mm_kwargs"]["image"][0].get_data(),
    )
    assert batched_tensors_equal(
        pair_items[1].get_data(),
        query["mm_kwargs"]["image"][0].get_data(),
    )
    assert [item.length for item in pair["mm_placeholders"]["image"]] == [
        document["mm_placeholders"]["image"][0].length,
        query["mm_placeholders"]["image"][0].length,
    ]


@pytest.mark.parametrize(
    ("image_mean", "image_std", "rescale_factor", "is_identity"),
    [
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, True),
        ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25], 1 / 255, False),
    ],
)
def test_fused_input_norm_initialization_on_device(
    monkeypatch: pytest.MonkeyPatch,
    image_mean: list[float],
    image_std: list[float],
    rescale_factor: float,
    is_identity: bool,
):
    """Identity detection must not synchronize the default device."""
    original_allclose = torch.allclose

    def cpu_allclose(input: torch.Tensor, other: torch.Tensor, *args, **kwargs):
        assert input.device.type == "cpu"
        assert other.device.type == "cpu"
        return original_allclose(input, other, *args, **kwargs)

    monkeypatch.setattr(torch, "allclose", cpu_allclose)
    # Exercise the real accelerator when available. The meta device gives the
    # CPU-only test shard the same non-CPU default-device semantics without
    # requiring a CUDA-enabled PyTorch build.
    default_device = "cuda" if torch.cuda.is_available() else "meta"
    with torch.device(default_device):
        input_norm = FusedInputNorm(image_mean, image_std, rescale_factor)

    assert input_norm.is_identity is is_identity
    if is_identity:
        assert input_norm.weight is None
        assert input_norm.bias is None
    else:
        assert input_norm.weight.device.type == default_device
        assert input_norm.bias.device.type == default_device


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img", "expected_pixels_shape"),
    [
        ({}, 1426, (5704, 1176)),
        ({"min_pixels": 64**2, "max_pixels": 512**2}, 330, (1320, 1176)),
        (
            {
                "size": {
                    "shortest_edge": 64**2,
                    "longest_edge": 512**2,
                },
            },
            330,
            (1320, 1176),
        ),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_toks_per_img: int,
    expected_pixels_shape: tuple[int, int],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Qwen2VLMultiModalProcessor handles min/max pixels properly."""
    if (
        Version(TRANSFORMERS_VERSION) < Version("5.2.0")
        and "size" in mm_processor_kwargs
    ):
        pytest.skip("`size` ignored by `Qwen2VLProcessor.__call__`")

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape[0] * num_imgs
    assert pixel_shape[1] == expected_pixels_shape[1]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28},
        {"min_pixels": 28 * 28, "max_pixels": 1283 * 28 * 28},
        {"size": {"shortest_edge": 28 * 28, "longest_edge": 1280 * 28 * 28}},
        {"size": {"shortest_edge": 28 * 28, "longest_edge": 1283 * 28 * 28}},
    ],
)
def test_get_image_size_with_most_features(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
):
    if (
        Version(TRANSFORMERS_VERSION) < Version("5.2.0")
        and "size" in mm_processor_kwargs
    ):
        pytest.skip("`size` ignored by `Qwen2VLProcessor.__call__`")

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    hf_processor = processor.info.get_hf_processor(**mm_processor_kwargs)
    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size

    max_image_size = processor.info.get_image_size_with_most_features()
    max_tokens = processor.info.get_num_image_tokens(
        image_width=max_image_size.width,
        image_height=max_image_size.height,
        image_processor=hf_processor.image_processor,
        mm_kwargs=mm_processor_kwargs,
    )

    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    for asset in image_assets:
        mm_data = {"image": [asset.pil_image]}
        processed_inputs = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=mm_processor_kwargs,
        )
        grid_thw = processed_inputs["mm_kwargs"].get_data()["image_grid_thw"].tolist()
        t, h, w = grid_thw[0]
        tokens = (t * h * w) // (merge_size**2)
        assert tokens < max_tokens


@pytest.mark.parametrize(
    "model_id", ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]
)
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_mm_device_do_normalize(
    image_assets: ImageTestAssets,
    model_id: str,
    num_imgs: int,
):
    """Ensure that enable mm_device_do_normalize yields the correct result."""

    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": num_imgs},
    )
    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs_with_normalize = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
    )
    pixel_values_with_normalize = processed_inputs_with_normalize[
        "mm_kwargs"
    ].get_data()["pixel_values"]
    dtype = pixel_values_with_normalize.dtype

    processed_inputs_without_normalize = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"do_normalize": False, "do_rescale": False},
    )
    pixel_values_without_normalize = processed_inputs_without_normalize[
        "mm_kwargs"
    ].get_data()["pixel_values"]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    input_norm = FusedInputNorm.from_model_config(ctx.model_config)
    pixel_values_do_input_norm = input_norm(
        pixel_values_without_normalize.to(dtype), dtype
    )

    torch.testing.assert_close(pixel_values_with_normalize, pixel_values_do_input_norm)
