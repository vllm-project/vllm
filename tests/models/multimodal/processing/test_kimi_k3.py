# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ECTransferConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.renderers import ChatParams, renderer_from_config
from vllm.renderers.chat_utils.item_tracker import _get_embeds_data
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

from ...utils import build_model_context

_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="module")
def image_processors():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    k25_fused = getattr(processor.info, "image_processor")  # noqa: B009
    assert isinstance(k25_fused, KimiK25FusedVisionProcessor)
    k3_hf = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return k25_fused, k3_hf


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("with_embeddings", [False, True])
def test_image_metadata_only_roundtrip(with_embeddings: bool):
    """EC metadata preserves original dimensions and repeated-image placeholders."""
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 4})
    mm_config = ctx.model_config.multimodal_config
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [_noise_image(w, h) for w, h in [(310, 470), (7300, 30), (13, 9)]]
    images.append(images[0])
    prompt = processor.dummy_inputs.get_dummy_text({"image": len(images)})
    raw = processor(prompt, mm_items=processor.info.parse_mm_data({"image": images}))
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier=str(index),
            mm_position=position,
        )
        for index, (item, position) in enumerate(
            zip(raw["mm_kwargs"]["image"], raw["mm_placeholders"]["image"])
        )
    ]
    published = collect_ec_item_metadata(
        features, PlaceholderMetadataResolver(ctx.model_config)
    )
    metadata = json.loads(
        json.dumps([entry["metadata"] for entry in published.values()])
    )
    for image, item in zip(images, metadata):
        assert item == {"image_sizes": list(image.size)}

    if with_embeddings:
        embeddings = [
            torch.full(
                (position.get_num_embeds(), ctx.model_config.get_hidden_size()),
                float(index),
            )
            for index, position in enumerate(raw["mm_placeholders"]["image"])
        ]
        remote_items = [
            {**item, "image_embeds": embedding}
            for item, embedding in zip(metadata, embeddings)
        ]
    else:
        remote_items = metadata
    remote_data = _get_embeds_data("image", remote_items, processor)
    remote = processor(
        prompt, mm_items=processor.info.parse_mm_data({"image": remote_data})
    )
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]
    assert "pixel_values" not in remote["mm_kwargs"].get_data()
    if with_embeddings:
        for item, embedding in zip(remote["mm_kwargs"]["image"], embeddings):
            assert item is not None
            torch.testing.assert_close(item["image_embeds"].data, embedding)

    mm_config.allow_missing_mm_embeddings = False
    ordinary_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    with pytest.raises(ValueError, match="should contain the fields"):
        ordinary_processor.info.parse_mm_data(
            {"image": {"image_sizes": remote_data["image_sizes"]}}
        )


def test_image_metadata_only_chat():
    """image_embeds inputs preserve K3's native chat tokens and ranges."""
    config = VllmConfig(
        model_config=ModelConfig(
            _MODEL_ID,
            tokenizer_mode="kimi_k3",
            trust_remote_code=True,
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 3},
            mm_processor_cache_gb=0,
        ),
        # Rendering needs no KV cache; keep CPU MLA and hybrid-cache defaults aligned.
        cache_config=CacheConfig(enable_prefix_caching=False),
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector", ec_role="ec_consumer"
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    images = [_noise_image(31, 47), _noise_image(73, 30), _noise_image(31, 47)]
    raw_parts = [
        {"type": "image_url", "image_url": {"url": encode_image_url(image)}}
        for image in images
    ]
    metadata_parts = [
        {
            "type": "image_embeds",
            "image_embeds": {
                "image_sizes": list(image.size),
            },
        }
        for image in images
    ]
    renderer = renderer_from_config(config)
    try:
        results = []
        for parts in (raw_parts, metadata_parts):
            _, prompts = renderer.render_chat(
                [[{"role": "user", "content": parts}]], ChatParams()
            )
            results.append(prompts[0])
    finally:
        renderer.shutdown()
    raw, remote = results
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]


@pytest.mark.parametrize(
    ("width", "height"),
    [(310, 470), (7300, 30), (13, 9)],
)
def test_fused_image_processor_matches_hf(image_processors, width: int, height: int):
    fused, hf = image_processors
    image = _noise_image(width, height)
    expected = hf.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    actual = fused.preprocess([{"type": "image", "image": image}], return_tensors="pt")

    torch.testing.assert_close(actual["grid_thws"], expected["grid_thws"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    assert fused.media_tokens_calculator(
        {"type": "image", "image": image}
    ) == hf.media_tokens_calculator({"type": "image", "image": image})


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 2})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = processor.info.get_hf_config().image_placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(normalized_values, torch.Tensor)

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(raw_values, torch.Tensor)
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )
