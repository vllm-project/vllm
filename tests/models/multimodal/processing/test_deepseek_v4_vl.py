# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 VL processing helpers."""

import json

import pytest
import torch
from PIL import Image
from transformers import DeepseekV4Config

from tests.models.utils import build_model_context
from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    build_image_block_pad_free,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.renderers.chat_utils.item_tracker import _get_embeds_data
from vllm.renderers.deepseek_v4 import DeepseekV41Renderer
from vllm.renderers.params import ChatParams
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer

# Vision fields of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
VISION_CONFIG = dict(
    vision_n_layers=24,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    vision_max_n_token=384,
    vision_min_pixels=147456,
    vision_max_wh_ratio=8,
)


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
    config = DeepseekV4Config(**VISION_CONFIG)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    processor = DeepseekV4VLProcessor(config)

    dummy = info.get_image_size_with_most_features()
    dummy_features = _feature_counts(processor, dummy.width, dummy.height)
    wide_features = _feature_counts(processor, 1932, 336)

    assert dummy == (1932, 336)
    assert wide_features <= dummy_features


def test_get_image_size_with_most_features_respects_max_wh_ratio():
    config = DeepseekV4Config(**VISION_CONFIG)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    dummy = info.get_image_size_with_most_features()
    assert dummy.width / dummy.height <= config.vision_max_wh_ratio


@pytest.mark.parametrize("sizes", [[(224, 224)], [(336, 560), (840, 224), (336, 560)]])
def test_v41_image_metadata_only_roundtrip(sizes, monkeypatch):
    """EC metadata preserves image spans without running image preprocessing."""
    ctx = build_model_context(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        limit_mm_per_prompt={"image": len(sizes)},
    )
    mm_config = ctx.model_config.multimodal_config
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", size) for size in sizes]
    prompt = processor.dummy_inputs.get_dummy_text({"image": len(images)})
    raw = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality="image",
            identifier=raw["mm_hashes"]["image"][index],
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
        json.dumps([published[feature.identifier]["metadata"] for feature in features])
    )
    assert all(set(item) == {"llm_grid"} for item in metadata)
    image_data = _get_embeds_data("image", metadata, processor)
    renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
    renderer.model_config = ctx.model_config
    renderer.mm_processor = processor
    renderer.tokenizer = get_deepseek_v41_tokenizer(processor.info.get_tokenizer())
    _, rendered = renderer.render_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_embeds", "image_embeds": item, "uuid": str(i)}
                    for i, item in enumerate(metadata)
                ],
            }
        ],
        ChatParams(chat_template_kwargs={"thinking": False, "tokenize": False}),
    )
    assert rendered["multi_modal_uuids"]["image"] == [str(i) for i in range(len(sizes))]
    assert torch.equal(
        rendered["multi_modal_data"]["image"]["llm_grid"], image_data["llm_grid"]
    )

    def unexpected_preprocessing(*args, **kwargs):
        pytest.fail("The EC consumer must not preprocess images")

    monkeypatch.setattr(processor, "_call_hf_processor", unexpected_preprocessing)
    remote = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"image": image_data}),
        hf_processor_mm_kwargs={},
    )
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]
    remote_data = remote["mm_kwargs"].get_data()
    assert set(remote_data) == {"llm_grid"}
    assert torch.equal(remote_data["llm_grid"], raw["mm_kwargs"].get_data()["llm_grid"])

    mm_config.allow_missing_mm_embeddings = False
    ordinary_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    with pytest.raises(ValueError, match="should contain the fields"):
        ordinary_processor.info.parse_mm_data({"image": image_data})

    from vllm.models.deepseek_v41 import DeepseekV41ForCausalLM

    assert DeepseekV41ForCausalLM.get_placeholder_str(
        "image_embeds", 0
    ) == DeepseekV41ForCausalLM.get_placeholder_str("image", 0)
    lengths = [position.length for position in raw["mm_placeholders"]["image"]]
    embeddings = torch.randn(sum(lengths), ctx.model_config.hf_config.hidden_size)
    supplied = ordinary_processor(
        prompt,
        mm_items=ordinary_processor.info.parse_mm_data(
            {"image": {**image_data, "image_embeds": embeddings}}
        ),
        hf_processor_mm_kwargs={},
    )
    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    outputs = model.embed_multimodal(**supplied["mm_kwargs"].get_data())
    assert supplied["prompt_token_ids"] == raw["prompt_token_ids"]
    assert [len(output) for output in outputs] == lengths
    assert torch.equal(torch.cat(outputs), embeddings)
