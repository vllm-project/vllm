# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from transformers import CLIPVisionConfig, LlavaConfig, SiglipVisionConfig

from vllm.model_executor.models.llava import LlavaForConditionalGeneration


@pytest.mark.parametrize(
    ("strategy", "num_selected_tokens"),
    [("default", 576), ("full", 577)],
)
@pytest.mark.parametrize("num_images", [0, 1, 10])
def test_clip_lora_counts_include_cls_before_feature_selection(
    strategy, num_selected_tokens, num_images
):
    model = SimpleNamespace(
        config=LlavaConfig(
            vision_config=CLIPVisionConfig(image_size=336, patch_size=14),
            vision_feature_select_strategy=strategy,
        )
    )

    counts = LlavaForConditionalGeneration.get_mm_lora_token_counts(
        model,
        modality="image",
        mm_kwargs=None,
        num_mm_embeds=num_images * num_selected_tokens,
    )

    assert counts == (num_images * 577, num_images * num_selected_tokens)


def test_siglip_lora_counts_do_not_add_cls():
    model = SimpleNamespace(config=LlavaConfig(vision_config=SiglipVisionConfig()))

    assert LlavaForConditionalGeneration.get_mm_lora_token_counts(
        model, modality="image", mm_kwargs=None, num_mm_embeds=5760
    ) == (5760, 5760)
