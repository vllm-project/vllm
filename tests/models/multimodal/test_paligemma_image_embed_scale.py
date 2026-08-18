# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for PaliGemma image-embedding scaling (issue #52667).

Gemma applies its ``sqrt(hidden_size)`` normalizer to *text* embeddings inside
``GemmaModel.embed_input_ids`` and consumes the merged ``inputs_embeds`` as-is in
``forward``. PaliGemma must therefore hand its projected image features to the
language model unscaled — the previous ``* hidden_size**-0.5`` left image rows at
``1/hidden_size`` of the intended magnitude relative to text.

This exercises the ``image_embeds`` pass-through path, which needs no model
weights: ``embed_multimodal`` must return user-provided image embeddings
unchanged.
"""

import torch

from vllm.model_executor.models.paligemma import (
    PaliGemmaForConditionalGeneration,
)


def test_image_embeds_are_not_rescaled():
    # Bypass __init__: the image_embeds path only uses two pure instance methods
    # (_parse_and_validate_image_input, _process_image_input) and no weights.
    model = PaliGemmaForConditionalGeneration.__new__(PaliGemmaForConditionalGeneration)

    image_embeds = torch.randn(4, 16)
    out = model.embed_multimodal(image_embeds=image_embeds)

    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, image_embeds), (
        "image embeddings must pass through unscaled; a residual inverse scale "
        "(hidden_size**-0.5) mismatches Gemma's text-side sqrt(hidden_size) "
        "normalizer"
    )
