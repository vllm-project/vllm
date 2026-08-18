# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.paligemma import PaliGemmaForConditionalGeneration


def test_embed_multimodal_preserves_projected_image_embeddings():
    """Projected image embeddings should keep their input scale."""
    model = object.__new__(PaliGemmaForConditionalGeneration)
    model.config = SimpleNamespace(hidden_size=2048)
    image_embeds = torch.randn(2, 3, 2048)

    output = model.embed_multimodal(image_embeds=image_embeds)

    torch.testing.assert_close(output, image_embeds)
