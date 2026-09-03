# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Step3-VL precomputed image embedding inputs."""

import pytest
import torch

from vllm.model_executor.models.step3_vl import (
    Step3VLForConditionalGeneration,
    Step3VLImageEmbeddingInputs,
)


class _FakeStep3VL:
    @staticmethod
    def _process_image_features(image_features: torch.Tensor) -> torch.Tensor:
        return image_features


def test_image_embedding_inputs_construction():
    """Step3VLImageEmbeddingInputs should store embeddings in the data field."""
    image_embeds = torch.randn(2, 16, 64)

    inputs = Step3VLImageEmbeddingInputs(
        type="image_embeds",
        data=image_embeds,
    )

    assert inputs["type"] == "image_embeds"
    assert torch.equal(inputs["data"], image_embeds)
    assert torch.equal(inputs.data, image_embeds)


def test_image_embedding_inputs_validation_rejects_wrong_rank():
    """Validation should reject tensors with wrong rank."""
    with pytest.raises(ValueError, match="rank"):
        Step3VLImageEmbeddingInputs(
            type="image_embeds",
            data=torch.randn(16, 64),
        )


def test_process_image_embeds_does_not_require_pixel_input_fields():
    """The image_embeds branch should not reference patch pixel metadata."""
    image_embeds = torch.randn(2, 4, 8)
    image_input = Step3VLImageEmbeddingInputs(
        type="image_embeds",
        data=image_embeds,
    )

    outputs = Step3VLForConditionalGeneration._process_image_input(
        _FakeStep3VL(),
        image_input,
    )

    assert len(outputs) == 2
    assert torch.equal(outputs[0], image_embeds[0])
    assert torch.equal(outputs[1], image_embeds[1])
