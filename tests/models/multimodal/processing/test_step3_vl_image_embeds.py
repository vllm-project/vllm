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
    img_output_tokens = 2
    patch_output_tokens = 1

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


@pytest.mark.parametrize("clone", [True, False])
def test_postprocess_encoder_output_respects_clone_for_zero_patch_output(clone):
    """A replay must overwrite only output whose alias was requested."""
    graph_output = torch.arange(4, dtype=torch.float16).reshape(2, 2)
    dest: dict[int, torch.Tensor] = {}

    Step3VLForConditionalGeneration.postprocess_encoder_output(
        _FakeStep3VL(),
        {"global": graph_output},
        indices=[0],
        per_item_out_tokens=[2],
        dest=dest,
        clone=clone,
        batch_mm_kwargs={"num_patches": [0]},
    )

    postprocessed = dest[0]
    snapshot = postprocessed.clone()
    graph_output.fill_(-1)

    if clone:
        assert torch.equal(postprocessed, snapshot)
    else:
        assert torch.equal(postprocessed, graph_output)
