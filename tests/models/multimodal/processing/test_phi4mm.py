# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for phi4mm's multimodal preprocessing kwargs."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["microsoft/Phi-4-multimodal-instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img"),
    [
        ({"dynamic_hd": 4}, 1329),
        ({"dynamic_hd": 16}, 4433),
        # the default num_crops of phi-4-multimodal is 36
        ({}, 9585),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Phi4MMMultiModalProcessor handles dynamic_hd properly."""
    # Avoid initializing CUDA early
    from vllm.model_executor.models.phi4mm import _IMAGE_PLACEHOLDER_TOKEN_ID

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"

    image_size = ctx.get_hf_config().embd_layer["image_embd_layer"]["crop_size"]
    dummy_image_size = (image_size * 7, image_size * 7)
    dummy_image = image_assets[0].pil_image.resize(dummy_image_size)
    mm_data = {"image": [dummy_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = processed_inputs["prompt_token_ids"].count(
        _IMAGE_PLACEHOLDER_TOKEN_ID
    )
    assert img_tok_count == expected_toks_per_img * num_imgs


class _StubModel:
    """get_mm_lora_token_counts for Phi4MM reads only its arguments
    (modality, num_mm_embeds), not any instance state, so an empty
    stub is sufficient to exercise it without constructing the full
    `nn.Module` (vision tower, audio tower, language model, etc.)."""


def _get_mm_lora_token_counts():
    # Imported lazily to avoid initializing CUDA early, consistent
    # with the pattern used in test_processor_override above.
    from vllm.model_executor.models.phi4mm import Phi4MMForCausalLM

    return Phi4MMForCausalLM.get_mm_lora_token_counts


@pytest.mark.parametrize("modality", ["image", "audio"])
@pytest.mark.parametrize("num_mm_embeds", [0, 1, 16, 197, 1500])
def test_mm_lora_token_counts_passthrough(modality, num_mm_embeds):
    """Phi4MM's tower/connector modules are shape-preserving in the
    token dimension (all downsampling happens upstream, inside tower
    processing), so tower and connector token counts should both
    equal the input embed count, for both image and audio modalities.
    """
    get_mm_lora_token_counts = _get_mm_lora_token_counts()
    stub = _StubModel()

    tower_tokens, connector_tokens = get_mm_lora_token_counts(
        stub,
        modality=modality,
        mm_kwargs=None,
        num_mm_embeds=num_mm_embeds,
    )

    assert tower_tokens == num_mm_embeds
    assert connector_tokens == num_mm_embeds


@pytest.mark.parametrize("modality", ["image_embeds", "audio_embeds"])
def test_mm_lora_token_counts_modality_prefix_match(modality):
    """Modality strings are matched by prefix (e.g. `image_embeds`
    should still match `image`), consistent with how vLLM passes
    modality-variant strings elsewhere in the codebase.
    """
    get_mm_lora_token_counts = _get_mm_lora_token_counts()
    stub = _StubModel()

    tower_tokens, connector_tokens = get_mm_lora_token_counts(
        stub,
        modality=modality,
        mm_kwargs=None,
        num_mm_embeds=42,
    )

    assert tower_tokens == 42
    assert connector_tokens == 42


def test_mm_lora_token_counts_unsupported_modality_raises():
    get_mm_lora_token_counts = _get_mm_lora_token_counts()
    stub = _StubModel()

    with pytest.raises(ValueError, match="Unsupported modality"):
        get_mm_lora_token_counts(
            stub,
            modality="video",
            mm_kwargs=None,
            num_mm_embeds=10,
        )
