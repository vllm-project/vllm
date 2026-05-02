# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# TODO: to be updated to "google/gemma-4-e2b-it" once the models are available
GEMMA4_MODEL_ID = "google/gemma-4-E2B-it"


@pytest.mark.parametrize(
    "image_width,image_height,max_soft_tokens",
    [
        # Production repro: a 3x900 image (extreme aspect ratio) made the
        # prompt-side estimator return 289 while the HF Gemma 4 image
        # processor's vision tower output capped at 280, producing the
        # "Attempted to assign 280 multimodal tokens to 289 placeholders"
        # mismatch that crashed EngineCore.
        (900, 3, 280),
        (3, 900, 280),
        # Same pathology should hold for the video-frame budget (70 tokens).
        (900, 3, 70),
        # And for any other supported budget.
        (4000, 2, 1120),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
    model_id: str,
    image_width: int,
    image_height: int,
    max_soft_tokens: int,
):
    """Regression for the Gemma 3/4 multimodal crash.

    `_compute_num_soft_tokens` must never return a value larger than
    `max_soft_tokens`. The HF Gemma 4 image processor clamps its vision
    tower output to that value; if the prompt-side estimator returns more,
    the prompt has more `image` placeholder tokens than the encoder will
    fill, and `_merge_multimodal_embeddings` raises `ValueError` deep in
    the model forward.
    """
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    num_soft_tokens = processor.info._compute_num_soft_tokens(
        image_width=image_width,
        image_height=image_height,
        max_soft_tokens=max_soft_tokens,
    )

    assert num_soft_tokens <= max_soft_tokens, (
        f"_compute_num_soft_tokens returned {num_soft_tokens} for "
        f"image_width={image_width}, image_height={image_height}, "
        f"max_soft_tokens={max_soft_tokens} — exceeds the cap that the HF "
        f"image processor enforces on its vision tower output. This is "
        f"the placeholder/encoder count mismatch that crashes EngineCore."
    )


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt accurately restricts multiple images."""
    # We only allow 1 image
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Provide 2 images in the prompt
    prompt = "<image><image>"
    # image_assets usually has multiple images
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    # Expect ValueError when exceeding limit
    with pytest.raises(ValueError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


# ---------------------------------------------------------------------------
# Test-local pin for the typed exception contract
# ---------------------------------------------------------------------------
#
# The clamp in ``Gemma4ProcessingInfo._compute_num_soft_tokens`` keeps the
# prompt-side estimate from ever exceeding ``max_soft_tokens``, so under
# normal operation production code never needs a typed exception for the
# placeholder/encoder count mismatch — and ``gemma4_mm.py`` does not export
# one. The class and row-counting helper below are defined here to pin the
# shape any future defense-in-depth reintroduction must match: subclass
# ``ValueError`` so existing ``except ValueError`` handlers keep catching
# it; keyword-only ``actual`` / ``expected`` constructor; message string
# carries both counts; row-counting helper walks Tensor / list / tuple.


class Gemma4MultimodalPlaceholderMismatch(ValueError):
    """Pinned shape for a typed placeholder/encoder count mismatch."""

    def __init__(self, *, actual: int, expected: int) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(
            f"Gemma 4 multimodal placeholder/encoder count mismatch: "
            f"prompt has {expected} placeholder token(s) but the "
            f"vision/audio tower produced {actual} embedding(s)."
        )


def _count_multimodal_embedding_rows(mm_embeds) -> int:
    """Recursively count leading-dim rows across Tensor / list / tuple."""
    if isinstance(mm_embeds, torch.Tensor):
        return int(mm_embeds.shape[0])
    return sum(_count_multimodal_embedding_rows(t) for t in mm_embeds)


def test_gemma4_multimodal_placeholder_mismatch_is_value_error():
    """The typed exception must subclass ``ValueError`` so that existing
    ``except ValueError`` handlers (e.g. inside vLLM's preprocess and
    serving paths) keep catching it."""
    exc = Gemma4MultimodalPlaceholderMismatch(actual=289, expected=280)
    assert isinstance(exc, ValueError)


def test_gemma4_multimodal_placeholder_mismatch_carries_counts():
    """``actual`` and ``expected`` must be exposed as attributes so the
    engine layer can build a structured client-input error report
    without re-parsing the message string."""
    exc = Gemma4MultimodalPlaceholderMismatch(actual=289, expected=280)

    assert exc.actual == 289
    assert exc.expected == 280
    # Message should mention both counts so it is self-explanatory in
    # logs even if the structured fields are dropped somewhere.
    msg = str(exc)
    assert "289" in msg
    assert "280" in msg


def test_gemma4_multimodal_placeholder_mismatch_requires_kwargs():
    """The constructor is keyword-only to keep call sites self-documenting
    (``actual=`` vs ``expected=`` are easy to swap positionally)."""
    with pytest.raises(TypeError):
        Gemma4MultimodalPlaceholderMismatch(289, 280)  # type: ignore[misc]


@pytest.mark.parametrize(
    "mm_embeds_factory,expected_rows",
    [
        # Single tensor: rows along leading dim.
        (lambda: torch.zeros(280, 16), 280),
        # List of tensors: sum of leading dims (e.g. 2 images).
        (lambda: [torch.zeros(280, 16), torch.zeros(70, 16)], 350),
        # Tuple of tensors: same accounting as list.
        (lambda: (torch.zeros(70, 16), torch.zeros(70, 16)), 140),
        # Empty list: zero rows.
        (lambda: [], 0),
    ],
)
def test_count_multimodal_embedding_rows(mm_embeds_factory, expected_rows):
    """``_count_multimodal_embedding_rows`` must mirror
    ``_merge_multimodal_embeddings``'s scattering: each leading-dim row
    of every tensor (recursively flattened) is one multimodal token.
    """
    assert _count_multimodal_embedding_rows(mm_embeds_factory()) == expected_rows
