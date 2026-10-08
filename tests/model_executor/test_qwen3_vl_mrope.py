# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.image_prune.rate import (
    compute_retained_tokens_count as rate_retained_tokens_count,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune.evs import compute_mrope_for_media


@pytest.fixture(autouse=True, scope="module")
def _force_cpu_default_device():
    # _get_mrope_input_positions returns CPU tensors (via torch.from_numpy).
    # Ensure the default device is CPU so the rest of the test tensors match.
    original = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(original)


IMAGE_TOKEN_ID = 999
VIDEO_TOKEN_ID = 888
VISION_START_TOKEN_ID = 777
VISION_END_TOKEN_ID = 778


@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass
class DummyConfig:
    image_token_id: int = IMAGE_TOKEN_ID
    video_token_id: int = VIDEO_TOKEN_ID
    vision_start_token_id: int = VISION_START_TOKEN_ID
    vision_end_token_id: int = VISION_END_TOKEN_ID
    vision_config: DummyVisionConfig = dataclasses.field(
        default_factory=DummyVisionConfig
    )


def make_video_embedding(
    t, h, w, interleave_text_tokens: tuple[int, int], video_pruning_rate: float = 0.0
):
    """Make a video embedding for a given video size and pruning rate.

    Args:
        t: Number of frames.
        h: Number of rows.
        w: Number of columns.
        interleave_text_tokens: Tuple of minimum and maximum number of text tokens to
            interleave with the video.
        video_pruning_rate: Pruning rate for the video.

    Returns:
        Tuple of (unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask)

    """
    unpruned_tokens_sequence = []
    population = list(range(1, 100))

    for _ in range(t):
        num_prefix_tokens = random.randint(
            interleave_text_tokens[0], interleave_text_tokens[1]
        )

        prefix_tokens = random.choices(population, k=num_prefix_tokens)
        vision_tokens = (
            [VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * h * w + [VISION_END_TOKEN_ID]
        )

        unpruned_tokens_sequence.extend(prefix_tokens)
        unpruned_tokens_sequence.extend(vision_tokens)

    unpruned_tokens_sequence = torch.tensor(unpruned_tokens_sequence, dtype=torch.long)
    video_token_mask = unpruned_tokens_sequence == VIDEO_TOKEN_ID

    pruning_mask = torch.bernoulli(video_token_mask.float() * video_pruning_rate).bool()  # type: ignore[attr-defined]
    # Sanity check that we don't prune what should not be pruned.
    assert not pruning_mask[~video_token_mask].any()

    retention_mask = ~pruning_mask
    pruned_tokens_sequence = unpruned_tokens_sequence[retention_mask]
    return unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_thw", [[3, 8, 7], [128, 10, 12]])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
@pytest.mark.parametrize("num_suffix_tokens", [0, 7])
@pytest.mark.parametrize("video_pruning_rate", [0, 0.25, 0.75])
@pytest.mark.parametrize("interleave_text_tokens", [(0, 0), (1, 4)])
def test_match_qwen3vl_mrope_evs_on(
    spatial_merge_size: int,
    num_prefix_tokens: int,
    grid_thw: tuple[int, int, int],
    num_suffix_tokens: int,
    video_pruning_rate: float,
    interleave_text_tokens: tuple[int, int],
):
    hf_config = DummyConfig()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size

    t, h, w = grid_thw
    population = list(range(1, 100))
    prefix_tokens = random.choices(population, k=num_prefix_tokens)
    suffix_tokens = random.choices(population, k=num_suffix_tokens)

    video_tokens, video_tokens_pruned, retention_mask = make_video_embedding(
        t,
        h // spatial_merge_size,
        w // spatial_merge_size,
        interleave_text_tokens=interleave_text_tokens,
        video_pruning_rate=video_pruning_rate,
    )
    assert len(video_tokens) == len(retention_mask)

    input_tokens = prefix_tokens + video_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + video_tokens_pruned.tolist() + suffix_tokens

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ],
        dim=0,
    )

    # Build the GT mrope for unpruned input.
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=len(input_tokens)),
    )
    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[mm_feature],
        config=hf_config,
    )

    # Compute mrope for a video-only media (unpruned).
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor(grid_thw),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="video",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=0, length=video_tokens.numel()),
    )
    video_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=video_tokens.tolist(),
        mm_features=[mm_feature],
        config=hf_config,
    )
    video_mrope = video_mrope.permute(1, 0)  # [N, 3]
    hidden_size = 16

    is_video_embed = torch.isin(
        video_tokens_pruned, torch.tensor([VIDEO_TOKEN_ID], dtype=torch.long)
    )

    expanded_positions = torch.full(
        (len(video_tokens_pruned), 5),
        fill_value=-100,
        device=video_mrope.device,
        dtype=torch.long,
    )
    expanded_positions[is_video_embed, :3] = video_mrope[retention_mask][is_video_embed]
    expanded_positions[~is_video_embed, :3] = video_mrope[retention_mask][
        ~is_video_embed
    ]

    is_vision_start = video_tokens_pruned == VISION_START_TOKEN_ID
    expanded_positions[..., 3] = is_vision_start
    expanded_positions[..., 4] = is_video_embed

    # Check that all positions were filled, since we initialized them as negative.
    assert (expanded_positions >= 0).all()

    video_embeddings = torch.empty(
        (len(video_tokens_pruned), hidden_size), device=video_mrope.device
    )

    video_embeddings = torch.cat(
        [
            video_embeddings,
            expanded_positions.float(),
        ],
        dim=1,
    )
    multimodal_embeddings = [video_embeddings]

    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # Initialize computed_mrope with sequential positions for all prefix tokens
    computed_mrope = torch.empty((3, len(input_tokens_pruned)), dtype=torch.long)
    computed_mrope[:, 0 : len(prefix_tokens)] = expected_mrope[
        :, 0 : len(prefix_tokens)
    ]

    # Paranoia check that computed_mrope is wrong.
    assert not torch.equal(computed_mrope, expected_mrope_masked)

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=multimodal_embeddings,
        mrope_positions=computed_mrope,
        num_computed_tokens=len(prefix_tokens),
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    assert torch.equal(actual_mrope, expected_mrope_masked)


def make_image_embedding(h, w, image_pruning_rate: float = 0.0, seed: int = 1234):
    """Make an image embedding for a given image size and pruning rate.

    Args:
        h: Number of rows (post spatial merge).
        w: Number of columns (post spatial merge).
        image_pruning_rate: Pruning rate for the image.
        seed: Seed for the deterministic retention subset.

    Returns:
        Tuple of (unpruned_tokens_sequence, pruned_tokens_sequence,
        retention_mask)

    """
    unpruned_tokens_sequence = (
        [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * h * w + [VISION_END_TOKEN_ID]
    )
    unpruned_tokens_sequence = torch.tensor(unpruned_tokens_sequence, dtype=torch.long)
    image_token_mask = unpruned_tokens_sequence == IMAGE_TOKEN_ID

    retained_count = rate_retained_tokens_count(h * w, image_pruning_rate)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(h * w, generator=g)
    image_retention_mask = torch.zeros(h * w, dtype=torch.bool)
    image_retention_mask[perm[:retained_count]] = True
    # Retain vision_start / vision_end; only image tokens are pruned.
    retention_mask = torch.cat(
        [
            torch.ones(1, dtype=torch.bool),
            image_retention_mask,
            torch.ones(1, dtype=torch.bool),
        ]
    )
    # Sanity check that only image tokens are ever pruned.
    assert retention_mask[~image_token_mask].all()

    pruned_tokens_sequence = unpruned_tokens_sequence[retention_mask]
    return unpruned_tokens_sequence, pruned_tokens_sequence, retention_mask


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("grid_hw", [[8, 7], [12, 10]])
@pytest.mark.parametrize("num_prefix_tokens", [1, 11])
@pytest.mark.parametrize("num_suffix_tokens", [0, 7])
@pytest.mark.parametrize("image_pruning_rate", [0.0, 0.25, 0.75])
@pytest.mark.parametrize(
    "num_computed_extra_tokens",
    [0, 1],
    ids=["at_vision_start", "past_vision_start"],
)
def test_match_qwen3vl_mrope_rate_on(
    spatial_merge_size: int,
    grid_hw: tuple[int, int],
    num_prefix_tokens: int,
    num_suffix_tokens: int,
    image_pruning_rate: float,
    num_computed_extra_tokens: int,
):
    hf_config = DummyConfig()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size

    h, w = grid_hw
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size
    num_tokens = llm_grid_h * llm_grid_w
    population = list(range(1, 100))
    prefix_tokens = random.choices(population, k=num_prefix_tokens)
    suffix_tokens = random.choices(population, k=num_suffix_tokens)

    image_tokens, image_tokens_pruned, retention_mask = make_image_embedding(
        llm_grid_h,
        llm_grid_w,
        image_pruning_rate=image_pruning_rate,
    )

    input_tokens = prefix_tokens + image_tokens.tolist() + suffix_tokens
    input_tokens_pruned = prefix_tokens + image_tokens_pruned.tolist() + suffix_tokens

    whole_sequence_retention_mask = torch.cat(
        [
            torch.ones(len(prefix_tokens), dtype=torch.bool),
            retention_mask,
            torch.ones(len(suffix_tokens), dtype=torch.bool),
        ],
        dim=0,
    )

    # Build the GT mrope for the unpruned input.
    image_offset = len(prefix_tokens) + 1  # First image token index.
    mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "image_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([1, h, w]),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="image",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=image_offset, length=num_tokens),
    )
    expected_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens,
        mm_features=[mm_feature],
        config=hf_config,
    )

    # Mirror `_postprocess_image_embeds_evs`: retained image embeddings carry
    # their original grid positions (plus a dummy fifth channel).
    image_retention_mask = retention_mask[1:-1]
    num_retained = int(image_retention_mask.sum().item())
    media_positions = compute_mrope_for_media(
        torch.tensor([1, h, w]), spatial_merge_size
    )
    expanded_positions = torch.zeros(
        (num_retained, 5),
        dtype=torch.long,
    )
    expanded_positions[:, :4] = media_positions[image_retention_mask]

    hidden_size = 16
    image_embeddings = torch.empty(
        (num_retained, hidden_size),
    )
    image_embeddings = torch.cat([image_embeddings, expanded_positions.float()], dim=1)
    multimodal_embeddings = [image_embeddings]

    expected_mrope_masked = expected_mrope[:, whole_sequence_retention_mask]

    # Number of tokens computed before this prefill chunk. `+1` simulates a
    # chunk boundary right past the vision_start token, which exercises the
    # media-start fallback in `recompute_mrope_positions`.
    num_computed_tokens = len(prefix_tokens) + num_computed_extra_tokens

    # Placeholder mrope for the pruned sequence, as produced by
    # `_get_mrope_input_positions` in the real pipeline.
    pruned_mm_feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            {
                "image_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([1, h, w]),
                    field=None,  # HACK.
                ),
            }
        ),
        modality="image",
        identifier="DUMMY",
        mm_position=PlaceholderRange(offset=image_offset, length=num_retained),
    )
    computed_mrope, _ = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=input_tokens_pruned,
        mm_features=[pruned_mm_feature],
        config=hf_config,
    )
    if image_pruning_rate > 0.0:
        # Paranoia check that the placeholder mrope is wrong.
        assert not torch.equal(computed_mrope, expected_mrope_masked)

    _, actual_mrope, _ = Qwen3VLForConditionalGeneration._recompute_mrope_positions(
        input_ids=input_tokens_pruned,
        multimodal_embeddings=multimodal_embeddings,
        mrope_positions=computed_mrope,
        num_computed_tokens=num_computed_tokens,
        vision_start_token_id=hf_config.vision_start_token_id,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
    )

    # Prefix and media positions must match the masked unpruned mrope
    # exactly. Trailing text (vision_end + suffix) instead continues right
    # after the last retained media position: unlike the video path, image
    # mm positions are raw grid coords, so no gap is left for pruned tokens.
    trailing_len = 1 + num_suffix_tokens  # vision_end + suffix text
    media_end = len(input_tokens_pruned) - trailing_len
    assert torch.equal(
        actual_mrope[:, :media_end], expected_mrope_masked[:, :media_end]
    )
    last_media_max = int(expected_mrope_masked[:, media_end - 1].max())
    expected_trailing = (
        torch.arange(trailing_len).unsqueeze(0) + last_media_max + 1
    ).expand(3, -1)
    assert torch.equal(actual_mrope[:, media_end:], expected_trailing)
