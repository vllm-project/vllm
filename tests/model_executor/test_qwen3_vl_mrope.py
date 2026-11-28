# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import random
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.video_prune import evs, vidcom2


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


@dataclass(slots=True)
class DummyVisionConfig:
    spatial_merge_size: int = 1


@dataclass(slots=True)
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

    pruning_mask = torch.bernoulli(video_token_mask.float() * video_pruning_rate).bool()
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


TEXT_TOKEN_ID = 1


def _build_prompt(
    layout: list[str],
    video_grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    num_retained: int | None,
):
    """Build `TEXT <media> TEXT ... <media> TEXT` as the Qwen3-VL processor does.

    Per frame, a video has timestamp tokens, `VISION_START`, its video tokens and
    `VISION_END`. With `num_retained`, each video keeps that many tokens, all in
    its first frame, as with video pruning.

    Returns the tokens, their features, their positions in the unpruned layout
    (-1 inside a pruned video, where they are only known after encoding) and the
    position of the first generated token.
    """
    tokens: list[int] = []
    mm_features = []
    expected: list[list[int]] = []
    pos = 0

    def add_text(*token_ids: int) -> None:
        nonlocal pos
        tokens.extend(token_ids)
        expected.extend([p] * 3 for p in range(pos, pos + len(token_ids)))
        pos += len(token_ids)

    def add_grid(token_id: int, grid_h: int, grid_w: int) -> None:
        nonlocal pos
        tokens.extend([token_id] * grid_h * grid_w)
        expected.extend(
            [pos, pos + i, pos + j] for i in range(grid_h) for j in range(grid_w)
        )
        pos += max(grid_h, grid_w)

    add_text(*[TEXT_TOKEN_ID] * 3)
    for modality in layout:
        grid_thw = video_grid_thw if modality == "video" else (1, 8, 12)
        t, grid_h, grid_w = grid_thw
        grid_h //= spatial_merge_size
        grid_w //= spatial_merge_size
        if modality == "image":
            add_text(VISION_START_TOKEN_ID)
            offset = len(tokens)
            add_grid(IMAGE_TOKEN_ID, grid_h, grid_w)
            length = len(tokens) - offset
            add_text(VISION_END_TOKEN_ID)
        else:
            offset = len(tokens)
            for frame in range(t):
                add_text(*[TEXT_TOKEN_ID] * (2 + frame % 3), VISION_START_TOKEN_ID)
                if num_retained is None:
                    add_grid(VIDEO_TOKEN_ID, grid_h, grid_w)
                else:
                    if frame == 0:
                        tokens.extend([VIDEO_TOKEN_ID] * num_retained)
                    pos += max(grid_h, grid_w)
                add_text(VISION_END_TOKEN_ID)
            length = len(tokens) - offset
            if num_retained is not None:
                expected[offset:] = [[-1] * 3] * length
        mm_features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem(
                    {
                        f"{modality}_grid_thw": MultiModalFieldElem(
                            data=torch.tensor(grid_thw),
                            field=None,  # HACK.
                        ),
                    }
                ),
                modality=modality,
                identifier="DUMMY",
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        )
        add_text(*[TEXT_TOKEN_ID] * 2)
    add_text(*[TEXT_TOKEN_ID] * 4)
    return tokens, mm_features, torch.tensor(expected).T, pos


def _num_retained(
    pruning: tuple[str, float] | None, grid_thw: tuple[int, int, int], merge: int
) -> int | None:
    if pruning is None:
        return None
    method, q = pruning
    t, h, w = grid_thw
    module = vidcom2 if method == "vidcom2" else evs
    return module.compute_retained_tokens_count(
        tokens_per_frame=(h // merge) * (w // merge), num_frames=t, q=q
    )


@pytest.mark.parametrize(
    ("video_grid_thw", "spatial_merge_size"),
    [((12, 34, 60), 2), ((4, 6, 4), 1), ((5, 6, 10), 1), ((1, 4, 5), 1)],
    ids=["12x34x60", "4x6x4", "5x6x10", "1x4x5"],
)
@pytest.mark.parametrize(
    "pruning",
    # VidCom2 at 0.95 keeps fewer tokens than one frame has.
    [None, ("evs", 0.3), ("evs", 0.5), ("vidcom2", 0.75), ("vidcom2", 0.95)],
    ids=["unpruned", "evs0.3", "evs0.5", "vidcom2-0.75", "vidcom2-0.95"],
)
@pytest.mark.parametrize(
    "layout",
    [["video"], ["video", "video"], ["image", "video", "image"]],
    ids="+".join,
)
def test_get_mrope_input_positions_pruned_video(
    video_grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    pruning: tuple[str, float] | None,
    layout: list[str],
):
    """A pruned video advances the positions as much as the unpruned video does.

    The tokens after a video are not recomputed when the video is a prefix cache
    hit, so they must already have the positions of the unpruned layout.
    """
    hf_config = DummyConfig(vision_config=DummyVisionConfig(spatial_merge_size))
    num_retained = _num_retained(pruning, video_grid_thw, spatial_merge_size)
    tokens, mm_features, expected, next_pos = _build_prompt(
        layout, video_grid_thw, spatial_merge_size, num_retained
    )

    positions, delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=tokens, mm_features=mm_features, config=hf_config
    )

    assert positions.shape == (3, len(tokens))
    known = expected[0] >= 0
    torch.testing.assert_close(positions[:, known], expected[:, known], rtol=0, atol=0)
    assert len(tokens) + delta == next_pos


@pytest.mark.parametrize(
    ("video_grid_thw", "spatial_merge_size", "pruning"),
    [((12, 34, 60), 2, ("evs", 0.3)), ((5, 6, 10), 1, ("vidcom2", 0.95))],
    ids=["12x34x60-evs0.3", "5x6x10-vidcom2-0.95"],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_pruned_video_prefix_cache_hit_matches_recompute(
    video_grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    pruning: tuple[str, float],
    seed: int,
):
    """Whichever tokens are kept, the positions after a pruned video are the ones
    that `_recompute_mrope_positions` gives them when the video is encoded.
    """
    hf_config = DummyConfig(vision_config=DummyVisionConfig(spatial_merge_size))
    num_retained = _num_retained(pruning, video_grid_thw, spatial_merge_size)
    tokens, mm_features, _, _ = _build_prompt(
        ["video"], video_grid_thw, spatial_merge_size, num_retained
    )
    unpruned_tokens, unpruned_features, unpruned_positions, _ = _build_prompt(
        ["video"], video_grid_thw, spatial_merge_size, None
    )

    # The positions that `_get_expanded_positions` appends to the embeddings:
    # those of the unpruned video, for the tokens that are kept.
    start = unpruned_features[0].mm_position.offset
    end = start + unpruned_features[0].mm_position.length
    video_tokens = torch.tensor(unpruned_tokens[start:end])
    is_video = video_tokens == VIDEO_TOKEN_ID
    video_token_indices = is_video.nonzero().squeeze(1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(video_token_indices), generator=generator)
    kept = torch.ones(len(video_tokens), dtype=torch.bool)
    kept[video_token_indices[order[num_retained:]]] = False
    expanded_positions = torch.cat(
        [
            unpruned_positions[:, start:end][:, kept] - unpruned_positions[0, start],
            (video_tokens[kept] == VISION_START_TOKEN_ID).long()[None],
            is_video[kept].long()[None],
        ]
    )
    assert expanded_positions.shape[1] == mm_features[0].mm_position.length
    embeddings = torch.cat(
        [torch.zeros(expanded_positions.shape[1], 4), expanded_positions.T.float()],
        dim=1,
    )

    positions, delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        input_tokens=tokens, mm_features=mm_features, config=hf_config
    )
    _, recomputed, recomputed_delta = (
        Qwen3VLForConditionalGeneration._recompute_mrope_positions(
            input_ids=torch.tensor(tokens),
            multimodal_embeddings=[embeddings],
            mrope_positions=positions.clone(),
            num_computed_tokens=0,
            vision_start_token_id=hf_config.vision_start_token_id,
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
        )
    )

    video_end = mm_features[0].mm_position.offset + mm_features[0].mm_position.length
    torch.testing.assert_close(
        positions[:, video_end:], recomputed[:, video_end:], rtol=0, atol=0
    )
    assert delta == recomputed_delta
