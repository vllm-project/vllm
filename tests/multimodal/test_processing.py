# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from contextlib import nullcontext

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.context import (
    InputProcessingContext,
    overlay_modality_mm_kwargs,
)
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    PlaceholderFeaturesInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptReplacement,
    _apply_matches,
    _apply_token_matches_with_placeholders,
    apply_token_matches,
    find_mm_placeholders,
    iter_token_matches,
    replace_token_matches,
)
from vllm.utils.collection_utils import flatten_2d_lists

from .utils import random_image

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("token_ids", "match_ids", "expected"),
    [
        ([], [], []),
        ([], [32000], []),
        (
            [32000, 32000, 32000],
            [32000],
            [
                {"start_idx": 0, "end_idx": 1},
                {"start_idx": 1, "end_idx": 2},
                {"start_idx": 2, "end_idx": 3},
            ],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000],
            [{"start_idx": 0, "end_idx": 2}],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000, 32000],
            [{"start_idx": 0, "end_idx": 3}],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000],
            [
                {"start_idx": 1, "end_idx": 3},
                {"start_idx": 6, "end_idx": 8},
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000, 32000, 32000],
            [
                {"start_idx": 1, "end_idx": 5},
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 0, 32000],
            [],
        ),
    ],
)
@pytest.mark.parametrize("start_idx", [0, 4, 8])
def test_iter_token_matches(token_ids, match_ids, expected, start_idx):
    result = list(iter_token_matches(token_ids, match_ids, start_idx=start_idx))

    # Manually constructed results
    assert [item._asdict() for item in result] == [
        item for item in expected if item["start_idx"] >= start_idx
    ]

    # Invariants
    match_lens = [end - start for start, end in result]
    print("match_lens:", match_lens)  # Only displayed on error
    assert all(match_len == len(match_ids) for match_len in match_lens)


@pytest.mark.parametrize(
    ("token_ids", "match_ids", "new_ids", "expected"),
    [
        ([], [], [-1], []),
        ([], [32000], [-1], []),
        (
            [32000, 32000, 32000],
            [32000],
            [-1],
            [-1, -1, -1],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000],
            [-1],
            [-1, 32000],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000, 32000],
            [-1],
            [-1],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000],
            [-1],
            [9833, -1, 32000, 32000, 9833, -1, 32000, 918],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000, 32000, 32000],
            [-1],
            [9833, -1, 9833, 28747, 32000, 32000, 918],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 0, 32000],
            [-1],
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
        ),
    ],
)
def test_replace_token_matches(token_ids, match_ids, new_ids, expected):
    result = replace_token_matches(token_ids, match_ids, new_ids)

    # Manually constructed results
    assert result == expected


@pytest.mark.parametrize(
    ("prompt", "target_by_key", "expected_by_key"),
    [
        (
            [],
            {
                "pattern_1": [],
                "pattern_2": [32000],
                "pattern_3": PromptIndexTargets.start(),
                "pattern_4": PromptIndexTargets.prefix([32000]),
                "pattern_5": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [],
                "pattern_2": [],
                "pattern_3": [
                    {"start_idx": 0, "end_idx": 0},
                ],
                "pattern_4": [],
                "pattern_5": [
                    {"start_idx": 0, "end_idx": 0},
                ],
            },
        ),
        (
            [32000, 32000, 32000, 32000],
            {
                "pattern_1": [32000],
                "pattern_2": [32000, 32000],
                "pattern_3": [32000, 32000, 32000],
                "pattern_4": PromptIndexTargets.start(),
                "pattern_5": PromptIndexTargets.prefix([32000]),
                "pattern_6": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [
                    {"start_idx": 0, "end_idx": 1},
                    {"start_idx": 1, "end_idx": 2},
                    {"start_idx": 2, "end_idx": 3},
                    {"start_idx": 3, "end_idx": 4},
                ],
                "pattern_2": [
                    {"start_idx": 0, "end_idx": 2},
                    {"start_idx": 2, "end_idx": 4},
                ],
                "pattern_3": [
                    {"start_idx": 0, "end_idx": 3},
                ],
                "pattern_4": [
                    {"start_idx": 0, "end_idx": 0},
                ],
                "pattern_5": [
                    {"start_idx": 1, "end_idx": 1},
                ],
                "pattern_6": [
                    {"start_idx": 4, "end_idx": 4},
                ],
            },
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            {
                "pattern_1": [28747, 32000],
                "pattern_2": [28747, 32000, 32000, 32000],
                "pattern_3": [28747, 0, 32000],
                "pattern_4": PromptIndexTargets.start(),
                "pattern_5": PromptIndexTargets.prefix([28747, 32000]),
                "pattern_6": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [
                    {"start_idx": 1, "end_idx": 3},
                    {"start_idx": 6, "end_idx": 8},
                ],
                "pattern_2": [
                    {"start_idx": 1, "end_idx": 5},
                ],
                "pattern_3": [],
                "pattern_4": [
                    {"start_idx": 0, "end_idx": 0},
                ],
                "pattern_5": [],
                "pattern_6": [
                    {"start_idx": 10, "end_idx": 10},
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
def test_find_token_matches(
    prompt,
    target_by_key,
    expected_by_key,
    update_type,
):
    prompt_updates = {
        key: update_type(key, target, []).resolve(0)
        for key, target in target_by_key.items()
    }
    result = {
        key: list(update.iter_token_matches(prompt))
        for key, update in prompt_updates.items()
    }

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert {
        key: [
            dict(start_idx=item.start_idx, end_idx=item.end_idx)
            for item in result.get(key, [])
        ]
        for key in expected_by_key
    } == expected_by_key


FIND_UPDATE_TOKENS_TEST_CASES = [
    # Tokenized test cases of `test_find_update_text`
    # using the vocab of llava-hf/llava-v1.6-mistral-7b-hf
    (
        [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918],
        {
            # We use `<image>` before `Image:` to test matches that
            # occur out of order
            "pattern_1": [32000],
            "pattern_2": [9833, 28747],
            "pattern_3": [918],
        },
        {
            # Test whether target is confused with replacement
            "pattern_1": [32000, 32000],
            # Test empty replacement
            "pattern_2": [],
            # Test dynamic replacement (beyond the form of `unit * count`)
            "pattern_3": [1550, 918, 1550],
        },
        {
            PromptInsertion: {
                0: [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918],
                1: [
                    1,
                    9833,
                    28747,
                    32000,
                    32000,
                    32000,
                    9833,
                    28747,
                    32000,
                    32000,
                    918,
                    1550,
                    918,
                    1550,
                ],  # noqa: E501
                2: [
                    1,
                    9833,
                    28747,
                    32000,
                    32000,
                    32000,
                    32000,
                    32000,
                    9833,
                    28747,
                    32000,
                    32000,
                    918,
                    1550,
                    918,
                    1550,
                    1550,
                    918,
                    1550,
                ],  # noqa: E501
            },
            PromptReplacement: {
                0: [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918],
                1: [1, 32000, 32000, 9833, 28747, 32000, 32000, 1550, 918, 1550],  # noqa: E501
                2: [1, 32000, 32000, 32000, 32000, 32000, 1550, 918, 1550],
            },
        },
    ),
    # Test index targets
    (
        [],
        {
            "pattern_1": PromptIndexTargets.start(),
            "pattern_2": PromptIndexTargets.prefix([32000]),
            "pattern_3": PromptIndexTargets.end(),
        },
        {
            "pattern_1": [-1],
            "pattern_2": [-2],
            "pattern_3": [-3],
        },
        {
            PromptInsertion: {
                0: [],
                1: [-1, -3],
                2: [-1, -1, -3, -3],
            },
            PromptReplacement: {
                0: [],
                1: [-1, -3],
                2: [-1, -1, -3, -3],
            },
        },
    ),
    (
        [32000],
        {
            "pattern_1": PromptIndexTargets.start(),
            "pattern_2": PromptIndexTargets.prefix([32000]),
            "pattern_3": PromptIndexTargets.end(),
        },
        {
            "pattern_1": [-1],
            "pattern_2": [-2],
            "pattern_3": [-3],
        },
        {
            PromptInsertion: {
                0: [32000],
                1: [-1, 32000, -2, -3],
                2: [-1, -1, 32000, -2, -2, -3, -3],
            },
            PromptReplacement: {
                0: [32000],
                1: [-1, 32000, -2, -3],
                2: [-1, -1, 32000, -2, -2, -3, -3],
            },
        },
    ),
    # Test different replacement per item
    (
        [32000, 32000, 32000],
        {
            "pattern_1": [32000],
        },
        {
            "pattern_1": lambda idx: [-(idx + 1)],
        },
        {
            PromptInsertion: {
                0: [32000, 32000, 32000],
                1: [32000, -1, 32000, 32000],
                2: [32000, -1, -2, 32000, 32000],
            },
            PromptReplacement: {
                0: [32000, 32000, 32000],
                1: [-1, 32000, 32000],
                2: [-1, -2, 32000],
            },
        },
    ),
    (
        [32000, 32000, 32000],
        {
            "pattern_1": PromptIndexTargets.prefix([32000]),
        },
        {
            "pattern_1": lambda idx: [-(idx + 1)],
        },
        {
            PromptInsertion: {
                0: [32000, 32000, 32000],
                1: [32000, -1, 32000, 32000],
                2: [32000, -1, -2, 32000, 32000],
            },
            PromptReplacement: {
                0: [32000, 32000, 32000],
                1: [32000, -1, 32000, 32000],
                2: [32000, -1, -2, 32000, 32000],
            },
        },
    ),
]


def _placeholder(modality, item_idx, start_idx, tokens):
    return PlaceholderFeaturesInfo(
        modality=modality,
        item_idx=item_idx,
        start_idx=start_idx,
        tokens=tokens,
        is_embed=None,
    )


FIND_UPDATE_TOKENS_PLACEHOLDER_EXPECTED = [
    {
        PromptInsertion: {
            0: {},
            1: {
                "pattern_1": [_placeholder("pattern_1", 0, 4, [32000, 32000])],
                "pattern_3": [_placeholder("pattern_3", 0, 11, [1550, 918, 1550])],
            },
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 4, [32000, 32000]),
                    _placeholder("pattern_1", 1, 6, [32000, 32000]),
                ],
                "pattern_3": [
                    _placeholder("pattern_3", 0, 13, [1550, 918, 1550]),
                    _placeholder("pattern_3", 1, 16, [1550, 918, 1550]),
                ],
            },
        },
        PromptReplacement: {
            0: {},
            1: {
                "pattern_1": [_placeholder("pattern_1", 0, 1, [32000, 32000])],
                "pattern_3": [_placeholder("pattern_3", 0, 7, [1550, 918, 1550])],
            },
            2: {},
        },
    },
    {
        PromptInsertion: {0: {}, 1: {}, 2: {}},
        PromptReplacement: {0: {}, 1: {}, 2: {}},
    },
    {
        PromptInsertion: {
            0: {},
            1: {
                "pattern_1": [_placeholder("pattern_1", 0, 0, [-1])],
                "pattern_2": [_placeholder("pattern_2", 0, 2, [-2])],
                "pattern_3": [_placeholder("pattern_3", 0, 3, [-3])],
            },
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 0, [-1]),
                    _placeholder("pattern_1", 1, 1, [-1]),
                ],
                "pattern_2": [
                    _placeholder("pattern_2", 0, 3, [-2]),
                    _placeholder("pattern_2", 1, 4, [-2]),
                ],
                "pattern_3": [
                    _placeholder("pattern_3", 0, 5, [-3]),
                    _placeholder("pattern_3", 1, 6, [-3]),
                ],
            },
        },
        PromptReplacement: {
            0: {},
            1: {
                "pattern_1": [_placeholder("pattern_1", 0, 0, [-1])],
                "pattern_2": [_placeholder("pattern_2", 0, 2, [-2])],
                "pattern_3": [_placeholder("pattern_3", 0, 3, [-3])],
            },
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 0, [-1]),
                    _placeholder("pattern_1", 1, 1, [-1]),
                ],
                "pattern_2": [
                    _placeholder("pattern_2", 0, 3, [-2]),
                    _placeholder("pattern_2", 1, 4, [-2]),
                ],
                "pattern_3": [
                    _placeholder("pattern_3", 0, 5, [-3]),
                    _placeholder("pattern_3", 1, 6, [-3]),
                ],
            },
        },
    },
    {
        PromptInsertion: {
            0: {},
            1: {"pattern_1": [_placeholder("pattern_1", 0, 1, [-1])]},
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 1, [-1]),
                    _placeholder("pattern_1", 1, 2, [-2]),
                ]
            },
        },
        PromptReplacement: {
            0: {},
            1: {"pattern_1": [_placeholder("pattern_1", 0, 0, [-1])]},
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 0, [-1]),
                    _placeholder("pattern_1", 1, 1, [-2]),
                ]
            },
        },
    },
    {
        PromptInsertion: {
            0: {},
            1: {"pattern_1": [_placeholder("pattern_1", 0, 1, [-1])]},
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 1, [-1]),
                    _placeholder("pattern_1", 1, 2, [-2]),
                ]
            },
        },
        PromptReplacement: {
            0: {},
            1: {"pattern_1": [_placeholder("pattern_1", 0, 1, [-1])]},
            2: {
                "pattern_1": [
                    _placeholder("pattern_1", 0, 1, [-1]),
                    _placeholder("pattern_1", 1, 2, [-2]),
                ]
            },
        },
    },
]


@pytest.mark.parametrize(
    ("prompt", "target_by_key", "repl_by_key", "expected_by_update_type_mm_count"),  # noqa: E501
    FIND_UPDATE_TOKENS_TEST_CASES,
)
def test_find_update_tokens(
    prompt,
    target_by_key,
    repl_by_key,
    expected_by_update_type_mm_count,
):
    for (
        update_type,
        expected_by_mm_count,
    ) in expected_by_update_type_mm_count.items():
        for mm_count, expected in expected_by_mm_count.items():
            mm_prompt_updates = {
                key: [
                    [update_type(key, target, repl_by_key[key]).resolve(i)]
                    for i in range(mm_count)
                ]
                for key, target in target_by_key.items()
            }

            new_prompt, result = apply_token_matches(prompt, mm_prompt_updates)

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_prompt_updates:", mm_prompt_updates)
            print("new_prompt:", new_prompt)
            print("result:", result)

            # Manually constructed results
            assert new_prompt == expected


@pytest.mark.parametrize(
    (
        "prompt",
        "target_by_key",
        "repl_by_key",
        "expected_by_update_type_mm_count",
        "expected_placeholders_by_update_type_mm_count",
    ),
    [
        (*case, placeholder_expected)
        for case, placeholder_expected in zip(
            FIND_UPDATE_TOKENS_TEST_CASES,
            FIND_UPDATE_TOKENS_PLACEHOLDER_EXPECTED,
            strict=True,
        )
    ],
)
def test_apply_token_matches_with_placeholders(
    prompt,
    target_by_key,
    repl_by_key,
    expected_by_update_type_mm_count,
    expected_placeholders_by_update_type_mm_count,
):
    for update_type, expected_by_mm_count in expected_by_update_type_mm_count.items():
        for mm_count, expected in expected_by_mm_count.items():
            mm_prompt_updates = {
                key: [
                    [update_type(key, target, repl_by_key[key]).resolve(i)]
                    for i in range(mm_count)
                ]
                for key, target in target_by_key.items()
            }

            new_prompt, result, placeholders = _apply_token_matches_with_placeholders(
                prompt,
                mm_prompt_updates,
            )

            if any(
                update_idx is None
                for update_idxs in result.values()
                for update_idx in update_idxs
            ):
                continue

            expected_placeholders = expected_placeholders_by_update_type_mm_count[
                update_type
            ][mm_count]

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_prompt_updates:", mm_prompt_updates)
            print("new_prompt:", new_prompt)
            print("result:", result)
            print("placeholders:", placeholders)

            assert new_prompt == expected
            assert {
                modality: ph_list
                for modality, ph_list in placeholders.items()
                if ph_list
            } == expected_placeholders


@pytest.mark.parametrize(
    "repl_by_key",
    [
        {
            "pattern_1": [32000, 32000],
            "pattern_2": [],
            "pattern_3": [1550, 918, 1550],
            # Test different modalities having the same tokens (32000)
            "pattern_4": [32000],
        },
    ],
)
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918],
            {
                "pattern_1": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=0,
                        start_idx=6,
                        tokens=[32000, 32000],
                        is_embed=None,
                    ),
                ],
                "pattern_4": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_4",
                        item_idx=0,
                        start_idx=3,
                        tokens=[32000],
                        is_embed=None,
                    ),
                ],
            },
        ),
        (
            [1, 32000, 32000, 9833, 28747, 32000, 32000, 1550, 918, 1550],
            {
                "pattern_1": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=0,
                        start_idx=1,
                        tokens=[32000, 32000],
                        is_embed=None,
                    ),
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=1,
                        start_idx=5,
                        tokens=[32000, 32000],
                        is_embed=None,
                    ),
                ],
                "pattern_3": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_3",
                        item_idx=0,
                        start_idx=7,
                        tokens=[1550, 918, 1550],
                        is_embed=None,
                    ),
                ],
                # No match for pattern_4 as it has lower priority than pattern_1
            },
        ),
        (
            [1, 32000, 32000, 32000, 32000, 32000, 1550, 918, 1550],
            {
                "pattern_1": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=0,
                        start_idx=1,
                        tokens=[32000, 32000],
                        is_embed=None,
                    ),
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=1,
                        start_idx=3,
                        tokens=[32000, 32000],
                        is_embed=None,
                    ),
                ],
                "pattern_4": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_4",
                        item_idx=0,
                        start_idx=5,
                        tokens=[32000],
                        is_embed=None,
                    ),
                ],
                "pattern_3": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_3",
                        item_idx=0,
                        start_idx=6,
                        tokens=[1550, 918, 1550],
                        is_embed=None,
                    ),
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
def test_find_mm_placeholders(
    repl_by_key,
    prompt,
    expected,
    update_type,
):
    mm_prompt_updates = {
        key: [[update_type(key, [], repl).resolve(i)] for i in range(3)]
        for key, repl in repl_by_key.items()
    }

    result = find_mm_placeholders(prompt, mm_prompt_updates)

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert result == expected


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("num_images", "limit", "is_valid"),
    [
        (0, 0, True),
        (0, 1, True),
        (1, 0, False),
        (1, 1, True),
        (1, 2, True),
        (2, 1, False),
        (2, 2, True),
    ],
)
def test_limit_mm_per_prompt_apply(model_id, num_images, limit, is_valid):
    limit_mm_per_prompt = {"image": limit}

    model_config = ModelConfig(
        model=model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    rng = np.random.RandomState(0)
    image = random_image(rng, min_wh=128, max_wh=256)
    if num_images == 0:
        mm_data = {}
    elif num_images == 1:
        mm_data = {"image": image}
    else:
        mm_data = {"image": [image] * num_images}

    exc_ctx = (
        nullcontext()
        if is_valid
        else pytest.raises(VLLMValidationError, match="At most")
    )

    with exc_ctx:
        processor(
            "<image>" * num_images,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("user_limit", "supported_limit"),
    [
        (0, 0),
        (0, 1),
        (1, 0),  # user wants 1, model supports 0 → capped to 0
        (1, 1),
        (1, 2),
        (2, 1),  # user wants 2, model supports 1 → capped to 1
        (2, 2),
        (5, 1),  # large user limit, low model support → capped to 1
        (1, 5),
        (10, 0),  # large user limit, no model support → capped to 0
    ],
)
def test_budget_caps_prevent_dummy_input_validation_failure(
    model_id, user_limit, supported_limit
):
    limit_mm_per_prompt = {"image": user_limit}

    model_config = ModelConfig(
        model=model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    processor.info.get_supported_mm_limits = lambda: {"image": supported_limit}

    # This is what budget.py uses to derive mm_counts
    allowed = processor.info.allowed_mm_limits

    assert allowed["image"] <= supported_limit, (
        f"allowed_mm_limits['image']={allowed['image']} exceeds "
        f"supported_limit={supported_limit}"
    )

    assert allowed["image"] <= user_limit, (
        f"allowed_mm_limits['image']={allowed['image']} exceeds user_limit={user_limit}"
    )

    assert allowed["image"] == min(user_limit, supported_limit)


class DummyProcessor:
    def __init__(self, a: int = 0, b: int = 0) -> None:
        super().__init__()

        self.a = a
        self.b = b

    def __call__(
        self,
        a: int = 0,
        c: int = 0,
        return_tensors: str | None = None,
    ) -> dict[str, int]:
        return dict(a=a, c=c)


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])  # Dummy
@pytest.mark.parametrize(
    ("config_kwargs", "inference_kwargs", "expected_kwargs"),
    [
        ({"a": 1}, {}, {"a": 1, "b": 0}),
        ({}, {"a": 1}, {"a": 1, "b": 0}),
        # inference_kwargs should take precedence
        ({"a": 1}, {"a": 2}, {"a": 2, "b": 0}),
        # Should ignore extra kwargs
        ({"a": 1, "c": 1}, {}, {"a": 1, "b": 0}),
        ({"b": 1, "c": 1}, {}, {"a": 0, "b": 1}),
    ],
)
def test_hf_processor_init_kwargs(
    model_id,
    config_kwargs,
    inference_kwargs,
    expected_kwargs,
):
    ctx = InputProcessingContext(
        model_config=ModelConfig(model_id, mm_processor_kwargs=config_kwargs),
        tokenizer=None,
    )

    processor = ctx.get_hf_processor(
        DummyProcessor,  # type: ignore[arg-type]
        **inference_kwargs,
    )
    assert processor.a == expected_kwargs["a"]
    assert processor.b == expected_kwargs["b"]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])  # Dummy
@pytest.mark.parametrize(
    ("config_kwargs", "inference_kwargs", "expected_kwargs"),
    [
        ({"a": 1}, {}, {"a": 1, "c": 0}),
        ({}, {"a": 1}, {"a": 1, "c": 0}),
        # inference_kwargs should take precedence
        ({"a": 1}, {"a": 2}, {"a": 2, "c": 0}),
        # Should ignore extra kwargs
        ({"a": 1, "c": 1}, {}, {"a": 1, "c": 1}),
        ({"b": 1, "c": 1}, {}, {"a": 0, "c": 1}),
    ],
)
def test_hf_processor_call_kwargs(
    model_id,
    config_kwargs,
    inference_kwargs,
    expected_kwargs,
):
    ctx = InputProcessingContext(
        model_config=ModelConfig(model_id, mm_processor_kwargs=config_kwargs),
        tokenizer=None,
    )

    processor = ctx.get_hf_processor(DummyProcessor)  # type: ignore[arg-type]

    result = ctx.call_hf_processor(processor, {}, inference_kwargs)
    assert result == expected_kwargs


def test_apply_matches_no_match_exits_quickly():
    """
    Test that _apply_matches exits quickly when no matches are found.

    Previously, _apply_matches had O(n²) behavior when no match was found
    because it would increment start_idx by 1 each iteration while
    re-scanning the entire prompt from prev_end_idx=0.

    With the fix, it should exit immediately when no match is found.
    """
    # Create a long prompt with no placeholder
    long_prompt = [1] * 10000

    # Create update looking for a placeholder that doesn't exist
    mm_prompt_updates = {"image": [[PromptReplacement("image", [0], [-1]).resolve(0)]]}

    start = time.perf_counter()
    result, _ = _apply_matches(long_prompt, mm_prompt_updates)
    elapsed = time.perf_counter() - start

    # Should complete in < 100ms (was taking seconds before the fix)
    assert elapsed < 0.1, f"_apply_matches took {elapsed:.2f}s, expected < 0.1s"
    assert flatten_2d_lists(result) == long_prompt


def test_apply_matches_many_shared_targets_scales_linearly():
    """Shared replacement targets must not trigger per-item rescanning."""
    replacement = [1] * 50
    update = PromptReplacement("image", [0], replacement)

    def measure(item_count: int) -> float:
        mm_prompt_updates = {
            "image": [[update.resolve(item_idx)] for item_idx in range(item_count)]
        }
        prompt = [0] * item_count

        start = time.perf_counter()
        result, match_result = apply_token_matches(prompt, mm_prompt_updates)
        elapsed = time.perf_counter() - start

        assert len(result) == item_count * len(replacement)
        assert all(token_id == 1 for token_id in result)
        assert match_result == {"image": [0] * item_count}

        return elapsed

    measure(100)
    small_time = measure(1_000)
    large_time = measure(4_000)

    time_ratio = large_time / small_time
    assert time_ratio < 8, f"Expected linear scaling, got {time_ratio:.1f}x"


def test_iter_token_matches_rejects_negative_start_idx():
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_token_matches([1, 2, 3], [2], start_idx=-1))


def test_find_mm_placeholders_avoids_quadratic_false_prefixes():
    """
    Test that placeholder scanning stays linear under adversarial candidates.

    The fast-forward scan must not rescan the prompt tail per position when
    one candidate's first token never occurs (forcing a full search) while
    another's occurs at every position (forcing single-step advances).
    """
    prompt = [1] * 30_000
    mm_prompt_updates = {
        "absent": [[PromptReplacement("absent", [0], [999, 0]).resolve(0)]],
        "frequent_false_prefix": [
            [PromptReplacement("frequent_false_prefix", [0], [1, 2]).resolve(0)]
        ],
    }

    start = time.perf_counter()
    result = find_mm_placeholders(prompt, mm_prompt_updates)
    elapsed = time.perf_counter() - start

    assert result == {}
    assert elapsed < 0.5, f"find_mm_placeholders took {elapsed:.2f}s, expected < 0.5s"


@pytest.mark.parametrize(
    "prompt",
    [
        # Empty prompt: the scan loop is never entered
        [],
        # Non-empty prompt: the scan runs but never finds the first item,
        # so the second item must stay unresolved
        [1, 2, 3, 4, 5],
    ],
)
def test_find_mm_placeholders_stops_at_missing_item(prompt):
    """
    Test that the scan returns no placeholders once it fails to find
    an item's placeholder, leaving later items unresolved.
    """
    result = find_mm_placeholders(
        prompt,
        {
            "image": [
                [PromptReplacement("image", [0], [999]).resolve(0)],
                [PromptReplacement("image", [0], [998]).resolve(1)],
            ]
        },
    )

    assert result == {}


class _FakeTokenizer:
    """
    Character-level tokenizer where "foo" merges into one token differently
    depending on whether it is followed by "d", like BPE merging "foo" in
    "food" across the search-text boundary.
    """

    _MERGES = {"food": (1000,), "foo": (101, 111, 111)}
    _INVERSE = {ids: text for text, ids in _MERGES.items()}

    def encode(self, text: str, **kwargs) -> list[int]:
        token_ids = list[int]()
        pos = 0
        while pos < len(text):
            for length in (4, 3):
                word = text[pos : pos + length]
                if word in self._MERGES:
                    token_ids.extend(self._MERGES[word])
                    pos += length
                    break
            else:
                token_ids.append(ord(text[pos]))
                pos += 1
        return token_ids

    def decode(self, token_ids: list[int], **kwargs) -> str:
        chars = list[str]()
        pos = 0
        while pos < len(token_ids):
            for length in (3, 1):
                key = tuple(token_ids[pos : pos + length])
                if key in self._INVERSE:
                    chars.append(self._INVERSE[key])
                    pos += length
                    break
            else:
                chars.append(chr(token_ids[pos]))
                pos += 1
        return "".join(chars)


class _FakeProcessingInfo:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        return self._tokenizer


class _TextFallbackProcessor(BaseMultiModalProcessor):
    """Only `self.info.get_tokenizer()` is needed by the text fallback."""

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.info = _FakeProcessingInfo(tokenizer)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        raise NotImplementedError

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        raise NotImplementedError


def _text_fallback_processor() -> BaseMultiModalProcessor:
    return _TextFallbackProcessor(_FakeTokenizer())


def test_apply_prompt_updates_falls_back_to_text_matching():
    """
    Test that the fallback in `_apply_prompt_updates` finds targets that
    tokenize differently inside the prompt ("foo" in "food").
    """
    processor = _text_fallback_processor()

    new_token_ids, placeholders = processor._apply_prompt_updates(
        [1000],  # "food"
        {
            "image": [
                [PromptReplacement("image", [101, 111, 111], [200, 201]).resolve(0)]
            ]
        },
    )

    assert new_token_ids == [200, 201, ord("d")]
    assert [p.to_range().offset for p in placeholders["image"]] == [0]
    assert [p.tokens for p in placeholders["image"]] == [[200, 201]]


def test_apply_prompt_updates_falls_back_with_prefix_target():
    """
    Test that `PromptIndexTargets.prefix` targets are resolved against the
    decoded text in the fallback path of `_apply_prompt_updates`.
    """
    processor = _text_fallback_processor()

    new_token_ids, placeholders = processor._apply_prompt_updates(
        [1000],  # "food"
        {
            "image": [
                [
                    PromptInsertion(
                        "image",
                        PromptIndexTargets.prefix([101, 111, 111]),
                        [9],
                    ).resolve(0)
                ]
            ]
        },
    )

    assert new_token_ids == [101, 111, 111, 9, ord("d")]
    assert [p.tokens for p in placeholders["image"]] == [[9]]


def test_apply_prompt_updates_falls_back_with_index_targets():
    """
    Test that the text resolvers of `PromptIndexTargets.start`/`end`
    match against the decoded text when another item forces the
    fallback in `_apply_prompt_updates`.
    """
    processor = _text_fallback_processor()

    new_token_ids, placeholders = processor._apply_prompt_updates(
        [1000],  # "food"
        {
            "image": [
                [PromptReplacement("image", [101, 111, 111], [200, 201]).resolve(0)],
                [PromptInsertion("image", PromptIndexTargets.end(), [9]).resolve(1)],
            ]
        },
    )

    assert new_token_ids == [200, 201, ord("d"), 9]
    assert [p.tokens for p in placeholders["image"]] == [[200, 201], [9]]


@pytest.mark.skip_global_cleanup
def test_overlay_modality_mm_kwargs_scoped_video_does_not_leak_to_image():
    """HF-style videos_kwargs must overlay only when modality is video."""
    video_size = {"longest_edge": 469762048, "shortest_edge": 4096}
    kwargs = {"videos_kwargs": {"size": video_size}}

    assert overlay_modality_mm_kwargs(kwargs, None) == kwargs
    assert "size" not in overlay_modality_mm_kwargs(kwargs, "image")
    assert overlay_modality_mm_kwargs(kwargs, "video")["size"] == video_size


@pytest.mark.skip_global_cleanup
def test_overlay_modality_mm_kwargs_flat_size_stays_shared():
    """A flat size override keeps the current shared-namespace behavior."""
    size = {"longest_edge": 469762048, "shortest_edge": 4096}
    kwargs = {"size": size}

    for modality in (None, "image", "video"):
        assert overlay_modality_mm_kwargs(kwargs, modality)["size"] == size


@pytest.mark.skip_global_cleanup
def test_overlay_modality_mm_kwargs_scoped_wins_over_flat_for_modality():
    """A nested videos_kwargs size wins over a flat size for video reads."""
    kwargs = {
        "size": {"longest_edge": 1},
        "videos_kwargs": {"size": {"longest_edge": 2}},
        "images_kwargs": {"size": {"longest_edge": 3}},
    }

    assert overlay_modality_mm_kwargs(kwargs, "video")["size"] == {"longest_edge": 2}
    assert overlay_modality_mm_kwargs(kwargs, "image")["size"] == {"longest_edge": 3}
    assert overlay_modality_mm_kwargs(kwargs, None)["size"] == {"longest_edge": 1}


@pytest.mark.skip_global_cleanup
def test_overlay_modality_mm_kwargs_ignores_non_mapping_scoped_value():
    kwargs = {"images_kwargs": "not-a-dict", "size": {"longest_edge": 1}}
    assert overlay_modality_mm_kwargs(kwargs, "image")["size"] == {"longest_edge": 1}


@pytest.mark.skip_global_cleanup
def test_mm_processor_kwargs_merge_then_overlay_preserves_scoping():
    """Configured videos_kwargs overlay only for video reads after merge."""
    from vllm.config.multimodal import MultiModalConfig

    size = {"longest_edge": 469762048, "shortest_edge": 4096}
    mm_config = MultiModalConfig(mm_processor_kwargs={"videos_kwargs": {"size": size}})
    merged = mm_config.merge_mm_processor_kwargs({})
    assert overlay_modality_mm_kwargs(merged, "video")["size"] == size
    assert "size" not in overlay_modality_mm_kwargs(merged, "image")
    assert "size" not in overlay_modality_mm_kwargs(merged, None)


def test_processor_inputs_hashes_partial_uuids():
    rng = np.random.RandomState(0)
    images = [random_image(rng, min_wh=8, max_wh=9) for _ in range(2)]
    inputs = ProcessorInputs(
        prompt=[],
        mm_data_items=MultiModalDataParser().parse_mm_data({"image": images}),
        mm_uuid_items={"image": ["image-uuid", None]},
    )

    assert inputs.get_mm_hashes("test-model", "blake3") == {
        "image": [
            "image-uuid",
            MultiModalHasher.hash_kwargs(
                "blake3", model_id="test-model", image=images[1]
            ),
        ]
    }
