# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig, MultiModalConfig, SchedulerConfig
from vllm.exceptions import VLLMUnprocessableEntityError, VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.media import MediaRef
from vllm.multimodal.parse import MultiModalDataParser, ProcessorBatchItems
from vllm.multimodal.processing.context import (
    InputProcessingContext,
    TimingContext,
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

from ..models.utils import build_model_context
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
    """Test that _apply_matches exits quickly when no matches are found.

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
    """Test that placeholder scanning stays linear under adversarial candidates.

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
    """Test that the scan returns no placeholders once it fails to find
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
    """Character-level tokenizer where "foo" merges into one token differently
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
    """Test that the fallback in `_apply_prompt_updates` finds targets that
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
    """Test that `PromptIndexTargets.prefix` targets are resolved against the
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
    """Test that the text resolvers of `PromptIndexTargets.start`/`end`
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


def test_processor_inputs_hashes_scope_kwargs_by_modality():
    """Changing one modality's options must not invalidate another item."""
    rng = np.random.RandomState(0)
    mm_data_items = MultiModalDataParser().parse_mm_data(
        {
            "image": [random_image(rng, min_wh=8, max_wh=9)],
            "video": [np.zeros((2, 8, 8, 3), dtype=np.uint8)],
        }
    )
    mm_uuid_items = {"image": ["image-uuid"], "video": ["video-uuid"]}

    def get_hashes(video_frames: int, image_size: int, video_size: int):
        return ProcessorInputs(
            prompt=[],
            mm_data_items=mm_data_items,
            mm_uuid_items=mm_uuid_items,
            media_io_kwargs={"video": {"num_frames": video_frames}},
            hf_processor_mm_kwargs={
                "images_kwargs": {"size": {"longest_edge": image_size}},
                "videos_kwargs": {"size": {"longest_edge": video_size}},
            },
        ).get_mm_hashes("test-model", "blake3")

    base = get_hashes(video_frames=4, image_size=224, video_size=224)
    changed_video = get_hashes(video_frames=16, image_size=224, video_size=448)
    changed_image = get_hashes(video_frames=4, image_size=448, video_size=224)

    assert changed_video["image"] == base["image"]
    assert changed_video["video"] != base["video"]
    assert changed_image["image"] != base["image"]
    assert changed_image["video"] == base["video"]


def test_processor_inputs_hashes_ignore_unrelated_kwargs():
    """An image-only request ignores video-only processing configuration."""
    image = random_image(np.random.RandomState(0), min_wh=8, max_wh=9)
    inputs = ProcessorInputs(
        prompt=[],
        mm_data_items=MultiModalDataParser().parse_mm_data({"image": [image]}),
        mm_uuid_items={"image": ["image-uuid"]},
        media_io_kwargs={"video": {"num_frames": 16}},
        hf_processor_mm_kwargs={"videos_kwargs": {"size": {"longest_edge": 448}}},
    )

    assert inputs.get_mm_hashes("test-model", "blake3") == {"image": ["image-uuid"]}


def test_processor_inputs_hashes_ref_spec_replaces_media_io_kwargs():
    """A ref's decode spec lives inside its key, so `media_io_kwargs` is not
    hashed a second time for it. A client-supplied UUID replaces the item
    entirely, so there the factor must stay -- the key is never consulted."""
    from vllm.multimodal.media import ImageMediaIO

    data = b"encoded-image-bytes"
    keep_ref = ImageMediaIO(image_mode=None).load_bytes_ref(data)
    rgb_ref = ImageMediaIO().load_bytes_ref(data)

    def hash_of(ref, uuid_item, media_io_kwargs):
        return ProcessorInputs(
            prompt=[],
            mm_data_items=MultiModalDataParser().parse_mm_data({"image": [ref]}),
            mm_uuid_items={"image": [uuid_item]},
            media_io_kwargs=media_io_kwargs,
        ).get_mm_hashes("test-model", "blake3")["image"][0]

    # Different decode settings still produce different identities...
    assert keep_ref.key != rgb_ref.key
    # ...but the request-level copy of those settings adds nothing on top.
    assert hash_of(keep_ref, None, {}) == hash_of(
        keep_ref, None, {"image": {"image_mode": None}}
    )

    # With a UUID the ref is not hashed, so media_io_kwargs is the only thing
    # keeping two decode settings apart.
    assert hash_of(keep_ref, "image-uuid", {}) != hash_of(
        keep_ref, "image-uuid", {"image": {"image_mode": None}}
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # Shifting the key/value boundary: both flatten to the dotted key
        # "mm_processor_kwargs.abc" followed by no value bytes.
        ({"ab": "c"}, {"a": "bc"}),
        # A nested mapping and a caller-supplied dotted key flatten alike.
        ({"size": {"shortest_edge": 224}}, {"size.shortest_edge": 224}),
        # A sequence and a mapping keyed by stringified indices flatten alike.
        ({"fps": [2, 4]}, {"fps": {"0": 2, "1": 4}}),
        # None contributes the key alone, which a zero-byte value also does.
        ({"video_pruning_rate": None}, {"video_pruning_rate": ""}),
        # An empty container contributes nothing, as does omitting the key.
        ({"size": {}}, {}),
    ],
)
def test_processor_inputs_hashes_distinguish_kwargs_shapes(left, right):
    """Distinct processor kwargs must not share a multi-modal hash.

    ``hf_processor_mm_kwargs`` is per-request input, so both the keys and the
    values here are caller-controlled. The hash is the identity of the
    processor cache entry and is mixed into the prefix-cache block key, so two
    requests sharing one is a cross-request cache hit.
    """
    image = random_image(np.random.RandomState(0), min_wh=8, max_wh=9)
    mm_data_items = MultiModalDataParser().parse_mm_data({"image": [image]})

    def hash_with(hf_processor_mm_kwargs):
        return ProcessorInputs(
            prompt=[],
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        ).get_mm_hashes("test-model", "blake3")["image"][0]

    assert hash_with(left) != hash_with(right)


class _LazyTestProcessingInfo:
    """Minimal ProcessingInfo for exercising the deferred-decode orchestration
    in `_cached_apply_hf_processor` without a real HF model."""

    model_id = "lazy-test-model"

    def __init__(self) -> None:
        self.data_parser = MultiModalDataParser()
        self.ctx = SimpleNamespace(
            tokenizer=None,
            get_mm_config=lambda: MultiModalConfig(),
        )

    def get_data_parser(self):
        return self.data_parser

    def parse_mm_data(self, mm_data, *, validate=True):
        return self.data_parser.parse_mm_data(mm_data)


class _LazyTestModelConfig:
    def __init__(self, mm_processor_cache_gb: float) -> None:
        self._mm_config = MultiModalConfig(mm_processor_cache_gb=mm_processor_cache_gb)

    def get_multimodal_config(self) -> MultiModalConfig:
        return self._mm_config


class _LazyTestProcessor(BaseMultiModalProcessor):
    """Processor whose HF call fabricates one dummy tensor per image item."""

    requires_tokenizer = False

    def __init__(self) -> None:
        super().__init__(
            _LazyTestProcessingInfo(),  # type: ignore[arg-type]
            dummy_inputs=None,  # type: ignore[arg-type]
        )

        # Encoded bytes visible to byte-consuming models (see dots3_note) at
        # HF-processing time, one entry per media ref.
        self.seen_encoded_bytes = list[bytes]()
        self.fail_hf_processor = False
        self.hf_calls = 0

    def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
        for items in mm_items.values():
            if not isinstance(items, ProcessorBatchItems):
                continue
            for idx in range(items.get_count()):
                raw = items.get_raw(idx)
                if isinstance(raw, MediaRef):
                    self.seen_encoded_bytes.append(raw.data)
        return super()._get_hf_mm_inputs(mm_items, hf_kwargs)

    def _call_hf_processor(self, hf_data, hf_kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        if self.fail_hf_processor:
            raise RuntimeError("boom")

        images = hf_data.get("images") or []
        if not images:
            return BatchFeature()
        self.hf_calls += 1
        return BatchFeature({"pixel_values": torch.zeros(len(images), 1)})

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        if "pixel_values" not in hf_inputs:
            return {}
        return {
            "pixel_values": MultiModalFieldConfig.shared(
                "image", hf_inputs["pixel_values"].shape[0]
            )
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        # One placeholder update per image item so the modality is present
        # in the grouped updates consumed by `_merge_mm_kwargs`. An index
        # target keeps `apply()` usable without a tokenizer.
        return [PromptInsertion("image", PromptIndexTargets.start(), [0])]


def _lazy_cache():
    return MultiModalProcessorOnlyCache(
        _LazyTestModelConfig(mm_processor_cache_gb=1)  # type: ignore[arg-type]
    )


class _CountingDecoder:
    """Decoder that records how many times it ran."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def _lazy_inputs(processor, lazy_items, cache):
    mm_items = MultiModalDataParser().parse_mm_data({"image": lazy_items})
    return ProcessorInputs(prompt=[], mm_data_items=mm_items, cache=cache)


def _lazy_apply(processor, lazy_items, cache=None, timing_ctx=None):
    inputs = _lazy_inputs(processor, lazy_items, cache)
    return processor._cached_apply_hf_processor(
        inputs, timing_ctx or TimingContext(enabled=False)
    )


def test_lazy_cache_hit_skips_decode():
    """A cache hit must not decode; a miss decodes once and releases bytes."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    data = b"fake-image-bytes"

    # First request (miss): the item is decoded once and its bytes released.
    decoder_1 = _CountingDecoder(Image.new("RGB", (4, 4)))
    lazy_1 = MediaRef(decoder_1, data)
    timing_ctx = TimingContext(enabled=True)
    _lazy_apply(processor, [lazy_1], cache, timing_ctx)
    assert decoder_1.calls == 1
    assert lazy_1.data == b""
    assert "decode_mm_items" in timing_ctx.stage_secs

    # Second request with identical bytes (hit): no decode, bytes released
    # right after hashing.
    decoder_2 = _CountingDecoder(Image.new("RGB", (4, 4)))
    lazy_2 = MediaRef(decoder_2, data)
    _lazy_apply(processor, [lazy_2], cache)
    assert decoder_2.calls == 0
    assert lazy_2.data == b""


def test_lazy_cache_miss_decodes_in_parallel():
    """Cache-miss items must decode concurrently, not serialized."""
    num_items = 4
    barrier = threading.Barrier(num_items)
    calls = [0] * num_items

    def make_decoder(idx):
        def decode():
            calls[idx] += 1
            # Deadlocks (BrokenBarrierError after timeout) if the decodes
            # were serialized.
            barrier.wait(timeout=10)
            return Image.new("RGB", (4, 4))

        return decode

    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    lazy_items = [
        MediaRef(make_decoder(idx), f"image-{idx}".encode()) for idx in range(num_items)
    ]
    _lazy_apply(processor, lazy_items, cache)

    assert calls == [1] * num_items


def test_lazy_decode_error_becomes_unprocessable():
    """A decode failure surfaces as VLLMUnprocessableEntityError, and
    in-flight sibling decodes are still awaited rather than abandoned."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()

    def bad_decode():
        raise ValueError("corrupt media")

    slow_completed = threading.Event()

    def slow_decode():
        time.sleep(0.2)
        slow_completed.set()
        return Image.new("RGB", (4, 4))

    with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
        _lazy_apply(
            processor,
            [MediaRef(bad_decode, b"broken"), MediaRef(slow_decode, b"good")],
            cache,
        )

    assert exc_info.value.parameter is None
    assert "image media at index 0" in str(exc_info.value)
    assert slow_completed.is_set()


def test_lazy_miss_bytes_available_during_hf_processing():
    """A cache-miss ref still holds its encoded bytes while the HF processor
    runs, and they are released afterwards."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    lazy = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), b"image-bytes")

    _lazy_apply(processor, [lazy], cache)

    # The HF-facing layer saw the raw bytes for the miss item...
    assert processor.seen_encoded_bytes == [b"image-bytes"]
    # ...and the bytes are released by the time processing returns.
    assert lazy.data == b""


def test_lazy_miss_bytes_released_on_hf_processor_error():
    """Bytes of cache-miss refs are released even if HF processing raises
    (try/finally in `_cached_apply_hf_processor`)."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    processor.fail_hf_processor = True
    lazy = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), b"image-bytes")

    with pytest.raises(RuntimeError, match="boom"):
        _lazy_apply(processor, [lazy], cache)

    assert processor.seen_encoded_bytes == [b"image-bytes"]
    assert lazy.data == b""


@pytest.mark.asyncio
async def test_lazy_phase1_does_not_block_mm_worker():
    """While request A's decode is in flight, request B's phase 1 must be
    able to run on the same single-worker executor (i.e. phase 1 submits
    decodes without joining them)."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    executor = ThreadPoolExecutor(max_workers=1)  # stands in for _mm_executor

    decode_entered = threading.Event()
    release_decode = threading.Event()

    def gated_decode():
        decode_entered.set()
        release_decode.wait(timeout=30)
        return Image.new("RGB", (4, 4))

    inputs_a = _lazy_inputs(processor, [MediaRef(gated_decode, b"a-bytes")], cache)
    inputs_b = _lazy_inputs(
        processor,
        [MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), b"b-bytes")],
        cache,
    )

    try:
        loop = asyncio.get_running_loop()
        state_a = await loop.run_in_executor(
            executor, processor.apply_phase1, inputs_a, TimingContext(enabled=False)
        )
        assert len(state_a.decodes) == 1
        await asyncio.to_thread(decode_entered.wait, 30)
        assert decode_entered.is_set()

        # B's phase 1 + decode + phase 2 all complete while A's decode is
        # still blocked: the mm worker is not occupied by A's decode.
        state_b = await asyncio.wait_for(
            loop.run_in_executor(
                executor, processor.apply_phase1, inputs_b, TimingContext(enabled=False)
            ),
            timeout=30,
        )
        await state_b.wait_decodes_async()
        result_b = await asyncio.wait_for(
            loop.run_in_executor(executor, processor.apply_phase2, state_b),
            timeout=30,
        )
        assert not release_decode.is_set()
        assert result_b["mm_kwargs"]["image"][0] is not None

        release_decode.set()
        await state_a.wait_decodes_async()
        result_a = await asyncio.wait_for(
            loop.run_in_executor(executor, processor.apply_phase2, state_a),
            timeout=30,
        )
        assert result_a["mm_kwargs"]["image"][0] is not None
    finally:
        release_decode.set()
        executor.shutdown(wait=True)


def test_lazy_phase2_rederives_miss_to_hit():
    """If another request caches a miss item between phase 1 and phase 2
    (A1, B1, B2, A2 interleaving), phase 2 must re-check the cache and skip
    reprocessing instead of blindly applying the HF processor."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    data = b"shared-bytes"

    # A starts first: its phase 1 sees a cache miss and submits the decode.
    lazy_a = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), data)
    inputs_a = _lazy_inputs(processor, [lazy_a], cache)
    state_a = processor.apply_phase1(inputs_a, TimingContext(enabled=False))
    state_a.wait_decodes()

    # B's whole apply interleaves between A's phases and caches the content.
    lazy_b = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), data)
    hf_before = processor.hf_calls
    processor.apply(
        _lazy_inputs(processor, [lazy_b], cache), TimingContext(enabled=False)
    )
    assert processor.hf_calls == hf_before + 1

    # A's phase 2 re-checks the cache, finds the hit, and skips reprocessing.
    result_a = processor.apply_phase2(state_a)
    assert processor.hf_calls == hf_before + 1
    assert result_a["mm_kwargs"]["image"][0] is not None


def test_lazy_phase2_handles_hit_eviction():
    """If a phase-1 hit is evicted before phase 2, phase 2 must fall back to
    decoding and processing it instead of asserting on a missing cache
    entry."""
    processor = _LazyTestProcessor()
    cache = _lazy_cache()
    data = b"evict-me"

    first = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), data)
    _lazy_apply(processor, [first], cache)
    assert processor.hf_calls == 1

    lazy2 = MediaRef(_CountingDecoder(Image.new("RGB", (4, 4))), data)
    inputs2 = _lazy_inputs(processor, [lazy2], cache)
    state = processor.apply_phase1(inputs2, TimingContext(enabled=False))

    # Hit in phase 1: nothing to decode, and the bytes are still held --
    # releasing them here would leave the eviction fallback below with
    # nothing to decode from.
    assert not state.decodes
    assert lazy2.data == data

    # The item is evicted between the phases, so phase 2 has to decode and
    # process it after all.
    cache.clear_cache()
    result = processor.apply_phase2(state)

    assert processor.hf_calls == 2
    assert result["mm_kwargs"]["image"][0] is not None
    assert lazy2.data == b""


@pytest.mark.parametrize(
    ("chunked_prefill", "max_model_len", "expected_seq_len"),
    [(None, 491520, 491520), (True, 491520, 8192), (True, 128, 128), (False, 128, 128)],
)
def test_dummy_inputs_scheduler_budget(
    chunked_prefill, max_model_len, expected_seq_len
):
    ctx = build_model_context(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": 1},
    )
    ctx.model_config.max_model_len = max_model_len

    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )
    processor.apply = lambda *args, **kwargs: {"prompt_token_ids": [7]}

    kwargs = {}
    if chunked_prefill is not None:
        kwargs["scheduler_config"] = SchedulerConfig(
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            max_num_batched_tokens=8192,
            max_num_seqs=1,
            enable_chunked_prefill=chunked_prefill,
        )

    result = processor.get_dummy_mm_inputs({"image": 1}, **kwargs)
    assert len(result["prompt_token_ids"]) == expected_seq_len
