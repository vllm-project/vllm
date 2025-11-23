# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from typing import cast

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    InputProcessingContext,
    PlaceholderFeaturesInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptReplacement,
    apply_text_matches,
    apply_token_matches,
    find_mm_placeholders,
    iter_token_matches,
    replace_token_matches,
)
from vllm.multimodal.profiling import MultiModalProfiler
from vllm.transformers_utils.tokenizer import AnyTokenizer

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
    # Should not be used since there is nothing to convert to token IDs
    mock_tokenizer = cast(AnyTokenizer, object())

    prompt_updates = {
        key: update_type(key, target, []).resolve(0)
        for key, target in target_by_key.items()
    }
    result = {
        key: list(update.iter_token_matches(prompt, mock_tokenizer))
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


@pytest.mark.parametrize(
    ("prompt", "target_by_key", "expected_by_key"),
    [
        # Detokenized test cases of `test_find_token_matches`
        # using the vocab of llava-hf/llava-v1.6-mistral-7b-hf
        (
            "",
            {
                "pattern_1": "",
                "pattern_2": "<image>",
                "pattern_3": PromptIndexTargets.start(),
                "pattern_4": PromptIndexTargets.prefix("<image>"),
                "pattern_5": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [{"start_idx": 0, "end_idx": 0}],
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
            "<image><image><image><image>",
            {
                "pattern_1": "<image>",
                "pattern_2": "<image><image>",
                "pattern_3": "<image><image><image>",
                "pattern_4": PromptIndexTargets.start(),
                "pattern_5": PromptIndexTargets.prefix("<image>"),
                "pattern_6": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [
                    {"start_idx": 0, "end_idx": 7},
                    {"start_idx": 7, "end_idx": 14},
                    {"start_idx": 14, "end_idx": 21},
                    {"start_idx": 21, "end_idx": 28},
                ],
                "pattern_2": [
                    {"start_idx": 0, "end_idx": 14},
                    {"start_idx": 14, "end_idx": 28},
                ],
                "pattern_3": [
                    {"start_idx": 0, "end_idx": 21},
                ],
                "pattern_4": [
                    {"start_idx": 0, "end_idx": 0},
                ],
                "pattern_5": [
                    {"start_idx": 7, "end_idx": 7},
                ],
                "pattern_6": [
                    {"start_idx": 28, "end_idx": 28},
                ],
            },
        ),
        (
            "Image:<image><image><image>Image:<image><image>!",
            {
                "pattern_1": "Image:<image>",
                "pattern_2": "Image:<image><image><image>",
                "pattern_3": "Image:<unk><image>",
                "pattern_4": PromptIndexTargets.start(),
                "pattern_5": PromptIndexTargets.prefix("Image:<image>"),
                "pattern_6": PromptIndexTargets.end(),
            },
            {
                "pattern_1": [
                    {"start_idx": 0, "end_idx": 13},
                    {"start_idx": 27, "end_idx": 40},
                ],
                "pattern_2": [
                    {"start_idx": 0, "end_idx": 27},
                ],
                "pattern_3": [],
                "pattern_4": [
                    {"start_idx": 0, "end_idx": 0},
                ],
                "pattern_5": [
                    {"start_idx": 13, "end_idx": 13},
                ],
                "pattern_6": [
                    {"start_idx": 48, "end_idx": 48},
                ],
            },
        ),
        # Test regex escape
        (
            "<|image|><image><|image|><image>",
            {
                "pattern_1": "<|image|>",
                "pattern_2": "<|image|><image>",
                "pattern_3": "<|image|><image><|image|>",
            },
            {
                "pattern_1": [
                    {"start_idx": 0, "end_idx": 9},
                    {"start_idx": 16, "end_idx": 25},
                ],
                "pattern_2": [
                    {"start_idx": 0, "end_idx": 16},
                    {"start_idx": 16, "end_idx": 32},
                ],
                "pattern_3": [
                    {"start_idx": 0, "end_idx": 25},
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
def test_find_text_matches(
    prompt,
    target_by_key,
    expected_by_key,
    update_type,
):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

    prompt_updates = {
        key: update_type(key, target, []).resolve(0)
        for key, target in target_by_key.items()
    }
    result = {
        key: list(update.iter_text_matches(prompt, mock_tokenizer))
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


@pytest.mark.parametrize(
    ("prompt", "target_by_key", "repl_by_key", "expected_by_update_type_mm_count"),  # noqa: E501
    [
        (
            "Image:<image>Image:<image><image>!",
            {
                # We use `<image>` before `Image:` to test matches that
                # occur out of order
                "pattern_1": "<image>",
                "pattern_2": "Image:",
                "pattern_3": "!",
            },
            {
                # Test whether target is confused with replacement
                "pattern_1": "<image><image>",
                # Test empty replacement
                "pattern_2": "",
                # Test dynamic replacement (beyond the form of `unit * count`)
                "pattern_3": "?!?",
            },
            {
                PromptInsertion: {
                    0: "Image:<image>Image:<image><image>!",
                    1: "Image:<image><image><image>Image:<image><image>!?!?",
                    2: "Image:<image><image><image><image><image>Image:<image><image>!?!??!?",  # noqa: E501
                },
                PromptReplacement: {
                    0: "Image:<image>Image:<image><image>!",
                    1: "<image><image>Image:<image><image>?!?",
                    2: "<image><image><image><image><image>?!?",
                },
            },
        ),
        # Test index targets
        (
            "",
            {
                "pattern_1": PromptIndexTargets.start(),
                "pattern_2": PromptIndexTargets.prefix("<image>"),
                "pattern_3": PromptIndexTargets.end(),
            },
            {
                "pattern_1": "1",
                "pattern_2": "2",
                "pattern_3": "3",
            },
            {
                PromptInsertion: {
                    0: "",
                    1: "13",
                    2: "1133",
                },
                PromptReplacement: {
                    0: "",
                    1: "13",
                    2: "1133",
                },
            },
        ),
        (
            "<image>",
            {
                "pattern_1": PromptIndexTargets.start(),
                "pattern_2": PromptIndexTargets.prefix("<image>"),
                "pattern_3": PromptIndexTargets.end(),
            },
            {
                "pattern_1": "1",
                "pattern_2": "2",
                "pattern_3": "3",
            },
            {
                PromptInsertion: {
                    0: "<image>",
                    1: "1<image>23",
                    2: "11<image>2233",
                },
                PromptReplacement: {
                    0: "<image>",
                    1: "1<image>23",
                    2: "11<image>2233",
                },
            },
        ),
        # Test different replacement per item
        (
            "<image><image><image>",
            {
                "pattern_1": "<image>",
            },
            {
                "pattern_1": lambda idx: str(idx + 1),
            },
            {
                PromptInsertion: {
                    0: "<image><image><image>",
                    1: "<image>1<image><image>",
                    2: "<image>12<image><image>",
                },
                PromptReplacement: {
                    0: "<image><image><image>",
                    1: "1<image><image>",
                    2: "12<image>",
                },
            },
        ),
        (
            "<image><image><image>",
            {
                "pattern_1": PromptIndexTargets.prefix("<image>"),
            },
            {
                "pattern_1": lambda idx: str(idx + 1),
            },
            {
                PromptInsertion: {
                    0: "<image><image><image>",
                    1: "<image>1<image><image>",
                    2: "<image>12<image><image>",
                },
                PromptReplacement: {
                    0: "<image><image><image>",
                    1: "<image>1<image><image>",
                    2: "<image>12<image><image>",
                },
            },
        ),
    ],
)
def test_find_update_text(
    prompt,
    target_by_key,
    repl_by_key,
    expected_by_update_type_mm_count,
):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

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

            new_prompt, result = apply_text_matches(
                prompt,
                mm_prompt_updates,
                mock_tokenizer,
            )

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_prompt_updates:", mm_prompt_updates)
            print("new_prompt:", new_prompt)
            print("result:", result)

            # Manually constructed results
            assert new_prompt == expected


@pytest.mark.parametrize(
    ("prompt", "target_by_key", "repl_by_key", "expected_by_update_type_mm_count"),  # noqa: E501
    [
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
    ],
)
def test_find_update_tokens(
    prompt,
    target_by_key,
    repl_by_key,
    expected_by_update_type_mm_count,
):
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

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

            new_prompt, result = apply_token_matches(
                prompt,
                mm_prompt_updates,
                mock_tokenizer,
            )

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_prompt_updates:", mm_prompt_updates)
            print("new_prompt:", new_prompt)
            print("result:", result)

            # Manually constructed results
            assert new_prompt == expected


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
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    mm_prompt_updates = {
        key: [[update_type(key, [], repl).resolve(i)] for i in range(3)]
        for key, repl in repl_by_key.items()
    }

    result = find_mm_placeholders(prompt, mm_prompt_updates, mock_tokenizer)

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert result == expected


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("limit", "num_supported", "is_valid"),
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
def test_limit_mm_per_prompt_dummy(model_id, limit, num_supported, is_valid):
    limit_mm_per_prompt = {"image": limit}

    model_config = ModelConfig(
        model=model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    processor._supported_mm_limits = {"image": num_supported}

    profiler = MultiModalProfiler(processor)

    exc_ctx = nullcontext() if is_valid else pytest.raises(ValueError, match="At most")

    with exc_ctx:
        profiler.get_decoder_dummy_data(
            model_config.max_model_len,
            mm_counts=limit_mm_per_prompt,
        )


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

    exc_ctx = nullcontext() if is_valid else pytest.raises(ValueError, match="At most")

    with exc_ctx:
        processor.apply(
            "<image>" * num_images,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )


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
    processor = _get_dummy_processor(
        model_id=model_id,
# Test Qwen3 Omni audio_sample_rate preservation
class TestQwen3OmniAudioSampleRatePreservation:
    """Test that audio_sample_rate is preserved during kwargs restructuring.

    These tests validate the fix for the audio_sample_rate bug in Qwen3 Omni
    where the parameter was lost during kwargs restructuring. The tests don't
    require importing the actual model classes - they just test the kwargs
    manipulation logic.
    """

    def test_audio_sample_rate_preserved_in_audio_kwargs(self) -> None:
        """
        Test that audio_sample_rate is moved from top-level mm_kwargs
        into audio_kwargs during kwargs restructuring.

        This is the core fix: when transformers < 4.58.0, the code
        restructures kwargs into audio_kwargs and text_kwargs, and
        audio_sample_rate must be preserved in audio_kwargs.
        """
        from packaging.version import Version

        # Setup: Create mm_kwargs with audio_sample_rate at top level
        mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": 16000,
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {
            "truncation": False,
        }

        # Execute: Simulate the kwargs processing (the fix)
        mm_kwargs_copy = dict(mm_kwargs)
        tok_kwargs_copy = dict(tok_kwargs)

        transformers_ver = "4.57.0"
        if Version(transformers_ver) < Version("4.58.0"):
            # Extract audio_sample_rate before restructuring (THE FIX)
            audio_sample_rate = mm_kwargs_copy.pop("audio_sample_rate", None)

            # Restructure kwargs
            mm_kwargs_copy["audio_kwargs"] = {
                "truncation": mm_kwargs_copy.pop("truncation", False)
            }
            mm_kwargs_copy["text_kwargs"] = {
                "truncation": tok_kwargs_copy.pop("truncation", False)
            }

            # Put audio_sample_rate into audio_kwargs (THE FIX)
            if audio_sample_rate is not None:
                mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] = (
                    audio_sample_rate
                )

        # Assert: Verify audio_sample_rate is in audio_kwargs
        assert "audio_kwargs" in mm_kwargs_copy
        assert "audio_sample_rate" in mm_kwargs_copy["audio_kwargs"]
        assert mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] == 16000

        # Assert: Verify truncation is also in audio_kwargs
        assert mm_kwargs_copy["audio_kwargs"]["truncation"] is True

        # Assert: Verify text_kwargs is created correctly
        assert "text_kwargs" in mm_kwargs_copy
        assert mm_kwargs_copy["text_kwargs"]["truncation"] is False

    def test_audio_sample_rate_absent_when_not_provided(self) -> None:
        """
        Test that when audio_sample_rate is not provided in mm_kwargs,
        the restructured audio_kwargs doesn't contain it.
        """
        from packaging.version import Version

        # Setup: Create mm_kwargs WITHOUT audio_sample_rate
        mm_kwargs: dict[str, Any] = {
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {
            "truncation": False,
        }

        # Execute: Simulate the kwargs processing
        mm_kwargs_copy = dict(mm_kwargs)
        tok_kwargs_copy = dict(tok_kwargs)

        transformers_ver = "4.57.0"
        if Version(transformers_ver) < Version("4.58.0"):
            # Extract audio_sample_rate (will be None)
            audio_sample_rate = mm_kwargs_copy.pop("audio_sample_rate", None)

            # Restructure kwargs
            mm_kwargs_copy["audio_kwargs"] = {
                "truncation": mm_kwargs_copy.pop("truncation", False)
            }
            mm_kwargs_copy["text_kwargs"] = {
                "truncation": tok_kwargs_copy.pop("truncation", False)
            }

            # Only add audio_sample_rate if it exists
            if audio_sample_rate is not None:
                mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] = (
                    audio_sample_rate
                )

        # Assert: Verify audio_sample_rate is NOT in audio_kwargs
        assert "audio_kwargs" in mm_kwargs_copy
        assert "audio_sample_rate" not in mm_kwargs_copy["audio_kwargs"]

        # Assert: Verify truncation is still in audio_kwargs
        assert mm_kwargs_copy["audio_kwargs"]["truncation"] is True

    @pytest.mark.parametrize(
        "sample_rate", [8000, 16000, 22050, 24000, 44100, 48000]
    )
    def test_various_audio_sample_rates_preserved(
        self, sample_rate: int
    ) -> None:
        """
        Test that various common audio sample rates are preserved.

        Common sample rates:
        - 8000: Telephone quality
        - 16000: Wideband speech (Qwen3 Omni default)
        - 22050: Low-quality audio
        - 24000: High-quality speech
        - 44100: CD quality
        - 48000: Professional audio
        """
        from packaging.version import Version

        # Setup: Create mm_kwargs with specific sample rate
        mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": sample_rate,
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {"truncation": False}

        # Execute: Simulate the kwargs processing
        mm_kwargs_copy = dict(mm_kwargs)
        tok_kwargs_copy = dict(tok_kwargs)

        transformers_ver = "4.57.0"
        if Version(transformers_ver) < Version("4.58.0"):
            audio_sample_rate_val = mm_kwargs_copy.pop(
                "audio_sample_rate", None
            )
            mm_kwargs_copy["audio_kwargs"] = {
                "truncation": mm_kwargs_copy.pop("truncation", False)
            }
            mm_kwargs_copy["text_kwargs"] = {
                "truncation": tok_kwargs_copy.pop("truncation", False)
            }
            if audio_sample_rate_val is not None:
                mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] = (
                    audio_sample_rate_val
                )

        # Assert: Verify the specific sample rate is preserved
        assert mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] == sample_rate


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
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    ctx = InputProcessingContext(
        model_config=ModelConfig(model_id, mm_processor_kwargs=config_kwargs),
        tokenizer=mock_tokenizer,
    )

    processor = ctx.get_hf_processor(DummyProcessor)  # type: ignore[arg-type]

    result = ctx.call_hf_processor(processor, {}, inference_kwargs)
    assert result == expected_kwargs
