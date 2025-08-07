# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from typing import Optional, cast

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.inputs import InputProcessingContext
from vllm.multimodal import MULTIMODAL_REGISTRY
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (PlaceholderFeaturesInfo,
                                        PromptIndexTargets, PromptInsertion,
                                        PromptReplacement, apply_text_matches,
                                        apply_token_matches,
                                        find_mm_placeholders,
                                        find_text_matches, find_token_matches,
                                        iter_token_matches,
                                        replace_token_matches)
# yapf: enable
from vllm.multimodal.profiling import MultiModalProfiler
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import full_groupby

from .utils import random_image


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "match_ids", "expected"),
    [
        ([], [], []),
        ([], [32000], []),
        (
            [32000, 32000, 32000],
            [32000],
            [
                { "start_idx": 0, "end_idx": 1 },
                { "start_idx": 1, "end_idx": 2 },
                { "start_idx": 2, "end_idx": 3 },
            ],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000],
            [{ "start_idx": 0, "end_idx": 2 }],
        ),
        (
            [32000, 32000, 32000],
            [32000, 32000, 32000],
            [{ "start_idx": 0, "end_idx": 3 }],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000],
            [
                { "start_idx": 1, "end_idx": 3 },
                { "start_idx": 6, "end_idx": 8 },
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 32000, 32000, 32000],
            [
                { "start_idx": 1, "end_idx": 5 },
            ],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [28747, 0, 32000],
            [],
        ),
    ],
)
# yapf: enable
def test_iter_token_matches(token_ids, match_ids, expected):
    result = list(iter_token_matches(token_ids, match_ids))

    # Manually constructed results
    assert [item._asdict() for item in result] == expected

    # Invariants
    match_lens = [end - start for start, end in result]
    print("match_lens:", match_lens)  # Only displayed on error
    assert all(match_len == len(match_ids) for match_len in match_lens)


# yapf: disable
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
# yapf: enable
def test_replace_token_matches(token_ids, match_ids, new_ids, expected):
    result = replace_token_matches(token_ids, match_ids, new_ids)

    # Manually constructed results
    assert result == expected


# yapf: disable
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
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_4": [],
                "pattern_5": [
                    { "start_idx": 0, "end_idx": 0 },
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
                    { "start_idx": 0, "end_idx": 1 },
                    { "start_idx": 1, "end_idx": 2 },
                    { "start_idx": 2, "end_idx": 3 },
                    { "start_idx": 3, "end_idx": 4 },
                ],
                "pattern_2": [
                    { "start_idx": 0, "end_idx": 2 },
                    { "start_idx": 2, "end_idx": 4 },
                ],
                "pattern_3": [
                    { "start_idx": 0, "end_idx": 3 },
                ],
                "pattern_4": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_5": [
                    { "start_idx": 1, "end_idx": 1 },
                ],
                "pattern_6": [
                    { "start_idx": 4, "end_idx": 4 },
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
                    { "start_idx": 1, "end_idx": 3 },
                    { "start_idx": 6, "end_idx": 8 },
                ],
                "pattern_2": [
                    { "start_idx": 1, "end_idx": 5 },
                ],
                "pattern_3": [],
                "pattern_4": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_5": [],
                "pattern_6": [
                    { "start_idx": 10, "end_idx": 10 },
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
# yapf: enable
def test_find_token_matches(
    prompt,
    target_by_key,
    expected_by_key,
    update_type,
):
    # Should not be used since there is nothing to convert to token IDs
    mock_tokenizer = cast(AnyTokenizer, object())

    prompt_updates = [
        update_type(key, target, []).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    result = find_token_matches(prompt, prompt_updates)

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    result_groups = dict(full_groupby(result, key=lambda x: x.modality))
    assert {
        key: [
            dict(start_idx=item.start_idx, end_idx=item.end_idx)
            for item in result_groups.get(key, [])
        ]
        for key in expected_by_key
    } == expected_by_key


# yapf: disable
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
                "pattern_1": [{ "start_idx": 0, "end_idx": 0 }],
                "pattern_2": [],
                "pattern_3": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_4": [],
                "pattern_5": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
            }
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
                    { "start_idx": 0, "end_idx": 7 },
                    { "start_idx": 7, "end_idx": 14 },
                    { "start_idx": 14, "end_idx": 21 },
                    { "start_idx": 21, "end_idx": 28 },
                ],
                "pattern_2": [
                    { "start_idx": 0, "end_idx": 14 },
                    { "start_idx": 14, "end_idx": 28 },
                ],
                "pattern_3": [
                    { "start_idx": 0, "end_idx": 21 },
                ],
                "pattern_4": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_5": [
                    { "start_idx": 7, "end_idx": 7 },
                ],
                "pattern_6": [
                    { "start_idx": 28, "end_idx": 28 },
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
                    { "start_idx": 0, "end_idx": 13 },
                    { "start_idx": 27, "end_idx": 40 },
                ],
                "pattern_2": [
                    { "start_idx": 0, "end_idx": 27 },
                ],
                "pattern_3": [],
                "pattern_4": [
                    { "start_idx": 0, "end_idx": 0 },
                ],
                "pattern_5": [
                    { "start_idx": 13, "end_idx": 13 },
                ],
                "pattern_6": [
                    { "start_idx": 48, "end_idx": 48 },
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
                    { "start_idx": 0, "end_idx": 9 },
                    { "start_idx": 16, "end_idx": 25 },
                ],
                "pattern_2": [
                    { "start_idx": 0, "end_idx": 16 },
                    { "start_idx": 16, "end_idx": 32 },
                ],
                "pattern_3": [
                    { "start_idx": 0, "end_idx": 25 },
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
# yapf: enable
def test_find_text_matches(
    prompt,
    target_by_key,
    expected_by_key,
    update_type,
):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

    prompt_updates = [
        update_type(key, target, []).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    result = find_text_matches(prompt, prompt_updates)

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    result_groups = dict(full_groupby(result, key=lambda x: x.modality))
    assert {
        key: [
            dict(start_idx=item.start_idx, end_idx=item.end_idx)
            for item in result_groups.get(key, [])
        ]
        for key in expected_by_key
    } == expected_by_key


# yapf: disable
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
    ]
)
# yapf: enable
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
        mm_prompt_updates = {
            key:
            [update_type(key, target, repl_by_key[key]).bind(mock_tokenizer)]
            for key, target in target_by_key.items()
        }
        mm_matches = {
            key: find_text_matches(prompt, updates)
            for key, updates in mm_prompt_updates.items()
        }

        for mm_count, expected in expected_by_mm_count.items():
            result = apply_text_matches(
                prompt,
                mm_matches,
                {key: mm_count
                 for key in repl_by_key},
            )

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_matches:", mm_matches)
            print("result:", result)

            # Manually constructed results
            assert result == expected


# yapf: disable
@pytest.mark.parametrize(
    ("prompt", "target_by_key", "repl_by_key", "expected_by_update_type_mm_count"),  # noqa: E501
    [
        # Tokenized test cases of `test_find_replace_text`
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
                    1: [1, 9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918, 1550, 918, 1550],  # noqa: E501
                    2: [1, 9833, 28747, 32000, 32000, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918, 1550, 918, 1550, 1550, 918, 1550],  # noqa: E501
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
)
# yapf: enable
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
        mm_prompt_updates = {
            key:
            [update_type(key, target, repl_by_key[key]).bind(mock_tokenizer)]
            for key, target in target_by_key.items()
        }
        mm_matches = {
            key: find_token_matches(prompt, updates)
            for key, updates in mm_prompt_updates.items()
        }

        for mm_count, expected in expected_by_mm_count.items():
            result = apply_token_matches(
                prompt,
                mm_matches,
                {key: mm_count
                 for key in repl_by_key},
            )

            # Only displayed on error
            print("update_type:", update_type)
            print("mm_count:", mm_count)
            print("mm_matches:", mm_matches)
            print("result:", result)

            # Manually constructed results
            assert result == expected


# yapf: disable
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
            }

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
            }
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
            }
        ),
    ]
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
# yapf: enable
def test_find_mm_placeholders(
    repl_by_key,
    prompt,
    expected,
    update_type,
):
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    mm_prompt_updates = {
        key: [update_type(key, [], repl).bind(mock_tokenizer)]
        for key, repl in repl_by_key.items()
    }

    result = find_mm_placeholders(
        mm_prompt_updates,
        prompt,
        # Effectively match all occurrences in the prompt
        {key: 3
         for key in repl_by_key},
    )

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert result == expected


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("limit", "num_supported", "is_valid"),
    [(0, 0, True), (0, 1, True), (1, 0, False), (1, 1, True), (1, 2, True),
     (2, 1, False), (2, 2, True)],
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

    if is_valid:
        exc_ctx = nullcontext()
    else:
        exc_ctx = pytest.raises(ValueError, match="At most")

    with exc_ctx:
        profiler.get_decoder_dummy_data(
            model_config.max_model_len,
            mm_counts=limit_mm_per_prompt,
        )


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("num_images", "limit", "is_valid"),
    [(0, 0, True), (0, 1, True), (1, 0, False), (1, 1, True), (1, 2, True),
     (2, 1, False), (2, 2, True)],
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

    if is_valid:
        exc_ctx = nullcontext()
    else:
        exc_ctx = pytest.raises(ValueError, match="At most")

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
        return_tensors: Optional[str] = None,
    ) -> dict[str, int]:
        return dict(a=a, c=c)


# yapf: disable
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
# yapf: enable
def test_hf_processor_init_kwargs(
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

    processor = ctx.get_hf_processor(
        DummyProcessor,  # type: ignore[arg-type]
        **inference_kwargs,
    )

    for k, v in expected_kwargs.items():
        assert getattr(processor, k) == v


# yapf: disable
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
# yapf: enable
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
