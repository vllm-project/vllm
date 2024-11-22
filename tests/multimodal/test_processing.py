from typing import cast

import pytest
from transformers import BatchFeature

from vllm.multimodal.processing import (PromptReplacement, find_text_matches,
                                        find_token_matches, iter_token_matches,
                                        iter_token_runs, replace_text_matches)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import full_groupby


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], []),
        (
            [32000, 32000, 32000],
            [{ "token_id": 32000, "start_idx": 0, "length": 3 }],
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [
                { "token_id": 9833, "start_idx": 0, "length": 1 },
                { "token_id": 28747, "start_idx": 1, "length": 1 },
                { "token_id": 32000, "start_idx": 2, "length": 3 },
                { "token_id": 9833, "start_idx": 5, "length": 1 },
                { "token_id": 28747, "start_idx": 6, "length": 1 },
                { "token_id": 32000, "start_idx": 7, "length": 2 },
                { "token_id": 918, "start_idx": 9, "length": 1 },
            ],
        ),
    ],
)
# yapf: enable
def test_iter_token_runs(token_ids, expected):
    result = list(iter_token_runs(token_ids))

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert [item._asdict() for item in result] == expected

    # Invariants
    assert sum(run_info.length for run_info in result) == len(token_ids)


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "match_ids", "expected"),
    [
        ([], [], [{ "start_idx": 0, "end_idx": 0 }]),
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
    ("prompt", "target_by_key", "expected_by_key"),
    [
        (
            [],
            {
                "pattern_1": [],
                "pattern_2": [32000],
            },
            {
                "pattern_1": [{ "start_idx": 0, "end_idx": 0 }],
                "pattern_2": [],
            }
        ),
        (
            [32000, 32000, 32000, 32000],
            {
                "pattern_1": [32000],
                "pattern_2": [32000, 32000],
                "pattern_3": [32000, 32000, 32000],
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
            },
        ),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            {
                "pattern_1": [28747, 32000],
                "pattern_2": [28747, 32000, 32000, 32000],
                "pattern_3": [28747, 0, 32000],
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
            },
        ),
    ],
)
# yapf: enable
def test_find_token_matches(prompt, target_by_key, expected_by_key):
    # Should not be used since there is nothing to convert to token IDs
    mock_tokenizer = cast(AnyTokenizer, object())

    result = find_token_matches(
        prompt,
        [
            PromptReplacement(target, [], 0).bind(key, mock_tokenizer)
            for key, target in target_by_key.items()
        ],
    )

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
            },
            {
                "pattern_1": [{ "start_idx": 0, "end_idx": 0 }],
                "pattern_2": [],
            }
        ),
        (
            "<image><image><image><image>",
            {
                "pattern_1": "<image>",
                "pattern_2": "<image><image>",
                "pattern_3": "<image><image><image>",
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
            },
        ),
        (
            "Image:<image><image><image>Image:<image><image>!",
            {
                "pattern_1": "Image:<image>",
                "pattern_2": "Image:<image><image><image>",
                "pattern_3": "Image:<unk><image>",
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
# yapf: enable
def test_find_text_matches(prompt, target_by_key, expected_by_key):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

    result = find_text_matches(
        prompt,
        [
            PromptReplacement(target, [], 0).bind(key, mock_tokenizer)
            for key, target in target_by_key.items()
        ],
    )

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
    ("prompt", "target_by_key", "repl_by_key", "expected_by_mm_count"),
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
                # Test whether target is confused with repl_unit
                "pattern_1": ("<image><image>", 1),
                # Test empty repl_unit
                "pattern_2": ("", 1),
                # Test multiple repl_count
                "pattern_3": ("?", 2),
            },
            {
                # Test no replacement
                0: "Image:<image>Image:<image><image>!",
                # Test single replacement
                1: "<image><image>Image:<image><image>??",
                # Test repeated replacement
                2: "<image><image><image><image><image>??",
            },
        ),
    ]
)
# yapf: enable
def test_find_replace_text(
    prompt,
    target_by_key,
    repl_by_key,
    expected_by_mm_count,
):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

    matches = find_text_matches(
        prompt,
        [
            PromptReplacement(target, *repl_by_key[key]) \
                .bind(key, mock_tokenizer)
            for key, target in target_by_key.items()
        ],
    )
    result_by_mm_count = {
        mm_count: replace_text_matches(
            prompt,
            matches,
            {key: list(range(mm_count))
             for key in repl_by_key},
            BatchFeature(),
        )
        for mm_count in expected_by_mm_count
    }

    # Only displayed on error
    print("matches:", matches)
    print("result_by_mm_count:", result_by_mm_count)

    # Manually constructed results
    assert result_by_mm_count == expected_by_mm_count
