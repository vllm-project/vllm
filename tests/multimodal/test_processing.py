from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (PlaceholderFeaturesInfo,
                                        PromptReplacement,
                                        find_mm_placeholders,
                                        find_text_matches, find_token_matches,
                                        iter_token_matches,
                                        replace_text_matches,
                                        replace_token_matches)
# yapf: enable
from vllm.multimodal.profiling import MultiModalProfiler
from vllm.multimodal.utils import cached_get_tokenizer
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
    ("prompt", "target_by_key", "expected_by_key"),
    [
        (
            [],
            {
                "pattern_1": [],
                "pattern_2": [32000],
            },
            {
                "pattern_1": [],
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

    prompt_repls = [
        PromptReplacement(key, target, []).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    result = find_token_matches(prompt, prompt_repls)

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

    prompt_repls = [
        PromptReplacement(key, target, []).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    result = find_text_matches(prompt, prompt_repls)

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
    ("prompt", "target_by_key", "repl_by_key"),
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
        ),
    ]
)
@pytest.mark.parametrize(
    ("mm_count", "expected"),
    [
        (0, "Image:<image>Image:<image><image>!"),
        (1, "<image><image>Image:<image><image>?!?"),
        (2, "<image><image><image><image><image>?!?"),
    ]
)
# yapf: enable
def test_find_replace_text(
    prompt,
    target_by_key,
    repl_by_key,
    mm_count,
    expected,
):
    # Should not be used since there is nothing to convert to text
    mock_tokenizer = cast(AnyTokenizer, object())

    mm_prompt_repls = {
        key: [
            PromptReplacement(key, target,
                              repl_by_key[key]).bind(mock_tokenizer)
        ]
        for key, target in target_by_key.items()
    }
    mm_matches = {
        key: find_text_matches(prompt, prompt_repls)
        for key, prompt_repls in mm_prompt_repls.items()
    }

    result = replace_text_matches(
        prompt,
        mm_matches,
        {key: mm_count
         for key in repl_by_key},
    )

    # Only displayed on error
    print("mm_matches:", mm_matches)
    print("result:", result)

    # Manually constructed results
    assert result == expected


# yapf: disable
@pytest.mark.parametrize(
    ("prompt", "target_by_key", "repl_by_key"),
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
        ),
    ]
)
@pytest.mark.parametrize(
    ("mm_count", "expected"),
    [
        (0, [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918]),
        (1, [1, 32000, 32000, 9833, 28747, 32000, 32000, 1550, 918, 1550]),
        (2, [1, 32000, 32000, 32000, 32000, 32000, 1550, 918, 1550]),
    ]
)
# yapf: enable
def test_find_replace_tokens(
    prompt,
    target_by_key,
    repl_by_key,
    mm_count,
    expected,
):
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    mm_prompt_repls = {
        key: [
            PromptReplacement(key, target,
                              repl_by_key[key]).bind(mock_tokenizer)
        ]
        for key, target in target_by_key.items()
    }
    mm_matches = {
        key: find_token_matches(prompt, prompt_repls)
        for key, prompt_repls in mm_prompt_repls.items()
    }

    result = replace_token_matches(
        prompt,
        mm_matches,
        {key: mm_count
         for key in repl_by_key},
    )

    # Only displayed on error
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
                    ),
                ],
                "pattern_4": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_4",
                        item_idx=0,
                        start_idx=3,
                        tokens=[32000],
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
                    ),
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=1,
                        start_idx=5,
                        tokens=[32000, 32000],
                    ),
                ],
                "pattern_3": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_3",
                        item_idx=0,
                        start_idx=7,
                        tokens=[1550, 918, 1550],
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
                    ),
                    PlaceholderFeaturesInfo(
                        modality="pattern_1",
                        item_idx=1,
                        start_idx=3,
                        tokens=[32000, 32000],
                    ),
                ],
                "pattern_4": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_4",
                        item_idx=0,
                        start_idx=5,
                        tokens=[32000],
                    ),
                ],
                "pattern_3": [
                    PlaceholderFeaturesInfo(
                        modality="pattern_3",
                        item_idx=0,
                        start_idx=6,
                        tokens=[1550, 918, 1550],
                    ),
                ],
            }
        ),
    ]
)
# yapf: enable
def test_find_mm_placeholders(
    repl_by_key,
    prompt,
    expected,
):
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    mm_prompt_repls = {
        key: [PromptReplacement(key, [], repl).bind(mock_tokenizer)]
        for key, repl in repl_by_key.items()
    }

    result = find_mm_placeholders(
        mm_prompt_repls,
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
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="half",
        revision=None,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(
        model_config,
        tokenizer=cached_get_tokenizer(model_config.tokenizer),
    )
    profiler = MultiModalProfiler(processor)

    mock_supported_mm_limits = MagicMock(return_value={"image": num_supported})
    processor.info.get_supported_mm_limits = mock_supported_mm_limits

    if is_valid:
        exc_ctx = nullcontext()
    else:
        exc_ctx = pytest.raises(ValueError, match="this model only supports")

    with exc_ctx:
        profiler.get_dummy_data(model_config.max_model_len)


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
        task="auto",
        tokenizer=model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="half",
        revision=None,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )

    processor = MULTIMODAL_REGISTRY.create_processor(
        model_config,
        tokenizer=cached_get_tokenizer(model_config.tokenizer),
    )

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
        exc_ctx = pytest.raises(ValueError, match=f"passed {num_images} image")

    with exc_ctx:
        processor.apply(
            "<image>" * num_images,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )
