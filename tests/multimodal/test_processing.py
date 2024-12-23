from typing import cast

import numpy as np
import pytest
from PIL import Image

from vllm.config import ModelConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (InputProcessingContext,
                                        ProcessingCache, PromptReplacement,
                                        _PlaceholderInfo, find_text_matches,
                                        find_token_matches, iter_placeholders,
                                        iter_token_matches,
                                        replace_text_matches,
                                        replace_token_matches)
# yapf: enable
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import full_groupby


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

    prompt_repls = [
        PromptReplacement(key, target, repl_by_key[key]).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    matches = find_text_matches(prompt, prompt_repls)

    result = replace_text_matches(
        prompt,
        matches,
        {key: mm_count
         for key in repl_by_key},
    )

    # Only displayed on error
    print("matches:", matches)
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

    prompt_repls = [
        PromptReplacement(key, target, repl_by_key[key]).bind(mock_tokenizer)
        for key, target in target_by_key.items()
    ]
    matches = find_token_matches(prompt, prompt_repls)

    result = replace_token_matches(
        prompt,
        matches,
        {key: mm_count
         for key in repl_by_key},
    )

    # Only displayed on error
    print("matches:", matches)
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
        },
    ],
)
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            [1, 9833, 28747, 32000, 9833, 28747, 32000, 32000, 918],
            [
                _PlaceholderInfo(
                    modality="pattern_1",
                    start_idx=6,
                    replacement=[32000, 32000],
                ),
            ],
        ),
        (
            [1, 32000, 32000, 9833, 28747, 32000, 32000, 1550, 918, 1550],
            [
                _PlaceholderInfo(
                    modality="pattern_1",
                    start_idx=1,
                    replacement=[32000, 32000],
                ),
                _PlaceholderInfo(
                    modality="pattern_1",
                    start_idx=5,
                    replacement=[32000, 32000],
                ),
                _PlaceholderInfo(
                    modality="pattern_3",
                    start_idx=7,
                    replacement=[1550, 918, 1550],
                ),
            ],
        ),
        (
            [1, 32000, 32000, 32000, 32000, 32000, 1550, 918, 1550],
            [
                _PlaceholderInfo(
                    modality="pattern_1",
                    start_idx=1,
                    replacement=[32000, 32000],
                ),
                _PlaceholderInfo(
                    modality="pattern_1",
                    start_idx=3,
                    replacement=[32000, 32000],
                ),
                _PlaceholderInfo(
                    modality="pattern_3",
                    start_idx=6,
                    replacement=[1550, 918, 1550],
                ),
            ],
        ),
    ]
)
def test_iter_placeholders(
    repl_by_key,
    prompt,
    expected,
):
    # Should not be used since there is nothing to convert to tokens
    mock_tokenizer = cast(AnyTokenizer, object())

    prompt_repls = [
        PromptReplacement(key, [], repl).bind(mock_tokenizer)
        for key, repl in repl_by_key.items()
    ]

    result = list(
        iter_placeholders(
            prompt_repls,
            prompt,
            # Effectively match all occurrences in the prompt
            {key: 3 for key in repl_by_key},
         ))

    # Only displayed on error
    print("result:", result)

    # Manually constructed results
    assert result == expected


def _rand_img(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def _rand_video(
    rng: np.random.RandomState,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    num_frames = rng.randint(max_frames)
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)



@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
def test_processing_cache_correctness_mixed_modality(model_id, hit_rate):
    from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor

    model_config = ModelConfig(model_id,
                               task="auto",
                               tokenizer=model_id,
                               tokenizer_mode="auto",
                               trust_remote_code=False,
                               seed=0,
                               dtype="float16",
                               revision=None)

    ctx = InputProcessingContext(
        model_config,
        cached_get_tokenizer(model_config.tokenizer),
    )
    baseline_processor = Qwen2VLMultiModalProcessor(
        ctx,
        cache=None,
        enable_sanity_checks=True,
    )
    cached_processor = Qwen2VLMultiModalProcessor(
        ctx,
        # Ensure that the cache capacity is sufficient to contain all inputs
        cache=ProcessingCache(1 << 30),
        enable_sanity_checks=True,
    )

    rng = np.random.RandomState(0)

    hf_processor =  baseline_processor._get_hf_processor()
    image_token = hf_processor.image_token
    video_token = hf_processor.video_token
    repeating_image = Image.new("RGB", size=(128, 128))
    repeating_video = [repeating_image] * 8

    baseline_results = []
    cached_results = []
    for batch_idx in range(10):
        images = []
        for image_idx in range(rng.randint(3)):
            if rng.rand() < hit_rate:
                image = repeating_image
            else:
                image = _rand_img(rng, min_wh=128, max_wh=256)

            images.append(image)

        videos = []
        for video_idx in range(rng.randint(1)):
            if rng.rand() < hit_rate:
                video = repeating_video
            else:
                video = _rand_video(rng, max_frames=8, min_wh=128, max_wh=256)

            videos.append(video)

        prompt = "".join(["Images: ", image_token * len(images),
                          "\nVideos: ", video_token * len(videos)])
        mm_data = {"image": images, "video": videos}

        # Drop unnecessary keys and test single -> multi conversion
        if rng.rand() < 1:
            if not images:
                del mm_data["image"]
            elif len(images) == 1:
                mm_data["image"] = mm_data["image"][0]

            if not videos:
                del mm_data["video"]
            elif len(videos) == 1:
                mm_data["video"] = mm_data["video"][0]

        baseline_result = baseline_processor.apply(
            prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )
        baseline_results.append(baseline_result)

        cached_result = cached_processor.apply(
            prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs={},
        )
        cached_results.append(cached_result)

    assert baseline_results == cached_results
