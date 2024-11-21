from typing import cast

import pytest
from transformers import BatchFeature

from vllm.config import ModelConfig
from vllm.multimodal.processing import (InputProcessingContext,
                                        ModalityProcessingMetadata,
                                        MultiModalProcessor, PromptReplacement,
                                        iter_token_matches, iter_token_runs)
from vllm.multimodal.utils import cached_get_tokenizer


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

    # Manually constructed results
    assert result == expected

    # Invariants
    assert sum(run_info["length"] for run_info in result) == len(token_ids)


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
            [
                { "start_idx": 0, "end_idx": 2 },
                { "start_idx": 1, "end_idx": 3 },
            ],
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


@pytest.mark.parametrize("tokenizer_id", [
    "llava-hf/llava-1.5-7b-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
])
@pytest.mark.parametrize(
    "prompt_str",
    [
        "What is in this image?",
        # LLaVA
        "<image>What is in this image?",
        "What is<image>in this image?",
        "What is in this image?<image>",
        # LLama-3.2
        "<|image|>What is in this image?",
        "What is<|image|>in this image?",
        "What is in this image?<|image|>",
        # Phi-3-vision
        "<image_1>What is in this image?",
        "What is<image_1>in this image?",
        "What is in this image?<image_1>",
        # Qwen2-VL
        "<|vision_start|><|image_pad|><|vision_end|>What is in this image?",
        "What is<|vision_start|><|image_pad|><|vision_end|>in this image?",
        "What is in this image?<|vision_start|><|image_pad|><|vision_end|>",
    ])
@pytest.mark.parametrize(
    "repl_target_str",
    [
        # No match
        "No",
        # Has match
        "i",
        "image",
        "image?",
        "<image>",
        "<|image|>",
        "<image_1>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<s>",
    ])
@pytest.mark.parametrize(
    "repl_unit_str",
    [
        # No match
        "No",
        # Has match
        "i",
        "image",
        "image?",
        "<image>",
        "<|image|>",
        "<image_1>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<s>",
    ])
@pytest.mark.parametrize("repl_count", [0, 1, 2])
@pytest.mark.parametrize("mm_count", [0, 1, 2])
def test_processor_prompt_replacements(
    tokenizer_id,
    prompt_str,
    repl_target_str,
    repl_unit_str,
    repl_count,
    mm_count,
):
    model_config = cast(ModelConfig, object())
    tokenizer = cached_get_tokenizer(tokenizer_id)

    ctx = InputProcessingContext(model_config, tokenizer)
    metadata = ModalityProcessingMetadata(prompt_repls=[
        PromptReplacement(repl_target_str, repl_unit_str,
                          lambda *args, **kwargs: repl_count),
    ])
    modality = "image"

    dummy_metadata = {modality: metadata}
    dummy_mm_data = {modality: list(range(mm_count))}
    prompt_ids = tokenizer.encode(prompt_str)

    processor = MultiModalProcessor(ctx, dummy_metadata)
    new_token_ids, placeholder_ranges = processor._apply_prompt_replacements(
        dummy_mm_data,
        BatchFeature(),
        prompt_ids,
    )

    repl_unit_ids = tokenizer.encode(repl_unit_str)

    # Only displayed on error
    print("prompt_ids:", prompt_ids)
    print("repl_unit_ids:", repl_unit_ids)
    print("new_token_ids:", new_token_ids)
    print("placeholder_ranges:", placeholder_ranges)

    # Invariants
    if mm_count == 0:
        assert new_token_ids == prompt_ids

    assert len(placeholder_ranges) <= mm_count

    for placeholder_range in placeholder_ranges:
        repl_offset = placeholder_range["offset"]
        repl_len = placeholder_range["length"]

        assert new_token_ids[repl_offset:repl_offset +
                             repl_len] == repl_unit_ids * repl_count
