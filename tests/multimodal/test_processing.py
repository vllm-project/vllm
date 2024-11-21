import pytest
from transformers import PreTrainedTokenizerBase

from vllm.multimodal.processing import (find_token_match_by_text,
                                        iter_token_runs)
from vllm.multimodal.utils import cached_get_tokenizer


# yapf: disable
@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], []),
        ([32000, 32000, 32000], [(32000, { "offset": 0, "length": 3 })]),
        (
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [
                (9833, { "offset": 0, "length": 1 }),
                (28747, { "offset": 1, "length": 1 }),
                (32000, { "offset": 2, "length": 3 }),
                (9833, { "offset": 5, "length": 1 }),
                (28747, { "offset": 6, "length": 1 }),
                (32000, { "offset": 7, "length": 2 }),
                (918, { "offset": 9, "length": 1 }),
            ],
        ),
    ],
)
# yapf: enable
def test_iter_token_runs(token_ids, expected):
    result = list(iter_token_runs(token_ids))

    # Invariants
    assert sum(run_info["length"] for _, run_info in result) == len(token_ids)

    # Manually constructed results
    assert result == expected


@pytest.mark.parametrize("tokenizer_id", [
    "llava-hf/llava-1.5-7b-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
])
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize(
    "text",
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
    "match_str",
    [
        # No match
        "No",
        # Has match
        "i",
        "What",
        "What is",
        "image",
        "image?",
        "<image>",
        "<|image|>",
        "<image_1>",
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<s>",
        "</s>",
    ])
def test_token_match_by_text(
    tokenizer_id,
    add_special_tokens,
    text,
    match_str,
):
    tokenizer = cached_get_tokenizer(tokenizer_id)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    match = find_token_match_by_text(tokenizer, token_ids, text, match_str)

    # These are only shown in the output if the test fails
    print("token_ids:", token_ids)
    print("match:", match)

    # Invariants
    if (match_str in text
            or match_str in tokenizer.decode(token_ids,
                                             skip_special_tokens=False)):
        assert match is not None
        match_start_idx, match_end_idx, *_ = match

        assert match_str in tokenizer.decode(
            token_ids[match_start_idx:match_end_idx],
            skip_special_tokens=False,
        )
        assert match_str not in tokenizer.decode(
            token_ids[match_start_idx + 1:match_end_idx],
            skip_special_tokens=False,
        )
        assert match_str not in tokenizer.decode(
            token_ids[match_start_idx:match_end_idx - 1],
            skip_special_tokens=False,
        )
    else:
        assert match is None
