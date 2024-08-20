import pytest
from transformers import AutoConfig, AutoTokenizer

from vllm.multimodal.image import repeat_and_pad_image_tokens


@pytest.mark.parametrize("model", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_repeat_and_pad_image_tokens(model):
    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)

    test_cases = [
        ("<image>", 2, "<image><image>", [32000, 32000]),
        ("<image><image>", 2, "<image><image><image>", [32000, 32000, 32000]),
        ("<image><image>", [3, 2], "<image><image><image><image><image>",
         [32000, 32000, 32000, 32000, 32000]),
        ("Image:<image>Image:<image>!", [3, 2],
         "Image:<image><image><image>Image:<image><image>!",
         [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918]),
        ("<image>", [3, 2], "<image><image><image>", [32000, 32000, 32000]),
    ]

    for prompt, repeat_count, expected_prompt, expected_token_ids in test_cases:
        new_prompt, new_token_ids = repeat_and_pad_image_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_token_ids=tokenizer.encode(prompt,
                                              add_special_tokens=False),
            image_token_id=image_token_id,
            repeat_count=repeat_count,
        )
        assert new_prompt == expected_prompt
        assert new_token_ids == expected_token_ids
