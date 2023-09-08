import pytest

from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import detokenize_incrementally


@pytest.fixture(scope="module")
def llama_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


truth_and_input_ids = [
    ("Hello here, this is a simple test",
     [15043, 1244, 29892, 445, 338, 263, 2560, 1243]),
    ("<s> vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It is designed to be used in production environments, where inference and serving",
     [
         1,
         325,
         2208,
         29924,
         338,
         263,
         1880,
         29899,
         20678,
         649,
         322,
         3370,
         29899,
         8462,
         27262,
         322,
         16330,
         6012,
         363,
         365,
         26369,
         29879,
         29889,
         739,
         338,
         8688,
         304,
         367,
         1304,
         297,
         5802,
         23136,
         29892,
         988,
         27262,
         322,
         16330,
     ])
]


@pytest.mark.parametrize("truth_and_input_ids", truth_and_input_ids)
def test_decode_streaming_english_spaces(llama_tokenizer, truth_and_input_ids):
    truth, all_input_ids = truth_and_input_ids
    assert (all_input_ids == llama_tokenizer(
        truth, add_special_tokens=False)["input_ids"])

    decoded_text = ""
    offset = 0
    token_offset = 0
    for i in range(len(all_input_ids)):
        text, offset, token_offset = detokenize_incrementally(
            llama_tokenizer, all_input_ids[:i + 1], offset, token_offset)
        decoded_text += text

    assert decoded_text == truth


def test_decode_streaming_chinese_utf8(llama_tokenizer):
    truth = "我很感谢你的热情"
    all_input_ids = [
        30672,
        232,
        193,
        139,
        233,
        135,
        162,
        235,
        179,
        165,
        30919,
        30210,
        234,
        134,
        176,
        30993,
    ]

    decoded_text = ""
    offset = 0
    token_offset = 0
    for i in range(len(all_input_ids)):
        text, offset, token_offset = detokenize_incrementally(
            llama_tokenizer, all_input_ids[:i + 1], offset, token_offset)
        decoded_text += text

    assert decoded_text == truth
