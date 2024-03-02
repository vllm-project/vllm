# This unit test should be moved to a new
# tests/test_guided_decoding directory.

from transformers import AutoTokenizer
import torch

from vllm.model_executor.guided_logits_processors import (RegexLogitsProcessor,
                                                          JSONLogitsProcessor)

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 10
            },
            "minItems": 3
        },
        "work history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string"
                    },
                    "duration": {
                        "type": "string"
                    },
                    "position": {
                        "type": "string"
                    }
                },
                "required": ["company", "position"]
            }
        }
    },
    "required": ["name", "age", "skills", "work history"]
}

TEST_REGEX = r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}" + \
             r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"


def test_guided_logits_processors():
    """Basic unit test for RegexLogitsProcessor and JSONLogitsProcessor."""
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
    regex_LP = RegexLogitsProcessor(TEST_REGEX, tokenizer)
    json_LP = JSONLogitsProcessor(TEST_SCHEMA, tokenizer)

    regex_LP.init_state()
    token_ids = tokenizer.encode(
        f"Give an example IPv4 address with this regex: {TEST_REGEX}")
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    regex_LP(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    json_LP.init_state()
    token_ids = tokenizer.encode(
        f"Give an employee profile that fits this schema: {TEST_SCHEMA}")
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    json_LP(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)
