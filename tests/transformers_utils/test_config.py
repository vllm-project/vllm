# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_rwkv7_raw_pth_generation_config_uses_model_config(tmp_path, monkeypatch):
    from transformers import GenerationConfig

    model_path = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    model_path.write_bytes(b"")

    def fail_from_pretrained(*args, **kwargs):
        raise AssertionError("raw RWKV .pth should not be parsed as JSON")

    monkeypatch.setattr(GenerationConfig, "from_pretrained", fail_from_pretrained)

    generation_config = try_get_generation_config(
        str(model_path),
        trust_remote_code=False,
    )

    assert generation_config is not None
