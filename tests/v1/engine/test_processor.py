# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         VllmConfig)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.engine.processor import Processor


@pytest.fixture
def opt_125m_huggingface_id():
    return "facebook/opt-125m"


@pytest.fixture
def fake_lora_adapter_files(tmp_path, opt_125m_huggingface_id):
    adapter_config = {
        "base_model_name_or_path": opt_125m_huggingface_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    }
    with open(tmp_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f)

    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "added_tokens_decoder": {
            "50300": {
                "content": "_A_",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
                "special": False
            },
            "50301": {
                "content": "_B_",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
                "special": False
            },
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "chat_template":
        "{{ bos_token }}{% for message in messages %}{% endfor %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "</s>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False
    }
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    base_vocab = {f"tok_{i}": i for i in range(3, 50300)}
    added_vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "_A_": 50300,
        "_B_": 50301,
    }
    vocab = {**base_vocab, **added_vocab}
    tokenizer = {
        "version":
        "1.0",
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": 1,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": 2,
                "content": "</s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": 50300,
                "content": "_A_",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
                "special": False
            },
            {
                "id": 50301,
                "content": "_B_",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
                "special": False
            },
        ],
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(tmp_path / "tokenizer.json", "w") as f:
        json.dump(tokenizer, f)

    with open(tmp_path / "adapter_model.bin", "wb") as f:
        f.write(b"")

    return tmp_path


def test_allowed_token_ids_with_lora_vocab(opt_125m_huggingface_id,
                                           fake_lora_adapter_files):
    model = opt_125m_huggingface_id
    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    device_config = DeviceConfig()

    lora_config = LoRAConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        device_config=device_config,
        lora_config=lora_config,
    )

    tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
        scheduler_config=vllm_config.scheduler_config,
        lora_config=vllm_config.lora_config)
    processor = Processor(vllm_config, tokenizer)

    # We define tokens 50300, 50301 in our fake lora adapter
    lora_token_ids = [50300, 50301]
    sampling_params = SamplingParams(allowed_token_ids=lora_token_ids)
    lora_request = LoRARequest("1", 1, str(fake_lora_adapter_files))
    processor._validate_sampling_params(sampling_params, lora_request)
