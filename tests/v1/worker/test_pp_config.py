# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import LlamaConfig

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.worker.pp.config import get_pp_config


@pytest.fixture
def config_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=12,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
    ).save_pretrained(tmp_path / "model")

    def build(*, prefix=False, chunk=False, eager=True, size=2, pp=2):
        return VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path / "model"),
                skip_tokenizer_init=True,
                enforce_eager=eager,
                max_model_len=512,
                dtype="float32",
            ),
            device_config=DeviceConfig(device="cpu"),
            parallel_config=ParallelConfig(
                pipeline_parallel_size=pp, distributed_executor_backend="mp"
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=512,
                is_encoder_decoder=False,
                max_num_batched_tokens=512,
                enable_chunked_prefill=chunk,
            ),
            cache_config=CacheConfig(enable_prefix_caching=prefix),
            additional_config={
                "pp_send_buffer_size": size,
                "dynamic_pp_config": {"enabled": True},
            },
        )

    return build


def test_disabled_feature_leaves_default_configuration_unchanged():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    size, dynamic = get_pp_config(config)
    assert size == 0 and not dynamic.enabled


def test_dynamic_config_accepts_the_complete_prefill_runtime(config_factory):
    size, dynamic = get_pp_config(config_factory())
    assert size == 2 and dynamic.enabled


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"prefix": True}, "KV replicas"),
        ({"chunk": True}, "chunked prefill"),
        ({"eager": False}, "eager execution"),
        ({"pp": 3}, "PP=2"),
        ({"size": 1}, "high_watermark"),
        ({"size": True}, "non-negative integer"),
    ],
)
def test_incompatible_runtime_is_rejected(config_factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_pp_config(config_factory(**kwargs))
