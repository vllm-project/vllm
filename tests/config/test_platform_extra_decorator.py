# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

from vllm.config.cache import CacheConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config.utils import config, get_hash_factors, replace, update_config
from vllm.config.vllm import VllmConfig


def test_platform_extra_in_cache_config():
    c_default = CacheConfig(block_size=16)
    assert c_default.platform_extra == {}

    c = CacheConfig(block_size=32)
    c.platform_extra["oot_mode"] = "turbo"
    c.platform_extra["dma_buffer"] = 1024

    factors = get_hash_factors(c, set())
    assert "platform_extra" in factors
    assert factors["platform_extra"] == (("dma_buffer", 1024), ("oot_mode", "turbo"))

    c_pickled = pickle.loads(pickle.dumps(c))
    assert c_pickled.platform_extra["oot_mode"] == "turbo"
    assert c_pickled.platform_extra["dma_buffer"] == 1024


def test_platform_extra_in_vllm_config():
    v = VllmConfig(
        cache_config=CacheConfig(block_size=32),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=100,
            max_num_seqs=50,
            max_model_len=1024,
            is_encoder_decoder=False,
        ),
        load_config=LoadConfig(),
    )

    v.platform_extra["global_hardware_id"] = "OOT-GPU-1"
    v.device_config.platform_extra["clock_speed"] = 1500
    v.scheduler_config.platform_extra["custom_scheduler_policy"] = "LIFO"

    v_worker = pickle.loads(pickle.dumps(v))

    assert v_worker.platform_extra["global_hardware_id"] == "OOT-GPU-1"
    assert v_worker.device_config.platform_extra["clock_speed"] == 1500
    assert v_worker.scheduler_config.platform_extra["custom_scheduler_policy"] == "LIFO"


def test_platform_extra_inheritance():
    @config
    class BaseConfig:
        base_val: int

    @config
    class SubConfig(BaseConfig):
        sub_val: int

    s = SubConfig(base_val=1, sub_val=2)
    s.platform_extra["injected"] = True
    assert s.base_val == 1
    assert s.sub_val == 2
    assert s.platform_extra["injected"] is True

    # Validate MRO logic: the shared extension point is inherited once.
    import dataclasses

    fields = [f.name for f in dataclasses.fields(SubConfig)]
    assert "platform_extra" not in fields


def test_platform_extra_replace_preserves_state():
    c = CacheConfig(block_size=16)
    c.platform_extra["oot_mode"] = "turbo"

    updated = replace(c, block_size=32)

    assert updated.block_size == 32
    assert updated.platform_extra == {"oot_mode": "turbo"}


def test_platform_extra_replace_override():
    c = CacheConfig(block_size=16)
    c.platform_extra["oot_mode"] = "turbo"

    updated = replace(c, platform_extra={"oot_mode": "safe", "lanes": 2})

    assert updated.platform_extra == {"oot_mode": "safe", "lanes": 2}


def test_platform_extra_update_config_override():
    c = CacheConfig(block_size=16)
    c.platform_extra["oot_mode"] = "turbo"

    updated = update_config(c, {"platform_extra": {"oot_mode": "safe"}})

    assert updated.platform_extra == {"oot_mode": "safe"}
