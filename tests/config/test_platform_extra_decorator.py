# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import pytest

from vllm.config.cache import CacheConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config.utils import config, get_hash_factors, replace, update_config
from vllm.config.vllm import VllmConfig


def _make_vllm_config() -> VllmConfig:
    return VllmConfig(
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
    v = _make_vllm_config()

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


def test_platform_extra_set_dispatches_to_existing_config_targets():
    v = _make_vllm_config()

    v.set_platform_extra("cache_config.oot_mode", "turbo")
    v.set_platform_extra("device_config.clock_speed", 1500)
    v.set_platform_extra("scheduler_config.custom_scheduler_policy", "LIFO")

    assert v.cache_config.platform_extra == {"oot_mode": "turbo"}
    assert v.device_config.platform_extra == {"clock_speed": 1500}
    assert v.scheduler_config.platform_extra == {"custom_scheduler_policy": "LIFO"}


def test_platform_extra_set_survives_replace_update_and_hash():
    v = _make_vllm_config()
    v.set_platform_extra("cache_config.oot_mode", "turbo")

    replaced = replace(v, load_config=replace(v.load_config))
    updated = update_config(v, {"load_config": {}})
    factors = get_hash_factors(v.cache_config, set())

    assert replaced.cache_config.platform_extra == {"oot_mode": "turbo"}
    assert updated.cache_config.platform_extra == {"oot_mode": "turbo"}
    assert factors["platform_extra"] == (("oot_mode", "turbo"),)


def test_platform_extra_rejects_unsupported_unstable_values():
    class UnsupportedValue:
        pass

    c = CacheConfig(block_size=16)
    c.platform_extra["bad"] = UnsupportedValue()

    with pytest.raises(TypeError, match="unsupported type"):
        get_hash_factors(c, set())


def test_platform_extra_read_does_not_materialize_state():
    c = CacheConfig(block_size=16)

    assert "_platform_extra" not in c.__dict__

    _ = c.platform_extra

    assert "_platform_extra" not in c.__dict__


@pytest.mark.parametrize("mutator", [replace, update_config])
def test_platform_extra_rejects_unsupported_values_deterministically(mutator):
    c = CacheConfig(block_size=16)

    with pytest.raises(TypeError, match="platform_extra"):
        if mutator is replace:
            mutator(c, platform_extra={"unsafe": object()})
        else:
            mutator(c, {"platform_extra": {"unsafe": object()}})



def test_platform_extra_attribute_fallback():
    c = CacheConfig(block_size=16)
    
    # 2. missing key raises AttributeError
    with pytest.raises(AttributeError, match="'CacheConfig' object has no attribute 'missing'"):
        _ = c.missing

    # 1. read-only attribute fallback from platform_extra
    c.platform_extra["oot_policy"] = "turbo"
    assert c.oot_policy == "turbo"
    
    # 3. real field name collisions do not get shadowed by platform_extra
    c.platform_extra["block_size"] = 99
    assert c.block_size == 16
    
    # 4. worker/pickle path still works if attribute is read after restore
    c_pickled = pickle.loads(pickle.dumps(c))
    assert c_pickled.oot_policy == "turbo"
    assert c_pickled.block_size == 16
    with pytest.raises(AttributeError):
        _ = c_pickled.missing

