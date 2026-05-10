# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle

import pytest

from vllm.config.cache import CacheConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms.interface import Platform, PlatformEnum


class _OOTPlatformStub(Platform):
    _enum = PlatformEnum.OOT
    device_name = "StubDevice"
    device_type: str = "privateuseone"
    dispatch_key: str = "PrivateUse1"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        oot_vals = vllm_config.additional_config.get("oot_platform_params", {})

        dma_buffer = oot_vals.get("dma_buffer")
        if dma_buffer is not None:
            vllm_config.set_platform_extra("cache_config.dma_buffer", dma_buffer)

        hw_id = oot_vals.get("hardware_id")
        if hw_id is not None:
            vllm_config.set_platform_extra("hardware_id", hw_id)

        scheduler_policy = oot_vals.get("scheduler_policy")
        if scheduler_policy is not None:
            vllm_config.set_platform_extra("scheduler_config.oot_policy", scheduler_policy)


def _make_minimal_vllm_config(additional_config=None) -> VllmConfig:
    return VllmConfig(
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=256,
            max_num_seqs=32,
            max_model_len=512,
            is_encoder_decoder=False,
        ),
        load_config=LoadConfig(),
        additional_config=additional_config or {},
    )


def test_oot_platform_check_and_update_reads_cli_derived_values():
    cli_derived = {
        "oot_platform_params": {
            "dma_buffer": 4096,
            "hardware_id": "OOT-GPU-42",
            "scheduler_policy": "LIFO",
        }
    }

    vllm_config = _make_minimal_vllm_config(additional_config=cli_derived)
    _OOTPlatformStub.check_and_update_config(vllm_config)

    assert vllm_config.cache_config.platform_extra.get("dma_buffer") == 4096, (
        "CLI-derived dma_buffer must be propagated into cache_config.platform_extra "
        "by the OOT platform's check_and_update_config. "
        "Missing: path-based dispatch from additional_config to sub-config platform_extra."
    )
    assert vllm_config.platform_extra.get("hardware_id") == "OOT-GPU-42", (
        "CLI-derived hardware_id must be propagated into top-level platform_extra."
    )
    assert vllm_config.scheduler_config.platform_extra.get("oot_policy") == "LIFO", (
        "CLI-derived scheduler_policy must be propagated into scheduler_config.platform_extra."
    )


def test_oot_platform_extra_survives_pickle_worker_boundary():
    cli_derived = {
        "oot_platform_params": {
            "dma_buffer": 8192,
            "hardware_id": "OOT-ACCEL-7",
        }
    }

    vllm_config = _make_minimal_vllm_config(additional_config=cli_derived)
    _OOTPlatformStub.check_and_update_config(vllm_config)

    worker_config = pickle.loads(pickle.dumps(vllm_config))

    assert worker_config.cache_config.platform_extra.get("dma_buffer") == 8192, (
        "After pickle round-trip (worker boundary), cache_config.platform_extra['dma_buffer'] "
        "must equal the CLI-derived value. "
        "Missing: end-to-end propagation from CLI -> check_and_update_config -> platform_extra -> pickle."
    )
    assert worker_config.platform_extra.get("hardware_id") == "OOT-ACCEL-7", (
        "After pickle round-trip, top-level platform_extra['hardware_id'] must survive."
    )


def test_oot_platform_extra_no_phantom_injection():
    vllm_config = _make_minimal_vllm_config(additional_config={})
    _OOTPlatformStub.check_and_update_config(vllm_config)

    assert "dma_buffer" not in vllm_config.cache_config.platform_extra, (
        "dma_buffer must not appear in cache_config.platform_extra when not provided via CLI."
    )
    assert "hardware_id" not in vllm_config.platform_extra, (
        "hardware_id must not appear in top-level platform_extra when not provided via CLI."
    )


def test_engine_args_additional_config_to_worker_propagation():
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        additional_config={
            "oot_platform_params": {
                "dma_buffer": 2048,
                "hardware_id": "CLI-OOT-1",
            }
        },
    )

    vllm_config = _make_minimal_vllm_config(
        additional_config=engine_args.additional_config
    )
    _OOTPlatformStub.check_and_update_config(vllm_config)

    worker_config = pickle.loads(pickle.dumps(vllm_config))

    assert worker_config.cache_config.platform_extra.get("dma_buffer") == 2048, (
        "CLI-derived dma_buffer (via EngineArgs.additional_config) must be visible "
        "to workers after pickle round-trip. "
        "This is the core Engine->Worker propagation requirement."
    )
    assert worker_config.platform_extra.get("hardware_id") == "CLI-OOT-1", (
        "CLI-derived hardware_id must survive the Engine->Worker boundary."
    )


def test_platform_dispatch_platform_extra_helper_exists():
    from vllm.platforms.interface import Platform

    assert hasattr(Platform, "dispatch_platform_extra"), (
        "Platform must expose a dispatch_platform_extra(vllm_config, path, value) "
        "class method so OOT platforms can propagate CLI-derived values into "
        "sub-config platform_extra fields without manual path traversal. "
        "This is the MISSING standardized path-dispatch mechanism."
    )


def test_vllm_config_set_platform_extra_by_path():
    vllm_config = _make_minimal_vllm_config()

    assert hasattr(vllm_config, "set_platform_extra"), (
        "VllmConfig must expose a set_platform_extra(path, value) method "
        "so OOT platforms can use path-based dispatch (e.g., 'cache_config.dma_buffer') "
        "instead of manual attribute traversal. "
        "This is the MISSING path-dispatch API on VllmConfig."
    )

    vllm_config.set_platform_extra("cache_config.dma_buffer", 4096)
    assert vllm_config.cache_config.platform_extra.get("dma_buffer") == 4096, (
        "set_platform_extra('cache_config.dma_buffer', 4096) must set "
        "vllm_config.cache_config.platform_extra['dma_buffer'] = 4096."
    )

    vllm_config.set_platform_extra("hardware_id", "OOT-GPU-1")
    assert vllm_config.platform_extra.get("hardware_id") == "OOT-GPU-1", (
        "set_platform_extra('hardware_id', value) must set top-level platform_extra."
    )

    worker_config = pickle.loads(pickle.dumps(vllm_config))
    assert worker_config.cache_config.platform_extra.get("dma_buffer") == 4096
    assert worker_config.platform_extra.get("hardware_id") == "OOT-GPU-1"

def test_invalid_dispatch_path_raises_error():
    vllm_config = _make_minimal_vllm_config()
    with pytest.raises(AttributeError, match="VllmConfig has no attribute 'invalid_config'"):
        vllm_config.set_platform_extra("invalid_config.foo", 123)



def test_non_oot_default_no_platform_extra_materialized():
    vllm_config = _make_minimal_vllm_config(additional_config={})

    assert "oot_platform_params" not in vllm_config.additional_config, (
        "Non-OOT path must not inject oot_platform_params into additional_config."
    )
    assert vllm_config.platform_extra == {}, (
        "Top-level platform_extra must be empty on the non-OOT default path."
    )
    assert vllm_config.cache_config.platform_extra == {}, (
        "cache_config.platform_extra must be empty on the non-OOT default path."
    )
    assert vllm_config.scheduler_config.platform_extra == {}, (
        "scheduler_config.platform_extra must be empty on the non-OOT default path."
    )


def test_non_oot_default_pickle_round_trip_clean():
    import pickle

    vllm_config = _make_minimal_vllm_config(additional_config={})
    worker_config = pickle.loads(pickle.dumps(vllm_config))

    assert worker_config.platform_extra == {}, (
        "After pickle round-trip on non-OOT config, top-level platform_extra must be empty."
    )
    assert worker_config.cache_config.platform_extra == {}, (
        "After pickle round-trip on non-OOT config, cache_config.platform_extra must be empty."
    )
