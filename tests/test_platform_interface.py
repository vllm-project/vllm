# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.parallel_state as parallel_state
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import Platform

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("platform_cls", [Platform, CpuPlatform])
def test_hybrid_alignment_syncs_dynamic_backend_config_across_pp(
    monkeypatch, platform_cls
):
    dynamic_backend = type("DynamicBackend", (), {})
    backends = iter([dynamic_backend, None])
    monkeypatch.setattr(
        Platform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, config: next(backends)),
    )
    monkeypatch.setattr(
        parallel_state,
        "model_parallel_is_initialized",
        lambda: True,
    )

    gathered_local_values = iter([True, False])

    def fake_all_gather_object(output, local_value, group):
        assert local_value is next(gathered_local_values)
        output[:] = [False, True]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )

    synced_config = None

    def source_broadcast(config, src):
        nonlocal synced_config
        assert src == 1
        synced_config = config
        return config

    def receiver_broadcast(config, src):
        assert config is None
        assert src == 1
        return synced_config

    pp_groups = iter(
        [
            SimpleNamespace(
                world_size=2,
                rank_in_group=1,
                cpu_group=object(),
                broadcast_object=source_broadcast,
            ),
            SimpleNamespace(
                world_size=2,
                rank_in_group=0,
                cpu_group=object(),
                broadcast_object=receiver_broadcast,
            ),
        ]
    )
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: next(pp_groups))

    def align_hybrid(config, backend):
        assert backend is dynamic_backend
        config.cache_config.block_size = 512
        config.cache_config.mamba_block_size = 512
        config.cache_config.mamba_page_size_padded = 4096
        config.cache_config.skip_page_size_padded = 8192

    monkeypatch.setattr(
        platform_cls,
        "_align_hybrid_block_size",
        classmethod(lambda cls, config, backend: align_hybrid(config, backend)),
    )

    def make_config():
        return SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=True),
            cache_config=SimpleNamespace(
                block_size=16,
                user_specified_block_size=True,
                kv_cache_dtype_skip_layers=[],
                mamba_block_size=None,
                mamba_page_size_padded=None,
                skip_page_size_padded=None,
            ),
        )

    source_config = make_config()
    platform_cls.update_block_size_for_backend(source_config)

    receiver_config = make_config()
    platform_cls.update_block_size_for_backend(receiver_config)

    assert receiver_config.cache_config.block_size == 512
    assert receiver_config.cache_config.mamba_block_size == 512
    assert receiver_config.cache_config.mamba_page_size_padded == 4096
    assert receiver_config.cache_config.skip_page_size_padded == 8192
