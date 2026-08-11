# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.parallel_state as parallel_state
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import Platform

pytestmark = pytest.mark.skip_global_cleanup


class _RemoteBackend:
    @classmethod
    def full_cls_name(cls):
        return cls.__module__, cls.__qualname__


@pytest.mark.parametrize("platform_cls", [Platform, CpuPlatform])
def test_hybrid_alignment_uses_backend_from_another_pp_rank(monkeypatch, platform_cls):
    monkeypatch.setattr(
        Platform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, config: None),
    )
    monkeypatch.setattr(
        parallel_state,
        "model_parallel_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=object()),
    )

    def fake_all_gather_object(output, local_value, group):
        output[:] = [
            None,
            (_RemoteBackend.__module__, _RemoteBackend.__qualname__),
        ]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )

    align_hybrid = Mock()
    monkeypatch.setattr(
        platform_cls,
        "_align_hybrid_block_size",
        align_hybrid,
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=True,
            kv_cache_dtype_skip_layers=[],
        ),
    )

    platform_cls.update_block_size_for_backend(config)

    align_hybrid.assert_called_once_with(config, _RemoteBackend)
