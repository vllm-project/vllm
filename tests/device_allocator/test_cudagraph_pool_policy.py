# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cuMem CUDA graph pool policy is one VllmConfig property; the NCCL env
check depends only on it."""

import os
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

_ON = dict(v2=True, sleep=True, backend="cumem", mode=CUDAGraphMode.FULL_AND_PIECEWISE)


def _cfg(v2, sleep, backend, mode, model=True):
    model_config = (
        SimpleNamespace(enable_sleep_mode=sleep, sleep_mode_backend=backend)
        if model
        else None
    )
    return SimpleNamespace(
        use_v2_model_runner=v2,
        model_config=model_config,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="the pool is CUDA-only")
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"mode": CUDAGraphMode.FULL}, True),
        ({"mode": CUDAGraphMode.PIECEWISE}, True),
        ({"v2": False}, False),
        ({"sleep": False}, False),
        ({"backend": "other"}, False),
        ({"mode": CUDAGraphMode.NONE}, False),
        ({"model": False}, False),
    ],
    ids=[
        "default",
        "full",
        "piecewise",
        "v1",
        "no-sleep",
        "backend",
        "eager",
        "nomodel",
    ],
)
def test_use_cumem_cudagraph_pool(overrides, expected):
    cfg = _cfg(**{**_ON, **overrides})
    assert VllmConfig.use_cumem_cudagraph_pool.fget(cfg) is expected


@pytest.mark.parametrize(
    ("pool", "env", "expected_register", "raises"),
    [
        (True, {}, "0", False),
        (True, {"NCCL_GRAPH_REGISTER": "0"}, "0", False),
        (True, {"NCCL_GRAPH_REGISTER": "1"}, "1", True),
        (False, {}, None, False),
        (False, {"NCCL_GRAPH_REGISTER": "1"}, "1", False),
    ],
    ids=["unset->0", "explicit-0", "explicit-1", "off-untouched", "off-allowed"],
)
def test_nccl_graph_register_policy(monkeypatch, pool, env, expected_register, raises):
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = SimpleNamespace(use_cumem_cudagraph_pool=pool)
    if raises:
        with pytest.raises(ValueError, match="NCCL_GRAPH_REGISTER"):
            VllmConfig._verify_cumem_cudagraph_pool_env(cfg)
    else:
        VllmConfig._verify_cumem_cudagraph_pool_env(cfg)
    assert os.environ.get("NCCL_GRAPH_REGISTER") == expected_register


def test_engine_config_applies_policy(monkeypatch):
    """VllmConfig validation runs the env policy: building a V2 sleep-mode
    engine config defaults NCCL_GRAPH_REGISTER to 0."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    config = EngineArgs(
        "hmellor/tiny-random-LlamaForCausalLM", enable_sleep_mode=True
    ).create_engine_config()
    assert config.use_cumem_cudagraph_pool is current_platform.is_cuda()
    assert os.environ.get("NCCL_GRAPH_REGISTER") == (
        "0" if current_platform.is_cuda() else None
    )
