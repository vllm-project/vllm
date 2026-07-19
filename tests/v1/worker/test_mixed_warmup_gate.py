# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mixed prefill+decode warmup behavior."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.distributed.parallel_state as parallel_state
from vllm.model_executor.warmup import (
    flashinfer_sparse_mla_warmup as sparse_mla_warmup,
)
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

pytestmark = pytest.mark.skip_global_cleanup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_sparse_mla_autotune_forwards_skip_ops(monkeypatch, tmp_path):
    captured = []

    @contextlib.contextmanager
    def fake_autotune(tune_mode, **kwargs):
        captured.append((tune_mode, kwargs))
        yield

    class Backend:
        @staticmethod
        def get_name():
            return "FLASHINFER_MLA_SPARSE_DSV4"

    class AutotunerModule(ModuleType):
        AutoTuner = object

    runner = SimpleNamespace(
        attn_groups=[[SimpleNamespace(backend=Backend())]],
        vllm_config=SimpleNamespace(use_v2_model_runner=False),
        _dummy_run=Mock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
        ),
    )
    world = SimpleNamespace(
        rank_in_group=0,
        broadcast_object=lambda value, src: value,
        barrier=lambda: None,
    )
    autotuner_module = AutotunerModule("flashinfer.autotuner")

    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner_module)
    monkeypatch.setattr(sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        sparse_mla_warmup,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda capability: True),
    )
    monkeypatch.setattr(sparse_mla_warmup, "flashinfer_autotune", fake_autotune)
    monkeypatch.setattr(
        sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: tmp_path / "cache",
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)

    skip_ops = {
        "trtllm::fused_moe::gemm1",
        "trtllm::fused_moe::gemm2",
    }
    assert sparse_mla_warmup._run_flashinfer_sparse_mla_decode_autotune(
        worker,
        16,
        sparse_mla_warmup._DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS,
        skip_ops,
    )
    assert captured == [
        (
            True,
            {
                "cache": str(tmp_path / "cache"),
                "skip_ops": skip_ops,
            },
        )
    ]
