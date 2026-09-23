# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch

import vllm.model_executor.warmup.kernel_warmup as warmup
import vllm.utils.flashinfer as fi_utils
from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend)
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ],
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 32, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run():
    runner = _make_runner([])
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner)

    assert get_buckets.call_args_list == [call(8192), call(128)]
    assert autotune.call_args_list == [
        call(tuning_buckets=max_buckets),
        call(tuning_buckets=deferred_buckets),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
        ),
    ]


@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("skip_ops", [[], ["fp4_gemm", "bmm_fp8"]])
@pytest.mark.parametrize("failure", [None, "dummy", "replayssm", "kimi"])
def test_flashinfer_autotune_lifecycle(
    monkeypatch, tmp_path, use_v2, world_size, rank, skip_ops, failure
):
    runner = _make_runner([])
    model = runner.get_model()
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    events = Mock()
    events.broadcast.return_value = b"cached"
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        cpu_group=object(),
        broadcast_object=events.broadcast,
        barrier=events.barrier,
    )

    @contextmanager
    def autotune_context(**kwargs):
        assert torch.is_inference_mode_enabled()
        events.enter()
        try:
            yield
        finally:
            events.exit()

    events.autotune.side_effect = autotune_context
    if failure is not None:
        getattr(events, failure).side_effect = RuntimeError("warmup failed")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(
            AutoTuner=SimpleNamespace(get=lambda: events.tuner),
            set_autotune_process_group=events.set_group,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=events.autotune, autotune_v2_reload=events.reload),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: use_v2)
    monkeypatch.setattr(fi_utils, "autotune", events.autotune)
    monkeypatch.setattr(warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", skip_ops)
    resolve_file = Mock(return_value=cache_path)
    resolve_root = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_file", resolve_file)
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_v2_root", resolve_root)
    monkeypatch.setattr(warmup, "write_flashinfer_autotune_cache", events.write)
    monkeypatch.setattr(warmup, "_run_flashinfer_autotune_dummy_runs", events.dummy)
    monkeypatch.setattr(warmup, "replayssm_autotune_warmup", events.replayssm)
    monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", events.kimi)

    if failure is None:
        warmup.flashinfer_autotune(runner)
    else:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.flashinfer_autotune(runner)

    kwargs = {"skip_ops": set(skip_ops)} if skip_ops else {}
    expected = []
    if use_v2:
        resolve_file.assert_not_called()
        resolve_root.assert_called_once_with()
        expected.append(call.autotune(mode="tune", cache_root=str(tmp_path), **kwargs))
    else:
        resolve_root.assert_not_called()
        resolve_file.assert_called_once_with(runner)
        expected.extend(
            [
                call.broadcast(b"cached" if rank == 0 else None, src=0),
                call.write(cache_path, b"cached"),
                call.barrier(),
                call.tuner.load_configs(str(cache_path)),
                call.autotune(tune_mode=True, **kwargs),
            ]
        )
    expected.extend(
        [call.set_group(world.cpu_group if world_size > 1 else None), call.enter()]
    )
    for name, arg in [("dummy", runner), ("replayssm", runner), ("kimi", model)]:
        expected.append(getattr(call, name)(arg))
        if failure == name:
            break
    expected.extend([call.exit(), call.set_group(None)])
    if failure is None:
        if world_size > 1:
            expected.append(call.barrier())
        if use_v2 and world_size > 1:
            expected.extend([call.reload(), call.barrier()])
        elif not use_v2 and rank == 0:
            expected.append(call.tuner.save_configs(str(cache_path)))
    assert events.mock_calls == expected
