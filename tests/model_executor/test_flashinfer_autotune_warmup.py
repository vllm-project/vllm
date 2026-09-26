# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    iter_flashinfer_autotune_cache_files,
    read_flashinfer_autotune_cache_bytes,
    reset_flashinfer_autotuner_state,
    share_flashinfer_autotune_cache,
    write_flashinfer_autotune_cache_aliases,
)
from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
)

pytestmark = pytest.mark.cpu_test

_VLLM_CACHE = "autotune_configs.json"
_FLASHINFER_CACHE = "autotune_config.json"
_CACHE_PAYLOAD = b'{"trtllm::fused_moe::gemm1": ["MoERunner", 0]}'


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


@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(skip_attn):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
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
        _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)

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
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


class _FakeTuner:
    """Stand-in for FlashInfer AutoTuner cache state."""

    def __init__(self, *, file_hits: dict[str, Any] | None = None) -> None:
        self.profiling_cache: dict[str, Any] = {}
        self._file_configs: dict[str, Any] = dict(file_hits or {})
        self.loaded_paths: list[str] = []
        self.load_return: bool = True

    def clear_cache(self) -> None:
        self.profiling_cache.clear()
        self._file_configs.clear()

    def load_configs(self, path: str) -> bool:
        self.loaded_paths.append(path)
        self._file_configs["from_file"] = Path(path).read_bytes()
        return self.load_return


def _canonical_cache_path(tmp_path: Path) -> Path:
    return tmp_path / _VLLM_CACHE


def _broadcast_from_leader(leader_payload: bytes | None):
    """Simulate rank-0 ``broadcast_object`` across sequential rank calls."""
    shared = {"value": leader_payload}

    def broadcast_object(obj: bytes | None) -> bytes | None:
        if obj is not None:
            shared["value"] = obj
        return shared["value"]

    return broadcast_object


def test_iter_cache_files_includes_vllm_and_flashinfer_names(tmp_path):
    cache_path = _canonical_cache_path(tmp_path)
    files = iter_flashinfer_autotune_cache_files(cache_path)
    assert files == (cache_path, tmp_path / _FLASHINFER_CACHE)


def test_read_cache_bytes_falls_back_to_flashinfer_native_filename(tmp_path):
    cache_path = _canonical_cache_path(tmp_path)
    native = tmp_path / _FLASHINFER_CACHE
    native.write_bytes(_CACHE_PAYLOAD)

    assert read_flashinfer_autotune_cache_bytes(cache_path) == _CACHE_PAYLOAD


def test_read_cache_bytes_prefers_vllm_filename_when_both_exist(tmp_path):
    cache_path = _canonical_cache_path(tmp_path)
    cache_path.write_bytes(b'{"vllm": 1}')
    (tmp_path / _FLASHINFER_CACHE).write_bytes(b'{"flashinfer": 1}')

    assert read_flashinfer_autotune_cache_bytes(cache_path) == b'{"vllm": 1}'


def test_write_aliases_keeps_both_filenames_in_sync(tmp_path):
    cache_path = _canonical_cache_path(tmp_path)
    write_flashinfer_autotune_cache_aliases(cache_path, _CACHE_PAYLOAD)

    assert cache_path.read_bytes() == _CACHE_PAYLOAD
    assert (tmp_path / _FLASHINFER_CACHE).read_bytes() == _CACHE_PAYLOAD


def test_share_cache_loads_identical_configs_from_native_only_file(tmp_path):
    """Leader has only autotune_config.json; every rank must load that payload.

    This is the #57423 split: rank 0 hits FlashInfer's native dump while
    peers look at vLLM's autotune_configs.json, miss, and block forever.
    """
    leader_dir = tmp_path / "rank0"
    follower_dir = tmp_path / "rank1"
    leader_dir.mkdir()
    follower_dir.mkdir()
    (leader_dir / _FLASHINFER_CACHE).write_bytes(_CACHE_PAYLOAD)

    leader_path = leader_dir / _VLLM_CACHE
    follower_path = follower_dir / _VLLM_CACHE
    leader_tuner = _FakeTuner(file_hits={"stale": True})
    follower_tuner = _FakeTuner()
    broadcast = _broadcast_from_leader(
        read_flashinfer_autotune_cache_bytes(leader_path)
    )

    assert share_flashinfer_autotune_cache(
        leader_tuner,
        leader_path,
        is_leader=True,
        broadcast_object=broadcast,
    )
    assert share_flashinfer_autotune_cache(
        follower_tuner,
        follower_path,
        is_leader=False,
        broadcast_object=broadcast,
    )

    assert leader_tuner._file_configs["from_file"] == _CACHE_PAYLOAD
    assert follower_tuner._file_configs["from_file"] == _CACHE_PAYLOAD
    assert "stale" not in leader_tuner._file_configs
    for directory in (leader_dir, follower_dir):
        assert (directory / _VLLM_CACHE).read_bytes() == _CACHE_PAYLOAD
        assert (directory / _FLASHINFER_CACHE).read_bytes() == _CACHE_PAYLOAD


def test_share_cache_clears_asymmetric_hits_when_no_file_exists(tmp_path):
    """No on-disk cache: drop rank-local hits so every rank misses together."""
    cache_path = _canonical_cache_path(tmp_path)
    leader_tuner = _FakeTuner(file_hits={"rank0-only": True})
    leader_tuner.profiling_cache["rank0-only"] = True
    follower_tuner = _FakeTuner()
    broadcast = _broadcast_from_leader(None)

    assert not share_flashinfer_autotune_cache(
        leader_tuner,
        cache_path,
        is_leader=True,
        broadcast_object=broadcast,
    )
    assert not share_flashinfer_autotune_cache(
        follower_tuner,
        cache_path,
        is_leader=False,
        broadcast_object=broadcast,
    )

    assert leader_tuner._file_configs == {}
    assert leader_tuner.profiling_cache == {}
    assert follower_tuner._file_configs == {}
    assert leader_tuner.loaded_paths == []
    assert follower_tuner.loaded_paths == []


def test_reset_autotuner_state_falls_back_without_clear_cache():
    tuner = SimpleNamespace(
        profiling_cache={"hit": 1},
        _file_configs={"hit": 1},
        _ranked_tactics_cache={"hit": 1},
    )
    reset_flashinfer_autotuner_state(tuner)
    assert tuner.profiling_cache == {}
    assert tuner._file_configs == {}
    assert tuner._ranked_tactics_cache == {}
