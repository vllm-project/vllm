# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm.model_executor.warmup import (
    flashinfer_autotune_cache,
    flashinfer_sparse_mla_warmup,
)


def make_runner(backend_names: list[str], **kwargs):
    groups = [
        SimpleNamespace(
            backend=SimpleNamespace(get_name=lambda name=name: name),
        )
        for name in backend_names
    ]
    return SimpleNamespace(attn_groups=[groups], **kwargs)


def test_deepseek_v4_sparse_mla_backend_detection_is_dsv4_scoped() -> None:
    assert flashinfer_sparse_mla_warmup._has_deepseek_v4_sparse_mla_backend(
        make_runner(["FLASHMLA_SPARSE_DSV4"])
    )
    assert flashinfer_sparse_mla_warmup._has_deepseek_v4_sparse_mla_backend(
        make_runner(["FLASHINFER_MLA_SPARSE_DSV4"])
    )
    assert not flashinfer_sparse_mla_warmup._has_deepseek_v4_sparse_mla_backend(
        make_runner(["FLASHINFER_MLA_SPARSE"])
    )


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "aot_compile_hash_factors",
        lambda _: ["env-hash", "config-hash"],
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs, "VLLM_CACHE_ROOT", str(tmp_path)
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        flashinfer_autotune_cache.envs,
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        flashinfer_autotune_cache,
        "aot_compile_hash_factors",
        lambda _: ["env-hash", "config-hash"],
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = flashinfer_autotune_cache.resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"


def test_sparse_mla_autotune_broadcast_writes_nonlocal_rank_cache(
    monkeypatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "pp-rank-1" / "autotune_configs.json"
    tune_results = b'{"cached": true}'
    loaded_paths: list[str] = []

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def load_configs(self, path: str) -> bool:
            loaded_paths.append(path)
            assert Path(path).read_bytes() == tune_results
            return True

    fake_autotuner = SimpleNamespace(AutoTuner=FakeAutoTuner)
    fake_sparse = SimpleNamespace(
        sparse_mla_sm120_decode_dsv4_autotune=lambda cache_path=None: nullcontext()
    )
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(sparse_mla_sm120=fake_sparse)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setitem(sys.modules, "flashinfer.sparse_mla_sm120", fake_sparse)
    monkeypatch.setattr(flashinfer_sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: cache_path,
    )

    class FakeWorld:
        rank_in_group = 5
        local_rank = 5

        def __init__(self) -> None:
            self.barrier_count = 0

        def broadcast_object(self, obj, src: int = 0):
            assert obj is None
            assert src == 0
            return tune_results

        def barrier(self) -> None:
            self.barrier_count += 1

    world = FakeWorld()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )

    dummy_runs = []
    runner = make_runner(
        ["FLASHINFER_MLA_SPARSE_DSV4"],
        _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
        ),
    )

    assert not cache_path.exists()

    assert (
        flashinfer_sparse_mla_warmup._deepseek_v4_sparse_mla_decode_autotune(worker, 16)
        is True
    )

    assert cache_path.read_bytes() == tune_results
    assert loaded_paths == [str(cache_path)]
    assert world.barrier_count == 1
    assert dummy_runs == [
        {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
    ]


def test_sparse_mla_autotune_uses_dsv3_2_context_on_leader(
    monkeypatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "pp-rank-0" / "autotune_configs.json"
    tune_results = b'{"cached": "dsv3_2"}'
    loaded_paths: list[str] = []
    context_calls: list[str] = []

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def load_configs(self, path: str) -> bool:
            loaded_paths.append(path)
            assert Path(path).read_bytes() == tune_results
            return True

    @contextmanager
    def dsv3_2_autotune(cache_path=None):
        assert cache_path == str(cache_path_obj)
        context_calls.append("dsv3_2")
        yield
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_bytes(tune_results)

    def dsv4_autotune(cache_path=None):
        raise AssertionError("DSv3.2 sparse MLA warmup must not use DSv4 autotune")

    cache_path_obj = cache_path
    fake_autotuner = SimpleNamespace(AutoTuner=FakeAutoTuner)
    fake_sparse = SimpleNamespace(
        sparse_mla_sm120_decode_dsv3_2_autotune=dsv3_2_autotune,
        sparse_mla_sm120_decode_dsv4_autotune=dsv4_autotune,
    )
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(sparse_mla_sm120=fake_sparse)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setitem(sys.modules, "flashinfer.sparse_mla_sm120", fake_sparse)
    monkeypatch.setattr(flashinfer_sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: cache_path,
    )

    class FakeWorld:
        rank_in_group = 0
        local_rank = 0

        def __init__(self) -> None:
            self.barrier_count = 0

        def broadcast_object(self, obj, src: int = 0):
            assert obj == tune_results
            assert src == 0
            return obj

        def barrier(self) -> None:
            self.barrier_count += 1

    world = FakeWorld()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )

    dummy_runs = []
    runner = make_runner(
        ["FLASHINFER_MLA_SPARSE"],
        _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
        ),
    )

    assert not cache_path.exists()

    assert (
        flashinfer_sparse_mla_warmup._flashinfer_sparse_mla_decode_autotune(worker, 16)
        is True
    )

    assert context_calls == ["dsv3_2"]
    assert loaded_paths == [str(cache_path)]
    assert world.barrier_count == 1
    assert dummy_runs == [
        {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
    ]


def test_sparse_mla_autotune_v2_uses_execute_model_mixed_warmup(
    monkeypatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "pp-rank-0" / "autotune_configs.json"
    tune_results = b'{"cached": "dsv4-v2"}'
    loaded_paths: list[str] = []
    context_calls: list[tuple[str, int, int]] = []
    execute_outputs: list[Any] = []
    sample_calls: list[Any] = []
    kv_disabled: list[bool] = []

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def load_configs(self, path: str) -> bool:
            loaded_paths.append(path)
            assert Path(path).read_bytes() == tune_results
            return True

    @contextmanager
    def dsv4_autotune(cache_path=None):
        assert cache_path == str(cache_path_obj)
        context_calls.append(("enter", len(execute_outputs), len(sample_calls)))
        yield
        context_calls.append(("exit", len(execute_outputs), len(sample_calls)))
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_bytes(tune_results)

    cache_path_obj = cache_path
    fake_autotuner = SimpleNamespace(AutoTuner=FakeAutoTuner)
    fake_sparse = SimpleNamespace(sparse_mla_sm120_decode_dsv4_autotune=dsv4_autotune)
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(sparse_mla_sm120=fake_sparse)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setitem(sys.modules, "flashinfer.sparse_mla_sm120", fake_sparse)
    monkeypatch.setattr(flashinfer_sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: cache_path,
    )

    class FakeWorld:
        rank_in_group = 0
        local_rank = 0

        def __init__(self) -> None:
            self.barrier_count = 0

        def broadcast_object(self, obj, src: int = 0):
            assert obj == tune_results
            assert src == 0
            return obj

        def barrier(self) -> None:
            self.barrier_count += 1

    world = FakeWorld()
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )

    class FakeKVConnector:
        def set_disabled(self, disabled: bool) -> None:
            kv_disabled.append(disabled)

    def fail_dummy_run(**kwargs):
        raise AssertionError("V2 sparse MLA warmup must not call _dummy_run")

    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=True),
    )

    runner = make_runner(
        ["FLASHINFER_MLA_SPARSE_DSV4"],
        is_pooling_model=False,
        vllm_config=vllm_config,
        kv_cache_config=SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
        ),
        kv_connector=FakeKVConnector(),
        _dummy_run=fail_dummy_run,
    )

    def execute_model(scheduler_output):
        execute_outputs.append(scheduler_output)

    def sample_tokens(grammar_output):
        sample_calls.append(grammar_output)

    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=vllm_config,
        execute_model=execute_model,
        sample_tokens=sample_tokens,
    )

    assert (
        flashinfer_sparse_mla_warmup._deepseek_v4_sparse_mla_decode_autotune(worker, 16)
        is True
    )

    assert context_calls == [("enter", 1, 1), ("exit", 2, 2)]
    assert loaded_paths == [str(cache_path)]
    assert world.barrier_count == 1
    assert kv_disabled == [True, False]
    assert sample_calls == [None, None]
    assert len(execute_outputs) == 3

    decode_prefill_output = execute_outputs[0]
    assert decode_prefill_output.total_num_scheduled_tokens == 2
    assert decode_prefill_output.num_scheduled_tokens == {
        "_sparse_mla_v2_decode_warmup_": 2,
    }
    assert [
        req.num_computed_tokens for req in decode_prefill_output.scheduled_new_reqs
    ] == [0]
    assert [req.block_ids for req in decode_prefill_output.scheduled_new_reqs] == [
        ([1],),
    ]
    decode_sampling_params = decode_prefill_output.scheduled_new_reqs[0].sampling_params
    assert decode_sampling_params is not None
    assert decode_sampling_params.max_tokens == 2

    mixed_output = execute_outputs[1]
    assert mixed_output.total_num_scheduled_tokens == 16
    assert mixed_output.num_scheduled_tokens == {
        "_sparse_mla_v2_decode_warmup_": 1,
        "_sparse_mla_v2_prefill_warmup_": 15,
    }
    assert mixed_output.scheduled_cached_reqs.req_ids == [
        "_sparse_mla_v2_decode_warmup_"
    ]
    assert mixed_output.scheduled_cached_reqs.num_computed_tokens == [2]
    assert mixed_output.scheduled_cached_reqs.num_output_tokens == [1]
    assert mixed_output.scheduled_cached_reqs.new_block_ids == [None]
    assert [req.num_computed_tokens for req in mixed_output.scheduled_new_reqs] == [0]
    assert [req.block_ids for req in mixed_output.scheduled_new_reqs] == [
        ([2],),
    ]

    cleanup_output = execute_outputs[2]
    assert cleanup_output.total_num_scheduled_tokens == 0
    assert cleanup_output.finished_req_ids == {
        "_sparse_mla_v2_decode_warmup_",
        "_sparse_mla_v2_prefill_warmup_",
    }


def test_sparse_mla_autotune_v2_returns_false_when_mixed_warmup_skips(
    monkeypatch, tmp_path: Path
) -> None:
    cache_path = tmp_path / "pp-rank-0" / "autotune_configs.json"
    context_entries: list[str] = []
    loaded_paths: list[str] = []

    class FakeAutoTuner:
        @classmethod
        def get(cls):
            return cls()

        def load_configs(self, path: str) -> bool:
            loaded_paths.append(path)
            return True

    @contextmanager
    def dsv4_autotune(cache_path=None):
        context_entries.append(str(cache_path))
        yield
        raise AssertionError("Skipped V2 warmup must not write autotune cache")

    fake_autotuner = SimpleNamespace(AutoTuner=FakeAutoTuner)
    fake_sparse = SimpleNamespace(sparse_mla_sm120_decode_dsv4_autotune=dsv4_autotune)
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(sparse_mla_sm120=fake_sparse)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setitem(sys.modules, "flashinfer.sparse_mla_sm120", fake_sparse)
    monkeypatch.setattr(flashinfer_sparse_mla_warmup, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup,
        "resolve_flashinfer_autotune_file",
        lambda runner: cache_path,
    )

    class FakeWorld:
        rank_in_group = 0
        local_rank = 0

        def broadcast_object(self, obj, src: int = 0):
            raise AssertionError("Skipped V2 warmup must not broadcast autotune cache")

        def barrier(self) -> None:
            raise AssertionError("Skipped V2 warmup must not enter cache barrier")

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: FakeWorld()
    )

    class FakeKVConnector:
        def set_disabled(self, disabled: bool) -> None:
            raise AssertionError("Skipped V2 warmup must not disable KV connector")

    def fail_execute_model(scheduler_output):
        raise AssertionError("Skipped V2 warmup must not execute model")

    def fail_sample_tokens(grammar_output):
        raise AssertionError("Skipped V2 warmup must not sample tokens")

    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=True),
    )

    def fail_dummy_run(**kwargs):
        raise AssertionError("Skipped V2 warmup must not call _dummy_run")

    runner = make_runner(
        ["FLASHINFER_MLA_SPARSE_DSV4"],
        is_pooling_model=False,
        vllm_config=vllm_config,
        kv_cache_config=SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
        ),
        kv_connector=FakeKVConnector(),
        _dummy_run=fail_dummy_run,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=vllm_config,
        execute_model=fail_execute_model,
        sample_tokens=fail_sample_tokens,
    )

    assert (
        flashinfer_sparse_mla_warmup._deepseek_v4_sparse_mla_decode_autotune(worker, 2)
        is False
    )
    assert context_entries == []
    assert loaded_paths == []
    assert not cache_path.exists()


def test_flashinfer_sparse_mla_warmup_skips_non_sm120_without_dummy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        flashinfer_sparse_mla_warmup.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )

    dummy_runs = []
    runner = make_runner(
        ["FLASHINFER_MLA_SPARSE"],
        is_pooling_model=False,
        _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
        ),
    )

    flashinfer_sparse_mla_warmup.flashinfer_sparse_mla_decode_autotune_warmup(worker)

    assert dummy_runs == []
