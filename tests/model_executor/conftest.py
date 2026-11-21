# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, local
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def autotune_model_path(tmp_path_factory):
    from transformers import GPT2Config

    path = tmp_path_factory.mktemp("autotune-model")
    GPT2Config(n_embd=16, n_head=4, n_layer=1, n_positions=128).save_pretrained(path)
    return path


@pytest.fixture
def autotune_model_state(autotune_model_path):
    import torch

    from vllm.config import DeviceConfig, ModelConfig, VllmConfig
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    def make(*, max_model_len=64):
        config = VllmConfig(
            model_config=ModelConfig(
                str(autotune_model_path),
                skip_tokenizer_init=True,
                max_model_len=max_model_len,
                dtype="float32",
            ),
            device_config=DeviceConfig("cpu"),
        )
        return DefaultModelState(config, torch.nn.Identity(), None, torch.device("cpu"))

    return make


class _Gather:
    """Exchange the actual submissions with the ranks in this group."""

    def __init__(self, size):
        self.barrier = Barrier(size, timeout=15)
        self.values = [None] * size

    def __call__(self, rank, out, value):
        self.values[rank] = value
        self.barrier.wait()
        out[:] = self.values
        self.barrier.wait()


class _AutotuneGroup:
    def __init__(self, run, ranks):
        self.run = run
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(run.rank)
        self.cpu_group = self

    def record(self, operation):
        self.run.collectives[self.ranks][self.run.rank].append(operation)

    def all_gather_object(self, out, value):
        self.record(("all_gather",))
        self.run.gathers[self.ranks](self.rank_in_group, out, value)

    def barrier(self):
        self.record(("barrier",))


class _AutotuneTuner:
    def __init__(self, run, *, validating=False):
        self.run = run
        self.validating = validating
        self.profiling_cache = {}
        self._file_configs = {}
        self._namespaced_records = {}
        self.protocol_records = {}
        self.loaded = None
        self.load_paths = []
        self._dirty = False

    @property
    def cache(self):
        return {**self._file_configs, **self.profiling_cache}

    def load_configs(self, path):
        path = Path(path)
        rank = self.run.rank
        if self.validating:
            self.run.validation_paths[rank].append(path)
        else:
            self.load_paths.append(path)
        if rank in self.run.unreadable_ranks:
            raise FileNotFoundError(path)
        if rank in self.run.rejected_load_ranks:
            return False
        if not self.validating and rank in self.run.rejected_install_ranks:
            return False
        self.loaded = json.loads(path.read_text())
        self._file_configs.update(self.loaded)
        if self.validating and rank in self.run.replace_during_validation:
            snapshot = path.with_suffix(".replacement")
            snapshot.write_text(json.dumps(self.loaded))
            snapshot.replace(path)
        return True

    def clear_cache(self):
        self.profiling_cache.clear()
        self._file_configs.clear()
        self._namespaced_records.clear()
        self.protocol_records.clear()
        self._dirty = False

    def save_configs(self, path):
        self.run.saves.append((self.run.rank, Path(path), dict(self.cache)))
        Path(path).write_text(json.dumps(self.cache))

    def profile(self, operation):
        if operation in self.cache:
            return
        group = self.run.tuning_group
        rank = self.run.rank
        self.run.profile_groups[rank].append(None if group is None else group.ranks)
        for tactic in range(2):
            if group is not None:
                group.record(("all_reduce", operation, tactic))
        self.profiling_cache[operation] = rank // self.run.tp
        self._dirty = True


class _AutotuneRun:
    def __init__(self, pp, tp, cache_dir, make_model_state):
        self.pp, self.tp = pp, tp
        self.cache_dir = cache_dir
        self.local = local()
        self.ranks = tuple(range(pp * tp))
        self.collectives: dict[tuple[int, ...], dict[int, list[tuple[object, ...]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        groups = [self.ranks] + [self.ranks[i : i + tp] for i in range(0, pp * tp, tp)]
        self.gathers = {ranks: _Gather(len(ranks)) for ranks in groups}
        self.rejected_load_ranks: set[int] = set()
        self.unreadable_ranks: set[int] = set()
        self.rejected_install_ranks: set[int] = set()
        self.replace_during_validation: set[int] = set()
        self.validation_paths = defaultdict(list)
        self.tuners = {rank: _AutotuneTuner(self) for rank in self.ranks}
        self.states = {rank: make_model_state() for rank in self.ranks}
        for state in self.states.values():
            state.vllm_config.parallel_config.pipeline_parallel_size = pp
        self.saves = []
        self.profile_groups = defaultdict(list)
        self.dummy_run_ranks = []

    @property
    def rank(self):
        return self.local.rank

    @property
    def tuning_group(self):
        return getattr(self.local, "tuning_group", None)

    def cache_path(self, rank):
        return self.cache_dir / f"autotune_configs_dp0_rank{rank}.json"

    def world(self):
        return _AutotuneGroup(self, self.ranks)

    def tensor_group(self):
        start = self.rank // self.tp * self.tp
        return _AutotuneGroup(self, self.ranks[start : start + self.tp])

    def pipeline_group(self):
        return SimpleNamespace(world_size=self.pp)

    def set_group(self, group):
        self.local.tuning_group = group

    def dummy_runs(self, runner, *, skip_attn=False):
        self.dummy_run_ranks.append(self.rank)
        self.tuners[self.rank].profile("shared_gemm")
        if self.pp > 1 and self.rank // self.tp == 0:
            self.tuners[self.rank].profile("pp0_extra_gemm")

    def on_ranks(self, fn):
        def run(rank):
            self.local.rank = rank
            return fn(rank)

        with ThreadPoolExecutor(max_workers=len(self.ranks)) as pool:
            return list(pool.map(run, self.ranks))

    def execute(self, *, expected_error=None):
        from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune

        def tune(rank):
            state = self.states[rank]
            # Only the FlashInfer warmup boundary is under test. The state and
            # config are real; no CUDA model execution is needed here.
            runner = SimpleNamespace(
                model_state=state,
                vllm_config=state.vllm_config,
                get_model=lambda: state.model,
            )
            outcome = (
                pytest.raises(RuntimeError, match=expected_error)
                if expected_error is not None
                else nullcontext()
            )
            with outcome:
                flashinfer_autotune(runner)

        self.on_ranks(tune)
        return self

    def assert_collectives_match(self):
        for ranks, traces in self.collectives.items():
            assert set(traces) == set(ranks)
            assert all(trace == traces[ranks[0]] for trace in traces.values()), traces


@pytest.fixture
def autotune_run(monkeypatch, tmp_path, autotune_model_state):
    import vllm.model_executor.warmup.kernel_warmup as warmup
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed import parallel_state

    def make_run(*, pp=2, tp=4):
        run = _AutotuneRun(pp, tp, tmp_path, autotune_model_state)
        monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: pp > 1)
        monkeypatch.setattr(parallel_state, "get_node_count", lambda: 1)

        class AutoTuner(_AutotuneTuner):
            def __init__(self):
                super().__init__(run, validating=True)

            @staticmethod
            def get():
                return run.tuners[run.rank]

        autotuner = ModuleType("flashinfer.autotuner")
        monkeypatch.setattr(autotuner, "AutoTuner", AutoTuner, raising=False)
        monkeypatch.setattr(
            autotuner, "set_autotune_process_group", run.set_group, raising=False
        )
        monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner)
        monkeypatch.setattr(parallel_state, "get_world_group", run.world)
        monkeypatch.setattr(parallel_state, "get_tp_group", run.tensor_group)
        monkeypatch.setattr(parallel_state, "get_pp_group", run.pipeline_group)
        monkeypatch.setattr(fi_utils, "autotune", lambda **kwargs: nullcontext())
        monkeypatch.setattr(
            "torch.distributed.all_gather_object",
            lambda out, value, group: group.all_gather_object(out, value),
        )
        monkeypatch.setattr(
            warmup,
            "resolve_rank_flashinfer_autotune_file",
            lambda _: run.cache_path(run.rank),
        )
        monkeypatch.setattr(warmup, "_flashinfer_autotune_skip_ops", lambda _: None)
        monkeypatch.setattr(
            warmup, "_run_flashinfer_autotune_dummy_runs", run.dummy_runs
        )
        monkeypatch.setattr(warmup, "replayssm_autotune_warmup", lambda _: None)
        monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", lambda _: None)
        return run

    return make_run
