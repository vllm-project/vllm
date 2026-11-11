#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_eplb_control_cpu_group,
)
from vllm.model_executor.model_loader.weight_cache.eplb import (
    EplbPlacement,
    bind_weight_cache_eplb,
    get_weight_cache_eplb_binding,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.utils.network_utils import get_open_port
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu import eplb_utils as eplb
from vllm.v1.worker.gpu import model_runner as mrv2


class FakeMemoryProfiler:
    def __enter__(self):
        self.consumed_memory = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEplbState:
    instances: list["FakeEplbState"] = []

    def __init__(self, parallel_config: Any, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.add_model_calls: list[tuple[Any, Any, str | None]] = []
        self.step_calls: list[tuple[bool, bool, bool]] = []
        self.async_started = False
        self.is_async = True
        self.update_mapping_calls: list[tuple[Any, torch.Tensor]] = []
        FakeEplbState.instances.append(self)

    def add_model(
        self,
        model: Any,
        model_config: Any,
        *,
        model_name: str | None = None,
    ) -> None:
        self.add_model_calls.append((model, model_config, model_name))

    def recover_weight_cache_placement(self, model_config: Any, binding: Any) -> None:
        pass

    def step(self, is_dummy: bool, is_profile: bool, *, log_stats: bool) -> None:
        self.step_calls.append((is_dummy, is_profile, log_stats))

    def start_async_loop(self) -> None:
        self.async_started = True

    def update_mapping(self, model_config: Any, mapping: torch.Tensor) -> None:
        self.update_mapping_calls.append((model_config, mapping))


def _make_runner(**overrides: Any) -> Any:
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(model="test-model", logits_processors=None)
    runner.req_states = SimpleNamespace(
        device=runner.device,
        max_num_reqs=8,
        vocab_size=0,
        all_token_ids=None,
        prompt_len=None,
        prefill_len=None,
        total_len=None,
    )
    runner.load_config = SimpleNamespace(load_format="hf")
    runner.parallel_config = SimpleNamespace(
        enable_eplb=True,
        enable_elastic_ep=False,
        enable_batch_sharded_sampling=False,
        eplb_config=SimpleNamespace(log_balancedness=True),
    )
    runner.compilation_config = CompilationConfig(mode=CompilationMode.NONE)
    runner.vllm_config = SimpleNamespace(
        compilation_config=runner.compilation_config,
        load_config=runner.load_config,
        model_config=runner.model_config,
    )
    runner.lora_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.speculative_config = None
    runner.speculator = None
    runner.pcp_manager = None
    runner.num_speculative_steps = 0
    runner.encoder_cache = None
    runner.is_pooling_model = False
    runner.is_last_pp_rank = True
    runner.is_first_pp_rank = True
    runner.max_num_reqs = 8
    runner.max_num_tokens = 16
    runner.decode_query_len = 1
    runner.kv_connector = SimpleNamespace(
        set_disabled=lambda *_: None,
        post_forward=lambda *_, **__: None,
    )
    runner.eplb = eplb.EPLBController(runner.parallel_config, runner.device)
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.pooling_runner = None
    runner.execute_model_state = None
    for key, value in overrides.items():
        setattr(runner, key, value)
    spec_config = runner.speculative_config
    runner._draft_workspace_lane = int(
        spec_config is not None
        and (
            spec_config.use_dspark()
            if hasattr(spec_config, "use_dspark")
            else getattr(spec_config, "method", None) == "dspark"
        )
    )
    return runner


def test_v2_load_model_registers_moe_with_eplb(monkeypatch):
    FakeEplbState.instances.clear()
    model = SimpleNamespace(is_moe=True)

    monkeypatch.setattr(mrv2, "DeviceMemoryProfiler", FakeMemoryProfiler)
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(
        mrv2,
        "get_model_loader",
        lambda load_config: SimpleNamespace(load_model=lambda **_: model),
    )
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *args: SimpleNamespace(num_new_sampled_tokens_per_step=1),
    )
    monkeypatch.setattr(
        eplb,
        "get_mixture_of_experts_model",
        lambda loaded_model: (
            loaded_model if getattr(loaded_model, "is_moe", False) else None
        ),
    )

    runner = _make_runner(is_last_pp_rank=False)
    mrv2.GPUModelRunner.load_model(runner)

    assert runner.model is model
    assert runner.model_state is not None
    assert runner.eplb_state is not None
    assert runner.eplb_state.add_model_calls == [(model, runner.model_config, None)]
    assert runner.eplb_state.async_started is True


def test_v2_load_model_with_dummy_weights_defers_eplb_async_loop(monkeypatch):
    FakeEplbState.instances.clear()
    model = SimpleNamespace(is_moe=True)

    monkeypatch.setattr(mrv2, "DeviceMemoryProfiler", FakeMemoryProfiler)
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(
        mrv2,
        "get_model_loader",
        lambda load_config: SimpleNamespace(load_model=lambda **_: model),
    )
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *args: SimpleNamespace(num_new_sampled_tokens_per_step=1),
    )
    monkeypatch.setattr(eplb, "get_mixture_of_experts_model", lambda model: model)

    runner = _make_runner(is_last_pp_rank=False)
    mrv2.GPUModelRunner.load_model(runner, load_dummy_weights=True)

    assert runner.load_config.load_format == "dummy"
    assert runner.eplb_state is not None
    # A scaling-up worker registers so it can join the EPLB communicator.
    assert runner.eplb_state.add_model_calls == [(model, runner.model_config, None)]
    assert runner.eplb_state.async_started is False


def test_v2_setup_eplb_from_mapping_updates_state_in_place():
    runner = _make_runner()
    state = FakeEplbState(runner.parallel_config, runner.device)
    runner.eplb_state = state
    mapping = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    mrv2.GPUModelRunner.setup_eplb_from_mapping(runner, mapping)

    assert runner.eplb_state is state
    assert state.update_mapping_calls == [(runner.model_config, mapping)]


def test_v2_load_model_registers_dspark_speculator_with_eplb(monkeypatch):
    FakeEplbState.instances.clear()
    target_model = SimpleNamespace(is_moe=True)
    draft_model = SimpleNamespace(is_moe=True)
    draft_model_config = SimpleNamespace(
        model="dspark-draft",
        hf_config=SimpleNamespace(model_type="deepseek_v4"),
    )

    class FakeDSparkSpeculator:
        def __init__(self):
            self.model = draft_model
            self.eplb_state = None
            self.draft_logits = None

        def load_model(self, target_model):
            return None

        def set_eplb_state(self, state) -> None:
            self.eplb_state = state

    monkeypatch.setattr(mrv2, "DeviceMemoryProfiler", FakeMemoryProfiler)
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(mrv2, "DraftModelSpeculator", FakeDSparkSpeculator)
    monkeypatch.setattr(
        mrv2,
        "get_model_loader",
        lambda load_config: SimpleNamespace(load_model=lambda **_: target_model),
    )
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *args: SimpleNamespace(num_new_sampled_tokens_per_step=1),
    )
    monkeypatch.setattr(
        eplb,
        "get_mixture_of_experts_model",
        lambda model: model if getattr(model, "is_moe", False) else None,
    )

    speculator = FakeDSparkSpeculator()
    runner = _make_runner(
        is_last_pp_rank=False,
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=draft_model_config,
        ),
        speculator=speculator,
    )
    mrv2.GPUModelRunner.load_model(runner)

    assert runner.eplb_state is not None
    assert runner.eplb_state.add_model_calls == [
        (draft_model, draft_model_config, "dspark-draft (draft)"),
        (target_model, runner.model_config, None),
    ]
    assert speculator.eplb_state is runner.eplb_state


def test_v2_sample_tokens_runs_eplb_on_non_last_pp_rank(monkeypatch):
    events = []
    runner = _make_runner(is_last_pp_rank=False, num_speculative_steps=0)
    runner.execute_model_state = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2, idx_mapping=torch.zeros(2, dtype=torch.int32)
        ),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        routed_experts=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )
    runner.req_states = SimpleNamespace()

    def fake_receive(*args, **kwargs):
        events.append("receive")
        # all_decode_next=True, so model_state.postprocess_state is skipped.
        return True

    runner.pp_handler = SimpleNamespace(receive=fake_receive)
    runner.postprocess_num_computed_tokens = lambda *args, **kwargs: events.append(
        "postprocess_num_computed_tokens"
    )
    runner.eplb.step = lambda *args, **kwargs: events.append("eplb")

    output = mrv2.GPUModelRunner.sample_tokens(runner, None)
    assert output in (EMPTY_MODEL_RUNNER_OUTPUT, None)
    assert events == ["receive", "postprocess_num_computed_tokens", "eplb"]


@pytest.fixture
def eplb_dist_init(tmp_path):
    config = VllmConfig()
    config.parallel_config.enable_eplb = True
    config.parallel_config.eplb_config.use_async = False
    config.parallel_config.eplb_config.communicator = "torch_gloo"
    config.parallel_config.eplb_config.num_redundant_experts = 1
    config.parallel_config.eplb_config.window_size = 2
    config.parallel_config.eplb_config.step_interval = 100
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'eplb_dist'}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield config.parallel_config
    cleanup_dist_env_and_memory()


class _CpuMoeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_moe_layers = 2
        self.num_expert_groups = 1
        self.num_logical_experts = 2
        self.num_physical_experts = 3
        self.num_local_physical_experts = 3
        self.num_routed_experts = 2
        self.num_shared_experts = 0
        self.num_redundant_experts = 1
        self.expert_weights = [[torch.zeros(3, 1)] for _ in range(2)]
        self.moe_layers = []

    def set_eplb_state(
        self, expert_load_view, logical_to_physical_map, logical_replica_count
    ):
        self.expert_load_view = expert_load_view
        self.logical_to_physical_map = logical_to_physical_map
        self.logical_replica_count = logical_replica_count

    def update_physical_experts_metadata(self, *args):
        pass


def _make_cpu_model_config(tmp_path) -> ModelConfig:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "n_embd": 8,
                "n_head": 1,
                "n_layer": 1,
                "n_positions": 16,
                "vocab_size": 16,
            }
        )
    )
    return ModelConfig(model=str(tmp_path), tokenizer_mode="skip", max_model_len=16)


def test_weight_cache_recovery_restores_maps_without_version_bump(
    eplb_dist_init, tmp_path
):
    model = _CpuMoeModel()
    placement = EplbPlacement(7, ((1, 0, 1), (0, 1, 0)))
    binding = bind_weight_cache_eplb(model, placement)
    controller = eplb.EPLBController(eplb_dist_init, torch.device("cpu"))
    controller.prepare_load()

    assert controller.maybe_register_model(model, _make_cpu_model_config(tmp_path))
    assert controller.state is not None
    model_state = next(iter(controller.state.model_states.values()))
    assert model_state.physical_to_logical_map.tolist() == [[1, 0, 1], [0, 1, 0]]
    assert model_state.logical_replica_count.tolist() == [[1, 2], [2, 1]]
    assert model_state.logical_to_physical_map.tolist() == [
        [[1, -1], [0, 2]],
        [[0, 2], [1, -1]],
    ]
    assert model_state.weight_cache_binding is binding
    assert binding.version == 7
    assert binding.placement is placement


def test_eplb_without_weight_cache_binding_keeps_initial_mapping(
    eplb_dist_init, tmp_path
):
    model = _CpuMoeModel()
    controller = eplb.EPLBController(eplb_dist_init, torch.device("cpu"))
    controller.prepare_load()

    assert controller.maybe_register_model(model, _make_cpu_model_config(tmp_path))
    assert controller.state is not None
    model_state = next(iter(controller.state.model_states.values()))
    assert model_state.physical_to_logical_map.tolist() == [[0, 1, 0], [0, 1, 0]]
    assert model_state.logical_replica_count.tolist() == [[2, 1], [2, 1]]
    assert model_state.weight_cache_binding is None


def test_weight_cache_pause_resume_controls_steps_and_active_copies(
    eplb_dist_init, tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    model = _CpuMoeModel()
    commands = []
    can_unfreeze = True
    unfreeze_failures = 1

    def request(message):
        nonlocal unfreeze_failures
        commands.append(message["cmd"])
        if message["cmd"] == "eplb_can_unfreeze":
            return {"status": "ok", "can_unfreeze": can_unfreeze}
        if message["cmd"] == "eplb_unfreeze" and unfreeze_failures:
            unfreeze_failures -= 1
            return {"status": "error", "message": "injected unfreeze failure"}
        return {"status": "ok"}

    binding = bind_weight_cache_eplb(
        model,
        EplbPlacement(3, ((1, 0, 1), (0, 1, 0))),
        token="test-token",
        request=request,
    )
    controller = eplb.EPLBController(eplb_dist_init, torch.device("cpu"))
    controller.prepare_load()
    assert controller.maybe_register_model(model, _make_cpu_model_config(tmp_path))
    assert controller.state is not None
    initial_step = controller.state.expert_rearrangement_step

    controller.pause_for_weight_seed()
    controller.pause_for_weight_seed()
    controller.step()
    assert controller.paused
    assert controller.state.expert_rearrangement_step == initial_step
    assert commands == ["eplb_freeze", "eplb_freeze"]

    can_unfreeze = False
    with pytest.raises(RuntimeError, match="remain paused"):
        controller.resume_for_weight_seed()
    assert controller.paused
    can_unfreeze = True
    with pytest.raises(RuntimeError, match="remain paused"):
        controller.resume_for_weight_seed()
    assert controller.paused
    assert controller.state.expert_rearrangement_step == initial_step

    controller.resume_for_weight_seed()
    controller.step()
    assert not controller.paused
    assert controller.state.expert_rearrangement_step == initial_step + 1
    assert commands[-2:] == ["eplb_can_unfreeze", "eplb_unfreeze"]
    assert commands.count("eplb_unfreeze") == 2

    controller.parallel_config.data_parallel_size = 2
    with pytest.raises(RuntimeError, match="requires DP=1"):
        controller.pause_for_weight_seed()
    assert not controller.paused
    controller.parallel_config.data_parallel_size = 1

    controller.detach_weight_cache_bindings()
    controller.detach_weight_cache_bindings()
    assert get_weight_cache_eplb_binding(model) is None
    assert commands.count("eplb_detach") == 1
    assert binding.version == 3


def _pause_after_ep_collective_worker(
    rank: int,
    init_method: str,
    model_path: str,
    pause_started,
) -> None:
    torch.accelerator.synchronize = lambda: None
    config = VllmConfig()
    config.parallel_config.enable_eplb = True
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.eplb_config.use_async = False
    config.parallel_config.eplb_config.communicator = "torch_gloo"
    config.parallel_config.eplb_config.num_redundant_experts = 1
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=2,
                rank=rank,
                distributed_init_method=init_method,
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(2, 1)
            assert get_ep_group().world_size == 2
            assert get_eplb_control_cpu_group() is not get_ep_group().cpu_group
            assert (
                torch.distributed.get_process_group_ranks(get_eplb_control_cpu_group())
                == get_ep_group().ranks
            )

            model = _CpuMoeModel()
            unfreeze_failures = int(rank == 0)

            def request(message):
                nonlocal unfreeze_failures
                if message["cmd"] == "eplb_unfreeze" and unfreeze_failures:
                    unfreeze_failures -= 1
                    return {"status": "error", "message": "injected failure"}
                return {"status": "ok", "can_unfreeze": True}

            bind_weight_cache_eplb(
                model,
                EplbPlacement(0, ((1, 0, 1), (0, 1, 0))),
                token="test-token",
                request=request,
            )
            controller = eplb.EPLBController(
                config.parallel_config, torch.device("cpu")
            )
            controller.prepare_load()
            model_config = ModelConfig(
                model=model_path, tokenizer_mode="skip", max_model_len=16
            )
            assert controller.maybe_register_model(model, model_config)

            rearrange_result = torch.tensor(1, dtype=torch.int32)
            torch.distributed.all_reduce(
                rearrange_result,
                group=get_ep_group().cpu_group,
            )
            assert rearrange_result.item() == 2, (
                rank,
                rearrange_result,
                get_ep_group().world_size,
            )
            if rank == 1:
                pause_started.set()
            else:
                assert pause_started.wait(10)

            for round_idx in range(2):
                controller.pause_for_weight_seed()
                assert controller.paused
                if round_idx == 0:
                    with pytest.raises(RuntimeError, match="remain paused"):
                        controller.resume_for_weight_seed()
                    assert controller.paused
                controller.resume_for_weight_seed()
                assert not controller.paused
            controller.detach_weight_cache_bindings()
            controller.detach_weight_cache_bindings()
            assert get_weight_cache_eplb_binding(model) is None
    finally:
        cleanup_dist_env_and_memory()


def test_weight_cache_pause_is_ordered_after_ep_collectives(tmp_path):
    """Exercise the EP-local half of the DP=1 top-level pause protocol."""
    _make_cpu_model_config(tmp_path)
    context = mp.get_context("spawn")
    pause_started = context.Event()
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    processes = [
        context.Process(
            target=_pause_after_ep_collective_worker,
            args=(rank, init_method, str(tmp_path), pause_started),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(10)


def _empty_pp_shard_worker(rank, init_method, model_path):
    torch.accelerator.synchronize = lambda: None
    config = VllmConfig()
    config.parallel_config.enable_eplb = True
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.eplb_config.use_async = False
    config.parallel_config.eplb_config.communicator = "torch_gloo"
    config.parallel_config.eplb_config.num_redundant_experts = 1
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=2,
                rank=rank,
                distributed_init_method=init_method,
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(1, 2)
            assert get_ep_group().ranks == [rank]
            assert (
                torch.distributed.get_process_group_ranks(get_eplb_control_cpu_group())
                == get_ep_group().ranks
            )

            model = _CpuMoeModel() if rank == 1 else torch.nn.Module()
            placement = EplbPlacement.initial(model)
            assert EplbPlacement.from_json_dict(placement.to_json_dict()) == placement
            if rank == 0:
                assert placement.physical_to_logical_map == ()
            commands = []

            def request(message):
                commands.append(message["cmd"])
                return {"status": "ok", "can_unfreeze": True}

            binding = bind_weight_cache_eplb(
                model,
                placement,
                token="test-token",
                request=request,
            )
            controller = eplb.EPLBController(
                config.parallel_config, torch.device("cpu")
            )
            controller.prepare_load()
            model_config = ModelConfig(
                model=model_path, tokenizer_mode="skip", max_model_len=16
            )
            registered = controller.maybe_register_model(model, model_config)
            assert registered == (rank == 1)
            if rank == 0:
                assert not controller.maybe_register_model(model, model_config)
                assert len(controller._weight_cache_bindings) == 1
                assert controller.state is not None
                assert not controller.state.model_states

            controller.pause_for_weight_seed()
            controller.resume_for_weight_seed()
            assert binding.version == 0
            assert commands == ["eplb_freeze", "eplb_can_unfreeze", "eplb_unfreeze"]
            controller.detach_weight_cache_bindings()
            controller.detach_weight_cache_bindings()
            assert get_weight_cache_eplb_binding(model) is None
            assert commands.count("eplb_detach") == 1
    finally:
        cleanup_dist_env_and_memory()


def test_weight_cache_empty_pp_shard_joins_control_without_commit(tmp_path):
    _make_cpu_model_config(tmp_path)
    context = mp.get_context("spawn")
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    processes = [
        context.Process(
            target=_empty_pp_shard_worker,
            args=(rank, init_method, str(tmp_path)),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(10)


@pytest.mark.parametrize("failure_point", ["move", "map"])
def test_weight_cache_aborts_after_rearrange_failure_without_collective(
    eplb_dist_init, tmp_path, monkeypatch, failure_point
):
    class NoOpEvent:
        def __init__(self, *args, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.0

    monkeypatch.setattr("vllm.distributed.eplb.eplb_state.torch.Event", NoOpEvent)
    model = _CpuMoeModel()
    writer_state = "clean"
    commands = []

    def request(message):
        nonlocal writer_state
        commands.append(message["cmd"])
        if message["cmd"] == "eplb_begin":
            assert writer_state == "clean"
            writer_state = "dirty"
        elif message["cmd"] == "eplb_abort":
            assert writer_state == "dirty"
            writer_state = "poisoned"
        return {"status": "ok"}

    binding = bind_weight_cache_eplb(
        model,
        EplbPlacement(0, ((0, 1, 0), (0, 1, 0))),
        token="test-token",
        request=request,
    )
    controller = eplb.EPLBController(eplb_dist_init, torch.device("cpu"))
    controller.prepare_load()
    assert controller.maybe_register_model(model, _make_cpu_model_config(tmp_path))
    assert controller.state is not None

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "map":
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.rearrange_expert_weights_inplace",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr("vllm.distributed.eplb.eplb_state._commit_eplb_maps", fail)
    else:
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.rearrange_expert_weights_inplace",
            fail,
        )

    action_checks = 0
    original_check = controller.state._check_weight_cache_action

    def checked_action(*args):
        nonlocal action_checks
        action_checks += 1
        return original_check(*args)

    monkeypatch.setattr(controller.state, "_check_weight_cache_action", checked_action)
    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        controller.state.rearrange()

    assert action_checks == 1
    assert commands == ["eplb_begin", "eplb_abort"]
    assert writer_state == "poisoned"
    assert binding.version == 0


def _runtime_failure_worker(rank, init_method, model_path):
    torch.accelerator.synchronize = lambda: None
    config = VllmConfig()
    config.parallel_config.enable_eplb = True
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.eplb_config.use_async = False
    config.parallel_config.eplb_config.communicator = "torch_gloo"
    config.parallel_config.eplb_config.num_redundant_experts = 1
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=2,
                rank=rank,
                distributed_init_method=init_method,
                local_rank=rank,
                backend="gloo",
            )
            initialize_model_parallel(2, 1)
            model_config = ModelConfig(
                model=model_path, tokenizer_mode="skip", max_model_len=16
            )

            model = _CpuMoeModel()
            placement = (
                EplbPlacement(0, ()) if rank == 0 else EplbPlacement.initial(model)
            )
            bind_weight_cache_eplb(
                model,
                placement,
                token="test-token",
                request=lambda message: {"status": "ok"},
            )
            controller = eplb.EPLBController(
                config.parallel_config, torch.device("cpu")
            )
            controller.prepare_load()
            with pytest.raises(RuntimeError, match="placement shape"):
                controller.maybe_register_model(model, model_config)
            controller.detach_weight_cache_bindings()

            writer_state = "clean"
            begin_commands = []

            def begin_request(message):
                nonlocal writer_state
                begin_commands.append(message["cmd"])
                if message["cmd"] == "eplb_begin":
                    if rank == 0:
                        return {"status": "error", "message": "injected begin"}
                    writer_state = "dirty"
                elif message["cmd"] == "eplb_abort":
                    assert writer_state == "dirty"
                    writer_state = "poisoned"
                return {"status": "ok"}

            model = _CpuMoeModel()
            binding = bind_weight_cache_eplb(
                model,
                EplbPlacement.initial(model),
                token="test-token",
                request=begin_request,
            )
            controller = eplb.EPLBController(
                config.parallel_config, torch.device("cpu")
            )
            controller.prepare_load()
            assert controller.maybe_register_model(model, model_config)
            assert controller.state is not None
            model_state = next(iter(controller.state.model_states.values()))
            with pytest.raises(RuntimeError, match="begin failed"):
                controller.state._begin_weight_cache_rearrangement(model_state)
            if rank == 0:
                assert begin_commands == ["eplb_begin"]
                assert writer_state == "clean"
            else:
                assert begin_commands == ["eplb_begin", "eplb_abort"]
                assert writer_state == "poisoned"
            assert binding.version == 0
            controller.detach_weight_cache_bindings()

            writer_state = "clean"
            commit_commands = []

            def commit_request(message):
                nonlocal writer_state
                commit_commands.append(message["cmd"])
                if message["cmd"] == "eplb_begin":
                    assert writer_state == "clean"
                    writer_state = "dirty"
                elif message["cmd"] == "eplb_commit":
                    if rank == 0:
                        return {"status": "error", "message": "injected commit"}
                    writer_state = "clean"
                elif message["cmd"] == "eplb_abort":
                    assert writer_state == "dirty"
                    writer_state = "poisoned"
                return {"status": "ok"}

            model = _CpuMoeModel()
            binding = bind_weight_cache_eplb(
                model,
                EplbPlacement.initial(model),
                token="test-token",
                request=commit_request,
            )
            controller = eplb.EPLBController(
                config.parallel_config, torch.device("cpu")
            )
            controller.prepare_load()
            assert controller.maybe_register_model(model, model_config)
            assert controller.state is not None
            model_state = next(iter(controller.state.model_states.values()))
            controller.state._begin_weight_cache_rearrangement(model_state)
            with pytest.raises(RuntimeError, match="commit failed"):
                controller.state._commit_weight_cache_rearrangement(
                    model_state,
                    model_state.physical_to_logical_map.clone(),
                )
            if rank == 0:
                assert commit_commands == ["eplb_begin", "eplb_commit", "eplb_abort"]
                assert writer_state == "poisoned"
                assert binding.version == 0
            else:
                assert commit_commands == ["eplb_begin", "eplb_commit"]
                assert writer_state == "clean"
                assert binding.version == 1
            controller.detach_weight_cache_bindings()
    finally:
        cleanup_dist_env_and_memory()


def test_weight_cache_begin_commit_failures_abort_only_dirty_writers(tmp_path):
    _make_cpu_model_config(tmp_path)
    context = mp.get_context("spawn")
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    processes = [
        context.Process(
            target=_runtime_failure_worker,
            args=(rank, init_method, str(tmp_path)),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(10)


@pytest.mark.parametrize("failure_point", ["synchronize", "placement"])
def test_weight_cache_pre_commit_failure_aborts_dirty_writer(
    eplb_dist_init, tmp_path, monkeypatch, failure_point
):
    model = _CpuMoeModel()
    writer_state = "clean"
    commands = []

    def request(message):
        nonlocal writer_state
        commands.append(message["cmd"])
        if message["cmd"] == "eplb_begin":
            writer_state = "dirty"
        elif message["cmd"] == "eplb_abort":
            writer_state = "poisoned"
        return {"status": "ok"}

    binding = bind_weight_cache_eplb(
        model,
        EplbPlacement.initial(model),
        token="test-token",
        request=request,
    )
    controller = eplb.EPLBController(eplb_dist_init, torch.device("cpu"))
    controller.prepare_load()
    assert controller.maybe_register_model(model, _make_cpu_model_config(tmp_path))
    assert controller.state is not None
    model_state = next(iter(controller.state.model_states.values()))
    controller.state._begin_weight_cache_rearrangement(model_state)

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    if failure_point == "synchronize":
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.torch.accelerator.synchronize", fail
        )
    else:
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.torch.accelerator.synchronize",
            lambda: None,
        )
        monkeypatch.setattr("vllm.distributed.eplb.eplb_state.EplbPlacement", fail)
    with pytest.raises(RuntimeError, match="commit failed") as error:
        controller.state._commit_weight_cache_rearrangement(
            model_state,
            model_state.physical_to_logical_map.clone(),
        )

    assert error.value.__cause__ is not None
    assert f"injected {failure_point} failure" in str(error.value.__cause__)
    assert commands == ["eplb_begin", "eplb_abort"]
    assert writer_state == "poisoned"
    assert binding.version == 0
