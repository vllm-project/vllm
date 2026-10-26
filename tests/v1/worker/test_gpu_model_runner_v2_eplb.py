#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CompilationConfig, CompilationMode
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu import eplb_utils as eplb
from vllm.v1.worker.gpu import model_runner as mrv2


def test_v2_reports_logical_sleep_state_and_skips_sleeping_rank(monkeypatch):
    group = SimpleNamespace(world_size=4, rank=3)
    monkeypatch.setattr(eplb, "get_ep_group", lambda: group)
    runner = _make_runner()
    runner.parallel_config.enable_elastic_ep = True
    runner.eplb.state = SimpleNamespace(
        logical_sleep_state=SimpleNamespace(rank_mapping={0: 0, 1: 1, 2: -1, 3: -1})
    )
    assert runner.get_ep_sleep_state() == {
        "ep_world_size": 4,
        "active_ep_size": 2,
        "sleeping_ep_ranks": [2, 3],
    }
    assert runner._dummy_run(1) == (None, None)
    assert runner.execute_model(None) is EMPTY_MODEL_RUNNER_OUTPUT


@pytest.mark.parametrize("already_sleeping", [False, True])
def test_v2_resize_remaps_before_masking_and_dp_split(monkeypatch, already_sleeping):
    calls = []
    manager = SimpleNamespace(set_masked_ranks=lambda ranks: calls.append("mask"))
    group = SimpleNamespace(
        world_size=4, device_communicator=SimpleNamespace(all2all_manager=manager)
    )
    monkeypatch.setattr(eplb, "get_ep_group", lambda: group)
    monkeypatch.setattr(eplb, "get_dp_group", lambda size: calls.append(size))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_runner()
    runner.eplb.state = SimpleNamespace(
        is_logical_sleep_active=lambda: already_sleeping,
        prepare_logical_sleep=lambda ranks: calls.append("prepare"),
        resize_logical_sleep=lambda ranks: calls.append("resize"),
        restore_logical_sleep=lambda: calls.append("restore"),
    )
    runner.resize_sleep_ep_ranks([2, 3])
    assert calls == ["resize" if already_sleeping else "prepare", "mask", 2]
    calls.clear()
    runner.resize_sleep_ep_ranks([])
    assert calls == ["restore", "mask", 4]


def test_v2_does_not_run_periodic_eplb_over_sleeping_peers():
    runner = _make_runner()
    runner.parallel_config.enable_elastic_ep = True
    runner.eplb._has_registered_models = True
    step = Mock()
    runner.eplb.state = SimpleNamespace(logical_sleep_state=object(), step=step)
    runner.eplb.step()
    step.assert_not_called()


@pytest.mark.parametrize(
    "connected_ranks,expected_experts", [(1, 64), (2, 128), (4, 256)]
)
def test_nixl_dispatch_preserves_local_expert_shape_after_resize(
    monkeypatch, connected_ranks, expected_experts
):
    pytest.importorskip("nixl_ep")
    from vllm.distributed.device_communicators.all2all import NixlEPAll2AllManager
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )

    monkeypatch.setattr(
        NixlEPAll2AllManager,
        "_buffer",
        SimpleNamespace(max_num_ep_ranks=4, connected_ep_size=connected_ranks),
    )
    prepare = NixlEPPrepareAndFinalize.__new__(NixlEPPrepareAndFinalize)
    prepare.expert_capacity = 256
    assert prepare.dispatch_expert_capacity == expected_experts


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
        self.add_model_calls: list[tuple[Any, Any]] = []
        self.step_calls: list[tuple[bool, bool, bool]] = []
        self.async_started = False
        self.is_async = True
        self.update_mapping_calls: list[tuple[Any, torch.Tensor]] = []
        FakeEplbState.instances.append(self)

    def add_model(self, model: Any, model_config: Any) -> None:
        self.add_model_calls.append((model, model_config))

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
    assert runner.eplb_state.add_model_calls == [(model, runner.model_config)]
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
    assert runner.eplb_state.add_model_calls == [(model, runner.model_config)]
    assert runner.eplb_state.async_started is False


def test_v2_setup_eplb_from_mapping_updates_state_in_place():
    runner = _make_runner()
    state = FakeEplbState(runner.parallel_config, runner.device)
    runner.eplb_state = state
    mapping = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    mrv2.GPUModelRunner.setup_eplb_from_mapping(runner, mapping)

    assert runner.eplb_state is state
    assert state.update_mapping_calls == [(runner.model_config, mapping)]


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
