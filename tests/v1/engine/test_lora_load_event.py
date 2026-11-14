# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-core delivery of LoRALoadEvent. tests/lora/test_lora_metrics.py runs
the same paths on a real engine."""

import queue
from collections import deque

import msgspec

from vllm.lora.load_event import LoRALoadEvent, LoRALoadTiming
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.executor.uniproc_executor import UniProcExecutor


def _engine_core(cls: type[EngineCore] = EngineCore) -> EngineCore:
    """The engine core state the LoRA event methods use, without a model."""
    engine_core = object.__new__(cls)
    engine_core._pending_lora_load_event = None
    engine_core._lora_load_event = None
    engine_core._num_stale_lora_snapshots = 0
    engine_core.batch_queue = deque()
    engine_core.model_executor = UniProcExecutor.__new__(UniProcExecutor)
    engine_core.model_executor._lora_load_event = None
    return engine_core


def test_pending_event_keeps_latest_snapshot_and_every_timing():
    engine_core = _engine_core()
    engine_core._publish_lora_load_event(None)
    assert engine_core.take_pending_lora_load_event() is None

    engine_core._publish_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["alpha"],
            cpu_adapters=["alpha"],
            loads=[LoRALoadTiming("alpha", "load", 0.5)],
        )
    )
    engine_core._publish_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["beta"],
            cpu_adapters=["alpha", "beta"],
            loads=[LoRALoadTiming("beta", "activate", 0.1)],
        )
    )

    assert engine_core.take_pending_lora_load_event() == LoRALoadEvent(
        gpu_adapters=["beta"],
        cpu_adapters=["alpha", "beta"],
        loads=[
            LoRALoadTiming("alpha", "load", 0.5),
            LoRALoadTiming("beta", "activate", 0.1),
        ],
    )
    assert engine_core.take_pending_lora_load_event() is None


def test_proc_sends_adapters_to_every_frontend_and_timings_to_one():
    """API servers share Prometheus counters, so only client 0 gets the
    timings; each serves the adapters from its own state."""
    proc = _engine_core(EngineCoreProc)
    proc.addresses = EngineZmqAddresses(inputs=["a", "b", "c"], outputs=["a", "b", "c"])
    proc.output_queue = queue.Queue()

    event = LoRALoadEvent(
        gpu_adapters=["alpha"],
        cpu_adapters=["alpha"],
        ranks={"alpha": 8},
        loads=[LoRALoadTiming("alpha", "load", 0.5)],
    )
    proc._publish_lora_load_event(event)
    proc._publish_lora_load_event(None)

    sent = [proc.output_queue.get_nowait() for _ in range(3)]
    assert proc.output_queue.empty()
    assert [client_index for client_index, _ in sent] == [0, 1, 2]
    assert sent[0][1].lora_load_event is event
    adapters_only = LoRALoadEvent(
        gpu_adapters=["alpha"], cpu_adapters=["alpha"], ranks={"alpha": 8}
    )
    for _, out in sent[1:]:
        assert out.lora_load_event == adapters_only
    for _, out in sent:
        assert not out.outputs and out.scheduler_stats is None


def test_rpc_snapshot_outlives_steps_queued_before_it():
    """With a batch queue, steps already queued during an adapter RPC took
    their snapshots before the RPC's. Their timings count; their adapters do
    not replace the RPC's. A step queued after the RPC reports normally."""
    engine_core = _engine_core()
    engine_core.batch_queue.extend(["step-1", "step-2"])

    rpc = LoRALoadEvent(
        gpu_adapters=["beta"],
        cpu_adapters=["beta", "gamma"],
        loads=[LoRALoadTiming("beta", "load", 0.1)],
    )
    engine_core.model_executor._lora_load_event = rpc
    engine_core._publish_lora_rpc_load_event()
    assert engine_core.take_pending_lora_load_event() == rpc

    # step-1 still holds alpha, which the RPC removed.
    engine_core._publish_step_lora_load_event(
        LoRALoadEvent(
            gpu_adapters=["alpha", "gamma"],
            cpu_adapters=["alpha", "beta", "gamma"],
            loads=[LoRALoadTiming("gamma", "load", 0.2)],
        )
    )
    assert engine_core.take_pending_lora_load_event() == LoRALoadEvent(
        gpu_adapters=["beta"],
        cpu_adapters=["beta", "gamma"],
        loads=[LoRALoadTiming("gamma", "load", 0.2)],
    )

    # step-2 has no timings, so it sends nothing.
    engine_core._publish_step_lora_load_event(
        LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha", "beta"])
    )
    assert engine_core.take_pending_lora_load_event() is None

    after_rpc = LoRALoadEvent(gpu_adapters=["gamma"], cpu_adapters=["beta", "gamma"])
    engine_core._publish_step_lora_load_event(after_rpc)
    assert engine_core.take_pending_lora_load_event() == after_rpc


def test_unchanged_rpc_leaves_queued_steps_alone():
    """An adapter RPC that changed nothing has no snapshot, so the queued
    steps' snapshots are still the newest."""
    engine_core = _engine_core()
    engine_core.batch_queue.append("step-1")
    engine_core._publish_lora_rpc_load_event()

    step = LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"])
    engine_core._publish_step_lora_load_event(step)
    assert engine_core.take_pending_lora_load_event() == step


def test_engine_core_outputs_carry_the_event_as_the_last_field():
    """EngineCoreOutputs is array_like, so frontends decode the field by
    position."""
    event = LoRALoadEvent(
        gpu_adapters=["alpha"],
        cpu_adapters=["alpha", "beta"],
        ranks={"alpha": 64},
        loads=[LoRALoadTiming("alpha", "load", 0.25)],
    )
    encoded = msgspec.msgpack.encode(
        EngineCoreOutputs(engine_index=1, lora_load_event=event)
    )

    as_array = msgspec.msgpack.decode(encoded)
    assert as_array[-1] == {
        "gpu_adapters": ["alpha"],
        "cpu_adapters": ["alpha", "beta"],
        "ranks": {"alpha": 64},
        "loads": [{"adapter_name": "alpha", "transition": "load", "seconds": 0.25}],
    }
    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreOutputs)
    assert decoded.lora_load_event == event
