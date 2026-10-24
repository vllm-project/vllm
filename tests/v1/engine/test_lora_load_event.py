# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-core delivery of LoRALoadEvent. The idle-RPC and step paths run
against a real engine in tests/lora/test_lora_metrics.py."""

import queue

import msgspec

from vllm.lora.load_event import LoRALoadEvent, LoRALoadTiming
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.utils import EngineZmqAddresses


def _engine_core() -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core._pending_lora_load_event = None
    return engine_core


def test_pending_event_keeps_latest_snapshot_and_every_timing():
    engine_core = _engine_core()
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

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_lora_load_event(outputs)

    assert list(outputs) == [0]
    assert outputs[0].lora_load_event == LoRALoadEvent(
        gpu_adapters=["beta"],
        cpu_adapters=["alpha", "beta"],
        loads=[
            LoRALoadTiming("alpha", "load", 0.5),
            LoRALoadTiming("beta", "activate", 0.1),
        ],
    )


def test_none_publishes_nothing():
    engine_core = _engine_core()
    engine_core._publish_lora_load_event(None)

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_lora_load_event(outputs)
    assert outputs == {}


def test_flush_attaches_to_the_step_outputs_and_clears():
    engine_core = _engine_core()
    event = LoRALoadEvent(cpu_adapters=["alpha"])
    engine_core._publish_lora_load_event(event)

    step_outputs = EngineCoreOutputs(
        outputs=[EngineCoreOutput(request_id="req-1", new_token_ids=[1])]
    )
    outputs = {0: step_outputs}
    engine_core._flush_lora_load_event(outputs)
    assert outputs == {0: step_outputs}
    assert step_outputs.lora_load_event is event

    again: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_lora_load_event(again)
    assert again == {}


def test_proc_broadcasts_to_every_frontend():
    """Every API server gets the event, so their /metrics agree."""
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc.addresses = EngineZmqAddresses(inputs=["a", "b", "c"], outputs=["a", "b", "c"])
    proc.output_queue = queue.Queue()

    event = LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"])
    proc._publish_lora_load_event(event)
    proc._publish_lora_load_event(None)

    delivered = [proc.output_queue.get_nowait() for _ in range(3)]
    assert proc.output_queue.empty()
    assert [client_index for client_index, _ in delivered] == [0, 1, 2]
    assert all(out.lora_load_event is event for _, out in delivered)
    assert all(not out.outputs and out.scheduler_stats is None for _, out in delivered)


def test_engine_core_outputs_carry_the_event_as_the_last_field():
    """array_like, so the field is positional; non-Python frontends decode it
    by index."""
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
