# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.snapshot.resources import (
    SnapshotEngineOps,
    SnapshotMemoryPlan,
    SnapshotResourcePlan,
    build_checkpoint_resources,
    build_session_resources,
)

_PHASES = [
    "prepare memory",
    "prepare communicators",
    "synchronize device",
    "restore communicators",
    "restore memory",
]


class DeviceState:
    """Device-boundary callbacks with observable partial resource changes."""

    def __init__(self, *, retain_weights=False, fail_at=None):
        self.retain_weights = retain_weights
        self.fail_at = fail_at
        self.error = RuntimeError(f"injected {fail_at}")
        self.weights = bytearray(b"model weights")
        self.kv_resident = True
        self.communicators = "active"
        self.reloads = 0
        self.events = []

    def step(self, phase):
        self.events.append(phase)
        if phase == self.fail_at:
            raise self.error

    def prepare_memory(self):
        assert self.communicators == "active"
        self.kv_resident = False
        if not self.retain_weights:
            self.weights = None
        self.step("prepare memory")

    def prepare_communicators(self):
        assert not self.kv_resident
        self.communicators = "unknown"
        self.step("prepare communicators")
        self.communicators = "released"

    def synchronize(self):
        assert self.communicators == "released"
        assert not self.kv_resident
        self.step("synchronize device")

    def restore_memory(self):
        assert self.communicators == "active"
        self.step("restore memory")
        if not self.retain_weights:
            self.weights = bytearray(b"model weights")
            self.reloads += 1
        self.kv_resident = True

    def restore_communicators(self):
        assert not self.kv_resident
        self.communicators = "unknown"
        self.step("restore communicators")
        self.communicators = "active"


def engine_ops(device):
    return SnapshotEngineOps(
        SnapshotResourcePlan(
            memory=SnapshotMemoryPlan(device.prepare_memory, device.restore_memory),
            prepare_communicators=device.prepare_communicators,
            restore_communicators=device.restore_communicators,
            synchronize=device.synchronize,
        )
    )


@pytest.mark.parametrize("retain_weights", [True, False], ids=["retain", "reload"])
def test_prepare_restore_preserves_the_memory_plan(retain_weights):
    device = DeviceState(retain_weights=retain_weights)
    ops = engine_ops(device)
    original_weights = device.weights
    assert (ops.state, ops.phase) == ("new", "new")
    assert ops.is_idle
    with pytest.raises(RuntimeError, match="cannot restore.*new"):
        ops.restore()
    assert device.events == []
    assert device.kv_resident

    ops.prepare()
    assert (ops.state, ops.phase) == ("prepared", "prepared")
    assert not ops.is_idle
    assert device.events == _PHASES[:3]
    assert device.communicators == "released"
    assert not device.kv_resident
    if retain_weights:
        assert device.weights is original_weights
    else:
        assert device.weights is None

    with pytest.raises(RuntimeError, match="cannot prepare.*prepared"):
        ops.prepare()
    assert device.events == _PHASES[:3]
    assert (ops.state, ops.phase) == ("prepared", "prepared")

    ops.restore()
    assert (ops.state, ops.phase) == ("ready", "ready")
    assert ops.is_idle
    assert device.events == _PHASES
    assert device.communicators == "active"
    assert device.kv_resident
    assert device.weights == original_weights
    assert (device.weights is original_weights) is retain_weights
    assert device.reloads == (0 if retain_weights else 1)

    restored_weights = device.weights
    ops.restore()
    with pytest.raises(RuntimeError, match="cannot prepare.*ready"):
        ops.prepare()
    assert (ops.state, ops.phase) == ("ready", "ready")
    assert device.events == _PHASES
    assert device.weights is restored_weights
    assert device.kv_resident
    assert device.reloads == (0 if retain_weights else 1)


@pytest.mark.parametrize(
    ("phase", "communicators"),
    [
        ("prepare memory", "active"),
        ("prepare communicators", "unknown"),
        ("synchronize device", "released"),
        ("restore communicators", "unknown"),
        ("restore memory", "active"),
    ],
)
def test_failure_retains_phase_and_forbids_restoration(phase, communicators):
    device = DeviceState(fail_at=phase)
    ops = engine_ops(device)
    if phase in _PHASES[3:]:
        ops.prepare()
        operation = ops.restore
    else:
        operation = ops.prepare

    with pytest.raises(RuntimeError) as error:
        operation()
    assert error.value is device.error
    assert (ops.state, ops.phase) == ("failed", phase)
    assert device.weights is None
    assert not device.kv_resident
    assert device.communicators == communicators
    expected_events = _PHASES[: _PHASES.index(phase) + 1]
    assert device.events == expected_events

    with pytest.raises(RuntimeError, match="cannot restore.*failed"):
        ops.restore()
    with pytest.raises(RuntimeError, match="cannot prepare.*failed"):
        ops.prepare()
    assert (ops.state, ops.phase) == ("failed", phase)
    assert device.events == expected_events
    assert device.weights is None
    assert not device.kv_resident
    assert device.communicators == communicators
    assert device.reloads == 0
    assert not ops.is_idle


def test_session_resources_restore_weights_before_kv():
    events = []
    ops = SnapshotEngineOps(
        build_session_resources(
            sleep=lambda: events.append("sleep(2)"),
            wake_up_memory=lambda tags: events.append(("wake", tuple(tags))),
            reload_weights=lambda: events.append("reload"),
            rpc=lambda method: events.append(method),
        )
    )

    ops.prepare()
    ops.restore()

    assert events == [
        "sleep(2)",
        "checkpoint_prepare",
        "synchronize_device",
        "checkpoint_restore",
        ("wake", ("weights",)),
        "reload",
        ("wake", ("kv_cache",)),
    ]


def test_checkpoint_resources_keep_weights_resident():
    events = []
    ops = SnapshotEngineOps(
        build_checkpoint_resources(
            release_kv_cache=lambda: events.append("release_kv"),
            wake_up_kv_cache=lambda: events.append("wake_kv"),
            rpc=lambda method: events.append(method),
        )
    )

    ops.prepare()
    ops.restore()

    assert events == [
        "release_kv",
        "checkpoint_prepare",
        "synchronize_device",
        "checkpoint_restore",
        "wake_kv",
    ]
