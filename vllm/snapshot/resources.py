# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SnapshotMemoryPlan:
    prepare: Callable[[], object]
    restore: Callable[[], object]


@dataclass(frozen=True)
class SnapshotResourcePlan:
    """Engine-side operations for one snapshot consumer.

    Both current consumers release memory before preparing communicators and
    restore communicators before memory. A failure keeps its phase and is
    terminal for the engine.
    """

    memory: SnapshotMemoryPlan
    prepare_communicators: Callable[[], object]
    restore_communicators: Callable[[], object]
    synchronize: Callable[[], object]


class SnapshotEngineOps:
    def __init__(self, resources: SnapshotResourcePlan) -> None:
        self.resources = resources
        self.state: Literal["new", "prepared", "ready", "failed"] = "new"
        self.phase = "new"

    @property
    def is_idle(self) -> bool:
        """Whether the engine may start another resource operation."""
        return self.state in ("new", "ready")

    def prepare(self) -> None:
        if self.state != "new":
            raise RuntimeError(
                f"cannot prepare snapshot resources in state {self.state}"
            )
        try:
            self.phase = "prepare memory"
            self.resources.memory.prepare()
            self.phase = "prepare communicators"
            self.resources.prepare_communicators()
            self.phase = "synchronize device"
            self.resources.synchronize()
        except BaseException:
            self.state = "failed"
            raise
        self.state = "prepared"
        self.phase = "prepared"

    def restore(self) -> None:
        if self.state == "ready":
            return
        if self.state != "prepared":
            raise RuntimeError(
                f"cannot restore snapshot resources in state {self.state}"
            )
        try:
            self.phase = "restore communicators"
            self.resources.restore_communicators()
            self.phase = "restore memory"
            self.resources.memory.restore()
        except BaseException:
            self.state = "failed"
            raise
        self.state = "ready"
        self.phase = "ready"


def build_session_resources(
    *,
    sleep: Callable[[], object],
    wake_up_memory: Callable[[list[str] | None], object],
    reload_weights: Callable[[], object],
    rpc: Callable[..., object],
) -> SnapshotResourcePlan:
    """Level-2 sleep for a process-image snapshot of a private engine."""

    def restore_memory() -> None:
        wake_up_memory(["weights"])
        reload_weights()
        wake_up_memory(["kv_cache"])

    return SnapshotResourcePlan(
        memory=SnapshotMemoryPlan(sleep, restore_memory),
        prepare_communicators=lambda: rpc("checkpoint_prepare"),
        restore_communicators=lambda: rpc("checkpoint_restore"),
        synchronize=lambda: rpc("synchronize_device"),
    )


def build_checkpoint_resources(
    *,
    release_kv_cache: Callable[[], object],
    wake_up_kv_cache: Callable[[], object],
    rpc: Callable[..., object],
) -> SnapshotResourcePlan:
    """KV-only release for a checkpoint of a serving engine."""
    return SnapshotResourcePlan(
        memory=SnapshotMemoryPlan(release_kv_cache, wake_up_kv_cache),
        prepare_communicators=lambda: rpc("checkpoint_prepare"),
        restore_communicators=lambda: rpc("checkpoint_restore"),
        synchronize=lambda: rpc("synchronize_device"),
    )
