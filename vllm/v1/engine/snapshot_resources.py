# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from typing import TYPE_CHECKING

from vllm.snapshot.resources import (
    SnapshotEngineOps,
    SnapshotResourcePlan,
    build_checkpoint_resources,
    build_session_resources,
)
from vllm.v1.engine import CheckpointState

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore


class EngineSnapshotResources:
    """Bind snapshot resource plans to one engine's state and operations.

    The shared layer owns operation ordering and failure state. This adapter
    owns the engine checks and the callbacks into the executor and workers.
    """

    def __init__(self, engine: "EngineCore") -> None:
        self.engine = engine
        self.ops: SnapshotEngineOps | None = None

    @property
    def is_idle(self) -> bool:
        return self.ops is None or self.ops.is_idle

    @property
    def phase(self) -> str:
        return "new" if self.ops is None else self.ops.phase

    def require_idle(self) -> None:
        if not self.is_idle:
            raise RuntimeError(
                f"Snapshot resources are unavailable in phase {self.phase}"
            )

    def prepare_checkpoint(self) -> None:
        self.require_idle()
        self._start(
            build_checkpoint_resources(
                release_kv_cache=self.engine._release_kv_cache_memory,
                wake_up_kv_cache=partial(self.engine._wake_up_memory, ["kv_cache"]),
                rpc=self.engine.collective_rpc,
            )
        )

    def prepare_session(self) -> None:
        engine = self.engine
        if engine._checkpoint_state != CheckpointState.RUNNING:
            raise RuntimeError(
                f"Cannot prepare snapshot resources while checkpoint state is "
                f"{engine._checkpoint_state}"
            )
        if not engine.vllm_config.model_config.enable_sleep_mode:
            raise RuntimeError("Snapshot resources require --enable-sleep-mode")
        self.require_idle()
        if any(engine.collective_rpc("has_incomplete_weight_update")):
            raise RuntimeError("Snapshot preparation requires completed weight updates")
        if (
            not engine.is_scheduler_paused()
            or engine.scheduler.has_requests()
            or engine.batch_queue
            or engine.model_executor.is_sleeping
        ):
            raise RuntimeError("Snapshot preparation requires a completed idle pause")
        self._start(
            build_session_resources(
                sleep=partial(engine.model_executor.sleep, 2),
                wake_up_memory=engine._wake_up_memory,
                reload_weights=partial(engine.collective_rpc, "reload_weights"),
                rpc=engine.collective_rpc,
            )
        )

    def restore(self) -> None:
        if self.ops is None:
            raise RuntimeError("Snapshot resources have not been prepared")
        self.ops.restore()

    def _start(self, resources: SnapshotResourcePlan) -> None:
        self.ops = SnapshotEngineOps(resources)
        self.ops.prepare()
