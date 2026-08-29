# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast KV side channels for the NIXL pull connector.

Optional ipc channels between the EngineCore and TP worker processes that
bypass the per-step SchedulerOutput/ModelRunnerOutput round trip: fast
dispatch (EngineCore PUB -> worker SUB) starts NIXL reads right after
schedule(), and fast notify (worker PUSH -> EngineCore PULL) reports pull
completion without waiting for the step boundary. Both are hints only; the
regular get_finished() path still reports every request (scheduler dedups).
Enable via kv_connector_extra_config: fast_kv_dispatch / fast_kv_notify.
"""

import contextlib
import os
import pickle
import threading
from typing import TYPE_CHECKING, Any

import zmq

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_PULL_CONNECTORS = ("NixlConnector", "NixlPullConnector")


def fast_dispatch_enabled(vllm_config: VllmConfig) -> bool:
    ktc = vllm_config.kv_transfer_config
    return ktc is not None and bool(
        ktc.get_from_extra_config("fast_kv_dispatch", False)
    )


def fast_notify_enabled(vllm_config: VllmConfig) -> bool:
    ktc = vllm_config.kv_transfer_config
    return ktc is not None and bool(ktc.get_from_extra_config("fast_kv_notify", False))


def _fast_kv_base_port(vllm_config: VllmConfig) -> int:
    # Unique per vLLM instance on this host, like the NIXL side channel port.
    return (
        envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        + vllm_config.parallel_config.data_parallel_index
    )


def fast_kv_dispatch_path(vllm_config: VllmConfig) -> str:
    return f"ipc:///tmp/vllm_nixl_fkd_{_fast_kv_base_port(vllm_config)}.sock"


def fast_kv_notify_path(vllm_config: VllmConfig) -> str:
    return f"ipc:///tmp/vllm_nixl_fkn_{_fast_kv_base_port(vllm_config)}.sock"


def _unlink_ipc_path(path: str) -> None:
    if path.startswith("ipc://"):
        with contextlib.suppress(OSError):
            os.unlink(path[len("ipc://") :])


class NixlFastKVEngineCoreBridge:
    """EngineCore-process endpoints of the fast KV side channels (owns the
    bound ends of both ipc sockets; workers connect)."""

    def __init__(self, vllm_config: VllmConfig, scheduler: Any, output_queue: Any):
        self.scheduler = scheduler
        self.output_queue = output_queue
        self.world_size = vllm_config.parallel_config.world_size
        self.dispatch_enabled = fast_dispatch_enabled(vllm_config)
        self.notify_enabled = fast_notify_enabled(vllm_config) and hasattr(
            scheduler, "on_fast_kv_recv_finished"
        )
        # Scheduler-side connector, for on_pd_kv_ready (early-arm).
        self.connector_scheduler = getattr(
            getattr(scheduler, "connector", None), "connector_scheduler", None
        )
        self._ctx = zmq.Context()
        self._stop_event = threading.Event()
        self._dispatch_sock: zmq.Socket | None = None
        # The dispatch socket is written from the busy loop and the notify
        # thread; zmq sockets are not thread-safe.
        self._dispatch_lock = threading.Lock()
        self._notify_t: threading.Thread | None = None
        if self.dispatch_enabled:
            path = fast_kv_dispatch_path(vllm_config)
            _unlink_ipc_path(path)
            sock = self._ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(path)
            self._dispatch_sock = sock
        if self.notify_enabled:
            self._notify_path = fast_kv_notify_path(vllm_config)
            _unlink_ipc_path(self._notify_path)
            self._notify_t = threading.Thread(
                target=self._notify_loop,
                daemon=True,
                name="nixl-fast-kv-notify",
            )
            self._notify_t.start()
        logger.info(
            "NIXL fast KV EngineCore bridge initialized "
            "(dispatch=%s notify=%s world_size=%d)",
            self.dispatch_enabled,
            self.notify_enabled,
            self.world_size,
        )

    def dispatch(self, scheduler_output: "SchedulerOutput") -> None:
        """Publish new pull metadata to the workers right after schedule()."""
        if self._dispatch_sock is None:
            return
        meta = scheduler_output.kv_connector_metadata
        reqs_to_recv = getattr(meta, "reqs_to_recv", None)
        if not reqs_to_recv:
            return
        try:
            with self._dispatch_lock:
                self._dispatch_sock.send(pickle.dumps(reqs_to_recv))
        except Exception:
            logger.exception(
                "Fast KV dispatch failed; workers will pick the metadata up "
                "with the next execute_model call."
            )

    def _notify_loop(self):
        sock = self._ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(self._notify_path)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        pending: dict[str, set[int]] = {}
        try:
            while not self._stop_event.is_set():
                if not poller.poll(timeout=200):
                    continue
                msg = pickle.loads(sock.recv())
                if len(msg) == 3 and msg[0] == "pd_ready":
                    self._handle_pd_ready(msg[1], msg[2])
                    continue
                req_id, rank = msg
                ranks = pending.setdefault(req_id, set())
                ranks.add(rank)
                if len(ranks) < self.world_size:
                    continue
                del pending[req_id]
                try:
                    result = self.scheduler.on_fast_kv_recv_finished(req_id)
                except Exception:
                    logger.exception(
                        "on_fast_kv_recv_finished failed for %s; the regular "
                        "get_finished path will resume the request.",
                        req_id,
                    )
                    continue
                if result is None:
                    continue
                client_index, engine_core_output = result
                # Same cross-thread pattern as _send_finish_outputs_to_client.
                from vllm.v1.engine import EngineCoreOutputs

                self.output_queue.put_nowait(
                    (client_index, EngineCoreOutputs(outputs=[engine_core_output]))
                )
        except Exception:
            logger.exception(
                "NIXL fast-notify bridge thread died; completions fall back "
                "to the regular per-step path."
            )
        finally:
            sock.close(linger=0)

    def _handle_pd_ready(self, raw_request_id: str, kv_transfer_params: dict) -> None:
        """Early-arm: merge the ready params into the armed request and
        fast-publish its pull metadata. Runs on the notify thread."""
        cs = self.connector_scheduler
        if cs is None or not getattr(cs, "pd_early_arm_enabled", False):
            return
        try:
            result = cs.on_pd_kv_ready(raw_request_id, kv_transfer_params)
        except Exception:
            logger.exception(
                "on_pd_kv_ready failed for %s; the armed-timeout fallback "
                "will recompute locally.",
                raw_request_id,
            )
            return
        if result is None or self._dispatch_sock is None:
            # The next step's connector metadata delivers the pull.
            return
        req_id, req_meta = result
        try:
            with self._dispatch_lock:
                self._dispatch_sock.send(pickle.dumps({req_id: req_meta}))
        except Exception:
            logger.exception(
                "Fast publish of pd_ready pull metadata failed for %s; the "
                "next step's connector metadata delivers it.",
                req_id,
            )

    def shutdown(self):
        self._stop_event.set()
        if self._dispatch_sock is not None:
            with contextlib.suppress(Exception):
                self._dispatch_sock.close(linger=0)
            self._dispatch_sock = None
        if self._notify_t is not None and self._notify_t.is_alive():
            self._notify_t.join(timeout=1.0)
        self._notify_t = None
        # The zmq context is deliberately not term()'d: term would block on
        # sockets owned by already-killed daemon threads.


def maybe_create_fast_kv_bridge(
    vllm_config: VllmConfig, scheduler: Any, output_queue: Any
) -> NixlFastKVEngineCoreBridge | None:
    """Create the bridge iff the NIXL pull connector is in use and at least
    one fast path is enabled."""
    ktc = vllm_config.kv_transfer_config
    if ktc is None or ktc.kv_connector not in _PULL_CONNECTORS:
        return None
    if not (fast_dispatch_enabled(vllm_config) or fast_notify_enabled(vllm_config)):
        return None
    return NixlFastKVEngineCoreBridge(vllm_config, scheduler, output_queue)
