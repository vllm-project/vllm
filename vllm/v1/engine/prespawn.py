# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-spawn helper for EngineCore startup.

Fires off the EngineCore subprocess before the parent has finished building
`VllmConfig`, so the child's heavy imports (torch, vllm) overlap the
parent's own init. When the parent is ready, it sends the kwargs for
`EngineCoreProc.run_engine_core` over a ZMQ PAIR socket and the child
dispatches into the normal EngineCore boot path.

Usage pattern:
    # CLI entrypoint, before VllmConfig is constructed:
    prespawn_engine_core()

    # Later, CoreEngineProcManager calls take_pending() to find the handle
    # and send kwargs to the already-running child.
"""

import atexit
import pickle
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

import zmq

from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.utils.system_utils import get_mp_context

# Upper bound on how long the child will wait for config before exiting.
# Parent-side work is normally a few seconds; this just prevents orphaned
# children from hanging forever if the parent dies mid-startup.
_CONFIG_TIMEOUT_MS = 120_000

_pending: list["PrespawnedEngine"] = []


@dataclass
class PrespawnedEngine:
    """Handle for an EngineCore subprocess spawned before its config is ready."""

    proc: BaseProcess
    sock: zmq.Socket

    def send_config(self, **run_engine_core_kwargs) -> None:
        """Hand `run_engine_core` kwargs to the waiting child."""
        self.sock.send(pickle.dumps(run_engine_core_kwargs))
        self.sock.close()

    def shutdown(self) -> None:
        """Kill the child if it's still waiting for config (e.g. on fallback)."""
        self.sock.close(linger=0)
        if self.proc.is_alive():
            self.proc.terminate()
        self.proc.join(timeout=5.0)


def prespawn_engine_core() -> PrespawnedEngine:
    """Spawn EngineCore now; `CoreEngineProcManager` will hand it kwargs later."""
    addr = get_open_zmq_ipc_path()
    sock = zmq.Context.instance().socket(zmq.PAIR)
    sock.bind(addr)

    proc = get_mp_context().Process(
        target=_child_main,
        kwargs={"config_addr": addr},
        name="EngineCore",
    )
    proc.start()
    handle = PrespawnedEngine(proc=proc, sock=sock)
    _pending.append(handle)
    return handle


def take_pending() -> list[PrespawnedEngine]:
    """Drain and return any pre-spawned handles registered with the module."""
    handles, _pending[:] = list(_pending), []
    return handles


@atexit.register
def _cleanup_unconsumed() -> None:
    """Terminate any handles still waiting at interpreter exit (parent-side)."""
    for handle in take_pending():
        handle.shutdown()


def _child_main(config_addr: str) -> None:
    # Heavy imports here run in parallel with the parent's argparse +
    # VllmConfig construction — that's the whole point of prespawn.
    from vllm.v1.engine.core import EngineCoreProc

    sock = zmq.Context.instance().socket(zmq.PAIR)
    sock.connect(config_addr)
    # Bail out if the parent dies before sending config: PAIR sockets don't
    # surface peer death, so without a bounded wait the child would hang.
    if not sock.poll(timeout=_CONFIG_TIMEOUT_MS):
        return
    kwargs = pickle.loads(sock.recv())
    sock.close()
    EngineCoreProc.run_engine_core(**kwargs)
