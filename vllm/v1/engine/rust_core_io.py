# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native output path of the Rust engine core (`VLLM_RUST_CORE_NATIVE_IO`).

The Rust scheduler sends the request outputs of each step itself: the output
thread of `vllm._rust_core` encodes them in the Rust frontend's wire types,
with no Python `EngineCoreOutputs` built. Everything else the engine sends
(utility results, DP coordinator messages, error outputs, ...) goes through
`NativeOutputQueue`, which encodes it in Python as before and hands the frames
to the same thread, so the order of messages is kept.
"""

import threading
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.serial_utils import MsgpackEncoder

# `EngineCoreProc.ENGINE_CORE_DEAD`.
_ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"


def native_output_unsupported_reason(
    vllm_config: VllmConfig, scheduler: SchedulerInterface
) -> str | None:
    """Why the engine core cannot use the native output path, if so."""
    from vllm.v1.core.sched.rust_scheduler import RustScheduler

    if not isinstance(scheduler, RustScheduler):
        return "it needs the Rust scheduler"
    # TODO: pooling outputs, which the Rust frontend does not request yet.
    if vllm_config.model_config.runner_type == "pooling":
        return "it does not support pooling models"
    return None


class NativeOutputQueue:
    """`EngineCoreProc.output_queue` on the native output path: encodes each
    item as the output thread would, and hands it to the native output thread.
    """

    def __init__(self, writer: Any, engine_index: int):
        self._writer = writer
        self._engine_index = engine_index
        # `MsgpackEncoder` is not thread-safe, and the busy loop, the input
        # thread and the fault-tolerance sentinel all send outputs.
        self._local = threading.local()

    def put_nowait(self, item: tuple[int, EngineCoreOutputs] | bytes) -> None:
        if item == _ENGINE_CORE_DEAD:
            self._writer.send_dead()
            return
        assert not isinstance(item, bytes)
        client_index, outputs = item
        outputs.engine_index = self._engine_index
        encoder = getattr(self._local, "encoder", None)
        if encoder is None:
            encoder = self._local.encoder = MsgpackEncoder()
        # Copy the frames that view tensor memory: the output thread sends them
        # later, without the GIL.
        frames = [
            b if isinstance(b, bytes) else bytes(b) for b in encoder.encode(outputs)
        ]
        self._writer.send_encoded(client_index, frames)
