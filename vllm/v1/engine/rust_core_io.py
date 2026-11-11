# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native I/O of the Rust engine core, which serves the Rust frontend.

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
from vllm.v1.engine import EngineCoreOutputs, PrefixHashConfig
from vllm.v1.serial_utils import MsgpackEncoder

# `EngineCoreProc.ENGINE_CORE_DEAD`.
_ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"


def native_input_unsupported_reason(
    vllm_config: VllmConfig, prefix_hash_config: PrefixHashConfig | None
) -> str | None:
    """Why the engine core cannot use the native input path, which needs the
    native output path too, if so."""
    # The hooks of these connectors take Python requests, which the fast path
    # does without.
    if vllm_config.kv_transfer_config is not None:
        return "it does not support KV connectors"
    if vllm_config.ec_transfer_config is not None:
        return "it does not support EC connectors"
    if vllm_config.aux_output_config.enabled:
        return "it does not support aux output connectors"
    if prefix_hash_config is not None and not prefix_hash_config.hash_algo.endswith(
        "_cbor"
    ):
        return (
            f"it hashes prompt blocks with *_cbor algorithms only, not "
            f"{prefix_hash_config.hash_algo}"
        )
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
