# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRI-IO implementation of registered peer reads."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import msgspec
import torch

from ..transport import PeerReadTransport, ReadBatch, ReadState, RemoteSession


@dataclass(frozen=True)
class Advertisement:
    version: int
    engine: bytes
    memory: bytes


class MoRIBatch:
    """Keep owned submission arguments and status alive across both worker threads."""

    def __init__(self, submission: Any):
        self.submission = submission

    def state(self) -> ReadState:
        if self.submission.status.Succeeded():
            return ReadState.SUCCEEDED
        if self.submission.status.Failed():
            return ReadState.FAILED
        return ReadState.PENDING

    def quiesced(self) -> bool:
        return self.submission.status.Succeeded()


class MoRISession:
    """Prepare and post MoRI reads on the worker I/O thread.

    The session and remote memory descriptors are keep-alives for submissions.
    """

    def __init__(self, wrapper: Any, session: Any, memory: Any):
        self.wrapper, self.session, self.memory = wrapper, session, memory

    def prepare(self, ranges: Sequence[tuple[int, int, int]]) -> MoRIBatch:
        local, remote, sizes = map(list, zip(*ranges))
        return MoRIBatch(
            self.wrapper.prepare_owned_read(sizes, local, remote, self.memory)
        )

    def post(self, batch: ReadBatch) -> None:
        assert isinstance(batch, MoRIBatch)
        self.wrapper.submit_owned_read(batch.submission)


class MoRITransport(PeerReadTransport):
    """Register worker KV memory at startup and open MoRI sessions on its I/O thread."""

    name = "mori"
    max_ranges_per_post = 64
    max_range_bytes = 16 << 20

    @classmethod
    def available(cls) -> bool:
        try:
            import mori.io  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, storage: torch.UntypedStorage, host: str) -> None:
        os.environ.setdefault("MORI_DISABLE_AUTO_XGMI", "1")
        if os.environ["MORI_DISABLE_AUTO_XGMI"] != "1":
            raise ValueError("RDMA peers require MORI_DISABLE_AUTO_XGMI=1")
        from mori.io import BackendType, IOEngine, IOEngineConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
            MoRIIOWrapper,
        )

        self.tensor = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        self.engine = IOEngine(
            f"peer-l1-{uuid4().hex}", IOEngineConfig(host=host, port=0)
        )
        self.wrapper = MoRIIOWrapper(self.engine)
        self.wrapper.set_backend_type(BackendType.RDMA)
        memory = self.wrapper.register_local_tensor(self.tensor)
        self._advertisement = msgspec.json.encode(
            Advertisement(1, self.wrapper.get_agent_metadata(), memory)
        )

    def advertisement(self) -> bytes:
        return self._advertisement

    def open(self, advertisement: bytes) -> RemoteSession:
        advertised = msgspec.json.decode(advertisement, type=Advertisement)
        if advertised.version != 1:
            raise ValueError("Unsupported MoRI peer advertisement version")
        self.wrapper.register_remote_engine(advertised.engine)
        memory = self.wrapper.get_unpack_memory_metadata(advertised.memory)
        session = self.wrapper.build_session(self.wrapper.local_memory_metadata, memory)
        if session is None:
            raise RuntimeError("MoRI could not create an RDMA read session")
        return MoRISession(self.wrapper, session, memory)

    def close(self) -> None:
        # Published registrations cannot be revoked safely.
        pass
