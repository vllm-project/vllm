# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async NIXL CPU DRAM <-> S3 transfer engine."""

import contextlib
from collections import deque
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle

from vllm.logger import init_logger

from vllm.v1.kv_offload.tiering.obj.obj_store_config import ObjStoreConfig

logger = init_logger(__name__)

WRITE = "WRITE"
READ = "READ"


class TransferEntry(NamedTuple):
    job_id: int
    xfer_handle: nixl_xfer_handle
    files_desc: object


class NixlEngine:
    """Manages async CPU DRAM <-> S3 transfers via NIXL."""

    def __init__(self, obj_config: ObjStoreConfig, io_threads: int = 4):
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjNixlEngine", agent_config)
        params = {**obj_config.to_nixl_params(), "num_threads": str(io_threads)}
        self._agent.create_backend("OBJ", params)

        self._in_flight: deque[TransferEntry] = deque()
        self._primary_tensor: torch.Tensor | None = None
        self._primary_reg = None
        self._base_addr: int = 0
        self._stride: int = 0

    def set_primary_view(self, view: memoryview) -> None:
        """Register the entire primary CPU buffer with NIXL once."""
        np_arr = np.asarray(view)
        self._primary_tensor = torch.as_tensor(np_arr)
        self._primary_reg = self._agent.register_memory([self._primary_tensor])
        self._base_addr = self._primary_tensor.data_ptr()
        self._stride = view.strides[0]

    def submit_transfer(
        self,
        job_id: int,
        block_ids: Iterable[int],
        s3_keys: Iterable[str],
        op: str,
    ) -> bool:
        """Submit an async transfer. op is 'WRITE' (store) or 'READ' (load)."""
        blocks_data = [
            (self._base_addr + int(bid) * self._stride, self._stride, 0)
            for bid in block_ids
        ]
        nixl_files = [
            (0, self._stride, 0, key) for key in s3_keys
        ]

        xfer_desc = self._agent.get_xfer_descs(blocks_data, "DRAM")
        if xfer_desc is None:
            logger.error("get_xfer_descs failed for job %d", job_id)
            return False

        files_desc = self._agent.register_memory(nixl_files, "OBJ")
        if files_desc is None:
            logger.error("register_memory (OBJ) failed for job %d", job_id)
            return False

        xfer_handle = self._agent.initialize_xfer(
            op, xfer_desc, files_desc.trim(), "ObjNixlEngine"
        )
        if not xfer_handle:
            logger.error("initialize_xfer failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            return False

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.error("agent.transfer failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            self._agent.release_xfer_handle(xfer_handle)
            return False

        self._in_flight.append(TransferEntry(job_id, xfer_handle, files_desc))
        return True

    def get_finished(self) -> list[tuple[int, bool]]:
        """Poll in-flight transfers; return completed (job_id, success) pairs."""
        results: list[tuple[int, bool]] = []
        still_in_flight: deque[TransferEntry] = deque()

        for entry in self._in_flight:
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                logger.error("check_xfer_state raised for job %d: %s", entry.job_id, exc)
                state = "ERR"
            if state == "PROC":
                still_in_flight.append(entry)
                continue
            self._agent.deregister_memory(entry.files_desc)
            self._agent.release_xfer_handle(entry.xfer_handle)
            success = state == "DONE"
            if not success:
                logger.error("transfer failed job=%d state=%s", entry.job_id, state)
            results.append((entry.job_id, success))

        self._in_flight = still_in_flight
        return results

    def shutdown(self) -> None:
        for entry in self._in_flight:
            self._agent.deregister_memory(entry.files_desc)
            self._agent.release_xfer_handle(entry.xfer_handle)
        self._in_flight.clear()
        if self._primary_reg is not None:
            self._agent.deregister_memory(self._primary_reg)
            self._primary_reg = None
        self._primary_tensor = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
