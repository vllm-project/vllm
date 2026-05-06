# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async NIXL CPU DRAM <-> S3 transfer engine."""

import contextlib
from collections import deque
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.obj.nixl_lookup import obj_key_to_dev_id

logger = init_logger(__name__)


class _TransferEntry(NamedTuple):
    job_id: int
    xfer_handle: object
    files_desc: object


class NixlEngine:
    """Manages async CPU DRAM <-> S3 transfers via NIXL."""

    def __init__(
        self,
        bucket: str,
        endpoint_override: str,
        access_key: str,
        secret_key: str,
        scheme: str = "http",
        ca_bundle: str = "",
        io_threads: int = 4,
    ):
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjNixlEngine", agent_config)
        params: dict[str, str] = {
            "bucket": bucket,
            "endpoint_override": endpoint_override,
            "scheme": scheme,
            "access_key": access_key,
            "secret_key": secret_key,
            "num_threads": str(io_threads),
        }
        if ca_bundle:
            params["ca_bundle"] = ca_bundle
        self._agent.create_backend("OBJ", params)

        self._in_flight: deque[_TransferEntry] = deque()
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
        block_ids: Sequence[int],
        s3_keys: Sequence[str],
        op: str,
    ) -> bool:
        """Submit an async transfer. op is 'WRITE' (store) or 'READ' (load)."""
        blocks_data = [
            (self._base_addr + int(bid) * self._stride, self._stride, 0)
            for bid in block_ids
        ]
        nixl_files = [
            (0, self._stride, obj_key_to_dev_id(key), key) for key in s3_keys
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

        self._in_flight.append(_TransferEntry(job_id, xfer_handle, files_desc))
        return True

    def get_finished(self) -> list[tuple[int, bool]]:
        """Poll in-flight transfers; return completed (job_id, success) pairs."""
        results: list[tuple[int, bool]] = []
        to_remove: list[_TransferEntry] = []

        for entry in self._in_flight:
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                logger.error("check_xfer_state raised for job %d: %s", entry.job_id, exc)
                state = "ERR"
            if state == "PROC":
                continue
            self._agent.deregister_memory(entry.files_desc)
            self._agent.release_xfer_handle(entry.xfer_handle)
            success = state == "DONE"
            if not success:
                logger.error("transfer failed job=%d state=%s", entry.job_id, state)
            results.append((entry.job_id, success))
            to_remove.append(entry)

        for entry in to_remove:
            self._in_flight.remove(entry)
        return results

    def shutdown(self) -> None:
        if self._primary_reg is not None:
            self._agent.deregister_memory(self._primary_reg)
            self._primary_reg = None
        self._primary_tensor = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
