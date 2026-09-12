# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torchcomms
from torchcomms import ReduceOp

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _infer_torchcomms_backend(device: torch.device | None) -> str:
    if device is None:
        raise RuntimeError(
            "Could not infer torchcomms backend because device is None. "
            "Set VLLM_TORCHCOMMS_BACKEND explicitly."
        )

    if device.type == "xpu":
        return "xccl"

    raise RuntimeError(
        f"Could not infer torchcomms backend for device={device}. "
        "Set VLLM_TORCHCOMMS_BACKEND explicitly."
    )


def _wait_if_needed(work: Any) -> None:
    if work is None:
        return
    if hasattr(work, "wait"):
        work.wait()
        return
    if hasattr(work, "wait_blocking"):
        work.wait_blocking()
        return
    raise RuntimeError(f"Unexpected torchcomms work object: {type(work)}")


def _setup_torchcomms_rank_env(
    *,
    group_rank: int,
    group_world_size: int,
) -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", group_rank))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", group_world_size))

    os.environ["RANK"] = str(group_rank)
    os.environ["WORLD_SIZE"] = str(group_world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    os.environ["TORCHCOMM_RANK"] = str(group_rank)
    os.environ["TORCHCOMM_SIZE"] = str(group_world_size)

    return local_rank, local_world_size


def _setup_backend_specific_env(
    *,
    backend: str,
    local_rank: int,
    local_world_size: int,
) -> None:
    if backend == "xccl":
        os.environ.setdefault("CCL_PROCESS_LAUNCHER", "none")
        os.environ["CCL_LOCAL_RANK"] = str(local_rank)
        os.environ["CCL_LOCAL_SIZE"] = str(local_world_size)
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")


class TorchCommsCommunicator(DeviceCommunicatorBase):
    """vLLM device communicator backed directly by torchcomms."""

    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
        )

        self.backend = os.environ.get("VLLM_TORCHCOMMS_BACKEND")
        if self.backend is None:
            self.backend = _infer_torchcomms_backend(self.device)

        self.comm = None

        self.group_rank = dist.get_rank(group=self.cpu_group)
        self.group_world_size = dist.get_world_size(group=self.cpu_group)

        if self.group_world_size == 1:
            logger.info(
                "TorchCommsCommunicator initialized with world_size=1; "
                "collectives will be local no-ops."
            )
            return

        local_rank, local_world_size = _setup_torchcomms_rank_env(
            group_rank=self.group_rank,
            group_world_size=self.group_world_size,
        )

        _setup_backend_specific_env(
            backend=self.backend,
            local_rank=local_rank,
            local_world_size=local_world_size,
        )

        logger.info(
            "[torchcomms] env before new_comm: "
            "backend=%s TORCHCOMM_RANK=%s TORCHCOMM_SIZE=%s "
            "RANK=%s WORLD_SIZE=%s LOCAL_RANK=%s LOCAL_WORLD_SIZE=%s",
            self.backend,
            os.environ.get("TORCHCOMM_RANK"),
            os.environ.get("TORCHCOMM_SIZE"),
            os.environ.get("RANK"),
            os.environ.get("WORLD_SIZE"),
            os.environ.get("LOCAL_RANK"),
            os.environ.get("LOCAL_WORLD_SIZE"),
        )

        if self.backend == "xccl":
            logger.info(
                "[torchcomms] xccl env before new_comm: "
                "CCL_PROCESS_LAUNCHER=%s CCL_LOCAL_RANK=%s CCL_LOCAL_SIZE=%s "
                "CCL_ATL_TRANSPORT=%s CCL_LOG_LEVEL=%s",
                os.environ.get("CCL_PROCESS_LAUNCHER"),
                os.environ.get("CCL_LOCAL_RANK"),
                os.environ.get("CCL_LOCAL_SIZE"),
                os.environ.get("CCL_ATL_TRANSPORT"),
                os.environ.get("CCL_LOG_LEVEL"),
            )

        comm_name = f"{unique_name or 'vllm'}_torchcomms"

        orig_master_addr = os.environ.get("MASTER_ADDR")
        orig_master_port = os.environ.get("MASTER_PORT")

        torchcomm_addr = os.environ.get(
            "TORCHCOMM_MASTER_ADDR",
            os.environ.get("MASTER_ADDR", "127.0.0.1"),
        )
        torchcomm_port = os.environ.get("TORCHCOMM_MASTER_PORT")

        if torchcomm_port is None:
            raise RuntimeError(
                "TORCHCOMM_MASTER_PORT is not set. TorchComms needs a separate "
                "rendezvous port from MASTER_PORT to avoid port collisions."
            )

        os.environ["MASTER_ADDR"] = torchcomm_addr
        os.environ["MASTER_PORT"] = str(torchcomm_port)

        logger.info(
            "[torchcomms] using rendezvous MASTER_ADDR=%s MASTER_PORT=%s "
            "original_MASTER_ADDR=%s original_MASTER_PORT=%s "
            "TORCHCOMM_MASTER_ADDR=%s TORCHCOMM_MASTER_PORT=%s name=%s",
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
            orig_master_addr,
            orig_master_port,
            os.environ.get("TORCHCOMM_MASTER_ADDR"),
            os.environ.get("TORCHCOMM_MASTER_PORT"),
            comm_name,
        )

        try:
            self.comm = torchcomms.new_comm(
                self.backend,
                self.device,
                name=comm_name,
            )
        except Exception:
            self.comm = None
            logger.exception(
                "[torchcomms] new_comm failed backend=%s device=%s "
                "name=%s group_rank=%s group_world_size=%s",
                self.backend,
                self.device,
                comm_name,
                self.group_rank,
                self.group_world_size,
            )
            raise

        finally:
            if orig_master_addr is not None:
                os.environ["MASTER_ADDR"] = orig_master_addr
            else:
                os.environ.pop("MASTER_ADDR", None)

            if orig_master_port is not None:
                os.environ["MASTER_PORT"] = orig_master_port
            else:
                os.environ.pop("MASTER_PORT", None)

        logger.info(
            "[torchcomms] initialized backend=%s device=%s group_rank=%s "
            "group_world_size=%s unique_name=%s",
            self.backend,
            self.device,
            self.group_rank,
            self.group_world_size,
            unique_name,
        )

    def _get_comm(self) -> Any:
        if self.comm is None:
            raise RuntimeError("TorchComms communicator is not initialized")
        return self.comm

    def _log_tensor(self, op: str, tensor: torch.Tensor) -> None:
        logger.info(
            "[torchcomms] rank=%s op=%s shape=%s dtype=%s device=%s",
            self.group_rank,
            op,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self._log_tensor("all_reduce:start", input_)

        if self.group_world_size == 1:
            self._log_tensor("all_reduce:done_world_size_1", input_)
            return input_

        output = input_.clone()

        logger.info(
            " ".join(
                (
                    "[torchcomms] all_reduce backend=%s rank=%s input_shape=%s",
                    "dtype=%s device=%s",
                )
            ),
            self.backend,
            self.group_rank,
            tuple(input_.shape),
            input_.dtype,
            input_.device,
        )

        comm = self._get_comm()
        work = comm.all_reduce(
            output,
            ReduceOp.SUM,
            async_op=False,
        )
        _wait_if_needed(work)

        self._log_tensor("all_reduce:done", output)
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._log_tensor("all_gather:start", input_)

        if self.group_world_size == 1:
            self._log_tensor("all_gather:done_world_size_1", input_)
            return input_

        if dim < 0:
            dim += input_.dim()

        moved = input_.movedim(dim, 0).contiguous()

        output = torch.empty(
            (self.group_world_size * moved.shape[0],) + moved.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )

        logger.info(
            "[torchcomms] all_gather backend=%s rank=%s input_shape=%s "
            "moved_shape=%s output_shape=%s dim=%s",
            self.backend,
            self.group_rank,
            tuple(input_.shape),
            tuple(moved.shape),
            tuple(output.shape),
            dim,
        )

        comm = self._get_comm()
        work = comm.all_gather_single(
            output,
            moved,
            async_op=False,
        )
        _wait_if_needed(work)

        result = output.movedim(0, dim).contiguous()
        self._log_tensor("all_gather:done", result)
        return result

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._log_tensor("reduce_scatter:start", input_)

        if self.group_world_size == 1:
            self._log_tensor("reduce_scatter:done_world_size_1", input_)
            return input_

        if dim < 0:
            dim += input_.dim()

        moved = input_.movedim(dim, 0).contiguous()

        if moved.shape[0] % self.group_world_size != 0:
            raise RuntimeError(
                "reduce_scatter requires the scatter dimension to be divisible "
                f"by world_size. shape={tuple(input_.shape)}, dim={dim}, "
                f"world_size={self.group_world_size}"
            )

        chunk = moved.shape[0] // self.group_world_size

        output = torch.empty(
            (chunk,) + moved.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )

        logger.info(
            "[torchcomms] reduce_scatter backend=%s rank=%s input_shape=%s "
            "moved_shape=%s output_shape=%s dim=%s",
            self.backend,
            self.group_rank,
            tuple(input_.shape),
            tuple(moved.shape),
            tuple(output.shape),
            dim,
        )

        comm = self._get_comm()
        work = comm.reduce_scatter_single(
            output,
            moved,
            ReduceOp.SUM,
            async_op=False,
        )
        _wait_if_needed(work)

        result = output.movedim(0, dim).contiguous()
        self._log_tensor("reduce_scatter:done", result)
        return result

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        self._log_tensor("broadcast:start", tensor)

        if self.group_world_size == 1:
            self._log_tensor("broadcast:done_world_size_1", tensor)
            return tensor

        logger.info(
            " ".join(
                (
                    "[torchcomms] broadcast backend=%s rank=%s src=%s shape=%s",
                    "dtype=%s device=%s",
                )
            ),
            self.backend,
            self.group_rank,
            src,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

        comm = self._get_comm()
        work = comm.broadcast(
            tensor,
            root=src,
            async_op=False,
        )
        _wait_if_needed(work)

        self._log_tensor("broadcast:done", tensor)
        return tensor

    def destroy(self) -> None:
        # Temporary debugging workaround:
        # Do not call torchcomms finalize here because XCCL watchdog cleanup
        # is currently crashing in TorchCommXCCL::timeoutWatchdog().
        #
        # Let the process exit without explicitly finalizing this communicator.
        return

    def __del__(self) -> None:
        # Never run TorchComms cleanup from Python object destruction.
        return
