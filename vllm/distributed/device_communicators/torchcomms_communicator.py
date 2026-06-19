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


class TorchCommsCommunicator(DeviceCommunicatorBase):
    """vLLM device communicator backed directly by torchcomms"""

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

        self.backend = os.environ.get(
            "VLLM_TORCHCOMMS_BACKEND")
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


        local_rank = int(os.environ.get("LOCAL_RANK", self.group_rank))
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.group_world_size))

        os.environ["TORCHCOMM_RANK"] = str(self.group_rank)
        os.environ["TORCHCOMM_SIZE"] = str(self.group_world_size)


        os.environ.setdefault("CCL_PROCESS_LAUNCHER", "none")
        os.environ["CCL_LOCAL_RANK"] = str(local_rank)
        os.environ["CCL_LOCAL_SIZE"] = str(local_size)

        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")


        logger.info(
        "[torchcomms] env before new_comm: "
        "TORCHCOMM_RANK=%s TORCHCOMM_SIZE=%s "
        "CCL_PROCESS_LAUNCHER=%s CCL_LOCAL_RANK=%s CCL_LOCAL_SIZE=%s "
        "CCL_ATL_TRANSPORT=%s CCL_LOG_LEVEL=%s",
        os.environ.get("TORCHCOMM_RANK"),
        os.environ.get("TORCHCOMM_SIZE"),
        os.environ.get("CCL_PROCESS_LAUNCHER"),
        os.environ.get("CCL_LOCAL_RANK"),
        os.environ.get("CCL_LOCAL_SIZE"),
        os.environ.get("CCL_ATL_TRANSPORT"),
        os.environ.get("CCL_LOG_LEVEL"),
        )
        self.comm = torchcomms.new_comm(
            self.backend,
            self.device,
            name=f"{unique_name or 'vllm'}_torchcomms",
        )

        logger.info(
            "[torchcomms] initialized backend=%s device=%s group_rank=%s "
            "group_world_size=%s unique_name=%s",
            self.backend,
            self.device,
            self.group_rank,
            self.group_world_size,
            unique_name,
        )

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
            "[torchcomms] all_reduce backend=%s rank=%s input_shape=%s dtype=%s device=%s",
            self.backend,
            self.group_rank,
            tuple(input_.shape),
            input_.dtype,
            input_.device,
        )

        work = self.comm.all_reduce(
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

        work = self.comm.all_gather_single(
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

        work = self.comm.reduce_scatter_single(
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
            "[torchcomms] broadcast backend=%s rank=%s src=%s shape=%s dtype=%s device=%s",
            self.backend,
            self.group_rank,
            src,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )

        work = self.comm.broadcast(
            tensor,
            root=src,
            async_op=False,
        )
        _wait_if_needed(work)

        self._log_tensor("broadcast:done", tensor)
        return tensor

    def destroy(self) -> None:
        logger.info(
            "[torchcomms] destroy backend=%s rank=%s world_size=%s",
            self.backend,
            self.group_rank,
            self.group_world_size,
        )

        if self.comm is not None and hasattr(self.comm, "finalize"):
            self.comm.finalize()

        self.comm = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
