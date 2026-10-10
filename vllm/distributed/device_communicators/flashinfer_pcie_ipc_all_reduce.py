# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer PCIe CUDA-IPC all-reduce integration.

This backend targets small tensor-parallel all-reduces on a single PCIe-only
node.  Its workspace has a stricter lifetime and stream contract than the
existing FlashInfer MNNVL/TRT-LLM all-reduce implementation, so it intentionally
lives behind a separate wrapper and an opt-in environment variable.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

try:
    import flashinfer.comm as flashinfer_comm

    _pcie_ipc_available = hasattr(flashinfer_comm, "PcieIpcAllReduceWorkspace")
except ImportError:
    flashinfer_comm = None  # type: ignore[assignment]
    _pcie_ipc_available = False


class FlashInferPcieIpcAllReduce:
    """vLLM lifecycle wrapper for FlashInfer's PCIe IPC all-reduce."""

    def __init__(
        self,
        group: ProcessGroup,
        tune_group: ProcessGroup,
        device: int | str | torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.tune_group = tune_group
        self.device = torch.device(device)
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.workspace: Any | None = None
        self.hidden_dim = 0
        self.dtype: torch.dtype | None = None

        if not _pcie_ipc_available:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce was requested but this "
                "FlashInfer build does not provide PcieIpcAllReduceWorkspace; "
                "falling back to another all-reduce backend."
            )
            return
        if not current_platform.is_cuda():
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce requires the CUDA platform."
            )
            return
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce does not support world_size=%d; "
                "supported sizes are %s.",
                self.world_size,
                _SUPPORTED_WORLD_SIZES,
            )
            return
        if not all(in_the_same_node_as(tune_group, source_rank=0)):
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce requires every rank in the TP "
                "group to be on one node."
            )
            return

        # Setup is deferred until kernel_warmup, where the model hidden size and
        # exact CUDA Graph buckets are known. Until then dispatch falls through.
        self.disabled = False

    @property
    def initialized(self) -> bool:
        return not self.disabled and self.workspace is not None

    def setup(
        self,
        *,
        hidden_dim: int,
        dtype: torch.dtype,
        capture_sizes: Sequence[int],
        tune_cache: Path,
    ) -> None:
        """Allocate, tune, and prepare the exact graph-capture shapes."""
        if self.disabled or self.workspace is not None:
            return
        if dtype not in _SUPPORTED_DTYPES:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce does not support dtype=%s; "
                "falling back to another all-reduce backend.",
                dtype,
            )
            self.disabled = True
            return

        batches = tuple(sorted({int(size) for size in capture_sizes if size > 0}))
        if not batches:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce has no CUDA Graph capture sizes "
                "to prepare; falling back to another all-reduce backend."
            )
            self.disabled = True
            return

        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype
        max_numel = batches[-1] * self.hidden_dim
        workspace = flashinfer_comm.PcieIpcAllReduceWorkspace(
            group=self.group,
            max_numel=max_numel,
            dtype=dtype,
            tune_batches=batches,
            tune_cache=str(tune_cache),
        )
        self.workspace = workspace

        # tune() reuses a complete persisted cache and profiles only cache
        # misses. It is deliberately mandatory: FlashInfer's seed policy was
        # substantially slower than NCCL for the target TP4 decode workload.
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        workspace.tune(
            [self.hidden_dim],
            dtype=dtype,
            tune_group=self.tune_group,
        )
        workspace.prepare([(batch, self.hidden_dim) for batch in batches], dtype=dtype)
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        logger.info_once(
            "Initialized FlashInfer PCIe IPC all-reduce for TP%d, hidden_dim=%d, "
            "dtype=%s, max_tokens=%d.",
            self.world_size,
            self.hidden_dim,
            dtype,
            batches[-1],
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        workspace = self.workspace
        return bool(
            not self.disabled
            and workspace is not None
            and inp.is_cuda
            and inp.is_contiguous()
            and inp.dim() == 2
            and inp.shape[1] == self.hidden_dim
            and inp.dtype == self.dtype
            and workspace.supports(inp)
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self.workspace is not None
        return self.workspace.all_reduce(inp)

    @contextmanager
    def capture(self):
        """Move the single-stream workspace to and from the capture stream."""
        workspace = self.workspace
        if workspace is None:
            yield
            return

        # The caller orders graph capture against normal execution. A device
        # sync makes that assertion explicit before relaxing FlashInfer's
        # single-stream guard. Captured launches themselves are exempt from the
        # guard, while capture warmups execute and bind to the capture stream.
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        try:
            yield
        finally:
            torch.accelerator.synchronize(self.device)
            workspace.rebind_stream()

    def destroy(self) -> None:
        workspace = self.workspace
        if workspace is not None:
            workspace.destroy()
            self.workspace = None


def warmup_flashinfer_pcie_ipc_allreduce(worker: "Worker") -> None:
    """Initialize the TP PCIe IPC backend immediately before graph capture."""
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    tp_group = get_tp_group()
    communicator = tp_group.device_communicator
    if not isinstance(communicator, CudaCommunicator):
        return
    pcie_comm = communicator.fi_pcie_ipc_ar_comm
    if pcie_comm is None or pcie_comm.disabled:
        return
    if worker.vllm_config.parallel_config.use_ubatching:
        logger.warning_once(
            "FlashInfer PCIe IPC all-reduce does not yet support DBO or "
            "multi-ubatch execution; falling back to another backend."
        )
        pcie_comm.disabled = True
        return

    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    if not capture_sizes:
        return

    base_cache = resolve_flashinfer_autotune_file(worker.model_runner)
    ranks = "-".join(str(rank) for rank in tp_group.ranks)
    tune_cache = base_cache.with_name(f"pcie_ipc_allreduce_tp_{ranks}.json")
    pcie_comm.setup(
        hidden_dim=worker.model_config.get_hidden_size(),
        dtype=worker.model_config.dtype,
        capture_sizes=capture_sizes,
        tune_cache=tune_cache,
    )
