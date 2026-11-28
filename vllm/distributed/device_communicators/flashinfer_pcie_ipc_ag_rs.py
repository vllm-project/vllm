# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer PCIe CUDA-IPC all-gather and reduce-scatter integration."""

import json
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flashinfer
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.custom_all_reduce import (
    _all_ranks_true,
    _can_p2p,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

try:
    import flashinfer.comm as flashinfer_comm

    _pcie_ipc_available = hasattr(
        flashinfer_comm, "PcieIpcAllGatherWorkspace"
    ) and hasattr(flashinfer_comm, "PcieIpcReduceScatterWorkspace")
except ImportError:
    flashinfer_comm = None  # type: ignore[assignment]
    _pcie_ipc_available = False


class FlashInferPcieIpcAgRs:
    """Own the paired EP workspaces and their stream lifetimes."""

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
        self.ag_workspace: Any | None = None
        self.rs_workspace: Any | None = None
        self.hidden_dim = 0
        self.dtype: torch.dtype | None = None

        if not _pcie_ipc_available:
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS was requested but this FlashInfer "
                "build does not provide its collective workspaces; using NCCL."
            )
            return
        if not current_platform.is_cuda():
            logger.warning_once("FlashInfer PCIe IPC AG/RS requires CUDA.")
            return
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS does not support world_size=%d; "
                "supported sizes are %s.",
                self.world_size,
                _SUPPORTED_WORLD_SIZES,
            )
            return
        if not all(in_the_same_node_as(tune_group, source_rank=0)):
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS requires every EP rank on one node."
            )
            return
        if not _all_ranks_true(tune_group, _can_p2p(self.rank, self.world_size)):
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS requires CUDA peer access between "
                "all ranks; using NCCL."
            )
            return

        self.disabled = False

    def _cache_metadata(
        self, batches: tuple[int, ...], ag_hiddens: list[int]
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "world_size": self.world_size,
            "max_numel": batches[-1] * self.hidden_dim,
            "capture_sizes": list(batches),
            "ag_hiddens": ag_hiddens,
            "dtype": str(self.dtype),
            "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
            "device_name": torch.cuda.get_device_name(self.device),
            "device_capability": list(torch.cuda.get_device_capability(self.device)),
        }

    def _cache_complete(self, tune_cache: Path, metadata: dict[str, Any]) -> bool:
        paths = [
            tune_cache.with_name(f"ag_{tune_cache.name}"),
            tune_cache.with_name(f"rs_{tune_cache.name}"),
        ]
        if not all(path.is_file() for path in paths):
            return False
        try:
            cached = json.loads(tune_cache.with_suffix(".complete").read_text())
        except (OSError, ValueError, TypeError):
            return False
        if cached != metadata:
            differences = [
                f"{key}: cached={cached.get(key)!r}, requested={metadata[key]!r}"
                for key in metadata
                if cached.get(key) != metadata[key]
            ]
            logger.info_once(
                "PCIe IPC tuning metadata mismatch at %s: %s",
                tune_cache,
                "; ".join(differences),
            )
        return cached == metadata

    def _write_cache_complete(self, tune_cache: Path, metadata: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        path = tune_cache.with_suffix(".complete")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, sort_keys=True))
        os.replace(temporary, path)

    def setup(
        self,
        *,
        hidden_dim: int,
        dtype: torch.dtype,
        capture_sizes: Sequence[int],
        tune_cache: Path,
    ) -> None:
        if self.disabled or self.ag_workspace is not None:
            return
        if dtype not in _SUPPORTED_DTYPES:
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS does not support dtype=%s; using NCCL.",
                dtype,
            )
            self.disabled = True
            return

        batches = tuple(sorted({int(size) for size in capture_sizes if size > 0}))
        if not batches:
            logger.warning_once(
                "FlashInfer PCIe IPC AG/RS has no CUDA Graph capture sizes "
                "to tune; using NCCL."
            )
            self.disabled = True
            return

        setup_started = time.monotonic()
        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype
        max_numel = batches[-1] * self.hidden_dim
        ag_hiddens = [self.hidden_dim]
        if self.dtype == torch.bfloat16 and self.hidden_dim % 2 == 0:
            ag_hiddens.append(self.hidden_dim // 2)
        metadata = self._cache_metadata(batches, ag_hiddens)
        cache_hit = _all_ranks_true(
            self.tune_group,
            self._cache_complete(tune_cache, metadata),
        )
        common = dict(
            group=self.group,
            max_numel=max_numel,
            dtype=dtype,
            tune_batches=batches,
        )
        self.ag_workspace = flashinfer_comm.PcieIpcAllGatherWorkspace(
            **common,
            tune_cache=str(tune_cache.with_name(f"ag_{tune_cache.name}")),
        )
        self.rs_workspace = flashinfer_comm.PcieIpcReduceScatterWorkspace(
            **common,
            tune_cache=str(tune_cache.with_name(f"rs_{tune_cache.name}")),
        )

        torch.accelerator.synchronize(self.device)
        if not cache_hit:
            self.ag_workspace.rebind_stream()
            self.ag_workspace.tune(ag_hiddens)
            self.rs_workspace.rebind_stream()
            self.rs_workspace.tune([self.hidden_dim])
            _all_ranks_true(self.tune_group, True)
            self._write_cache_complete(tune_cache, metadata)
        torch.accelerator.synchronize(self.device)
        self.ag_workspace.rebind_stream()
        self.rs_workspace.rebind_stream()
        setup_ms = round((time.monotonic() - setup_started) * 1000)
        logger.info_once(
            "Initialized FlashInfer PCIe IPC AG/RS for EP%d, hidden_dim=%d, "
            "dtype=%s, max_tokens=%d, setup_ms=%d, tune_cache=%s.",
            self.world_size,
            self.hidden_dim,
            dtype,
            batches[-1],
            setup_ms,
            "hit" if cache_hit else "miss",
        )

    def _should_use(self, inp: torch.Tensor, workspace: Any | None) -> bool:
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

    def should_use_ag(self, inp: torch.Tensor) -> bool:
        if self.dtype != torch.bfloat16 and inp.dtype in _FP8_DTYPES:
            return False
        if inp.dtype in _FP8_DTYPES and inp.shape[-1] % 2 != 0:
            return False
        ag_inp = inp.view(torch.bfloat16) if inp.dtype in _FP8_DTYPES else inp
        return bool(
            not self.disabled
            and self.ag_workspace is not None
            and inp.is_cuda
            and inp.is_contiguous()
            and inp.dim() == 2
            and inp.shape[1] == self.hidden_dim
            and self.ag_workspace.supports(ag_inp)
        )

    def should_use_rs(self, inp: torch.Tensor) -> bool:
        return self._should_use(inp, self.rs_workspace)

    def all_gather(self, inp: torch.Tensor) -> torch.Tensor:
        assert self.ag_workspace is not None
        if inp.dtype in _FP8_DTYPES:
            return self.ag_workspace.all_gather(inp.view(torch.bfloat16)).view(
                inp.dtype
            )
        return self.ag_workspace.all_gather(inp)

    def reduce_scatter(
        self, inp: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.rs_workspace is not None
        return self.rs_workspace.reduce_scatter(inp, out=output)

    @contextmanager
    def capture(self):
        if self.ag_workspace is None:
            yield
            return
        assert self.rs_workspace is not None

        torch.accelerator.synchronize(self.device)
        self.ag_workspace.rebind_stream()
        self.rs_workspace.rebind_stream()
        try:
            yield
        finally:
            torch.accelerator.synchronize(self.device)
            self.ag_workspace.rebind_stream()
            self.rs_workspace.rebind_stream()

    def destroy(self) -> None:
        if self.ag_workspace is not None:
            self.ag_workspace.destroy()
            self.ag_workspace = None
        if self.rs_workspace is not None:
            self.rs_workspace.destroy()
            self.rs_workspace = None


def warmup_flashinfer_pcie_ipc_ag_rs(worker: "Worker") -> None:
    """Initialize the EP PCIe IPC backend immediately before graph capture."""
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.parallel_state import (
        get_dp_group,
        get_ep_group,
        get_pcp_group,
    )
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    use_ep = (
        worker.vllm_config.parallel_config.enable_expert_parallel
        and get_pcp_group().world_size > 1
    )
    group = get_ep_group() if use_ep else get_dp_group()
    communicator = group.device_communicator
    if not isinstance(communicator, CudaCommunicator):
        return
    pcie_comm = communicator.fi_pcie_ipc_ag_rs_comm
    if pcie_comm is None or pcie_comm.disabled:
        return
    if worker.vllm_config.parallel_config.use_ubatching:
        logger.warning_once(
            "FlashInfer PCIe IPC AG/RS does not support DBO or multi-ubatch "
            "execution; using NCCL."
        )
        pcie_comm.disabled = True
        return

    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    if not capture_sizes:
        return

    base_cache = resolve_flashinfer_autotune_file(worker.model_runner)
    ranks = "-".join(str(rank) for rank in group.ranks)
    role = group.unique_name.replace(":", "_")
    tune_cache = base_cache.with_name(f"pcie_ipc_ag_rs_{role}_{ranks}.json")
    pcie_comm.setup(
        hidden_dim=worker.model_config.get_hidden_size(),
        dtype=worker.model_config.dtype,
        capture_sizes=capture_sizes,
        tune_cache=tune_cache,
    )
