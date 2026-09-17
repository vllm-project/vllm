# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""

import ctypes
import errno
import mmap
import os
import tempfile
import time
import weakref
from contextlib import ExitStack
from functools import cache
from pathlib import Path

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    _engram_head_shard_weight_loader,
    _engram_select_rows,
)
from vllm.models.deepseek_v41.common.engram import (
    Engram as BaseEngram,
)
from vllm.models.deepseek_v41.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


_ENGRAM_PSI_PAUSE = 20.0
_ENGRAM_PSI_RESUME = 10.0
_ENGRAM_PSI_WAIT_SECONDS = 120.0
_ENGRAM_PAGE_CHUNK_BYTES = 256 * 1024**2
_ENGRAM_PSI_POLL_SECONDS = 0.25


class _EngramPageThrottle:
    """Share a bounded PSI wait budget across one table's fault and collapse."""

    def __init__(self) -> None:
        self.remaining = _ENGRAM_PSI_WAIT_SECONDS
        self.exhausted = False
        self.paths: list[Path] = []
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    relative = line[3:].lstrip("/")
                    # A cgroup namespace can hide the process's actual cgroup.
                    if ".." not in Path(relative).parts:
                        self.paths.append(
                            Path("/sys/fs/cgroup") / relative / "memory.pressure"
                        )
                    break
        except OSError:
            pass
        self.paths.append(Path("/proc/pressure/memory"))

    def _pressure(self) -> float | None:
        while self.paths:
            try:
                for line in self.paths[0].read_text().splitlines():
                    if line.startswith("some "):
                        fields = dict(field.split("=", 1) for field in line.split()[1:])
                        pressure = float(fields["avg10"])
                        if 0 <= pressure <= 100:
                            return pressure
            except (OSError, ValueError, KeyError):
                pass
            self.paths.pop(0)
            if self.paths:
                logger.info_once(
                    "Engram cgroup v2 PSI unavailable; using system memory PSI."
                )
            else:
                logger.info_once(
                    "Engram memory PSI unavailable; page throttling is disabled."
                )
        return None

    def wait(self) -> None:
        if self.exhausted:
            return
        pressure = self._pressure()
        if pressure is None or pressure < _ENGRAM_PSI_PAUSE:
            return
        logger.info("Pausing Engram page work: memory PSI some avg10=%.2f%%.", pressure)
        while True:
            if self.remaining <= 0:
                self.exhausted = True
                logger.warning(
                    "Engram PSI wait budget exhausted; completing remaining page "
                    "faults without throttling and skipping this table's collapse. "
                    "MADV_HUGEPAGE remains enabled; insufficient memory can still "
                    "cause allocation failure or an OOM kill."
                )
                return
            if pressure is None or pressure <= _ENGRAM_PSI_RESUME:
                return
            start = time.monotonic()
            time.sleep(min(_ENGRAM_PSI_POLL_SECONDS, self.remaining))
            self.remaining -= time.monotonic() - start
            pressure = self._pressure()


def _engram_page_chunk_size(page_size: int) -> int:
    return (_ENGRAM_PAGE_CHUNK_BYTES + page_size - 1) // page_size * page_size


def _fault_engram_host_pages(
    mapping: mmap.mmap,
    offset: int,
    size: int,
    page_size: int,
    throttle: _EngramPageThrottle,
) -> None:
    chunk_size = _engram_page_chunk_size(page_size)
    for start in range(0, size, chunk_size):
        throttle.wait()
        np.frombuffer(
            mapping,
            dtype=np.uint8,
            count=min(chunk_size, size - start),
            offset=offset + start,
        )[:: mmap.PAGESIZE] = 0


def _engram_thp_size() -> int | None:
    try:
        size = int(
            Path("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").read_text()
        )
    except (OSError, ValueError):
        return None
    return size if size >= mmap.PAGESIZE and size & (size - 1) == 0 else None


def _engram_thp_mode(page_size: int) -> str:
    root = Path("/sys/kernel/mm/transparent_hugepage")

    def read_mode(path: Path) -> str:
        try:
            return next(
                (
                    word[1:-1]
                    for word in path.read_text().split()
                    if word.startswith("[") and word.endswith("]")
                ),
                "unknown",
            )
        except OSError:
            return "unknown"

    mode = read_mode(root / f"hugepages-{page_size // 1024}kB" / "enabled")
    if mode in ("inherit", "unknown"):
        mode = read_mode(root / "enabled")
    return mode


def _engram_hugepage_bytes(pointer: int, size: int) -> int | None:
    try:
        lines = Path("/proc/self/smaps").read_text().splitlines()
    except OSError:
        return None
    total, active = 0, False
    for line in lines:
        fields = line.split()
        if "-" in fields[0]:
            start, end = (int(x, 16) for x in fields[0].split("-"))
            active = start >= pointer and end <= pointer + size
        elif active and fields[0] == "AnonHugePages:":
            total += int(fields[1]) * 1024
    return total


def _allocate_engram_host_storage(
    num_bytes: int,
    checkpoint_dir: Path | None = None,
    throttle: _EngramPageThrottle | None = None,
) -> torch.Tensor | None:
    """Register a PMD-aligned private mapping, retaining ordinary-page fallback."""
    page_size = _engram_thp_size()
    if page_size is None:
        logger.info("Engram PMD size unavailable; using Torch pinned memory.")
        return None
    if num_bytes < page_size:
        logger.info(
            "Engram host table (%.2f MiB) is smaller than one PMD (%.2f MiB); "
            "skipping THP packing and using Torch pinned memory.",
            num_bytes / 1024**2,
            page_size / 1024**2,
        )
        return None
    mode = _engram_thp_mode(page_size)
    logger.info("Engram PMD THP policy: %s (%.2f MiB).", mode, page_size / 1024**2)
    if mode == "never":
        logger.info(
            "Automatic PMD THP allocation is disabled; retaining explicit "
            "MADV_COLLAPSE recovery after loading, which ignores this policy."
        )
    # Include the partial final PMD in the advised VMA.
    size = (num_bytes + page_size - 1) // page_size * page_size
    mapping = owner = tensor = finalizer = None
    try:
        mapping = mmap.mmap(
            -1, size + 2 * page_size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        address = np.frombuffer(mapping, dtype=np.uint8, count=1).ctypes.data
        offset = page_size - address % page_size
        # Unfaulted guards isolate the advised VMA for coverage accounting.
        mapping.madvise(mmap.MADV_NOHUGEPAGE)
        mapping.madvise(mmap.MADV_HUGEPAGE, offset, size)
        owner = np.frombuffer(mapping, dtype=np.uint8, count=num_bytes, offset=offset)
        _prepare_engram_host_pages(checkpoint_dir)
        # Fault the whole advised range, including the rounded tail, before
        # registration can pin pages at base size. An unfaulted tail VMA has
        # no anon_vma, which fails the recovery MADV_COLLAPSE with EINVAL.
        _fault_engram_host_pages(
            mapping, offset, size, page_size, throttle or _EngramPageThrottle()
        )
        tensor = torch.from_numpy(owner)
        pointer = tensor.data_ptr()
        # Register the PMD-rounded range, not num_bytes: ending the
        # registration mid-PMD splits the VMA there, and a huge page cannot
        # span the split, so the tail PMD would fall back to base pages.
        result = torch.cuda.cudart().cudaHostRegister(pointer, size, 0)
        if result.value != 0:
            raise RuntimeError(f"cudaHostRegister failed: {result}")
        finalizer = weakref.finalize(
            owner, DPSharedEngramStorage._unregister, mapping, pointer
        )
        finalizer.atexit = False  # type: ignore[misc]
        if not tensor.is_pinned():
            raise RuntimeError("CUDA did not recognize the Engram registration")
    except (OSError, RuntimeError) as exc:
        if finalizer is not None:
            finalizer()
        tensor = owner = None
        if mapping is not None:
            mapping.close()
        logger.warning_once(
            "Engram huge-page allocation failed (%s); using Torch pinned memory.", exc
        )
        return None
    return tensor


def _engram_checkpoint_dir() -> Path | None:
    from vllm.transformers_utils.repo_utils import try_get_local_file

    config = get_current_vllm_config()
    model = config.model_config
    if model is None:
        return None
    model_path = model.model_weights or model.model
    path = Path(model_path)
    if path.is_dir():
        return path
    for filename in (
        "model.safetensors.index.json",
        "model.safetensors",
        "config.json",
    ):
        cached_file = try_get_local_file(
            model_path,
            filename,
            revision=model.revision,
            cache_dir=config.load_config.download_dir,
        )
        if isinstance(cached_file, Path):
            return cached_file.parent
    return None


def _drop_engram_checkpoint_cache(checkpoint_dir: Path | None) -> None:
    if checkpoint_dir is None or not hasattr(os, "posix_fadvise"):
        return
    files, num_bytes = 0, 0
    for path in checkpoint_dir.glob("*.safetensors"):
        try:
            with path.open("rb") as file:
                size = os.fstat(file.fileno()).st_size
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
            num_bytes += size
        except OSError as exc:
            logger.warning(
                "Could not release Engram checkpoint cache for %s: %s", path, exc
            )
    logger.info(
        "Requested page-cache release for %d checkpoint files (%.2f GiB) "
        "before Engram huge-page allocation/recovery.",
        files,
        num_bytes / 1024**3,
    )


@cache
def _prepare_engram_host_pages(checkpoint_dir: Path | None) -> None:
    _drop_engram_checkpoint_cache(checkpoint_dir)


def _collapse_engram_host_pages(pointer: int, size: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    for attempt in range(3):
        if libc.madvise(pointer, size, 25) == 0:  # MADV_COLLAPSE (Linux >= 6.1)
            return
        error = ctypes.get_errno()
        if error != errno.EAGAIN or attempt == 2:
            logger.warning("Engram MADV_COLLAPSE failed: %s", os.strerror(error))
            return
        time.sleep(1)


def _finish_engram_host_pages(
    pointer: int,
    num_bytes: int,
    checkpoint_dir: Path | None,
    throttle: _EngramPageThrottle | None = None,
) -> None:
    # Same PMD rounding as _allocate_engram_host_storage, so coverage is
    # measured over exactly the advised VMA.
    page_size = _engram_thp_size() or mmap.PAGESIZE
    size = (num_bytes + page_size - 1) // page_size * page_size
    throttle = throttle or _EngramPageThrottle()
    huge_bytes = _engram_hugepage_bytes(pointer, size)
    if not throttle.exhausted and huge_bytes is not None and huge_bytes < size * 0.90:
        _drop_engram_checkpoint_cache(checkpoint_dir)
        logger.info(
            "Recovering Engram huge pages for %.2f GiB host table", size / 1024**3
        )
        chunk_size = _engram_page_chunk_size(page_size)
        for start in range(0, size, chunk_size):
            throttle.wait()
            if throttle.exhausted:
                break
            _collapse_engram_host_pages(pointer + start, min(chunk_size, size - start))
        huge_bytes = _engram_hugepage_bytes(pointer, size)
    log = logger.warning if huge_bytes == 0 else logger.info
    log(
        "Engram host table: %.2f GiB, PMD-rounded mapping: %.2f GiB, "
        "huge-page coverage %s. "
        "Low coverage can limit lookup performance.",
        num_bytes / 1024**3,
        size / 1024**3,
        "unknown" if huge_bytes is None else f"{100 * huge_bytes / size:.1f}%",
    )


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(
        dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max()
    )


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID
        )
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


class DPSharedEngramStorage:
    """Registered host weights shared by a node-local DP group with one writer."""

    def __init__(
        self, num_rows: int, dim: int, block_size: int, group: GroupCoordinator
    ) -> None:
        self.group = group
        weight_bytes = num_rows * dim
        storage = self._allocate(weight_bytes + weight_bytes // block_size)
        self.weight = storage[:weight_bytes].view(torch.float8_e4m3fn).view(-1, dim)
        self.weight_scale_inv = storage[weight_bytes:].view(-1, dim // block_size)
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Map and register one physical allocation across a node-local DP group."""
        from vllm.distributed.device_communicators.shm_broadcast import (
            check_shm_free_space,
        )

        group = self.group
        with ExitStack() as stack:
            path, error = None, None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(
                            prefix="vllm_engram_", dir="/dev/shm"
                        )
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            mapping = owner = tensor = finalizer = None
            stage = "open"
            try:
                try:
                    with open(path, "r+b") as file:
                        stage = "mmap"
                        mapping = mmap.mmap(
                            file.fileno(), num_bytes, flags=mmap.MAP_SHARED
                        )
                    stage = "cudaHostRegister"
                    owner = np.frombuffer(mapping, dtype=np.uint8)
                    pointer = owner.ctypes.data
                    tensor = torch.from_numpy(owner)
                    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                    if result.value != 0:
                        raise RuntimeError(f"cudaHostRegister failed: {result}")
                    finalizer = weakref.finalize(
                        owner, self._unregister, mapping, pointer
                    )
                    finalizer.atexit = False  # type: ignore[misc]
                    # The UVA helper otherwise allocates a private pinned copy.
                    if not tensor.is_pinned():
                        raise RuntimeError(
                            "CUDA did not recognize the shared Engram registration"
                        )
                except Exception as exc:
                    error = f"{stage}: {type(exc).__name__}: {exc}"

                errors: list[str | None] = [None] * group.world_size
                # Also fences peer mappings before the leader unlinks the file.
                torch.distributed.all_gather_object(
                    errors, error, group=group.cpu_group
                )
                failures = "; ".join(
                    f"EDP rank {rank}: {error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                )
                if failures:
                    raise RuntimeError(
                        "Engram shared-memory initialization failed: " + failures
                    )
                assert tensor is not None
                return tensor
            except Exception:
                if finalizer is not None:
                    finalizer()
                tensor = owner = None
                if mapping is not None:
                    mapping.close()
                raise

    @staticmethod
    def _unregister(mapping: mmap.mmap, pointer: int) -> None:
        # Torch storage retains the numpy owner, including through cached UVA views.
        # Keep its mmap alive until CUDA has released the registration.
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        if result.value != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if self.group.rank_in_group == 0:
            _engram_head_shard_weight_loader(param, loaded_weight)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self.group.cpu_group)

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        if self._views is None:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv),
            )
        return self._views


class ParallelEngramEmbedding(BaseParallelEngramEmbedding):
    """Extend TP lookup with DP head sharding or shared, CPU-offloaded TP slices."""

    _shared_memory: DPSharedEngramStorage | None = None

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        thp_packing: bool = False,
        checkpoint_dir: Path | None = None,
    ) -> None:
        if thp_packing and (not cpu_offload or dp_shared_memory):
            raise ValueError(
                "thp_packing requires cpu_offload=True and dp_shared_memory=False"
            )
        self.thp_packing = thp_packing
        self._checkpoint_dir = checkpoint_dir
        self._host_page_storage: torch.Tensor | None = None
        self._host_pages_dirty = False
        self._host_page_throttle: _EngramPageThrottle | None = None
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        super().__init__(num_embeddings, dim, head_sizes, block_size)
        if cpu_offload:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            # The ue8m0 encoding of scale 1.0 is exponent byte 127.
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB %s",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
                "shared across DP replicas" if dp_shared_memory else "per rank",
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return super()._get_shard_info()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = DPSharedEngramStorage(
                self.part_num_embeddings, self.dim, self.block_size, group
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        if not self.cpu_offload:
            return super()._allocate_weights()
        if self.thp_packing:
            weight_bytes = self.part_num_embeddings * self.dim
            self._host_page_throttle = _EngramPageThrottle()
            host_storage = _allocate_engram_host_storage(
                weight_bytes + weight_bytes // self.block_size,
                self._checkpoint_dir,
                self._host_page_throttle,
            )
            if host_storage is not None:
                self._host_page_storage = host_storage
                self._weight_loader = self._load_host_weight
                return (
                    host_storage[:weight_bytes]
                    .view(torch.float8_e4m3fn)
                    .view(-1, self.dim),
                    host_storage[weight_bytes:].view(-1, self.dim // self.block_size),
                )
        # Model initialization may be inside a CUDA device context.
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            ),
        )

    def _load_host_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        _engram_head_shard_weight_loader(param, loaded_weight)
        self._host_pages_dirty = True

    def finish_weight_loading(self) -> None:
        storage = self._host_page_storage
        if storage is not None and self._host_pages_dirty:
            _finish_engram_host_pages(
                storage.data_ptr(),
                storage.numel(),
                self._checkpoint_dir,
                self._host_page_throttle,
            )
            self._host_pages_dirty = False

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self.weight_scale_inv)
        if not self.cpu_offload:
            return super()._storage()
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight.data),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.dp_size == 1:
            return super().forward(indices)
        num_tokens = indices.shape[0]
        indices = gather_engram_hashes(indices)
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        out = _gather_engram_rows(out, num_tokens)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows


class Engram(BaseEngram):
    """NVIDIA Engram with asynchronous offload and node-local DP lookup."""

    _prefetch_stream: torch.cuda.Stream | None = None
    _prefetch_done: torch.cuda.Event | None = None

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
        *,
        prefetch_stream: torch.cuda.Stream | None,
    ) -> None:
        self._prefetch_stream = prefetch_stream
        super().__init__(
            config,
            quant_config,
            layout,
            layer_hash_index,
            use_sequence_parallel,
            prefix,
        )

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        return ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=engram_config.dp_shared_memory,
            thp_packing=engram_config.thp_packing,
            checkpoint_dir=(
                _engram_checkpoint_dir() if engram_config.thp_packing else None
            ),
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        super()._init_staging(max_tokens * self.embed_tokens.dp_size, head_dim)
        if self.embed_tokens.cpu_offload:
            if self._prefetch_stream is None:
                raise ValueError(
                    "CPU-offloaded Engram requires a caller-provided prefetch stream"
                )
            self._prefetch_done = torch.cuda.Event()
        else:
            self._prefetch_stream = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Prefetch local shared rows or the DP group's gathered hash IDs."""
        if self._prefetch_stream is None:
            return super().prepare_embeddings(hash_ids)
        rows = self.staged_rows[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self._start_prefetch(hash_ids, rows, self._prefetch_stream)

    @eager_break_during_capture
    def _start_prefetch(
        self, hash_ids: torch.Tensor, rows: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        # Eager boundaries let the lookup span piecewise graph segments.
        stream.wait_stream(torch.cuda.current_stream())
        # Keep temporary hash storage alive until lookup finishes reading it.
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            self.embed_tokens.lookup(hash_ids, rows, background=True)
            assert self._prefetch_done is not None
            self._prefetch_done.record(stream)

    @eager_break_during_capture
    def _finish_prefetch(self, event: torch.cuda.Event) -> None:
        torch.cuda.current_stream().wait_event(event)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_stream is not None:
            assert self._prefetch_done is not None
            self._finish_prefetch(self._prefetch_done)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return super()._ready_rows(num_tokens)
