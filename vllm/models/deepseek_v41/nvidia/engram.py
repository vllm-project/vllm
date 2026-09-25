# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA Engram DP sharding, shared host storage, and asynchronous prefetch."""

import ctypes
import mmap
import os
import shutil
import tempfile
import weakref
from contextlib import ExitStack

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
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
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


def _allocate_huge_page_storage(num_bytes: int) -> torch.Tensor | None:
    """Register prefaulted huge pages, or return None for pinned-memory fallback."""
    try:
        mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mapping.madvise(mmap.MADV_HUGEPAGE)
    except OSError as exc:
        logger.warning(
            "Engram huge-page allocation failed (%s); using pinned memory.", exc
        )
        return None

    owner = np.frombuffer(mapping, dtype=np.uint8)
    # Fault the pages in before CUDA pins them, so they can be huge.
    owner[:: mmap.PAGESIZE] = 0
    tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    if result.value != 0:
        logger.warning(
            "Engram cudaHostRegister failed (%s); using pinned memory.", result
        )
        return None
    # Tensor views retain owner; its finalizer retains mapping until unregister.
    finalizer = weakref.finalize(
        owner, DPSharedEngramStorage._unregister, mapping, pointer
    )
    finalizer.atexit = False  # type: ignore[misc]
    if not tensor.is_pinned():
        logger.warning(
            "CUDA did not recognize Engram registration; using pinned memory."
        )
        return None
    return tensor


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


def can_share_engram_tables(layout: EngramLayout, block_size: int = 32) -> bool:
    """Whether co-located DP replicas exist and /dev/shm can hold the full tables."""
    if not os.path.isdir(SHM_PATH):
        logger.warning_once(
            "%s is not mounted; storing the offloaded Engram tables per rank "
            "instead of sharing them across DP replicas.",
            SHM_PATH,
        )
        return False
    if get_engram_dp_size() == 1:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded tables per rank instead of sharing them."
        )
        return False
    num_bytes = sum(layout.num_embeddings) * (
        layout.head_dim + layout.head_dim // block_size
    )
    if shutil.disk_usage(SHM_PATH).total < num_bytes:
        logger.warning_once(
            "Sharing Engram tables across DP replicas needs %.1f GiB of %s "
            "(--shm-size or --ipc=host); sharding them across replicas instead.",
            num_bytes / 1024**3,
            SHM_PATH,
        )
        return False
    return True


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
        group = self.group
        with ExitStack() as stack:
            path = error = None
            if group.rank_in_group == 0:
                try:
                    check_shm_free_space(num_bytes)
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_engram_", dir=SHM_PATH)
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

            mapping = owner = tensor = None
            try:
                with open(path, "r+b") as file:
                    mapping = mmap.mmap(file.fileno(), num_bytes, flags=mmap.MAP_SHARED)
                owner = np.frombuffer(mapping, dtype=np.uint8)
                pointer = owner.ctypes.data
                tensor = torch.from_numpy(owner)
                result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                if result.value != 0:
                    raise RuntimeError(f"cudaHostRegister failed: {result}")
                finalizer = weakref.finalize(owner, self._unregister, mapping, pointer)
                finalizer.atexit = False  # type: ignore[misc]
                # The UVA helper otherwise allocates a private pinned copy.
                if not tensor.is_pinned():
                    raise RuntimeError(
                        "cudaHostRegister did not pin the shared Engram mapping"
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            # Also fences peer mappings before the leader unlinks the file.
            errors: list[str | None] = [None] * group.world_size
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            if failures:
                # Dropping the owner runs the finalizer; the mapping then unmaps.
                del tensor, owner, mapping
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            assert tensor is not None
            return tensor

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
        use_thp: bool = False,
    ) -> None:
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.use_thp = use_thp
        self._packed: torch.Tensor | None = None
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
        if self.use_thp:
            weight_bytes = self.part_num_embeddings * self.dim
            packed = _allocate_huge_page_storage(
                weight_bytes + weight_bytes // self.block_size
            )
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(torch.float8_e4m3fn).view(-1, self.dim),
                    packed[weight_bytes:].view(-1, self.dim // self.block_size),
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

    def collapse_huge_pages(self) -> None:
        """Best-effort MADV_COLLAPSE (Linux >= 6.1) of pages that faulted small."""
        if self._packed is None:
            return
        addr, num_bytes = self._packed.data_ptr(), self._packed.nbytes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        if libc.madvise(addr, num_bytes, 25) != 0:
            logger.warning(
                "Engram MADV_COLLAPSE failed; keeping existing pages: %s",
                os.strerror(ctypes.get_errno()),
            )

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
        self, *args, prefetch_stream: torch.cuda.Stream | None = None, **kwargs
    ) -> None:
        # Layers sharing one stream serialize their offloaded lookups, so they
        # take turns instead of jointly starving decoder compute of SMs.
        self._prefetch_stream = prefetch_stream
        super().__init__(*args, **kwargs)

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
            dp_shared_memory=bool(engram_config.dp_shared_memory),
            use_thp=engram_config.use_thp,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        super()._init_staging(max_tokens * self.embed_tokens.dp_size, head_dim)
        if not self.embed_tokens.cpu_offload:
            self._prefetch_stream = None
            return
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.staged_rows.device)
        self._prefetch_done = torch.cuda.Event()

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
        # Wait for this layer's lookup only, not for later layers on the stream.
        torch.cuda.current_stream().wait_event(event)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_done is not None:
            self._finish_prefetch(self._prefetch_done)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return super()._ready_rows(num_tokens)
