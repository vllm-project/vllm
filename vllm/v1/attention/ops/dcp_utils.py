# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP managers, collective selection, and symmetric-memory implementations."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.ops.common import cp_lse_ag_out_ar, cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

try:
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False


@functools.cache
def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        device = torch.device("cuda", torch.accelerator.current_device_index())
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
            return False
        probe = symm_mem.empty(8, dtype=torch.uint8, device=device)
        probe.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(probe, group.device_group.group_name)
        spans = handle is not None and handle.multicast_ptr != 0
    except Exception as error:
        logger.debug("Direct DCP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct DCP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def _direct_dcp_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if use_direct is not None:
        return use_direct
    return (
        symm_mem_available
        and current_platform.is_cuda()
        and (supported_dtypes is None or dtype in supported_dtypes)
        and (
            all(in_the_same_node_as(group.cpu_group, source_rank=0))
            or _symm_mem_spans_group(group)
        )
    )


def _direct_dcp_multicast_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    return _direct_dcp_enabled(
        group, dtype, use_direct, supported_dtypes
    ) and _symm_mem_spans_group(group)


def get_dcp_workspace_max_num_tokens(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    speculative_tokens = vllm_config.num_speculative_tokens
    tokens_per_seq = (
        1
        + (
            2
            if speculative_config is not None and speculative_config.parallel_drafting
            else 1
        )
        * speculative_tokens
    )
    return min(
        scheduler_config.max_num_batched_tokens,
        max(
            scheduler_config.max_num_seqs * tokens_per_seq,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        ),
    )


class _DirectDCPWorkspace:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.device = torch.device(device)
        self.num_ubatches = num_ubatches
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=self.device)
        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []

    def _allocate(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = symm_mem.empty(shape, device=self.device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "DCP symmetric memory rendezvous returned None"
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        self.device = storage.device
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
        disabled = [0] * self.num_ubatches
        for allocated, handle, _ in self._allocations:
            if allocated is storage:
                break
        else:
            return disabled
        try:
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, storage.device.index
            ):
                return disabled
            multicast_base = handle.multicast_ptr
        except Exception:
            return disabled
        if not multicast_base:
            return disabled
        storage_base = storage.data_ptr()
        return [
            multicast_base + (storage[ubatch].data_ptr() - storage_base)
            for ubatch in range(self.num_ubatches)
        ]


def reserve_query_head_storage(
    query: torch.Tensor, padded_num_heads: int
) -> torch.Tensor:
    """Reserve backing storage for fixed-head decode kernels."""
    assert query.ndim == 3
    assert query.shape[1] <= padded_num_heads
    padded = query.new_empty((query.shape[0], padded_num_heads, query.shape[2]))
    padded.resize_(query.shape)
    padded.copy_(query)
    return padded


_A2A_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


class DirectDCPA2AWorkspace(_DirectDCPWorkspace):
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _A2A_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                f"Direct DCP A2A requires at least one ubatch slot, got {num_ubatches}"
            )
        super().__init__(group, device, num_ubatches)
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            output_shape, dtype
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(lse_shape, torch.float32)
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        output = partial_output.new_empty(
            (num_tokens, self.heads_per_rank, self.head_dim)
        )
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            seq_lens,
            query_start_loc,
            self.peer_output_ptrs[ubatch],
            self.peer_lse_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_output[ubatch],
            self.received_lse[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output,
            self.world_size,
            self.rank,
            self.max_num_tokens,
            is_lse_base_on_e,
        )
        return output


@functools.cache
def get_direct_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace | None:
    if not _direct_dcp_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_A2A, _A2A_SUPPORTED_DTYPES
    ):
        return None
    return DirectDCPA2AWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )


def _q_gather_layout_supported(
    world_size: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    padded_num_heads: int | None,
) -> bool:
    element_size = torch.empty((), dtype=dtype).element_size()
    gathered_num_heads = world_size * heads_per_rank
    storage_num_heads = (
        gathered_num_heads if padded_num_heads is None else padded_num_heads
    )
    return (
        heads_per_rank * head_dim * element_size % 16 == 0
        and storage_num_heads * head_dim * element_size % 16 == 0
    )


class DirectDCPQGatherWorkspace(_DirectDCPWorkspace):
    """Publish query shards directly into the consumer-final symmetric buffer.

    The final buffer is reusable after the downstream DCP output combine. That
    combine orders all ranks after attention has consumed the gathered query.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
        padded_num_heads: int | None = None,
    ) -> None:
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP q-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_num_tokens < 1 or heads_per_rank < 1 or head_dim < 1:
            raise ValueError(
                "Direct DCP q-gather dimensions must be positive, got "
                f"T={max_num_tokens}, H={heads_per_rank}, D={head_dim}"
            )
        gathered_num_heads = group.size() * heads_per_rank
        if not _q_gather_layout_supported(
            group.size(), heads_per_rank, head_dim, dtype, padded_num_heads
        ):
            raise ValueError("Direct DCP q-gather requires 16-byte-aligned query rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP q-gather requires at least two ranks")
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.gathered_num_heads = gathered_num_heads
        self.padded_num_heads = (
            self.gathered_num_heads if padded_num_heads is None else padded_num_heads
        )
        if self.padded_num_heads < self.gathered_num_heads:
            raise ValueError(
                "Direct DCP q-gather padded heads must cover gathered heads: "
                f"{self.padded_num_heads} < {self.gathered_num_heads}"
            )
        self.head_dim = head_dim

        query_shape = (
            num_ubatches,
            max_num_tokens,
            self.padded_num_heads,
            head_dim,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.final_query, _ = self._allocate(query_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        query_multicast_ptrs = self._multicast_ptrs(self.final_query)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(query_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(
            query_ptr and signal_ptr for query_ptr, signal_ptr in self.multicast_ptrs
        ):
            raise RuntimeError(
                "Direct DCP q-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 1))
        torch.accelerator.synchronize()

    def gather(self, local_query: torch.Tensor) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP q-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        if local_query.ndim == 3 and local_query.shape[1] != self.heads_per_rank:
            raise ValueError(
                f"DCP q-gather expected {self.heads_per_rank} local query heads, "
                f"got {local_query.shape[1]}"
            )

        num_tokens = local_query.shape[0]
        output = torch.as_strided(
            self.final_query[ubatch],
            size=(num_tokens, self.gathered_num_heads, self.head_dim),
            stride=(
                self.gathered_num_heads * self.head_dim,
                self.head_dim,
                1,
            ),
        )
        query_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_q_gather(
            local_query,
            output,
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            self.world_size,
            self.rank,
            self.max_num_tokens,
            self.padded_num_heads,
            query_multicast_ptr,
            signal_multicast_ptr,
        )
        return output


@functools.cache
def get_direct_dcp_q_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
    padded_num_heads: int | None = None,
) -> DirectDCPQGatherWorkspace | None:
    if not _direct_dcp_multicast_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_Q_GATHER
    ):
        return None
    if not _q_gather_layout_supported(
        group.world_size, heads_per_rank, head_dim, dtype, padded_num_heads
    ):
        return None
    return DirectDCPQGatherWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
        padded_num_heads,
    )


_KV_GATHER_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)


def _kv_gather_layout_supported(token_dim: int, dtype: torch.dtype) -> bool:
    return token_dim * torch.empty((), dtype=dtype).element_size() % 16 == 0


class DirectDCPKVGatherWorkspace(_DirectDCPWorkspace):
    """Persistent symmetric buffers for direct DCP KV gather."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_gathered_tokens: int,
        token_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _KV_GATHER_SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP kv-gather does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP kv-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_gathered_tokens < 1 or token_dim < 1:
            raise ValueError(
                "Direct DCP kv-gather dimensions must be positive, got "
                f"T={max_gathered_tokens}, D={token_dim}"
            )
        if not _kv_gather_layout_supported(token_dim, dtype):
            raise ValueError("Direct DCP kv-gather requires 16-byte-aligned KV rows.")
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP kv-gather requires at least two ranks")
        if max_gathered_tokens % self.world_size != 0:
            raise ValueError(
                "Direct DCP kv-gather capacity must divide evenly across "
                f"ranks: {max_gathered_tokens} % {self.world_size} != 0"
            )
        self.max_gathered_tokens = max_gathered_tokens

        kv_shape = (num_ubatches, 2, max_gathered_tokens, token_dim)
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_kv, _ = self._allocate(kv_shape, dtype)
        self.received_signal, _ = self._allocate(signal_shape, torch.int32)
        kv_multicast_ptrs = self._multicast_ptrs(self.received_kv)
        signal_multicast_ptrs = self._multicast_ptrs(self.received_signal)
        self.multicast_ptrs = list(
            zip(kv_multicast_ptrs, signal_multicast_ptrs, strict=True)
        )
        if not all(kv_ptr and signal_ptr for kv_ptr, signal_ptr in self.multicast_ptrs):
            raise RuntimeError(
                "Direct DCP kv-gather requires NVLS symmetric-memory multicast."
            )
        self.completion = self.received_signal.new_zeros((num_ubatches, 2))
        torch.accelerator.synchronize()

    def gather(self, gathered_kv: torch.Tensor, local_kv: torch.Tensor) -> None:
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        kv_multicast_ptr, signal_multicast_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            gathered_kv,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
            kv_multicast_ptr,
            signal_multicast_ptr,
        )


@functools.cache
def get_direct_dcp_kv_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    token_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPKVGatherWorkspace | None:
    if not _direct_dcp_multicast_enabled(
        group,
        dtype,
        envs.VLLM_USE_DIRECT_DCP_KV_GATHER,
        _KV_GATHER_SUPPORTED_DTYPES,
    ):
        return None
    if not _kv_gather_layout_supported(token_dim, dtype):
        return None
    return DirectDCPKVGatherWorkspace(
        group.device_group,
        device,
        max_gathered_tokens,
        token_dim,
        dtype,
        num_ubatches,
    )


class DCPCombine(Protocol):
    def __call__(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor: ...


class FlashInferDCPManager:
    """Own persistent buffers and collectives for FlashInfer DCP decode."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        speculative_config = vllm_config.speculative_config

        self.group = get_dcp_group()
        self.world_size = parallel_config.decode_context_parallel_size
        self.max_seq_len = scheduler_config.max_num_seqs
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.max_num_tokens = min(
            scheduler_config.max_num_batched_tokens,
            self.max_seq_len * (1 + num_spec_tokens),
        )

        num_heads_in_dcp = num_heads * self.world_size
        num_ubatches = max(parallel_config.num_ubatches, 1)
        self.output = torch.empty(
            (
                num_ubatches,
                self.max_num_tokens,
                num_heads_in_dcp,
                head_size,
            ),
            dtype=dtype,
            device="cuda",
        )
        self.lse = torch.empty(
            (num_ubatches, self.max_num_tokens, num_heads_in_dcp),
            dtype=torch.float32,
            device="cuda",
        )

        combine_fn = (
            dcp_a2a_lse_reduce
            if parallel_config.dcp_comm_backend == "a2a"
            else cp_lse_ag_out_rs
        )
        self.combine = functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=False,
        )

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(query.contiguous(), dim=-2)

    def get_decode_buffers(
        self, num_decode_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_decode_tokens > self.max_num_tokens:
            raise RuntimeError(
                "FlashInfer DCP decode batch exceeds the persistent buffer: "
                f"{num_decode_tokens} > {self.max_num_tokens}."
            )
        ubatch_id = dbo_current_ubatch_id()
        return (
            self.output[ubatch_id, :num_decode_tokens],
            self.lse[ubatch_id, :num_decode_tokens],
        )


class MLADCPManager:
    """Select and own layer-level collective implementations for MLA DCP."""

    _kv_gather: Callable[[torch.Tensor, torch.Tensor], object]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        num_heads: int,
        query_head_dim: int,
        output_head_dim: int,
        query_dtype: torch.dtype,
        output_dtype: torch.dtype,
        padded_num_heads: int | None,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.group = get_dcp_group()
        self.device = torch.device(device)
        self.num_ubatches = max(parallel_config.num_ubatches, 1)
        self.max_num_tokens = get_dcp_workspace_max_num_tokens(vllm_config)
        self.use_a2a = parallel_config.dcp_comm_backend == "a2a"
        self.padded_num_heads = padded_num_heads

        self.combine = self._init_combine(
            num_heads,
            output_head_dim,
            output_dtype,
            is_lse_base_on_e,
            use_pcp,
        )
        self.query_gather = (
            None
            if use_pcp
            else self._init_query_gather(
                num_heads,
                query_head_dim,
                query_dtype,
            )
        )

    def _init_combine(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        is_lse_base_on_e: bool,
        use_pcp: bool,
    ) -> DCPCombine:
        direct_workspace = None
        if self.use_a2a:
            direct_workspace = get_direct_dcp_a2a_workspace(
                self.group,
                self.device,
                self.max_num_tokens,
                num_heads,
                head_dim,
                dtype,
                self.num_ubatches,
            )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP A2A for MLA.")
            return functools.partial(
                direct_workspace.lse_reduce,
                is_lse_base_on_e=is_lse_base_on_e,
            )

        combine_fn = (
            dcp_a2a_lse_reduce
            if self.use_a2a
            else cp_lse_ag_out_ar
            if use_pcp
            else cp_lse_ag_out_rs
        )
        return functools.partial(
            combine_fn,
            cp_group=self.group,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    def _init_query_gather(
        self,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        direct_workspace = get_direct_dcp_q_gather_workspace(
            self.group,
            self.device,
            self.max_num_tokens,
            num_heads,
            head_dim,
            dtype,
            self.num_ubatches,
            self.padded_num_heads,
        )
        if direct_workspace is not None:
            logger.info_once("Using direct symmetric-memory DCP query gather for MLA.")
            return direct_workspace.gather
        return self._gather_query

    def _gather_query(self, query: torch.Tensor) -> torch.Tensor:
        query = self.group.all_gather(query, dim=1)
        if self.padded_num_heads is not None:
            query = reserve_query_head_storage(query, self.padded_num_heads)
        return query

    def init_kv_gather(
        self,
        workspace: torch.Tensor,
        max_gathered_tokens: int,
    ) -> None:
        world_size = self.group.world_size
        assert max_gathered_tokens > 0
        assert max_gathered_tokens % world_size == 0
        assert workspace.ndim == 2
        assert workspace.is_contiguous()
        assert workspace.shape[0] == (
            max_gathered_tokens + max_gathered_tokens // world_size
        )
        assert workspace.shape[1] > 0

        direct_workspace = get_direct_dcp_kv_gather_workspace(
            self.group,
            workspace.device,
            max_gathered_tokens,
            workspace.shape[1],
            workspace.dtype,
            self.num_ubatches,
        )
        if direct_workspace is not None:
            logger.info_once(
                "Using direct symmetric-memory DCP chunked-context KV gather for MLA."
            )
            self._kv_gather = direct_workspace.gather
        else:
            self._kv_gather = functools.partial(
                torch.distributed.all_gather_into_tensor,
                group=self.group.device_group,
            )

    def kv_gather(
        self,
        gathered_kv: torch.Tensor,
        local_kv: torch.Tensor,
    ) -> object:
        return self._kv_gather(gathered_kv, local_kv)
