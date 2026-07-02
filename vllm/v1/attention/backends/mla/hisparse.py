# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental HiSparse hot-buffer decode for sparse MLA (GLM-5 / DSA).

Port of SGLang's HiSparse decode design to vLLM's attention-backend boundary.
HiSparse keeps the full-size MLA KV off the GPU and serves decode from a small
per-request device hot buffer; that is what frees GPU memory for higher decode
concurrency.

- Full-size KV residency. In the host-resident deployment the MLA KV layers
  are allocated in pinned host memory (see ``_is_hisparse_host_layer`` and
  ``_allocate_kv_cache_tensors`` in the cache utils / model runner); only the
  GPU indexer pool and the per-request hot buffers stay on device. A
  GPU-cache deployment is also supported: the full GPU KV cache remains the
  store and every written row is mirrored into a pinned host cache keyed by
  its global KV slot id. Either way the mirror/backup is done by a CUDA kernel
  writing straight into pinned memory, so it is stream-ordered with the decode
  kernels and needs no host synchronization.
- Each batch row owns a fixed per-layer hot region of
  ``region_stride = round_up(device_buffer_size + 1, 128)`` KV rows (padded to
  128 so the flat buffer can be viewed with any kernel page size up to 128).
  Slots ``[0, device_buffer_size)`` are LRU-managed; slot ``device_buffer_size``
  is reserved for the newest token, which the KV-cache update writes there
  directly (mirroring SGLang's reserved newest page).
- At decode time, the indexer's top-k positions are converted to global slot
  ids and a swap-in kernel resolves them against the hot region: hits are
  reused, misses are copied from the pinned host pool (or the GPU cache in
  mirror mode), and the LRU order is updated — the same algorithm as SGLang's
  ``load_cache_to_device_buffer`` kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

logger = init_logger(__name__)


def is_hisparse_decode_batch(
    *,
    max_query_len: int,
    num_reqs: int,
    num_actual_tokens: int,
) -> bool:
    return max_query_len == 1 and num_reqs == num_actual_tokens


@dataclass(frozen=True)
class HiSparseConfig:
    top_k: int
    device_buffer_size: int
    host_to_device_ratio: float
    host_pool_gib: float | None = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        model_top_k: int,
    ) -> HiSparseConfig | None:
        attention_config = vllm_config.attention_config
        if attention_config is None or not attention_config.enable_hisparse:
            return None
        raw_config = attention_config.hisparse_config
        if raw_config is None:
            raise ValueError(
                "HiSparse requires attention_config.hisparse_config with "
                "top_k, device_buffer_size, and host_to_device_ratio."
            )

        top_k = int(raw_config.get("top_k", model_top_k))
        device_buffer_size = int(raw_config["device_buffer_size"])
        # Optional: only used to size the host pool when host_pool_gib is unset.
        host_to_device_ratio = float(raw_config.get("host_to_device_ratio", 2))
        host_pool_gib = raw_config.get("host_pool_gib")
        host_pool_gib = None if host_pool_gib is None else float(host_pool_gib)

        if top_k != model_top_k:
            raise ValueError(
                f"HiSparse top_k ({top_k}) must match model index_topk "
                f"({model_top_k}) for lossless sparse MLA."
            )
        if device_buffer_size < top_k:
            raise ValueError(
                "HiSparse device_buffer_size must be at least top_k. "
                f"Got device_buffer_size={device_buffer_size}, top_k={top_k}."
            )
        if host_to_device_ratio < 1:
            raise ValueError("HiSparse host_to_device_ratio must be >= 1.")
        if host_pool_gib is not None and host_pool_gib <= 0:
            raise ValueError("HiSparse host_pool_gib must be positive when set.")

        return cls(
            top_k=top_k,
            device_buffer_size=device_buffer_size,
            host_to_device_ratio=host_to_device_ratio,
            host_pool_gib=host_pool_gib,
        )


# One per device, shared by all per-layer coordinators: number of real
# (non-CUDA-graph-padding) requests in the current batch. Updated by the
# model runner outside captured regions; read by the swap-in kernel from
# device memory so graph replays observe the per-step value.
_NUM_REAL_REQS: dict[torch.device, torch.Tensor] = {}


def get_num_real_reqs_tensor(device: torch.device) -> torch.Tensor:
    tensor = _NUM_REAL_REQS.get(device)
    if tensor is None:
        # Default to "all rows are real" so standalone use (tests, eager
        # deployments without the runner hook) processes every row; the model
        # runner overwrites this each step when driving CUDA graphs.
        tensor = torch.full((1,), torch.iinfo(torch.int32).max, dtype=torch.int32,
                            device=device)
        _NUM_REAL_REQS[device] = tensor
    return tensor


def set_num_real_reqs(num_reqs: int) -> None:
    """Called by the model runner before each (real or dummy) forward."""
    for tensor in _NUM_REAL_REQS.values():
        tensor.fill_(num_reqs)


def invalidate_blocks(
    static_forward_context: dict, block_ids: list[int], block_size: int
) -> None:
    """Drop cached HiSparse state for the given blocks in every layer.

    Called by the model runner when blocks are (re)assigned to newly scheduled
    or preemption-resumed requests, before any forward step can select them.
    This makes block recycling safe for any writer (local prefill, connector
    RDMA into GPU or host memory) without per-connector reporting hooks.
    """
    if not block_ids:
        return
    slots: torch.Tensor | None = None
    for layer in static_forward_context.values():
        impl = getattr(layer, "impl", None)
        coordinator = getattr(impl, "hisparse_coordinator", None)
        if coordinator is None:
            continue
        if slots is None:
            blocks = torch.tensor(
                block_ids, dtype=torch.long, device=coordinator.device
            )
            offsets = torch.arange(
                block_size, dtype=torch.long, device=coordinator.device
            )
            slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        coordinator.invalidate_slots(slots)


def reset_rows(static_forward_context: dict, row_ids: list[int]) -> None:
    """Drop per-row HiSparse hot-buffer state in every layer."""
    if not row_ids:
        return
    rows: torch.Tensor | None = None
    for layer in static_forward_context.values():
        impl = getattr(layer, "impl", None)
        coordinator = getattr(impl, "hisparse_coordinator", None)
        if coordinator is None:
            continue
        if rows is None:
            rows = torch.tensor(row_ids, dtype=torch.long, device=coordinator.device)
        coordinator.reset_hot_rows(rows)


# Shared all-true validity bitmap for host-resident pools: every slot the
# scheduler hands out is written before the indexer can select it, so the
# swap-in kernel may treat the whole pool as valid.
_ALL_VALID: torch.Tensor | None = None


def get_all_valid_tensor(num_rows: int) -> torch.Tensor:
    global _ALL_VALID
    if _ALL_VALID is None or _ALL_VALID.shape[0] < num_rows:
        _ALL_VALID = torch.ones(num_rows, dtype=torch.bool, pin_memory=True)
    return _ALL_VALID


def _has_hisparse_ops() -> bool:
    try:
        try:
            import vllm._C_stable_libtorch  # noqa: F401
        except ImportError:
            import vllm._C  # noqa: F401
    except ImportError:
        return False
    if not hasattr(torch.ops, "_C_cache_ops"):
        return False
    if not (
        hasattr(torch.ops._C_cache_ops, "hisparse_swap_in")
        and hasattr(torch.ops._C_cache_ops, "hisparse_backup")
    ):
        return False
    # Guard against stale builds that carry an older hisparse_swap_in schema.
    schema = str(torch.ops._C_cache_ops.hisparse_swap_in.default._schema)
    return "num_real_reqs" in schema


class _GroupPlan:
    """Group-shared swap-in plan for GLM-5.2 index sharing.

    A "full" layer's swap_in(produce_plan=True) writes the resolved plan here;
    its "shared" layers replay it via apply_plan without re-resolving LRU. One
    set per (device, max_rows, top_k), shared across all coordinators -- mirrors
    the model-global topk_indices_buffer (layers run sequentially: full writes,
    its shared layers read, the next full overwrites). Static shapes so the
    replay is CUDA-graph-capture safe.
    """

    __slots__ = ("global_indices", "hot_indices", "miss_mask", "valid_counts")

    def __init__(self, device: torch.device, max_rows: int, top_k: int) -> None:
        self.global_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.hot_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.miss_mask = torch.zeros(
            (max_rows, top_k), dtype=torch.int32, device=device
        )
        # Populated only when the producer resolves with return_valid_counts
        # (the bf16 path); shared layers reuse it (identical top-k -> identical
        # counts). None until first produced.
        self.valid_counts: torch.Tensor | None = None


_GROUP_PLANS: dict[tuple[str, int, int], _GroupPlan] = {}


def _get_group_plan(device: torch.device, max_rows: int, top_k: int) -> _GroupPlan:
    key = (str(device), max_rows, top_k)
    plan = _GROUP_PLANS.get(key)
    if plan is None:
        plan = _GroupPlan(device, max_rows, top_k)
        _GROUP_PLANS[key] = plan
    return plan


_COPY_STREAMS: dict[str, "torch.cuda.Stream"] = {}


def _get_copy_stream(device: torch.device) -> "torch.cuda.Stream":
    """One dedicated copy stream per device, shared across a group's coordinators
    for overlapped prefetch (#1). Capturing a fork/join around work on it into a
    FULL_DECODE_ONLY graph is supported (verified by the capture spike)."""
    key = str(device)
    s = _COPY_STREAMS.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _COPY_STREAMS[key] = s
    return s


class HiSparseCoordinator:
    """Per-layer decode-time hot buffer for sparse MLA KV rows.

    Hot-buffer hits are keyed by global KV slot id, so correctness relies on
    one invariant: a recycled slot's stale state is dropped before reuse —
    write paths (``mirror_slots``/``write_newest_rows``) invalidate the slots
    they write, and the model runner invalidates all blocks (re)assigned to
    incoming requests (covering connector RDMA loads of any kind).
    """

    def __init__(
        self,
        config: HiSparseConfig,
        max_num_reqs: int,
        row_width: int,
        kv_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.max_num_reqs = max_num_reqs
        self.row_width = row_width
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)
        # One reserved slot for the newest token, padded so the flat hot
        # buffer can be viewed with any kernel page size up to 128 (the
        # FlashMLA FP8 / FlashInfer sparse kernels need a paged layout; the
        # actual block size is only known at swap-in time).
        self.region_stride = round_up(config.device_buffer_size + 1, 128)

        row_bytes = row_width * kv_dtype.itemsize
        if row_bytes % 16 != 0:
            raise ValueError(
                f"HiSparse requires 16-byte aligned KV rows, got {row_bytes}B."
            )

        hot_rows = max_num_reqs * self.region_stride
        # Allocated eagerly so vLLM's memory profiling accounts for it when
        # sizing the main KV cache.
        self.hot_cache = torch.zeros(
            (hot_rows, row_width), dtype=kv_dtype, device=self.device
        )
        self.device_global_indices = torch.full(
            (max_num_reqs, config.device_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._lru_init = torch.arange(
            config.device_buffer_size, dtype=torch.int16, device=self.device
        )
        self.lru_slots = self._lru_init.repeat(max_num_reqs, 1).contiguous()
        self._newest_hot_slots = (
            torch.arange(max_num_reqs, dtype=torch.int64, device=self.device)
            * self.region_stride
            + config.device_buffer_size
        )

        self.num_real_reqs = get_num_real_reqs_tensor(self.device)

        # Group-shared plan buffers for index-sharing plan-once (see _GroupPlan).
        self._plan = _get_group_plan(self.device, max_num_reqs, config.top_k)

        # Overlapped prefetch (#1, opt-in): a "full" layer issues its group's
        # "shared" gathers early on a copy stream, overlapping the PCIe transfer
        # with intervening compute. group_shared/leader are wired by the impl
        # (which knows full vs shared); _prefetch_event is recorded on the copy
        # stream by the leader and awaited in the shared layer's apply_plan.
        self.group_shared: list[HiSparseCoordinator] = []
        self.leader: HiSparseCoordinator | None = None
        self._prefetch_event: torch.cuda.Event | None = None
        self._overlap_enabled = (
            os.environ.get("VLLM_HISPARSE_OVERLAP", "0") == "1"
            and self.device.type == "cuda"
            and _has_hisparse_ops()
        )
        self._copy_stream = (
            _get_copy_stream(self.device) if self._overlap_enabled else None
        )

        self._source_cache: torch.Tensor | None = None
        self._source_is_host = False
        self._host_cache: torch.Tensor | None = None
        self._host_cache_valid: torch.Tensor | None = None
        self._host_is_pinned = False
        self._use_cuda_ops = _has_hisparse_ops()
        # Host backup runs INLINE on the current/capture stream (no dedicated
        # backup stream). The decode KV-update (write_newest_rows -> _backup_rows)
        # executes inside the FULL_DECODE_ONLY CUDA graph; a cross-stream wait
        # there raises "dependency created on uncaptured work in another stream"
        # and aborts graph capture (the bug that previously forced enforce_eager).
        # Running the backup on the capture stream is graph-safe, and the
        # per-step decode backup is only the batch's newest rows (tiny), so
        # there is no overlap worth reclaiming. (SGLang overlaps its backup by
        # issuing it from the scheduler prepare phase, outside capture -- the
        # only sound place for a future async variant.)
        if not self._use_cuda_ops:
            logger.warning_once(
                "HiSparse CUDA ops (_C_cache_ops.hisparse_swap_in/backup) are "
                "not compiled. Host-resident HiSparse requires these ops; "
                "mirror-mode tests may still use the Python reference path."
            )

    def hot_cache_paged(self, block_size: int) -> torch.Tensor:
        """Hot buffer shaped like a regular paged MLA cache."""
        return self.hot_cache.view(-1, block_size, self.row_width)

    @property
    def source_is_host(self) -> bool:
        return self._source_is_host

    def bind_source_cache(self, kv_cache: torch.Tensor) -> None:
        flat = kv_cache.view(-1, kv_cache.shape[-1])
        if self._source_cache is not None and (
            self._source_cache.data_ptr() == flat.data_ptr()
            and self._source_cache.shape == flat.shape
        ):
            return

        if kv_cache.dtype != self.kv_dtype or kv_cache.shape[-1] != self.row_width:
            raise ValueError(
                "HiSparse coordinator bound to a KV cache with mismatched "
                f"layout: expected ({self.row_width}, {self.kv_dtype}), got "
                f"({kv_cache.shape[-1]}, {kv_cache.dtype})."
            )

        self._source_cache = flat
        self.reset_hot_state()

        num_slots = flat.shape[0]
        if kv_cache.device.type == "cpu":
            # Host-resident pool: the cache itself is the host store. No
            # mirror is needed; every allocated slot is written before use.
            if not kv_cache.is_pinned():
                raise ValueError(
                    "HiSparse host-resident KV pool must be pinned memory."
                )
            if not self._use_cuda_ops:
                raise RuntimeError(
                    "HiSparse host-resident KV requires compiled "
                    "_C_cache_ops.hisparse_swap_in/backup kernels. Rebuild "
                    "vLLM from source so csrc/hisparse_kernels.cu is included."
                )
            self._source_is_host = True
            self._host_cache = flat
            self._host_cache_valid = get_all_valid_tensor(num_slots)
            self._host_is_pinned = True
            return

        self._source_is_host = False
        if (
            self._host_cache is not None
            and self._host_cache.shape[0] >= num_slots
        ):
            self._host_cache_valid.zero_()  # type: ignore[union-attr]
            return

        try:
            self._host_cache = torch.empty(
                (num_slots, self.row_width),
                dtype=self.kv_dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host_cache_valid = torch.zeros(
                num_slots, dtype=torch.bool, pin_memory=True
            )
            self._host_is_pinned = True
        except RuntimeError:
            logger.warning_once(
                "HiSparse failed to pin %.2f GiB of host mirror memory; "
                "falling back to pageable memory and the slow Python path.",
                num_slots * self.row_width * self.kv_dtype.itemsize / 2**30,
            )
            self._host_cache = torch.empty(
                (num_slots, self.row_width), dtype=self.kv_dtype, device="cpu"
            )
            self._host_cache_valid = torch.zeros(num_slots, dtype=torch.bool)
            self._host_is_pinned = False

    def reset_hot_state(self) -> None:
        """Drop all hot-buffer bookkeeping (hits become misses)."""
        self.device_global_indices.fill_(-1)
        self.lru_slots.copy_(self._lru_init.expand_as(self.lru_slots))

    def reset_hot_rows(self, rows: torch.Tensor) -> None:
        """Drop hot-buffer bookkeeping for newly assigned batch rows."""
        if rows.numel() == 0:
            return
        rows = rows[(rows >= 0) & (rows < self.max_num_reqs)]
        if rows.numel() == 0:
            return
        self.device_global_indices[rows] = -1
        self.lru_slots[rows] = self._lru_init

    def _invalidate_hot_copies(self, slots: torch.Tensor) -> None:
        stale = torch.isin(
            self.device_global_indices, slots.to(device=self.device, dtype=torch.int32)
        )
        self.device_global_indices[stale] = -1

    def invalidate_slots(self, slots: torch.Tensor) -> None:
        """Drop all cached state for the given global slots.

        Called when blocks are (re)assigned to a request, regardless of who
        writes them (local prefill, KV connector RDMA, ...). Hot-buffer
        copies of recycled slots must never be served as hits, and in
        mirror mode the stale host rows must not be preferred over the
        rewritten GPU source rows.
        """
        if self._source_cache is None:
            return
        self._invalidate_hot_copies(slots)
        if not self._source_is_host and self._host_cache_valid is not None:
            slots_cpu = slots.to(device="cpu", dtype=torch.long)
            slots_cpu = slots_cpu[
                (slots_cpu >= 0) & (slots_cpu < self._host_cache_valid.shape[0])
            ]
            self._host_cache_valid[slots_cpu] = False

    def _kernel_path(self) -> bool:
        return self._use_cuda_ops and self._host_is_pinned

    def _backup_rows(
        self,
        src_cache: torch.Tensor,
        src_indices: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        assert self._host_cache is not None and self._host_cache_valid is not None
        torch.ops._C_cache_ops.hisparse_backup(
            src_cache,
            src_indices,
            self._host_cache,
            self._host_cache_valid,
            dst_slots,
        )

    # -------------------------------------------------------------- mirroring

    def mirror_slots(
        self,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        """Mirror the given global KV slots from the cache into host memory."""
        if slot_mapping is None or kv_cache.numel() == 0:
            return
        self.bind_source_cache(kv_cache)
        valid_slots = slot_mapping.flatten()
        valid_slots = valid_slots[valid_slots >= 0].to(torch.int64)
        if valid_slots.numel() == 0:
            return

        assert self._source_cache is not None
        if self._source_is_host:
            # Data already lives in the host pool; only stale hot-buffer
            # copies of these (possibly recycled) slots must be dropped.
            pass
        elif self._kernel_path():
            self._backup_rows(
                self._source_cache,
                valid_slots,
                valid_slots,
            )
        else:
            self._mirror_slots_fallback(valid_slots)
        self._invalidate_hot_copies(valid_slots)

    def _mirror_slots_fallback(self, valid_slots: torch.Tensor) -> None:
        assert self._source_cache is not None
        assert self._host_cache is not None and self._host_cache_valid is not None
        rows = self._source_cache.index_select(0, valid_slots).to(device="cpu")
        slots_cpu = valid_slots.to(device="cpu")
        self._host_cache[slots_cpu] = rows
        self._host_cache_valid[slots_cpu] = True

    # ------------------------------------------------------- newest-token path

    def write_newest_rows(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Decode-step KV update.

        Writes the newest token of each batch row to:
        - its global KV slot (keeps the full GPU cache complete, so mixed
          batches, preemption-resume, and prefix sharing stay correct),
        - its reserved hot slot (so the swap-in kernel can serve the newest
          token without a host roundtrip, like SGLang's reserved page),
        - the pinned host mirror (so later steps can miss-load it).
        """
        from vllm import _custom_ops as ops

        if kv_cache.numel() == 0:
            return
        self.bind_source_cache(kv_cache)
        num_tokens = kv_c_normed.shape[0]
        assert num_tokens <= self.max_num_reqs
        newest_slots = self._newest_hot_slots[:num_tokens]
        global_slots = slot_mapping[:num_tokens].to(torch.int64)
        assert self._source_cache is not None

        if self._source_is_host:
            # Host-resident pool: write the newest rows into the reserved hot
            # slots (GPU), then scatter them into the host pool. -1-padded
            # slots (CUDA-graph padding) are skipped by the backup kernel.
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe.squeeze(1),
                self.hot_cache.view(-1, 1, self.row_width),
                newest_slots,
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
            )
            self._backup_rows(
                self.hot_cache,
                newest_slots,
                global_slots,
            )
        else:
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                global_slots,
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
            )
            # CUDA-graph padding fills slot_mapping with -1; clamp so the
            # gather stays in bounds (padded rows then just hold row 0's
            # bytes, which is harmless since their hot indices are never
            # consumed).
            rows = self._source_cache.index_select(0, global_slots.clamp(min=0))
            self.hot_cache.index_copy_(0, newest_slots, rows)

            if self._kernel_path():
                self._backup_rows(
                    self._source_cache,
                    global_slots,
                    global_slots,
                )
            else:
                self._mirror_slots_fallback(global_slots)
        # Global slots can be recycled blocks; drop any stale hot copies so
        # they cannot be served as hits.
        self._invalidate_hot_copies(global_slots)

    # ---------------------------------------------------------------- swap-in

    def swap_in(
        self,
        *,
        kv_cache: torch.Tensor,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor | None,
        return_valid_counts: bool = False,
        produce_plan: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Resolve top-k positions against the hot buffer.

        Returns ``(hot_cache_paged, hot_indices)`` (plus ``valid_counts`` when
        requested). ``hot_cache_paged`` has the same paged layout as a regular
        MLA KV cache; ``hot_indices`` are global token ids within it.

        When ``produce_plan`` (an index-sharing "full" layer), the resolved plan
        (global ids, hot slots, miss mask) is also written to the group-shared
        buffers so the group's "shared" layers can replay it via ``apply_plan``
        without re-resolving the LRU.
        """
        num_tokens = topk_indices.shape[0]
        top_k = topk_indices.shape[1]
        if top_k > self.config.device_buffer_size:
            raise ValueError(
                f"HiSparse top-k width {top_k} exceeds device_buffer_size "
                f"{self.config.device_buffer_size}."
            )
        if self.region_stride % block_size != 0:
            raise ValueError(
                f"HiSparse region_stride {self.region_stride} is not "
                f"divisible by the kernel block_size {block_size}."
            )
        self.bind_source_cache(kv_cache)

        converted = triton_convert_req_index_to_global_index(
            req_id_per_token[:num_tokens],
            block_table,
            topk_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=top_k,
            BLOCK_N=128 if top_k % 128 == 0 else top_k,
            return_valid_counts=return_valid_counts,
        )
        if return_valid_counts:
            global_indices, valid_counts = converted
        else:
            global_indices = converted
            valid_counts = None

        # When slot_mapping is None (mixed batches: the newest rows were
        # written to the global store, not the reserved hot slots), newest
        # tokens are resolved like any other entry (miss -> host load); the
        # backup kernel runs stream-ordered, so those rows are already
        # visible in the host pool.
        newest_global = (
            slot_mapping[:num_tokens].to(torch.int32).contiguous()
            if slot_mapping is not None
            else None
        )

        # For a plan producer, resolve into the group-shared buffers so the
        # group's shared layers can replay the identical plan.
        if produce_plan:
            self._plan.global_indices[:num_tokens].copy_(global_indices)
            hot_indices = self._plan.hot_indices[:num_tokens]
            miss_mask = self._plan.miss_mask[:num_tokens]
            # The kernel skips CUDA-graph padding rows, so their entries in
            # the persistent plan buffer would otherwise keep stale values;
            # refill so padded rows come out -1 (masked by attention).
            hot_indices.fill_(-1)
        else:
            hot_indices = torch.full_like(global_indices, -1)
            miss_mask = None

        if self._kernel_path():
            # Padded rows are skipped by the kernel (num_real_reqs) and must
            # come out as -1 so the attention kernel masks them.
            # In host-resident mode every miss is served from the host pool
            # (validity is all-true), so the GPU source fallback is dead; pass
            # the hot cache to satisfy the kernel's CUDA-source requirement.
            kernel_source = (
                self.hot_cache if self._source_is_host else self._source_cache
            )
            torch.ops._C_cache_ops.hisparse_swap_in(
                kernel_source,
                self._host_cache,
                self._host_cache_valid,
                self.hot_cache,
                global_indices,
                newest_global,
                hot_indices,
                self.device_global_indices,
                self.lru_slots,
                self.num_real_reqs,
                self.region_stride,
                miss_mask,
            )
        else:
            self._swap_in_fallback(
                global_indices, newest_global, hot_indices, miss_mask
            )

        if produce_plan:
            # Shared layers reuse the group's counts (identical top-k).
            self._plan.valid_counts = valid_counts
            # Overlap (#1): issue the group's shared gathers early on the copy
            # stream so their PCIe transfer overlaps intervening compute.
            if self._overlap_enabled and self.group_shared:
                self._prefetch_group(num_tokens)

        if valid_counts is None:
            return self.hot_cache_paged(block_size), hot_indices
        return self.hot_cache_paged(block_size), hot_indices, valid_counts

    def _gather_plan_into(self, num_tokens: int) -> None:
        """Gather THIS coord's misses per the group-shared plan into its own hot
        cache, on the current stream. Host-resident: misses come from the host
        pool (kernel_source is the hot cache, a no-op fallback source)."""
        # At prefetch time the shared layer may not have bound its per-forward
        # source yet; the hot cache is always a valid CUDA source and host-valid
        # misses are served from the host pool regardless.
        kernel_source = (
            self._source_cache
            if (not self._source_is_host and self._source_cache is not None)
            else self.hot_cache
        )
        torch.ops._C_cache_ops.hisparse_gather_plan(
            kernel_source,
            self._host_cache,
            self._host_cache_valid,
            self.hot_cache,
            self._plan.global_indices[:num_tokens],
            self._plan.hot_indices[:num_tokens],
            self._plan.miss_mask[:num_tokens],
            self.num_real_reqs,
        )

    def _prefetch_group(self, num_tokens: int) -> None:
        """Leader-side: fork the copy stream from the compute stream (plan is
        ready) and issue each shared layer's gather on it, recording a per-shared
        event the shared layer's apply_plan awaits. Fork here + wait_event there
        is captured into the decode graph (verified by the capture spike)."""
        compute = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(compute)  # fork
        with torch.cuda.stream(self._copy_stream):
            for shared in self.group_shared:
                # Need the host pool bound to serve misses. Until then, the
                # shared layer gathers inline in apply_plan.
                if shared._host_cache is None:
                    shared._prefetch_event = None
                    continue
                shared._gather_plan_into(num_tokens)
                if shared._prefetch_event is None:
                    shared._prefetch_event = torch.cuda.Event()
                shared._prefetch_event.record(self._copy_stream)

    def _swap_in_fallback(
        self,
        global_indices: torch.Tensor,
        newest_global: torch.Tensor | None,
        hot_indices: torch.Tensor,
        miss_mask: torch.Tensor | None = None,
    ) -> None:
        """Slow reference implementation of the swap-in kernel semantics.

        Writes resolved slots into ``hot_indices`` in place; when ``miss_mask``
        is given, sets it to 1 at columns resolved as a miss (matching the
        kernel's plan output for the group-shared replay path).
        """
        assert self._source_cache is not None
        assert self._host_cache is not None and self._host_cache_valid is not None
        num_tokens, _ = global_indices.shape
        buf = self.config.device_buffer_size
        hot_indices.fill_(-1)
        if miss_mask is not None:
            miss_mask.fill_(0)

        global_cpu = global_indices.cpu().tolist()
        newest_cpu = (
            newest_global.cpu().tolist()
            if newest_global is not None
            else [-1] * num_tokens
        )
        dgi_cpu = self.device_global_indices[:num_tokens].cpu().tolist()
        lru_cpu = self.lru_slots[:num_tokens].cpu().tolist()

        miss_src: list[int] = []
        miss_dst: list[int] = []
        for row in range(num_tokens):
            base = row * self.region_stride
            slot_of_global = {
                g: slot for slot, g in enumerate(dgi_cpu[row]) if g >= 0
            }
            hit_slots: list[int] = []
            hit_cols: dict[int, int] = {}
            for col, g in enumerate(global_cpu[row]):
                if g < 0:
                    continue
                if g == newest_cpu[row]:
                    hot_indices[row, col] = base + buf
                elif g in slot_of_global:
                    hit_cols[slot_of_global[g]] = col

            # Classify slots in LRU order, like the kernel does.
            evictables = [s for s in lru_cpu[row] if s not in hit_cols]
            hit_slots = [s for s in lru_cpu[row] if s in hit_cols]
            for slot in hit_slots:
                hot_indices[row, hit_cols[slot]] = base + slot

            misses = [
                (col, g)
                for col, g in enumerate(global_cpu[row])
                if g >= 0 and g != newest_cpu[row] and g not in slot_of_global
            ]
            miss_slots = []
            for m, (col, g) in enumerate(misses):
                slot = evictables[m]
                miss_slots.append(slot)
                hot_indices[row, col] = base + slot
                if miss_mask is not None:
                    miss_mask[row, col] = 1
                dgi_cpu[row][slot] = g
                miss_src.append(g)
                miss_dst.append(base + slot)

            lru_cpu[row] = evictables[len(misses) :] + miss_slots + hit_slots

        self.device_global_indices[:num_tokens] = torch.tensor(
            dgi_cpu, dtype=torch.int32, device=self.device
        )
        self.lru_slots[:num_tokens] = torch.tensor(
            lru_cpu, dtype=torch.int16, device=self.device
        )

        if miss_src:
            src_cpu = torch.tensor(miss_src, dtype=torch.long)
            dst = torch.tensor(miss_dst, dtype=torch.long, device=self.device)
            host_valid = self._host_cache_valid[src_cpu]
            rows = torch.empty(
                (len(miss_src), self.row_width),
                dtype=self.kv_dtype,
                device=self.device,
            )
            if torch.any(host_valid):
                rows[host_valid.to(self.device)] = self._host_cache[
                    src_cpu[host_valid]
                ].to(self.device)
            if not torch.all(host_valid):
                gpu_src = src_cpu[~host_valid].to(self.device)
                rows[(~host_valid).to(self.device)] = self._source_cache.index_select(
                    0, gpu_src
                )
            self.hot_cache.index_copy_(0, dst, rows)

    # ---------------------------------------------------------- plan replay

    def apply_plan(
        self,
        *,
        kv_cache: torch.Tensor,
        block_size: int,
        num_tokens: int,
        return_valid_counts: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Replay the group's plan for an index-sharing "shared" layer.

        A "full" layer resolved the plan via ``swap_in(produce_plan=True)``;
        this gathers only THIS layer's own missed rows into the identical
        planned hot slots, with no LRU resolution. Fixed shape -> capture-safe.
        """
        n = num_tokens
        hot_indices = self._plan.hot_indices[:n]
        if self._overlap_enabled and self._prefetch_event is not None:
            # Leader already gathered this layer's misses on the copy stream;
            # just join so attention sees them, then consume the event so the
            # next step re-prefetches (or falls back if it can't).
            torch.cuda.current_stream(self.device).wait_event(self._prefetch_event)
            self._prefetch_event = None
        else:
            # Not prefetched (overlap off, or host pool not yet bound): gather
            # this layer's misses inline, same path the leader's prefetch uses.
            self.bind_source_cache(kv_cache)
            if self._kernel_path():
                self._gather_plan_into(n)
            else:
                self._apply_plan_fallback(
                    self._plan.global_indices[:n], hot_indices, self._plan.miss_mask[:n]
                )
        if return_valid_counts:
            assert self._plan.valid_counts is not None, (
                "apply_plan(return_valid_counts=True) requires the group's full "
                "layer to have produced the plan with counts (bf16 path)."
            )
            return (
                self.hot_cache_paged(block_size),
                hot_indices,
                self._plan.valid_counts[:n],
            )
        return self.hot_cache_paged(block_size), hot_indices

    def _apply_plan_fallback(
        self,
        global_indices: torch.Tensor,
        hot_indices: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> None:
        """Python reference for hisparse_gather_plan (miss-column gather)."""
        assert self._source_cache is not None
        assert self._host_cache is not None and self._host_cache_valid is not None
        mask = (miss_mask != 0) & (global_indices >= 0) & (hot_indices >= 0)
        src = global_indices[mask]
        dst = hot_indices[mask].to(torch.long)
        if src.numel() == 0:
            return
        src_cpu = src.cpu().to(torch.long)
        host_valid = self._host_cache_valid[src_cpu]
        rows = torch.empty(
            (src.numel(), self.row_width), dtype=self.kv_dtype, device=self.device
        )
        if torch.any(host_valid):
            rows[host_valid.to(self.device)] = self._host_cache[
                src_cpu[host_valid]
            ].to(self.device)
        if not torch.all(host_valid):
            gpu_src = src_cpu[~host_valid].to(self.device)
            rows[(~host_valid).to(self.device)] = self._source_cache.index_select(
                0, gpu_src
            )
        self.hot_cache.index_copy_(0, dst, rows)


def create_hisparse_coordinator(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    row_width: int,
    kv_dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> HiSparseCoordinator | None:
    config = HiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    # HiSparse is decode-only: intended for PD decode instances, where KV
    # arrives via a consumer connector (NIXL) into the host pool and the
    # instance never prefills locally. The unified / non-PD prefill-from-host
    # path has been removed (do_kv_cache_update asserts a decode batch).
    hf_config = vllm_config.model_config.hf_config
    if not hasattr(hf_config, "index_topk"):
        raise ValueError("HiSparse is only supported for DSA models with index_topk.")
    if hasattr(hf_config, "compress_ratios") and len(hf_config.compress_ratios) > 0:
        raise ValueError("This HiSparse path targets GLM-5/DSA, not DeepSeek V4.")
    if vllm_config.speculative_config is not None:
        raise ValueError("HiSparse MVP does not support speculative decoding.")

    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    coordinator = HiSparseCoordinator(
        config=config,
        max_num_reqs=max_num_reqs,
        row_width=row_width,
        kv_dtype=kv_dtype,
        device=device,
    )
    hot_bytes = coordinator.hot_cache.numel() * kv_dtype.itemsize
    logger.info_once(
        "Enabled experimental HiSparse sparse MLA hot buffer: top_k=%d, "
        "device_buffer_size=%d (region_stride=%d), host_to_device_ratio=%d, "
        "host_pool_gib=%s, %.1f MiB GPU hot buffer per layer "
        "(max_num_seqs=%d).",
        config.top_k,
        config.device_buffer_size,
        coordinator.region_stride,
        config.host_to_device_ratio,
        config.host_pool_gib,
        hot_bytes / 2**20,
        max_num_reqs,
    )
    return coordinator
