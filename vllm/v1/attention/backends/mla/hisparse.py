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


def validate_hisparse_decode_batch(
    *,
    max_query_len: int,
    num_reqs: int,
    num_actual_tokens: int,
) -> None:
    if is_hisparse_decode_batch(
        max_query_len=max_query_len,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
    ):
        return

    raise NotImplementedError(
        "HiSparse decode hot-buffer mode only supports pure decode batches. "
        "Prefill or mixed sparse MLA batches would need full logical GPU KV "
        "or a prefill host-staging path; use a decode-only/disaggregated "
        "deployment for this experimental path."
    )


@dataclass(frozen=True)
class HiSparseConfig:
    top_k: int
    device_buffer_size: int
    host_to_device_ratio: int
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
        host_to_device_ratio = int(raw_config.get("host_to_device_ratio", 2))
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
        # Running the backup on the capture stream is graph-safe. The per-step
        # decode backup is only the batch's newest rows (tiny), so losing the
        # async overlap is negligible; the one-time prefill backup is synchronous
        # but prefill is not graph-captured. SGLang reclaims the overlap by doing
        # the backup eagerly in the scheduler prepare phase -- a future option,
        # toggleable here via VLLM_HISPARSE_ASYNC_BACKUP=1 (only safe with
        # cudagraphs disabled).
        _async_backup = (
            os.environ.get("VLLM_HISPARSE_ASYNC_BACKUP", "0") == "1"
            and self.device.type == "cuda"
            and self._use_cuda_ops
        )
        self._backup_stream = (
            torch.cuda.Stream(device=self.device) if _async_backup else None
        )
        self._backup_done_event = torch.cuda.Event() if _async_backup else None
        self._has_pending_backup = False
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
        self.wait_for_pending_backup()
        self.device_global_indices.fill_(-1)
        self.lru_slots.copy_(self._lru_init.expand_as(self.lru_slots))

    def reset_hot_rows(self, rows: torch.Tensor) -> None:
        """Drop hot-buffer bookkeeping for newly assigned batch rows."""
        self.wait_for_pending_backup()
        if rows.numel() == 0:
            return
        rows = rows[(rows >= 0) & (rows < self.max_num_reqs)]
        if rows.numel() == 0:
            return
        self.device_global_indices[rows] = -1
        self.lru_slots[rows] = self._lru_init

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        assert self._backup_done_event is not None
        self._backup_done_event.wait(torch.cuda.current_stream(self.device))
        self._has_pending_backup = False

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
        self.wait_for_pending_backup()
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
        if self._backup_stream is None:
            torch.ops._C_cache_ops.hisparse_backup(
                src_cache,
                src_indices,
                self._host_cache,
                self._host_cache_valid,
                dst_slots,
            )
            return

        self.wait_for_pending_backup()
        current_stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._backup_stream):
            self._backup_stream.wait_stream(current_stream)
            torch.ops._C_cache_ops.hisparse_backup(
                src_cache,
                src_indices,
                self._host_cache,
                self._host_cache_valid,
                dst_slots,
            )
            assert self._backup_done_event is not None
            self._backup_done_event.record(self._backup_stream)
            for tensor in (src_cache, src_indices, dst_slots):
                tensor.record_stream(self._backup_stream)
        self._has_pending_backup = True

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

    def write_rows_to_host(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write prefill/mixed-batch rows directly to a host-resident pool."""
        from vllm import _custom_ops as ops

        if kv_cache.numel() == 0:
            return
        self.bind_source_cache(kv_cache)
        if not self._source_is_host:
            raise RuntimeError("write_rows_to_host requires a host-resident KV pool.")
        flat_slots = slot_mapping.flatten()[: kv_c_normed.shape[0]]
        valid_mask = flat_slots >= 0
        valid = flat_slots[valid_mask].to(device=self.device, dtype=torch.int64)
        if valid.numel() == 0:
            return
        # CUDA graph padding can make kv_c_normed/k_pe longer than slot_mapping.
        # Only rows represented by slot_mapping correspond to real KV writes.
        real_kv_rows = kv_c_normed[: flat_slots.numel()]
        real_pe_rows = k_pe[: flat_slots.numel()]

        if kv_cache_dtype == "fp8_ds_mla":
            kv_rows = real_kv_rows[valid_mask]
            pe_rows = real_pe_rows[valid_mask]
            rows = torch.empty(
                (valid.numel(), self.row_width),
                dtype=self.kv_dtype,
                device=self.device,
            )
            ops.concat_and_cache_mla(
                kv_rows,
                pe_rows.squeeze(1),
                rows.view(-1, 1, self.row_width),
                torch.arange(valid.numel(), dtype=torch.int64, device=self.device),
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
            )
        else:
            rows = torch.cat(
                [real_kv_rows[valid_mask], real_pe_rows[valid_mask].squeeze(1)],
                dim=-1,
            ).contiguous()
        src = torch.arange(valid.numel(), dtype=torch.int64, device=self.device)
        self._backup_rows(rows, src, valid)
        self._invalidate_hot_copies(valid)

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
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Resolve top-k positions against the hot buffer.

        Returns ``(hot_cache_paged, hot_indices)`` (plus ``valid_counts`` when
        requested). ``hot_cache_paged`` has the same paged layout as a regular
        MLA KV cache; ``hot_indices`` are global token ids within it.
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
        # tokens are resolved like any other entry (miss -> host load).
        newest_global = (
            slot_mapping[:num_tokens].to(torch.int32).contiguous()
            if slot_mapping is not None
            else None
        )
        if newest_global is None:
            # Mixed batches do not use the reserved newest slot, so a selected
            # newly written row must be visible in the host pool before miss
            # handling can read it.
            self.wait_for_pending_backup()

        if self._kernel_path():
            # Padded rows are skipped by the kernel (num_real_reqs) and must
            # come out as -1 so the attention kernel masks them.
            # In host-resident mode every miss is served from the host pool
            # (validity is all-true), so the GPU source fallback is dead; pass
            # the hot cache to satisfy the kernel's CUDA-source requirement.
            kernel_source = (
                self.hot_cache if self._source_is_host else self._source_cache
            )
            hot_indices = torch.full_like(global_indices, -1)
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
            )
        else:
            hot_indices = self._swap_in_fallback(global_indices, newest_global)

        if valid_counts is None:
            return self.hot_cache_paged(block_size), hot_indices
        return self.hot_cache_paged(block_size), hot_indices, valid_counts

    def _swap_in_fallback(
        self,
        global_indices: torch.Tensor,
        newest_global: torch.Tensor | None,
    ) -> torch.Tensor:
        """Slow reference implementation of the swap-in kernel semantics."""
        assert self._source_cache is not None
        assert self._host_cache is not None and self._host_cache_valid is not None
        num_tokens, _ = global_indices.shape
        buf = self.config.device_buffer_size
        hot_indices = torch.full_like(global_indices, -1)

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

        return hot_indices


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

    # HiSparse works on any deployment. PD-decode instances are the intended
    # fast path (KV arrives via a consumer connector and the instance never
    # prefills). Unified / non-PD instances are also supported: prefill gathers
    # KV from the host pool (see _hisparse_host_prefill_cache) while decode uses
    # the hot buffer. No kv_transfer_config requirement here; the non-PD prefill
    # cost is surfaced as a warning in VllmConfig.__post_init__.
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
