# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NixlConnectorWithAux — extends NixlConnector to support auxiliary
dense tensor transfer (e.g. hidden states) via NIXL RDMA, alongside
the paged KV cache.

Design:
  - Subclass only; does NOT modify original NixlConnector / base classes.
  - Aux buffer is a contiguous GPU tensor, registered with NIXL.
  - Transfers use the same RDMA backend as KV cache (UCX/IB).
  - Aux metadata (buffer addr, slot, length) travels through
    kv_transfer_params → proxy → decode scheduler → worker metadata.

Usage (in kv_transfer_config):
    {
        "kv_connector": "NixlConnectorWithAux",
        "kv_connector_module_path":
            "examples.pd_eagle_test.nixl_connector_with_aux",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "transfer_aux": "true",
            "aux_max_slots": 8,
            "aux_max_seq_len": 2048
        }
    }
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
)

if TYPE_CHECKING:
    from vllm.config import KVCacheConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.forward_context import ForwardContext
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


# ── Aux-specific metadata that rides alongside kv_transfer_params ──────


@dataclass
class AuxTransferInfo:
    """Per-request aux transfer info, passed from scheduler to worker."""

    remote_aux_base_addr: int  # GPU ptr on the prefill side
    remote_device_id: int
    remote_engine_id: str
    slot_index: int  # slot in prefill's aux buffer
    num_elements: int  # actual number of elements (tokens * hidden)
    slot_size_bytes: int  # full slot size in bytes


# ── Slab allocator for the aux buffer ──────────────────────────────────


class AuxSlabAllocator:
    """Simple fixed-size slab allocator for aux buffer slots."""

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self._free: list[int] = list(range(max_slots))
        self._used: dict[str, int] = {}  # req_id -> slot_index

    def alloc(self, req_id: str) -> int | None:
        if not self._free:
            return None
        slot = self._free.pop(0)
        self._used[req_id] = slot
        return slot

    def free(self, req_id: str) -> None:
        slot = self._used.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)

    def get_slot(self, req_id: str) -> int | None:
        return self._used.get(req_id)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)


# ── Worker subclass ───────────────────────────────────────────────────


class NixlConnectorWorkerWithAux(NixlConnectorWorker):
    """Extends NixlConnectorWorker with an RDMA-registered aux buffer."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.aux_enabled = str(extra.get("transfer_aux", "false")).lower() == "true"
        # Default aux_max_slots to max_num_seqs so we never run out
        # under normal scheduling conditions (matching KV cache behavior).
        default_max_slots = vllm_config.scheduler_config.max_num_seqs
        self.aux_max_slots = int(extra.get("aux_max_slots", default_max_slots))
        self.aux_max_seq_len = int(extra.get("aux_max_seq_len", 2048))

        # Will be set after model config is available (in register_kv_caches).
        self.aux_buffer: torch.Tensor | None = None
        self.aux_base_addr: int = 0
        self.aux_slot_size_bytes: int = 0
        self.aux_hidden_size: int = 0
        self.aux_dtype: torch.dtype = torch.bfloat16
        self.aux_allocator = AuxSlabAllocator(self.aux_max_slots)

        # Per-request actual lengths (in elements, not bytes).
        self._aux_lengths: dict[str, int] = {}

        # NIXL descriptor handles for aux.
        self._aux_local_xfer_handle: int | None = None
        # Track pending aux transfers: req_id -> list of NIXL handles
        self._aux_recving_transfers: dict[str, list[int]] = defaultdict(list)
        # Completed aux transfers ready for get_aux.
        self._aux_ready: set[str] = set()
        # Pending aux loads waiting for remote agent handshake.
        self._pending_aux_loads: dict[str, AuxTransferInfo] = {}

    # ── Registration ──────────────────────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches AND allocate/register the aux buffer.

        When speculative decoding is enabled, the decode side has extra
        KV cache layers from the drafter (e.g. EAGLE).  These layers
        don't exist on the prefill side, so including them in the NIXL
        registration causes a handshake mismatch.  We filter them out
        before calling super(), keeping only target-model layers.
        """
        filtered = kv_caches
        if self.vllm_config.speculative_config is not None:
            target_num_layers = (
                self.vllm_config.model_config.hf_config.num_hidden_layers
            )
            filtered = {}
            for name, cache in kv_caches.items():
                # Layer names look like "model.layers.36.self_attn.attn".
                # EAGLE3 uses start_layer_id = target_num_layers, so
                # drafter layers have index >= target_num_layers.
                m = re.search(r"layers\.(\d+)\.", name)
                if m:
                    layer_idx = int(m.group(1))
                    if layer_idx < target_num_layers:
                        filtered[name] = cache
                else:
                    # Cannot parse index — keep it (conservative).
                    filtered[name] = cache

            if len(filtered) != len(kv_caches):
                logger.info(
                    "Filtered KV caches for NIXL: kept %d/%d layers "
                    "(excluded %d drafter layers)",
                    len(filtered),
                    len(kv_caches),
                    len(kv_caches) - len(filtered),
                )

        # ── Allocate and register the aux buffer BEFORE super() ────────
        # super().register_kv_caches() snapshots the NIXL agent metadata
        # at the very end (via get_agent_metadata()).  The aux buffer must
        # already be registered with NIXL so that the metadata exchanged
        # during the handshake includes the aux memory region.
        if self.aux_enabled:
            # Derive device_id from the KV cache tensors (self.device_id
            # isn't set until super() runs).
            first_cache = next(iter(kv_caches.values()))
            device_id = max(first_cache.get_device(), 0)

            self.aux_hidden_size = self.vllm_config.model_config.get_hidden_size()
            self.aux_dtype = self.vllm_config.model_config.dtype

            element_size = torch.tensor([], dtype=self.aux_dtype).element_size()
            slot_elements = self.aux_max_seq_len * self.aux_hidden_size
            self.aux_slot_size_bytes = slot_elements * element_size
            total_bytes = self.aux_max_slots * self.aux_slot_size_bytes

            # Allocate contiguous GPU buffer.
            self.aux_buffer = torch.zeros(
                self.aux_max_slots * slot_elements,
                dtype=self.aux_dtype,
                device=f"cuda:{device_id}",
            )
            self.aux_base_addr = self.aux_buffer.data_ptr()

            logger.info(
                "Allocated aux buffer: %d slots x %d bytes = %.1f MB, "
                "hidden_size=%d, dtype=%s, base_addr=0x%x",
                self.aux_max_slots,
                self.aux_slot_size_bytes,
                total_bytes / 1e6,
                self.aux_hidden_size,
                self.aux_dtype,
                self.aux_base_addr,
            )

            # Register with NIXL for RDMA (before super snapshots metadata).
            caches_data = [
                (self.aux_base_addr, total_bytes, device_id, ""),
            ]
            descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
            self._registered_descs.append(descs)
            logger.info("Aux buffer registered with NIXL (pre-handshake).")

        # ── Now call super() — this registers KV caches AND publishes
        # agent metadata (which now includes the aux memory region).
        super().register_kv_caches(filtered)

        # ── Create local aux transfer descriptors (post-super, since we
        # now have self.device_id set by the parent).
        if self.aux_enabled and self.aux_buffer is not None:
            slot_descs = []
            for i in range(self.aux_max_slots):
                addr = self.aux_base_addr + i * self.aux_slot_size_bytes
                slot_descs.append((addr, self.aux_slot_size_bytes, self.device_id))
            local_descs = self.nixl_wrapper.get_xfer_descs(
                slot_descs, self.nixl_memory_type
            )
            self._aux_local_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", local_descs
            )
            logger.info("Aux NIXL local xfer descriptors ready.")

    # ── put_aux (prefill side) ────────────────────────────────────────

    def put_aux(
        self, name: str, req_id: str, tensor: torch.Tensor
    ) -> dict[str, Any] | None:
        """Copy tensor into a slot of the registered aux buffer.

        Returns metadata dict to be included in kv_transfer_params,
        or None if aux is disabled / no slot available.
        """
        if not self.aux_enabled or self.aux_buffer is None:
            return None

        slot = self.aux_allocator.alloc(req_id)
        if slot is None:
            raise RuntimeError(
                f"put_aux: no free aux slot for req={req_id} "
                f"(max_slots={self.aux_allocator.max_slots}, "
                f"used={self.aux_allocator.num_used}). "
                f"Increase aux_max_slots or check for slot leaks."
            )

        # Flatten and copy into buffer.
        flat = tensor.detach().reshape(-1)
        num_elements = flat.numel()
        slot_offset = slot * (self.aux_max_seq_len * self.aux_hidden_size)
        self.aux_buffer[slot_offset : slot_offset + num_elements].copy_(flat)
        self._aux_lengths[req_id] = num_elements

        return {
            "aux_base_addr": self.aux_base_addr,
            "aux_device_id": self.device_id,
            "aux_slot_index": slot,
            "aux_num_elements": num_elements,
            "aux_slot_size_bytes": self.aux_slot_size_bytes,
        }

    # ── get_aux (decode side) ─────────────────────────────────────────

    def get_aux(
        self,
        name: str,
        req_id: str,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor | None:
        """Return transferred aux tensor, or None if not ready."""
        if not self.aux_enabled or self.aux_buffer is None:
            return None

        if req_id not in self._aux_ready:
            # Check if still in-progress.
            self._poll_aux_transfers()
            if req_id not in self._aux_ready:
                return None

        slot = self.aux_allocator.get_slot(req_id)
        if slot is None:
            return None

        num_elements = self._aux_lengths.get(req_id, 0)
        if num_elements == 0:
            return None

        slot_offset = slot * (self.aux_max_seq_len * self.aux_hidden_size)
        flat = self.aux_buffer[slot_offset : slot_offset + num_elements]
        # Reshape to [num_tokens, hidden_size].
        num_tokens = num_elements // self.aux_hidden_size
        tensor = flat.reshape(num_tokens, self.aux_hidden_size)

        self._aux_ready.discard(req_id)
        return tensor.to(device)

    # ── free_aux ──────────────────────────────────────────────────────

    def free_aux(self, req_id: str) -> None:
        self.aux_allocator.free(req_id)
        self._aux_lengths.pop(req_id, None)
        self._aux_ready.discard(req_id)
        self._aux_recving_transfers.pop(req_id, None)

    # ── NIXL aux transfer (decode side) ───────────────────────────────

    def start_load_aux(
        self,
        metadata: NixlConnectorMetadata,
    ) -> None:
        """Issue NIXL READs for aux data alongside KV loads."""
        if not self.aux_enabled or self.aux_buffer is None:
            return

        # ── Retry pending aux loads whose handshakes have completed ──
        if self._pending_aux_loads:
            still_pending: dict[str, AuxTransferInfo] = {}
            for req_id, info in self._pending_aux_loads.items():
                remote_agents = self._remote_agents.get(info.remote_engine_id)
                if remote_agents:
                    logger.info(
                        "start_load_aux: retrying pending aux for "
                        "req=%s (handshake now complete)",
                        req_id,
                    )
                    self._issue_aux_rdma_read(req_id, info)
                else:
                    still_pending[req_id] = info
            self._pending_aux_loads = still_pending

        # ── Process new aux requests from this batch ─────────────────
        aux_infos: dict[str, AuxTransferInfo] = getattr(metadata, "aux_to_recv", {})
        if not aux_infos:
            return

        for req_id, info in aux_infos.items():
            remote_agents = self._remote_agents.get(info.remote_engine_id)
            if not remote_agents:
                # Handshake not done yet — queue for retry.
                logger.info(
                    "start_load_aux: no remote agent yet for "
                    "engine=%s, queuing req=%s for retry",
                    info.remote_engine_id,
                    req_id,
                )
                self._pending_aux_loads[req_id] = info
                continue

            self._issue_aux_rdma_read(req_id, info)

    def _issue_aux_rdma_read(self, req_id: str, info: AuxTransferInfo) -> None:
        """Allocate a local slot and issue NIXL RDMA READ for aux data."""
        # Allocate a local slot.
        local_slot = self.aux_allocator.alloc(req_id)
        if local_slot is None:
            # No slot available — queue for retry on the next step,
            # matching how KV cache applies backpressure instead of
            # silently dropping data.
            logger.warning(
                "start_load_aux: no local slot for req=%s "
                "(used=%d/%d), queuing for retry",
                req_id,
                self.aux_allocator.num_used,
                self.aux_allocator.max_slots,
            )
            self._pending_aux_loads[req_id] = info
            return

        self._aux_lengths[req_id] = info.num_elements

        remote_agents = self._remote_agents.get(info.remote_engine_id)
        assert remote_agents, (
            f"_issue_aux_rdma_read called without remote agent "
            f"for engine={info.remote_engine_id}"
        )
        # Use rank 0 (aux is not TP-sharded).
        remote_agent_name = next(iter(remote_agents.values()))

        # Create remote descriptor (lazily, per-request).
        remote_addr = info.remote_aux_base_addr + info.slot_index * info.slot_size_bytes
        remote_descs = self.nixl_wrapper.get_xfer_descs(
            [(remote_addr, info.slot_size_bytes, info.remote_device_id)],
            self.nixl_memory_type,
        )
        remote_handle = self.nixl_wrapper.prep_xfer_dlist(
            remote_agent_name, remote_descs
        )

        # Issue RDMA READ.
        try:
            local_ids = np.array([local_slot], dtype=np.uint64)
            remote_ids = np.array([0], dtype=np.uint64)
            xfer_handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                self._aux_local_xfer_handle,
                local_ids,
                remote_handle,
                remote_ids,
            )
            self.nixl_wrapper.transfer(xfer_handle)
            self._aux_recving_transfers[req_id].append(xfer_handle)
        except Exception as e:
            logger.error(
                "start_load_aux: NIXL transfer failed for req=%s: %s",
                req_id,
                e,
            )
            self.aux_allocator.free(req_id)

    def _poll_aux_transfers(self) -> None:
        """Check completion of pending aux transfers."""
        for req_id, handles in list(self._aux_recving_transfers.items()):
            still_pending = []
            for handle in handles:
                try:
                    state = self.nixl_wrapper.check_xfer_state(handle)
                    if state == "DONE":
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif state == "PROC":
                        still_pending.append(handle)
                    else:
                        logger.warning(
                            "aux transfer failed for req=%s state=%s",
                            req_id,
                            state,
                        )
                        self.nixl_wrapper.release_xfer_handle(handle)
                except Exception as e:
                    logger.error(
                        "aux transfer poll error for req=%s: %s",
                        req_id,
                        e,
                    )
            if still_pending:
                self._aux_recving_transfers[req_id] = still_pending
            else:
                del self._aux_recving_transfers[req_id]
                self._aux_ready.add(req_id)

    # ── Override get_finished to clean up aux ─────────────────────────

    def get_finished(self) -> tuple[set[str], set[str]]:
        finished_sending, finished_recving = super().get_finished()
        # Free aux slots for requests done sending (prefill side).
        for req_id in finished_sending:
            self.free_aux(req_id)
        return finished_sending, finished_recving


# ── Scheduler subclass ────────────────────────────────────────────────


class NixlConnectorSchedulerWithAux(NixlConnectorScheduler):
    """Extends scheduler to track aux metadata per request."""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        super().__init__(vllm_config, engine_id)

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.aux_enabled = str(extra.get("transfer_aux", "false")).lower() == "true"
        # req_id -> aux metadata dict (from put_aux on the worker)
        self._aux_put_info: dict[str, dict[str, Any]] = {}
        # req_id -> AuxTransferInfo (from remote prefill, to be sent to worker)
        self._pending_aux_recv: dict[str, AuxTransferInfo] = {}

    def set_aux_info(self, req_id: str, info: dict[str, Any]) -> None:
        """Called (indirectly) when the worker finishes put_aux."""
        self._aux_put_info[req_id] = info

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        delay, params = super().request_finished(request, block_ids)

        if params is not None and self.aux_enabled:
            aux = self._aux_put_info.pop(request.request_id, None)
            if aux:
                params.update(aux)
                params["aux_available"] = True
            else:
                params["aux_available"] = False

        return delay, params

    def update_state_after_alloc(
        self,
        request: Request,
        blocks,
        num_external_tokens: int,
    ):
        super().update_state_after_alloc(request, blocks, num_external_tokens)

        # If the request has aux info from the remote prefill, store
        # it so build_connector_meta can attach it to worker metadata.
        if not self.aux_enabled:
            return
        params = request.kv_transfer_params
        if not params or not params.get("aux_available"):
            return

        # This request has transferred aux data. Store the info
        # so the worker can issue an RDMA READ.
        self._pending_aux_recv[request.request_id] = AuxTransferInfo(
            remote_aux_base_addr=params["aux_base_addr"],
            remote_device_id=params["aux_device_id"],
            slot_index=params["aux_slot_index"],
            num_elements=params["aux_num_elements"],
            slot_size_bytes=params["aux_slot_size_bytes"],
            remote_engine_id=params.get("remote_engine_id", ""),
        )

    def build_connector_meta(self, scheduler_output) -> NixlConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)

        # Attach pending aux recv info for the decode worker.
        aux_to_recv: dict[str, AuxTransferInfo] = {}
        for req_id in list(self._pending_aux_recv):
            if req_id in meta.reqs_to_recv:
                aux_to_recv[req_id] = self._pending_aux_recv.pop(req_id)

        # Dynamically attach aux info to the metadata object.
        # The worker will check for this attribute.
        meta.aux_to_recv = aux_to_recv  # type: ignore[attr-defined]
        return meta


# ── Top-level connector ───────────────────────────────────────────────


class NixlConnectorWithAux(NixlConnector):
    """NixlConnector extended with RDMA-based auxiliary tensor transfer.

    Lifecycle (same as KV, with aux piggybacking):

        Prefill worker               Decode worker
        ──────────────               ─────────────
        register_kv_caches
        + register aux buffer
           │
        model forward
           │
        put_aux(name, req_id, tensor)
           │  (copy into RDMA-registered GPU buffer)
           ▼
        request_finished
           │  (kv_transfer_params includes aux_base_addr, slot, etc.)
           ▼
        [HTTP proxy] ──────────►  update_state_after_alloc
                                     │  (stores AuxTransferInfo)
                                     ▼
                                  build_connector_meta
                                     │  (attaches aux_to_recv)
                                     ▼
                                  start_load_kv + start_load_aux
                                     │  (NIXL RDMA READ for aux)
                                     ▼
                                  get_aux(name, req_id)
                                     │  (returns tensor from local buffer)
                                     ▼
                                  EAGLE warm-up uses tensor
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # Call grandparent init (KVConnectorBase_V1), skip NixlConnector's
        # __init__ so we can use our own worker/scheduler subclasses.
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            KVConnectorRole,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorSchedulerWithAux(
                vllm_config, self.engine_id
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorkerWithAux(
                vllm_config, self.engine_id
            )

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.aux_enabled = str(extra.get("transfer_aux", "false")).lower() == "true"
        # Pending aux metadata from put_aux, to be sent via KVConnectorOutput.
        self._pending_aux_meta: dict[str, dict[str, Any]] = {}

    # ── Worker-side: put / get / free ─────────────────────────────────

    def put_aux(self, name: str, req_id: str, tensor: torch.Tensor) -> None:
        """Save an aux tensor into the RDMA-registered buffer."""
        if not self.aux_enabled:
            return
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, NixlConnectorWorkerWithAux)
        info = self.connector_worker.put_aux(name, req_id, tensor)

        # Store aux metadata on the instance. It will be collected by
        # get_aux_meta() and sent to the scheduler via KVConnectorOutput.
        if info is not None:
            self._pending_aux_meta[req_id] = info

    def get_aux_meta(self) -> dict[str, dict[str, Any]] | None:
        """Return and clear pending aux metadata from put_aux calls.

        Called by the model runner mixin after get_finished() to populate
        KVConnectorOutput.aux_meta, which flows back to the scheduler.
        """
        if not self._pending_aux_meta:
            return None
        meta = dict(self._pending_aux_meta)
        self._pending_aux_meta.clear()
        return meta

    def get_aux(
        self,
        name: str,
        req_id: str,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor | None:
        """Retrieve a transferred aux tensor."""
        if not self.aux_enabled:
            return None
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, NixlConnectorWorkerWithAux)
        return self.connector_worker.get_aux(name, req_id, device)

    def free_aux(self, req_id: str) -> None:
        if self.connector_worker is not None:
            assert isinstance(self.connector_worker, NixlConnectorWorkerWithAux)
            self.connector_worker.free_aux(req_id)

    # ── Override: start_load_kv ───────────────────────────────────────

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        """Start KV loads (NIXL RDMA) and aux loads."""
        super().start_load_kv(forward_context, **kwargs)

        if not self.aux_enabled:
            return
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        assert isinstance(self.connector_worker, NixlConnectorWorkerWithAux)
        self.connector_worker.start_load_aux(self._connector_metadata)

    # ── Override: update_connector_output (scheduler side) ──────────────

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Receive aux metadata from the worker via KVConnectorOutput.

        This is called on the scheduler side after each step. The worker
        populates connector_output.aux_meta in get_aux_meta(); we forward
        it to the scheduler-side connector so request_finished() can
        include it in kv_transfer_params.
        """
        if (
            self.aux_enabled
            and self.connector_scheduler is not None
            and connector_output.aux_meta
        ):
            assert isinstance(
                self.connector_scheduler,
                NixlConnectorSchedulerWithAux,
            )
            for req_id, info in connector_output.aux_meta.items():
                self.connector_scheduler.set_aux_info(req_id, info)

    # ── Override: request_finished (scheduler side) ────────────────────

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)
