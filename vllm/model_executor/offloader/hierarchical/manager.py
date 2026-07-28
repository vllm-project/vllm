# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExpertTierManager: ensure / remap / prefetch across device↔RAM↔disk."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config.offload import HierarchicalOffloadConfig
from vllm.logger import init_logger
from vllm.model_executor.offloader.hierarchical.device_slots import ExpertSlotPool
from vllm.model_executor.offloader.hierarchical.disk_store import (
    ExpertStoreReader,
    ensure_store_or_none,
)
from vllm.model_executor.offloader.hierarchical.format import (
    convert_layer_from_device_params,
    pack_expert_row_torch,
)
from vllm.model_executor.offloader.hierarchical.metrics import (
    TierStats,
    increment_prom,
    record_stats,
)
from vllm.model_executor.offloader.hierarchical.planner import (
    build_tier_plan,
    log_tier_plan,
    resolve_ram_budget_bytes,
)
from vllm.model_executor.offloader.hierarchical.ram_cache import PinnedExpertRamCache
from vllm.model_executor.offloader.hierarchical.usage import (
    ExpertUsageStore,
    default_usage_path,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.offloader.hierarchical.pilot import PilotPrefetcher

logger = init_logger(__name__)

_WEIGHT_PARAM_NAMES = (
    # Unquantized / MXFP4 / compressed-tensors style
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    # AWQ / GPTQ / WNA16 MoE style
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
    "w13_g_idx",
    "w2_g_idx",
    "w13_g_idx_sort_indices",
    "w2_g_idx_sort_indices",
)


class LayerTierState:
    """Per-MoE-layer staging state."""

    def __init__(
        self,
        layer_id: int,
        module: nn.Module,
        host_weights: list[torch.Tensor],
        param_names: list[str],
        slot_pool: ExpertSlotPool,
        row_nbytes: int,
        expert_map: torch.Tensor | None = None,
    ):
        self.layer_id = layer_id
        self.module = module
        self.host_weights = host_weights  # full [E, ...] on CPU (pinned preferred)
        self.param_names = param_names
        self.slot_pool = slot_pool
        self.row_nbytes = row_nbytes
        self.num_experts = host_weights[0].shape[0]
        # Optional global→local map captured before slot rebind.
        self.expert_map = expert_map

    def to_local(self, expert_id: int) -> int:
        """Map a router expert id to a local host-pack index, or -1."""
        if expert_id < 0:
            return -1
        if self.expert_map is not None:
            if expert_id >= self.expert_map.numel():
                return -1
            local = int(self.expert_map[expert_id].item())
            return local if local >= 0 else -1
        if expert_id >= self.num_experts:
            return -1
        return expert_id


class ExpertTierManager:
    """Owns the 3-tier hierarchy for all MoE layers on this worker."""

    def __init__(self, config: HierarchicalOffloadConfig, model_path: str | None = None):
        self.config = config
        self.layers: dict[int, LayerTierState] = {}
        self.stats = TierStats()
        self.copy_stream = current_platform.Stream()
        self._ram: PinnedExpertRamCache | None = None
        self._disk: ExpertStoreReader | None = None
        self._usage: ExpertUsageStore | None = None
        self._pilot: PilotPrefetcher | None = None
        self._tokens_since_repin = 0
        self._initialized = False
        self._model_path = model_path
        self._pending_modules: list[tuple[int, nn.Module]] = []
        # MoE expert modules still holding full packs on device during
        # construct/load. Oldest are spilled to host only under VRAM pressure
        # so we use GPU+RAM together instead of host-only (which causes swap).
        self._device_resident_experts: list[nn.Module] = []

    def register_moe_module(self, layer_id: int, module: nn.Module) -> None:
        """Queue a RoutedExperts (or parent) module for post_init staging."""
        self._pending_modules.append((layer_id, module))

    def _device_mem_bytes(self) -> tuple[int, int]:
        """Return (free, total) device bytes; conservative fallback on error."""
        try:
            free, total = torch.accelerator.get_memory_info()
            return int(free), int(total)
        except Exception:
            try:
                free, total = torch.xpu.mem_get_info()
                return int(free), int(total)
            except Exception:
                return 2 * 1024**3, 32 * 1024**3

    def _load_vram_reserve_bytes(self) -> int:
        """Keep this much device memory free for the next layer alloc."""
        # ~3 GiB headroom for next MoE create_weights + attention.
        return 3 * 1024**3

    def _park_module_to_host(self, module: nn.Module) -> int:
        """Move one module's expert packs to pageable CPU. Returns #params moved."""
        params = dict(module.named_parameters(recurse=False))
        moved = 0
        for name in _WEIGHT_PARAM_NAMES:
            if name not in params:
                continue
            p = params[name]
            if p.device.type == "cpu":
                continue
            cpu = torch.empty_like(p, device="cpu", pin_memory=False)
            cpu.copy_(p.detach())
            p.data = cpu
            moved += 1
        if hasattr(module, "w13_qweight"):
            module.w13_weight = module.w13_qweight
        if hasattr(module, "w2_qweight"):
            module.w2_weight = module.w2_qweight
        return moved

    def park_experts_on_host(self, module: nn.Module) -> None:
        """Keep experts on device until VRAM is tight, then spill oldest.

        Combined GPU+host capacity is what avoids swap for models like
        Mixtral-8x22B AWQ (~69 GiB) on a 32 GiB Arc + 62 GiB host. Blindly
        parking every layer to host left the GPU idle and overflowed RAM.
        """
        self._device_resident_experts.append(module)
        reserve = self._load_vram_reserve_bytes()
        parked_modules = 0
        parked_params = 0
        while len(self._device_resident_experts) > 1:
            free, total = self._device_mem_bytes()
            if free >= reserve:
                break
            oldest = self._device_resident_experts.pop(0)
            n = self._park_module_to_host(oldest)
            if n:
                parked_modules += 1
                parked_params += n
                try:
                    torch.accelerator.empty_cache()
                except Exception:
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
        if parked_modules:
            free, total = self._device_mem_bytes()
            logger.info(
                "VRAM pressure: parked %d MoE layer(s) (%d params) to host; "
                "device free=%.2f/%.2f GiB; still resident=%d",
                parked_modules,
                parked_params,
                free / 1024**3,
                total / 1024**3,
                len(self._device_resident_experts),
            )

    def post_init(self) -> None:
        if self._initialized:
            return
        if not self._pending_modules:
            logger.warning(
                "HierarchicalOffloader active but no MoE modules registered"
            )
            self._initialized = True
            return

        # Discover weight params from the first module to size the plan.
        sample_layer_id, sample_mod = self._pending_modules[0]
        param_names, host_like = self._extract_expert_params(sample_mod)
        if not host_like:
            logger.warning("No expert weight params found; hierarchical disabled")
            self._initialized = True
            return

        row_nbytes = sum(
            (p.numel() // p.shape[0]) * p.element_size() for p in host_like
        )
        num_local = host_like[0].shape[0]
        top_k = getattr(sample_mod, "top_k", 8) or 8
        plan = build_tier_plan(
            self.config,
            num_moe_layers=len(self._pending_modules),
            num_local_experts=num_local,
            expert_row_bytes=row_nbytes,
            top_k=int(top_k),
        )
        log_tier_plan(plan)
        slots = plan.slots_per_layer

        # RAM cache sized to max row across layers (assume uniform).
        ram_budget = resolve_ram_budget_bytes(self.config)
        self._ram = PinnedExpertRamCache(ram_budget, row_nbytes)

        usage_path = self.config.tier_usage_path or default_usage_path(
            self.config.tier_disk_path, self._model_path
        )
        self._usage = ExpertUsageStore(usage_path)

        self._disk = ensure_store_or_none(
            self.config.tier_disk_path,
            num_workers=self.config.tier_io_workers,
            prefer_direct=self.config.tier_direct,
        )

        for layer_id, module in self._pending_modules:
            # Dedupe by layer_id (MoERunner + wrap_modules may both register).
            if layer_id in self.layers:
                continue
            pnames, weights = self._extract_expert_params(module)
            if not weights:
                continue
            # Host copies of full expert packs. Reuse storage when params are
            # already parked on CPU (avoid 2× RAM for large MoE).
            host_weights: list[torch.Tensor] = []
            pin = True
            try:
                sample_nbytes = sum(w.numel() * w.element_size() for w in weights)
                pin = sample_nbytes <= 512 * 1024 * 1024
            except Exception:
                pin = False
            for w in weights:
                if w.device.type == "cpu":
                    host_weights.append(w.detach())
                    continue
                cpu = torch.empty_like(w, device="cpu", pin_memory=pin)
                cpu.copy_(w.detach())
                host_weights.append(cpu)

            # Optionally spill to ExpertStore.
            if self.config.tier_disk_path and self._disk is None:
                try:
                    convert_layer_from_device_params(
                        self.config.tier_disk_path,
                        layer_id,
                        host_weights,
                        model_id=self._model_path or "unknown",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build ExpertStore for layer %d: %s", layer_id, e
                    )
                self._disk = ensure_store_or_none(
                    self.config.tier_disk_path,
                    num_workers=self.config.tier_io_workers,
                    prefer_direct=self.config.tier_direct,
                )

            # Seed RAM with hottest experts (or first slots-worth).
            hot = self._usage.hottest(layer_id, slots * 4, host_weights[0].shape[0])
            for eid in hot:
                if not self._ram.enabled:
                    break
                row = pack_expert_row_torch([w[eid] for w in host_weights])
                self._ram.put(layer_id, eid, row, pinned=True)

            slot_pool = ExpertSlotPool(
                layer_id=layer_id,
                weight_templates=host_weights,
                num_slots=slots,
                copy_stream=self.copy_stream,
            )

            # Point module parameters at device slot buffers and rebuild
            # MoE kernels for the reduced slot pack size.
            for name, slot_buf in zip(pnames, slot_pool.slot_weights):
                param = dict(module.named_parameters())[name]
                param.data = slot_buf

            # Preserve global→local map for ensure(); kernel sees slot ids.
            expert_map = None
            if hasattr(module, "_expert_map"):
                expert_map = getattr(module, "_expert_map", None)
                if expert_map is not None:
                    expert_map = expert_map.detach().clone()

            self._rebind_module_for_slots(module, slots)

            self.layers[layer_id] = LayerTierState(
                layer_id=layer_id,
                module=module,
                host_weights=host_weights,
                param_names=pnames,
                slot_pool=slot_pool,
                row_nbytes=row_nbytes,
                expert_map=expert_map,
            )

            # Prefetch initial hot set into device slots (fill all slots so
            # VRAM is front-loaded rather than left half-empty).
            init_ids = list(hot[:slots])
            if len(init_ids) < slots:
                for eid in range(host_weights[0].shape[0]):
                    if eid not in init_ids:
                        init_ids.append(eid)
                    if len(init_ids) >= slots:
                        break
            self._ensure_layer(layer_id, init_ids, record_usage=False)

        if self.config.tier_pilot:
            from vllm.model_executor.offloader.hierarchical.pilot import (
                PilotPrefetcher,
            )

            self._pilot = PilotPrefetcher(self, real=self.config.tier_pilot_real)

        self._pending_modules.clear()
        self._initialized = True
        logger.info(
            "ExpertTierManager ready: %d MoE layers, slots/layer=%d",
            len(self.layers),
            slots,
        )

    @staticmethod
    def _extract_expert_params(
        module: nn.Module,
    ) -> tuple[list[str], list[torch.Tensor]]:
        params = dict(module.named_parameters())
        names: list[str] = []
        tensors: list[torch.Tensor] = []
        for name in _WEIGHT_PARAM_NAMES:
            if name in params and params[name].dim() >= 2:
                names.append(name)
                tensors.append(params[name].data)
        # Also accept nested RoutedExperts
        if not tensors:
            for child in module.modules():
                if child is module:
                    continue
                cparams = dict(child.named_parameters(recurse=False))
                for name in _WEIGHT_PARAM_NAMES:
                    if name in cparams and cparams[name].dim() >= 2:
                        names.append(name)
                        tensors.append(cparams[name].data)
                if tensors:
                    # Rebind module to the child that owns weights
                    break
        return names, tensors

    @staticmethod
    def _rebind_module_for_slots(module: nn.Module, num_slots: int) -> None:
        """Update expert counts and recreate MoE kernels for the slot pack.

        After ``param.data`` is replaced with ``[E_slots, ...]`` tensors, any
        kernel that captured the old full pack / expert count must be rebuilt.
        """
        if hasattr(module, "local_num_experts"):
            module.local_num_experts = num_slots
        moe_cfg = getattr(module, "moe_config", None)
        if moe_cfg is not None and hasattr(moe_cfg, "num_local_experts"):
            moe_cfg.num_local_experts = num_slots

        # Clear EP expert_map so remapped slot ids are used directly.
        if hasattr(module, "_expert_map"):
            module.register_buffer("_expert_map", None, persistent=False)

        # Keep AWQ aliases pointing at the (possibly rebound) qweight params.
        if hasattr(module, "w13_qweight"):
            module.w13_weight = module.w13_qweight
        if hasattr(module, "w2_qweight"):
            module.w2_weight = module.w2_qweight

        for child in module.modules():
            if hasattr(child, "fused_moe_impl"):
                child.fused_moe_impl = None

        quant = getattr(module, "quant_method", None)
        if quant is None:
            return

        if hasattr(quant, "fused_moe_impl"):
            quant.fused_moe_impl = None

        # MXFP4 / compressed-tensors path builds moe_kernel eagerly.
        if hasattr(quant, "moe") and hasattr(quant.moe, "num_local_experts"):
            quant.moe.num_local_experts = num_slots

        if hasattr(quant, "moe_kernel"):
            # Prefer quant-method helpers that know how to rebuild (AWQ WNA16).
            if hasattr(quant, "_setup_kernel"):
                try:
                    quant._setup_kernel(module)
                    logger.info(
                        "Rebuilt WNA16/AWQ MoE kernel for %d expert slots",
                        num_slots,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to rebuild AWQ MoE kernel after slot rebind: %s",
                        e,
                    )
                    quant.moe_kernel = None
            try:
                from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
                    make_mxfp4_moe_kernel,
                )

                if (
                    getattr(quant, "moe_quant_config", None) is not None
                    and getattr(quant, "moe", None) is not None
                ):
                    # Refresh quant config against the new slot-sized weights.
                    if hasattr(quant, "get_fused_moe_quant_config"):
                        quant.moe_quant_config = quant.get_fused_moe_quant_config(
                            module
                        )
                    routing_tables = None
                    if hasattr(module, "_expert_routing_tables"):
                        try:
                            routing_tables = module._expert_routing_tables()
                        except Exception:
                            routing_tables = None
                    quant.moe_kernel = make_mxfp4_moe_kernel(
                        moe_quant_config=quant.moe_quant_config,
                        moe_config=quant.moe,
                        experts_cls=getattr(quant, "experts_cls", None),
                        mxfp4_backend=getattr(quant, "mxfp4_backend", None),
                        routing_tables=routing_tables,
                    )
                    logger.info(
                        "Rebuilt MXFP4 MoE kernel for %d expert slots", num_slots
                    )
                else:
                    quant.moe_kernel = None
            except Exception as e:
                logger.warning(
                    "Failed to rebuild MoE kernel after slot rebind: %s", e
                )
                quant.moe_kernel = None

    @staticmethod
    def _invalidate_fused_impl(module: nn.Module) -> None:
        ExpertTierManager._rebind_module_for_slots(
            module,
            num_slots=getattr(module, "local_num_experts", 1) or 1,
        )

    def ensure_and_remap(
        self, layer_id: int, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """Ensure experts for ``topk_ids`` are on device; return remapped ids."""
        if layer_id not in self.layers:
            return topk_ids

        state = self.layers[layer_id]
        flat = topk_ids.reshape(-1)
        unique = torch.unique(flat).tolist()
        # Router ids may be global; convert to local pack indices first.
        global_to_local: dict[int, int] = {}
        local_ids: list[int] = []
        for gid in unique:
            g = int(gid)
            if g < 0:
                continue
            local = state.to_local(g)
            if local < 0:
                continue
            global_to_local[g] = local
            local_ids.append(local)

        local_remap = self._ensure_layer(
            layer_id, list(set(local_ids)), record_usage=True
        )

        # Remap original (global) topk ids → slot ids for the kernel.
        remapped = topk_ids.clone()
        for gid, local in global_to_local.items():
            sid = local_remap.get(local)
            if sid is None:
                continue
            remapped = torch.where(topk_ids == gid, sid, remapped)
        return remapped

    def maybe_pilot_prefetch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        if self._pilot is None:
            return
        ids = [int(x) for x in torch.unique(topk_ids).tolist() if int(x) >= 0]
        self._pilot.prefetch_next(layer_id, hidden_states, ids)

    def prefetch_experts(
        self, layer_id: int, expert_ids: list[int], *, block: bool = False
    ) -> None:
        if layer_id not in self.layers:
            return
        self._ensure_layer(layer_id, expert_ids, record_usage=False)
        if block:
            current_platform.current_stream().wait_stream(self.copy_stream)

    def notify_tokens(self, n: int) -> None:
        """Advance repin clock by ``n`` emitted tokens."""
        if self.config.tier_policy != "balanced":
            return
        if self.config.tier_repin_tokens <= 0:
            return
        self._tokens_since_repin += n
        if self._tokens_since_repin < self.config.tier_repin_tokens:
            return
        self._tokens_since_repin = 0
        if self._ram is None or self._usage is None:
            return
        for layer_id, state in self.layers.items():
            hot = self._usage.hottest(
                layer_id, state.slot_pool.num_slots, state.num_experts
            )
            self._ram.repin_hottest(layer_id, hot, max_swaps=4)

    def shutdown(self) -> None:
        if self._usage is not None:
            self._usage.flush()
        if self._disk is not None:
            self._disk.close()
        record_stats(self.stats)

    def _ensure_layer(
        self,
        layer_id: int,
        expert_ids: list[int],
        *,
        record_usage: bool,
    ) -> dict[int, int]:
        state = self.layers[layer_id]
        pool = state.slot_pool
        host_rows: dict[int, list[torch.Tensor]] = {}
        needed = [e for e in expert_ids if e >= 0 and not pool.contains(e)]

        t0 = time.perf_counter_ns()
        # Always materialize host rows for every requested expert up front.
        # Slot eviction within ensure_from_host_rows can invalidate earlier
        # contains() hits from this same batch.
        for eid in expert_ids:
            if eid < 0:
                continue
            if eid in host_rows:
                continue
            if pool.contains(eid):
                self.stats.device_hits += 1
                increment_prom("device")
                # Still provide a host row in case this slot is evicted while
                # bringing in other experts from this batch.
            row_view = self._ram.get(layer_id, eid) if self._ram else None
            if row_view is not None:
                if eid not in host_rows:
                    self.stats.ram_hits += 1
                    increment_prom("ram")
                host_rows[eid] = self._unpack_host_row(state, row_view)
                continue
            if self._disk is not None and self._disk.has_layer(layer_id):
                blob = self._disk.read_row_sync(layer_id, eid)
                if self._ram and self._ram.enabled:
                    self._ram.put(layer_id, eid, blob)
                self.stats.disk_hits += 1
                increment_prom("disk")
                host_rows[eid] = self._disk.unpack_row(layer_id, blob)
                continue
            if eid >= state.num_experts:
                raise KeyError(
                    f"layer {layer_id}: expert id {eid} out of range "
                    f"for host pack size {state.num_experts}"
                )
            if not pool.contains(eid):
                self.stats.ram_hits += 1
                increment_prom("ram")
            host_rows[eid] = [w[eid] for w in state.host_weights]
            if self._ram and self._ram.enabled:
                packed = pack_expert_row_torch(host_rows[eid])
                self._ram.put(layer_id, eid, packed)

        remap, events = pool.ensure_from_host_rows(expert_ids, host_rows)

        # Wait for in-flight copies of experts we need
        compute = current_platform.current_stream()
        for ev in events:
            compute.wait_event(ev)
        pool.mark_ready(list(remap.keys()))

        stall = time.perf_counter_ns() - t0
        self.stats.stall_ns += stall
        self.stats.ensures += 1
        self.stats.unique_experts += len(set(expert_ids))
        dma = sum(
            state.row_nbytes for e in needed if e in remap
        )
        self.stats.dma_bytes += dma
        if dma:
            increment_prom("device", dma_bytes=dma, stall_ns=stall)

        if record_usage and self._usage is not None:
            self._usage.record(layer_id, expert_ids)

        # Build full remap including already-resident
        full: dict[int, int] = {}
        for eid in expert_ids:
            if eid < 0:
                continue
            sid = pool.slot_of(eid)
            if sid is not None:
                full[eid] = sid
        full.update(remap)
        return full

    def _unpack_host_row(
        self, state: LayerTierState, row_view: torch.Tensor
    ) -> list[torch.Tensor]:
        """Unpack a packed uint8 RAM row into typed expert slices."""
        out: list[torch.Tensor] = []
        offset = 0
        for w in state.host_weights:
            nbytes = (w.numel() // w.shape[0]) * w.element_size()
            chunk = row_view[offset : offset + nbytes]
            offset += nbytes
            out.append(
                chunk.view(w.dtype).reshape(w.shape[1:]).clone()
            )
        return out


_manager: ExpertTierManager | None = None


def get_tier_manager() -> ExpertTierManager | None:
    return _manager


def set_tier_manager(manager: ExpertTierManager | None) -> None:
    global _manager
    _manager = manager
