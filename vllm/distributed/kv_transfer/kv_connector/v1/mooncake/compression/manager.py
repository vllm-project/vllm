# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from .codec import EncodedChunk
from .config import MooncakeCompressionConfig
from .layout import (
    ALIGNMENT,
    CompressionLayout,
    KVPart,
    LayerPartLayout,
    build_compression_layout,
)
from .memory import (
    FixedWorkspaceAllocator,
    MooncakeCompressionMemoryPlan,
    register_workspace_allocators,
    unregister_workspace_allocators,
)
from .pipeline import CompressionPipeline, CompressionSlotBuffers
from .scores import build_head_selection

logger = init_logger(__name__)


@dataclass(frozen=True)
class ExecutionPlan:
    config: MooncakeCompressionConfig
    layout: CompressionLayout
    signs: dict[tuple[str, KVPart], torch.Tensor]
    orders: dict[str, torch.Tensor]
    identity_layers: frozenset[str]

    @property
    def raw(self) -> bool:
        return not self.config.enabled or not self.config.pipeline

    @property
    def aggregate(self) -> bool:
        return "aggregate" in self.config.pipeline

    def data_bytes(self, kv_part: int, tokens: dict[int, int] | None = None) -> int:
        if self.aggregate:
            if tokens is None:
                return self.layout.raw_bytes_by_part[kv_part]
            total = 0
            for part in self.layout.parts(kv_part):
                count = tokens.get(part.group_index, 0)
                blocks = (count + part.block_size - 1) // part.block_size
                total += blocks * part.num_heads * part.block_size * part.head_size * 2
            return total
        if tokens is not None and not self.layout.has_body(tokens):
            return 0
        lengths = (
            self.layout.u8_bytes_by_part
            if "quantizer" in self.config.pipeline
            else self.layout.raw_bytes_by_part
        )
        return lengths[kv_part]

    def aux_bytes(self, kv_part: int, tokens: dict[int, int] | None = None) -> int:
        if tokens is not None and not self.layout.has_body(tokens):
            return 0
        return (
            self.layout.aux_bytes_by_part[kv_part]
            if "quantizer" in self.config.pipeline
            else 0
        )


def prepare_execution(
    memory: MooncakeCompressionMemoryPlan,
    config: MooncakeCompressionConfig,
    bound_layout: CompressionLayout,
    tp_size: int,
) -> ExecutionPlan:
    effective = config
    if not config.enabled or not config.pipeline:
        effective = replace(config, pipeline=())
    if "quantizer" not in effective.pipeline:
        effective = replace(effective, split_type="layer")
    if memory.specs:
        selection = build_head_selection(
            memory.specs, effective, memory.tp_rank, tp_size, memory.scores
        )
        num_layers = (
            memory.num_layers
            or max(part.layer_index for part in bound_layout.layer_parts) + 1
        )
        layout = build_compression_layout(
            memory.specs, effective, selection.low_counts, num_layers
        )
        groups = {
            part.layer_name: part.group_index for part in bound_layout.layer_parts
        }
        layout = replace(
            layout,
            layer_parts=tuple(
                replace(part, group_index=groups[part.layer_name])
                for part in layout.layer_parts
            ),
            fa_group_indices=bound_layout.fa_group_indices,
            index_rows_by_group=bound_layout.index_rows_by_group,
            arena_bytes=bound_layout.arena_bytes,
            aux_arena_bytes=bound_layout.aux_arena_bytes,
        )
    else:
        selection, layout = memory.head_selection, bound_layout
    signs, orders = {}, {}
    for part in layout.layer_parts:
        order = selection.local_orders[part.layer_index]
        if part.layer_name not in orders:
            orders[part.layer_name] = torch.tensor(
                order, dtype=torch.int64
            ).pin_memory()
        if not effective.uses_transformer:
            continue
        signs[(part.layer_name, part.kv_part)] = build_signs(
            part, order, effective.seed, pin_memory=True
        )
    identity_layers = frozenset(
        name
        for name, order in orders.items()
        if tuple(order.tolist()) == tuple(range(order.numel()))
    )
    return ExecutionPlan(effective, layout, signs, orders, identity_layers)


def build_signs(
    part: LayerPartLayout,
    order: tuple[int, ...],
    seed: int,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    cpu = torch.empty(
        (part.num_heads, part.head_size), dtype=part.dtype, pin_memory=pin_memory
    )
    generator = torch.Generator(device="cpu")
    for output_head, original_head in enumerate(order):
        generator.manual_seed(seed ^ (part.layer_index << 16) ^ original_head)
        cpu[output_head].copy_(
            torch.randint(
                0, 2, (part.head_size,), generator=generator, dtype=torch.int8
            )
        ).mul_(2).sub_(1)
    return cpu.view(1, 1, part.num_heads, part.head_size)


class _PoolCarver:
    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self.offset = 0

    def take(self, nbytes: int) -> torch.Tensor:
        self.offset = round_up(self.offset, ALIGNMENT)
        end = self.offset + nbytes
        result = self.buffer[self.offset : end]
        self.offset = end
        return result


class MooncakeCompressionManager:
    def __init__(
        self,
        plan: MooncakeCompressionMemoryPlan,
        kv_cache_layout: str,
        blocks_per_logical: int,
        tp_size: int,
        pp_size: int,
        device: torch.device,
    ):
        self.plan = plan
        self.config = plan.config
        self.layout = plan.layout
        self.tp_size = tp_size
        self._owns_nvcomp_allocator = False

        self._raw_tail_capacity = self.layout.raw_tail_capacity
        arena_b_bytes = 0 if self.config.uses_aggregate else self.layout.arena_bytes
        slot_bytes = (
            self.layout.arena_bytes
            + arena_b_bytes
            + self.layout.aux_arena_bytes
            + self._raw_tail_capacity
        )
        fixed_bytes = (
            self.config.slot_count * slot_bytes
            + plan.codec_workspace_bytes
            + plan.metadata_bytes
        )
        self.bulk = torch.empty(fixed_bytes, dtype=torch.uint8, device=device)
        carver = _PoolCarver(self.bulk)

        max_physical_blocks = self.config.logical_blocks_per_chunk * blocks_per_logical
        self.slots: list[CompressionSlotBuffers] = []
        num_index_rows = max(
            len(self.layout.index_rows_by_group),
            len({part.layer_name for part in self.layout.layer_parts}),
            1,
        )
        for _ in range(self.config.slot_count):
            arena_a = carver.take(self.layout.arena_bytes)
            arena_b = carver.take(arena_b_bytes)
            aux = carver.take(self.layout.aux_arena_bytes)
            raw_tail = carver.take(self._raw_tail_capacity)
            indices_bytes = max_physical_blocks * num_index_rows * 8
            indices = (
                carver.take(indices_bytes)
                .view(torch.int64)
                .view(num_index_rows, max_physical_blocks)
            )
            self.slots.append(
                CompressionSlotBuffers(
                    arena_a=arena_a,
                    arena_b=arena_b,
                    aux=aux,
                    block_indices=indices,
                    completion_event=torch.cuda.Event(),
                    host_indices=torch.empty(
                        (num_index_rows, max_physical_blocks),
                        dtype=torch.int64,
                        pin_memory=True,
                    ),
                    raw_tail=raw_tail,
                )
            )

        # Compression kernels are tiny (gather/FWHT/quant/ANS) but sit on the
        # TTFT critical path; run them on high-priority streams so a saturated
        # prefill stream cannot starve them.
        self.streams = [
            torch.cuda.Stream(device=device, priority=-1) for _ in self.slots
        ]
        self.workspace_allocators: list[FixedWorkspaceAllocator] = []
        if self.config.uses_codec:
            self.workspace_allocators = [
                FixedWorkspaceAllocator(
                    carver.take(plan.codec_workspace_bytes // len(self.slots))
                )
                for _ in self.slots
            ]
        self._workspace_by_stream = dict(
            zip(
                (stream.cuda_stream for stream in self.streams),
                self.workspace_allocators,
            )
        )
        if self.config.uses_codec:
            register_workspace_allocators(self._workspace_by_stream)
            self._owns_nvcomp_allocator = True
        try:
            self.stream = self.streams[0]

            from .codec import ANSCodec

            self.pipelines: list[CompressionPipeline] = []
            self.slot_plans: list[ExecutionPlan | None] = [None] * len(self.slots)
            for index, stream in enumerate(self.streams):
                orders: dict[str, torch.Tensor] = {}
                signs: dict[tuple[str, KVPart], torch.Tensor] = {}
                for part in self.layout.layer_parts:
                    if part.layer_name not in orders:
                        order = carver.take(part.num_heads * 8).view(torch.int64)
                        if index == 0:
                            order.copy_(
                                torch.tensor(
                                    self.plan.head_selection.local_orders[
                                        part.layer_index
                                    ],
                                    dtype=torch.int64,
                                )
                            )
                        orders[part.layer_name] = order
                    if not self.config.uses_transformer:
                        continue
                    key = (part.layer_name, part.kv_part)
                    signs[key] = (
                        carver.take(part.num_heads * part.head_size * 2)
                        .view(part.dtype)
                        .view(1, 1, part.num_heads, part.head_size)
                    )
                    if index == 0:
                        order_values = self.plan.head_selection.local_orders[
                            part.layer_index
                        ]
                        signs[key].copy_(
                            build_signs(part, order_values, self.config.seed)
                        )
                codec = (
                    ANSCodec(stream, self.layout.packed_bytes_by_part)
                    if self.config.uses_codec
                    else None
                )
                self.pipelines.append(
                    CompressionPipeline(
                        layout=self.layout,
                        kv_cache_layout=kv_cache_layout,
                        blocks_per_logical=blocks_per_logical,
                        slots=self.slots,
                        codec=codec,
                        stream=stream,
                        signs=signs,
                        head_orders=orders,
                        stages=self.config.pipeline,
                    )
                )
            self.head_orders = self.pipelines[0].head_orders
            self.signs = self.pipelines[0].signs
            self._init_event = torch.cuda.Event()
            self._init_event.record(torch.cuda.current_stream(device))
            for stream in self.streams:
                stream.wait_event(self._init_event)
        except BaseException:
            self._release_workspace_allocator()
            raise
        logger.info(
            "Mooncake compression manager initialized: layout=%s pipeline=%s "
            "tp=%d pp=%d slots=%d bulk_bytes=%d arena_bytes=%d aux_arena_bytes=%d "
            "workspace_bytes=%d blocks_per_logical=%d",
            kv_cache_layout,
            self.config.pipeline,
            tp_size,
            pp_size,
            self.config.slot_count,
            fixed_bytes,
            self.layout.arena_bytes,
            self.layout.aux_arena_bytes,
            plan.codec_workspace_bytes,
            blocks_per_logical,
        )

    @property
    def fa_group_indices(self) -> frozenset[int]:
        return self.layout.fa_group_indices

    def _release_workspace_allocator(self):
        if self._owns_nvcomp_allocator:
            unregister_workspace_allocators(self._workspace_by_stream)
            self._owns_nvcomp_allocator = False

    def prepare(self) -> ExecutionPlan:
        return prepare_execution(self.plan, self.config, self.layout, self.tp_size)

    def _execution_pipeline(self, slot_id: int, execution: ExecutionPlan | None):
        pipeline = self.pipelines[slot_id]
        if execution is None:
            execution = self.default_execution
        previous = self.slot_plans[slot_id]
        if previous is not execution:
            with torch.cuda.stream(self.streams[slot_id]):
                for name, order in execution.orders.items():
                    pipeline.head_orders[name].copy_(order, non_blocking=True)
                for key, signs in execution.signs.items():
                    pipeline.signs[key].copy_(signs, non_blocking=True)
            pipeline.layout = execution.layout
            pipeline.stages = execution.config.pipeline
            pipeline.identity_orders = set(execution.identity_layers)
            pipeline._cpp_metas = None
            self.slot_plans[slot_id] = execution
        return pipeline

    def bind_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        group_indices: dict[str, int],
    ) -> None:
        layer_parts = tuple(
            replace(part, group_index=group_indices[part.layer_name])
            for part in self.layout.layer_parts
        )
        self.layout = replace(
            self.layout,
            layer_parts=layer_parts,
            fa_group_indices=frozenset(
                group_indices[part.layer_name] for part in layer_parts
            ),
            index_rows_by_group={
                group_index: row
                for row, group_index in enumerate(
                    sorted({part.group_index for part in layer_parts})
                )
            },
        )
        for pipeline in self.pipelines:
            pipeline.layout = self.layout
            pipeline.bind_kv_caches(kv_caches)
        self.default_execution = self.prepare()
        config = asdict(self.default_execution.config)
        for key in ("score_path", "slot_count"):
            config.pop(key, None)
        layout_description = asdict(self.default_execution.layout)
        layout_description["fa_group_indices"] = sorted(
            self.default_execution.layout.fa_group_indices
        )
        wire_layout = {
            "version": 1,
            "config": config,
            "layout": layout_description,
            "cache_layout": self.pipelines[0].kv_cache_layout,
            "blocks_per_logical": self.pipelines[0].blocks_per_logical,
            "orders": {
                name: order.tolist()
                for name, order in self.default_execution.orders.items()
            },
        }
        self.wire_profile = hashlib.sha256(
            json.dumps(wire_layout, sort_keys=True, default=str).encode()
        ).hexdigest()
        logger.info(
            "Mooncake compression KV caches bound: layers=%d fa_groups=%s",
            len(self.layout.layer_parts) // 2,
            sorted(self.layout.fa_group_indices),
        )
        logger.debug(
            "Mooncake compression cache tensors: %s",
            {
                name: {
                    "shape": tuple(cache.shape),
                    "stride": tuple(cache.stride()),
                    "data_ptr": cache.data_ptr(),
                }
                for name, cache in kv_caches.items()
                if name in {part.layer_name for part in self.layout.layer_parts}
            },
        )

    def registered_memory(self) -> tuple[list[int], list[int]]:
        storage = self.bulk.untyped_storage()
        return [storage.data_ptr()], [storage.nbytes()]

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        execution: ExecutionPlan | None = None,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> EncodedChunk:
        return self._execution_pipeline(slot_id, execution).encode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            valid_tokens_by_group=valid_tokens_by_group,
        )

    def decode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        codec_bytes: int,
        u8_bytes: int,
        execution: ExecutionPlan | None = None,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> torch.cuda.Event:
        plan = execution or self.default_execution
        expected_bytes = plan.data_bytes(kv_part, valid_tokens_by_group)
        if u8_bytes != expected_bytes:
            raise ValueError("Compressed chunk has an incompatible decoded size")
        if not 0 <= codec_bytes <= self.slots[slot_id].arena_a.numel():
            raise ValueError("Compressed chunk exceeds receive buffer capacity")
        if "codec" not in plan.config.pipeline and codec_bytes != u8_bytes:
            raise ValueError("Unencoded chunk payload size does not match its layout")
        return self._execution_pipeline(slot_id, execution).decode(
            slot_id,
            logical_ids_by_group,
            kv_part,
            codec_bytes,
            u8_bytes,
            valid_tokens_by_group=valid_tokens_by_group,
        )

    def shutdown(self) -> None:
        if not self._owns_nvcomp_allocator:
            return
        for stream in self.streams:
            stream.synchronize()
        self.pipelines.clear()
        self._release_workspace_allocator()
