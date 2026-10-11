# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

from . import cpp_backend
from .codec import ANSCodec, EncodedChunk
from .layout import CompressionLayout, KVPart, LayerPartLayout, QuantSection


@dataclass
class CompressionSlotBuffers:
    arena_a: torch.Tensor
    arena_b: torch.Tensor
    aux: torch.Tensor
    block_indices: torch.Tensor
    completion_event: torch.cuda.Event
    host_indices: torch.Tensor | None = None
    raw_tail: torch.Tensor | None = None


class CompressionPipeline:
    def __init__(
        self,
        layout: CompressionLayout,
        kv_cache_layout: str,
        blocks_per_logical: int,
        slots: list[CompressionSlotBuffers],
        codec: ANSCodec | None,
        stream: torch.cuda.Stream,
        signs: dict[tuple[str, KVPart], torch.Tensor],
        head_orders: dict[str, torch.Tensor],
        stages: tuple[str, ...] = ("transformer", "quantizer", "codec"),
    ):
        if kv_cache_layout == "LBHNC":
            kv_cache_layout = "HND"
        if kv_cache_layout != "HND":
            raise ValueError("Mooncake compression requires LBHNC/HND KV cache layout.")
        self.layout = layout
        self.kv_cache_layout = kv_cache_layout
        self.blocks_per_logical = blocks_per_logical
        self._physical_offsets = torch.arange(
            blocks_per_logical, dtype=torch.int64, device="cpu"
        )
        self.slots = slots
        self.codec = codec
        self.stream = stream
        self.signs = signs
        self.head_orders = head_orders
        self.stages = stages
        self.identity_orders: set[str] = set()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._cache_parts: dict[tuple[str, KVPart], torch.Tensor] = {}
        self._cpp_metas: dict[tuple[int, int], torch.Tensor] | None = None
        self.logger = init_logger(__name__)
        if cpp_backend.enabled():
            if any(part.dtype is not torch.bfloat16 for part in layout.layer_parts):
                raise ValueError(
                    "The CUDA Mooncake backend only supports BF16 KV caches."
                )
            if any(
                section.axis == "tensor"
                for part in layout.layer_parts
                for section in part.sections
            ):
                raise ValueError(
                    "The CUDA Mooncake backend does not support tensor-axis "
                    "quantization yet; use the Python backend."
                )

    def _use_cpp(self, logical_ids_by_group: list[list[int]]) -> bool:
        return (
            cpp_backend.enabled()
            and "aggregate" not in self.stages
            and len(self.layout.fa_group_indices) == 1
        )

    def _cpp_meta(self, slot_id: int, kv_part: int) -> torch.Tensor:
        if self._cpp_metas is None:
            self._cpp_metas = {}
        key = (slot_id, kv_part)
        meta = self._cpp_metas.get(key)
        if meta is None:
            meta = cpp_backend.build_meta(self, self.slots[slot_id], kv_part)
            self._cpp_metas[key] = meta
        return meta

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        value_sizes = {part.layer_name: part.head_size for part in self.layout.parts(1)}
        for part in self.layout.parts(0):
            cache = kv_caches[part.layer_name]
            expected = (
                part.num_heads,
                part.block_size // self.blocks_per_logical,
                part.head_size + value_sizes[part.layer_name],
            )
            if (
                cache.dtype is not part.dtype
                or cache.ndim != 4
                or tuple(cache.shape[1:]) != expected
                or cache.stride(-1) != 1
            ):
                raise ValueError(
                    f"Unsupported compression cache for {part.layer_name}: "
                    f"expected [blocks, {expected}] with dense BF16 content, "
                    f"got {tuple(cache.shape)} / {cache.dtype}."
                )
        if (
            cpp_backend.enabled()
            and "aggregate" not in self.stages
            and len(self.layout.fa_group_indices) != 1
        ):
            raise ValueError(
                "The CUDA compression backend requires one FA cache group; "
                "use the Python backend for multiple FA groups."
            )
        if cpp_backend.enabled() and "aggregate" not in self.stages:
            cpp_backend.load_kvs_compress(self.stream.device)
        self._cpp_metas = None
        key_sizes = {
            part.layer_name: part.head_size
            for part in self.layout.layer_parts
            if part.kv_part == 0
        }
        self.kv_caches = {name: kv_caches[name] for name in key_sizes}
        self._cache_parts = {
            (part.layer_name, part.kv_part): self.kv_caches[part.layer_name].narrow(
                -1,
                0 if part.kv_part == 0 else key_sizes[part.layer_name],
                part.head_size,
            )
            for part in self.layout.layer_parts
        }

    def _index_tensor(
        self, slot: CompressionSlotBuffers, logical_ids: list[int], row: int = 0
    ) -> torch.Tensor:
        physical_count = len(logical_ids) * self.blocks_per_logical
        storage = (
            slot.block_indices
            if slot.block_indices.ndim == 1
            else slot.block_indices[row]
        )
        indices = storage[:physical_count]
        logical = torch.tensor(logical_ids, dtype=torch.int64, device="cpu")
        physical = (
            logical.unsqueeze(1) * self.blocks_per_logical
            + self._physical_offsets.unsqueeze(0)
        ).flatten()
        if slot.host_indices is not None:
            host = slot.host_indices[row, :physical_count]
            host.copy_(physical)
            indices.copy_(host, non_blocking=True)
        else:
            indices.copy_(physical)
        return indices

    def _group_indices(
        self,
        slot: CompressionSlotBuffers,
        logical_ids_by_group: list[list[int]],
    ) -> dict[int, torch.Tensor]:
        """Build the block-index array once per KV group.

        Every layer in a group gathers with the exact same window block ids,
        so building and uploading the identical array once per layer is pure
        overhead.  Each group keeps its own row of ``block_indices``.
        """
        return {
            group_index: self._index_tensor(
                slot,
                logical_ids,
                self.layout.index_rows_by_group[group_index],
            )
            for group_index, logical_ids in enumerate(logical_ids_by_group)
            if logical_ids and group_index in self.layout.index_rows_by_group
        }

    def _cache_part(self, part: LayerPartLayout) -> torch.Tensor:
        return self._cache_parts[(part.layer_name, part.kv_part)]

    def _layer_views(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_part = self._cache_part(part)
        kernel_tokens = part.block_size // self.blocks_per_logical
        raw_numel = physical_blocks * kernel_tokens * part.num_heads * part.head_size
        raw_bytes = raw_numel * cache_part.element_size()
        raw_a = slot.arena_a[part.raw_offset : part.raw_offset + raw_bytes].view(
            cache_part.dtype
        )
        canonical_a = raw_a.view(
            physical_blocks, kernel_tokens, part.num_heads, part.head_size
        )
        scratch = raw_a.view(
            physical_blocks, part.num_heads, kernel_tokens, part.head_size
        )
        # Keep both raw ping-pong buffers in arena A.  Arena B is reserved for
        # the packed u8 stream and must not be touched by later layer gathers.
        canonical_b = (
            slot.arena_a[
                self.layout.scratch_stride
                + part.raw_offset : self.layout.scratch_stride
                + part.raw_offset
                + raw_bytes
            ]
            .view(cache_part.dtype)
            .view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        )
        return scratch, canonical_a, canonical_b

    def _gather_rearrange(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
    ) -> bool:
        """Gather raw blocks and move them into canonical layout.

        Returns True when the canonical data landed in the second ping-pong
        buffer (buffer B) because a head permutation was applied. Identity
        head orders land directly in the buffer the next stage consumes, so
        the transpose/reorder copy is skipped.
        """
        physical_blocks = indices.numel()
        scratch, canonical_a, canonical_b = self._layer_views(
            slot, part, physical_blocks
        )
        source = self._cache_part(part)
        if part.layer_name in self.identity_orders:
            target = canonical_b if "transformer" in self.stages else canonical_a
            torch.index_select(source, 0, indices, out=target.permute(0, 2, 1, 3))
            return target is canonical_b
        torch.index_select(source, 0, indices, out=scratch)
        source_canonical = scratch.permute(0, 2, 1, 3)
        torch.index_select(
            source_canonical,
            2,
            self.head_orders[part.layer_name],
            out=canonical_b,
        )
        return True

    def _hadamard_forward(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
    ) -> None:
        _, canonical_a, canonical_b = self._layer_views(slot, part, physical_blocks)
        canonical_b.mul_(self.signs[(part.layer_name, part.kv_part)])
        self._hadamard_transform(canonical_b, canonical_a, part.head_size)

    @staticmethod
    def _hadamard_transform(
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        block_size: int,
    ) -> None:
        """Run normalized FWHT using the two preallocated arena buffers.

        Humming's JIT runtime requires its call site to be the process main
        thread, while Mooncake serves compression requests from an asyncio
        worker thread. The buffer-ping-pong implementation keeps the same
        normalized Walsh-Hadamard transform without allocating a temporary
        CUDA tensor in the request path.
        """
        if threading.current_thread() is threading.main_thread():
            from humming.ops import process_input

            process_input(
                inputs.reshape(-1, block_size),
                hadamard_block_size=block_size,
                outputs=outputs.reshape(-1, block_size),
            )
            return

        src = inputs.reshape(-1, block_size)
        output_flat = outputs.reshape(-1, block_size)
        dst = output_flat
        step = 1
        while step < block_size:
            src_pairs = src.view(-1, block_size // (2 * step), 2, step)
            dst_pairs = dst.view(-1, block_size // (2 * step), 2, step)
            even = src_pairs[:, :, 0, :]
            odd = src_pairs[:, :, 1, :]
            torch.add(even, odd, out=dst_pairs[:, :, 0, :])
            torch.sub(even, odd, out=dst_pairs[:, :, 1, :])
            src, dst = dst, src
            step <<= 1

        if src.data_ptr() != output_flat.data_ptr():
            output_flat.copy_(src)
        output_flat.mul_(1.0 / math.sqrt(block_size))

    @staticmethod
    def _axes(section: QuantSection) -> tuple[int, ...]:
        if section.axis == "channel":
            return (0, 1)
        if section.axis == "token":
            return (2, 3)
        return (0, 1, 2, 3)

    def _aux_view(
        self,
        slot: CompressionSlotBuffers,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        temporary: bool = False,
    ) -> torch.Tensor:
        if temporary:
            offset += self.layout.aux_arena_bytes // 2
        nbytes = math.prod(shape) * 2
        return slot.aux[offset : offset + nbytes].view(dtype).view(shape)

    @staticmethod
    def _valid_rows(tensor: torch.Tensor, valid_tokens: int) -> torch.Tensor:
        capacity = tensor.shape[0] * tensor.shape[1]
        # Preserve the full-block reduction shape and its numerical behavior.
        if valid_tokens == capacity:
            return tensor
        return tensor.flatten(0, 1)[:valid_tokens].unsqueeze(1)

    def _pack_raw_tail(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
        valid_tokens: int,
        offset: int,
    ) -> int:
        tail_tokens = valid_tokens % part.block_size
        if tail_tokens == 0:
            return offset
        assert slot.raw_tail is not None
        kernel_tokens = part.block_size // self.blocks_per_logical
        tail_indices = indices[-self.blocks_per_logical :]
        scratch, _, _ = self._layer_views(slot, part, self.blocks_per_logical)
        torch.index_select(self._cache_part(part), 0, tail_indices, out=scratch)
        remaining = tail_tokens
        source_offset = 0
        destination = slot.raw_tail[offset:]
        for physical in range(self.blocks_per_logical):
            count = min(remaining, kernel_tokens)
            if count <= 0:
                break
            source = scratch[physical, :, :count, :]
            nbytes = source.numel() * source.element_size()
            destination[source_offset : source_offset + nbytes].view(
                source.dtype
            ).view_as(source).copy_(source)
            source_offset += nbytes
            remaining -= count
        return offset + source_offset

    def _apply_raw_tail(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
        valid_tokens: int,
        offset: int,
    ) -> int:
        tail_tokens = valid_tokens % part.block_size
        if tail_tokens == 0:
            return offset
        assert slot.raw_tail is not None
        kernel_tokens = part.block_size // self.blocks_per_logical
        tail_indices = indices[-self.blocks_per_logical :]
        scratch, _, _ = self._layer_views(slot, part, self.blocks_per_logical)
        # Do not inherit any unused positions from either P or D's recycled KV.
        scratch.zero_()
        remaining = tail_tokens
        source_offset = 0
        source = slot.raw_tail[offset:]
        for physical in range(self.blocks_per_logical):
            count = min(remaining, kernel_tokens)
            if count <= 0:
                break
            destination = scratch[physical, :, :count, :]
            nbytes = destination.numel() * destination.element_size()
            destination.copy_(
                source[source_offset : source_offset + nbytes]
                .view(destination.dtype)
                .view_as(destination)
            )
            source_offset += nbytes
            remaining -= count
        self._cache_part(part).index_copy_(0, tail_indices, scratch)
        return offset + source_offset

    def _quantize(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
        valid_tokens: int,
    ) -> None:
        dtype = self._cache_part(part).dtype
        kernel_tokens = part.block_size // self.blocks_per_logical
        _, canonical_a, _ = self._layer_views(slot, part, physical_blocks)
        u8_layer = slot.arena_b[
            self.layout.u8_base + part.u8_offset : self.layout.u8_base
            + part.u8_offset
            + physical_blocks * kernel_tokens * part.num_heads * part.head_size
        ].view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        valid_a = self._valid_rows(canonical_a, valid_tokens)
        valid_u8 = self._valid_rows(u8_layer, valid_tokens)
        for section in part.sections:
            source = valid_a.narrow(2, section.head_start, section.head_count)
            output = valid_u8.narrow(2, section.head_start, section.head_count)
            aux_shape = (
                (1, 1, section.head_count, part.head_size)
                if section.axis == "channel"
                else (*valid_a.shape[:2], 1, 1)
                if section.axis == "token"
                else (1, 1, 1, 1)
            )
            if section.axis == "token":
                for offset in (section.aux_min_offset, section.aux_scale_offset):
                    self._aux_view(
                        slot, offset, (physical_blocks * kernel_tokens, 1), dtype
                    )[valid_tokens:].zero_()
            minimum = self._aux_view(slot, section.aux_min_offset, aux_shape, dtype)
            scale = self._aux_view(slot, section.aux_scale_offset, aux_shape, dtype)
            maximum = self._aux_view(
                slot,
                section.aux_min_offset,
                aux_shape,
                dtype,
                temporary=True,
            )
            inverse_scale = self._aux_view(
                slot,
                section.aux_scale_offset,
                aux_shape,
                dtype,
                temporary=True,
            )
            torch.amin(source, dim=self._axes(section), keepdim=True, out=minimum)
            torch.amax(source, dim=self._axes(section), keepdim=True, out=maximum)
            inverse_scale.copy_(maximum).sub_(minimum).clamp_(min=1e-5)
            inverse_scale.div_(section.num_levels - 1)
            scale.copy_(inverse_scale)
            inverse_scale.reciprocal_()
            source.sub_(minimum).mul_(inverse_scale).round_().clamp_(
                0, section.num_levels - 1
            )
            output.copy_(source)

    def _aggregate_segments(self, logical_ids_by_group, kv_part):
        """Native-layout payload offsets shared identically by both sides."""
        segments = []
        offset = 0
        for row, part in enumerate(self.layout.parts(kv_part)):
            logical_ids = logical_ids_by_group[part.group_index]
            blocks = len(logical_ids) * self.blocks_per_logical
            cache_part = self._cache_part(part)
            per_block = (
                cache_part.numel() // cache_part.shape[0] * cache_part.element_size()
            )
            nbytes = blocks * per_block
            segments.append((row, part, logical_ids, offset, nbytes))
            offset += nbytes
        return segments, offset

    def _aggregate_buffer(self, buffer, offset, nbytes, blocks, part):
        cache_part = self._cache_part(part)
        shape = (blocks, *cache_part.shape[1:])
        return buffer[offset : offset + nbytes].view(part.dtype).view(shape)

    def _encode_aggregate(
        self,
        slot: CompressionSlotBuffers,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
    ) -> EncodedChunk:
        segments, payload_bytes = self._aggregate_segments(
            logical_ids_by_group, kv_part
        )
        with torch.cuda.stream(self.stream):
            index_cache = self._group_indices(slot, logical_ids_by_group)
            for _row, part, _logical_ids, offset, nbytes in segments:
                indices = index_cache[part.group_index]
                out = self._aggregate_buffer(
                    slot.arena_a, offset, nbytes, indices.numel(), part
                )
                torch.index_select(self._cache_part(part), 0, indices, out=out)
            slot.completion_event.record(self.stream)
        return EncodedChunk(
            None, slot.completion_event, payload_bytes, payload_bytes, 0, 0
        )

    def _decode_aggregate(
        self,
        slot: CompressionSlotBuffers,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
    ) -> torch.cuda.Event:
        segments, _ = self._aggregate_segments(logical_ids_by_group, kv_part)
        with torch.cuda.stream(self.stream):
            index_cache = self._group_indices(slot, logical_ids_by_group)
            for _row, part, _logical_ids, offset, nbytes in segments:
                indices = index_cache[part.group_index]
                value = self._aggregate_buffer(
                    slot.arena_a, offset, nbytes, indices.numel(), part
                )
                self._cache_part(part).index_copy_(0, indices, value)
            slot.completion_event.record(self.stream)
        return slot.completion_event

    def encode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> EncodedChunk:
        slot = self.slots[slot_id]
        raw_tail_bytes = self.layout.raw_tail_bytes(kv_part, valid_tokens_by_group)
        logical_count = len(
            logical_ids_by_group[next(iter(self.layout.fa_group_indices))]
        )
        physical_count = logical_count * self.blocks_per_logical
        self.logger.debug(
            "Mooncake compression encode: slot=%d kv_part=%d logical_ids=%s "
            "physical_blocks=%d raw_bytes=%d u8_bytes=%d",
            slot_id,
            kv_part,
            [group for group in logical_ids_by_group if group],
            physical_count,
            self.layout.raw_bytes_by_part[kv_part],
            self.layout.u8_bytes_by_part[kv_part],
        )
        if "aggregate" in self.stages:
            return self._encode_aggregate(slot, logical_ids_by_group, kv_part)
        with torch.cuda.stream(self.stream):
            # Layout offsets and codec capacities are planned for the maximum
            # chunk. Clear unused tail regions so a short final chunk cannot
            # encode stale data from the previous slot owner.
            quantized = "quantizer" in self.stages
            u8_bytes = (
                self.layout.u8_bytes_by_part
                if quantized
                else self.layout.raw_bytes_by_part
            )[kv_part]
            aux_bytes = self.layout.aux_bytes_by_part[kv_part] if quantized else 0
            if not self.layout.has_body(valid_tokens_by_group):
                u8_bytes = aux_bytes = 0
            slot.arena_b[self.layout.u8_base : self.layout.u8_base + u8_bytes].zero_()
            slot.aux[:aux_bytes].zero_()
            index_cache = self._group_indices(slot, logical_ids_by_group)
            if self._use_cpp(logical_ids_by_group):
                group_index = next(iter(self.layout.fa_group_indices))
                indices = index_cache[group_index]
                valid_tokens = valid_tokens_by_group[group_index]
                cpp_backend.encode_chunk(
                    self._cpp_meta(slot_id, kv_part),
                    indices,
                    indices.numel(),
                    valid_tokens,
                    self.stream,
                )
                raw_tail_offset = raw_tail_bytes
            else:
                raw_tail_offset = 0
                for part in self.layout.parts(kv_part):
                    valid_tokens = valid_tokens_by_group[part.group_index]
                    indices = index_cache[part.group_index]
                    raw_tail_offset = self._pack_raw_tail(
                        slot, part, indices, valid_tokens, raw_tail_offset
                    )
                    body_tokens = valid_tokens - valid_tokens % part.block_size
                    if not body_tokens:
                        continue
                    body_blocks = (
                        body_tokens // part.block_size * self.blocks_per_logical
                    )
                    indices = indices[:body_blocks]
                    # Only complete logical blocks enter any compression stage.
                    gathered_in_b = self._gather_rearrange(slot, part, indices)
                    if "transformer" in self.stages:
                        self._hadamard_forward(slot, part, indices.numel())
                    elif gathered_in_b:
                        _, canonical_a, canonical_b = self._layer_views(
                            slot, part, indices.numel()
                        )
                        canonical_a.copy_(canonical_b)
                    if quantized:
                        self._quantize(slot, part, indices.numel(), body_tokens)
                    else:
                        _, canonical_a, _ = self._layer_views(
                            slot, part, indices.numel()
                        )
                        start = self.layout.u8_base + part.u8_offset * 2
                        packed = slot.arena_b[start : start + canonical_a.numel() * 2]
                        packed.copy_(canonical_a.reshape(-1).view(torch.uint8))
            if not u8_bytes or "codec" not in self.stages:
                slot.arena_a[:u8_bytes].copy_(
                    slot.arena_b[self.layout.u8_base : self.layout.u8_base + u8_bytes]
                )
                slot.completion_event.record(self.stream)
                return EncodedChunk(
                    None,
                    slot.completion_event,
                    u8_bytes,
                    u8_bytes,
                    raw_tail_bytes,
                    slot.raw_tail.data_ptr() if slot.raw_tail is not None else 0,
                )
            assert self.codec is not None
            encoded = self.codec.encode(
                slot.arena_b[self.layout.u8_base :],
                slot.arena_a,
                u8_bytes,
                slot.completion_event,
            )
            encoded.raw_tail_bytes = raw_tail_offset
            encoded.raw_tail_addr = (
                slot.raw_tail.data_ptr() if slot.raw_tail is not None else 0
            )
            return encoded

    def _dequantize(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        physical_blocks: int,
        valid_tokens: int,
    ) -> None:
        dtype = self._cache_part(part).dtype
        kernel_tokens = part.block_size // self.blocks_per_logical
        _, canonical_a, _ = self._layer_views(slot, part, physical_blocks)
        u8_layer = slot.arena_b[
            self.layout.u8_base + part.u8_offset : self.layout.u8_base
            + part.u8_offset
            + physical_blocks * kernel_tokens * part.num_heads * part.head_size
        ].view(physical_blocks, kernel_tokens, part.num_heads, part.head_size)
        valid_a = self._valid_rows(canonical_a, valid_tokens)
        valid_u8 = self._valid_rows(u8_layer, valid_tokens)
        for section in part.sections:
            source = valid_u8.narrow(2, section.head_start, section.head_count)
            output = valid_a.narrow(2, section.head_start, section.head_count)
            aux_shape = (
                (1, 1, section.head_count, part.head_size)
                if section.axis == "channel"
                else (*valid_a.shape[:2], 1, 1)
                if section.axis == "token"
                else (1, 1, 1, 1)
            )
            minimum = self._aux_view(slot, section.aux_min_offset, aux_shape, dtype)
            scale = self._aux_view(slot, section.aux_scale_offset, aux_shape, dtype)
            output.copy_(source).mul_(scale).add_(minimum)

    def _inverse_hadamard_scatter(
        self,
        slot: CompressionSlotBuffers,
        part: LayerPartLayout,
        indices: torch.Tensor,
    ) -> None:
        scratch, canonical_a, canonical_b = self._layer_views(
            slot, part, indices.numel()
        )
        if "transformer" in self.stages:
            self._hadamard_transform(canonical_a, canonical_b, part.head_size)
            canonical_b.mul_(self.signs[(part.layer_name, part.kv_part)])
        else:
            canonical_b.copy_(canonical_a)
        unordered = scratch.permute(0, 2, 1, 3)
        unordered.index_copy_(2, self.head_orders[part.layer_name], canonical_b)
        self._cache_part(part).index_copy_(0, indices, scratch)

    def decode(
        self,
        slot_id: int,
        logical_ids_by_group: list[list[int]],
        kv_part: KVPart,
        codec_bytes: int,
        u8_bytes: int,
        *,
        valid_tokens_by_group: dict[int, int],
    ) -> torch.cuda.Event:
        if "aggregate" in self.stages:
            return self._decode_aggregate(
                self.slots[slot_id],
                logical_ids_by_group,
                kv_part,
            )
        quantized = "quantizer" in self.stages
        slot = self.slots[slot_id]
        self.logger.debug(
            "Mooncake compression decode: slot=%d kv_part=%d logical_ids=%s "
            "codec_bytes=%d u8_bytes=%d",
            slot_id,
            kv_part,
            [group for group in logical_ids_by_group if group],
            codec_bytes,
            u8_bytes,
        )
        with torch.cuda.stream(self.stream):
            if u8_bytes and "codec" in self.stages:
                assert self.codec is not None
                self.codec.decode(
                    slot.arena_a,
                    slot.arena_b[self.layout.u8_base :],
                    codec_bytes,
                    u8_bytes,
                    slot.completion_event,
                )
            else:
                slot.arena_b[
                    self.layout.u8_base : self.layout.u8_base + u8_bytes
                ].copy_(slot.arena_a[:u8_bytes])
            index_cache = self._group_indices(slot, logical_ids_by_group)
            if self._use_cpp(logical_ids_by_group):
                group_index = next(iter(self.layout.fa_group_indices))
                indices = index_cache[group_index]
                valid_tokens = valid_tokens_by_group[group_index]
                cpp_backend.decode_chunk(
                    self._cpp_meta(slot_id, kv_part),
                    indices,
                    indices.numel(),
                    valid_tokens,
                    u8_bytes,
                    self.stream,
                )
            else:
                raw_tail_offset = 0
                for part in self.layout.parts(kv_part):
                    valid_tokens = valid_tokens_by_group[part.group_index]
                    indices = index_cache[part.group_index]
                    body_tokens = valid_tokens - valid_tokens % part.block_size
                    if body_tokens:
                        body_blocks = (
                            body_tokens // part.block_size * self.blocks_per_logical
                        )
                        body_indices = indices[:body_blocks]
                        if quantized:
                            self._dequantize(slot, part, body_blocks, body_tokens)
                        else:
                            _, canonical_a, _ = self._layer_views(
                                slot, part, body_blocks
                            )
                            start = self.layout.u8_base + part.u8_offset * 2
                            packed = slot.arena_b[
                                start : start + canonical_a.numel() * 2
                            ]
                            canonical_a.copy_(
                                packed.view(canonical_a.dtype).view_as(canonical_a)
                            )
                        self._inverse_hadamard_scatter(slot, part, body_indices)
                    raw_tail_offset = self._apply_raw_tail(
                        slot, part, indices, valid_tokens, raw_tail_offset
                    )
            slot.completion_event.record(self.stream)
        return slot.completion_event
