# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared owner objects for low-latency expert parallel decode.

Adapted from https://github.com/sgl-project/sglang/pull/32482.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class _SharedEPLayout:
    activation_offset: int
    scale_offset: int
    route_id_offset: int
    route_weight_offset: int
    output_offset: int
    input_signal_offset: int
    output_signal_offset: int
    total_bytes: int

    @classmethod
    def build(
        cls,
        *,
        max_tokens: int,
        hidden_size: int,
        top_k: int,
        world_size: int,
        quant_dtype: str,
    ) -> "_SharedEPLayout":
        offset = 0
        activation_offset = offset
        if quant_dtype == "nvfp4":
            offset += max_tokens * hidden_size // 2
            scale_offset = offset
            offset = scale_offset + max_tokens * (hidden_size // 16)
            # NVFP4 W2 writes canonical owner/top-k rows into its separate VMM
            # object. Only MXFP8 retains the rank-partial output slab.
            output_parts = 0
        elif quant_dtype == "mxfp8":
            offset += max_tokens * hidden_size
            scale_offset = offset
            offset = scale_offset + max_tokens * (hidden_size // 32)
            output_parts = world_size
        else:
            raise ValueError(f"Unsupported SharedEP activation format: {quant_dtype}")
        route_id_offset = _align_up(offset, torch.int32.itemsize)
        offset = route_id_offset + max_tokens * top_k * torch.int32.itemsize
        route_weight_offset = _align_up(offset, torch.float32.itemsize)
        offset = route_weight_offset + max_tokens * top_k * torch.float32.itemsize
        output_offset = _align_up(offset, torch.bfloat16.itemsize)
        offset = (
            output_offset
            + max_tokens * output_parts * hidden_size * torch.bfloat16.itemsize
        )
        input_signal_offset = _align_up(offset, torch.int32.itemsize)
        offset = input_signal_offset + world_size * torch.int32.itemsize
        output_signal_offset = _align_up(offset, torch.int32.itemsize)
        offset = output_signal_offset + world_size * torch.int32.itemsize
        return cls(
            activation_offset=activation_offset,
            scale_offset=scale_offset,
            route_id_offset=route_id_offset,
            route_weight_offset=route_weight_offset,
            output_offset=output_offset,
            input_signal_offset=input_signal_offset,
            output_signal_offset=output_signal_offset,
            total_bytes=offset,
        )

    def owner_pointer_table(
        self,
        peer_view: RankMajorPeerView,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert peer_view.global_view is not None
        base = peer_view.global_view.data_ptr()
        return torch.tensor(
            [
                base + owner * peer_view.bytes_per_rank + offset
                for owner in range(peer_view.world_size)
            ],
            dtype=torch.uint64,
            device=device,
        )


def _typed_view(
    storage: torch.Tensor,
    offset: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    byte_count = numel * dtype.itemsize
    return storage.narrow(0, offset, byte_count).view(dtype).view(shape)


@triton.jit
def _mxfp8_quantize_pack_kernel(
    source,
    source_ids,
    source_weights,
    target,
    target_scales,
    target_ids,
    target_weights,
    source_stride,
    id_stride,
    weight_stride,
    hidden_size: tl.constexpr,
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * 32 + tl.arange(0, 32)
    valid = (row < num_tokens) & (columns < hidden_size)
    values = tl.load(
        source + row * source_stride + columns,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1e-30)
    # Choose the smallest power-of-two scale that keeps the block inside the
    # finite E4M3 range. This matches FlashInfer's native MXFP8 quantizer.
    biased_scale = tl.clamp(
        tl.ceil(tl.log2(amax / 448.0)) + 127.0,
        0.0,
        254.0,
    )
    quantized = (values / tl.exp2(biased_scale - 127.0)).to(tl.float8e4nv)
    tl.store(target + row * hidden_size + columns, quantized, mask=valid)
    tl.store(
        target_scales + row * (hidden_size // 32) + group,
        biased_scale.to(tl.uint8),
        mask=row < num_tokens,
    )

    if group == 0:
        slots = tl.arange(0, triton.next_power_of_2(top_k))
        slot_mask = slots < top_k
        source_mask = slot_mask & (row < num_tokens)
        ids = tl.load(
            source_ids + row * id_stride + slots,
            mask=source_mask,
            other=-1,
        )
        weights = tl.load(
            source_weights + row * weight_stride + slots,
            mask=source_mask,
            other=0.0,
        )
        tl.store(target_ids + row * top_k + slots, ids, mask=slot_mask)
        tl.store(target_weights + row * top_k + slots, weights, mask=slot_mask)


@triton.jit
def _publish_routes_kernel(
    source_ids,
    source_weights,
    target_ids,
    target_weights,
    id_stride,
    weight_stride,
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, triton.next_power_of_2(top_k))
    slot_mask = slots < top_k
    source_mask = slot_mask & (row < num_tokens)
    ids = tl.load(
        source_ids + row * id_stride + slots,
        mask=source_mask,
        other=-1,
    )
    weights = tl.load(
        source_weights + row * weight_stride + slots,
        mask=source_mask,
        other=0.0,
    )
    tl.store(target_ids + row * top_k + slots, ids, mask=slot_mask)
    tl.store(target_weights + row * top_k + slots, weights, mask=slot_mask)


@triton.jit
def _gather_quantized_peer_rows_kernel(
    activation_peer_ptrs,
    scale_peer_ptrs,
    route_id_peer_ptrs,
    route_weight_peer_ptrs,
    target_activations,
    target_scales,
    target_ids,
    target_weights,
    rows_per_owner: tl.constexpr,
    activation_width: tl.constexpr,
    scale_width: tl.constexpr,
    top_k: tl.constexpr,
    block: tl.constexpr,
):
    global_row = tl.program_id(0)
    column_block = tl.program_id(1)
    owner = global_row // rows_per_owner
    local_row = global_row - owner * rows_per_owner
    columns = column_block * block + tl.arange(0, block)

    activation_ptrs = activation_peer_ptrs.to(tl.pointer_type(tl.uint64))
    activation_base = tl.load(activation_ptrs + owner).to(tl.pointer_type(tl.uint8))
    activation_mask = columns < activation_width
    activation = tl.load(
        activation_base + local_row * activation_width + columns,
        mask=activation_mask,
        other=0,
    )
    target_activation_bytes = target_activations.to(tl.pointer_type(tl.uint8))
    tl.store(
        target_activation_bytes + global_row * activation_width + columns,
        activation,
        mask=activation_mask,
    )

    scale_ptrs = scale_peer_ptrs.to(tl.pointer_type(tl.uint64))
    scale_base = tl.load(scale_ptrs + owner).to(tl.pointer_type(tl.uint8))
    scale_mask = columns < scale_width
    scale = tl.load(
        scale_base + local_row * scale_width + columns,
        mask=scale_mask,
        other=0,
    )
    target_scale_bytes = target_scales.to(tl.pointer_type(tl.uint8))
    tl.store(
        target_scale_bytes + global_row * scale_width + columns,
        scale,
        mask=scale_mask,
    )

    if column_block == 0:
        slots = tl.arange(0, triton.next_power_of_2(top_k))
        slot_mask = slots < top_k
        id_ptrs = route_id_peer_ptrs.to(tl.pointer_type(tl.uint64))
        id_base = tl.load(id_ptrs + owner).to(tl.pointer_type(tl.int32))
        weights_ptrs = route_weight_peer_ptrs.to(tl.pointer_type(tl.uint64))
        weight_base = tl.load(weights_ptrs + owner).to(tl.pointer_type(tl.float32))
        ids = tl.load(id_base + local_row * top_k + slots, mask=slot_mask)
        weights = tl.load(
            weight_base + local_row * top_k + slots,
            mask=slot_mask,
        )
        tl.store(target_ids + global_row * top_k + slots, ids, mask=slot_mask)
        tl.store(
            target_weights + global_row * top_k + slots,
            weights,
            mask=slot_mask,
        )


@triton.jit
def _store_release_epoch(addresses, epoch):
    return tl.inline_asm_elementwise(
        "atom.global.release.sys.exch.b32 $0, [$1], $2;",
        "=r,l,r",
        [addresses, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_acquire_epoch(addresses, epoch):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 value;
            .reg .pred pending;
        wait_epoch:
            ld.acquire.sys.global.u32 value, [$1];
            setp.ne.u32 pending, value, $2;
            @pending bra wait_epoch;
            mov.u32 $0, value;
        }
        """,
        "=r,l,r",
        [addresses, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _publish_epoch_kernel(
    signal_peer_ptrs,
    local_epoch,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    epoch = tl.load(local_epoch) + 1
    tl.store(local_epoch, epoch)
    ptrs = signal_peer_ptrs.to(tl.pointer_type(tl.uint64))
    lanes: tl.constexpr = triton.next_power_of_2(world_size)
    destinations = tl.arange(0, lanes)
    valid = destinations < world_size
    epochs = epoch + tl.zeros((lanes,), tl.int32)
    signals = tl.load(ptrs + destinations, mask=valid, other=0).to(
        tl.pointer_type(tl.int32)
    )
    if world_size == lanes:
        # Publish to every destination in parallel instead of issuing one
        # system-scope atomic per peer serially.
        _store_release_epoch(signals + rank, epochs)
    else:
        for destination in tl.static_range(world_size):
            fallback_signals = tl.load(ptrs + destination).to(tl.pointer_type(tl.int32))
            tl.atomic_xchg(
                fallback_signals + rank,
                epoch,
                sem="release",
                scope="sys",
            )


@triton.jit
def _wait_epoch_kernel(local_signals, local_epoch, world_size: tl.constexpr):
    epoch = tl.load(local_epoch)
    lanes: tl.constexpr = triton.next_power_of_2(world_size)
    if world_size == lanes:
        owners = tl.arange(0, lanes)
        epochs = epoch + tl.zeros((lanes,), tl.int32)
        # Every lane waits on one publisher with an acquire load.
        _wait_acquire_epoch(local_signals + owners, epochs)
    else:
        for owner in tl.static_range(world_size):
            flag = local_signals + owner
            observed = tl.atomic_cas(flag, epoch, epoch, sem="acquire", scope="sys")
            while observed != epoch:
                observed = tl.atomic_cas(
                    flag,
                    epoch,
                    epoch,
                    sem="acquire",
                    scope="sys",
                )


@triton.jit
def _scatter_partial_output_kernel(
    source,
    output_peer_ptrs,
    source_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    rows_per_owner: tl.constexpr,
    world_size: tl.constexpr,
    block: tl.constexpr,
):
    global_row = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    owner = global_row // rows_per_owner
    local_row = global_row - owner * rows_per_owner
    mask = columns < hidden_size
    values = tl.load(
        source + global_row * hidden_size + columns,
        mask=mask,
        other=0.0,
    )
    output_ptrs = output_peer_ptrs.to(tl.pointer_type(tl.uint64))
    output_base = tl.load(output_ptrs + owner).to(tl.pointer_type(tl.bfloat16))
    output_row = local_row * world_size + source_rank
    tl.store(
        output_base + output_row * hidden_size + columns,
        values,
        mask=(owner < world_size) & mask,
    )


@triton.jit
def _reduce_owner_output_kernel(
    source,
    output,
    output_stride,
    hidden_size: tl.constexpr,
    num_tokens: tl.constexpr,
    output_parts: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    mask = columns < hidden_size
    accumulator = tl.zeros([block], tl.float32)
    for slot in tl.static_range(output_parts):
        offset = (token * output_parts + slot) * hidden_size + columns
        accumulator += tl.load(source + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output + token * output_stride + columns,
        accumulator.to(tl.bfloat16),
        mask=(token < num_tokens) & mask,
    )


@dataclass
class SharedEPMemory:
    max_tokens: int
    hidden_size: int
    top_k: int
    quant_dtype: str
    world_size: int
    rank: int
    activations: torch.Tensor
    scales: torch.Tensor
    route_ids: torch.Tensor
    route_weights: torch.Tensor
    output_slots: torch.Tensor | None
    input_signals: torch.Tensor
    output_signals: torch.Tensor
    input_epoch: torch.Tensor
    output_epoch: torch.Tensor
    gathered_activations: torch.Tensor
    gathered_scales: torch.Tensor
    gathered_route_ids: torch.Tensor
    gathered_route_weights: torch.Tensor
    activation_peer_ptrs: torch.Tensor
    scale_peer_ptrs: torch.Tensor
    route_id_peer_ptrs: torch.Tensor
    route_weight_peer_ptrs: torch.Tensor
    output_peer_ptrs: torch.Tensor | None
    input_signal_peer_ptrs: torch.Tensor
    output_signal_peer_ptrs: torch.Tensor
    direct_output: torch.Tensor | None
    direct_output_local_slots: torch.Tensor | None
    direct_output_physical_rows_per_owner: int
    _peer_view: RankMajorPeerView
    _direct_output_peer_view: RankMajorPeerView | None

    @classmethod
    def create(
        cls,
        *,
        max_tokens: int,
        hidden_size: int,
        top_k: int,
        quant_dtype: str,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> "SharedEPMemory":
        if not current_platform.is_device_capability_family(100):
            raise NotImplementedError(
                "SharedEP currently requires an SM100-family CUDA device"
            )
        if quant_dtype not in ("nvfp4", "mxfp8"):
            raise ValueError(
                "SharedEP requires native NVFP4 or native MXFP8 activations"
            )
        alignment = 16 if quant_dtype == "nvfp4" else 32
        if hidden_size % alignment:
            raise ValueError(f"SharedEP hidden size must be divisible by {alignment}")
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        layout = _SharedEPLayout.build(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            world_size=world_size,
            quant_dtype=quant_dtype,
        )
        peer_view = create_rank_major_peer_view(
            (layout.total_bytes,),
            dtype=torch.uint8,
            group=group,
            device=device,
            require_native_atomics=True,
        )
        assert peer_view.local_view is not None
        local_storage = peer_view.local_view
        if quant_dtype == "nvfp4":
            activations = _typed_view(
                local_storage,
                layout.activation_offset,
                (max_tokens, hidden_size // 2),
                torch.uint8,
            )
            scales = _typed_view(
                local_storage,
                layout.scale_offset,
                (max_tokens, hidden_size // 16),
                torch.float8_e4m3fn,
            )
        else:
            activations = _typed_view(
                local_storage,
                layout.activation_offset,
                (max_tokens, hidden_size),
                torch.float8_e4m3fn,
            )
            scales = _typed_view(
                local_storage,
                layout.scale_offset,
                (max_tokens, hidden_size // 32),
                torch.uint8,
            )
        route_ids = _typed_view(
            local_storage,
            layout.route_id_offset,
            (max_tokens, top_k),
            torch.int32,
        )
        route_weights = _typed_view(
            local_storage,
            layout.route_weight_offset,
            (max_tokens, top_k),
            torch.float32,
        )
        output_slots = (
            _typed_view(
                local_storage,
                layout.output_offset,
                (max_tokens, world_size, hidden_size),
                torch.bfloat16,
            )
            if quant_dtype == "mxfp8"
            else None
        )
        input_signals = _typed_view(
            local_storage,
            layout.input_signal_offset,
            (world_size,),
            torch.int32,
        )
        output_signals = _typed_view(
            local_storage,
            layout.output_signal_offset,
            (world_size,),
            torch.int32,
        )
        input_signals.zero_()
        output_signals.zero_()
        input_epoch = torch.zeros(1, dtype=torch.int32, device=device)
        output_epoch = torch.zeros(1, dtype=torch.int32, device=device)
        global_rows = world_size * max_tokens
        activation_width = hidden_size // 2 if quant_dtype == "nvfp4" else hidden_size
        scale_width = hidden_size // 16 if quant_dtype == "nvfp4" else hidden_size // 32
        gathered_activations = torch.empty(
            global_rows,
            activation_width,
            dtype=(torch.uint8 if quant_dtype == "nvfp4" else torch.float8_e4m3fn),
            device=device,
        )
        gathered_scales = torch.empty(
            global_rows,
            scale_width,
            dtype=(torch.float8_e4m3fn if quant_dtype == "nvfp4" else torch.uint8),
            device=device,
        )
        gathered_route_ids = torch.empty(
            global_rows,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        gathered_route_weights = torch.empty(
            global_rows,
            top_k,
            dtype=torch.float32,
            device=device,
        )
        direct_output_peer_view: RankMajorPeerView | None = None
        direct_output: torch.Tensor | None = None
        direct_output_local_slots: torch.Tensor | None = None
        direct_output_physical_rows_per_owner = 0
        if quant_dtype == "nvfp4":
            try:
                direct_output_peer_view = create_rank_major_peer_view(
                    (max_tokens * top_k, hidden_size),
                    dtype=torch.bfloat16,
                    group=group,
                    device=device,
                )
            except Exception:
                peer_view.close()
                raise
            assert direct_output_peer_view.global_view is not None
            assert direct_output_peer_view.local_view is not None
            direct_output_physical_rows_per_owner = (
                direct_output_peer_view.rows_per_rank
            )
            # FlashInfer validates the logical token dimension but its direct
            # finalize epilogue maps each expanded token/top-k row through the
            # physical owner stride before issuing the VMM store.
            direct_output = torch.as_strided(
                direct_output_peer_view.global_view,
                size=(global_rows, hidden_size),
                stride=(hidden_size, 1),
            )
            direct_output_local_slots = direct_output_peer_view.local_view[
                : max_tokens * top_k
            ].view(max_tokens, top_k, hidden_size)
        dist.barrier(group=group)
        return cls(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            quant_dtype=quant_dtype,
            world_size=world_size,
            rank=rank,
            activations=activations,
            scales=scales,
            route_ids=route_ids,
            route_weights=route_weights,
            output_slots=output_slots,
            input_signals=input_signals,
            output_signals=output_signals,
            input_epoch=input_epoch,
            output_epoch=output_epoch,
            gathered_activations=gathered_activations,
            gathered_scales=gathered_scales,
            gathered_route_ids=gathered_route_ids,
            gathered_route_weights=gathered_route_weights,
            activation_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.activation_offset,
                device,
            ),
            scale_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.scale_offset,
                device,
            ),
            route_id_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.route_id_offset,
                device,
            ),
            route_weight_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.route_weight_offset,
                device,
            ),
            output_peer_ptrs=(
                layout.owner_pointer_table(
                    peer_view,
                    layout.output_offset,
                    device,
                )
                if quant_dtype == "mxfp8"
                else None
            ),
            input_signal_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.input_signal_offset,
                device,
            ),
            output_signal_peer_ptrs=layout.owner_pointer_table(
                peer_view,
                layout.output_signal_offset,
                device,
            ),
            direct_output=direct_output,
            direct_output_local_slots=direct_output_local_slots,
            direct_output_physical_rows_per_owner=(
                direct_output_physical_rows_per_owner
            ),
            _peer_view=peer_view,
            _direct_output_peer_view=direct_output_peer_view,
        )

    def publish_input(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        input_global_scale: torch.Tensor | None = None,
    ) -> int:
        num_tokens = hidden_states.shape[0]
        if not 0 <= num_tokens <= self.max_tokens:
            raise ValueError(
                f"SharedEP supports 0..{self.max_tokens} tokens, got {num_tokens}"
            )
        if hidden_states.shape != (num_tokens, self.hidden_size):
            raise ValueError("SharedEP hidden-state shape does not match its memory")
        if topk_ids.shape != (num_tokens, self.top_k):
            raise ValueError("SharedEP route shape does not match its memory")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("SharedEP route IDs and weights must have equal shapes")

        if self.quant_dtype == "nvfp4":
            if input_global_scale is None:
                raise ValueError("NVFP4 SharedEP requires an input global scale")
            if num_tokens:
                torch.ops._C.scaled_fp4_quant.out(
                    hidden_states,
                    input_global_scale,
                    False,
                    output=self.activations[:num_tokens],
                    output_scale=self.scales[:num_tokens].view(torch.uint8),
                )
            _publish_routes_kernel[(self.max_tokens,)](
                topk_ids,
                topk_weights,
                self.route_ids,
                self.route_weights,
                topk_ids.stride(0),
                topk_weights.stride(0),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_warps=1,
            )
        else:
            _mxfp8_quantize_pack_kernel[(self.max_tokens, self.hidden_size // 32)](
                hidden_states,
                topk_ids,
                topk_weights,
                self.activations,
                self.scales,
                self.route_ids,
                self.route_weights,
                hidden_states.stride(0),
                topk_ids.stride(0),
                topk_weights.stride(0),
                hidden_size=self.hidden_size,
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_warps=1,
            )
        self.publish_input_epoch()
        return num_tokens

    def publish_input_epoch(self) -> None:
        """Publish all owner input payloads written before this call."""
        _publish_epoch_kernel[(1,)](
            self.input_signal_peer_ptrs,
            self.input_epoch,
            rank=self.rank,
            world_size=self.world_size,
            num_warps=1,
        )

    def wait_input_epoch(self) -> None:
        """Acquire-wait until every owner has published the current input."""
        _wait_epoch_kernel[(1,)](
            self.input_signals,
            self.input_epoch,
            world_size=self.world_size,
            num_warps=1,
        )

    def gather_nvfp4_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.quant_dtype != "nvfp4":
            raise ValueError("NVFP4 gather requested from a non-NVFP4 SharedEP state")
        self.wait_input_epoch()
        global_rows = self.world_size * self.max_tokens
        _gather_quantized_peer_rows_kernel[
            (global_rows, triton.cdiv(self.hidden_size // 2, 512))
        ](
            self.activation_peer_ptrs,
            self.scale_peer_ptrs,
            self.route_id_peer_ptrs,
            self.route_weight_peer_ptrs,
            self.gathered_activations,
            self.gathered_scales,
            self.gathered_route_ids,
            self.gathered_route_weights,
            rows_per_owner=self.max_tokens,
            activation_width=self.hidden_size // 2,
            scale_width=self.hidden_size // 16,
            top_k=self.top_k,
            block=512,
            num_warps=4,
        )
        return (
            self.gathered_activations,
            self.gathered_scales,
            self.gathered_route_ids,
            self.gathered_route_weights,
        )

    def gather_mxfp8_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.quant_dtype != "mxfp8":
            raise ValueError("MXFP8 gather requested from a non-MXFP8 SharedEP state")
        self.wait_input_epoch()
        global_rows = self.world_size * self.max_tokens
        _gather_quantized_peer_rows_kernel[
            (global_rows, triton.cdiv(self.hidden_size, 256))
        ](
            self.activation_peer_ptrs,
            self.scale_peer_ptrs,
            self.route_id_peer_ptrs,
            self.route_weight_peer_ptrs,
            self.gathered_activations,
            self.gathered_scales,
            self.gathered_route_ids,
            self.gathered_route_weights,
            rows_per_owner=self.max_tokens,
            activation_width=self.hidden_size,
            scale_width=self.hidden_size // 32,
            top_k=self.top_k,
            block=256,
            num_warps=4,
        )
        return (
            self.gathered_activations,
            self.gathered_scales,
            self.gathered_route_ids,
            self.gathered_route_weights,
        )

    def publish_output(self) -> None:
        _publish_epoch_kernel[(1,)](
            self.output_signal_peer_ptrs,
            self.output_epoch,
            rank=self.rank,
            world_size=self.world_size,
            num_warps=1,
        )

    def wait_output_epoch(self) -> None:
        """Acquire-wait until every expert rank has published its output."""
        _wait_epoch_kernel[(1,)](
            self.output_signals,
            self.output_epoch,
            world_size=self.world_size,
            num_warps=1,
        )

    def publish_partial_output(
        self,
        partial_output: torch.Tensor,
    ) -> None:
        if self.output_peer_ptrs is None:
            raise RuntimeError("Rank-partial output is unavailable for NVFP4 SharedEP")
        expected = (self.world_size * self.max_tokens, self.hidden_size)
        if partial_output.shape != expected:
            raise ValueError(
                f"SharedEP partial output has shape {partial_output.shape}, "
                f"expected {expected}"
            )
        if not partial_output.is_contiguous():
            raise ValueError("SharedEP partial output must be contiguous")
        _scatter_partial_output_kernel[
            (
                partial_output.shape[0],
                triton.cdiv(self.hidden_size, 256),
            )
        ](
            partial_output,
            self.output_peer_ptrs,
            source_rank=self.rank,
            hidden_size=self.hidden_size,
            rows_per_owner=self.max_tokens,
            world_size=self.world_size,
            block=256,
            num_warps=4,
        )
        self.publish_output()

    def reduce_output(self, output: torch.Tensor, num_tokens: int) -> None:
        if self.output_slots is None:
            raise RuntimeError("Rank-partial output is unavailable for NVFP4 SharedEP")
        self.wait_output_epoch()
        if num_tokens == 0:
            return
        _reduce_owner_output_kernel[(num_tokens, triton.cdiv(self.hidden_size, 256))](
            self.output_slots,
            output,
            output.stride(0),
            hidden_size=self.hidden_size,
            num_tokens=num_tokens,
            output_parts=self.world_size,
            block=256,
            num_warps=4,
        )

    def reduce_direct_output(
        self,
        output: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Reduce this owner's canonical top-k slots after direct W2 stores."""
        if self.direct_output_local_slots is None:
            raise RuntimeError("Direct SharedEP output is unavailable")
        self.wait_output_epoch()
        if num_tokens == 0:
            return
        _reduce_owner_output_kernel[(num_tokens, triton.cdiv(self.hidden_size, 256))](
            self.direct_output_local_slots,
            output,
            output.stride(0),
            hidden_size=self.hidden_size,
            num_tokens=num_tokens,
            output_parts=self.top_k,
            block=256,
            num_warps=4,
        )

    def close(self) -> None:
        """Release both the payload and direct-output VMM mappings."""
        if self._direct_output_peer_view is not None:
            self._direct_output_peer_view.close()
        self._peer_view.close()


_SHARED_EP_MEMORY: dict[
    tuple[int, torch.device, int, int, int, str],
    SharedEPMemory,
] = {}


def get_shared_ep_memory(
    *,
    max_tokens: int,
    hidden_size: int,
    top_k: int,
    quant_dtype: str,
    group: dist.ProcessGroup,
    device: torch.device,
) -> SharedEPMemory:
    """Return the process-lifetime SharedEP state reused by every MoE layer."""

    key = (
        id(group),
        device,
        max_tokens,
        hidden_size,
        top_k,
        quant_dtype,
    )
    memory = _SHARED_EP_MEMORY.get(key)
    if memory is None:
        memory = SharedEPMemory.create(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            quant_dtype=quant_dtype,
            group=group,
            device=device,
        )
        _SHARED_EP_MEMORY[key] = memory
    return memory
