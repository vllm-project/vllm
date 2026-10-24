# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport TP all-reduce whose consumer runs an EFC epilogue.

The publish side is FlashInfer's LL protocol as vendored for DSV4.1
(``vllm.models.deepseek_v41.nvidia.ops.cute_dsl``): a plain publish or the MoE
finalize publish multicasts each rank's BF16 contribution into a
three-generation mailbox. The consumer below is the same Lamport spin/reduce
as ``_LamportMHCDeviceKernel`` and FlashInfer's
``_LamportResidualRMSNormDeviceKernel``, with everything after the reduction
generated from an ``efc.Epilogue``. Every epilogue shares one mailbox, so any
epilogue composes with either publish.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.cute_utils.cvt import fp32x4_to_fp8x4, fp32x8_to_fp4x8
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.all_reduce_mhc import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    _group_leader_block_sum,
    _QuadFinalizePublishDeviceKernel,
    _SharedOnlyPublishDeviceKernel,
)
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    store_global_u32,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

from . import efc
from .primitives import (
    abs_f32,
    f32_bits,
    f32_to_e4m3_bits,
    load_u32x4_pred,
    round_f32_to_e4m3,
    store_u8_pred,
    store_u32_pred,
    store_u32x2_pred,
    store_u32x4_pred,
    ue8m0_ceil,
)


@cute.jit
def _cluster_row_sum(
    value: Float32,
    warp_slots: cute.Pointer,
    cluster_slots: cute.Pointer,
    warps: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
) -> Float32:
    cta = _group_leader_block_sum(value, warp_slots, warps, 1)
    tidx, _, _ = cute.arch.thread_idx()
    cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    if tidx < cluster_size:
        remote_slot = map_shared_to_peer(cluster_slots + cluster_rank, Int32(tidx))
        store_shared_cluster_f32(remote_slot, cta)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    total = Float32(0.0)
    for peer in cutlass.range_constexpr(cluster_size):
        total = total + cute.arch.load((cluster_slots + peer).llvm_ptr, Float32)
    return total


def _rmem_f32(value) -> cute.Tensor:
    tensor = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    tensor.store(value)
    return tensor


def _is_vec(item) -> bool:
    return isinstance(item, cute.TensorSSA)


class Frag:
    """A thread's share of an FP32 token row: one item per trip, either an
    8-element ``TensorSSA`` or a ``Float32`` shared by its 8 elements."""

    __slots__ = ("items",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, items) -> None:
        self.items = list(items)

    def _zip(self, other, fn) -> Frag:
        if isinstance(other, Frag):
            return Frag(fn(a, b) for a, b in zip(self.items, other.items))
        return Frag(fn(a, other) for a in self.items)

    def __add__(self, other):
        return self._zip(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._zip(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._zip(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._zip(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._zip(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._zip(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._zip(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._zip(other, lambda a, b: b / a)

    def __neg__(self):
        return Frag(-a for a in self.items)

    def __lt__(self, other):
        return self._zip(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._zip(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._zip(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._zip(other, lambda a, b: a >= b)

    def __ne__(self, other):  # type: ignore[override]
        return self._zip(other, lambda a, b: a != b)

    def __eq__(self, other):  # type: ignore[override]
        return self._zip(other, lambda a, b: a == b)


def _items(value, trips: int) -> list:
    return value.items if isinstance(value, Frag) else [value] * trips


def _map(fn, *values, trips: int) -> Frag:
    return Frag(fn(*args) for args in zip(*(_items(v, trips) for v in values)))


def _vec_map(item, scalar_fn):
    source = _rmem_f32(item)
    result = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    for i in range(VEC_BF16):
        result[i] = scalar_fn(source[i])
    return result.load()


def _f32(value) -> Float32:
    return value if isinstance(value, Float32) else Float32(value)


class _DeviceConfig(efc.Config):
    phase = efc.Phase.DEVICE

    def __init__(self, ctx: _DeviceContext, accum: list) -> None:
        self.ctx = ctx
        self.trips = len(accum)
        self.hidden = float(ctx.kernel.hidden)
        self._accum = Frag(accum)

    def accum(self):
        return self._accum

    def zeros(self):
        def zero():
            tensor = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            tensor.fill(Float32(0.0))
            return tensor.load()

        return Frag(zero() for _ in range(self.trips))

    def round(self, x, dtype):
        def one(item):
            if dtype in (torch.bfloat16, torch.float16) and _is_vec(item):
                target = BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
                return item.to(target).to(Float32)
            if dtype == torch.float8_e4m3fn:
                if _is_vec(item):
                    return _vec_map(item, round_f32_to_e4m3)
                return round_f32_to_e4m3(_f32(item))
            raise NotImplementedError(f"round to {dtype}")

        return _map(one, x, trips=self.trips)

    def abs(self, x):
        return _map(
            lambda a: _vec_map(a, abs_f32) if _is_vec(a) else abs_f32(a),
            x,
            trips=self.trips,
        )

    def _select(self, cond, x, y):
        if _is_vec(cond):
            x = x if _is_vec(x) else _f32(x)
            y = y if _is_vec(y) else _f32(y)
            return cute.where(cond, x, y)
        if _is_vec(x) or _is_vec(y):
            raise NotImplementedError("cfg.where needs a full-row condition")
        return Float32(cutlass.select_(cond, _f32(x), _f32(y)))

    def maximum(self, x, y):
        return _map(lambda a, b: self._select(a > b, a, b), x, y, trips=self.trips)

    def minimum(self, x, y):
        return _map(lambda a, b: self._select(a < b, a, b), x, y, trips=self.trips)

    def where(self, cond, x, y):
        return _map(self._select, cond, x, y, trips=self.trips)

    def _scalar(self, x, fn):
        if isinstance(x, Frag):
            return Frag(fn(_f32(a)) for a in x.items)
        return fn(_f32(x))

    def rsqrt(self, x):
        return self._scalar(x, lambda a: cute.math.rsqrt(a, fastmath=True))

    def rcp_approx(self, x):
        return self._scalar(x, cute.arch.rcp_approx)

    def ue8m0_ceil(self, x):
        return self._scalar(x, ue8m0_ceil)

    def row_sum(self, x):
        partial = Float32(0.0)
        for item in _items(x, self.trips):
            if not _is_vec(item):
                raise NotImplementedError("row reductions take full rows")
            partial = partial + item.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
        return self.ctx.row_sum(partial)

    def group_max(self, x, group_size):
        lanes = group_size // VEC_BF16

        def one(item):
            if _is_vec(item):
                item = item.reduce(
                    cute.ReductionOp.MAX,
                    init_val=Float32(float("-inf")),
                    reduction_profile=0,
                )
            offset = 1
            while offset < lanes:
                item = cute.arch.fmax(
                    item,
                    cute.arch.shuffle_sync_bfly(
                        item, offset=offset, mask=-1, mask_and_clamp=31
                    ),
                )
                offset *= 2
            return item

        return Frag(one(item) for item in _items(x, self.trips))


class _DeviceRow:
    def __init__(self, ctx: _DeviceContext, index: int, stream: int | None = None):
        self.ctx, self.index, self.stream = ctx, index, stream

    def load(self):
        return Frag(
            packed_u32x4_to_bf16x8(packed).to(Float32)
            for packed in self.ctx.packed[(self.index, self.stream)]
        )

    def store(self, value) -> None:
        self.ctx.store_row(self.index, self.stream, value)


class _DeviceStreams:
    def __init__(self, ctx: _DeviceContext, index: int) -> None:
        self.ctx, self.index = ctx, index
        self.param = ctx.epilogue.params[index]

    def __len__(self) -> int:
        return self.param.shape[0]

    def __getitem__(self, stream: int) -> _DeviceRow:
        return _DeviceRow(self.ctx, self.index, stream)


class _DevicePerToken:
    def __init__(self, ctx: _DeviceContext, index: int) -> None:
        self.ctx, self.index = ctx, index
        self.param = ctx.epilogue.params[index]
        self.shape = self.param.shape

    def __getitem__(self, idx):
        return self.ctx.scalars[self.index][efc._flat_index(self.param, idx)]


class _DeviceScalarTensor:
    def __init__(self, ctx: _DeviceContext, index: int) -> None:
        self.ctx, self.index = ctx, index

    def load(self):
        return self.ctx.scalars[self.index][0]


class _DeviceGroupScale:
    def __init__(self, ctx: _DeviceContext, index: int) -> None:
        self.ctx, self.index = ctx, index

    def store(self, value) -> None:
        self.ctx.store_group_scale(self.index, value)


class _DeviceContext:
    """Per-thread state of the consumer: fragment indices, the hoisted loads
    and the smem slots of the row reductions."""

    def __init__(
        self, kernel: _LamportEFCDeviceKernel, params, token, base_fragment, smem
    ):
        self.kernel = kernel
        self.epilogue = kernel.epilogue
        self.params = params
        self.token = token
        self.fragments = [
            base_fragment + trip * kernel.fragment_stride
            for trip in range(kernel.trips)
        ]
        self.in_range = [f < kernel.fragments for f in self.fragments]
        self.packed: dict = {}
        self.scalars: dict = {}
        self.smem = smem

    def _row_element(self, index: int, stream: int | None, trip: int):
        hidden = self.kernel.hidden
        offset = Int64(self.fragments[trip]) * VEC_BF16
        if stream is None:
            return Int64(self.token) * hidden + offset
        streams = self.epilogue.params[index].shape[0]
        return (Int64(self.token) * streams + stream) * hidden + offset

    def _address(self, index: int, element) -> Int64:
        return Int64((self.params[index].iterator + element).toint())

    def _load_row(self, index: int, stream: int | None, weight: bool = False):
        loads = []
        for trip in range(self.kernel.trips):
            if weight:
                element = Int64(self.fragments[trip]) * VEC_BF16
            else:
                element = self._row_element(index, stream, trip)
            loads.append(
                load_u32x4_pred(self._address(index, element), self.in_range[trip])
            )
        self.packed[(index, stream)] = loads

    def preload(self) -> None:
        for index, param in enumerate(self.epilogue.params):
            if not param.read:
                continue
            kind = param.kind
            if isinstance(kind, efc.Row):
                self._load_row(index, None)
            elif isinstance(kind, efc.Streams):
                for stream in sorted(param.streams_read):
                    self._load_row(index, stream)
            elif isinstance(kind, efc.Weight):
                self._load_row(index, None, weight=True)
            elif isinstance(kind, efc.PerToken):
                base = Int64(self.token) * param.numel
                self.scalars[index] = [
                    cute.arch.load(
                        (self.params[index].iterator + base + i).llvm_ptr, Float32
                    )
                    for i in range(param.numel)
                ]
            elif isinstance(kind, efc.DeviceScalar):
                self.scalars[index] = [
                    cute.arch.load(self.params[index].iterator.llvm_ptr, Float32)
                ]

    def proxies(self) -> list:
        args = []
        for index, param in enumerate(self.epilogue.params):
            kind = param.kind
            if not kind.is_tensor:
                args.append(self.params[index])
            elif isinstance(kind, efc.Row | efc.Weight):
                args.append(_DeviceRow(self, index))
            elif isinstance(kind, efc.Streams):
                args.append(_DeviceStreams(self, index))
            elif isinstance(kind, efc.PerToken):
                args.append(_DevicePerToken(self, index))
            elif isinstance(kind, efc.DeviceScalar):
                args.append(_DeviceScalarTensor(self, index))
            else:
                args.append(_DeviceGroupScale(self, index))
        return args

    def row_sum(self, value: Float32) -> Float32:
        kernel = self.kernel
        warp_slots = self.smem.allocate_array(Float32, kernel.warps)
        cluster_slots = self.smem.allocate_array(Float32, kernel.cluster_size)
        return _cluster_row_sum(
            value, warp_slots, cluster_slots, kernel.warps, kernel.cluster_size
        )

    def store_row(self, index: int, stream: int | None, value) -> None:
        dtype = self.epilogue.params[index].dtype
        for trip, item in enumerate(_items(value, self.kernel.trips)):
            if not _is_vec(item):
                raise NotImplementedError("row stores take full rows")
            element = self._row_element(index, stream, trip)
            predicate = self.in_range[trip]
            if dtype == torch.bfloat16:
                store_u32x4_pred(
                    self._address(index, element),
                    bf16x8_to_packed_u32x4(item.to(BFloat16)),
                    predicate,
                )
                continue
            values = _rmem_f32(item)
            if dtype == torch.float8_e4m3fn:
                words = [
                    fp32x4_to_fp8x4(*(values[w * 4 + i] for i in range(4)))
                    for w in range(2)
                ]
                store_u32x2_pred(
                    self._address(index, element), words[0], words[1], predicate
                )
            else:
                store_u32_pred(
                    self._address(index, element // 2),
                    fp32x8_to_fp4x8(values, 0),
                    predicate,
                )

    def store_group_scale(self, index: int, value) -> None:
        param = self.epilogue.params[index]
        kind: efc.GroupScale = param.kind  # type: ignore[assignment]
        hidden, group = self.kernel.hidden, kind.group_size
        for trip, item in enumerate(_items(value, self.kernel.trips)):
            if _is_vec(item):
                raise NotImplementedError("group scales take cfg.group_max values")
            element = self.fragments[trip] * VEC_BF16
            group_index = Int64(element // group)
            predicate = self.in_range[trip] & (element % group == 0)
            token = Int64(self.token)
            if kind.swizzled:
                k_tiles = (hidden // group + 3) // 4
                offset = (
                    ((token >> 7) * k_tiles + (group_index >> 2)) * 512
                    + (token & 31) * 16
                    + ((token >> 5) & 3) * 4
                    + (group_index & 3)
                )
            else:
                offset = token * (hidden // group) + group_index
            address = self._address(index, offset)
            item = _f32(item)
            if param.dtype == torch.float32:
                store_u32_pred(address, f32_bits(item), predicate)
            elif param.dtype == torch.float8_e4m3fn:
                store_u8_pred(address, f32_to_e4m3_bits(item), predicate)
            else:
                exponent = (f32_bits(item) >> Uint32(23)) & Uint32(0xFF)
                store_u8_pred(address, exponent, predicate)


class _LamportEFCDeviceKernel:
    """``_LamportMHCDeviceKernel``'s spin/reduce with a generated epilogue."""

    def __init__(
        self,
        *,
        epilogue: efc.Epilogue,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        enable_pdl: bool,
    ) -> None:
        if threads % WARP_SIZE or hidden % VEC_BF16:
            raise ValueError("threads must be warps and hidden a multiple of 8")
        self.epilogue = epilogue
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.fragment_stride = cluster_size * threads
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE

    def smem_size_in_bytes(self) -> int:
        return self.epilogue.num_row_reductions * (self.warps + self.cluster_size) * 4

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        params: tuple,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(contribution_mailbox, stage_state, params).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        params: tuple,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        base_fragment = cluster_rank * self.threads + tidx

        ctx = _DeviceContext(
            self, params, token, base_fragment, cutlass.utils.SmemAllocator()
        )
        ctx.preload()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        accum = []
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            rank_packed = cute.make_rmem_tensor(cute.make_layout((self.tp, 4)), Uint32)
            rank_packed.fill(Uint32(0))
            dirty = fragment < self.fragments
            while dirty:
                dirty = False
                for source_rank in cutlass.range_constexpr(self.tp):
                    if fragment < self.fragments:
                        source_element = (
                            (Int64(active_stage) * self.tp + source_rank)
                            * self.capacity_m
                            + Int64(token)
                        ) * self.hidden + Int64(fragment) * VEC_BF16
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (contribution_mailbox.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        packed = load_global_u32x4(source_pointer, volatile=True)
                        dirty = dirty | fragment_has_negative_zero(packed)
                        for word in cutlass.range_constexpr(4):
                            rank_packed[source_rank, word] = packed[word]

            rank_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            rank_sum.fill(Float32(0.0))
            for source_rank in cutlass.range_constexpr(self.tp):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = rank_packed[source_rank, word]
                rank_sum.store(
                    rank_sum.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            accum.append(rank_sum.load())

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            for source_rank in cutlass.range_constexpr(self.tp):
                if fragment < self.fragments:
                    source_element = (
                        (Int64(active_stage) * self.tp + source_rank) * self.capacity_m
                        + Int64(token)
                    ) * self.hidden + Int64(fragment) * VEC_BF16
                    store_lamport_sentinel_u32x4(
                        Int64((contribution_mailbox.iterator + source_element).toint())
                    )

        self.epilogue.fn(_DeviceConfig(ctx, accum), *ctx.proxies())


def _cluster_size(hidden: int, threads: int) -> int:
    return min(8, math.ceil(hidden // VEC_BF16 / threads))


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


class LamportAllReduce:
    """A Lamport mailbox plus its publish kernels; epilogues bind to it.

    Construction is collective over ``group``. Calls through every bound
    epilogue must be serialized on one stream, as they share the mailbox.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_num_tokens: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        top_k: int | None = None,
        threads: int = 128,
    ) -> None:
        if group is None:
            from vllm.distributed import get_tp_group

            group = get_tp_group().device_group
        self.hidden_size = hidden = hidden_size
        self.capacity = capacity = max_num_tokens
        self.top_k = top_k
        self.threads = threads
        self.tp = tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.device = device = torch.device(device)
        with torch.accelerator.device_index(device.index):
            self._publish = cute.compile(
                _SharedOnlyPublishDeviceKernel(
                    hidden=hidden,
                    tp=tp,
                    rank=rank,
                    capacity_m=capacity,
                    elements_per_thread=VEC_BF16,
                    threads=threads,
                    release_before_store=False,
                    enable_pdl=True,
                ),
                make_fake_dynamic_compact_tensor(
                    BFloat16, alignment=16, divisibility=hidden
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            self._finalize_publish = None
            if top_k is not None:
                self._finalize_publish = cute.compile(
                    _QuadFinalizePublishDeviceKernel(
                        hidden=hidden,
                        top_k=top_k,
                        tp=tp,
                        rank=rank,
                        capacity_m=capacity,
                        threads=threads,
                        routed_scaling_factor=1.0,
                        include_shared_expert=True,
                        load_shared_expert_before_pdl=False,
                        enable_pdl=True,
                        prefetch_group=top_k,
                        fp32_weights=True,
                    ),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hidden
                    ),
                    make_fake_dynamic_compact_tensor(
                        Float32, alignment=4, divisibility=top_k
                    ),
                    make_fake_dynamic_compact_tensor(
                        Int32, alignment=4, divisibility=top_k
                    ),
                    make_fake_dynamic_compact_tensor(
                        BFloat16, alignment=16, divisibility=hidden
                    ),
                    make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                    Int64(0),
                    Int32(capacity),
                    current_cu_stream(),
                )
            self.mailbox = symm_mem.empty(
                (LAMPORT_GENERATIONS, tp, capacity, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            self._mailbox_handle = symm_mem.rendezvous(self.mailbox, group)
            multicast = int(self._mailbox_handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._multicast = multicast
            self.mailbox.view(torch.int16).fill_(-32768)
            self.stage_state = torch.zeros(2, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def bind(self, fn, **examples) -> FusedAllReduce:
        """Compile ``fn`` for the dtypes and per-token shapes of ``examples``,
        one value per epilogue parameter (M may differ from later calls)."""
        return FusedAllReduce(self, fn, examples)

    def publish(self, x: torch.Tensor) -> None:
        m = x.shape[0]
        self._check_m(m)
        if x.shape != (m, self.hidden_size) or x.dtype != torch.bfloat16:
            raise ValueError("x must be BF16 [M, hidden]")
        self._publish(
            to_cute_dynamic(x.flatten(), 16, divisibility=self.hidden_size),
            to_cute(self.stage_state, 4),
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )

    def publish_finalize(
        self,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> None:
        top_k = self.top_k
        if self._finalize_publish is None or top_k is None:
            raise RuntimeError("construct with top_k to publish a MoE finalize")
        m, hidden = shared_output.shape[0], self.hidden_size
        self._check_m(m)
        self._finalize_publish(
            to_cute_dynamic(gemm2_permuted.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(expert_weights.flatten(), 4, divisibility=top_k),
            to_cute_dynamic(
                expanded_idx_to_permuted_idx.flatten(), 4, divisibility=top_k
            ),
            to_cute_dynamic(shared_output.flatten(), 16, divisibility=hidden),
            to_cute(self.stage_state, 4),
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )

    def _check_m(self, m: int) -> None:
        if not 1 <= m <= self.capacity:
            raise ValueError(f"M={m} is outside [1, {self.capacity}]")


class FusedAllReduce:
    """An epilogue compiled against a ``LamportAllReduce``."""

    def __init__(self, ar: LamportAllReduce, fn, examples: dict) -> None:
        self.ar = ar
        hidden = ar.hidden_size
        self.epilogue = epilogue = efc.Epilogue(fn, hidden, examples)
        self.kernel = _LamportEFCDeviceKernel(
            epilogue=epilogue,
            hidden=hidden,
            tp=ar.tp,
            capacity_m=ar.capacity,
            cluster_size=_cluster_size(hidden, ar.threads),
            threads=ar.threads,
            enable_pdl=True,
        )
        with torch.accelerator.device_index(ar.device.index):
            self._compiled = cute.compile(
                self.kernel,
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * ar.tp * ar.capacity * hidden,),
                    assumed_align=16,
                ),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                tuple(self._fake(p) for p in epilogue.params),
                Int32(ar.capacity),
                current_cu_stream(),
            )

    def _layout(self, param: efc.Param) -> tuple:
        """(cute dtype, alignment, divisibility) of a flattened tensor."""
        hidden, kind, dtype = self.ar.hidden_size, param.kind, param.dtype
        if isinstance(kind, efc.Row):
            if dtype == torch.bfloat16:
                return BFloat16, 16, hidden
            if dtype == torch.float8_e4m3fn:
                return Uint8, 8, hidden
            return Uint8, 4, hidden // 2
        if isinstance(kind, efc.Streams):
            return BFloat16, 16, param.shape[0] * hidden
        if isinstance(kind, efc.PerToken):
            return Float32, 4, param.numel
        assert isinstance(kind, efc.GroupScale)
        groups = 512 if kind.swizzled else hidden // kind.group_size
        if dtype == torch.float32:
            return Float32, 4, groups
        return Uint8, 1, groups

    def _fake(self, param: efc.Param):
        kind = param.kind
        if not kind.is_tensor:
            return Float32(0.0)
        if isinstance(kind, efc.Weight):
            return make_fake_compact_tensor(
                BFloat16, (self.ar.hidden_size,), assumed_align=16
            )
        if isinstance(kind, efc.DeviceScalar):
            return make_fake_compact_tensor(Float32, (1,), assumed_align=4)
        dtype, alignment, divisibility = self._layout(param)
        return make_fake_dynamic_compact_tensor(
            dtype, alignment=alignment, divisibility=divisibility
        )

    def _argument(self, param: efc.Param, value, m: int):
        kind = param.kind
        if not kind.is_tensor:
            return Float32(value)
        if value.dtype != param.dtype:
            raise ValueError(f"{param.name}: expected {param.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{param.name} must be contiguous")
        if isinstance(kind, efc.Weight):
            return to_cute(value, 16)
        if isinstance(kind, efc.DeviceScalar):
            return to_cute(value.view(1), 4)
        if isinstance(kind, efc.GroupScale) and kind.swizzled:
            groups = self.ar.hidden_size // kind.group_size
            need = -(-m // 128) * 128 * (-(-groups // 4) * 4)
            if value.numel() * value.element_size() < need:
                raise ValueError(f"{param.name} is smaller than the 128x4 layout")
            value = value.view(torch.uint8)
        elif value.shape[0] != m:
            raise ValueError(f"{param.name}: expected {m} tokens")
        _, alignment, divisibility = self._layout(param)
        return to_cute_dynamic(
            _byte_view(value).flatten(), alignment, divisibility=divisibility
        )

    def consume(self, m: int, **values) -> None:
        """Run the consumer for a publish already issued on this stream."""
        self._compiled(
            to_cute(self.ar.mailbox.flatten(), 16),
            to_cute(self.ar.stage_state, 4),
            tuple(self._argument(p, values[p.name], m) for p in self.epilogue.params),
            Int32(m),
            current_cu_stream(),
        )

    def __call__(self, x: torch.Tensor, **values) -> None:
        self.ar.publish(x)
        self.consume(x.shape[0], **values)

    def finalize(
        self,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: torch.Tensor,
        **values,
    ) -> None:
        self.ar.publish_finalize(
            gemm2_permuted, expert_weights, expanded_idx_to_permuted_idx, shared_output
        )
        self.consume(shared_output.shape[0], **values)

    def reference(self, accum: torch.Tensor, **values) -> None:
        self.epilogue.reference(accum, values)
