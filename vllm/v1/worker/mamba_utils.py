# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def postprocess_mamba_fused_kernel(
    # Decision inputs (per-request)
    num_accepted_tokens_ptr,
    mamba_state_idx_ptr,
    num_scheduled_tokens_ptr,
    num_computed_tokens_ptr,
    num_draft_tokens_ptr,
    # Per-group block table base addresses: int64[num_groups]. Each entry is
    # the data_ptr of that group's persistent [max_reqs, max_blocks] int32
    # block table.
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,  # stride between requests (in elements)
    # Mamba state metadata (per-layer, per-state-type)
    # These are 1D arrays indexed by (layer_idx * num_state_types + state_type_idx)
    state_base_addrs_ptr,  # base address of each state tensor
    state_block_strides_ptr,  # bytes per block for each state
    state_elem_sizes_ptr,  # element size for each state
    state_inner_sizes_ptr,  # number of elements in inner dimensions
    state_conv_widths_ptr,  # conv width for conv states (0 for temporal)
    state_group_indices_ptr,  # maps state_idx to group index in block table
    # Output: num_accepted_tokens update (for src==dst case)
    num_accepted_tokens_out_ptr,
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    num_reqs,
    # Compile-time constants (fixed after model initialization)
    # block_size: determined by model config, constant for all invocations
    block_size: tl.constexpr,
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    COPY_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types)
    - program_id(0) = request index
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.
    """
    req_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    # Bounds check
    if req_idx >= num_reqs:
        return

    # Compute decision logic (mirrors postprocess_mamba Python reference)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)
    num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_idx)
    num_draft = tl.load(num_draft_tokens_ptr + req_idx)

    num_tokens_running_state = num_computed + num_scheduled - num_draft
    new_num_computed = num_tokens_running_state + num_accepted - 1
    aligned_new_computed = (new_num_computed // block_size) * block_size

    needs_copy = aligned_new_computed >= num_tokens_running_state

    if not needs_copy:
        return

    # Compute copy parameters
    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1

    # Load state metadata for this layer/state_type
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)

    # block_table_ptrs_ptr holds one pointer per group (each group owns its own
    # block table). Reinterpret as int32* since block ids are int32.
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + req_idx * block_table_stride_req

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    src_block_id = tl.load(block_table_base + src_block_idx).to(tl.int64)
    dest_block_id = tl.load(block_table_base + dest_block_idx).to(tl.int64)

    # Compute source and destination addresses based on state type
    # conv_width > 0 means this is a conv state (get_conv_copy_spec logic)
    # conv_width == 0 means this is a temporal state (get_temporal_copy_spec logic)
    is_conv_state = conv_width > 0

    if is_conv_state:
        # Conv state: copy
        #   state[block_table[req_idx, src_block_idx],  accept_token_bias:]
        # to
        #   state[block_table[req_idx, dest_block_idx], :conv_width - accept_token_bias]
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Number of elements to copy:
        # (conv_width - accept_token_bias) * inner_size
        num_elems_to_copy = (conv_width - accept_token_bias).to(
            tl.int64
        ) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
    else:
        # Temporal state: copy
        #   state[block_table[req_idx, src_block_idx + accept_token_bias]]
        # to
        #   state[block_table[req_idx, dest_block_idx]]
        actual_src_block_idx = src_block_idx + accept_token_bias
        actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(
            tl.int64
        )
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        # Use natural block data size (inner_size * elem_size), NOT
        # state_block_stride which is the page stride and can exceed the
        # actual data when the state tensor uses as_strided page padding.
        copy_size = state_inner_size * state_elem_size

    # Mirror postprocess_mamba's trailing
    #     if src_block_idx == dest_block_idx: num_accepted_tokens_cpu[i] = 1
    # This runs whether or not the copy below is skipped (it's per-request, so
    # only state_idx == 0 writes).
    if src_block_idx == dest_block_idx and state_idx == 0:
        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    # Mirror collect_mamba_copy_meta's early return: src==dst with no token
    # bias means source and destination ranges coincide, so the copy is a
    # no-op.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(curr_src, mask=mask)
        tl.store(curr_dst, data, mask=mask)


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


@dataclasses.dataclass
class MambaSpecDecodeGPUContext:
    """
    Context for GPU-side Mamba state copy operations during the
    fused postprocess path.

    Only used when speculative decoding is enabled on a hybrid model
    (and the mamba_cache_config is in align mode).

    Precomputes memory layout metadata (base addresses, strides, element sizes)
    so the GPU kernel can perform state copies without CPU-GPU sync.

    State types are distinguished by conv_width: >0 for conv states (sliding
    window with offset-based copies), 0 for temporal states (full block copies).
    """

    # Per-state metadata tensors (shape: [num_layers * num_state_types])
    # These are populated from forward_context during the first forward pass
    state_base_addrs: torch.Tensor  # int64: base address of each state tensor
    state_block_strides: torch.Tensor  # int64: bytes per block
    state_elem_sizes: torch.Tensor  # int32: element size in bytes
    state_inner_sizes: torch.Tensor  # int64: elements in inner dimensions
    state_conv_widths: torch.Tensor  # int32: conv width (0 for temporal states)
    state_group_indices: torch.Tensor  # int32: maps state_idx to group index

    # Configuration
    block_size: int
    num_layers: int
    num_state_types: int
    mamba_group_ids: list[int]
    num_groups: int

    # Output buffer for num_accepted_tokens updates
    num_accepted_tokens_out: torch.Tensor

    # Per-group block-table base addresses: int64[num_groups]. Populated in
    # initialize_from_forward_context from the persistent per-group block
    # table tensors (whose data_ptr is stable across steps).
    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0

    # Per-request staging buffers (CPU+GPU mirrors). The runner stages
    # values into the CPU view in ``_prepare_inputs`` and the fused kernel
    # reads the GPU side. These only exist when the postprocess kernel is
    # enabled (spec decode + hybrid + align mode).
    mamba_state_idx_buf: CpuGpuBuffer | None = None
    num_scheduled_tokens_buf: CpuGpuBuffer | None = None
    num_computed_tokens_buf: CpuGpuBuffer | None = None
    num_draft_tokens_buf: CpuGpuBuffer | None = None

    # Flag to track if metadata has been populated
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        num_state_types: int,
        device: torch.device,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeGPUContext":
        """Create context with allocated buffers (metadata populated later)."""
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)

        # Count total layers across all mamba groups
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types

        return cls(
            state_base_addrs=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_block_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_inner_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_group_indices=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            block_size=mamba_spec.block_size,
            num_layers=num_layers,
            num_state_types=num_state_types,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            block_table_ptrs=torch.zeros(
                len(mamba_group_ids), dtype=torch.int64, device=device
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            is_initialized=False,
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        block_tables: list[torch.Tensor],
    ) -> None:
        """
        Extract and cache memory layout metadata from Mamba state tensors.

        This method populates the pre-allocated metadata tensors with information
        needed by `postprocess_mamba_fused_kernel` to perform state copies entirely
        on the GPU without CPU-GPU synchronization.

        For each Mamba layer and state type, the following metadata is extracted:
        - state_base_addrs: GPU memory address (data_ptr) of the state tensor
        - state_block_strides: Bytes between consecutive blocks (stride * elem_size)
        - state_elem_sizes: Element size in bytes (e.g., 2 for float16)
        - state_inner_sizes: For conv states, elements per conv position (stride(1)),
          used to compute offset when slicing state[block, offset:]. For temporal
          states, this field is unused (set to 1).
        - state_conv_widths: Conv dimension size for conv states, 0 for temporal states

        The conv vs temporal state type is detected by inspecting the copy function
        name: functions containing "conv" are treated as conv states.

        This method is idempotent - it only executes once (guarded by is_initialized
        flag) since the metadata is static after model loading.

        Args:
            kv_cache_config: Configuration containing KV cache group info and
                layer name mappings.
            forward_context: Dictionary mapping layer names to attention objects,
                populated after the model is loaded. Each attention object must
                have a `kv_cache` attribute containing the list of state tensors.
            mamba_state_copy_funcs: Tuple of copy functions (one per state type)
                used to determine whether each state is a conv or temporal state.
            block_tables: per-mamba-group persistent block-table tensors, in
                the same order as `mamba_group_ids`. Their `data_ptr()` /
                `stride(0)` are captured once for the kernel to index into.
        """
        if self.is_initialized:
            return

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache

                for state_type_idx, state in enumerate(kv_caches):
                    # Base address
                    self.state_base_addrs[idx] = state.data_ptr()

                    # Block stride (bytes between consecutive blocks)
                    # state shape: [num_blocks, ...], stride(0) = elements per block
                    if state.dim() > 1:
                        block_stride_elems = state.stride(0)
                    else:
                        block_stride_elems = state.numel()
                    self.state_block_strides[idx] = (
                        block_stride_elems * state.element_size()
                    )

                    # Element size
                    self.state_elem_sizes[idx] = state.element_size()

                    copy_func = mamba_state_copy_funcs[state_type_idx]
                    assert (
                        copy_func is get_conv_copy_spec
                        or copy_func is get_temporal_copy_spec
                    ), f"unexpected copy func: {copy_func}"
                    if copy_func is get_conv_copy_spec:
                        # Conv state: conv_width is state.size(1)
                        # inner_size is stride(1) = elements per conv position,
                        # used to compute byte offset for state[block, offset:]
                        conv_w = state.size(1) if state.dim() > 1 else 0
                        self.state_conv_widths[idx] = conv_w
                        if state.dim() > 2:
                            # stride(1) = product of dims[2:] for contiguous tensor
                            self.state_inner_sizes[idx] = state.stride(1)
                        else:
                            # 2D tensor: [num_blocks, conv_dim], no inner dims
                            self.state_inner_sizes[idx] = 1
                    else:
                        # Temporal state: inner_size = natural elements per
                        # block (prod of inner dims).  The kernel uses this
                        # to compute copy_size = inner_size * elem_size,
                        # which gives the correct byte count even when the
                        # state tensor is as_strided with padded page strides
                        # (state_block_stride would be the page size, too big).
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = (
                            state[0].numel() if state.dim() > 1 else 1
                        )

                    self.state_group_indices[idx] = group_local_idx
                    idx += 1

        # Cache per-group block-table base addresses and per-request stride.
        # `block_tables[i]` is the persistent 2D int32 block-table tensor for
        # `mamba_group_ids[i]`; `data_ptr()` / `stride(0)` are stable for the
        # engine's lifetime, so we capture them once here.
        assert len(block_tables) == self.num_groups, (
            f"expected {self.num_groups} block tables, got {len(block_tables)}"
        )
        strides = {bt.stride(0) for bt in block_tables}
        assert len(strides) == 1, (
            f"all mamba block tables must share stride(0), got {strides}"
        )
        self.block_table_stride_req = int(next(iter(strides)))
        for i, bt in enumerate(block_tables):
            self.block_table_ptrs[i] = bt.data_ptr()

        self.is_initialized = True

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
    ) -> None:
        """
        Run the fused postprocess_mamba kernel on GPU.

        This computes decisions and performs mamba state copies entirely on GPU,
        eliminating the CPU-GPU sync that was previously needed.

        Args:
            num_reqs: Number of active requests
            num_accepted_tokens_gpu: [num_reqs] accepted token counts
            mamba_state_idx_gpu: [num_reqs] source block indices
            num_scheduled_tokens_gpu: [num_reqs] scheduled token counts
            num_computed_tokens_gpu: [num_reqs] computed token counts
            num_draft_tokens_gpu: [num_reqs] draft token counts
        """
        if num_reqs == 0 or not self.is_initialized:
            return

        # Initialize output to current values (unchanged unless src==dst)
        self.num_accepted_tokens_out[:num_reqs].copy_(
            num_accepted_tokens_gpu[:num_reqs]
        )

        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states)

        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            mamba_state_idx_gpu,
            num_scheduled_tokens_gpu,
            num_computed_tokens_gpu,
            num_draft_tokens_gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.num_accepted_tokens_out,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
        )


@dataclasses.dataclass
class MambaBuffers:
    """Single owner for all mamba-specific runner buffers.

    The two sub-objects have different gates:
    ``preprocess`` is needed whenever ``mamba_cache_mode == "align"``;
    ``postprocess_align`` is needed only when align is combined with
    speculative decoding on a hybrid model, and is ``None`` otherwise.
    """

    preprocess: MambaCopyBuffers
    postprocess_align: MambaSpecDecodeGPUContext | None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
        device: torch.device,
        with_postprocess_align: bool,
    ) -> "MambaBuffers":
        return cls(
            preprocess=MambaCopyBuffers.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            postprocess_align=(
                MambaSpecDecodeGPUContext.create(
                    max_num_reqs=max_num_reqs,
                    kv_cache_config=kv_cache_config,
                    num_state_types=len(copy_funcs),
                    device=device,
                    make_buffer=make_buffer,
                )
                if with_postprocess_align
                else None
            ),
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def cleanup_mamba_state_idx(
    scheduler_output: SchedulerOutput,
    mamba_state_idx: dict[str, int],
) -> None:
    """Pop stale `mamba_state_idx` entries for finished/preempted/resumed reqs.

    Force-preempted requests (e.g., during reset_prefix_cache / KV cache
    flush) appear in resumed_req_ids without a corresponding entry in
    preempted_req_ids, leaving stale entries that can point to block
    indices beyond the new (smaller) block allocation.
    """
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def postprocess_mamba_all(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    num_spec_tokens: int,
    num_reqs: int,
):
    """All-mode postprocess (only meaningful with num_spec_tokens > 0):
    record per-request the block index of the last token scheduled this
    step, so the next step can anchor its in-place writes when accepted
    drafts leave the sequence at a non-block-aligned position.
    """
    if num_spec_tokens <= 0:
        return
    _, mamba_spec = get_mamba_groups(kv_cache_config)
    block_size = mamba_spec.block_size
    full_decode_len = 1 + num_spec_tokens
    scheduled = scheduler_output.num_scheduled_tokens
    for req_id in input_batch.req_ids[:num_reqs]:
        num_query = scheduled.get(req_id, 0)
        if num_query == full_decode_len:
            req = requests[req_id]
            seq_len = req.num_computed_tokens + num_query
            mamba_state_idx[req_id] = max(0, (seq_len - 1) // block_size)
        else:
            mamba_state_idx.pop(req_id, None)


def preprocess_mamba_all_specdec(
    scheduler_output: SchedulerOutput,
    input_batch: GPUInputBatch,
    mamba_state_idx: dict[str, int],
    num_reqs: int,
    prev_last_scheduled_idx_buf: CpuGpuBuffer,
) -> None:
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)
    np_view = prev_last_scheduled_idx_buf.np
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        np_view[i] = mamba_state_idx.get(req_id, -1)
    np_view[num_reqs:].fill(-1)
    prev_last_scheduled_idx_buf.copy_to_gpu()


def postprocess_mamba_align_gpu(
    *,
    bufs: "MambaBuffers",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    """GPU-side mamba postprocess for spec decode + hybrid + align mode.

    Lazily binds the fused-kernel context to the persistent block tables and
    forward-context state pointers on the first call, runs the fused kernel,
    and async-copies the per-request accepted-token counts back to the input
    batch's CPU tensor for the next iteration's preprocess.
    """
    ctx = bufs.postprocess_align
    # Caller is responsible for gating on spec decode + hybrid; this assert is
    # a tripwire if those gates ever drift apart.
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [
                input_batch.block_table[gid].get_device_tensor(num_reqs)
                for gid in ctx.mamba_group_ids
            ],
        )

    ctx.run_fused_postprocess(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )

    # ``num_accepted_tokens_out`` is pre-initialized from
    # ``num_accepted_tokens_gpu``; the kernel only overwrites entries to 1
    # when src_block_idx == dest_block_idx (copy within the same block), so
    # the original count is preserved for everyone else.
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
        ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True
    )


def stage_postprocess_metadata_to_gpu(
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    num_scheduled_tokens_buf: CpuGpuBuffer,
    num_computed_tokens_buf: CpuGpuBuffer,
    num_draft_tokens_buf: CpuGpuBuffer,
) -> None:
    """Stage per-request postprocess metadata into GPU buffers (non-blocking).

    Walks ``req_ids[:num_reqs]`` in batch order and writes each request's
    scheduled/computed/draft token counts into the matching pinned numpy
    views, then issues three non-blocking H→D copies. These values don't
    change between ``_prepare_inputs`` and ``_update_states_after_model_execute``.
    The fused postprocess kernel indexes the resulting GPU tensors
    by ``req_idx``.
    """
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_scheduled = scheduler_output.num_scheduled_tokens
    scheduled_np = num_scheduled_tokens_buf.np
    computed_np = num_computed_tokens_buf.np
    draft_np = num_draft_tokens_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        scheduled_np[i] = num_scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, []))
    num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    num_computed_tokens_buf.copy_to_gpu(num_reqs)
    num_draft_tokens_buf.copy_to_gpu(num_reqs)


def stage_mamba_state_idx_to_gpu(
    mamba_state_idx: dict[str, int],
    req_ids: list[str],
    num_reqs: int,
    gpu_buf: CpuGpuBuffer,
) -> None:
    """Materialize ``mamba_state_idx`` into ``gpu_buf`` and copy to GPU.

    Walks ``req_ids[:num_reqs]`` in batch order, writing each request's block
    index into the buffer's pinned numpy view, then issues a non-blocking H→D
    copy. The fused kernel indexes the resulting GPU tensor by ``req_idx``.

    Invariant: ``preprocess_mamba`` must have run first for the same batch so
    that every ``req_ids[i]`` has an entry in ``mamba_state_idx``.
    """
    np_view = gpu_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before stage_mamba_state_idx_to_gpu"
        )
        np_view[i] = state_idx
    gpu_buf.copy_to_gpu(num_reqs)


def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    """Stage all per-request inputs the fused mamba postprocess kernel reads.

    Bundles ``stage_mamba_state_idx_to_gpu`` and
    ``stage_postprocess_metadata_to_gpu`` into a single call so the runner
    has one entry point for postprocess staging. Buffers live on ``ctx``
    and only exist when the postprocess kernel is enabled.
    """
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None
    stage_mamba_state_idx_to_gpu(
        mamba_state_idx,
        req_ids,
        num_reqs,
        ctx.mamba_state_idx_buf,
    )
    stage_postprocess_metadata_to_gpu(
        scheduler_output,
        req_ids,
        num_reqs,
        requests,
        ctx.num_scheduled_tokens_buf,
        ctx.num_computed_tokens_buf,
        ctx.num_draft_tokens_buf,
    )
