# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
import itertools
from typing import Any, Callable

import torch
import triton
import triton.language as tl

from vllm.config import CacheConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def batch_memcpy_kernel(
    src_ptrs,
    dst_ptrs,
    sizes,
    BLOCK_SIZE: tl.constexpr
):
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
    batch_memcpy_kernel[grid](
        src_ptrs,
        dst_ptrs,
        sizes,
        BLOCK_SIZE=BLOCK_SIZE
    )

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


def mamba_copy_block_for_qwen_next(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
):
    # TODO: general impl for all models
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[list[torch.Tensor]] = attention.kv_cache[0]
            conv_state, gdn_state = kv_caches
            # conv state
            conv_state_block_id = block_ids[src_block_idx]
            src_conv_state = conv_state[conv_state_block_id][accept_token_bias:]
            dest_conv_state = conv_state[dest_block_id]
            dest_conv_state[: len(src_conv_state)].copy_(src_conv_state.clone())
            # gdn state
            gdn_state_block_id = block_ids[src_block_idx + accept_token_bias]
            src_gdn_state = gdn_state[gdn_state_block_id]
            dest_gdn_state = gdn_state[dest_block_id]
            dest_gdn_state.copy_(src_gdn_state)


@dataclass
class CopySpec:
    block_idx_offset_func: Callable[[int], int]
    data_offset_func: Callable[[torch.Tensor, int], int]
    num_elements_func: Callable[[torch.Tensor, int], int]


conv_copy = CopySpec(
    block_idx_offset_func=lambda bias: 0,
    data_offset_func=lambda state, bias: bias * state.stride(0),
    num_elements_func=lambda state, bias: state.numel() - bias * state.stride(0),
)

full_copy = CopySpec(
    block_idx_offset_func=lambda bias: bias,
    data_offset_func=lambda state, bias: 0,
    num_elements_func=lambda state, bias: state.numel(),
)

def mamba_copy_block_for_qwen_next_v1(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
):
    # TODO: general impl for all models
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[list[torch.Tensor]] = attention.kv_cache[0]
            conv_state, gdn_state = kv_caches

            # conv state
            conv_state_block_id_ref = block_ids[src_block_idx]
            conv_state_block_id = block_ids[src_block_idx + conv_copy.block_idx_offset_func(accept_token_bias)]
            assert conv_state_block_id_ref == conv_state_block_id, f"{conv_state_block_id_ref} != {conv_state_block_id}"

            conv_state_block = conv_state[conv_state_block_id]
            data_offset = conv_copy.data_offset_func(conv_state_block, accept_token_bias)
            num_elements = conv_copy.num_elements_func(conv_state_block, accept_token_bias)
            src_conv_state_blk = conv_state_block.view(-1)[data_offset:data_offset + num_elements]
            dest_conv_state_blk = conv_state[dest_block_id].view(-1)[:num_elements]
            dest_conv_state_blk.copy_(src_conv_state_blk)

            src_conv_state = conv_state[conv_state_block_id][accept_token_bias:]
            dest_conv_state = conv_state[dest_block_id]
            # dest_conv_state[: len(src_conv_state)].copy_(src_conv_state.clone())
            dest_conv_state_ref = dest_conv_state[: len(src_conv_state)]
            src_conv_state_ref = conv_state[conv_state_block_id][accept_token_bias:]
            assert torch.equal(dest_conv_state_ref, src_conv_state_ref)

            # gdn state
            gdn_state_block_id_ref = block_ids[src_block_idx + accept_token_bias]
            gdn_state_block_id = block_ids[src_block_idx + full_copy.block_idx_offset_func(accept_token_bias)]
            assert gdn_state_block_id_ref == gdn_state_block_id, f"{gdn_state_block_id_ref} != {gdn_state_block_id}"

            gdn_state_block = gdn_state[gdn_state_block_id]
            data_offset = full_copy.data_offset_func(gdn_state_block, accept_token_bias)
            num_elements = full_copy.num_elements_func(gdn_state_block, accept_token_bias)
            src_gdn_state_blk = gdn_state_block.view(-1)[data_offset:data_offset + num_elements]
            dest_gdn_state_blk = gdn_state[dest_block_id].view(-1)[:num_elements]
            dest_gdn_state_blk.copy_(src_gdn_state_blk)

            src_gdn_state_ref = gdn_state[gdn_state_block_id]
            dest_gdn_state_ref = gdn_state[dest_block_id]
            # dest_gdn_state.copy_(src_gdn_state)
            assert torch.equal(dest_gdn_state_ref, src_gdn_state_ref)

def mamba_copy_block_for_qwen_next_v2(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
):
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    print('>>> [BEBUG] mamba_copy_block_for_qwen_next_v2', flush=True)
    copy_meta = []
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[list[torch.Tensor]] = attention.kv_cache[0]
            copy_specs = [conv_copy, full_copy]
            for state, copy_spec in zip(kv_caches, copy_specs):
                src_block_id = block_ids[src_block_idx + copy_spec.block_idx_offset_func(accept_token_bias)]
                data_offset = copy_spec.data_offset_func(state[0], accept_token_bias)
                num_elements = copy_spec.num_elements_func(state[0], accept_token_bias)
                copy_meta.append((state[src_block_id], state[dest_block_id], data_offset, num_elements))

    for src_state, dest_state, data_offset, num_elements in copy_meta:
        src_state_data = src_state.view(-1)[data_offset:data_offset + num_elements]
        dest_state_data = dest_state.view(-1)[:num_elements]
        dest_state_data.copy_(src_state_data)
            
def mamba_copy_block_for_qwen_next_v3(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
):
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    # print('>>> [BEBUG] mamba_copy_block_for_qwen_next_v3', flush=True)
    
    src_state_list = []
    dest_state_list = []
    # data_offset_list = []
    num_elements_list = []
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[list[torch.Tensor]] = attention.kv_cache[0]
            copy_specs = [conv_copy, full_copy]
            for state, copy_spec in zip(kv_caches, copy_specs):
                src_block_id = block_ids[src_block_idx + copy_spec.block_idx_offset_func(accept_token_bias)]
                data_offset = copy_spec.data_offset_func(state[0], accept_token_bias)
                num_elements = copy_spec.num_elements_func(state[0], accept_token_bias)
                src_state_list.append(state[src_block_id].data_ptr() + data_offset * state.element_size())
                dest_state_list.append(state[dest_block_id].data_ptr())
                # data_offset_list.append(data_offset)
                num_elements_list.append(num_elements * state.element_size())

    src_state_ptrs = torch.tensor(src_state_list, device='cuda', dtype=torch.int64)
    dst_state_ptrs = torch.tensor(dest_state_list, device='cuda', dtype=torch.int64)
    # data_offsets = torch.tensor(data_offset_list, device='cuda', dtype=torch.int32)
    num_elements = torch.tensor(num_elements_list, device='cuda', dtype=torch.int32)

    batch_memcpy(src_state_ptrs, dst_state_ptrs, num_elements)


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids):
        mamba_state_idx.pop(req_id, None)
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_blocks = len(req_state.block_ids[mamba_group_ids[0]])

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
            mamba_copy_block_for_qwen_next_v3(
                kv_cache_config,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    # NOTE: can be optimized as this function always returns the same result
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    # TODO: vectorize this loop
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            mamba_copy_block_for_qwen_next_v3(
                kv_cache_config,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
