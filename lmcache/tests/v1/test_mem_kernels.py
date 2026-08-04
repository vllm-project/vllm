# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is not available, skipping the test",
        allow_module_level=True,
    )

# First Party
if torch.cuda.is_available():
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        lmc_ops = None
else:
    lmc_ops = None

# Mock c_ops when not available
if lmc_ops is None:

    class MockEngineKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2
        NL_X_TWO_NB_NH_BS_HS = 3
        NL_X_NB_TWO_NH_BS_HS = 4

    class MockTransferDirection:
        H2D = 0
        D2H = 1

    class MockCOps:
        EngineKVFormat = MockEngineKVFormat
        GPUKVFormat = MockEngineKVFormat
        TransferDirection = MockTransferDirection

    lmc_ops = MockCOps()

# Local
from .utils import (
    check_mem_obj_equal,
    check_paged_kv_cache_equal,
    generate_kv_cache_paged,
    generate_kv_cache_paged_list_tensors,
)


def _tuple_kv_to_blob(
    kv_tensors,
) -> torch.Tensor:
    k_temp = []
    v_temp = []
    for kv_layer in kv_tensors:
        k_temp.append(kv_layer[0])
        v_temp.append(kv_layer[1])
    k_tensor_blob = torch.stack(k_temp)
    v_tensor_blob = torch.stack(v_temp)

    # kv_tensors: [num_layer, 2, num_tok, num_kv_head, head_size]
    kv_tensors_flatten = torch.stack((k_tensor_blob, v_tensor_blob))
    kv_tensors_flatten = kv_tensors_flatten.permute([1, 0, 2, 3, 4])

    return kv_tensors_flatten


def _slice_kv_at(
    start_idx: int,
    kv_tensors: torch.Tensor,
    chunk_size: int,
) -> List[torch.Tensor]:
    return [
        x.contiguous()
        for x in list(
            torch.split(
                kv_tensors[:, :, start_idx:, ...],
                chunk_size,
                dim=2,
            )
        )
    ]


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
def test_extract_and_load_back(num_tokens):
    device = "cuda"

    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged(num_blocks, device, block_size, dtype)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Old extract
    kv_tuple_list = []
    memory_obj_old_list = []
    chunk_size = 256
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for layer_id in range(num_layers):
        key_cache = kv_cache[layer_id][0].reshape(-1, num_heads, head_size)
        value_cache = kv_cache[layer_id][1].reshape(-1, num_heads, head_size)
        kv_tuple_list.append((key_cache[slot_mapping], value_cache[slot_mapping]))
    kv_blob = _tuple_kv_to_blob(kv_tuple_list)
    kv_chunked = _slice_kv_at(0, kv_blob, chunk_size)
    for chunk_id, chunk in enumerate(kv_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, chunk.shape[2], num_heads * head_size]
        )

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        chunk = chunk.contiguous()
        for layer_id in range(num_layers):
            memory_obj_old.tensor[0, layer_id].copy_(
                chunk[layer_id, 0].reshape(-1, 1024)
            )
            memory_obj_old.tensor[1, layer_id].copy_(
                chunk[layer_id, 1].reshape(-1, 1024)
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract (zero-copy kernels)
    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(num_layers):
            lmc_ops.load_and_reshape_flash(
                memory_obj_new.tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_new_list.append(memory_obj_new)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)
    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged(num_blocks, device, block_size, dtype)

    # New load back (zero-copy kernels)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        for layer_id in range(num_layers):
            lmc_ops.reshape_and_cache_back_flash(
                memory_obj_new.tensor,
                kv_cache_new[layer_id][0],
                kv_cache_new[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
    ],
)
def test_multi_layer_kernel(num_tokens, engine_kv_format):
    device = "cuda"

    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    chunk_size = 256
    num_layers = 32
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, engine_kv_format=engine_kv_format
    )
    # old deprecated kernels only handle vllm non-MLA flash attention format
    if engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
        kv_cache_force_old = [
            kv_layer.clone().permute(1, 0, 2, 3, 4).contiguous()
            for kv_layer in kv_cache
        ]
    else:
        kv_cache_force_old = kv_cache
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(num_layers):
            lmc_ops.load_and_reshape_flash(
                memory_obj_old.tensor,
                kv_cache_force_old[layer_id][0],
                kv_cache_force_old[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            page_buffer_size,
            lmc_ops.TransferDirection.D2H,
            engine_kv_format,
            block_size,
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, engine_kv_format=engine_kv_format
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            engine_kv_format,
            block_size,
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        engine_kv_format=engine_kv_format,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("head_size", [576, 66])  # Use 68 for dsv32 (132x int8)
@pytest.mark.parametrize(
    "engine_kv_format",
    [lmc_ops.EngineKVFormat.NL_X_NB_BS_HS],  # vllm MLA
)
def test_multi_layer_kernel_use_mla(num_tokens, head_size, engine_kv_format):
    device = "cuda"

    num_blocks = 1000
    block_size = 64
    chunk_size = 256
    dtype = torch.bfloat16
    num_layers = 32
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers,
        head_size,
        engine_kv_format=engine_kv_format,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size([1, num_layers, len(slot_mapping_temp), head_size])
        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)

        for layer_id in range(num_layers):
            for token_idx, slot_idx in enumerate(slot_mapping_temp):
                slot_idx = slot_idx.item()

                block_idx = slot_idx // block_size
                block_offset = slot_idx % block_size

                memory_obj_old.tensor[0][layer_id][token_idx] = kv_cache[layer_id][
                    block_idx
                ][block_offset]

        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    memory_obj_new_list = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size([1, num_layers, len(slot_mapping_temp), head_size])

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            0,
            lmc_ops.TransferDirection.D2H,
            engine_kv_format,
            block_size,
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    for left_mem_obj, right_mem_obj in zip(
        memory_obj_old_list, memory_obj_new_list, strict=False
    ):
        left_kv, right_kv = left_mem_obj.tensor[0], right_mem_obj.tensor[0]
        right_kv = right_kv.to(left_kv.device)

        assert len(left_kv.shape) == 3
        assert len(right_kv.shape) == 3

        assert (left_kv[:, :, :] == right_kv[:, :, :]).all()

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers,
        head_size,
        engine_kv_format=engine_kv_format,
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            0,
            lmc_ops.TransferDirection.H2D,
            engine_kv_format,
            block_size,
        )

    for left_kv, right_kv in zip(kv_cache, kv_cache_new, strict=False):
        assert len(left_kv.shape) == 3
        assert len(right_kv.shape) == 3

        left_reshaped = left_kv.reshape(
            left_kv.shape[0] * left_kv.shape[1], left_kv.shape[2]
        )
        right_reshaped = right_kv.reshape(
            right_kv.shape[0] * right_kv.shape[1], right_kv.shape[2]
        )

        assert (left_reshaped[slot_mapping, :] == right_reshaped[slot_mapping, :]).all()

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
    ],
)
def test_single_layer_kernel(num_tokens, token_major, engine_kv_format):
    device = "cuda"

    num_layers = 32
    num_blocks = 1000
    block_size = 16
    num_heads = 8
    head_size = 128
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, engine_kv_format=engine_kv_format
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype, engine_kv_format=engine_kv_format
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, 2, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (2, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    for layer_id in range(num_layers):
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id],
            slot_mapping,
            lmc_ops.TransferDirection.D2H,
            engine_kv_format,
            token_major,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            lmc_ops.TransferDirection.H2D,
            engine_kv_format,
            token_major,
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        engine_kv_format=engine_kv_format,
    )


@pytest.mark.parametrize("num_tokens", [256, 500, 1024])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,  # vllm HND flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,  # vllm HND flash infer
    ],
)
def test_multi_layer_kernel_hnd(num_tokens, engine_kv_format):
    """Round-trip test for HND multi-layer kernel: D2H then H2D."""
    device = "cuda"

    num_blocks = 200
    block_size = 16
    num_heads = 8
    head_size = 128
    chunk_size = 256
    num_layers = 4
    dtype = torch.bfloat16

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_size,
        engine_kv_format=engine_kv_format,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 512 * 1024 * 1024  # 512MB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # D2H: extract from paged cache into flat memory
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    memory_obj_list = []
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for slot_mapping_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            page_buffer_size,
            lmc_ops.TransferDirection.D2H,
            engine_kv_format,
            block_size,
            head_size=head_size,
        )
        memory_obj_list.append(memory_obj)
    torch.cuda.synchronize()

    # H2D: write back into a fresh paged cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_size,
        engine_kv_format=engine_kv_format,
    )
    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            engine_kv_format,
            block_size,
            head_size=head_size,
        )
    torch.cuda.synchronize()

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        engine_kv_format=engine_kv_format,
    )
    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize(
    "engine_kv_format",
    [
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,  # vllm HND flash attention
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,  # vllm HND flash infer
    ],
)
def test_single_layer_kernel_hnd(num_tokens, token_major, engine_kv_format):
    """Round-trip test for HND single-layer kernel: D2H then H2D per layer."""
    device = "cuda"

    num_layers = 4
    num_blocks = 200
    block_size = 16
    num_heads = 8
    head_size = 128
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_size,
        engine_kv_format=engine_kv_format,
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_size,
        engine_kv_format=engine_kv_format,
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, 2, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (2, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    for layer_id in range(num_layers):
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id],
            slot_mapping,
            lmc_ops.TransferDirection.D2H,
            engine_kv_format,
            token_major,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            lmc_ops.TransferDirection.H2D,
            engine_kv_format,
            token_major,
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        engine_kv_format=engine_kv_format,
    )


def test_lmcache_memcpy_async():
    # Register 4 memory regions and try to launch copy
    chunk_size = 1 << 26
    num_chunks = 4
    dtype = torch.bfloat16
    elements_per_chunk = chunk_size // dtype.itemsize

    big_cpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cpu"
    )
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cuda"
    )

    def check_gpu_and_cpu_equal(
        gpu_tensor: torch.Tensor,
        cpu_tensor: torch.Tensor,
    ):
        cpu_tensor_from_gpu = gpu_tensor.to("cpu")
        assert torch.allclose(cpu_tensor_from_gpu, cpu_tensor, atol=1e-5)

    rt = torch.cuda.cudart()

    # Register the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr + i * chunk_size, chunk_size, 0)

    # Launch default cuda copy
    with pytest.raises(RuntimeError):
        big_gpu_tensor.copy_(big_cpu_tensor, non_blocking=True)
        torch.cuda.synchronize()

    # Launc default cuda copy for a small page
    big_gpu_tensor[: elements_per_chunk // 2].copy_(
        big_cpu_tensor[: elements_per_chunk // 2], non_blocking=True
    )
    torch.cuda.synchronize()

    check_gpu_and_cpu_equal(
        big_gpu_tensor[: elements_per_chunk // 2],
        big_cpu_tensor[: elements_per_chunk // 2],
    )

    # Positions to copy
    # - 0.25 chunk -- 0.75 chunk
    # - 0.75 chunk -- 1.25 chunk
    # - 1.75 chunk -- 3.25 chunk
    # - whole 4 chunks
    starts = [chunk_size // 4, (3 * chunk_size) // 4, (7 * chunk_size) // 4, 0]
    ends = [
        (3 * chunk_size) // 4,
        (5 * chunk_size) // 4,
        (13 * chunk_size) // 4,
        chunk_size * num_chunks,
    ]

    # H2D copy
    for start, end in zip(starts, ends, strict=False):
        # should not be equal before copy
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        lmc_ops.lmcache_memcpy_async(
            big_gpu_tensor.data_ptr() + start,
            big_cpu_tensor.data_ptr() + start,
            end - start,
            lmc_ops.TransferDirection.H2D,
            start,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Reset the data in gpu
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cuda"
    )
    # D2H copy
    for start, end in zip(starts, ends, strict=False):
        # copy from gpu to cpu
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        lmc_ops.lmcache_memcpy_async(
            big_cpu_tensor.data_ptr() + start,
            big_gpu_tensor.data_ptr() + start,
            end - start,
            lmc_ops.TransferDirection.D2H,
            start,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Unregister the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr + i * chunk_size)


def test_lmcache_memcpy_async_int8_hidden132():
    """Test lmcache_memcpy_async with int8 dtype and hidden_size=132.

    hidden_size=132 means each token occupies 132 bytes, which is NOT a
    multiple of 4 (uint32 size). This tests that the kernel correctly handles
    non-uint32-aligned data sizes when splitting copies across registered
    memory chunk boundaries.

    The KV buffer shape is (2, num_layers, num_tokens, hidden_size), so the
    total bytes = 2 * num_layers * num_tokens * 132. We pick values such that
    the copy spans across at least one PIN_CHUNK_SIZE (64 MB) boundary.
    """
    # Use PIN_CHUNK_SIZE = 64 MB as the registration granularity
    chunk_size = 1 << 26  # 64 MB
    num_chunks = 2
    dtype = torch.int8
    hidden_size = 132  # NOT a multiple of 4 — tests uint32 alignment handling

    # Total bytes must span at least one chunk boundary.
    # We allocate num_chunks * chunk_size bytes worth of int8 elements.
    total_bytes = chunk_size * num_chunks
    total_elements = total_bytes  # int8: 1 byte per element

    cpu_tensor = torch.randint(-128, 127, (total_elements,), dtype=dtype, device="cpu")
    gpu_tensor = torch.zeros(total_elements, dtype=dtype, device="cuda")

    rt = torch.cuda.cudart()

    # Register the cpu memory in PIN_CHUNK_SIZE chunks
    ptr = cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr + i * chunk_size, chunk_size, 0)

    def check_equal(gpu_t: torch.Tensor, cpu_t: torch.Tensor):
        assert torch.equal(gpu_t.cpu(), cpu_t), "GPU and CPU tensors are not equal"

    # Test cases: copy ranges that cross chunk boundaries with non-4-byte-aligned
    # sizes, simulating real KV cache transfers where hidden_size=132 (int8).
    #
    # Each "KV row" is hidden_size=132 bytes. We pick copy sizes that are
    # multiples of 132 but NOT multiples of 4, so the copy boundary falls on
    # a non-uint32-aligned offset within a registered chunk.
    #
    # num_tokens=3: 3 * 132 = 396 bytes (396 % 4 == 0, but 132 % 4 != 0)
    # num_tokens=1: 1 * 132 = 132 bytes (132 % 4 == 0)
    # num_tokens=5: 5 * 132 = 660 bytes (660 % 4 == 0)
    # Use an offset that puts the copy right across the 64MB boundary.
    boundary = chunk_size  # 64 MB boundary in bytes

    # Place the copy so it straddles the chunk boundary:
    # start = boundary - 2 * hidden_size, length = 4 * hidden_size
    # This means 2 * hidden_size bytes before the boundary and 2 * hidden_size
    # bytes after — each segment is 264 bytes, not a multiple of 4*hidden_size.
    test_cases = [
        # (start_bytes, num_hidden_rows)
        (boundary - 2 * hidden_size, 4),  # straddles boundary, 528 bytes
        (boundary - hidden_size, 3),  # straddles boundary, 396 bytes
        (boundary - 3 * hidden_size, 6),  # straddles boundary, 792 bytes
        (0, 1),  # single row at start, 132 bytes
        (total_bytes - hidden_size, 1),  # single row at end, 132 bytes
    ]

    for start_bytes, num_rows in test_cases:
        nbytes = num_rows * hidden_size
        end_bytes = start_bytes + nbytes
        assert end_bytes <= total_bytes, (
            f"Test case out of bounds: start={start_bytes}, nbytes={nbytes}"
        )

        # Reset gpu_tensor slice to zeros so we can detect the copy
        gpu_tensor[start_bytes:end_bytes].zero_()

        # H2D copy
        lmc_ops.lmcache_memcpy_async(
            gpu_tensor.data_ptr() + start_bytes,
            cpu_tensor.data_ptr() + start_bytes,
            nbytes,
            lmc_ops.TransferDirection.H2D,
            start_bytes,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_equal(
            gpu_tensor[start_bytes:end_bytes],
            cpu_tensor[start_bytes:end_bytes],
        )

    # D2H copy: write known values to GPU, copy back to CPU, verify
    gpu_src = torch.randint(-128, 127, (total_elements,), dtype=dtype, device="cuda")
    cpu_dst = torch.zeros(total_elements, dtype=dtype, device="cpu")

    # Register cpu_dst for D2H
    ptr_dst = cpu_dst.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr_dst + i * chunk_size, chunk_size, 0)

    for start_bytes, num_rows in test_cases:
        nbytes = num_rows * hidden_size
        end_bytes = start_bytes + nbytes

        cpu_dst[start_bytes:end_bytes].zero_()

        lmc_ops.lmcache_memcpy_async(
            cpu_dst.data_ptr() + start_bytes,
            gpu_src.data_ptr() + start_bytes,
            nbytes,
            lmc_ops.TransferDirection.D2H,
            start_bytes,
            chunk_size,
        )
        torch.cuda.synchronize()

        check_equal(
            gpu_src[start_bytes:end_bytes],
            cpu_dst[start_bytes:end_bytes],
        )

    # Unregister memory
    ptr = cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr + i * chunk_size)

    ptr_dst = cpu_dst.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr_dst + i * chunk_size)
