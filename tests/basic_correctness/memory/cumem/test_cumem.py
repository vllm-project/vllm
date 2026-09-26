# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.device_allocator.cumem as cumem
from vllm import LLM, SamplingParams
from vllm.device_allocator import get_mem_allocator_instance
from vllm.platforms import current_platform

from ....utils import create_new_process_for_each_test

DEVICE_TYPE = current_platform.device_type


def mapped_usage(allocator) -> int:
    """Bytes the allocator still has mapped on the device.

    `torch.accelerator.get_memory_info()` is a device-wide `cudaMemGetInfo` /
    `hipMemGetInfo` query, so its delta also carries whatever else on the GPU
    allocates or frees while we measure. The allocator's own bookkeeping is
    process-local and exact.
    """
    return sum(
        data.handle[1]
        for data in allocator.pointer_to_data.values()
        if not data.is_asleep
    )


def _wake_up_with_poisoned_mappings(allocator, byte_value: int = 0xA5) -> None:
    """Wake discarded allocations with deterministic nonzero contents."""
    original_create_and_map = cumem.create_and_map

    def create_and_map_with_poison(handle) -> None:
        original_create_and_map(handle)
        _, size, ptr, _ = handle
        cumem.libcudart.cudaMemset(ptr, byte_value, size)

    cumem.create_and_map = create_and_map_with_poison
    try:
        allocator.wake_up()
    finally:
        cumem.create_and_map = original_create_and_map


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_python_error():
    """Test if Python error occurs when there's low-level
    error happening from the C++ side.
    """
    allocator = get_mem_allocator_instance()
    total_bytes = torch.accelerator.get_memory_info()[1]
    alloc_bytes = int(total_bytes * 0.7)
    tensors = []
    with allocator.use_memory_pool():
        # allocate 70% of the total memory
        x = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE_TYPE)
        tensors.append(x)
    # release the memory
    allocator.sleep(offload_tags=())

    # allocate more memory than the total memory
    y = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE_TYPE)
    tensors.append(y)
    with pytest.raises(RuntimeError):
        # when the allocator is woken up, it should raise an error
        # because we don't have enough memory
        allocator.wake_up()


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_basic_cumem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device=DEVICE_TYPE)
    x.zero_()

    # some tensors from custom memory pool
    allocator = get_mem_allocator_instance()
    with allocator.use_memory_pool():
        # custom memory pool
        y = torch.empty(shape, device=DEVICE_TYPE)
        y.zero_()
        y += 1
        z = torch.empty(shape, device=DEVICE_TYPE)
        z.zero_()
        z += 2

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)

    # Track the allocator's own mapped bytes, not device-wide free memory.
    mapped_bytes = mapped_usage(allocator)
    assert mapped_bytes >= y.nbytes + z.nbytes
    allocator.sleep()
    assert mapped_usage(allocator) == 0
    allocator.wake_up()
    assert mapped_usage(allocator) == mapped_bytes

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)


@pytest.mark.parametrize("full_sleep", [0, 1, 2], ids=["kv-only", "sleep-1", "sleep-2"])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_release_kv_cache_memory_preserves_generation(full_sleep, monkeypatch):
    """Rejected release and KV-only/full sleep cycles preserve greedy tokens."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    kv_cache_memory_bytes = 256 * 1024 * 1024
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        enable_sleep_mode=True,
        enforce_eager=True,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.05,
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    expected = llm.generate(prompt, sampling_params)[0].outputs[0].token_ids

    def get_mapped_bytes_by_tag(worker):
        allocator = get_mem_allocator_instance()
        mapped_bytes = mapped_usage(allocator)
        kv_cache_bytes = sum(
            data.handle[1]
            for data in allocator.pointer_to_data.values()
            if data.tag == "kv_cache" and not data.is_asleep
        )
        return mapped_bytes, kv_cache_bytes

    mapped_before, kv_cache_before = llm.collective_rpc(get_mapped_bytes_by_tag)[0]
    # Utility RPC transports the engine error as a plain Exception.
    with pytest.raises(Exception, match="requires a completed pause"):
        llm.release_kv_cache_memory()
    assert llm.collective_rpc(get_mapped_bytes_by_tag)[0][0] == mapped_before
    assert not llm.llm_engine.is_sleeping()
    assert llm.generate(prompt, sampling_params)[0].outputs[0].token_ids == expected

    llm.sleep(level=0)
    llm.release_kv_cache_memory()
    mapped_after, kv_cache_after = llm.collective_rpc(get_mapped_bytes_by_tag)[0]
    assert kv_cache_before > 0
    assert kv_cache_after == 0
    assert mapped_before - mapped_after >= kv_cache_before
    assert mapped_after > 0
    assert llm.llm_engine.is_sleeping()

    if full_sleep:
        llm.sleep(level=full_sleep)
        assert llm.collective_rpc(get_mapped_bytes_by_tag)[0][0] == 0
    llm.wake_up(tags=None if full_sleep else ["kv_cache"])
    if full_sleep == 2:
        llm.collective_rpc("reload_weights")
    assert llm.collective_rpc(get_mapped_bytes_by_tag)[0][0] == mapped_before
    assert not llm.llm_engine.is_sleeping()
    actual = llm.generate(prompt, sampling_params)[0].outputs[0].token_ids
    assert actual == expected


@pytest.mark.parametrize("first_level", [1, 2])
@pytest.mark.parametrize("second_level", [1, 2])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_sleep_with_only_weights_asleep(first_level, second_level, monkeypatch):
    """Repeated sleep after KV-only wake preserves mappings and recoverability."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        enable_sleep_mode=True,
        enforce_eager=True,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.05,
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    expected = llm.generate(prompt, sampling_params)[0].outputs[0].token_ids

    def get_mapped_bytes(worker):
        return mapped_usage(get_mem_allocator_instance())

    llm.sleep(level=first_level)
    llm.wake_up(tags=["kv_cache"])
    mapped_before = llm.collective_rpc(get_mapped_bytes)[0]
    assert mapped_before > 0
    llm.sleep(level=second_level)
    assert llm.collective_rpc(get_mapped_bytes)[0] == mapped_before
    llm.wake_up(tags=["weights"])
    if first_level == 2:
        llm.collective_rpc("reload_weights")
    assert not llm.llm_engine.is_sleeping()
    actual = llm.generate(prompt, sampling_params)[0].outputs[0].token_ids
    assert actual == expected


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_discard_tags():
    """Test that discard(tags) selectively frees GPU memory for specific
    tags while keeping other tags mapped and usable."""
    allocator = get_mem_allocator_instance()

    with allocator.use_memory_pool("weights"):
        weights = torch.ones(1024, 1024, device=DEVICE_TYPE)

    with allocator.use_memory_pool("kv_cache"):
        kv = torch.ones(512, 512, device=DEVICE_TYPE)

    mapped_bytes = mapped_usage(allocator)

    # Discard kv_cache only — weights should remain valid
    allocator.discard("kv_cache")

    mapped_bytes_after_discard = mapped_usage(allocator)
    assert mapped_bytes - mapped_bytes_after_discard >= kv.nbytes
    assert mapped_bytes_after_discard >= weights.nbytes

    # Weights are still usable
    assert torch.allclose(weights, torch.ones_like(weights))

    # Wake up and verify kv_cache is remapped; discarded contents are undefined.
    allocator.wake_up()
    assert kv.shape == (512, 512)

    # Full sleep/wake cycle still works after discard
    allocator.sleep(offload_tags="weights")
    allocator.wake_up()
    assert torch.allclose(weights, torch.ones_like(weights))


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_level2_discards_ordinary_tensor_with_weights_tag():
    """Reproduce the level-2 variant for an ordinary tensor in weights."""
    allocator = get_mem_allocator_instance()

    with allocator.use_memory_pool("weights"):
        fake_weight = torch.full((4096,), 0x44, dtype=torch.uint8, device=DEVICE_TYPE)
        ordinary_tensor = torch.full(
            (4096,), 0x55, dtype=torch.uint8, device=DEVICE_TYPE
        )

    pointers = (fake_weight.data_ptr(), ordinary_tensor.data_ptr())
    allocator.sleep(offload_tags=())
    _wake_up_with_poisoned_mappings(allocator)
    torch.accelerator.synchronize()

    assert (fake_weight.data_ptr(), ordinary_tensor.data_ptr()) == pointers
    assert torch.all(fake_weight == 0xA5)
    assert torch.all(ordinary_tensor == 0xA5)


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="CUDA graph not supported on XPU")
def test_cumem_with_cudagraph():
    allocator = get_mem_allocator_instance()
    with allocator.use_memory_pool():
        weight = torch.eye(1024, device=DEVICE_TYPE)
    with allocator.use_memory_pool(tag="discard"):
        cache = torch.empty(1024, 1024, device=DEVICE_TYPE)

    def model(x):
        out = x @ weight
        cache[: out.size(0)].copy_(out)
        return out + 1

    x = torch.empty(128, 1024, device=DEVICE_TYPE)

    # warmup
    model(x)

    # capture cudagraph
    model_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(model_graph):
        y = model(x)

    mapped_bytes = mapped_usage(allocator)
    assert mapped_bytes >= weight.nbytes + cache.nbytes
    allocator.sleep()
    assert mapped_usage(allocator) == 0
    allocator.wake_up()
    assert mapped_usage(allocator) == mapped_bytes

    # after waking up, the content in the weight tensor
    # should be restored, but the content in the cache tensor
    # should be discarded

    # this operation is also compatible with cudagraph

    x.random_()
    model_graph.replay()

    # cache content is as expected
    assert torch.allclose(x, cache[: x.size(0)])

    # output content is as expected
    assert torch.allclose(y, x + 1)
