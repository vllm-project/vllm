# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import gc

import pytest
import torch

from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test, requires_fp8

DEVICE_TYPE = current_platform.device_type


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_python_error():
    """
    Test if Python error occurs when there's low-level
    error happening from the C++ side.
    """
    allocator = CuMemAllocator.get_instance()
    total_bytes = torch.cuda.mem_get_info()[1]
    alloc_bytes = int(total_bytes * 0.7)
    tensors = []
    with allocator.use_memory_pool():
        # allocate 70% of the total memory
        x = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE_TYPE)
        tensors.append(x)
    # release the memory
    allocator.sleep()

    # allocate more memory than the total memory
    y = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE_TYPE)
    tensors.append(y)
    with pytest.raises(RuntimeError):
        # when the allocator is woken up, it should raise an error
        # because we don't have enough memory
        allocator.wake_up()


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_basic_cumem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device=DEVICE_TYPE)
    x.zero_()

    # some tensors from custom memory pool
    allocator = CuMemAllocator.get_instance()
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

    free_bytes = torch.cuda.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_cumem_with_cudagraph():
    allocator = CuMemAllocator.get_instance()
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

    free_bytes = torch.cuda.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()

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


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
@pytest.mark.parametrize(
    "model",
    [
        # sleep mode with safetensors
        "hmellor/tiny-random-LlamaForCausalLM",
        # sleep mode with pytorch checkpoint
        "facebook/opt-125m",
    ],
)
def test_end_to_end(model: str):
    free, total = torch.cuda.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
    # which is difficult to measure in the test. therefore, we only
    # test sleep level 1 here.
    llm.sleep(level=1)

    free_gpu_bytes_after_sleep, total = torch.cuda.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    # now the memory usage is mostly cudagraph memory pool,
    # and it should be less than the model weights (1B model, 2GiB weights)

    # NOTE: In V1, the memory buffer for logits (max_num_reqs x vocab_size)
    # is captured but cannot be releasesd from PyTorch due to a known bug,
    # therefore high memory usage after `llm.sleep` is called is expected.
    # FIXME(youkaichao & ywang96): Fix memory buffer issue with sleep mode
    # in V1.
    assert used_bytes < 7 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)
    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text

    llm.sleep(level=1)
    llm.wake_up(tags=["weights"])

    free_gpu_bytes_wake_up_w, total = torch.cuda.mem_get_info()
    used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline

    # should just reallocate memory for weights (1B model, ~2GiB weights)
    assert used_bytes < 10 * GiB_bytes

    # now allocate kv cache memory
    llm.wake_up(tags=["kv_cache"])
    output3 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output3[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep():
    model = "hmellor/tiny-random-LlamaForCausalLM"
    free, total = torch.cuda.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Put the engine to deep sleep
    llm.sleep(level=2)

    free_gpu_bytes_after_sleep, total = torch.cuda.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    assert used_bytes < 3 * GiB_bytes

    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    free_gpu_bytes_wake_up_w, total = torch.cuda.mem_get_info()
    used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline
    assert used_bytes < 4 * GiB_bytes

    # now allocate kv cache and cuda graph memory
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_async():
    async def test():
        model = "hmellor/tiny-random-LlamaForCausalLM"
        free, total = torch.cuda.mem_get_info()
        used_bytes_baseline = total - free  # in case other process is running
        engine_args = AsyncEngineArgs(
            model=model,
            enable_sleep_mode=True,
        )

        llm = AsyncLLMEngine.from_engine_args(engine_args)
        prompt = "How are you?"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm.generate(prompt, sampling_params, request_id="test_request_id1")
        async for output in outputs:
            pass

        # Put the engine to deep sleep
        await llm.sleep(level=2)

        await llm.wake_up(tags=["weights"])
        await llm.collective_rpc("reload_weights")
        free_gpu_bytes_wake_up_w, total = torch.cuda.mem_get_info()
        used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline
        assert used_bytes < 4 * GiB_bytes

        # now allocate kv cache and cuda graph memory
        await llm.wake_up(tags=["kv_cache"])
        outputs2 = llm.generate(prompt, sampling_params, request_id="test_request_id2")
        async for output2 in outputs2:
            pass

        # cmp output
        assert output.outputs[0].text == output2.outputs[0].text

    asyncio.run(test())


@requires_fp8
def test_deep_sleep_fp8_kvcache():
    model = "Qwen/Qwen2-0.5B"
    used_bytes_baseline = current_platform.get_current_memory_usage()

    llm = LLM(model, enable_sleep_mode=True, kv_cache_dtype="fp8")
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Put the engine to deep sleep
    llm.sleep(level=2)

    used_bytes = current_platform.get_current_memory_usage() - used_bytes_baseline

    # Rocm uses more memory for CudaGraphs, so we add 2 GiB more for the threshold
    rocm_extra_mem_bytes = 2 * GiB_bytes if current_platform.is_rocm() else 0
    mem_threshold_after_sleep = 3 * GiB_bytes + rocm_extra_mem_bytes
    assert used_bytes < mem_threshold_after_sleep

    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")

    used_bytes = current_platform.get_current_memory_usage() - used_bytes_baseline
    mem_threshold_after_wake_up = 4 * GiB_bytes + rocm_extra_mem_bytes
    assert used_bytes < mem_threshold_after_wake_up

    # now allocate kv cache and cuda graph memory
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_sleep_level3_preserves_weights_tag():
    """
    CuMem sleep with preserve_tags_on_gpu: weights stay mapped, kv_cache discarded.
    """
    allocator = CuMemAllocator.get_instance()

    # Allocates a GPU tensor tagged as "weights" and fills it with 7
    with allocator.use_memory_pool(tag="weights"):
        w = torch.empty(4096, dtype=torch.uint8, device=DEVICE_TYPE)
        w.fill_(7)
    # Allocates a GPU tensor tagged as "kv_cache"
    with allocator.use_memory_pool(tag="kv_cache"):
        torch.empty(4096, dtype=torch.uint8, device=DEVICE_TYPE)

    # Puts to sleep mode 3, which should preserve "weights" on GPU but not "kv_cache"
    allocator.sleep(
        offload_tags=tuple(),
        preserve_tags_on_gpu=frozenset({"weights"}),
    )

    # Checking the internal state of the allocator to ensure "weights" is preserved
    # and "kv_cache" is not
    for _ptr, d in allocator.pointer_to_data.items():
        if d.tag == "weights":
            assert d.preserved_during_sleep is True
            assert d.cpu_backup_tensor is None
        elif d.tag == "kv_cache":
            assert d.preserved_during_sleep is False
            assert d.cpu_backup_tensor is None

    allocator.wake_up(None)
    # The weights tensor should contain 7, proving the weight allocation stayed
    # valid on the GPU
    assert torch.all(w == 7)
    for d in allocator.pointer_to_data.values():
        assert d.preserved_during_sleep is False


@create_new_process_for_each_test()
def test_sleep_level3_llm_same_greedy_output():
    """
    Sleep level 3: weights stay on GPU; greedy decode unchanged after wake_up.
    End-to-end test with LLM to verify sleep(3) semantics.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(
        model=model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.05,
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=8)
    output = llm.generate(prompt, sampling_params)
    text_before = output[0].outputs[0].text

    torch.accelerator.synchronize()

    # Records free GPU memory
    free_before, _ = torch.cuda.mem_get_info()
    llm.sleep(level=3)
    torch.accelerator.synchronize()
    free_after, _ = torch.cuda.mem_get_info()

    message = "sleep(3) should return some KV pool VRAM"
    assert (free_after - free_before) > 0, message

    # Proves that after wake up, the model behaves identally for
    # deterministic generation
    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)
    assert output2[0].outputs[0].text == text_before


@create_new_process_for_each_test()
def test_sleep_level3_llm_partial_wake_kv_only():
    """
    After sleep(3), wake_up(tags=['kv_cache']) is enough; greedy output unchanged.
    Tests that after sleep(3) wakeup, the output is the same with deterministic model
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(
        model=model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.05,
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=8)
    output = llm.generate(prompt, sampling_params)
    text_before = output[0].outputs[0].text

    # Check if the second generated output is the same as the first one
    llm.sleep(level=3)
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    assert output2[0].outputs[0].text == text_before


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_sleep_level3_allocator_frees_less_than_sleep1():
    """
    This tests proves that sleep 3 frees less GPU memory than sleep 1
    Sleep 3 does not free the weights from the GPU memory
    """

    # creates two large tensors, "weight" and "kv cache"
    allocator = CuMemAllocator.get_instance()
    mb = 1024 * 1024
    size_w = 64 * mb
    size_kv = 64 * mb

    with allocator.use_memory_pool(tag="weights"):
        w = torch.empty(size_w, dtype=torch.uint8, device=DEVICE_TYPE)
    with allocator.use_memory_pool(tag="kv_cache"):
        kv = torch.empty(size_kv, dtype=torch.uint8, device=DEVICE_TYPE)

    # determine the amount of memory freed during sleep(3)
    torch.accelerator.synchronize()
    free_before_s3, _ = torch.cuda.mem_get_info()
    allocator.sleep(
        offload_tags=tuple(),
        preserve_tags_on_gpu=frozenset({"weights"}),
    )
    torch.accelerator.synchronize()
    free_after_s3, _ = torch.cuda.mem_get_info()
    freed_s3 = (free_after_s3 - free_before_s3) / mb

    allocator.wake_up(None)
    torch.accelerator.synchronize()

    # determine the amount of memory freed during sleep(1)
    free_before_s1, _ = torch.cuda.mem_get_info()
    allocator.sleep(offload_tags=("weights",))
    torch.accelerator.synchronize()
    free_after_s1, _ = torch.cuda.mem_get_info()
    freed_s1 = (free_after_s1 - free_before_s1) / mb

    allocator.wake_up(None)
    torch.accelerator.synchronize()

    del w, kv
    gc.collect()

    assert freed_s3 < freed_s1, (
        f"sleep(3) should free less GPU memory than sleep(1); "
        f"got {freed_s3:.1f} MB vs {freed_s1:.1f} MB"
    )
    assert freed_s3 > size_kv / mb * 0.5, (
        f"sleep(3) should free a substantial KV-sized amount; got {freed_s3:.1f} MB"
    )
    assert freed_s3 < size_w / mb + size_kv / mb * 1.5, (
        f"sleep(3) should not free weights+KV combined; got {freed_s3:.1f} MB"
    )
