# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest
import torch

from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test, requires_fp8

DEVICE = current_platform.device_type


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_python_error():
    """
    Test if Python error occurs when there's low-level
    error happening from the C++ side.
    """
    allocator = CuMemAllocator.get_instance()
    total_bytes = current_platform.mem_get_info()[1]
    alloc_bytes = int(total_bytes * 0.7)
    tensors = []
    with allocator.use_memory_pool():
        # allocate 70% of the total memory
        x = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE)
        tensors.append(x)
    # release the memory
    allocator.sleep()

    # allocate more memory than the total memory
    y = torch.empty(alloc_bytes, dtype=torch.uint8, device=DEVICE)
    tensors.append(y)
    with pytest.raises(RuntimeError):
        # when the allocator is woken up, it should raise an error
        # because we don't have enough memory
        allocator.wake_up()


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_basic_cumem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device=DEVICE)
    x.zero_()

    # some tensors from custom memory pool
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        # custom memory pool
        y = torch.empty(shape, device=DEVICE)
        y.zero_()
        y += 1
        z = torch.empty(shape, device=DEVICE)
        z.zero_()
        z += 2

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)
    free_bytes = current_platform.mem_get_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = current_platform.mem_get_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_cumem_with_cudagraph():
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        weight = torch.eye(1024, device=DEVICE)
    with allocator.use_memory_pool(tag="discard"):
        cache = torch.empty(1024, 1024, device=DEVICE)

    def model(x):
        out = x @ weight
        cache[: out.size(0)].copy_(out)
        return out + 1

    x = torch.empty(128, 1024, device="cuda")

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
    free, total = current_platform.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
    # which is difficult to measure in the test. therefore, we only
    # test sleep level 1 here.
    llm.sleep(level=1)
    free_gpu_bytes_after_sleep, total = current_platform.mem_get_info()
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

    free_gpu_bytes_wake_up_w, total = current_platform.mem_get_info()
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
    free, total = current_platform.mem_get_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Put the engine to deep sleep
    llm.sleep(level=2)

    free_gpu_bytes_after_sleep, total = current_platform.mem_get_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    assert used_bytes < 3 * GiB_bytes

    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    free_gpu_bytes_wake_up_w, total = current_platform.mem_get_info()
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
        free, total = current_platform.mem_get_info()
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
        free_gpu_bytes_wake_up_w, total = current_platform.mem_get_info()
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
