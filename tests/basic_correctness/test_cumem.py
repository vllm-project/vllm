# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes

from ..utils import fork_new_process_for_each_test


@fork_new_process_for_each_test
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
        x = torch.empty(alloc_bytes, dtype=torch.uint8, device='cuda')
        tensors.append(x)
    # release the memory
    allocator.sleep()

    # allocate more memory than the total memory
    y = torch.empty(alloc_bytes, dtype=torch.uint8, device='cuda')
    tensors.append(y)
    with pytest.raises(RuntimeError):
        # when the allocator is woken up, it should raise an error
        # because we don't have enough memory
        allocator.wake_up()


@fork_new_process_for_each_test
def test_basic_cumem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device='cuda')
    x.zero_()

    # some tensors from custom memory pool
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        # custom memory pool
        y = torch.empty(shape, device='cuda')
        y.zero_()
        y += 1
        z = torch.empty(shape, device='cuda')
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


@fork_new_process_for_each_test
def test_cumem_with_cudagraph():
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool():
        weight = torch.eye(1024, device='cuda')
    with allocator.use_memory_pool(tag="discard"):
        cache = torch.empty(1024, 1024, device='cuda')

    def model(x):
        out = x @ weight
        cache[:out.size(0)].copy_(out)
        return out + 1

    x = torch.empty(128, 1024, device='cuda')

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
    assert torch.allclose(x, cache[:x.size(0)])

    # output content is as expected
    assert torch.allclose(y, x + 1)


@fork_new_process_for_each_test
@pytest.mark.parametrize(
    "model, use_v1",
    [
        # sleep mode with safetensors
        ("meta-llama/Llama-3.2-1B", True),
        # sleep mode with pytorch checkpoint
        ("facebook/opt-125m", False),
    ])
def test_end_to_end(model: str, use_v1: bool):
    import os
    os.environ["VLLM_USE_V1"] = "1" if use_v1 else "0"
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
    assert used_bytes < 2 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text

    del os.environ["VLLM_USE_V1"]
