import psutil
import torch

from vllm import LLM, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes

from ..utils import fork_new_process_for_each_test


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
def test_end_to_end():
    llm = LLM("meta-llama/Llama-3.2-1B", enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    free_gpu_bytes = torch.cuda.mem_get_info()[0]
    print(
        f"Free GPU memory before sleep: {free_gpu_bytes / GiB_bytes:.2f} GiB")

    llm.sleep(level=1)

    free_gpu_bytes_after_sleep, total = torch.cuda.mem_get_info()
    print("Free GPU memory after sleep: "
          f"{free_gpu_bytes_after_sleep / GiB_bytes:.2f} GiB")
    used_bytes = total - free_gpu_bytes_after_sleep
    assert free_gpu_bytes_after_sleep > free_gpu_bytes
    # now the memory usage is mostly cudagraph memory pool,
    # and it should be less than the model weights
    assert used_bytes < 2 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@fork_new_process_for_each_test
def test_deep_sleep():
    llm = LLM("meta-llama/Llama-3.2-1B", enable_sleep_mode=True)

    cpu_used_bytes = psutil.virtual_memory().used
    print("CPU memory usage before sleep: "
          f"{cpu_used_bytes / GiB_bytes:.2f} GiB")

    # both model weights and kv cache are discarded
    llm.sleep(level=2)

    cpu_used_bytes_after_sleep = psutil.virtual_memory().used
    print("CPU memory usage after sleep: "
          f"{cpu_used_bytes_after_sleep / GiB_bytes:.2f} GiB")

    # the CPU memory usage should be similar to the memory usage before sleep
    assert abs(cpu_used_bytes_after_sleep - cpu_used_bytes) < 0.5 * GiB_bytes
