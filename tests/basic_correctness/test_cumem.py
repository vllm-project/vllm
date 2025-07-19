# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.utils import GiB_bytes

from ..utils import create_new_process_for_each_test


@create_new_process_for_each_test()
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


@create_new_process_for_each_test()
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


@create_new_process_for_each_test()
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


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model, use_v1",
    [
        # sleep mode with safetensors
        ("meta-llama/Llama-3.2-1B", True),
        # sleep mode with pytorch checkpoint
        ("facebook/opt-125m", False),
    ])
def test_end_to_end(monkeypatch: pytest.MonkeyPatch, model: str, use_v1: bool):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")
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
        if use_v1:
            assert used_bytes < 7 * GiB_bytes
        else:
            assert used_bytes < 2 * GiB_bytes

        llm.wake_up()
        output2 = llm.generate(prompt, sampling_params)
        # cmp output
        assert output[0].outputs[0].text == output2[0].outputs[0].text

        llm.sleep(level=1)
        llm.wake_up(tags=["weights"])

        free_gpu_bytes_wake_up_w, total = torch.cuda.mem_get_info()
        used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline

        # should just reallocate memory for weights (1B model, ~2GiB weights)
        if use_v1:
            assert used_bytes < 10 * GiB_bytes
        else:
            assert used_bytes < 6 * GiB_bytes

        # now allocate kv cache memory
        llm.wake_up(tags=["kv_cache"])
        output3 = llm.generate(prompt, sampling_params)

        # cmp output
        assert output[0].outputs[0].text == output3[0].outputs[0].text


@create_new_process_for_each_test()
def test_backup_memory_except():
    allocator = CuMemAllocator.get_instance()
    named_tensors = []
    tag = "discard"
    with allocator.use_memory_pool(tag=tag):
        x = torch.nn.Parameter(torch.randn(100, 100, device='cuda'))
        y = torch.nn.Parameter(torch.randn(200, 100, device='cuda'))
        z = torch.nn.Parameter(torch.randn(300, 100, device='cuda'))
        named_tensors.append(("x", x))
        named_tensors.append(("y", y))
    backup_named_tensors = [t.clone() for _, t in named_tensors]
    backup_z = z.clone()

    allocator.backup_memory_except(named_tensors, tags=tag)
    assert len(allocator.pointer_to_data) == 1
    ptr, data = next(iter(allocator.pointer_to_data.items()))
    assert ptr == x.data_ptr()
    size = data.handle[1]
    # torch's smallest segment size is 2MiB
    assert size == 2 * 1024 * 1024
    start = z.data_ptr() - x.data_ptr()
    for _, t in named_tensors:
        size -= t.nbytes
        start -= t.nbytes
    assert data.cpu_backup_tensor.nbytes == size
    end = start + z.nbytes
    assert (data.cpu_backup_tensor[start:end] == z.view(-1).view(
        torch.uint8).cpu()).all()

    allocator.sleep(offload_tags=tuple())
    allocator.wake_up()

    for t, (_, new_t) in zip(backup_named_tensors, named_tensors):
        assert (t != new_t).any()

    assert (backup_z == z).all()


@create_new_process_for_each_test()
def test_backup_memory_except_with_model():
    allocator = CuMemAllocator.get_instance()
    tag = "discard"

    class ToyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(12345, 100)
            self.register_buffer("x", torch.randn(22345, 100))
            self.y = torch.randn(12345, 5432)
            self.register_buffer("z", torch.randn(23333, 100))
            self.p = torch.nn.Parameter(torch.randn(31460, 100))

        def forward(self, x):
            return x

    with allocator.use_memory_pool(tag=tag):
        model = ToyModel()
        model.to('cuda')
        x = torch.randn(100, 100, device='cuda')

    backup_model = ToyModel().to('cuda')
    backup_model.load_state_dict(model.state_dict())
    backup_model.y = model.y.clone()
    backup_x = x.clone()

    def assert_equal_without_parameters():
        for b1, b2 in zip(backup_model.named_buffers(), model.named_buffers()):
            assert (b1[0] == b2[0])
            assert (b1[1] == b2[1]).all()

        assert (backup_model.y == model.y).all()

        assert (backup_x == x).all()

    # assert parameters are equal
    for b1, b2 in zip(backup_model.named_parameters(),
                      model.named_parameters()):
        assert (b1[0] == b2[0])
        assert (b1[1] == b2[1]).all()
    assert_equal_without_parameters()

    allocator.backup_memory_except(list(model.named_parameters()), tags=tag)

    allocator.sleep()
    allocator.wake_up()

    assert_equal_without_parameters()
    for p in model.parameters():
        assert (p == 0).all()
