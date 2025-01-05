import torch

from vllm.device_allocator.cumem import CuMemAllocator, CuMemMode


def test_basic_cumem():
    # some tensors from default memory pool
    shape = (1024, 1024)
    x = torch.empty(shape, device='cuda')
    x.zero_()

    # some tensors from custom memory pool
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(mode=CuMemMode.OFFLOAD):
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


def test_cumem_with_cudagraph():
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(mode=CuMemMode.OFFLOAD):
        weight = torch.eye(1024, device='cuda')
    with allocator.use_memory_pool(mode=CuMemMode.DISCARD):
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
