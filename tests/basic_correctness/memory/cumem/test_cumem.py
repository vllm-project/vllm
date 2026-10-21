# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

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


@pytest.mark.parametrize("tag", ["workspace", None], ids=["workspace", "default"])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_selective_wake_restores_internal_tags(tag):
    """A selective wake defers only weights/kv_cache; other tags always wake."""
    allocator = get_mem_allocator_instance()
    with allocator.use_memory_pool(tag):
        internal = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE_TYPE)
    with allocator.use_memory_pool("kv_cache"):
        kv = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE_TYPE)

    allocator.sleep(offload_tags=())
    assert mapped_usage(allocator) == 0

    allocator.wake_up(tags=["weights"])
    assert mapped_usage(allocator) == internal.nbytes
    allocator.wake_up(tags=["kv_cache"])
    assert mapped_usage(allocator) == internal.nbytes + kv.nbytes


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_workspace_scratch_discarded_on_sleep():
    """Workspace scratch lives in the sleep-mode pool: sleep discards it, any
    (selective) wake remaps it at the same address so captured graphs replay,
    and a buffer outgrown by a resize stays usable until it is released."""
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend
    from vllm.v1.worker.workspace import WorkspaceManager

    allocator = get_mem_allocator_instance()
    manager = WorkspaceManager(
        torch.device(DEVICE_TYPE),
        alloc_context=lambda: allocator.use_memory_pool("workspace"),
    )
    (small,) = manager.get_simultaneous(((32 << 20,), torch.uint8))
    # Resize while a view of the old buffer is alive: the tag is re-entered.
    (scratch,) = manager.get_simultaneous(((64 << 20,), torch.uint8))
    assert len(allocator.allocator_and_pools["workspace"]) == 2
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.add_(1)

    backend = CuMemBackend()
    backend.suspend(level=1)
    assert mapped_usage(allocator) == 0
    # Discarded, not copied.
    assert all(
        d.tag == "workspace" and d.cpu_backup_tensor is None
        for d in allocator.pointer_to_data.values()
    )

    backend.resume(tags=["kv_cache"])
    assert mapped_usage(allocator) == small.nbytes + scratch.nbytes
    scratch.zero_()
    graph.replay()
    assert int(scratch.sum()) == scratch.numel()
    small.fill_(1)

    # The next resize releases the outgrown buffer; wake does not remap it.
    del small
    (big,) = manager.get_simultaneous(((128 << 20,), torch.uint8))
    assert mapped_usage(allocator) == scratch.nbytes + big.nbytes
    backend.suspend(level=1)
    backend.resume(tags=["weights"])
    assert mapped_usage(allocator) == scratch.nbytes + big.nbytes


@pytest.mark.parametrize("free_x_early", [True, False], ids=["x-freed", "x-alive"])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_reentered_tag(free_x_early):
    """Enter a tag twice (two workspace ubatches): freeing X returns its memory
    by the next entry, without touching Y or breaking sleep/wake."""
    allocator = get_mem_allocator_instance()
    nbytes = 64 << 20
    with allocator.use_memory_pool("t"):
        xs = [torch.empty(nbytes, dtype=torch.uint8, device=DEVICE_TYPE)]
        if free_x_early:
            xs.clear()
    with allocator.use_memory_pool("t"):
        y = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE_TYPE)
    xs.clear()
    gc.collect()
    with allocator.use_memory_pool("t"):
        pass
    assert mapped_usage(allocator) == nbytes
    y.fill_(1)
    allocator.sleep(offload_tags="t")
    allocator.wake_up()
    assert int(y.sum()) == y.numel()
    allocator.release_pools()


@pytest.mark.parametrize("level", [1, 2], ids=["sleep-1", "sleep-2"])
@pytest.mark.parametrize("runner", ["v1", "v2"])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_kv_init_placement_survives_sleep(level, runner, monkeypatch):
    """KV-init state lands in runtime, KV in kv_cache, and connector memory
    outside every tag; runtime and KV survive sleep at both levels."""
    from functools import partial
    from types import SimpleNamespace as NS

    import vllm.v1.worker.gpu.kv_connector as v2_kv_connector
    import vllm.v1.worker.gpu_model_runner as v1_runner
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

    numel = 1 << 20

    class Connector:
        def get_mem_pool_context(self):
            return None

        def register_kv_caches(self, kv_caches):
            self.kv_caches = kv_caches
            self.buffer = torch.ones(numel, device=DEVICE_TYPE)

        def set_host_xfer_buffer_ops(self, copy_op):
            pass

    connector = Connector()
    for module in (gpu_worker, v1_runner, v2_kv_connector):
        monkeypatch.setattr(module, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(module, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_initialized", lambda *a: None)

    def initialize_kv_cache(kv_cache_config, kv_cache_allocation_context):
        model_runner.block_table = torch.arange(numel, device=DEVICE_TYPE)
        with kv_cache_allocation_context:
            model_runner.kv = torch.zeros(numel, device=DEVICE_TYPE)
        return {"layer": model_runner.kv}

    runner_cls = {"v1": v1_runner.GPUModelRunner, "v2": GPUModelRunnerV2}[runner]
    model_runner = NS(vllm_config=None, initialize_kv_cache=initialize_kv_cache)
    model_runner.init_kv_connector = partial(runner_cls.init_kv_connector, model_runner)
    model_config = NS(enable_cumem_allocator=True, enable_sleep_mode=True)
    worker = NS(
        cache_config=NS(num_gpu_blocks=None),
        vllm_config=NS(model_config=model_config),
        model_runner=model_runner,
    )
    worker._maybe_get_memory_pool_context = partial(
        gpu_worker.Worker._maybe_get_memory_pool_context, worker
    )
    kv_cache_config = NS(
        num_blocks=1, kv_cache_layout=None, needs_kv_cache_zeroing=False
    )
    gpu_worker.Worker.initialize_from_config(worker, kv_cache_config)

    allocator = get_mem_allocator_instance()

    def tag_of(tensor):
        ptr = tensor.data_ptr()
        for base, data in allocator.pointer_to_data.items():
            if base <= ptr < base + data.handle[1]:
                return data.tag
        return None

    assert tag_of(model_runner.block_table) == "runtime"
    assert tag_of(model_runner.kv) == "kv_cache"
    assert connector.kv_caches["layer"] is model_runner.kv
    assert tag_of(connector.buffer) is None

    backend = CuMemBackend()
    backend.suspend(level=level)
    assert mapped_usage(allocator) == 0

    backend.resume(tags=["weights"])
    assert torch.equal(
        model_runner.block_table, torch.arange(numel, device=DEVICE_TYPE)
    )
    backend.resume(tags=["kv_cache"])
    model_runner.kv.fill_(1)
    assert int(model_runner.kv.sum()) == numel


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_level2_discards_ordinary_tensor_with_weights_tag():
    """Discarded weights-tag memory is remapped; ROCm zeroes it over stale pages."""
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
    expected = 0 if current_platform.is_rocm() else 0xA5
    assert torch.all(fake_weight == expected)
    assert torch.all(ordinary_tensor == expected)


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


@pytest.mark.parametrize("level", [1, 2], ids=["sleep-1", "sleep-2"])
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="cuMem CUDA graph pool"
)
def test_cudagraph_pool_sleep(level):
    """Routing, and the graph pool backed up at both sleep levels and restored
    in place by any wake."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import vllm.distributed.device_communicators.pynccl_allocator as nccl_alloc
    from vllm.compilation.cudagraph_pool import (
        capture_outside_cumem_pool,
        capture_pool,
    )
    from vllm.device_allocator.sleep_mode_backend import CuMemBackend

    allocator = get_mem_allocator_instance()
    on, off = (SimpleNamespace(use_cumem_cudagraph_pool=v) for v in (True, False))
    handle = current_platform.graph_pool_handle()
    for cfg, ctx in ((off, nullcontext()), (on, capture_outside_cumem_pool())):
        with ctx, capture_pool(handle, cfg) as used:
            assert used == handle and allocator.current_tag != "cudagraph"
    with allocator.use_memory_pool("weights"):
        weight = torch.full((1 << 20,), 2.0, device=DEVICE_TYPE)
    x = torch.ones_like(weight)

    def capture() -> list:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with (
            capture_pool(handle, on) as pool,
            torch.cuda.graph(graph, pool=pool, stream=stream),
        ):
            assert nccl_alloc._graph_pool_id == pool != handle
            const = torch.empty_like(x)  # Never written by replay: needs a backup.
            y = x * weight + const
        return [graph, y, const.fill_(3.0)]

    def graph_pool() -> dict[int, int]:
        data = allocator.pointer_to_data.items()
        return {
            p: d.handle[1] for p, d in data if d.tag == "cudagraph" and not d.is_asleep
        }

    held, backend = capture(), CuMemBackend()
    mapped = graph_pool()
    backend.suspend(level=level)
    backend.resume(tags=["kv_cache"])
    assert graph_pool() == mapped
    backend.resume(tags=["weights"])
    weight.fill_(2.0)  # Level 2 discards weights; emulate the reload.
    held[0].replay()
    assert torch.equal(held[1], torch.full_like(x, 5.0))
