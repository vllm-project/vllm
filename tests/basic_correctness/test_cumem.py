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


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_keep_tags_preserves_gpu_allocation():
    """`CuMemAllocator.sleep(keep_tags=...)` must leave allocations whose
    tag is in `keep_tags` fully mapped on GPU and untouched: no CPU copy,
    no unmap, and the live tensor data must survive a sleep/wake cycle.
    This is the building block for tag-wise selective offload (hybrid
    co-location with partial rollout).
    """
    allocator = CuMemAllocator.get_instance()

    # weights: will be offloaded (CPU backup + unmap).
    with allocator.use_memory_pool(tag="weights"):
        weights = torch.eye(1024, device=DEVICE_TYPE)

    # kv_cache: must remain mapped on GPU across sleep().
    with allocator.use_memory_pool(tag="kv_cache"):
        kv = torch.full((1024, 1024), 7.0, device=DEVICE_TYPE)

    kv_ptr_before = kv.data_ptr()

    free_before = torch.cuda.mem_get_info()[0]
    allocator.sleep(offload_tags=("weights",), keep_tags=("kv_cache",))
    free_after = torch.cuda.mem_get_info()[0]

    # Some memory was freed (weights were unmapped) but not all of it
    # (kv_cache is still mapped).
    weights_bytes = weights.numel() * weights.element_size()
    kv_bytes = kv.numel() * kv.element_size()
    freed = free_after - free_before
    assert freed >= weights_bytes * 0.9, (
        f"expected weights ({weights_bytes} B) to be freed, only {freed} B was"
    )
    assert freed < weights_bytes + kv_bytes, (
        "kv_cache appears to have been unmapped despite keep_tags=('kv_cache',)"
    )

    # The live KV tensor must still be readable on GPU with its original
    # data intact, and at the same address.
    assert kv.data_ptr() == kv_ptr_before
    assert torch.allclose(kv, torch.full_like(kv, 7.0))

    allocator.wake_up()

    # After wake_up, weights are restored from CPU backup; kv was untouched.
    assert torch.allclose(weights, torch.eye(1024, device=DEVICE_TYPE))
    assert torch.allclose(kv, torch.full_like(kv, 7.0))


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_keep_and_offload_tags_must_be_disjoint():
    """Sanity guard: tag cannot be both kept and offloaded."""
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool(tag="weights"):
        _ = torch.zeros(8, device=DEVICE_TYPE)

    with pytest.raises(AssertionError):
        allocator.sleep(offload_tags=("weights",), keep_tags=("weights",))

    # Allocator state may be partially mutated; the test process is
    # isolated so we don't need to clean up.


@create_new_process_for_each_test()
def test_offload_tags_weights_only_preserves_kv_cache():
    """End-to-end: `LLM.sleep(offload_tags=["weights"])` must offload
    weights to CPU while keeping the KV cache live on GPU. After wake_up
    of just the weights tag, generation must still be possible without
    a separate KV-cache wake_up — this is the partial-rollout use case.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    free, total = torch.cuda.mem_get_info()
    used_bytes_baseline = total - free
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    reference = llm.generate(prompt, sampling_params)

    free_before_sleep = torch.cuda.mem_get_info()[0]
    llm.sleep(offload_tags=["weights"])
    free_after_sleep = torch.cuda.mem_get_info()[0]

    # Some GPU memory was freed (weights), but the KV cache pool must
    # still be mapped — i.e. less memory was freed than under the legacy
    # level=1 sleep.
    assert free_after_sleep > free_before_sleep, (
        "Sleep did not free any GPU memory."
    )
    used_bytes_after_sleep = total - free_after_sleep - used_bytes_baseline
    # KV cache for a 1B-class model under default settings is the
    # dominant chunk; if it had been freed we'd see usage drop further.
    # We just assert the engine still considers some GPU memory in use.
    assert used_bytes_after_sleep > 0

    # Wake only the weights — KV cache was never asleep.
    llm.wake_up(tags=["weights"])

    output_after = llm.generate(prompt, sampling_params)
    assert (
        reference[0].outputs[0].text == output_after[0].outputs[0].text
    ), "Output diverged after offload_tags=['weights'] sleep/wake cycle"


@create_new_process_for_each_test()
def test_offload_tags_empty_is_scheduler_only():
    """`LLM.sleep(offload_tags=[])` is a pure pause: no GPU memory is
    offloaded and the executor must NOT enter the sleeping state. If it
    did, a follow-up `wake_up(tags=["scheduling"])` would resume the
    scheduler while leaving the executor sleeping, and subsequent
    `sleep(...)` calls would be silently dropped.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=8)
    reference = llm.generate(prompt, sampling_params)

    # Empty offload_tags must not put the executor to sleep.
    llm.sleep(offload_tags=[])

    # Resume scheduling. After this, the engine must be fully usable
    # again — including for further sleep/wake cycles.
    llm.wake_up(tags=["scheduling"])

    # The next real sleep must take effect; if the executor were stuck
    # in sleeping state from the empty-tags call, this would be a no-op.
    free_before = torch.cuda.mem_get_info()[0]
    llm.sleep(level=1)
    free_after = torch.cuda.mem_get_info()[0]
    assert free_after > free_before, (
        "level=1 sleep after offload_tags=[] freed nothing — executor "
        "appears to have been stuck in sleeping state."
    )

    llm.wake_up()
    output_after = llm.generate(prompt, sampling_params)
    assert reference[0].outputs[0].text == output_after[0].outputs[0].text


@create_new_process_for_each_test("fork" if not current_platform.is_rocm() else "spawn")
def test_selective_sleep_is_reentrant_on_unmapped_allocations():
    """Calling `sleep` again while some allocations are already unmapped
    must be a no-op for those allocations: it must NOT re-issue
    `cudaMemcpy` from a freed source pointer or `unmap_and_release`
    on an already-released handle.

    Reproducer for the codex-flagged bug: first selective sleep offloads
    `weights` and keeps `kv_cache`; the second selective sleep offloads
    `kv_cache` and (in caller intent) keeps `weights`. Without the
    `is_mapped` guard the second pass would either treat the still-
    unmapped weights as "kept" (wrong accounting) or — without
    `keep_tags` — fall through to `cudaMemcpy`/`unmap` on a freed handle.
    """
    allocator = CuMemAllocator.get_instance()

    with allocator.use_memory_pool(tag="weights"):
        weights = torch.eye(512, device=DEVICE_TYPE)
    with allocator.use_memory_pool(tag="kv_cache"):
        kv = torch.full((512, 512), 3.0, device=DEVICE_TYPE)

    # First selective sleep: offload weights, keep kv_cache.
    allocator.sleep(offload_tags=("weights",), keep_tags=("kv_cache",))

    # Second selective sleep: now offload kv_cache too. Weights are
    # already unmapped; the loop must skip them rather than re-running
    # cudaMemcpy/unmap on a freed handle.
    allocator.sleep(offload_tags=("kv_cache",))

    # Both tags should now be restorable from CPU backups.
    allocator.wake_up()

    assert torch.allclose(weights, torch.eye(512, device=DEVICE_TYPE))
    assert torch.allclose(kv, torch.full_like(kv, 3.0))


@create_new_process_for_each_test()
def test_offload_tags_rejects_unknown_tag():
    """A typo like `offload_tags=["weight"]` must be rejected up front,
    not silently turned into a no-op sleep that wedges the executor.

    Without the worker-level validation, the bogus tag would be added to
    `_slept_tags`, no GPU pool would be freed, and `wake_up()` paths
    would refuse to wake the (real) tags — leaving the engine stuck in
    `is_sleeping=True`. We assert that the bad call raises eagerly and
    that the engine remains fully usable afterwards.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=8)
    reference = llm.generate(prompt, sampling_params)

    # Typo + blank must both be rejected. Validation must run BEFORE the
    # scheduler is paused, otherwise the failed call would leave the
    # scheduler in a paused state and `generate()` would hang.
    with pytest.raises(Exception):
        llm.sleep(offload_tags=["weight"])

    # Scheduler must be untouched: generate must work without an
    # explicit wake_up. If validation ran AFTER pause_scheduler, this
    # call would block forever in the paused scheduler queue.
    mid_check = llm.generate(prompt, sampling_params)
    assert reference[0].outputs[0].text == mid_check[0].outputs[0].text

    with pytest.raises(Exception):
        llm.sleep(offload_tags=[""])
    mid_check2 = llm.generate(prompt, sampling_params)
    assert reference[0].outputs[0].text == mid_check2[0].outputs[0].text

    # Real selective sleep + wake must still work end-to-end.
    llm.sleep(offload_tags=["weights"])
    llm.wake_up(tags=["weights"])

    output_after = llm.generate(prompt, sampling_params)
    assert reference[0].outputs[0].text == output_after[0].outputs[0].text


@create_new_process_for_each_test()
def test_recompute_sleep_staged_wake_keeps_scheduler_paused():
    """Recompute sleep releases KV cache, so staged wakeups must not resume
    scheduling after only weights are restored. Requests may resume only after
    the KV cache pool is remapped.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    reference = llm.generate(prompt, sampling_params)

    llm.sleep(mode="recompute")
    assert llm.llm_engine.is_sleeping()

    llm.wake_up(tags=["weights"])
    assert llm.llm_engine.is_sleeping(), (
        "staged weights wake_up must not resume scheduling before kv_cache wake_up"
    )

    llm.wake_up(tags=["kv_cache"])
    assert not llm.llm_engine.is_sleeping()

    output_after = llm.generate(prompt, sampling_params)
    assert reference[0].outputs[0].text == output_after[0].outputs[0].text


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
