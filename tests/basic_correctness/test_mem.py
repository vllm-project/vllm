# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os

import pytest
import torch
from torch import nn

import vllm.device_allocator.cumem as cumem
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.device_allocator import get_mem_allocator_instance
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test, requires_fp8

DEVICE_TYPE = current_platform.device_type

GEMMA3N_SLEEP_MODEL = "google/gemma-3n-E2B-it"
GEMMA3N_SLEEP_WAKE_CYCLES = int(os.getenv("VLLM_SLEEP_WAKE_REPRO_CYCLES", "3"))
GEMMA3N_SLEEP_TENSOR_NAMES = (
    "embed_scale",
    "embed_scale_per_layer",
    "per_layer_input_scale",
    "per_layer_projection_scale",
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


def _gemma3n_sleep_tensor_snapshot(model) -> dict[str, tuple[int, float]]:
    """Return addresses and values for Gemma 3n sleep-sensitive tensors."""
    from vllm.model_executor.models.gemma3n import Gemma3nSelfDecoder

    decoder = next(
        (
            module
            for module in model.modules()
            if isinstance(module, Gemma3nSelfDecoder)
        ),
        None,
    )
    assert decoder is not None
    return {
        name: (tensor.data_ptr(), tensor.float().item())
        for name in GEMMA3N_SLEEP_TENSOR_NAMES
        if (tensor := getattr(decoder, name, None)) is not None
    }


def _generation_signature(output) -> tuple[tuple[int, ...], tuple[float, ...]]:
    completion = output[0].outputs[0]
    token_ids = tuple(completion.token_ids)
    assert completion.logprobs is not None
    chosen_logprobs = tuple(
        step[token_id].logprob
        for token_id, step in zip(token_ids, completion.logprobs, strict=True)
    )
    return token_ids, chosen_logprobs


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_python_error():
    """
    Test if Python error occurs when there's low-level
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

    free_bytes = torch.accelerator.get_memory_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.accelerator.get_memory_info()[0]
    assert free_bytes_after_sleep > free_bytes
    allocator.wake_up()

    # they can be used together
    output = x + y + z
    assert torch.allclose(output, torch.ones_like(output) * 3)


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_discard_tags():
    """Test that discard(tags) selectively frees GPU memory for specific
    tags while keeping other tags mapped and usable."""
    allocator = get_mem_allocator_instance()

    with allocator.use_memory_pool("weights"):
        weights = torch.ones(1024, 1024, device=DEVICE_TYPE)

    with allocator.use_memory_pool("kv_cache"):
        kv = torch.ones(512, 512, device=DEVICE_TYPE)

    free_bytes = torch.accelerator.get_memory_info()[0]

    # Discard kv_cache only — weights should remain valid
    allocator.discard("kv_cache")

    free_bytes_after_discard = torch.accelerator.get_memory_info()[0]
    assert free_bytes_after_discard > free_bytes

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
def test_tagged_ordinary_tensor_is_discarded_with_kv_cache():
    """Reproduce allocation-tag contamination with an ordinary torch tensor.

    The allocator tags allocations, not semantic tensor owners. This test is a
    contract-level reproducer: production code must keep persistent metadata
    outside the discardable KV-cache scope.
    """
    allocator = get_mem_allocator_instance()

    with allocator.use_memory_pool("weights"):
        weight = torch.full((4096,), 0x11, dtype=torch.uint8, device=DEVICE_TYPE)
    with allocator.use_memory_pool("kv_cache"):
        fake_kv = torch.full((4096,), 0x22, dtype=torch.uint8, device=DEVICE_TYPE)
        ordinary_tensor = torch.full(
            (4096,), 0x33, dtype=torch.uint8, device=DEVICE_TYPE
        )

    pointers = tuple(t.data_ptr() for t in (weight, fake_kv, ordinary_tensor))
    allocator.sleep(offload_tags=("weights",))
    _wake_up_with_poisoned_mappings(allocator)
    torch.accelerator.synchronize()

    assert tuple(t.data_ptr() for t in (weight, fake_kv, ordinary_tensor)) == pointers
    assert torch.all(weight == 0x11)
    assert torch.all(fake_kv == 0xA5)
    assert torch.all(ordinary_tensor == 0xA5)


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses the CuMem allocator")
def test_level2_discards_ordinary_tensor_with_weights_tag():
    """Reproduce the level-2 variant for an ordinary tensor in weights."""
    allocator = get_mem_allocator_instance()

    with allocator.use_memory_pool("weights"):
        fake_weight = torch.full(
            (4096,), 0x44, dtype=torch.uint8, device=DEVICE_TYPE
        )
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
@pytest.mark.skipif(current_platform.is_xpu(), reason="Uses CUDA graph and CuMem")
def test_cudagraph_replays_with_corrupted_tagged_constant():
    """Reproduce silent model corruption despite a stable captured pointer."""

    class TinyScaleModel(nn.Module):
        """One-operation model with persistent runtime metadata.

        ``scale`` deliberately remains a plain tensor attribute. Constructing
        this model in the KV-cache pool reproduces a persistent model value
        accidentally inheriting the cache's discard-on-sleep policy.
        """

        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.tensor([3.0], device=DEVICE_TYPE)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    allocator = get_mem_allocator_instance()
    x = torch.tensor([2.0], device=DEVICE_TYPE)
    with allocator.use_memory_pool("kv_cache"):
        model = TinyScaleModel()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(x)
    scale_ptr = model.scale.data_ptr()

    allocator.sleep(offload_tags=())
    _wake_up_with_poisoned_mappings(allocator)
    graph.replay()
    torch.accelerator.synchronize()

    assert model.scale.data_ptr() == scale_ptr
    assert not torch.equal(output, torch.tensor([6.0], device=DEVICE_TYPE))

    # Owners that cannot move storage out of a discardable scope must recover
    # values in place so captured pointers remain valid.
    model.scale.fill_(3.0)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(output, torch.tensor([6.0], device=DEVICE_TYPE))


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

    free_bytes = torch.accelerator.get_memory_info()[0]
    allocator.sleep()
    free_bytes_after_sleep = torch.accelerator.get_memory_info()[0]
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


@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
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
    free, total = torch.accelerator.get_memory_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # the benefit of `llm.sleep(level=2)` is mainly CPU memory usage,
    # which is difficult to measure in the test. therefore, we only
    # test sleep level 1 here.
    llm.sleep(level=1)

    free_gpu_bytes_after_sleep, total = torch.accelerator.get_memory_info()
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

    free_gpu_bytes_wake_up_w, total = torch.accelerator.get_memory_info()
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
    free, total = torch.accelerator.get_memory_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Put the engine to deep sleep
    llm.sleep(level=2)

    free_gpu_bytes_after_sleep, total = torch.accelerator.get_memory_info()
    used_bytes = total - free_gpu_bytes_after_sleep - used_bytes_baseline
    assert used_bytes < 3 * GiB_bytes

    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    free_gpu_bytes_wake_up_w, total = torch.accelerator.get_memory_info()
    used_bytes = total - free_gpu_bytes_wake_up_w - used_bytes_baseline
    assert used_bytes < 4 * GiB_bytes

    # now allocate kv cache and cuda graph memory
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@create_new_process_for_each_test()
@pytest.mark.slow_test
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Reproduces the CUDA CuMemAllocator level-2 sleep path",
)
def test_gemma3n_level2_sleep_wake_preserves_generation(monkeypatch):
    """Detect naturally occurring Gemma 3n corruption after level-2 sleep."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    llm = LLM(
        model=GEMMA3N_SLEEP_MODEL,
        enable_sleep_mode=True,
        enforce_eager=True,
        attention_backend="TRITON_ATTN",
        max_model_len=256,
        max_num_seqs=1,
        seed=0,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    prompt = "Explain why the sky is blue in one short sentence."
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16, logprobs=1)

    before_output = llm.generate(prompt, sampling_params)
    before_token_ids, before_logprobs = _generation_signature(before_output)
    control_token_ids, control_logprobs = _generation_signature(
        llm.generate(prompt, sampling_params)
    )
    assert control_token_ids == before_token_ids
    assert control_logprobs == pytest.approx(before_logprobs, rel=1e-5, abs=1e-5)

    before_tensors = llm.apply_model(_gemma3n_sleep_tensor_snapshot)
    assert all(
        set(snapshot) == set(GEMMA3N_SLEEP_TENSOR_NAMES)
        for snapshot in before_tensors
    )

    for cycle in range(1, GEMMA3N_SLEEP_WAKE_CYCLES + 1):
        # Do not poison, zero, or allocate test-side GPU memory between sleep
        # and wake. Repeated cycles only increase the chance that the CUDA
        # driver naturally returns cleared or reused physical pages.
        llm.sleep(level=2)
        llm.wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights")
        after_tensors = llm.apply_model(_gemma3n_sleep_tensor_snapshot)
        llm.wake_up(tags=["kv_cache"])

        after_token_ids, after_logprobs = _generation_signature(
            llm.generate(prompt, sampling_params)
        )
        assert after_token_ids == before_token_ids, (
            f"cycle {cycle}: generated tokens changed after sleep/wake"
        )
        assert after_logprobs == pytest.approx(
            before_logprobs, rel=1e-5, abs=1e-5
        ), f"cycle {cycle}: selected-token logprobs changed after sleep/wake"
        # CuMem preserves virtual addresses. The model owner must restore the
        # original semantic values into those addresses after remapping.
        assert after_tensors == before_tensors, (
            f"cycle {cycle}: Gemma 3n runtime tensors changed after sleep/wake"
        )


@create_new_process_for_each_test()
def test_deep_sleep_lora():
    """Level-2 sleep/wake/reload with enable_lora=True.

    LoRA wrapping moves parameters under base_layer and adds LoRA
    stacked tensors that are plain attributes, not restored by the
    reload machinery — reload must forward checkpoint weights through
    the wrappers and reset the LoRA state afterwards.
    """
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(
        model,
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=8,
        enforce_eager=True,
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    # Level-2 sleep discards all GPU memory
    llm.sleep(level=2)

    # Reload weights from checkpoint
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    assert output[0].outputs[0].text == output2[0].outputs[0].text

    # Multiple cycles should not accumulate corruption
    for _ in range(3):
        llm.sleep(level=2)
        llm.wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])
    output3 = llm.generate(prompt, sampling_params)
    assert output[0].outputs[0].text == output3[0].outputs[0].text


def _lora_logits_mapping_present(model) -> bool:
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA

    return any(
        isinstance(m, LogitsProcessorWithLoRA)
        and m.sharded_to_full_mapping_gpu is not None
        for m in model.modules()
    )


@create_new_process_for_each_test()
def test_deep_sleep_lora_tp2(num_gpus_available, monkeypatch):
    """Level-2 sleep/wake/reload with enable_lora=True and TP=2.

    With TP > 1 the LoRA logits processor carries
    ``sharded_to_full_mapping_gpu``, a permanent index mapping used to
    reorder gathered logits. Like the LoRA stacked tensors it is a plain
    attribute allocated in the sleep-mode pool, so level-2 sleep destroys
    its contents — it must be restored after reload.
    """
    if num_gpus_available < 2:
        pytest.skip("Requires at least 2 GPUs")

    # Needed for apply_model to reach the multiproc TP workers below.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(
        model,
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=8,
        tensor_parallel_size=2,
        enforce_eager=True,
    )

    # Guard against this test silently not exercising the TP>1 reindex
    # path (e.g. if lm_head wrapping conditions change).
    assert all(llm.apply_model(_lora_logits_mapping_present))

    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_async():
    async def test():
        model = "hmellor/tiny-random-LlamaForCausalLM"
        free, total = torch.accelerator.get_memory_info()
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
        free_gpu_bytes_wake_up_w, total = torch.accelerator.get_memory_info()
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
