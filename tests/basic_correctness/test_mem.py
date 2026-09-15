# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import glob
import math
import os

import pytest
import torch

import vllm.device_allocator.cumem as cumem
import vllm.envs as envs
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.device_allocator import get_mem_allocator_instance
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test, requires_fp8

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


# --- level-2 sleep vs. attention calibration scales ---------------------------
# Placement is load-bearing: keep this section ABOVE the two
# test_deep_sleep_fp8_kvcache_mrv1* tests at the bottom of this file. The second
# of those sets VLLM_ENABLE_V1_MULTIPROCESSING=0, so it builds its engine inside
# the pytest parent and leaves its gpu_memory_utilization=0.9 KV pool mapped
# there: a third engine started afterwards once found only 9.17 of 95.07 GiB free
# and failed to start. This test forks, but a fork inherits the parent's device
# memory, so it has to run while the parent is still clean.
# A compressed-tensors checkpoint carrying a `kv_cache_scheme` is what selects
# CompressedTensorsKVCacheMethod -- the only quant method that rebinds
# _q/_k/_v_scale onto nn.Parameter. Plain kv_cache_dtype="fp8" selects
# Fp8KVCacheMethod, which keeps every scale as a buffer and is unaffected.
KV_SCALE_MODEL = os.environ.get(
    "VLLM_TEST_KV_SCALE_MODEL",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-kvcache-fp8-tensor",
)
KV_SCALE_NAMES = ("_q_scale", "_k_scale", "_v_scale", "_prob_scale")
CALIBRATED_NAMES = ("_k_scale", "_v_scale")


def _collect_attn_scales(worker) -> dict[str, tuple[str, list[float]]]:
    """Map every attention calibration scale to (registration form, values).

    The registration form is the point of the test: Worker.sleep(level=2)
    snapshots model.named_buffers() only, so a scale that ended up in
    module._parameters is dropped by a level-2 sleep and never restored.
    """
    model = worker.model_runner.model
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    scales = {}
    for name, tensor in [*buffers.items(), *parameters.items()]:
        if name.rsplit(".", 1)[-1] in KV_SCALE_NAMES:
            form = "buffer" if name in buffers else "parameter"
            scales[name] = (form, tensor.detach().float().cpu().flatten().tolist())
    return scales


def _reload_weights_only(worker, model_path: str) -> int:
    """Reload checkpoint weights in place, RL weight-sync style.

    model.load_weights() only: Worker.reload_weights() is deliberately NOT used
    because finalize_layerwise_reload() re-runs the quant method's
    process_weights_after_loading(), which re-derives the scales and hides the
    regression this test exists for.
    """
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    assert files, f"no safetensors under {model_path}"

    def iterate_weights():
        for path in files:
            with safe_open(path, framework="pt") as shard:
                for key in shard.keys():  # noqa: SIM118
                    tensor = shard.get_tensor(key)
                    # Skip non-float metadata entries: load_weights() expects
                    # tensors it can copy into a parameter, and a MoE checkpoint
                    # can carry int64 bias entries next to the weights.
                    if tensor.is_floating_point():
                        yield key, tensor

    return len(worker.model_runner.model.load_weights(iterate_weights()))


def _first_large_weight(worker) -> str:
    """Deterministic pick of one big weight parameter, for the probe below."""
    names = sorted(
        name
        for name, tensor in worker.model_runner.model.named_parameters()
        if tensor.numel() >= 1_000_000 and name.rsplit(".", 1)[-1] not in KV_SCALE_NAMES
    )
    assert names, "no large weight parameter found to probe"
    return names[0]


def _weight_probe(worker, name: str) -> dict:
    """(all_zero, float64 checksum) of one weight tensor.

    Anti-vacuity guard: it proves the level-2 sleep really discarded the weights
    pool and that the reload really put the same bytes back, so the scale
    assertions cannot pass on an engine that was never disturbed.  A returned
    count from load_weights() is NOT such a proof -- it counts parameter names
    that were processed, not values that were written.
    """
    tensor = dict(worker.model_runner.model.named_parameters())[name]
    flat = tensor.detach().double().flatten()
    return {
        "name": name,
        "numel": flat.numel(),
        "all_zero": bool(flat.abs().max().item() == 0.0),
        "checksum": flat.sum().item(),
    }


# 2.46 GiB (weights + non-torch) + 0.69 GiB (peak activation) is the measured
# floor for this 1.1B checkpoint before a single KV block is allocated, so the
# 0.6 reservation below still leaves ~7.7 GiB for KV on an 18 GiB device: this
# fits the smallest CI GPU slice, so the test needs no minimum-memory mark.
@requires_fp8
@create_new_process_for_each_test("fork" if current_platform.is_cuda() else "spawn")
def test_deep_sleep_fp8_kvcache_compressed_tensors_kv_scale(
    monkeypatch: pytest.MonkeyPatch,
):
    """Level-2 sleep must not zero the compressed-tensors KV calibration scales.

    CompressedTensorsKVCacheMethod.process_weights_after_loading() rebinds
    layer._q_scale/_k_scale/_v_scale onto the nn.Parameter placeholders, which
    moves those three names out of module._buffers. The only level-2 fallback in
    Worker.sleep()/Worker.wake_up() snapshots named_buffers(), so the rebound
    scales were discarded together with the "weights" pool and read back 0.0
    from the freshly mapped pages, while _prob_scale (never rebound, still a
    buffer) kept its value. Descaling by 0.0 then yielded NaN logprobs and a
    wrong first token on the very next request.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    envs.disable_envs_cache()

    model = KV_SCALE_MODEL
    # The reload helper globs a DIRECTORY, so resolve the hub id to the local
    # snapshot the engine itself loaded from (offline-safe: hits the HF cache).
    model_dir = hf_api().snapshot_download(model)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=16, logprobs=1)

    llm = LLM(
        model,
        enable_sleep_mode=True,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        max_model_len=2048,
        # 0.6 x 18 GiB = 10.8 GiB clears the ~3.2 GiB floor above with ~7.7 GiB left
        # for KV blocks -- far more than one 2048-token sequence needs.
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,  # deterministic; no cached-block masking
    )

    def scales_now() -> dict[str, tuple[str, list[float]]]:
        return llm.collective_rpc(_collect_attn_scales)[0]

    scales_before = scales_now()
    found = {name.rsplit(".", 1)[-1] for name in scales_before}
    assert found == set(KV_SCALE_NAMES), f"expected {KV_SCALE_NAMES}, found {found}"

    # (a) the invariant under test: every attention calibration scale must be a
    # BUFFER, because that is the only form the level-2 snapshot can see.
    as_parameter = sorted(
        name for name, (form, _) in scales_before.items() if form != "buffer"
    )
    assert not as_parameter, (
        "calibration scales registered as nn.Parameter are invisible to the "
        f"named_buffers() level-2 sleep snapshot and will be lost: {as_parameter}"
    )
    # Non-vacuity: k/v must carry real calibrated values from the checkpoint,
    # not the 1.0 placeholders, otherwise everything below could pass by luck.
    calibrated = {
        name: values
        for name, (_, values) in scales_before.items()
        if name.endswith(CALIBRATED_NAMES)
    }
    assert calibrated, f"no {CALIBRATED_NAMES} tensors found"
    assert any(v != 1.0 for values in calibrated.values() for v in values), (
        f"expected calibrated k/v scales, got placeholders: {calibrated}"
    )

    probe_name = llm.collective_rpc(_first_large_weight)[0]
    weight_before = llm.collective_rpc(_weight_probe, args=(probe_name,))[0]
    assert not weight_before["all_zero"], f"{probe_name} is zero before sleeping"

    expected = llm.generate(prompt, sampling_params)

    def assert_scales_intact(what: str) -> None:
        now = scales_now()
        assert set(now) == set(scales_before), f"{what}: scale set changed"
        zeroed = sorted(n for n, (_, v) in now.items() if any(x == 0.0 for x in v))
        assert not zeroed, f"{what}: zeroed calibration scales: {zeroed}"
        # (b) exact equality, not a tolerance: a 4.7e-9 perturbation of these
        # scales was measured to flip a greedy decision and move logprobs by 8.5.
        drifted = sorted(n for n in scales_before if scales_before[n] != now[n])
        assert not drifted, (
            f"{what}: calibration scales changed: "
            f"{ {n: (scales_before[n][1], now[n][1]) for n in drifted[:4]} }"
        )

    def assert_same_output(what: str) -> None:
        actual = llm.generate(prompt, sampling_params)[0].outputs[0]
        want = expected[0].outputs[0]
        # (c) greedy decoding must be untouched
        assert actual.token_ids == want.token_ids, f"{what}: token_ids diverged"
        assert actual.text == want.text, f"{what}: text diverged"
        # (d) logprobs finite and bit-identical
        assert want.logprobs and actual.logprobs, f"{what}: no logprobs returned"
        for pos, (want_lp, got_lp) in enumerate(zip(want.logprobs, actual.logprobs)):
            for token_id, want_entry in want_lp.items():
                got_entry = got_lp[token_id].logprob
                assert math.isfinite(got_entry), (
                    f"{what}: non-finite logprob at position {pos}"
                )
                assert got_entry == want_entry.logprob, (
                    f"{what}: logprob drift at position {pos}: "
                    f"{want_entry.logprob} -> {got_entry}"
                )

    # Negative control: level 1 offloads to pinned staging and remaps the same
    # handles, so neither the scales nor the output may change at all.
    llm.sleep(level=1)
    llm.wake_up()
    assert_scales_intact("after sleep(level=1)")
    assert_same_output("after sleep(level=1)")
    assert llm.collective_rpc(_weight_probe, args=(probe_name,))[0] == weight_before, (
        "sleep(level=1) must restore the weights byte-exactly"
    )

    # The regression itself: deep sleep, reload weights only (the shape of an RL
    # weight-sync), then bring the KV cache back.
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    discarded = llm.collective_rpc(_weight_probe, args=(probe_name,))[0]
    assert discarded["all_zero"], (
        "level-2 sleep did not discard the weights pool, so this test is not "
        f"exercising the reported path ({probe_name} checksum={discarded['checksum']})"
    )
    loaded = llm.collective_rpc(_reload_weights_only, args=(model_dir,))[0]
    assert loaded > 0, "the weight reload loaded nothing; the test would be vacuous"
    weight_after = llm.collective_rpc(_weight_probe, args=(probe_name,))[0]
    assert weight_after["checksum"] == weight_before["checksum"], (
        f"the weights-only reload did not restore {probe_name} bit-exactly: "
        f"{weight_before['checksum']} -> {weight_after['checksum']}; the output "
        "comparison below would then be measuring the reload, not the scales"
    )
    llm.wake_up(tags=["kv_cache"])

    assert_scales_intact("after sleep(level=2) + weights-only reload")
    assert_same_output("after sleep(level=2) + weights-only reload")


@requires_fp8
def test_deep_sleep_fp8_kvcache_mrv1(monkeypatch: pytest.MonkeyPatch):
    # Regression test for https://github.com/vllm-project/vllm/pull/28783.
    # In particular, verify that MRV1 does not rely on post_kv_cache_wake_up()
    # to restore correct output after level-2 sleep.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    envs.disable_envs_cache()

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


@requires_fp8
def test_deep_sleep_fp8_kvcache_mrv1_with_undefined_remap(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()

    llm = LLM(
        "Qwen/Qwen2-0.5B",
        enable_sleep_mode=True,
        kv_cache_dtype="fp8",
    )
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    expected = llm.generate(prompt, sampling_params)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")

    original_create_and_map = cumem.create_and_map

    def create_and_map_with_poison(handle) -> None:
        original_create_and_map(handle)
        _, size, ptr, _ = handle
        cumem.libcudart.cudaMemset(ptr, 0xA5, size)

    monkeypatch.setattr(cumem, "create_and_map", create_and_map_with_poison)

    # New requests must overwrite undefined remapped KV bytes before reading them.
    llm.wake_up(tags=["kv_cache"])
    actual = llm.generate(prompt, sampling_params)

    assert expected[0].outputs[0].text == actual[0].outputs[0].text
