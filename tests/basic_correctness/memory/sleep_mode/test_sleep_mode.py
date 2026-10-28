# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
import os

import numpy as np
import pytest
import torch

import vllm.device_allocator.cumem as cumem
import vllm.envs as envs
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

from ....utils import (
    create_new_process_for_each_test,
    multi_gpu_marks,
    multi_gpu_test,
    requires_fp8,
)


def _r3_capture_state(worker):
    runner = worker.model_runner
    connector = runner.aux_output_connector
    capturer = connector._capturer
    callbacks, sinks = {}, {}
    for module in runner.model.modules():
        for target in (module, getattr(module, "router", None)):
            callback = getattr(target, "capture_fn", None)
            if callback is not None:
                assert callback.func.__self__ is capturer
                callbacks[id(callback)] = True
        sink = getattr(getattr(module, "routed_experts", None), "routing_sink", None)
        if sink is not None:
            assert sink.capture_fn.func.__self__ is capturer
            sinks[id(sink)] = sink.buffer.data_ptr()
    assert len(callbacks) + len(sinks) == capturer.shape_per_token[0]
    assert connector._store is not None or worker.rank != 0
    return (
        id(connector),
        id(capturer),
        capturer.device_buffer.data_ptr(),
        callbacks,
        sinks,
    )


def _profile_r3_again(worker):
    before = _r3_capture_state(worker)
    worker.model_runner.profile_run()
    assert _r3_capture_state(worker) == before


def _has_r3_graphs(worker):
    return bool(worker.model_runner.cudagraph_manager.graphs)


@pytest.mark.parametrize(
    "eager,tp,kv_bytes",
    [
        pytest.param(True, 1, None, id="eager-auto-kv"),
        pytest.param(False, 1, None, id="graph-auto-kv"),
        pytest.param(True, 1, 64 << 20, id="eager-explicit-kv"),
        pytest.param(False, 2, None, marks=multi_gpu_marks(num_gpus=2), id="graph-tp2"),
    ],
)
@create_new_process_for_each_test()
def test_routed_experts_capture_survives_sleep_and_reload(
    monkeypatch, tmp_path, eager, tp, kv_bytes
):
    """R3 buffers and callbacks survive profiling, sleep/wake and weight reload."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    Qwen3MoeForCausalLM(config).save_pretrained(tmp_path)
    llm = LLM(
        str(tmp_path),
        skip_tokenizer_init=True,
        enable_sleep_mode=True,
        enable_return_routed_experts=True,
        enforce_eager=eager,
        tensor_parallel_size=tp,
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.3,
        kv_cache_memory_bytes=kv_bytes,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    state = llm.collective_rpc(_r3_capture_state)
    llm.collective_rpc(_profile_r3_again)
    if not eager:
        assert all(llm.collective_rpc(_has_r3_graphs))
    prompt = {"prompt_token_ids": [1, 2, 3, 4]}
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    expected = None
    for phase in ("initial", "wake", "reload"):
        if phase == "wake":
            llm.sleep(level=1)
            llm.wake_up()
        elif phase == "reload":
            llm.collective_rpc("reload_weights")
        output = llm.generate(prompt, params)[0].outputs[0]
        routes = output.routed_experts
        assert routes is not None
        assert routes.shape == (7, 2, 2)
        assert routes.dtype == np.uint8
        assert np.all(routes < 4)
        assert np.all(routes[..., 0] != routes[..., 1])
        if expected is None:
            expected = (output.token_ids, routes.copy())
        else:
            assert output.token_ids == expected[0]
            np.testing.assert_array_equal(routes, expected[1])
        assert llm.collective_rpc(_r3_capture_state) == state


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


def _model_state(worker) -> dict[str, list]:
    state = vars(worker.model_runner.model_state)
    return {k: v.tolist() for k, v in state.items() if isinstance(v, torch.Tensor)}


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "model",
    [
        "hmellor/tiny-random-LlamaForCausalLM",
        # Its MRV2 model state holds device tensors.
        "yujiepan/mamba2-codestral-v0.1-tiny-random",
    ],
)
def test_deep_sleep(model, monkeypatch, gpu_memory_cleared):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Needed for collective_rpc to run a function on the worker.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    free, total = torch.accelerator.get_memory_info()
    used_bytes_baseline = total - free  # in case other process is running
    llm = LLM(model, enable_sleep_mode=True, max_model_len=256)
    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)
    state = llm.collective_rpc(_model_state)

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
    # Sleep must not change any model state, not even values read only later.
    assert llm.collective_rpc(_model_state) == state
    output2 = llm.generate(prompt, sampling_params)

    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_lora(gpu_memory_cleared):
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


def _lora_logits_mapping_correct(model) -> bool:
    from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA

    processors = [m for m in model.modules() if isinstance(m, LogitsProcessorWithLoRA)]
    return bool(processors) and all(
        m.sharded_to_full_mapping_gpu is not None
        and torch.equal(
            m.sharded_to_full_mapping_gpu,
            m.sharded_to_full_mapping_gpu.new_tensor(m.sharded_to_full_mapping),
        )
        for m in processors
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="LoRA state is kept in the runtime pool only on CUDA/ROCm",
)
def test_deep_sleep_lora_tp2(monkeypatch):
    """Level-2 sleep/wake/reload with enable_lora=True and TP=2.

    With TP > 1 the LoRA logits processor carries
    ``sharded_to_full_mapping_gpu``, a permanent index mapping used to
    reorder gathered logits. Level-2 sleep must preserve it, like the rest
    of the LoRA state.
    """
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
    assert all(llm.apply_model(_lora_logits_mapping_correct))

    prompt = "How are you?"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    # Check before reload_weights, which also resets the mapping.
    assert all(llm.apply_model(_lora_logits_mapping_correct))
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    assert output[0].outputs[0].text == output2[0].outputs[0].text


def _cudagraph_bytes(worker) -> int:
    data = cumem.CuMemAllocator.get_instance().pointer_to_data.values()
    return sum(d.handle[1] for d in data if d.tag == "cudagraph")


def _custom_ar_active(worker) -> bool:
    from vllm.distributed.parallel_state import get_tp_group

    comm = get_tp_group().device_communicator
    return any(
        c is not None and not c.disabled for c in (comm.ca_comm, comm.aiter_ar_comm)
    )


def _capture_registers_buffers(worker) -> bool:
    from vllm.distributed.parallel_state import get_tp_group

    comm = get_tp_group().device_communicator
    ca, aiter = comm.ca_comm, comm.aiter_ar_comm
    return (ca is not None and ca._capture_registered) or (
        aiter is not None and aiter.aiter_ca.enable_register_for_capturing
    )


@pytest.mark.parametrize(
    ("mode", "breakable", "tp", "offload"),
    [
        ("FULL", False, 1, True),
        ("PIECEWISE", False, 1, True),
        ("PIECEWISE", True, 1, True),
        pytest.param(
            "FULL_AND_PIECEWISE", False, 2, True, marks=multi_gpu_marks(num_gpus=2)
        ),
        ("FULL_AND_PIECEWISE", False, 1, False),
    ],
    ids=["full", "piecewise", "breakable", "custom-ar-tp2", "off-by-default"],
)
@create_new_process_for_each_test()
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="cuMem CUDA graph pool"
)
def test_sleep_cudagraph_pool(monkeypatch, mode, breakable, tp, offload):
    """Each capture site uses the pool, exact across sleeps; TP=2 needs
    unregistered custom AR; off changes nothing."""
    for name, value in [
        ("VLLM_USE_V2_MODEL_RUNNER", "1"),
        ("VLLM_USE_BREAKABLE_CUDAGRAPH", "1" if breakable else "0"),
        ("VLLM_ALLOW_INSECURE_SERIALIZATION", "1"),
        ("VLLM_ALLREDUCE_USE_FLASHINFER", "0"),
        ("VLLM_ALLREDUCE_USE_SYMM_MEM", "0"),
    ]:
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    llm = LLM(
        "hmellor/tiny-random-LlamaForCausalLM",
        enable_sleep_mode=True,
        sleep_mode_offload_cudagraph=offload,
        tensor_parallel_size=tp,
        compilation_config={
            "cudagraph_mode": mode,
            "pass_config": {"fuse_allreduce_rms": False},
        },
    )
    graph_bytes = llm.collective_rpc(_cudagraph_bytes)
    assert os.environ.get("NCCL_GRAPH_REGISTER") == ("0" if offload else None)
    assert all(graph_bytes) if offload else not any(graph_bytes)
    if not offload:
        return
    if tp > 1:
        if not all(llm.collective_rpc(_custom_ar_active)):
            pytest.skip("Custom allreduce is unavailable on these GPUs")
        assert not any(llm.collective_rpc(_capture_registers_buffers))
    prompt, params = "How are you?", SamplingParams(temperature=0, max_tokens=10)
    expected = llm.generate(prompt, params)[0].outputs[0].text
    for level in (1, 2):
        llm.sleep(level=level)
        llm.wake_up(tags=["weights"])
        if level == 2:
            llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])
        assert llm.generate(prompt, params)[0].outputs[0].text == expected


@multi_gpu_test(num_gpus=2)
@create_new_process_for_each_test()
@pytest.mark.skipif(not current_platform.is_rocm(), reason="AITER fused allreduce")
def test_sleep_cudagraph_pool_aiter_fused_ar_group_quant(monkeypatch):
    """The fused AITER AR + RMSNorm + per-group FP8 quant captures into the
    cuMem pool without IPC-registering its input, across sleeps."""
    for name, value in [
        ("VLLM_USE_V2_MODEL_RUNNER", "1"),
        ("VLLM_ROCM_USE_AITER", "1"),
        ("VLLM_ALLOW_INSECURE_SERIALIZATION", "1"),
    ]:
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)
    llm = LLM(
        "Qwen/Qwen3-0.6B-FP8",
        enable_sleep_mode=True,
        sleep_mode_offload_cudagraph=True,
        tensor_parallel_size=2,
        max_model_len=1024,
        compilation_config={"pass_config": {"fuse_allreduce_rms": True}},
    )
    if not all(llm.collective_rpc(_custom_ar_active)):
        pytest.skip("AITER custom allreduce is unavailable on these GPUs")
    assert not any(llm.collective_rpc(_capture_registers_buffers))
    assert all(llm.collective_rpc(_cudagraph_bytes))
    prompt, params = "How are you?", SamplingParams(temperature=0, max_tokens=10)
    assert llm.generate(prompt, params)[0].outputs[0].text
    for level in (1, 2):
        llm.sleep(level=level)
        llm.wake_up(tags=["weights"])
        if level == 2:
            llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])
        assert llm.generate(prompt, params)[0].outputs[0].text


@create_new_process_for_each_test()
def test_deep_sleep_async(gpu_memory_cleared):
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


def _load_weights(worker) -> None:
    # Not reload_weights(): its layerwise finalize re-derives the KV scales.
    model = worker.model_runner.get_model()
    loader = get_model_loader(worker.load_config)
    model.load_weights(loader.get_all_weights(worker.model_config, model))


def _attn_kv_scales(model) -> dict[str, list[float]]:
    return {
        f"{name}.{attr}": getattr(module, attr).flatten().tolist()
        for name, module in model.named_modules()
        for attr in ("_q_scale", "_k_scale", "_v_scale")
        if isinstance(getattr(module, attr, None), torch.Tensor)
    }


@requires_fp8
@create_new_process_for_each_test()
def test_deep_sleep_compressed_tensors_kv_scales(
    monkeypatch: pytest.MonkeyPatch, gpu_memory_cleared
):
    """KV scales must survive level-2 sleep plus a weights reload (else NaN)."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-kvcache-fp8-tensor",
        enable_sleep_mode=True,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        gpu_memory_utilization=0.6,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=10, logprobs=0)
    scales = llm.apply_model(_attn_kv_scales)[0]
    assert any(v != 1.0 for values in scales.values() for v in values)
    expected = llm.generate("How are you?", sampling_params)[0].outputs[0]

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc(_load_weights)
    llm.wake_up(tags=["kv_cache"])

    assert llm.apply_model(_attn_kv_scales)[0] == scales
    actual = llm.generate("How are you?", sampling_params)[0].outputs[0]
    assert actual.token_ids == expected.token_ids
    logprobs = [
        [lp[t].logprob for lp, t in zip(o.logprobs, o.token_ids)]
        for o in (expected, actual)
    ]
    assert all(math.isfinite(x) for x in logprobs[1])
    assert logprobs[0] == logprobs[1]


@requires_fp8
def test_deep_sleep_fp8_kvcache(gpu_memory_cleared):
    # Regression test for https://github.com/vllm-project/vllm/pull/28783:
    # fp8 KV cache output must be restored correctly after level-2 sleep.
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
def test_deep_sleep_fp8_kvcache_with_undefined_remap(
    monkeypatch: pytest.MonkeyPatch,
):
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
