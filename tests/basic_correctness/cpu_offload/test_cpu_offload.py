# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from operator import attrgetter

import pytest
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ...utils import compare_two_settings


@pytest.mark.parametrize("disable_pin_memory", [False, True])
@pytest.mark.parametrize("disable_uva", [False, True])
def test_cpu_offload(disable_pin_memory, disable_uva):
    env_vars = {
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": str(int(disable_pin_memory)),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": str(int(disable_uva)),
    }

    args = ["--cpu-offload-gb", "1"]

    # cuda graph only works with UVA offloading
    if disable_uva:
        args.append("--enforce-eager")

    compare_two_settings(
        model="hmellor/tiny-random-LlamaForCausalLM",
        arg1=[],
        arg2=args,
        env1=None,
        env2=env_vars,
    )


@pytest.mark.parametrize(
    ("offload_kwargs", "offloader_type"),
    [
        ({"cpu_offload_gb": 1}, UVAOffloader),
        (
            {
                "offload_group_size": 1,
                "offload_num_in_group": 1,
                "offload_prefetch_step": 1,
            },
            PrefetchOffloader,
        ),
    ],
)
def test_mrv2_weight_offloading(
    vllm_runner, monkeypatch, offload_kwargs, offloader_type
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()
    original_offloader = get_offloader()

    try:
        with vllm_runner(
            "hmellor/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            gpu_memory_utilization=0.02,
            max_model_len=128,
            max_num_seqs=1,
            **offload_kwargs,
        ) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            assert isinstance(model_runner, GPUModelRunner)

            offloader = get_offloader()
            assert isinstance(offloader, offloader_type)
            if isinstance(offloader, UVAOffloader):
                assert offloader.cpu_offload_bytes > 0
            else:
                assert offloader.total_offloaded_bytes > 0
                assert offloader.buffer_pool is not None
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()


def _is_offloaded(p: nn.Parameter) -> bool:
    return p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False)


@pytest.mark.parametrize("disable_uva", [False, True])
def test_tower_weight_offloading(vllm_runner, monkeypatch, disable_uva):
    """`cpu_offload_params` segments must reach towers built outside make_layers.

    Regression test: `wrap_modules` was only called from `make_layers`, so a
    directly-constructed vision tower never reached the offloader and segments
    targeting it silently matched nothing.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if disable_uva:
        monkeypatch.setenv("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "1")
    envs.disable_envs_cache()
    original_offloader = get_offloader()

    try:
        with vllm_runner(
            "Qwen/Qwen3.5-0.8B",
            enforce_eager=True,
            # allocate more vram as Qwen 3.5 has 1.6 GiB of weights
            gpu_memory_utilization=0.3,
            max_model_len=128,
            max_num_seqs=1,
            enable_prefix_caching=False,
            cpu_offload_gb=1,
            cpu_offload_params={"visual"},
        ) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner

            offloader = get_offloader()
            assert isinstance(offloader, UVAOffloader)
            assert offloader.cpu_offload_bytes > 0

            model = model_runner.get_model()
            assert model._tower_model_names
            for name in model._tower_model_names:
                tower = attrgetter(name)(model)
                params = list(tower.parameters())
                assert params
                assert all(_is_offloaded(p) for p in params)
                if disable_uva:
                    # non-UVA fallback: weights live on CPU and are moved
                    # back on first forward
                    assert "forward" in vars(tower)

            # The language model must stay resident.
            assert not any(_is_offloaded(p) for p in model.language_model.parameters())
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="staging is CUDA only")
@pytest.mark.parametrize(
    ("model", "eager", "threshold"),
    # Eager: every step stages. Graphs up to 8 tokens: only the prefill
    # (longer than 8 tokens) stages, decode replays graphs that read in place.
    [
        ("TitanML/tiny-mixtral", True, 1),
        ("TitanML/tiny-mixtral", False, 9),
        ("Adamji/tiny-qwen3-moe-fp8-per-block", False, 9),
    ],
)
def test_uva_expert_staging(vllm_runner, monkeypatch, model, eager, threshold):
    """Staging UVA-offloaded experts gives the same greedy output as reading
    them in place (threshold 0)."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()
    original_offloader = get_offloader()
    prompts = [
        "The capital of France is a city that many people like to visit because",
        "Here is a short story about a robot who learned to paint the sea:",
    ]
    outputs = {}

    try:
        for min_tokens in (0, threshold):
            with vllm_runner(
                model,
                enforce_eager=eager,
                compilation_config=(
                    None if eager else {"cudagraph_capture_sizes": [1, 2, 4, 8]}
                ),
                moe_backend="triton",
                gpu_memory_utilization=0.3,
                max_model_len=256,
                max_num_seqs=2,
                cpu_offload_gb=1,
                cpu_offload_params={"experts"},
                cpu_offload_stage_min_tokens=min_tokens,
            ) as vllm_model:
                offloader = get_offloader()
                assert isinstance(offloader, UVAOffloader)
                assert offloader.cpu_offload_bytes > 0
                assert offloader.expert_stager.min_tokens == min_tokens
                outputs[min_tokens] = vllm_model.generate_greedy(prompts, 16)
                # Buffers exist only if a step was staged.
                assert bool(offloader.expert_stager._buffers) == (min_tokens > 0)
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()

    assert outputs[0] == outputs[threshold]
