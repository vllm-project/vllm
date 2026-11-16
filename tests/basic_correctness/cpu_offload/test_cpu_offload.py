# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from operator import attrgetter

import pytest
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
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
        model="vllm-project/tiny-random-LlamaForCausalLM",
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
            "vllm-project/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            gpu_memory_utilization=0.03,
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


@pytest.mark.parametrize(
    "model", ["vllm-project/tiny-random-LlamaForCausalLM", "TitanML/tiny-mixtral"]
)
def test_transformers_backend_weight_offloading(vllm_runner, monkeypatch, model):
    """Transformers backend decoder layers (including MoE experts) are offloaded as
    each one is built."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()
    original_offloader = get_offloader()
    original_wrap_modules = UVAOffloader.wrap_modules

    # Device memory as each layer reaches the offloader, and after all are offloaded
    on_arrival: list[int] = []
    layer_bytes: list[int] = []
    after_offload: list[int] = []

    def wrap_modules(self, modules_generator, prefix=""):
        def spy():
            for module in modules_generator:
                on_arrival.append(torch.accelerator.memory_allocated())
                layer_bytes.append(
                    sum(p.numel() * p.element_size() for p in module.parameters())
                )
                yield module

        modules = original_wrap_modules(self, spy(), prefix)
        after_offload.append(torch.accelerator.memory_allocated())
        return modules

    monkeypatch.setattr(UVAOffloader, "wrap_modules", wrap_modules)
    num_layers = 8

    try:
        with vllm_runner(
            model,
            model_impl="transformers",
            # Random weights, so the layers can be made large enough to measure
            load_format="dummy",
            hf_overrides={
                "num_hidden_layers": num_layers,
                "hidden_size": 1024,
                "intermediate_size": 4096,
            },
            enforce_eager=True,
            gpu_memory_utilization=0.1,
            max_model_len=128,
            max_num_seqs=1,
            # More than the decoder layers, so every one of them is offloaded
            cpu_offload_gb=4,
        ) as vllm_model:
            engine_core = vllm_model.llm.llm_engine.engine_core.engine_core
            model_runner = engine_core.model_executor.driver_worker.worker.model_runner
            layers = model_runner.get_model().model.get_decoder().layers
            assert all(_is_offloaded(p) for p in layers.parameters())
    finally:
        set_offloader(original_offloader)
        envs.disable_envs_cache()

    assert len(on_arrival) == num_layers, on_arrival
    assert len(after_offload) == 1, after_offload
    # About one layer is on the device at a time, not the whole decoder
    peak_above_resident = max(on_arrival) - after_offload[0]
    assert peak_above_resident <= 1.5 * max(layer_bytes), (
        peak_above_resident,
        layer_bytes,
    )


PREFETCH_ARGS = [
    "--offload-group-size",
    "1",
    "--offload-num-in-group",
    "1",
    "--offload-prefetch-step",
    "1",
]


@pytest.mark.parametrize(
    ("offload_args", "env"),
    [
        pytest.param(["--cpu-offload-gb", "1"], None, id="uva"),
        pytest.param(
            # cuda graph only works with UVA offloading
            ["--cpu-offload-gb", "1", "--enforce-eager"],
            {"VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": "1"},
            id="no-uva",
        ),
        pytest.param(PREFETCH_ARGS, None, id="prefetch"),
    ],
)
def test_transformers_backend_cpu_offload_outputs(offload_args, env):
    """The Transformers backend gives the same outputs with weights offloaded."""
    compare_two_settings(
        model="vllm-project/tiny-random-LlamaForCausalLM",
        arg1=["--model-impl", "transformers"],
        arg2=["--model-impl", "transformers", *offload_args],
        env1=None,
        env2=env,
    )
