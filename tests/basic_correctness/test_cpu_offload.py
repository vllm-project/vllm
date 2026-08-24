# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.offloader import (
    NoopOffloader,
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ..utils import compare_two_settings


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


class _FakeVLM(nn.Module):
    """A model shaped like a VLM: a tower built directly, plus a layer stack.

    Only `language_model.layers` goes through `make_layers` in a real model,
    so only those reach `wrap_modules`. `visual` is the blind spot.
    """

    def __init__(self, tower_dim: int = 64):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList(
            [nn.Linear(tower_dim, tower_dim) for _ in range(2)]
        )
        self.visual.merger = nn.Linear(tower_dim, tower_dim)

        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])


def _make_offloader(max_bytes: int, params: set[str] | None = None) -> UVAOffloader:
    return UVAOffloader(cpu_offload_max_bytes=max_bytes, cpu_offload_params=params)


def _is_offloaded(p: nn.Parameter) -> bool:
    return p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_model_reaches_directly_constructed_tower():
    """`--cpu-offload-params visual` must offload a tower built outside make_layers.

    This is the regression: the tower never reaches `wrap_modules`, so before
    this fix the segment matched nothing and nothing was offloaded.
    """
    model = _FakeVLM().cuda()
    offloader = _make_offloader(1024**3, {"visual"})

    offloader.offload_model(model)

    assert offloader.cpu_offload_bytes > 0
    assert all(_is_offloaded(p) for p in model.visual.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_model_leaves_unmatched_params_resident():
    """A segment must not drag in parameters outside it."""
    model = _FakeVLM().cuda()
    offloader = _make_offloader(1024**3, {"visual"})

    offloader.offload_model(model)

    assert not any(_is_offloaded(p) for p in model.language_model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_model_skips_params_already_offloaded_by_wrap_modules():
    """The sweep must not double-count what `make_layers` already offloaded."""
    model = _FakeVLM().cuda()
    offloader = _make_offloader(1024**3)

    offloader.wrap_modules(m for m in model.language_model.layers)
    after_wrap = offloader.cpu_offload_bytes
    assert after_wrap > 0

    offloader.offload_model(model)

    # The tower adds bytes; the already-offloaded layers must not be recounted.
    layer_bytes = sum(
        p.numel() * p.element_size() for p in model.language_model.parameters()
    )
    tower_bytes = sum(p.numel() * p.element_size() for p in model.visual.parameters())
    assert after_wrap == layer_bytes
    assert offloader.cpu_offload_bytes == layer_bytes + tower_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_model_respects_budget_consumed_by_layer_stack():
    """The layer stack has first claim on the budget.

    With no explicit segments and a budget the layers alone exhaust, the sweep
    finds nothing left to do -- so text-only-shaped configs are unaffected.
    """
    model = _FakeVLM().cuda()
    layer_bytes = sum(
        p.numel() * p.element_size() for p in model.language_model.parameters()
    )
    offloader = _make_offloader(layer_bytes)

    offloader.wrap_modules(m for m in model.language_model.layers)
    offloader.offload_model(model)

    assert offloader.cpu_offload_bytes <= layer_bytes
    assert not any(_is_offloaded(p) for p in model.visual.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_offload_model_fallback_hook_restores_weights_for_forward():
    """Without UVA, the leaf owning the offloaded weight must onload on forward."""
    model = _FakeVLM().cuda()
    offloader = _make_offloader(1024**3, {"visual"})
    offloader.uva_offloading = False

    merger = model.visual.merger
    expected = merger(torch.ones(1, 64, device="cuda"))

    offloader.offload_model(model)

    assert merger.weight.device.type == "cpu"
    # The hook is installed on the leaf that owns the parameter, which is the
    # module whose forward actually runs.
    assert "forward" in vars(merger)

    got = merger(torch.ones(1, 64, device="cuda"))
    torch.testing.assert_close(got, expected)


def test_offload_model_is_a_noop_for_other_backends():
    """Only UVA implements the sweep.

    `PrefetchOffloader.wrap_modules` asserts it is called exactly once and
    schedules over a circular layer stack, which a tower is not.
    """
    model = _FakeVLM()
    before = {name: p.data_ptr() for name, p in model.named_parameters()}

    NoopOffloader().offload_model(model)

    prefetch = PrefetchOffloader.__new__(PrefetchOffloader)
    prefetch.offload_model(model)

    assert {name: p.data_ptr() for name, p in model.named_parameters()} == before
    assert not any(
        getattr(p, "_vllm_is_uva_offloaded", False) for p in model.parameters()
    )
