# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from operator import attrgetter
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import maybe_offload_embeddings
from vllm.model_executor.offloader import (
    PrefetchOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from ..utils import compare_two_settings


def _model_with_direct_embedding() -> tuple[nn.Module, VocabParallelEmbedding]:
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    embedding = VocabParallelEmbedding(16, 8, disable_tp=True)
    model.language_model.model.embed_tokens = embedding
    return model, embedding


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_uva_available(),
    reason="UVA is not available.",
)
def test_direct_embedding_offload_preserves_prefix(monkeypatch, default_vllm_config):
    model, embedding = _model_with_direct_embedding()
    model.cuda()
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.should_pin_memory", lambda: False
    )
    offloader = UVAOffloader(
        cpu_offload_max_bytes=embedding.weight.nbytes,
        cpu_offload_params={"outer.language_model.model.embed_tokens"},
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.get_offloader", lambda: offloader
    )

    maybe_offload_embeddings(model, prefix="outer")

    assert offloader.cpu_offload_bytes == embedding.weight.nbytes
    assert getattr(embedding.weight, "_vllm_is_uva_offloaded", False)
    assert embedding.weight.device.type == "cuda"


def test_direct_embedding_offload_skips_unsupported_offloader(
    monkeypatch, default_vllm_config
):
    model, _ = _model_with_direct_embedding()
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.is_uva_available", lambda: False
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.uva.should_pin_memory", lambda: False
    )
    offloader = UVAOffloader(cpu_offload_max_bytes=1024)
    wrap_modules = Mock()
    monkeypatch.setattr(offloader, "wrap_modules", wrap_modules)
    monkeypatch.setattr(
        "vllm.model_executor.offloader.get_offloader", lambda: offloader
    )

    maybe_offload_embeddings(model)

    wrap_modules.assert_not_called()


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
