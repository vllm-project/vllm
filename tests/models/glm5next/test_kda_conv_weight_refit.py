# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA q|k|v short convs are one fused parameter.

The conv kernels read ``conv1d.weight`` directly, so an RL refit must reach it
through every write path (#55087), and FULL cudagraphs capture its address, so
it must never move. Weight offloading replaces parameter storage after
construction, which the fused parameter must also survive.
"""

import os

import pytest
import torch
from torch import nn

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.glm5next.common import kda
from vllm.platforms import current_platform

CHANNELS, KERNEL = 8, 4
CKPT_CONV_WEIGHTS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")
# Layers 0-3 of the real checkpoint at full width; needs SM90/SM100 (indexer).
PRUNED_MODEL = os.getenv("GLM5_PRUNED_MODEL", "JaredforReal/GLM-5.3-Flash-4L")


def _checkpoint_weights(seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(CHANNELS, 1, KERNEL, generator=gen) for _ in range(3)]


def _load(conv1d: nn.Module, weights: list[torch.Tensor]) -> None:
    for shard_id, weight in enumerate(weights):
        conv1d.weight.weight_loader(conv1d.weight, weight, shard_id)


def _kernel_view(conv1d: nn.Module) -> torch.Tensor:
    return conv1d.weight.view(conv1d.weight.size(0), conv1d.weight.size(-1))


@pytest.fixture
def conv1d(dist_init) -> nn.Module:
    return kda._fused_qkv_conv1d(KERNEL, CHANNELS, 1, 0, prefix="conv1d")


def test_fused_conv_loads_checkpoint_shards_in_qkv_order(conv1d):
    weights = _checkpoint_weights(seed=0)
    _load(conv1d, [w.view(CHANNELS, KERNEL) for w in weights[:1]] + weights[1:])
    assert conv1d.weight.shape == (3 * CHANNELS, 1, KERNEL)
    assert torch.equal(_kernel_view(conv1d), torch.cat(weights).squeeze(1))


def test_refit_updates_fused_conv_without_moving_it(conv1d):
    """Refits through load_weights (RL weight sync) and through the layerwise
    reload (``reload_weights``) both land at the original address."""
    data_ptr = conv1d.weight.data_ptr()
    _load(conv1d, _checkpoint_weights(seed=0))

    refit = _checkpoint_weights(seed=1)
    _load(conv1d, refit)
    assert torch.equal(_kernel_view(conv1d), torch.cat(refit).squeeze(1))

    reload = _checkpoint_weights(seed=2)
    model = nn.Sequential(conv1d)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    _load(conv1d, reload)
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(_kernel_view(conv1d), torch.cat(reload).squeeze(1))
    assert conv1d.weight.data_ptr() == data_ptr


def _conv_weight_ptrs(model: nn.Module) -> list[int]:
    return [
        module.conv1d.weight.data_ptr()
        for module in model.modules()
        if isinstance(module, kda.Glm5NextLinearAttention)
    ]


def _refit_from_checkpoint(worker, conv_scale: float) -> list[int]:
    """RL refit shape: stream the checkpoint through ``reload_weights`` (the
    layerwise reload), with the KDA conv weights scaled."""
    from vllm.model_executor.model_loader import get_model_loader

    runner = worker.model_runner
    model = runner.get_model()
    loader = get_model_loader(runner.load_config)

    def weights():
        for name, tensor in loader.get_all_weights(runner.model_config, model):
            if name.endswith(CKPT_CONV_WEIGHTS):
                tensor = tensor * conv_scale
            yield name, tensor

    worker.reload_weights(weights_iterator=weights())
    return _conv_weight_ptrs(model)


def _refit_kernel_format(worker, conv_scale: float) -> list[int]:
    """Kernel-format refit: ``reload_weights(is_checkpoint_format=False)``
    copies into the parameters without any weight loader."""
    model = worker.model_runner.get_model()
    weights = [
        (name, param.data * conv_scale)
        for name, param in model.named_parameters()
        if name.endswith("self_attn.conv1d.weight")
    ]
    assert weights
    worker.reload_weights(weights_iterator=iter(weights), is_checkpoint_format=False)
    return _conv_weight_ptrs(model)


def _greedy(llm: LLM) -> list[int]:
    params = SamplingParams(temperature=0.0, max_tokens=8)
    output = llm.generate(["The capital of France is"], params)[0]
    return list(output.outputs[0].token_ids)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (9, 10)
    ),
    reason="GLM-5.3-Flash sparse indexer needs SM90/SM100",
)
@pytest.mark.parametrize("cpu_offload_gb", [0, 10])
def test_refit_reaches_cudagraph_decode(
    monkeypatch: pytest.MonkeyPatch, cpu_offload_gb: int
):
    """With CUDA graphs on, decode must follow the refitted conv weights and
    return to the original output once they are restored (#55087), including
    when the KDA layers are offloaded to CPU."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=PRUNED_MODEL,
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        cpu_offload_gb=cpu_offload_gb,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    reference = _greedy(llm)
    ptrs = []
    for refit, scale, same in (
        (_refit_from_checkpoint, 2.0, False),
        (_refit_from_checkpoint, 1.0, True),
        (_refit_kernel_format, 2.0, False),
        (_refit_kernel_format, 0.5, True),
    ):
        ptrs.append(llm.collective_rpc(refit, kwargs={"conv_scale": scale})[0])
        assert (_greedy(llm) == reference) == same, (refit.__name__, scale)
    assert all(p == ptrs[0] for p in ptrs)

    del llm
    cleanup_dist_env_and_memory()
