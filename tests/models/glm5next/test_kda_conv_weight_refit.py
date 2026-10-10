# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA q|k|v conv weights share one merged buffer.

The conv kernels read ``_merged_conv_weight`` while weight loading writes the
three ``ColumnParallelLinear`` weights, so an RL refit must reach the merged
weight through every write path (#55087). FULL cudagraphs capture the merged
weight's address, so it must also never move.
"""

import os

import pytest
import torch
from torch import nn

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.glm5next.common import kda
from vllm.platforms import current_platform

CHANNELS, KERNEL = 8, 4
CONV_WEIGHTS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")
# Layers 0-3 of the real checkpoint at full width; needs SM90/SM100 (indexer).
PRUNED_MODEL = os.getenv("GLM5_PRUNED_MODEL", "JaredforReal/GLM-5.3-Flash-4L")


class _ConvLayer(nn.Module):
    """The conv part of ``Glm5NextLinearAttention``, wired like the layer."""

    def __init__(self) -> None:
        super().__init__()
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            ColumnParallelLinear(
                KERNEL,
                CHANNELS,
                bias=False,
                params_dtype=torch.float32,
                prefix=f"{name}_conv1d",
            )
            for name in "qkv"
        )
        self.register_buffer(
            "_merged_conv_weight",
            kda._merge_conv_weights(self.q_conv1d, self.k_conv1d, self.v_conv1d),
            persistent=False,
        )

    @property
    def convs(self) -> tuple[ColumnParallelLinear, ...]:
        return (self.q_conv1d, self.k_conv1d, self.v_conv1d)


def _checkpoint_weights(seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(CHANNELS, 1, KERNEL, generator=gen) for _ in range(3)]


def _merged(weights: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([w.view(CHANNELS, KERNEL) for w in weights])


@pytest.fixture
def layer(dist_init) -> _ConvLayer:
    return _ConvLayer()


def test_conv_weights_alias_the_merged_buffer(layer):
    """Loading each conv weight through its stock loader fills the merged
    weight in place; the kernels' view keeps the checkpoint channel order."""
    merged = layer._merged_conv_weight
    assert merged.shape == (3 * CHANNELS, KERNEL) and merged.is_contiguous()
    for conv in layer.convs:
        assert conv.weight.shape == (CHANNELS, 1, KERNEL)
        assert conv.weight.untyped_storage().data_ptr() == (
            merged.untyped_storage().data_ptr()
        )
    assert "_merged_conv_weight" not in layer.state_dict()

    weights = _checkpoint_weights(seed=0)
    for conv, weight in zip(layer.convs, weights):
        conv.weight.weight_loader(conv.weight, weight)
    assert torch.equal(merged, _merged(weights))


def test_refit_updates_merged_weight_without_moving_it(layer):
    """A refit rewrites the conv weights through load_weights (RL weight sync)
    and through the layerwise reload (``reload_weights``); both must show up
    in the merged weight at its original address."""
    merged = layer._merged_conv_weight
    data_ptr = merged.data_ptr()
    for conv, weight in zip(layer.convs, _checkpoint_weights(seed=0)):
        conv.weight.weight_loader(conv.weight, weight)

    refit = _checkpoint_weights(seed=1)
    for conv, weight in zip(layer.convs, refit):
        conv.weight.weight_loader(conv.weight, weight)
    assert torch.equal(merged, _merged(refit))
    assert merged.data_ptr() == data_ptr

    reload = _checkpoint_weights(seed=2)
    model = nn.Sequential(layer)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    for conv, weight in zip(layer.convs, reload):
        conv.weight.weight_loader(conv.weight, weight)
    finalize_layerwise_reload(model, model_config=None)
    assert layer._merged_conv_weight is merged
    assert torch.equal(merged, _merged(reload))
    assert merged.data_ptr() == data_ptr
    assert "_merged_conv_weight" in layer._non_persistent_buffers_set
    for conv in layer.convs:
        assert conv.weight.untyped_storage().data_ptr() == (
            merged.untyped_storage().data_ptr()
        )


def _merged_conv_weight_ptrs(model: nn.Module) -> list[int]:
    return [
        module._merged_conv_weight.data_ptr()
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
            if name.endswith(CONV_WEIGHTS):
                tensor = tensor * conv_scale
            yield name, tensor

    worker.reload_weights(weights_iterator=weights())
    return _merged_conv_weight_ptrs(model)


def _refit_kernel_format(worker, conv_scale: float) -> list[int]:
    """Kernel-format refit: ``reload_weights(is_checkpoint_format=False)``
    copies into the parameters without any weight loader."""
    model = worker.model_runner.get_model()
    weights = [
        (name, param.data * conv_scale)
        for name, param in model.named_parameters()
        if name.endswith(CONV_WEIGHTS)
    ]
    worker.reload_weights(weights_iterator=iter(weights), is_checkpoint_format=False)
    return _merged_conv_weight_ptrs(model)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (9, 10)
    ),
    reason="GLM-5.3-Flash sparse indexer needs SM90/SM100",
)
def test_refit_reaches_cudagraph_decode(monkeypatch: pytest.MonkeyPatch):
    """With CUDA graphs on, decode must follow the refitted conv weights and
    return to the original output once they are restored (#55087)."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=PRUNED_MODEL,
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=8)

    def generate() -> list[int]:
        return list(
            llm.generate(["The capital of France is"], params)[0].outputs[0].token_ids
        )

    reference = generate()
    ptrs = []
    for refit, scale, same in (
        (_refit_from_checkpoint, 2.0, False),
        (_refit_from_checkpoint, 1.0, True),
        (_refit_kernel_format, 2.0, False),
        (_refit_kernel_format, 0.5, True),
    ):
        ptrs.append(llm.collective_rpc(refit, kwargs={"conv_scale": scale})[0])
        assert (generate() == reference) == same, (refit.__name__, scale)
    assert all(p == ptrs[0] for p in ptrs)
