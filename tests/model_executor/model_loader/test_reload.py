# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from itertools import chain

import pytest
import torch

from vllm.model_executor.model_loader.reload.helpers import get_layer_params_buffers
from vllm.model_executor.model_loader.reload.meta import (
    get_numel_loaded,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.utils.torch_utils import cuda_device_count_stateless


def test_move_metatensors():
    tensor = torch.empty((1, 2, 3))
    meta_tensor = to_meta_tensor(tensor)
    materialized_tensor = materialize_meta_tensor(meta_tensor)

    assert meta_tensor.device.type == "meta"
    assert tensor.device == materialized_tensor.device

    assert tensor.dtype == meta_tensor.dtype == materialized_tensor.dtype
    assert tensor.shape == meta_tensor.shape == materialized_tensor.shape
    assert tensor.__class__ == meta_tensor.__class__ == materialized_tensor.__class__
    assert tensor.__dict__ == meta_tensor.__dict__ == materialized_tensor.__dict__


def test_materialize_layer():
    layer = torch.nn.Linear(2, 3)
    params, buffers = get_layer_params_buffers(layer)
    params = {name: to_meta_tensor(param) for name, param in params.items()}
    buffers = {name: to_meta_tensor(buffer) for name, buffer in buffers.items()}
    info = LayerReloadingInfo(restore_metadata=(params, buffers))

    restore_layer_on_meta(layer, info)
    for name, tensor in chain(params.items(), buffers.items()):
        meta_tensor = getattr(layer, name)
        assert tensor.dtype == meta_tensor.dtype
        assert tensor.shape == meta_tensor.shape
        assert tensor.__class__ == meta_tensor.__class__
        assert tensor.__dict__ == meta_tensor.__dict__

    materialize_layer(layer)
    for name, tensor in chain(params.items(), buffers.items()):
        materialized_tensor = getattr(layer, name)
        assert tensor.dtype == materialized_tensor.dtype
        assert tensor.shape == materialized_tensor.shape
        assert tensor.__class__ == materialized_tensor.__class__
        assert tensor.__dict__ == materialized_tensor.__dict__


def test_get_numel_loaded():
    param = torch.empty(10, device="meta")
    loaded_weight = torch.empty(10)

    def complex_weight_loader(param, loaded_weight):
        param[:3] = loaded_weight[:3]
        param[5:8] = loaded_weight[5:8]

    args = inspect.signature(complex_weight_loader).bind(param, loaded_weight)
    assert get_numel_loaded(complex_weight_loader, args) == 6


@pytest.mark.parametrize("tp_size", [1, 2])
def test_reload_weights(tp_size, vllm_runner):
    if cuda_device_count_stateless() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    base_model = "Qwen/Qwen3-0.6B"
    mul_model = "nm-testing/Qwen3-0.6B-debug-multiply"
    add_model = "nm-testing/Qwen3-0.6B-debug-add"

    with vllm_runner(model_name=base_model, tensor_parallel_size=tp_size) as llm:
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] >= 3.5
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] >= 8.5

        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] <= 1.1
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] >= 1e10

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] >= 1e10
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] <= 1.1


def test_reload_quantized():
    pass


def test_reload_online_quantized():
    pass
