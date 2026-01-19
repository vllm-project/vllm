# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from itertools import chain

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


def test_reload_weights():
    pass


def test_reload_quantized():
    pass


def test_reload_online_quantized():
    pass
