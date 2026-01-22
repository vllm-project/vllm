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
from vllm.platforms import current_platform
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
        return "value"

    args = inspect.signature(complex_weight_loader).bind(param, loaded_weight)
    num_loaded, ret = get_numel_loaded(complex_weight_loader, args)
    assert num_loaded == 6
    assert ret == "value"


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        (
            "Qwen/Qwen3-0.6B",
            "nm-testing/Qwen3-0.6B-debug-multiply",
            "nm-testing/Qwen3-0.6B-debug-add",
        ),
        (
            "nm-testing/Qwen3-0.6B-W4A16-G128",
            "nm-testing/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "nm-testing/Qwen3-0.6B-debug-add-W4A16-G128",
        ),
        (
            "nm-testing/Qwen3-0.6B-FP8_BLOCK",
            "nm-testing/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "nm-testing/Qwen3-0.6B-debug-add-FP8_BLOCK",
        ),
    ],
)
def test_reload_weights(base_model, mul_model, add_model, tp_size, vllm_runner):
    if cuda_device_count_stateless() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and not current_platform.supports_fp8():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(model_name=base_model, tensor_parallel_size=tp_size) as llm:
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] > 1.1
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] > 1.1

        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] <= 1.1
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] >= 1e10

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        assert llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0] >= 1e10
        assert llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0] <= 1.1


@pytest.mark.parametrize(
    "tp_size",
    [
        2,
    ],
)
def test_reload_expert_parallelism(tp_size, vllm_runner):
    if cuda_device_count_stateless() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    base_model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    prompts = ["The capital of France is Paris"]

    with vllm_runner(
        model_name=base_model,
        enable_expert_parallel=True,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
    ) as llm:
        exp_perp = llm.generate_prompt_perplexity(prompts)
        llm.collective_rpc("reload_weights")
        reload_perp = llm.generate_prompt_perplexity(prompts)

        assert reload_perp == exp_perp
