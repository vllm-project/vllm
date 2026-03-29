# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import inspect
from weakref import WeakKeyDictionary, ref

import pytest
import torch

from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
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


def test_reload_lifecycle():
    layer = torch.nn.Linear(2, 3)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    restore_layer_on_meta(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        meta_tensor = getattr(layer, name)
        assert tensor.dtype == meta_tensor.dtype
        assert tensor.shape == meta_tensor.shape
        assert tensor.__class__ == meta_tensor.__class__
        assert tensor.__dict__ == meta_tensor.__dict__

    materialize_layer(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        materialized_tensor = getattr(layer, name)
        assert tensor.dtype == materialized_tensor.dtype
        assert tensor.shape == materialized_tensor.shape
        assert tensor.__class__ == materialized_tensor.__class__
        assert tensor.__dict__ == materialized_tensor.__dict__


def test_model_cleanup(dist_init, default_vllm_config):
    layer = QKVParallelLinear(2, 3, 4)
    assert layer.weight.weight_loader.__self__ is layer
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    mock_info_dict: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
        WeakKeyDictionary()
    )
    mock_info_dict[layer] = info
    layer_ref = ref(layer)

    del layer
    gc.collect()

    assert layer_ref() is None
    assert len(mock_info_dict) == 0


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


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-add-FP8_BLOCK",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-add-W4A16-G128",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-add-NVFP4A16",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_reload_weights(base_model, mul_model, add_model, tp_size, vllm_runner):
    if cuda_device_count_stateless() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and not current_platform.supports_fp8():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=10,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model,quantization",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "fp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8",
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "mxfp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "mxfp8",
            marks=[
                pytest.mark.slow_test,
                pytest.mark.xfail(reason="mxfp4 & mla is not supported yet"),
            ],
        ),
    ],
)
def test_online_quantize_reload(
    base_model, mul_model, add_model, quantization, tp_size, vllm_runner
):
    if cuda_device_count_stateless() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if quantization == "fp8" and not current_platform.supports_fp8():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=10,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp
