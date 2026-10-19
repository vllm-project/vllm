# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PWAL keeps the parameter objects (class and loader attributes) it converts."""

from collections.abc import Callable

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.model_loader.reload.meta import materialize_meta_tensor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="NVFP4/MXFP8 cases select the converted kernels on SM100.",
)

SIZE = 512


def _load(layer: torch.nn.Module) -> None:
    """Stand in for the weight loader: materialize and fill with finite values."""
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.is_meta:
            param = materialize_meta_tensor(param)
            setattr(layer, name, param)
        if param.dtype == torch.uint8:  # packed FP4 values or E8M0 scales
            param.data.copy_(torch.randint(120, 130, param.shape))
        elif param.dtype.is_floating_point:
            param.data.copy_(torch.rand(param.shape) + 0.5)


def _dims(param: torch.Tensor) -> tuple:
    return tuple(
        getattr(param, k, None) for k in ("input_dim", "output_dim", "packed_dim")
    )


def _linear(quant_config: QuantizationConfig, output_size: int = SIZE):
    layer = ReplicatedLinear(SIZE, output_size, bias=False, quant_config=quant_config)
    return layer, layer.quant_method


def _online_linear():
    layer = torch.nn.Module()
    layer.input_size = layer.input_size_per_partition = SIZE
    layer.output_size = layer.output_size_per_partition = SIZE
    layer.params_dtype = torch.bfloat16
    method = Mxfp4OnlineLinearMethod()
    method.create_weights(
        layer,
        input_size_per_partition=SIZE,
        output_partition_sizes=[SIZE],
        input_size=SIZE,
        output_size=SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    return layer, method


def _moe(quant_config: QuantizationConfig | None):
    layer = FusedMoEFactory(
        num_experts=8,
        top_k=2,
        hidden_size=SIZE,
        intermediate_size=SIZE,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.experts",
    ).routed_experts
    return layer, layer.quant_method


def _online_moe():
    layer, _ = _moe(None)
    method = Mxfp4OnlineMoEMethod(moe=layer.moe_config)
    method.create_weights(
        layer=layer,
        num_experts=8,
        hidden_size=SIZE,
        intermediate_size_per_partition=SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    return layer, method


NVFP4 = dict(is_checkpoint_nvfp4_serialized=True, kv_cache_quant_algo=None)

CASES: list[Callable] = [
    lambda: _linear(Fp8Config(weight_block_size=[128, 128])),
    _online_linear,
    lambda: _moe(None),
    lambda: _moe(Fp8Config(weight_block_size=[128, 128])),
    lambda: _moe(ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"})),
    lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
    lambda: _moe(Mxfp4Config()),
    _online_moe,
]
IDS = [
    "fp8-block-linear",
    "online-mxfp4-linear",
    "unquantized-moe",
    "fp8-block-moe",
    "modelopt-mxfp8-moe",
    "modelopt-nvfp4-moe",
    "mxfp4-moe",
    "online-mxfp4-moe",
]


@pytest.mark.parametrize("build", CASES, ids=IDS)
def test_process_weights_after_loading_keeps_param_objects(
    build: Callable, default_vllm_config, dist_init
) -> None:
    default_vllm_config.model_config = ModelConfig()
    with torch.device("cuda"):
        layer, method = build()
        _load(layer)
        before = {
            name: (param, set(param.__dict__), _dims(param))
            for name, param in layer.named_parameters(recurse=False)
        }
        method.process_weights_after_loading(layer)

    after = dict(layer.named_parameters(recurse=False, remove_duplicate=False))
    for name, (param, attrs, dims) in before.items():
        assert after.get(name) is param, name
        assert attrs <= param.__dict__.keys(), name
        # The loader shards by these on reload, so they keep the checkpoint layout.
        assert _dims(param) == dims, name
        assert callable(getattr(param, "weight_loader", None)), name
