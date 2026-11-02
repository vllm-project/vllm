# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4/MXFP8 scale parameters are uint8 and must hold the checkpoint's E8M0
bytes whether the scales arrive as uint8 or as float8_e8m0fnu: a by-value
copy would turn 2^(b-127) into an integer. ``copy_weight_`` keeps the bytes;
these tests go through the loaders that MX scales take to it."""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers.fused_moe import (
    unquantized_fused_moe_method as unquantized_module,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w8a8_mxfp8 as ct_mxfp8_moe,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import GroupQuantScaleParameter
from vllm.model_executor.utils import set_weight_attrs

ROWS, COLS = 16, 32
SRC_DTYPES = [torch.uint8, torch.float8_e8m0fnu]


def _scale_bytes(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Every byte value, including 0x00 (2^-127), 0x7F (1.0), 0xFE (2^127)
    and 0xFF (NaN), shifted by ``seed`` so shards differ."""
    flat = (torch.arange(math.prod(shape)) + seed) % 256
    return flat.to(torch.uint8).view(shape)


def _no_loader(*args, **kwargs):
    raise AssertionError("not used")


def _vllm_parameter_row(dst, src):
    """Loaders of the vLLMParameter classes (CT, ModelOpt, Quark linear):
    this rank's columns of a row-parallel scale, a non-contiguous source."""
    param = GroupQuantScaleParameter(
        data=dst, input_dim=1, output_dim=0, weight_loader=_no_loader
    )
    param.load_row_parallel_weight(src)
    return param.data, src.narrow(1, COLS, COLS)


def _linear_v1_column(dst, src):
    """``weight_loader`` of linear layers without loader v2 (INC MXFP4/MXFP8
    linear): this rank's rows of a column-parallel scale."""
    layer = linear_module.ColumnParallelLinear(
        COLS * 32, ROWS * 2, bias=False, params_dtype=torch.bfloat16
    )
    param = torch.nn.Parameter(dst, requires_grad=False)
    set_weight_attrs(param, {"output_dim": 0})
    layer.weight_loader(param, src)
    return param.data, src.narrow(0, ROWS, ROWS)


def _default_loader(dst, src):
    param = torch.nn.Parameter(dst, requires_grad=False)
    default_weight_loader(param, src)
    return param.data, src


# name -> (source shape at TP=2, load into dst; returns (loaded, expected bytes))
LOADERS = {
    "vllm_parameter_row": ((ROWS, COLS * 2), _vllm_parameter_row),
    "linear_v1_column": ((ROWS * 2, COLS), _linear_v1_column),
    "default_weight_loader": ((ROWS, COLS), _default_loader),
}


@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
@pytest.mark.parametrize("loader", list(LOADERS))
def test_loader_keeps_e8m0_bytes(default_vllm_config, monkeypatch, loader, src_dtype):
    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    src_shape, load = LOADERS[loader]
    src = _scale_bytes(src_shape, seed=7)
    dst = torch.full((ROWS, COLS), 0xAB, dtype=torch.uint8)
    loaded, expected = load(dst, src.view(src_dtype))
    assert loaded.dtype == torch.uint8
    assert torch.equal(loaded, expected.view(torch.uint8))


@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
def test_moe_scale_keeps_e8m0_bytes(default_vllm_config, monkeypatch, src_dtype):
    """CT W8A8 MXFP8 MoE scales through the real RoutedExperts.weight_loader."""
    num_experts, hidden, inter = 2, 256, 128
    monkeypatch.setattr(
        unquantized_module, "select_unquantized_moe_backend", lambda **kw: (None, None)
    )
    monkeypatch.setattr(
        ct_mxfp8_moe, "select_mxfp8_moe_backend", lambda **kw: (None, None)
    )
    config = make_dummy_moe_config(
        num_experts=num_experts,
        hidden_dim=hidden,
        intermediate_size=inter,
        experts_per_token=num_experts,
    )
    layer = RoutedExperts(
        "model.layers.0.mlp.experts",
        torch.bfloat16,
        config,
        None,
        expert_map_manager=SimpleNamespace(
            local_num_experts=num_experts,
            num_fused_shared_experts=0,
            placement_strategy="linear",
            expert_map=None,
            expert_mask=None,
            routing_tables=None,
            map_global_to_local=lambda index: index,
        ),
    )
    method = ct_mxfp8_moe.CompressedTensorsW8A8Mxfp8MoEMethod(layer.moe_config)
    method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    # RoutedExperts.weight_loader dispatches on the quant method's class name.
    del layer.quant_method  # the unquantized method is a registered submodule
    layer.quant_method = method
    w13, w2 = layer.w13_weight_scale, layer.w2_weight_scale
    for p in (w13, w2):
        assert p.dtype == torch.uint8
        p.data.fill_(0xAB)
    for expert in range(num_experts):
        srcs = {
            "w1": _scale_bytes((inter, hidden // 32), 10 * expert + 1),
            "w3": _scale_bytes((inter, hidden // 32), 10 * expert + 3),
            "w2": _scale_bytes((hidden, inter // 32), 10 * expert + 2),
        }
        for shard_id, src in srcs.items():
            p = w2 if shard_id == "w2" else w13
            name = f"experts.{expert}.{shard_id}.weight_scale"
            assert p.weight_loader(
                p, src.view(src_dtype), name, shard_id, expert, return_success=True
            )
        assert torch.equal(w13.data[expert], torch.cat([srcs["w1"], srcs["w3"]]))
        assert torch.equal(w2.data[expert], srcs["w2"])
