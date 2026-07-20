# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_MODULE = "vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep"


@pytest.fixture
def nccl_ep_module(monkeypatch: pytest.MonkeyPatch):
    nccl = types.ModuleType("nccl")
    nccl.__path__ = []
    nccl_core = types.ModuleType("nccl.core")
    nccl_ep = types.ModuleType("nccl.ep")

    for name, value in {
        "BFLOAT16": 0,
        "FLOAT16": 1,
        "FLOAT32": 2,
        "INT64": 3,
        "INT32": 4,
        "UINT8": 5,
    }.items():
        setattr(nccl_core, name, value)
    nccl_ep.Group = object
    nccl_ep.Tensor = object
    nccl.core = nccl_core
    nccl.ep = nccl_ep

    monkeypatch.setitem(sys.modules, "nccl", nccl)
    monkeypatch.setitem(sys.modules, "nccl.core", nccl_core)
    monkeypatch.setitem(sys.modules, "nccl.ep", nccl_ep)
    sys.modules.pop(_MODULE, None)
    return importlib.import_module(_MODULE)


def _block_quant_config():
    return SimpleNamespace(
        is_block_quantized=True,
        quant_dtype=torch.float8_e4m3fn,
        a1_scale=torch.tensor(0.5),
        a1_gscale=None,
        per_act_token_quant=False,
        block_shape=[128, 128],
    )


def _standard_prepare_finalize(module):
    return module.NcclEPStandardPrepareAndFinalize(
        ep_group=object(),
        num_dispatchers=2,
        dp_size=2,
        rank_expert_offset=0,
        num_experts=4,
        num_topk=2,
        use_fp8_dispatch=True,
    )


def test_standard_dispatches_block_quantized_input_as_bfloat16(
    nccl_ep_module, monkeypatch: pytest.MonkeyPatch
):
    prepare_finalize = _standard_prepare_finalize(nccl_ep_module)
    dispatched = {}

    def fake_dispatch(**kwargs):
        dispatched.update(kwargs)
        return lambda: None

    monkeypatch.setattr(prepare_finalize, "_do_dispatch", fake_dispatch)
    monkeypatch.setattr(
        nccl_ep_module,
        "moe_kernel_quantize_input",
        lambda *args, **kwargs: pytest.fail("input was quantized before dispatch"),
    )

    a1 = torch.empty((4, 128), dtype=torch.bfloat16)
    prepare_finalize.prepare_async(
        a1=a1,
        topk_weights=torch.empty((4, 2), dtype=torch.float32),
        topk_ids=torch.zeros((4, 2), dtype=torch.int64),
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=_block_quant_config(),
    )

    assert dispatched["tokens"] is a1
    assert dispatched["tokens"].dtype == torch.bfloat16


def test_standard_quantizes_block_quantized_input_after_dispatch(
    nccl_ep_module, monkeypatch: pytest.MonkeyPatch
):
    prepare_finalize = _standard_prepare_finalize(nccl_ep_module)
    quant_config = _block_quant_config()
    recv_x = torch.empty((4, 128), dtype=torch.bfloat16)
    quantized = torch.empty_like(recv_x, dtype=torch.float8_e4m3fn)
    scale = torch.ones((1, 1), dtype=torch.float32)
    calls = []

    def fake_quantize(*args, **kwargs):
        calls.append((args, kwargs))
        return quantized, scale

    monkeypatch.setattr(nccl_ep_module, "moe_kernel_quantize_input", fake_quantize)

    expert_x, expert_x_scale, *_ = prepare_finalize._receiver(
        recv_x=recv_x,
        recv_topk_idx=torch.zeros((4, 2), dtype=torch.int64),
        recv_topk_weights=torch.empty((4, 2), dtype=torch.float32),
        num_experts=4,
        a1_scale=quant_config.a1_scale,
        quant_config=quant_config,
        defer_input_quant=False,
    )

    assert len(calls) == 1
    assert calls[0][0][0] is recv_x
    assert expert_x is quantized
    assert expert_x_scale is scale
