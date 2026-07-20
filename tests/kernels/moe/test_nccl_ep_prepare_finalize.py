# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_MODULE = "vllm.model_executor.layers.fused_moe.prepare_finalize.nccl_ep"


class _Binding:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class _Handle:
    def dispatch(self, inputs, outputs, **kwargs):
        self.dispatch_inputs = inputs
        self.dispatch_outputs = outputs
        self.dispatch_kwargs = kwargs

    def complete(self, **kwargs):
        self.complete_kwargs = kwargs


class _Group:
    def __init__(self):
        self.handle = _Handle()

    def create_handle(self, *args, **kwargs):
        self.create_handle_args = args
        self.create_handle_kwargs = kwargs
        return self.handle


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
        "FLOAT8E4M3": 10,
    }.items():
        setattr(nccl_core, name, value)
    for name in (
        "Tensor",
        "DispatchInputs",
        "DispatchOutputs",
        "DispatchConfig",
        "HandleConfig",
    ):
        setattr(nccl_ep, name, _Binding)
    nccl_ep.Group = _Group
    nccl_ep.Layout = SimpleNamespace(FLAT=0)
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


def _standard_prepare_finalize(module, group=None):
    return module.NcclEPStandardPrepareAndFinalize(
        ep_group=group or _Group(),
        num_dispatchers=2,
        dp_size=4,
        rank_expert_offset=0,
        num_experts=4,
        num_topk=2,
        use_fp8_dispatch=True,
    )


def test_standard_maps_torch_fp8_to_nccl_fp8(nccl_ep_module):
    assert (
        nccl_ep_module._TORCH_TO_NCCL_DTYPE[torch.float8_e4m3fn]
        == nccl_ep_module.nccl_core.FLOAT8E4M3
    )


def test_standard_passes_quantized_tokens_and_scales_to_dispatch(
    nccl_ep_module, monkeypatch: pytest.MonkeyPatch
):
    prepare_finalize = _standard_prepare_finalize(nccl_ep_module)
    dispatched = {}
    quantized = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    scales = torch.ones((4, 1), dtype=torch.float32)

    def fake_dispatch(**kwargs):
        dispatched.update(kwargs)
        return lambda: None

    monkeypatch.setattr(prepare_finalize, "_do_dispatch", fake_dispatch)
    quantize_args = {}

    def fake_quantize(x, group_size, **kwargs):
        quantize_args["x"] = x
        quantize_args["group_size"] = group_size
        return quantized, scales

    monkeypatch.setattr(nccl_ep_module, "per_token_group_quant_fp8", fake_quantize)

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

    assert dispatched["tokens"] is quantized
    assert dispatched["token_scales"].data_ptr() == scales.data_ptr()
    assert dispatched["token_scales"].shape == (4, 1)
    assert quantize_args == {"x": a1, "group_size": 128}


def test_standard_dispatch_carries_fp8_scales_end_to_end(
    nccl_ep_module, monkeypatch: pytest.MonkeyPatch
):
    group = _Group()
    prepare_finalize = _standard_prepare_finalize(nccl_ep_module, group)
    received = {}

    monkeypatch.setattr(nccl_ep_module, "_get_cuda_stream", lambda: 0)
    for name in (
        "dbo_yield_and_switch_from_compute_to_comm",
        "dbo_switch_to_compute_sync",
    ):
        monkeypatch.setattr(nccl_ep_module, name, lambda: None)

    def fake_receiver(recv_x, recv_x_scale, *args, **kwargs):
        received["tokens"] = recv_x
        received["scales"] = recv_x_scale
        return None

    monkeypatch.setattr(prepare_finalize, "_receiver", fake_receiver)

    receiver = prepare_finalize._do_dispatch(
        tokens=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
        token_scales=torch.ones((4, 1), dtype=torch.float32),
        rank_topk_ids=torch.zeros((4, 2), dtype=torch.int64),
        rank_topk_weights=torch.empty((4, 2), dtype=torch.float32),
        num_experts=4,
        a1_scale=None,
        quant_config=_block_quant_config(),
        defer_input_quant=False,
    )

    assert group.handle.dispatch_inputs.scales is not None
    assert group.handle.dispatch_inputs.scales.dtype == 2
    assert group.handle.dispatch_inputs.scales.shape == (4, 1)
    assert group.handle.dispatch_outputs.scales is not None
    assert group.handle.dispatch_outputs.scales.dtype == 2
    assert group.handle.dispatch_outputs.tokens.shape == (16, 128)
    assert group.handle.dispatch_outputs.topk_weights.shape == (16, 2)
    assert group.handle.dispatch_outputs.topk_idx.shape == (16, 2)
    assert group.handle.dispatch_outputs.scales.shape == (16, 1)

    receiver()
    assert received["scales"] is not None
    assert received["scales"].shape == (16, 1)


def test_standard_expands_transport_scales_for_block_quantization(nccl_ep_module):
    prepare_finalize = _standard_prepare_finalize(nccl_ep_module)
    recv_x = torch.empty((8, 6144), dtype=torch.float8_e4m3fn)
    transport_scales = torch.ones((8, 1), dtype=torch.float32)
    recv_topk_idx = torch.full((8, 2), -1, dtype=torch.int64)

    _, expert_scales, _, _, _ = prepare_finalize._receiver(
        recv_x=recv_x,
        recv_x_scale=transport_scales,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=torch.empty((8, 2), dtype=torch.float32),
        num_experts=4,
        a1_scale=None,
        quant_config=_block_quant_config(),
        defer_input_quant=False,
    )

    assert expert_scales is not None
    assert expert_scales.shape == (8, 48)
    assert expert_scales.is_contiguous()
    torch.testing.assert_close(expert_scales[:, :1], transport_scales)
