# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from math import prod
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.nvidia import kda
from vllm.v1.worker import workspace


@pytest.fixture
def make_kda_layer(monkeypatch):
    def init_base(self, config, vllm_config, prefix):
        nn.Module.__init__(self)
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = None
        self.tp_size, self.tp_rank = 2, 0
        self.hidden_size, self.num_spec = 8, 0

    def linear(*args, **kwargs):
        layer = nn.Linear(1, 1, bias=False)
        layer.weight.weight_loader = None
        return layer

    monkeypatch.setattr(kda.GatedDeltaNetAttention, "__init__", init_base)
    monkeypatch.setattr(
        kda.KimiK3DeltaAttention,
        "get_state_dtype",
        lambda self: (torch.bfloat16, torch.float32),
    )
    for name in (
        "_KimiGDNMergedColumnParallelLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "FusedRMSNormGated",
    ):
        monkeypatch.setattr(kda, name, linear)
    monkeypatch.setattr(kda, "resolve_kda_decode_backend", lambda *args: "triton")
    monkeypatch.setattr(kda, "resolve_kda_spec_decode_backend", lambda *args: "native")
    monkeypatch.setattr(
        kda, "resolve_kda_prefill_backend", lambda backend, *args: backend
    )
    monkeypatch.setitem(sys.modules, "vllm._flashkda_C", ModuleType("vllm._flashkda_C"))
    monkeypatch.setattr(
        torch.ops._flashkda_C, "get_workspace_size", lambda *args: 513, raising=False
    )
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)

    def make(backend, num_tokens=17):
        config = SimpleNamespace(
            linear_attn_config=dict(
                num_heads=4,
                head_dim=8,
                short_conv_kernel_size=4,
                use_full_rank_gate=True,
            )
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(use_kda_recoverssm=False, use_replayssm=False),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=num_tokens, max_num_seqs=3
            ),
            additional_config={"kda_prefill_backend": backend},
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with torch.device("cpu"):
            return kda.KimiK3DeltaAttention(config, vllm_config)

    return make


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer"])
def test_prefill_workspace_reserved_during_construction_and_reused(
    make_kda_layer, monkeypatch, backend
):
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)

    layer = make_kda_layer(backend)
    expected_specs = (((1, 17, 2, 8), torch.bfloat16),)
    if backend == "flashkda":
        expected_specs += (
            ((3, 2, 8, 8), torch.float32),
            ((3, 2, 8, 8), torch.float32),
            ((513,), torch.uint8),
        )
        assert layer._flashkda_buffer_specs == expected_specs
        assert layer._flashinfer_kda_output_spec is None
    else:
        assert layer._flashkda_buffer_specs is None
        assert layer._flashinfer_kda_output_spec == expected_specs[0]

    # No forward/warmup has run: the full aligned layout must already exist.
    reserved = manager._current_workspaces[0]
    assert reserved is not None
    assert reserved.numel() == sum(
        ((prod(shape) * dtype.itemsize + 255) // 256) * 256
        for shape, dtype in expected_specs
    )
    make_kda_layer(backend)
    make_kda_layer(backend, num_tokens=5)
    assert manager._current_workspaces[0] is reserved

    manager.lock()
    buffers = manager.get_simultaneous(*expected_specs)
    assert [(tuple(buf.shape), buf.dtype) for buf in buffers] == list(expected_specs)
    assert all(
        buf.untyped_storage().data_ptr() == reserved.data_ptr() for buf in buffers
    )


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer"])
def test_prefill_workspace_reservation_grows_for_larger_layout(
    make_kda_layer, monkeypatch, backend
):
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    make_kda_layer(backend, num_tokens=5)
    smaller_bytes = manager._current_workspaces[0].numel()
    make_kda_layer(backend, num_tokens=33)
    assert manager._current_workspaces[0].numel() > smaller_bytes


@pytest.mark.parametrize("backend", ["flashkda", "flashinfer", "triton"])
def test_prefill_workspace_construction_without_manager(
    make_kda_layer, monkeypatch, backend
):
    monkeypatch.setattr(workspace, "_manager", None)
    make_kda_layer(backend)
    assert not workspace.is_workspace_manager_initialized()


def test_triton_prefill_does_not_reserve_workspace(make_kda_layer, monkeypatch):
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    make_kda_layer("triton")
    assert manager._current_workspaces == [None]
