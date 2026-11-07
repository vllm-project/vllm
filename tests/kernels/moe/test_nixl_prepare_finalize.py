# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_test


@pytest.fixture(params=["nixl_ep", "deepep_ll"])
def prepare_finalize_module(request, monkeypatch):
    # Only the external transport is mocked; construct the real vLLM adapter.
    from vllm.utils.import_utils import has_deep_ep, has_nixl_ep

    backend = request.param
    transport_name, has_transport = (
        ("nixl_ep", has_nixl_ep) if backend == "nixl_ep" else ("deep_ep", has_deep_ep)
    )
    name = f"vllm.model_executor.layers.fused_moe.prepare_finalize.{backend}"
    if name in sys.modules or has_transport():
        yield importlib.import_module(name)
        return
    transport = types.ModuleType(transport_name)
    transport.Buffer = object  # type: ignore[attr-defined]
    transport.topk_idx_t = torch.int64  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, transport_name, transport)
    module = importlib.import_module(name)
    try:
        yield module
    finally:
        del sys.modules[name]
        delattr(sys.modules[name.rsplit(".", 1)[0]], backend)


class _RecordingTransport:
    def dispatch(self, inputs, ids, *args, **kwargs):
        self.dispatch_ids = ids.clone()
        return None, None, object(), None, lambda: None

    def combine(self, inputs, ids, *args, **kwargs):
        self.combine_ids = ids.clone()
        return None, None, lambda: None

    low_latency_dispatch = dispatch
    low_latency_combine = combine


@pytest.mark.parametrize(
    "ids, expected, round_robin",
    [
        ([[-1, 6], [0, 7]], [[-1, 5], [0, 7]], True),
        ([[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], True),
        ([[0, -1], [-1, 6]], [[0, -1], [-1, 6]], False),
    ],
    ids=["mixed-padding", "all-padding", "linear"],
)
def test_routing_preserves_padding(prepare_finalize_module, ids, expected, round_robin):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceDelegate,
    )

    transport = _RecordingTransport()
    mapping = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7]) if round_robin else None
    if prepare_finalize_module.__name__.endswith(".nixl_ep"):
        adapter_type = prepare_finalize_module.NixlEPPrepareAndFinalize
        backend_args = {"expert_capacity": 8}
    else:
        adapter_type = prepare_finalize_module.DeepEPLLPrepareAndFinalize
        backend_args = {}
    adapter = adapter_type(
        buffer=transport,
        max_tokens_per_rank=2,
        num_dispatchers=4,
        global_to_physical=mapping,
        **backend_args,
    )
    routes = torch.tensor(ids, dtype=adapter.topk_indices_dtype())
    inputs = torch.zeros((2, 2048), dtype=torch.bfloat16)
    weights = torch.ones_like(routes, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make()
    reducer = TopKWeightAndReduceDelegate()
    output = torch.empty_like(inputs)
    expert_output = torch.empty((2, 8, 2048), dtype=inputs.dtype)
    adapter.prepare_async(inputs, weights, routes, 8, None, False, quant_config)
    adapter.finalize_async(output, expert_output, weights, routes, False, reducer)
    reference = torch.tensor(expected, dtype=routes.dtype)
    torch.testing.assert_close(transport.dispatch_ids, reference)
    torch.testing.assert_close(transport.combine_ids, reference)
