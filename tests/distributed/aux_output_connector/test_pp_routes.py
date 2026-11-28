# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.aux_output_connector import worker as worker_module
from vllm.distributed.aux_output_connector.worker import (
    PP_ROUTED_EXPERTS_KEY,
    AuxOutputWorkerConnector,
    add_pp_routes_buffer,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.cpu_test

_NUM_LAYERS = 4
_TOP_K = 2
_MAX_TOKENS = 8


def _stage() -> AuxOutputWorkerConnector:
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.output_dtype = torch.int16
    capturer.device_buffer = torch.full(
        (_MAX_TOKENS, _NUM_LAYERS, _TOP_K), -1, dtype=torch.int32
    )
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._capturer = capturer
    return worker


@pytest.mark.parametrize(
    "stage_layers",
    [[[0], [1, 2], [3]], [[0, 1, 2], [3]], [[0, 2], [1, 3]]],
    ids=["three-stages", "two-stages", "interleaved-layer-visits"],
)
def test_last_stage_holds_the_rows_of_every_layer(stage_layers):
    num_tokens = 5
    routing = torch.randint(0, 64, (num_tokens, _NUM_LAYERS, _TOP_K), dtype=torch.int32)
    received = None
    for layers in stage_layers:
        worker = _stage()
        if received is not None:
            worker.receive_pp_routes(received, num_tokens)
        # The stage's forward writes the rows of its own layers.
        worker._capturer.device_buffer[:num_tokens, layers] = routing[:, layers]
        hidden = torch.zeros(_MAX_TOKENS, 3)
        sent = worker.send_pp_routes(IntermediateTensors({"hidden_states": hidden}))
        assert sent["hidden_states"] is hidden
        received = sent[:num_tokens]
    rows = RoutedExpertsCapturer.snapshot_routing_data(worker._capturer, num_tokens)
    assert torch.equal(rows, routing.to(torch.int16))


def test_receive_leaves_rows_past_the_step_untouched():
    worker = _stage()
    sent = torch.zeros(_MAX_TOKENS, _NUM_LAYERS, _TOP_K, dtype=torch.int32)
    worker.receive_pp_routes(IntermediateTensors({PP_ROUTED_EXPERTS_KEY: sent}), 3)
    assert (worker._capturer.device_buffer[:3] == 0).all()
    assert (worker._capturer.device_buffer[3:] == -1).all()


def test_routes_buffer_matches_the_capture_layout(monkeypatch):
    monkeypatch.setattr(
        worker_module,
        "_get_routed_experts_shape",
        lambda _: (_NUM_LAYERS, 64, _TOP_K),
    )
    tensors = IntermediateTensors({})
    add_pp_routes_buffer(tensors, None, _MAX_TOKENS, torch.device("cpu"))
    buffer = tensors[PP_ROUTED_EXPERTS_KEY]
    assert buffer.shape == (_MAX_TOKENS, _NUM_LAYERS, _TOP_K)
    assert buffer.dtype == _stage()._capturer.device_buffer.dtype
