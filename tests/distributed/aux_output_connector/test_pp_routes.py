# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

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


def _config(routed_experts: bool) -> SimpleNamespace:
    return SimpleNamespace(
        aux_output_config=SimpleNamespace(
            enabled=True, enable_return_routed_experts=routed_experts
        )
    )


def _stage(start_layer: int, max_num_tokens: int = 8) -> AuxOutputWorkerConnector:
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.shape_per_token = (_NUM_LAYERS, _TOP_K)
    capturer.output_dtype = torch.int16
    capturer.output_dtype_name = "int16"
    capturer.device_buffer = torch.full(
        (max_num_tokens, _NUM_LAYERS, _TOP_K), -1, dtype=torch.int32
    )
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._vllm_config = _config(routed_experts=True)
    worker._routed_experts_capturer = capturer
    worker._indexer = None
    worker._pp_start_layer = start_layer
    return worker


@pytest.mark.parametrize("start_layer", [1, 3], ids=["one-earlier", "three-earlier"])
def test_receive_assigns_only_the_earlier_stages_layers(start_layer):
    worker = _stage(start_layer)
    sent = torch.arange(8 * _NUM_LAYERS * _TOP_K, dtype=torch.int32).view(
        8, _NUM_LAYERS, _TOP_K
    )
    own = torch.full((5, _NUM_LAYERS - start_layer, _TOP_K), 7, dtype=torch.int32)
    worker._routed_experts_capturer.device_buffer[:5, start_layer:] = own
    worker.receive_pp_routes(IntermediateTensors({PP_ROUTED_EXPERTS_KEY: sent}), 5)
    buffer = worker._routed_experts_capturer.device_buffer
    assert torch.equal(buffer[:5, :start_layer], sent[:5, :start_layer])
    assert torch.equal(buffer[:5, start_layer:], own)
    assert (buffer[5:] == -1).all()


def test_last_stage_rows_equal_the_rows_of_every_layer():
    stages = [range(0, 1), range(1, 3), range(3, 4)]
    num_tokens = 5
    routing = torch.randint(0, 64, (num_tokens, _NUM_LAYERS, _TOP_K), dtype=torch.int32)
    received = None
    for layers in stages:
        worker = _stage(layers.start)
        if received is not None:
            worker.receive_pp_routes(received, num_tokens)
        worker._routed_experts_capturer.device_buffer[
            :num_tokens, layers.start : layers.stop
        ] = routing[:, layers.start : layers.stop]
        hidden = torch.zeros(8, 3)
        sent = worker.send_pp_routes(IntermediateTensors({"hidden_states": hidden}))
        assert sent["hidden_states"] is hidden
        received = sent[:num_tokens]
    rows = worker._routed_experts_capturer.snapshot_routing_data(num_tokens)
    assert torch.equal(rows, routing.to(torch.int16))


@pytest.mark.parametrize(
    "tp_first,pp_last,owns",
    [(True, True, True), (True, False, False), (False, True, False)],
    ids=["output-rank", "earlier-stage", "other-tp-rank"],
)
def test_only_the_last_stage_output_rank_owns_the_data_plane(tp_first, pp_last, owns):
    worker = _stage(0)
    worker._store = None
    worker._buffer = None
    worker._vllm_config = SimpleNamespace(
        aux_output_config=SimpleNamespace(
            enable_return_routed_experts=True,
            enable_return_indexer_topk=False,
            object_key_prefix=None,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        max_concurrent_batches=1,
    )
    with (
        patch.object(
            worker_module,
            "get_tp_group",
            return_value=SimpleNamespace(is_first_rank=tp_first),
        ),
        patch.object(
            worker_module,
            "get_pp_group",
            return_value=SimpleNamespace(is_last_rank=pp_last),
        ),
        patch.object(worker_module, "retained_objects", return_value=(4, 4)),
    ):
        worker.initialize_output_storage(None)
    assert (worker._buffer is not None) == owns
    if worker._store is not None:
        worker._store.close()


@pytest.mark.parametrize(
    "routed_experts", [True, False], ids=["routed-experts", "other-outputs"]
)
def test_routes_buffer_is_declared_only_with_routed_experts(routed_experts):
    tensors = IntermediateTensors({})
    with patch.object(
        worker_module,
        "_get_routed_experts_shape",
        return_value=(_NUM_LAYERS, 64, _TOP_K),
    ):
        add_pp_routes_buffer(tensors, _config(routed_experts), 8, torch.device("cpu"))
    assert (PP_ROUTED_EXPERTS_KEY in tensors.tensors) == routed_experts


def test_stage_without_routed_experts_relays_no_routes():
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._vllm_config = _config(routed_experts=False)
    tensors = IntermediateTensors({"hidden_states": torch.zeros(8, 3)})
    worker.receive_pp_routes(tensors, 5)
    assert set(worker.send_pp_routes(tensors).tensors) == {"hidden_states"}
