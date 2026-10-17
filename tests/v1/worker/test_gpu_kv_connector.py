# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(kv_connector_module, "get_forward_context", object)

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(SimpleNamespace(kv_transfer_config=kv_config), {})
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    input_batch = SimpleNamespace(
        idx_mapping=request_indices, req_ids=request_ids, num_tokens=5
    )
    attn_metadata = {"layer": object()}
    connector.handle_preemptions(output)
    connector.pre_forward(output, input_batch, attn_metadata=attn_metadata)
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]

    kwargs = connector.kv_connector.start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["num_tokens"] == 5
    assert kwargs["scheduler_output"] is output
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    no_forward_output = _scheduler_output(False)
    connector.handle_preemptions(no_forward_output)
    connector.no_forward(no_forward_output)
    assert connector.kv_connector.start_load_kv.call_count == 2
    assert connector.kv_connector.start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    output = _scheduler_output(False)
    connector.handle_preemptions(output)
    connector.no_forward(output)

    assert events == ["handle", "bind", "start", "wait", "clear"]


@pytest.mark.parametrize(
    "runner_cls,dummy_run,num_tokens,disabled",
    [
        (GPUModelRunner, False, 0, False),
        (GPUModelRunner, False, 1, False),
        (GPUModelRunner, True, 0, False),
        (GPUModelRunner, True, 1, False),
        (GPUModelRunner, True, 0, True),
        (GPUModelRunner, True, 1, True),
        (MMEncoderModelRunner, False, 0, False),
        (MMEncoderModelRunner, False, 1, False),
    ],
    ids=[
        "decoder-zero",
        "decoder-work",
        "dummy-zero",
        "dummy-work",
        "disabled-dummy-zero",
        "disabled-dummy-work",
        "encoder-zero",
        "encoder-work",
    ],
)
def test_runner_drains_preempted_saves_before_reusing_pages(
    monkeypatch: pytest.MonkeyPatch,
    runner_cls: type[GPUModelRunner],
    dummy_run: bool,
    num_tokens: int,
    disabled: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(False)
    if disabled:
        monkeypatch.setattr(
            kv_connector_module.kv_transfer_state, "_KV_CONNECTOR_AGENT", None
        )
        connector.set_disabled(True)
        output.kv_connector_metadata = None
    output.total_num_scheduled_tokens = num_tokens
    output.num_scheduled_tokens = {"new": num_tokens}
    runner = object.__new__(runner_cls)
    if isinstance(runner, MMEncoderModelRunner):
        runner.input_tensor_semaphore = nullcontext
    runner.kv_connector = connector
    runner.aux_output_connector = None
    runner.update_pp_decode_requests = Mock()
    runner.finish_requests = Mock()
    runner.free_states = Mock()
    runner.add_requests = Mock()

    def reuse_pages(_):
        assert events == ["handle"], "a pending save can still read reused pages"
        events.append("reuse")

    runner.update_requests = reuse_pages
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner._merge_ec_connector_no_forward = lambda _, result: result

    class InputsReached(Exception):
        pass

    runner.gather_batch_req_state = Mock(side_effect=InputsReached)
    expected_events = [] if disabled else ["handle"]
    if not dummy_run:
        expected_events.append("reuse")
    if dummy_run and not num_tokens:
        runner.gather_batch_req_state.return_value = (None, None)
        runner.gather_batch_req_state.side_effect = None
        runner.lora_config = None
        runner.is_encoder_decoder = False
        runner.cudagraph_manager = None
        runner.parallel_config = SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0
        )
        runner.ubatch_runner = None
        runner.decode_query_len = 1
        monkeypatch.setattr(
            model_runner_module,
            "dispatch_cg_and_sync_dp",
            lambda *_args, **_kwargs: (SimpleNamespace(num_tokens=0), None),
        )
        runner.execute_model(output, dummy_run=True)
    elif num_tokens:
        with pytest.raises(InputsReached):
            runner.execute_model(output, dummy_run=dummy_run)
    else:
        runner.execute_model(output)
    if not num_tokens and not disabled:
        expected_events.extend(["bind", "start", "wait", "clear"])
    assert events == expected_events
    assert connector.kv_connector.handle_preemptions.call_count == int(not disabled)
