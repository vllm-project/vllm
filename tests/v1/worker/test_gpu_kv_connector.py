# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from copy import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    polls: bool = False,
) -> ActiveKVConnector:
    backend = Mock()
    backend.supports_transfer_polling = polls
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")

    def get_transfer_results(_finished_req_ids):
        events.append("results")
        return KVConnectorTransferResults()

    backend.get_transfer_results.side_effect = get_transfer_results
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
    vllm_config = SimpleNamespace(
        kv_transfer_config=kv_config,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(runner_type="generate"),
    )
    connector = ActiveKVConnector(vllm_config, {})
    events.clear()
    return connector


def _scheduler_output(
    has_sync_kv_loads: bool, has_pending_kv_transfers: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
        has_pending_kv_transfers=has_pending_kv_transfers,
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
    connector.pre_forward(output, input_batch, attn_metadata=attn_metadata)
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "results", "clear"]

    start_load_kv = connector.kv_connector.start_load_kv
    assert isinstance(start_load_kv, Mock)
    kwargs = start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["num_tokens"] == 5
    assert kwargs["scheduler_output"] is output
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))
    assert start_load_kv.call_count == 2
    assert start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(False))

    assert events == ["handle", "bind", "start", "wait", "results", "clear"]


def test_deferred_finalize_collects_results_before_saving(
    monkeypatch: pytest.MonkeyPatch,
):
    """With a drafter, the transfer results are collected right after the
    target forward, and the saves and metadata are finalized after the
    drafter writes its KV."""
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.pre_forward(_scheduler_output(False))

    connector.post_forward(set(), defer_finalize=True)
    assert events == ["handle", "bind", "start", "results"]

    connector.finalize_step()
    assert events == ["handle", "bind", "start", "results", "wait", "clear"]


def _polling_connector(
    monkeypatch: pytest.MonkeyPatch, *results: KVConnectorTransferResults
) -> ActiveKVConnector:
    connector = _make_connector(monkeypatch, [], polls=True)
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    backend.get_transfer_results.side_effect = [
        *results,
        *(KVConnectorTransferResults() for _ in range(8)),
    ]
    return connector


def _empty_output():
    output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
    output.kv_connector_output = KVConnectorOutput()
    return output


def test_polling_only_while_loads_pending(monkeypatch: pytest.MonkeyPatch):
    connector = _polling_connector(monkeypatch)
    assert connector.next_poll_time() is None
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    assert connector.next_poll_time() is not None
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=False))
    assert connector.next_poll_time() is None

    connector = _make_connector(monkeypatch, [], polls=False)
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    assert connector.next_poll_time() is None


def test_held_results_go_to_next_output(monkeypatch: pytest.MonkeyPatch):
    """Results found by post_forward() and by polls are held until an output,
    after its step's GPU work, takes them, each exactly once."""
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(finished_recving={"a"}),
        KVConnectorTransferResults(finished_recving={"b"}, finished_sending={"c"}),
    )
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    take = connector.transfer_results_taker()
    assert take is not None
    step_kv_output = connector.post_forward(set())
    assert step_kv_output is not None and step_kv_output.finished_recving is None
    connector.poll()

    output = _empty_output()
    output.kv_connector_output = step_kv_output
    take(output)
    assert step_kv_output.finished_recving == {"a", "b"}
    assert step_kv_output.finished_sending == {"c"}

    later = _empty_output()
    take = connector.transfer_results_taker()
    assert take is not None
    take(later)
    assert later.kv_connector_output == KVConnectorOutput()


def test_worker_meta_stays_with_transfer_results(monkeypatch: pytest.MonkeyPatch):
    """A connector may also report a load's completion in its worker metadata
    (e.g. OffloadingConnector's completed_jobs); the scheduler needs it in the
    same output as the load's finished_recving."""
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(),
        KVConnectorTransferResults(finished_recving={"a"}),
    )
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    step_meta, poll_meta, merged_meta = Mock(), Mock(), Mock()
    step_meta.aggregate.return_value = merged_meta
    backend.build_connector_worker_meta.side_effect = [step_meta, poll_meta]
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    take = connector.transfer_results_taker()
    assert take is not None
    step_kv_output = connector.post_forward(set())
    assert step_kv_output is not None
    assert step_kv_output.kv_connector_worker_meta is None
    connector.poll()
    # Polls that find nothing leave worker metadata to the next step.
    connector.poll()
    assert backend.build_connector_worker_meta.call_count == 2

    output = _empty_output()
    output.kv_connector_output = step_kv_output
    take(output)
    assert step_kv_output.finished_recving == {"a"}
    assert step_kv_output.kv_connector_worker_meta is merged_meta
    step_meta.aggregate.assert_called_once_with(poll_meta)


def test_held_failures_stay_with_completions(monkeypatch: pytest.MonkeyPatch):
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(finished_recving={"a"}, failed_recving={"a"}),
    )
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    backend.get_block_ids_with_load_errors.return_value = {7}
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    connector.poll()

    output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
    take = connector.transfer_results_taker()
    assert take is not None
    take(output)
    kv_output = output.kv_connector_output
    assert kv_output is not None
    assert kv_output.finished_recving == {"a"}
    assert kv_output.failed_recving == {"a"}
    assert kv_output.invalid_block_ids == {7}


def test_no_forward_output_takes_held_results(monkeypatch: pytest.MonkeyPatch):
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(finished_recving={"a"}),
        KVConnectorTransferResults(finished_recving={"b"}),
    )
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    connector.poll()

    output = connector.no_forward(
        _scheduler_output(False, has_pending_kv_transfers=True)
    )
    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_recving == {"a", "b"}


def test_no_forward_output_defers_to_pending_output(monkeypatch: pytest.MonkeyPatch):
    """An output without GPU work must not take results while an earlier step's
    output, processed first, has yet to take them."""
    connector = _polling_connector(
        monkeypatch, KVConnectorTransferResults(finished_sending={"a"})
    )
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    take = connector.transfer_results_taker()
    assert take is not None
    connector.post_forward(set())

    empty = connector.no_forward(_scheduler_output(False))
    assert empty.kv_connector_output is None or not (
        empty.kv_connector_output.finished_sending
    )
    output = _empty_output()
    take(output)
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_sending == {"a"}


def test_no_polling_without_opt_in(monkeypatch: pytest.MonkeyPatch):
    connector = _make_connector(monkeypatch, [], polls=False)
    assert connector.transfer_results_taker() is None


def test_polling_paused_from_take_until_next_step(monkeypatch: pytest.MonkeyPatch):
    """Once an output takes held results, polling waits for the next step."""
    connector = _polling_connector(monkeypatch)
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    take = connector.transfer_results_taker()
    assert take is not None
    connector.post_forward(set())
    assert connector.next_poll_time() is not None

    take(_empty_output())
    assert connector.next_poll_time() is None

    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    assert connector.next_poll_time() is None
    connector.post_forward(set())
    assert connector.next_poll_time() is not None


def test_sync_load_failures_stay_with_their_step(monkeypatch: pytest.MonkeyPatch):
    """An earlier pending output must not take a later step's sync load failures,
    nor anything collected after them."""
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(),
        KVConnectorTransferResults(finished_recving={"async"}),
    )
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    earlier_take = connector.transfer_results_taker()

    connector.pre_forward(_scheduler_output(True, has_pending_kv_transfers=True))
    assert connector.next_poll_time() is None
    take = connector.transfer_results_taker()
    assert earlier_take is not None and take is not None
    backend.get_block_ids_with_load_errors.return_value = {7}
    connector.post_forward(set())
    backend.get_block_ids_with_load_errors.return_value = set()
    assert connector.next_poll_time() is not None
    connector.poll()

    earlier = _empty_output()
    earlier_take(earlier)
    assert earlier.kv_connector_output == KVConnectorOutput()
    output = _empty_output()
    take(output)
    assert output.kv_connector_output.invalid_block_ids == {7}
    assert output.kv_connector_output.finished_recving == {"async"}


def test_poll_error_fails_next_step(monkeypatch: pytest.MonkeyPatch):
    connector = _polling_connector(monkeypatch)
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    backend.get_transfer_results.side_effect = RuntimeError("lost transfer")
    connector.poll()
    assert connector.next_poll_time() is None
    backend.get_transfer_results.side_effect = None
    backend.get_transfer_results.return_value = KVConnectorTransferResults()

    with pytest.raises(RuntimeError, match="lost transfer"):
        connector.post_forward(set())
    assert connector.post_forward(set()) is not None


@pytest.mark.parametrize("on_model_thread", [True, False])
def test_output_polls_only_on_model_thread(
    monkeypatch: pytest.MonkeyPatch, on_model_thread: bool
):
    """Outputs collected by the model thread (uniproc or synchronous scheduling)
    poll first; the connector must not be called from another thread."""
    connector = _polling_connector(
        monkeypatch,
        KVConnectorTransferResults(),
        KVConnectorTransferResults(finished_recving={"a"}),
    )
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    take = connector.transfer_results_taker()
    assert take is not None
    connector.post_forward(set())

    output = _empty_output()
    if on_model_thread:
        take(output)
    else:
        thread = threading.Thread(target=take, args=(output,))
        thread.start()
        thread.join()
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_recving == (
        {"a"} if on_model_thread else None
    )


def test_uncollected_outputs_do_not_strand_results(monkeypatch: pytest.MonkeyPatch):
    """Warmup outputs, and ones lost to a failed step, are never collected."""
    # set_disabled() changes the global connector.
    monkeypatch.setattr(
        kv_connector_module.kv_transfer_state, "_KV_CONNECTOR_AGENT", None
    )
    connector = _polling_connector(
        monkeypatch, KVConnectorTransferResults(finished_recving={"a"})
    )
    connector.set_disabled(True)
    assert connector.transfer_results_taker() is None
    connector.set_disabled(False)
    output = connector.no_forward(_scheduler_output(False))
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_recving == {"a"}

    lost, take = connector.transfer_results_taker(), connector.transfer_results_taker()
    assert lost is not None and take is not None
    take(_empty_output())
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    backend.get_transfer_results.side_effect = [
        KVConnectorTransferResults(finished_recving={"b"})
    ]
    output = connector.no_forward(_scheduler_output(False))
    assert output.kv_connector_output is not None
    assert output.kv_connector_output.finished_recving == {"b"}


def test_consecutive_sync_load_failures_stay_with_their_steps(
    monkeypatch: pytest.MonkeyPatch,
):
    """With async scheduling, the next step's results are collected before this
    step's output takes its own."""
    connector = _polling_connector(monkeypatch)
    backend = connector.kv_connector
    assert isinstance(backend, Mock)
    takes = []
    for block_id in (1, 2, 3):
        connector.pre_forward(_scheduler_output(True, has_pending_kv_transfers=True))
        takes.append(connector.transfer_results_taker())
        backend.get_block_ids_with_load_errors.return_value = {block_id}
        connector.post_forward(set())
    backend.get_block_ids_with_load_errors.return_value = set()
    connector.pre_forward(_scheduler_output(False, has_pending_kv_transfers=True))
    last_take = connector.transfer_results_taker()
    backend.get_transfer_results.side_effect = [
        KVConnectorTransferResults(finished_recving={"async"})
    ]
    connector.post_forward(set())

    outputs = []
    # Outputs are collected on another thread, which doesn't poll.
    connector._model_thread = -1
    for take in (*takes, last_take):
        assert take is not None
        output = _empty_output()
        take(output)
        outputs.append(output.kv_connector_output)
    assert [o.invalid_block_ids for o in outputs] == [{1}, {2}, {3}, set()]
    assert outputs[2].finished_recving == {"async"}
    assert not (outputs[0].finished_recving or outputs[1].finished_recving)
