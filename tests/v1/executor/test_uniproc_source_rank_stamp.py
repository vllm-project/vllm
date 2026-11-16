# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H14-04: UniProcExecutor's authoritative TP=1 source-rank stamping.

UniProcExecutor runs no output aggregator (the single worker's output IS
the gathered output), so the executor itself is the only component that
can stamp which rank produced the transfer evidence. The blocking path
stamped directly; the non-blocking path returned a raw Future whose
result was never stamped -- every report and attestation reached the
scheduler with source_rank=None and the evidence gates rejected it
(conservative, but functionally broken), or a connector-supplied
impersonation value survived. The stamp now happens at result-resolution
time inside _SourceRankStampedFuture, with no completion/mutation race:
the stamped value is what set_result stores.
"""

import threading
import time
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from vllm.v1.executor.uniproc_executor import (
    AsyncOutputFuture,
    UniProcExecutor,
    _SourceRankStampedFuture,
    _stamp_tp1_source_rank,
)
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVLoadAttestation,
    KVLoadFailureReport,
    ModelRunnerOutput,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def make_output(reports=(), attestations=()) -> ModelRunnerOutput:
    """A ModelRunnerOutput carrying transfer evidence (the connector's
    rank attribution deliberately bogus: the stamp must overwrite it)."""
    output = Mock(spec=ModelRunnerOutput)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    return output


def report_for(req_id="r1", source_rank=7) -> KVLoadFailureReport:
    return KVLoadFailureReport(
        request_id=req_id,
        failed_block_ids=[3],
        token_span=(48, 64),
        rollback_boundary=48,
        observed_failed_layers=[1],
        healthy_above_span=True,
        restored_extent=64,
        source_rank=source_rank,
        transfer_generation=1,
    )


def attestation_for(req_id="r1", source_rank=9) -> KVLoadAttestation:
    return KVLoadAttestation(
        request_id=req_id,
        expected_objects=4,
        loaded_objects=4,
        failed_objects=0,
        certification="sha256",
        certified_objects=4,
        certification_failures=0,
        terminal=True,
        transfer_generation=1,
        source_rank=source_rank,
    )


def assert_stamped(output):
    kv = output.kv_connector_output
    assert kv is not None
    for report in kv.load_failure_reports or []:
        assert report.source_rank == 0
    for attestation in kv.load_attestations or []:
        assert attestation.source_rank == 0


def make_executor(reply):
    """A UniProcExecutor whose collective_rpc is faked to return the
    given reply (a value or an Exception) -- no worker, no device, no
    init."""
    executor = object.__new__(UniProcExecutor)

    def collective_rpc(
        method, args=(), kwargs=None, non_block=False, single_value=False
    ):
        if isinstance(reply, Exception):
            if not non_block:
                raise reply
            future = Future()
            future.set_exception(reply)
            return future
        if non_block:
            future = Future()
            future.set_result(reply if single_value else [reply])
            return future
        return reply if single_value else [reply]

    executor.collective_rpc = collective_rpc
    return executor


def pending_executor():
    """An executor whose non-blocking execute_model returns a pending
    future the test resolves explicitly."""
    executor = object.__new__(UniProcExecutor)
    holder = {}

    def collective_rpc(
        method, args=(), kwargs=None, non_block=False, single_value=False
    ):
        assert non_block
        holder["future"] = Future()
        return holder["future"]

    executor.collective_rpc = collective_rpc
    return executor, holder


# ---------------------------------------------------------------------------
# The stamping helper itself.
# ---------------------------------------------------------------------------


def test_stamp_overwrites_connector_supplied_rank():
    """The stamp is AUTHORITATIVE: a connector-supplied source_rank (here
    7 and 9 -- attempts to impersonate another worker) is overwritten to
    the single rank's index 0."""
    reports = [report_for(source_rank=7)]
    attestations = [attestation_for(source_rank=9)]
    output = make_output(reports, attestations)
    result = _stamp_tp1_source_rank(output)
    assert result is output
    assert_stamped(output)


def test_stamp_is_idempotent():
    reports = [report_for(source_rank=7)]
    output = make_output(reports, [])
    _stamp_tp1_source_rank(output)
    _stamp_tp1_source_rank(output)
    assert output.kv_connector_output.load_failure_reports[0].source_rank == 0


def test_stamp_none_and_empty_outputs_supported():
    assert _stamp_tp1_source_rank(None) is None
    output = make_output()
    assert _stamp_tp1_source_rank(output) is output
    output_no_kv = Mock(spec=ModelRunnerOutput)
    output_no_kv.kv_connector_output = None
    assert _stamp_tp1_source_rank(output_no_kv) is output_no_kv


# ---------------------------------------------------------------------------
# execute_model: blocking and non-blocking routing.
# ---------------------------------------------------------------------------


def test_blocking_execute_model_stamps():
    executor = make_executor(make_output([report_for()], [attestation_for()]))
    output = executor.execute_model(Mock(), non_block=False)
    assert output is not None
    assert_stamped(output)


def test_non_blocking_already_completed_future_stamps():
    executor = make_executor(make_output([report_for()], [attestation_for()]))
    future = executor.execute_model(Mock(), non_block=True)
    assert isinstance(future, Future)
    assert future.done()  # already resolved: adopted eagerly
    assert_stamped(future.result())


def test_non_blocking_future_completed_later_stamps():
    """The H14-04 core shape: execute_model() has already returned when
    the result resolves. The stamp is applied at result() time and the
    caller observes the stamped evidence."""
    executor, holder = pending_executor()
    future = executor.execute_model(Mock(), non_block=True)
    assert isinstance(future, Future)
    assert not future.done()  # non-blocking: nothing was resolved eagerly

    def resolve():
        time.sleep(0.05)
        holder["future"].set_result(make_output([report_for()], [attestation_for()]))

    thread = threading.Thread(target=resolve)
    thread.start()
    assert_stamped(future.result())
    thread.join()


def test_non_blocking_async_output_future_stamps():
    """The AsyncOutputFuture path (async scheduling): the future only
    completes when result() drives get_output(); the wrapper stamps the
    resolved value."""
    async_output = Mock()
    async_output.get_output.return_value = make_output(
        [report_for()], [attestation_for()]
    )
    inner = AsyncOutputFuture(async_output, single_value=True)
    assert not inner.done()
    future = _SourceRankStampedFuture(inner)
    assert not future.done()
    assert_stamped(future.result())
    async_output.get_output.assert_called_once()


def test_future_resolving_to_none_supported():
    executor, holder = pending_executor()
    future = executor.execute_model(Mock(), non_block=True)
    holder["future"].set_result(None)
    assert future.result() is None


def test_future_exception_propagates():
    executor, holder = pending_executor()
    future = executor.execute_model(Mock(), non_block=True)
    boom = RuntimeError("worker exploded")
    holder["future"].set_exception(boom)
    with pytest.raises(RuntimeError, match="worker exploded"):
        future.result()


def test_already_failed_future_surfaces_in_execute_model():
    """The original non-blocking contract: an execution that already
    failed before execute_model() returns raises in-line, not only at
    the first result() reader."""
    boom = RuntimeError("immediate failure")
    executor = make_executor(boom)
    with pytest.raises(RuntimeError, match="immediate failure"):
        executor.execute_model(Mock(), non_block=True)


def test_sample_tokens_blocking_stamps_evidence():
    """F2 (adversarial review): the sample path carries the same transfer
    evidence as execute_model (the worker attaches it post-forward) and
    on the batch-queue / None-fallback paths the engine core consumes
    THIS output as the step result -- it must be stamped identically,
    or every unstamped attestation fails the completeness gate and the
    repair policy silently degrades to permanent recompute on the
    uniproc path."""
    executor = make_executor(make_output([report_for()], [attestation_for()]))
    output = executor.sample_tokens(Mock(), non_block=False)
    assert output is not None
    assert_stamped(output)


def test_sample_tokens_non_blocking_stamps_evidence():
    """The non-blocking sample path (the async batch-queue consumer):
    resolved eagerly or later, the evidence is stamped at result
    time."""
    executor = make_executor(make_output([report_for()], [attestation_for()]))
    future = executor.sample_tokens(Mock(), non_block=True)
    assert isinstance(future, Future)
    assert_stamped(future.result())


def test_sample_tokens_pending_future_stamps_evidence():
    """The pending-future sample shape: sample_tokens() returns before
    the worker resolves; the stamp is applied at resolution."""
    executor, holder = pending_executor()
    future = executor.sample_tokens(Mock(), non_block=True)
    assert not future.done()

    def resolve():
        time.sleep(0.05)
        holder["future"].set_result(make_output([report_for()], [attestation_for()]))

    thread = threading.Thread(target=resolve)
    thread.start()
    assert_stamped(future.result())
    thread.join()


def test_no_evidence_output_untouched():
    executor = make_executor(make_output())
    output = executor.execute_model(Mock(), non_block=False)
    assert output.kv_connector_output is not None


def test_wrapper_does_not_block_early():
    """Non-blocking execution stays non-blocking: constructing and
    inspecting the wrapper never waits for the inner future."""
    pending = Future()
    start = time.monotonic()
    wrapper = _SourceRankStampedFuture(pending)
    assert not wrapper.done()
    assert time.monotonic() - start < 0.5
    pending.set_result(make_output())
    assert wrapper.done()
    assert wrapper.result() is not None
