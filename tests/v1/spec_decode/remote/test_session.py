# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier session against the fake draft server: negotiation, proposal
rounds, failure latching, stale-response discard, and shutdown recovery."""

import contextlib
import dataclasses
import socket
import threading
import time
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm.config.speculative import RemoteDraftConfig
from vllm.v1.spec_decode.remote import session as session_module
from vllm.v1.spec_decode.remote import transport as transport_module
from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    PROTOCOL_MINOR,
    CloseSequence,
    ErrorReply,
    FeatureSlot,
    Hello,
    HelloAck,
    OpenSequence,
    Ping,
    PrefillChunk,
    ProposalResponse,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
)
from vllm.v1.spec_decode.remote.server import (
    DEFAULT_LIMITS,
    AdapterError,
    FakeDraftAdapter,
    RemoteDraftServer,
)
from vllm.v1.spec_decode.remote.session import (
    RemoteDraftError,
    RemoteDraftSession,
    RemoteProposalHandle,
    VerifierIdentity,
    frame_to_tensor,
    tensor_to_frame,
)
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    HandleState,
    InvalidStateTransition,
    RemoteProposalBatch,
    SequenceState,
)
from vllm.v1.spec_decode.remote.transport import (
    TransportError,
    TransportTimeout,
    listen,
)

VOCAB = 50
NUM_SPEC_TOKENS = 3
OK = SpeculatorStatusCode.OK
IDENTITY = VerifierIdentity(
    verifier_instance_id="verifier-a",
    target_fingerprint="tf",
    tokenizer_fingerprint="kf",
    method="eagle3",
    num_speculative_tokens=NUM_SPEC_TOKENS,
    parallel_drafting=False,
    provided_features=(TargetFeatureKind.TOKEN_IDS.value,),
    max_num_seqs=8,
    max_num_batched_tokens=64,
    max_model_len=64,
    vocab_size=VOCAB,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.after_read = 0.0

    def __call__(self):
        now = self.now
        self.now += self.after_read
        self.after_read = 0.0
        return now

    def advance(self, seconds: float):
        self.now += seconds


def tokens(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype)


def features(values):
    return FeatureBatch(schema_id=1, slots=(tokens(values),))


def batch(batch_id, keys, recovery, accepted=None):
    accepted = accepted or [0] * len(keys)
    return RemoteProposalBatch(
        batch_id=batch_id,
        keys=tuple(keys),
        accepted_counts=tokens(accepted),
        recovery_tokens=tokens(recovery, torch.int64),
        features=features(recovery),
    )


def expected_row(last_token: int):
    return [(last_token + j + 1) % VOCAB for j in range(NUM_SPEC_TOKENS)]


class ProtocolPeer:
    """Socket peer for replies a conforming draft adapter cannot produce."""

    hello: Hello
    protocol_minor: int = PROTOCOL_MINOR

    def __init__(self, conn):
        self.conn = conn

    def recv(self):
        frames = []
        while True:
            frame = self.conn.recv(timeout=5)
            if frame.kind == transport_module.FRAME_DATA:
                frames.append(transport_module.decode_data_frame(frame.body))
                continue
            envelope = decode_envelope(frame.body)
            return envelope, decode_payload(envelope), frames

    def reply(self, envelope, message, *, frames=()):
        for frame in frames:
            self.conn.send_data(frame)
        reply = decode_envelope(
            encode_message(message, session_id="s", request_id=envelope.request_id)
        )
        self.conn.send_control(
            msgspec.msgpack.encode(
                msgspec.structs.replace(reply, protocol_minor=self.protocol_minor)
            )
        )


@pytest.fixture
def protocol_peer():
    listener = listen("tcp://127.0.0.1:0")
    connections, threads, errors = [], [], []

    def start(script, *, protocol_minor=PROTOCOL_MINOR):
        def serve():
            try:
                conn = listener.accept(timeout=5)
                connections.append(conn)
                peer = ProtocolPeer(conn)
                peer.protocol_minor = protocol_minor
                envelope, hello, _ = peer.recv()
                assert isinstance(hello, Hello)
                peer.hello = hello
                adapter = FakeDraftAdapter(vocab_size=VOCAB)
                peer.reply(
                    envelope,
                    HelloAck(
                        server_id="scripted",
                        session_id="s",
                        session_epoch=1,
                        selected_transport="inline",
                        capabilities=adapter.capabilities(),
                        feature_schema=adapter.feature_schema(),
                        limits=DEFAULT_LIMITS,
                    ),
                )
                script(peer)
            except BaseException as exc:
                errors.append(exc)
            finally:
                for conn in connections:
                    conn.close()

        thread = threading.Thread(target=serve)
        threads.append(thread)
        thread.start()
        return listener.endpoint

    yield start
    listener.close()
    for conn in connections:
        conn.close()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not errors


def test_pipelined_prefills_cannot_renew_an_expired_round_budget():
    """An expired queued prefill must not delay an existing healthy request."""

    class DelayedPrefill(FakeDraftAdapter):
        def __init__(self):
            super().__init__(vocab_size=VOCAB)
            self.release_first = threading.Event()
            self.release_second = threading.Event()
            self.second_calls = 0

        def prefill(self, key, inputs):
            if key.sequence_id == 1:
                assert self.release_first.wait(5)
            if key.sequence_id == 2:
                self.second_calls += 1
                assert self.release_second.wait(5)
            return super().prefill(key, inputs)

    adapter = DelayedPrefill()
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, request_timeout_ms=200)
    try:
        session.connect()
        first, second, healthy = session.open_sequences((1, 2, 3))
        session.prefill(healthy, features([4]), is_final=True)
        session.prefill_many(
            [(first, features([4]), False), (second, features([4]), False)]
        )
        assert session.sequence_state(first) is SequenceState.TARGET_ONLY
        assert session.sequence_state(second) is SequenceState.TARGET_ONLY
        adapter.release_first.set()
        result = session.collect(session.dispatch(batch(0, [healthy], [7])))
        assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS]
        assert result.output.token_ids.tolist() == [expected_row(7)]
        assert adapter.second_calls == 0
        assert session.sequence_state(healthy) is SequenceState.READY
    finally:
        adapter.release_first.set()
        adapter.release_second.set()
        session.close()
        server.stop()


def test_legacy_peer_does_not_receive_more_prefills_after_round_timeout(protocol_peer):
    """Older peers cannot anchor pipelined commands to one execution budget."""
    from vllm.v1.spec_decode.remote.protocol import Pong

    def script(peer):
        for _ in range(3):
            envelope, opened, _ = peer.recv()
            assert isinstance(opened, OpenSequence)
            peer.reply(envelope, SequenceAck(opened.key))
        _, first, _ = peer.recv()
        assert isinstance(first, PrefillChunk)
        # No first ACK: the shared client deadline must expire before another
        # prefill can be sent. The next RPC flushes logical retirements.
        envelope, closed, _ = peer.recv()
        assert isinstance(closed, CloseSequence)
        for key in closed.keys:
            peer.reply(envelope, SequenceAck(key))
        envelope, ping, _ = peer.recv()
        assert isinstance(ping, Ping)
        peer.reply(envelope, Pong(nonce=ping.nonce))

    endpoint = protocol_peer(script, protocol_minor=1)
    session = RemoteDraftSession(endpoint, IDENTITY, request_timeout_ms=200)
    try:
        session.connect()
        first, second, _ = session.open_sequences((1, 2, 3))
        session.prefill_many(
            [(first, features([4]), False), (second, features([4]), False)]
        )
        assert session.sequence_state(first) is SequenceState.TARGET_ONLY
        assert session.sequence_state(second) is SequenceState.TARGET_ONLY
        assert session.ping().healthy
    finally:
        session.close()


@pytest.fixture
def adapter():
    return FakeDraftAdapter(vocab_size=VOCAB)


@pytest.fixture
def server(adapter):
    server = RemoteDraftServer(adapter, server_id="srv")
    server.start("tcp://127.0.0.1:0")
    yield server
    adapter.release()
    server.stop()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_session(server, clock):
    sessions = []

    def factory(identity=IDENTITY, *, connect=True, **kwargs):
        kwargs.setdefault("request_timeout_ms", 100)
        session = RemoteDraftSession(server.endpoint, identity, clock=clock, **kwargs)
        sessions.append(session)
        if connect:
            session.connect()
        return session

    yield factory
    for session in sessions:
        session.close()


@pytest.fixture
def session(make_session):
    return make_session()


def ready(session, sequence_id, prompt=(1, 2, 3)):
    key = session.open_sequence(sequence_id)
    session.prefill(key, features(list(prompt)), is_final=True)
    assert session.sequence_state(key) is SequenceState.READY
    return key


def round_trip(session, batch_):
    return session.collect(session.dispatch(batch_))


@pytest.mark.parametrize("effective_k", [-1, NUM_SPEC_TOKENS + 1])
def test_invalid_round_budget_fails_before_sending(session, monkeypatch, effective_k):
    key = ready(session, 1)
    send = Mock()
    monkeypatch.setattr(session, "_send", send)
    with pytest.raises(ValueError, match="draft budget"):
        session.dispatch(
            dataclasses.replace(
                batch(0, [key], [7]), num_speculative_tokens=effective_k
            )
        )
    send.assert_not_called()
    assert session.sequence_state(key) is SequenceState.READY


def test_adapter_cannot_exceed_the_requested_round_budget(
    session, adapter, monkeypatch
):
    key = ready(session, 1)
    propose = adapter.advance_and_propose

    def ignore_budget(batch_):
        return propose(
            dataclasses.replace(batch_, effective_num_speculative_tokens=None)
        )

    monkeypatch.setattr(adapter, "advance_and_propose", ignore_budget)
    result = round_trip(
        session, dataclasses.replace(batch(0, [key], [7]), num_speculative_tokens=0)
    )
    assert result.row_statuses == (SpeculatorStatusCode.INTERNAL_ERROR,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.sequence_state(key) is SequenceState.TARGET_ONLY


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_local_failure_is_terminal_before_policy(make_session, policy):
    session = make_session(failure_policy=policy)
    key = ready(session, 1)
    expectation = (
        pytest.raises(RemoteDraftError, match="feature allocation failed")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expectation:
        session.fail_sequences(
            (key,),
            detail="feature allocation failed",
            deadline=session.preparation_deadline(),
        )
    assert session.sequence_state(key) is SequenceState.TARGET_ONLY
    assert session.connected
    assert session._records[key.sequence_id].last_status is (
        SpeculatorStatusCode.OUT_OF_MEMORY
    )


def test_terminal_retirement_uses_only_the_remaining_preparation_budget(
    session, adapter, clock, monkeypatch
):
    key = ready(session, 1)
    deadline = session.preparation_deadline()
    clock.advance(0.075)
    sent_deadlines = []
    send = session._conn.send_control

    def send_control(payload, **kwargs):
        message = decode_payload(decode_envelope(payload))
        assert isinstance(message, CloseSequence)
        sent_deadlines.append(kwargs["deadline"])
        send(payload, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(session_module.time, "monotonic", lambda: 5000.0)
        patch.setattr(session._conn, "send_control", send_control)
        patch.setattr(
            session,
            "_await",
            lambda *a, **k: pytest.fail("retirement waited for an ACK"),
        )
        session.fail_sequences((key,), detail="feature OOM", deadline=deadline)
    assert sent_deadlines == [pytest.approx(5000.025)]
    assert session.sequence_state(key) is SequenceState.TARGET_ONLY
    assert session.connected
    assert not session._retiring


@pytest.mark.parametrize("healthy_sibling", [False, True])
def test_expired_preparation_budget_reclaims_idle_connection_or_defers_to_sibling(
    session, adapter, server, clock, monkeypatch, healthy_sibling
):
    a = ready(session, 1)
    b = ready(session, 2) if healthy_sibling else None
    closed = threading.Event()
    close = adapter.close_sequences

    def close_sequences(keys):
        close(keys)
        if a in keys:
            closed.set()

    monkeypatch.setattr(adapter, "close_sequences", close_sequences)
    deadline = session.preparation_deadline()
    clock.advance(0.101)
    with monkeypatch.context() as patch:
        send = Mock(wraps=session._conn.send_control)
        patch.setattr(session._conn, "send_control", send)
        session.fail_sequences((a,), detail="feature OOM", deadline=deadline)
        send.assert_not_called()
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    if healthy_sibling:
        assert session.connected and session._retiring == {a}
        result = round_trip(session, batch(0, [b], [4]))
        assert result.row_statuses == (OK,)
        assert session.sequence_state(b) is SequenceState.READY
        assert session.connected
    else:
        assert not session.connected
    assert closed.wait(2), "terminal remote state was not retired"
    assert adapter.prefix(a) is None
    assert server.registry.get(a) is None
    assert not session._retiring


@pytest.mark.parametrize("send_failure", [TransportTimeout, TransportError])
def test_unsent_terminal_retirement_drops_connection_and_releases_peer_state(
    session, adapter, server, monkeypatch, send_failure
):
    a, b = ready(session, 1), ready(session, 2)
    closed = threading.Event()
    close = adapter.close_sequences

    def close_sequences(keys):
        close(keys)
        if a in keys:
            closed.set()

    def fail_send(payload, **kwargs):
        assert isinstance(decode_payload(decode_envelope(payload)), CloseSequence)
        raise send_failure("injected retirement send failure")

    monkeypatch.setattr(adapter, "close_sequences", close_sequences)
    monkeypatch.setattr(session._conn, "send_control", fail_send)
    session.fail_sequences(
        (a,), detail="feature OOM", deadline=session.preparation_deadline()
    )
    assert not session.connected
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    assert session.sequence_state(b) is SequenceState.INVALID
    assert closed.wait(2)
    assert server.registry.get(a) is None
    assert not session._retiring


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("healthy_sibling", [False, True])
@pytest.mark.parametrize("advance_clock", [False, True])
def test_prefill_timeout_retires_idle_state_or_preserves_healthy_sibling(
    make_session,
    adapter,
    server,
    clock,
    monkeypatch,
    policy,
    healthy_sibling,
    advance_clock,
):
    session = make_session(failure_policy=policy)
    a = session.open_sequence(1)
    b = ready(session, 2) if healthy_sibling else None
    completed, closed = threading.Event(), threading.Event()
    prefill, close = adapter.prefill, adapter.close_sequences

    def prefill_and_notify(*args, **kwargs):
        result = prefill(*args, **kwargs)
        completed.set()
        return result

    def close_and_notify(keys):
        close(keys)
        if a in keys:
            closed.set()

    def timeout(matches, deadline, **kwargs):
        assert completed.wait(2), "peer did not accept the prefill"
        if advance_clock:
            clock.now = deadline
        raise TransportTimeout("prefill acknowledgement expired")

    monkeypatch.setattr(adapter, "prefill", prefill_and_notify)
    monkeypatch.setattr(adapter, "close_sequences", close_and_notify)
    expectation = (
        pytest.raises(RemoteDraftError, match="prefill failed.*TIMEOUT")
        if policy == "error"
        else contextlib.nullcontext()
    )
    retire = session._send_retirements

    def reject_late_retirement(*args, **kwargs):
        if session._retiring:
            pytest.fail("cleanup wrote after a transport timeout")
        return retire(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(session, "_await", timeout)
        patch.setattr(session, "_send_retirements", reject_late_retirement)
        with expectation:
            session.prefill(a, features([1, 2]), is_final=False)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    assert session.connected is healthy_sibling
    if healthy_sibling:
        assert session._retiring == {a}
        assert round_trip(session, batch(0, [b], [4])).row_statuses == (OK,)
        assert session.sequence_state(b) is SequenceState.READY
    assert closed.wait(2), "terminal prefill retained peer state"
    assert adapter.prefix(a) is None and server.registry.get(a) is None
    assert not session._retiring


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_prefill_failure_retires_with_only_the_remaining_ack_budget(
    make_session, adapter, clock, monkeypatch, policy
):
    session = make_session(failure_policy=policy)
    session._request_timeout_s = 10.0
    key = session.open_sequence(1)
    closed = threading.Event()
    prefill, close = adapter.prefill, adapter.close_sequences
    read, retire = session._await, session._send_retirements
    budgets = []

    def fail_after_prefill(*args, **kwargs):
        prefill(*args, **kwargs)
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "prefill allocation")

    def read_ack(*args, **kwargs):
        result = read(*args, **kwargs)
        clock.advance(2.5)
        return result

    def retire_with_budget(conn, deadline):
        if session._retiring:
            budgets.append(deadline - time.monotonic())
        return retire(conn, deadline)

    def close_and_notify(keys):
        close(keys)
        if key in keys:
            closed.set()

    monkeypatch.setattr(adapter, "prefill", fail_after_prefill)
    monkeypatch.setattr(adapter, "close_sequences", close_and_notify)
    monkeypatch.setattr(session, "_await", read_ack)
    monkeypatch.setattr(session, "_send_retirements", retire_with_budget)
    expectation = (
        pytest.raises(RemoteDraftError, match="prefill failed.*OUT_OF_MEMORY")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expectation:
        session.prefill(key, features([1, 2]), is_final=False)
    assert len(budgets) == 1 and budgets[0] <= 7.5
    assert session.connected and not session._retiring
    assert session.sequence_state(key) is SequenceState.TARGET_ONLY
    assert closed.wait(2), "prefill failure retained remote state despite spare budget"
    assert adapter.prefix(key) is None


def test_normal_close_removes_deferred_terminal_retirement(session, clock, monkeypatch):
    a, _ = ready(session, 1), ready(session, 2)
    deadline = session.preparation_deadline()
    clock.advance(0.101)
    session.fail_sequences((a,), detail="feature OOM", deadline=deadline)
    assert session._retiring == {a}
    sent = []
    send = session._conn.send_control

    def send_control(payload, **kwargs):
        sent.append(decode_payload(decode_envelope(payload)))
        send(payload, **kwargs)

    monkeypatch.setattr(session._conn, "send_control", send_control)
    session.close_sequences((a,))
    assert len(sent) == 1 and isinstance(sent[0], CloseSequence)
    assert session.sequence_state(a) is None
    assert not session._retiring


# ----------------------------------------------------------------------
# Handshake
# ----------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["auto", "inline"])
def test_handshake_negotiates_inline_transport_and_schema(
    make_session, adapter, transport
):
    session = make_session(transport=transport)
    assert session.connected
    assert session.selected_transport == "inline"
    assert session.session_epoch >= 1
    assert session.feature_schema == adapter.feature_schema()
    assert session.capabilities == adapter.capabilities()
    assert session.limits.max_batch_size >= 1


def test_handshake_rejects_incompatible_capabilities(clock):
    adapter = FakeDraftAdapter(
        vocab_size=VOCAB,
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="target_kv",
            required_features=(TargetFeatureKind.TOKEN_IDS.value,),
            standalone_weights="complete",
        ),
    )
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    try:
        session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
        with pytest.raises(RemoteDraftError, match="target-owned state"):
            session.connect()
        assert not session.connected
    finally:
        server.stop()


def test_handshake_rejects_missing_provided_feature(make_session):
    identity = dataclasses.replace(IDENTITY, provided_features=())
    with pytest.raises(RemoteDraftError, match="cannot provide"):
        make_session(identity)


def test_handshake_surfaces_server_rejection(make_session):
    identity = dataclasses.replace(IDENTITY, method="ngram")
    with pytest.raises(RemoteDraftError, match="method 'ngram'"):
        make_session(identity)


def test_handshake_timeout_is_reported():
    # Nothing answers HELLO on a bare listener, so the handshake must give
    # up at the startup deadline instead of hanging (real clock here: the
    # deadline loop is what is under test).
    from vllm.v1.spec_decode.remote.transport import listen

    listener = listen("tcp://127.0.0.1:0")
    try:
        session = RemoteDraftSession(
            listener.endpoint, IDENTITY, startup_timeout_ms=200
        )
        with pytest.raises(RemoteDraftError, match="timed out"):
            session.connect()
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("max_num_seqs", 300, "max_batch_size=256"),
        ("max_num_batched_tokens", 20000, "max_feature_tokens=16384"),
        ("max_model_len", 40000, "max_model_len=32768"),
    ],
)
def test_handshake_rejects_limits_below_verifier_config(
    make_session, field, value, reason
):
    # Both sides check: the server refuses the HELLO, and a server that
    # does not check is still rejected by the verifier.
    identity = dataclasses.replace(IDENTITY, **{field: value})
    with pytest.raises(RemoteDraftError, match=reason):
        make_session(identity)
    session = make_session(connect=False, identity=identity)
    ack = HelloAck(
        server_id="srv",
        session_id="s",
        session_epoch=1,
        selected_transport="inline",
        capabilities=FakeDraftAdapter().capabilities(),
        feature_schema=FakeDraftAdapter().feature_schema(),
        limits=DEFAULT_LIMITS,
    )
    assert any(
        reason in r for r in session._handshake_rejections(session._hello(), ack)
    )


def test_unimplemented_transport_is_rejected():
    with pytest.raises(NotImplementedError):
        RemoteDraftSession("tcp://127.0.0.1:1", IDENTITY, transport="zmq")


# ----------------------------------------------------------------------
# Proposal rounds
# ----------------------------------------------------------------------


def test_round_trip_matches_fake_formula(session, adapter):
    a = session.open_sequence(1)
    session.prefill(a, features([1, 2]), is_final=False)
    session.prefill(a, features([3]), is_final=True)
    b = ready(session, 2, prompt=(9,))
    assert adapter.prefix(a) == [1, 2, 3]

    result = round_trip(session, batch(0, [a, b], recovery=[5, 47], accepted=[0, 1]))
    assert result.row_statuses == (OK, OK)
    assert result.output.token_ids.tolist() == [expected_row(5), expected_row(47)]
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS] * 2
    assert adapter.prefix(a) == [1, 2, 3, 5]
    assert session.sequence_state(a) is SequenceState.READY

    result = round_trip(session, batch(1, [b], recovery=[48]))
    assert result.output.token_ids.tolist() == [expected_row(48)]
    assert adapter.proposed_batches == [0, 1]


class PaddedAdapter(FakeDraftAdapter):
    """Proposes one valid token per row and pads the rest with junk."""

    def __init__(self, padding: int, first: int):
        super().__init__(vocab_size=VOCAB)
        self.padding = padding
        self.first = first

    def advance_and_propose(self, batch_):
        result = super().advance_and_propose(batch_)
        rows = len(batch_.keys)
        row = [self.first] + [self.padding] * (NUM_SPEC_TOKENS - 1)
        return dataclasses.replace(
            result,
            token_ids=tensor_to_frame(0, torch.tensor([row] * rows)),
            valid_lengths=tensor_to_frame(0, torch.ones(rows, dtype=torch.int32)),
        )


@pytest.mark.parametrize(
    ("padding", "first", "status", "expected"),
    [
        (-1, 7, OK, [7, 0, 0]),
        (7, VOCAB, SpeculatorStatusCode.INTERNAL_ERROR, [0, 0, 0]),
        (7, -3, SpeculatorStatusCode.INTERNAL_ERROR, [0, 0, 0]),
    ],
    ids=["junk-padding-is-zeroed", "too-large", "negative"],
)
def test_draft_tokens_are_checked_against_the_target_vocab(
    clock, padding, first, status, expected
):
    adapter = PaddedAdapter(padding, first)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
    try:
        session.connect()
        a = ready(session, 1)
        result = round_trip(session, batch(0, [a], recovery=[4]))
        assert result.row_statuses == (status,)
        assert result.output.token_ids.tolist() == [expected]
        if status is OK:
            assert result.output.valid_lengths.tolist() == [1]
        else:
            assert result.output.valid_lengths.tolist() == [0]
            assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


def test_handle_records_dispatch_metadata(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    assert isinstance(handle, RemoteProposalHandle)
    assert handle.state is HandleState.DISPATCHED
    assert handle.active_rows == (0,)
    assert handle.session_epoch == session.session_epoch
    assert handle.session_id == session.session_id
    assert session.sequence_state(a) is SequenceState.IN_FLIGHT
    session.collect(handle)
    assert handle.state is HandleState.COLLECTED


def test_unready_rows_are_skipped_locally(session, adapter):
    a = ready(session, 1)
    opened = session.open_sequence(2)
    unknown = SequenceKey("verifier-a", 99, 0)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, a.generation + 1)

    handle = session.dispatch(
        batch(0, [a, opened, unknown, stale], recovery=[4, 5, 6, 7])
    )
    assert handle.active_rows == (0,)
    result = session.collect(handle)
    assert result.row_statuses == (
        OK,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.STALE_GENERATION,
    )
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS, 0, 0, 0]
    assert result.output.token_ids[0].tolist() == expected_row(4)
    assert adapter.proposed_batches == [0]


def test_all_rows_skipped_completes_without_wire_traffic(session, adapter):
    handle = session.dispatch(batch(0, [SequenceKey("verifier-a", 5, 0)], recovery=[1]))
    assert handle.state is HandleState.COMPLETED
    result = session.collect(handle)
    assert result.output.valid_lengths.tolist() == [0]
    assert adapter.proposed_batches == []


def test_collect_is_one_shot(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    session.collect(handle)
    with pytest.raises(InvalidStateTransition, match="collected"):
        session.collect(handle)


def test_batch_ids_must_increase(session):
    a = ready(session, 1)
    session.collect(session.dispatch(batch(3, [a], recovery=[1])))
    with pytest.raises(ValueError, match="not newer"):
        session.dispatch(batch(3, [a], recovery=[1]))


def test_duplicate_keys_are_rejected_without_consuming_the_batch(session, adapter):
    a = ready(session, 1)
    with pytest.raises(ValueError, match="duplicate sequence keys"):
        session.dispatch(batch(0, [a, a], recovery=[4, 4]))

    assert session.sequence_state(a) is SequenceState.READY
    assert adapter.proposed_batches == []
    assert adapter.prefix(a) == [1, 2, 3]
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (OK,)
    assert adapter.prefix(a) == [1, 2, 3, 4]


def test_only_one_proposal_may_be_in_flight(session):
    a = ready(session, 1)
    b = ready(session, 2)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    with pytest.raises(RemoteDraftError, match="in flight"):
        session.dispatch(batch(1, [b], recovery=[2]))
    session.collect(handle)
    result = round_trip(session, batch(1, [b], recovery=[2]))
    assert result.row_statuses == (OK,)


def test_other_rpcs_cannot_consume_an_inflight_proposal(session):
    a = ready(session, 1)
    b = session.open_sequence(2)
    handle = session.dispatch(batch(0, [a], recovery=[1]))

    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.ping()
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.open_sequence(3)
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.prefill(b, features([2]), is_final=True)
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.close_sequences((b,))

    assert session.sequence_state(b) is SequenceState.PREFILLING
    assert session.sequence_state(SequenceKey("verifier-a", 3, 0)) is None
    assert session.collect(handle).row_statuses == (OK,)
    assert session.pending_frames == 0


def ragged_batch(batch_id, keys, query_lens, token_ids, *extra_slots):
    return RemoteProposalBatch(
        batch_id=batch_id,
        keys=tuple(keys),
        accepted_counts=tokens([0] * len(keys)),
        recovery_tokens=tokens(list(range(len(keys))), torch.int64),
        features=FeatureBatch(1, (tokens(token_ids), *extra_slots)),
        query_lens=tokens(query_lens),
    )


def test_ragged_token_features_follow_their_rows(session, adapter):
    # Token-level features carry each row's target tokens back to back
    # (MRV2's [num_tokens, ...] layout); skipping an unready row must drop
    # exactly its segment.
    a = ready(session, 1)
    skipped = session.open_sequence(2)
    b = ready(session, 3)
    result = round_trip(
        session,
        ragged_batch(0, [a, skipped, b], [2, 3, 1], [10, 11, 20, 21, 22, 30]),
    )
    assert result.row_statuses[0] is OK
    assert result.row_statuses[2] is OK
    assert adapter.round_tokens == {a: [10, 11], b: [30]}


@pytest.mark.parametrize(
    ("query_lens", "token_ids", "match"),
    [
        ([2], [1, 2], "query_lens must be"),
        ([-1, 3], [1, 2], "non-negative"),
        ([1, 1], [1, 2, 3], "expected 2 rows"),
    ],
)
def test_query_lens_must_describe_token_features(
    session, adapter, query_lens, token_ids, match
):
    a = ready(session, 1)
    b = ready(session, 2)
    with pytest.raises(ValueError, match=match):
        session.dispatch(ragged_batch(0, [a, b], query_lens, token_ids))
    assert adapter.proposed_batches == []
    # A rejected batch does not consume its id.
    assert round_trip(session, batch(0, [a], recovery=[1])).row_statuses == (OK,)


def test_per_sequence_slots_hold_one_row_per_key(clock):
    schema = TargetFeatureSchema(
        schema_id=1,
        slots=(
            FeatureSlot(kind="token_ids", dtype="int32", trailing_shape=()),
            FeatureSlot(
                kind="temperature",
                dtype="float32",
                trailing_shape=(),
                per_sequence=True,
            ),
        ),
    )
    adapter = FakeDraftAdapter(vocab_size=VOCAB, feature_schema=schema)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    identity = dataclasses.replace(
        IDENTITY, provided_features=("token_ids", "temperature")
    )
    session = RemoteDraftSession(server.endpoint, identity, clock=clock)
    temps = torch.tensor([0.5, 0.0, 1.0])
    try:
        session.connect()
        a = session.open_sequence(1)
        with pytest.raises(ValueError, match="temperature"):
            session.prefill(
                a, FeatureBatch(1, (tokens([1, 2]), temps[:2])), is_final=True
            )
        session.prefill(a, FeatureBatch(1, (tokens([1, 2]), temps[:1])), is_final=True)
        skipped = session.open_sequence(2)
        b = session.open_sequence(3)
        session.prefill(b, FeatureBatch(1, (tokens([4]), temps[:1])), is_final=True)

        with pytest.raises(ValueError, match="temperature"):
            session.dispatch(
                ragged_batch(0, [a, skipped, b], [1, 1, 2], [7, 8, 9, 9], temps[:2])
            )
        result = round_trip(
            session,
            ragged_batch(0, [a, skipped, b], [1, 1, 2], [7, 8, 9, 9], temps),
        )
        assert result.row_statuses[0] is OK
        assert result.row_statuses[2] is OK
        assert adapter.round_tokens == {a: [7], b: [9, 9]}
    finally:
        session.close()
        adapter.release()
        server.stop()


def test_dispatch_validates_schema_and_shapes(session):
    a = ready(session, 1)
    wrong_schema = RemoteProposalBatch(
        batch_id=0,
        keys=(a,),
        accepted_counts=tokens([0]),
        recovery_tokens=tokens([1]),
        features=FeatureBatch(schema_id=2, slots=(tokens([1]),)),
    )
    with pytest.raises(ValueError, match="schema"):
        session.dispatch(wrong_schema)
    with pytest.raises(ValueError, match="accepted_counts"):
        session.dispatch(
            RemoteProposalBatch(1, (a,), tokens([0, 0]), tokens([1]), features([1]))
        )
    with pytest.raises(ValueError, match="recovery_tokens"):
        session.dispatch(
            RemoteProposalBatch(
                2,
                (a,),
                tokens([0]),
                tokens([[1]]),
                features([1]),
            )
        )
    with pytest.raises(ValueError, match="required feature slot"):
        session.dispatch(
            RemoteProposalBatch(
                3, (a,), tokens([0]), tokens([1]), FeatureBatch(1, (None,))
            )
        )


# ----------------------------------------------------------------------
# Failure semantics
# ----------------------------------------------------------------------


def test_queue_full_latches_target_only_without_skipping_recovery(
    session, adapter, monkeypatch
):
    a = ready(session, 1)
    propose = adapter.advance_and_propose

    def busy_once(batch_):
        if batch_.batch_id == 0:
            result = propose(dataclasses.replace(batch_, active_rows=()))
            return dataclasses.replace(
                result,
                valid_lengths=tensor_to_frame(0, tokens([0])),
                row_statuses=(SpeculatorStatusCode.QUEUE_FULL,),
            )
        return propose(batch_)

    monkeypatch.setattr(adapter, "advance_and_propose", busy_once)
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (SpeculatorStatusCode.QUEUE_FULL,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY

    result = round_trip(session, batch(1, [a], recovery=[5]))
    assert result.row_statuses == (SpeculatorStatusCode.QUEUE_FULL,)
    assert result.output.valid_lengths.tolist() == [0]
    assert adapter.proposed_batches == [0]
    assert adapter.prefix(a) is None


def test_timeout_latches_target_only_and_purges_stale_result(session, adapter, clock):
    a = ready(session, 1)
    b = ready(session, 2)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a, b], recovery=[1, 2]))
    clock.advance(10.0)
    result = session.collect(handle)
    assert handle.state is HandleState.COLLECTED
    assert result.row_statuses == (
        SpeculatorStatusCode.TIMEOUT,
        SpeculatorStatusCode.TIMEOUT,
    )
    assert result.output.valid_lengths.tolist() == [0, 0]
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY

    # The late result must never surface: the next round only sees the
    # fresh sequence and the stale frames are dropped.
    adapter.release()
    c = ready(session, 3)
    result = round_trip(session, batch(1, [a, c], recovery=[8, 9]))
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT, OK)
    assert result.output.token_ids[1].tolist() == expected_row(9)
    assert result.output.valid_lengths.tolist() == [0, NUM_SPEC_TOKENS]
    assert session.pending_frames == 0
    assert adapter.proposed_batches == [0, 1]

    # Latched rows ignore prefill and stay latched until closed.
    session.prefill(a, features([1]), is_final=True)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    session.close_sequences((a,))
    assert session.sequence_state(a) is None


def test_dispatch_expired_during_encoding_sends_no_proposal(
    session, adapter, clock, monkeypatch
):
    # Nothing was written, so the round times out without tearing down the
    # connection the other requests still use.
    a = ready(session, 1)
    b = ready(session, 2)

    to_numpy = torch.Tensor.numpy

    def slow_host_copy(tensor, *args, **kwargs):
        array = to_numpy(tensor, *args, **kwargs)
        clock.advance(1.0)
        return array

    monkeypatch.setattr(torch.Tensor, "numpy", slow_host_copy)
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.connected
    assert adapter.proposed_batches == []

    monkeypatch.undo()
    assert round_trip(session, batch(1, [b], recovery=[5])).row_statuses == (OK,)


def test_large_writes_cannot_extend_the_round_deadline(make_session, monkeypatch):
    session = make_session(dataclasses.replace(IDENTITY, max_num_batched_tokens=1000))
    a = session.open_sequence(1)
    budgets = []
    sock = session._conn._sock
    settimeout = socket.socket.settimeout

    def record_budget(connection, timeout):
        if connection is sock and timeout is not None:
            budgets.append(timeout)
        return settimeout(connection, timeout)

    monkeypatch.setattr(socket.socket, "settimeout", record_budget)
    session.prefill(a, features(list(range(1000))), is_final=True)
    assert budgets and 0 < budgets[0] <= 0.101
    assert session.sequence_state(a) is SequenceState.READY


def test_open_prefill_and_proposal_share_one_round_budget(session, clock, monkeypatch):
    deadline = session.preparation_deadline()
    await_reply = session._await
    waits = []
    durations = iter((0.04, 0.08))

    def delayed(matches, reply_deadline, **kwargs):
        waits.append(reply_deadline)
        clock.advance(next(durations))
        return await_reply(matches, reply_deadline, **kwargs)

    monkeypatch.setattr(session, "_await", delayed)
    (key,) = session.open_sequences((1,), deadline=deadline)
    session.prefill_many([(key, features([1, 2]), True)], deadline=deadline)
    result = session.collect(session.dispatch(batch(0, [key], [7]), deadline=deadline))
    assert waits == [deadline, deadline]
    assert result.output.valid_lengths.tolist() == [0]
    assert session.sequence_state(key) in (
        SequenceState.TARGET_ONLY,
        SequenceState.INVALID,
    )


def timeout_after_frames(monkeypatch, conn, completed_frames):
    send_data = conn.send_data
    sent = 0

    def send(frame, *, deadline=None, drain_incoming=False):
        nonlocal sent
        if sent == completed_frames:
            raise TransportTimeout("deadline elapsed before the next frame")
        send_data(frame, deadline=deadline, drain_incoming=drain_incoming)
        sent += 1

    def reject_control(*args, **kwargs):
        raise TransportTimeout("deadline elapsed before the control message")

    monkeypatch.setattr(conn, "send_data", send)
    monkeypatch.setattr(conn, "send_control", reject_control)


@pytest.mark.parametrize("completed_frames", [0, 1, 3])
def test_request_timeout_preserves_only_an_untouched_stream(
    session, adapter, monkeypatch, completed_frames
):
    a = ready(session, 1)
    b = ready(session, 2)
    conn = session._conn
    timeout_after_frames(monkeypatch, conn, completed_frames)

    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    assert result.output.valid_lengths.tolist() == [0]
    assert adapter.proposed_batches == []
    assert session.connected is (completed_frames == 0)
    assert conn.closed is (completed_frames != 0)

    monkeypatch.undo()
    if completed_frames:
        assert session.sequence_state(b) is SequenceState.INVALID
        session.connect()
        b = ready(session, 3)
    assert round_trip(session, batch(1, [b], recovery=[5])).row_statuses == (OK,)


@pytest.mark.parametrize("completed_frames", [0, 1, 3])
def test_result_timeout_closes_stream_and_allows_reconnect(
    server, monkeypatch, completed_frames
):
    session = RemoteDraftSession(server.endpoint, IDENTITY, request_timeout_ms=1000)
    try:
        session.connect()
        a = ready(session, 1)
        conn = next(iter(server._connections))
        timeout_after_frames(monkeypatch, conn, completed_frames)

        result = round_trip(session, batch(0, [a], recovery=[4]))
        assert result.output.valid_lengths.tolist() == [0]
        assert not session.connected
        assert conn.closed
        assert session.pending_frames == 0

        monkeypatch.undo()
        session.connect()
        b = ready(session, 2)
        result = round_trip(session, batch(0, [b], recovery=[5]))
        assert result.row_statuses == (OK,)
        assert result.output.token_ids[0].tolist() == expected_row(5)
        assert session.pending_frames == 0
    finally:
        session.close()


def test_collect_timeout_does_not_write_after_deadline(
    session, adapter, clock, monkeypatch
):
    a = ready(session, 1)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a], recovery=[4]))
    clock.advance(1.0)

    sock = session._conn._sock
    sendall = socket.socket.sendall

    def reject_late_write(connection, *args, **kwargs):
        if connection is sock:
            pytest.fail("fallback must not wait on a cancellation write")
        return sendall(connection, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "sendall", reject_late_write)
    result = session.collect(handle)
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    assert result.output.valid_lengths.tolist() == [0]
    assert not session.connected
    adapter.release()


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("num_requests", [1, 2])
def test_all_timed_out_requests_retire_on_disconnect_without_another_rpc(
    make_session, adapter, server, clock, monkeypatch, policy, num_requests
):
    session = make_session(failure_policy=policy)
    keys = tuple(ready(session, i) for i in range(num_requests))
    retired = threading.Event()
    close = adapter.close_sequences

    def close_sequences(closed):
        close(closed)
        if set(keys).issubset(closed):
            retired.set()

    monkeypatch.setattr(adapter, "close_sequences", close_sequences)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, keys, recovery=[4] * num_requests))
    clock.advance(1.0)
    monkeypatch.setattr(
        session._conn, "send_control", Mock(side_effect=AssertionError("late write"))
    )
    expectation = (
        pytest.raises(RemoteDraftError, match="TIMEOUT")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expectation:
        session.collect(handle)
    assert not session.connected
    assert not session._retiring
    assert session._inflight_batch_id is None
    assert all(session.sequence_state(k) is SequenceState.TARGET_ONLY for k in keys)
    adapter.release()
    assert retired.wait(2), "idle terminal requests retained their remote state"
    assert all(server.registry.get(k) is None for k in keys)
    assert all(adapter.prefix(k) is None for k in keys)


def test_latched_rows_report_their_failure_once(session, adapter, clock, monkeypatch):
    a = ready(session, 1)
    b = ready(session, 2)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    clock.advance(10.0)
    assert session.collect(handle).row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    adapter.release()

    warning = Mock()
    monkeypatch.setattr(session_module.logger, "warning", warning)
    for batch_id in (1, 2, 3):
        result = round_trip(session, batch(batch_id, [a, b], recovery=[1, 2]))
        assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT, OK)
    warning.assert_not_called()


def test_configured_timeout_raises_after_latching(server, adapter, clock):
    config = RemoteDraftConfig(
        endpoint=server.endpoint,
        transport="inline",
        failure_policy="error",
        request_timeout_ms=250,
        startup_timeout_ms=1000,
    )
    session = RemoteDraftSession.from_config(config, IDENTITY, clock=clock)
    try:
        session.connect()
        a = ready(session, 1)
        adapter.hold_batches.add(0)
        handle = session.dispatch(batch(0, [a], recovery=[1]))
        assert handle.deadline == clock() + 0.25
        clock.advance(0.251)
        with pytest.raises(RemoteDraftError, match="TIMEOUT"):
            session.collect(handle)
        assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    finally:
        adapter.release()
        session.close()


def limited_server(**limits):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    fields = dict(
        max_batch_size=8, max_feature_tokens=64, max_sequences=64, max_model_len=64
    )
    fields.update(limits)
    server = RemoteDraftServer(adapter, limits=RemoteServerLimits(**fields))
    server.start("tcp://127.0.0.1:0")
    return server, adapter


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_rows_beyond_max_batch_size_follow_failure_policy(clock, policy):
    # Reachable only when the runner sends more rows than it declared; the
    # overflow must fall back to target-only rather than kill the engine,
    # unless the error policy asks for failures to surface.
    server, adapter = limited_server(max_batch_size=2)
    identity = dataclasses.replace(IDENTITY, max_num_seqs=2)
    session = RemoteDraftSession(
        server.endpoint, identity, failure_policy=policy, clock=clock
    )
    try:
        session.connect()
        keys = [ready(session, i) for i in range(3)]
        handle = session.dispatch(batch(0, keys, recovery=[1, 2, 3]))
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="QUEUE_FULL"):
                session.collect(handle)
        else:
            result = session.collect(handle)
            assert result.row_statuses == (OK, OK, SpeculatorStatusCode.QUEUE_FULL)
            assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS] * 2 + [0]
        assert session.sequence_state(keys[2]) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


def test_rows_beyond_max_feature_tokens_degrade_instead_of_raising(clock):
    server, adapter = limited_server(max_feature_tokens=4)
    identity = dataclasses.replace(IDENTITY, max_num_batched_tokens=4)
    session = RemoteDraftSession(server.endpoint, identity, clock=clock)
    try:
        session.connect()
        keys = [ready(session, i, prompt=(1,)) for i in range(3)]
        result = round_trip(
            session, ragged_batch(0, keys, [3, 2, 1], [1, 2, 3, 4, 5, 6])
        )
        assert result.row_statuses == (OK, SpeculatorStatusCode.QUEUE_FULL, OK)
        assert adapter.round_tokens == {keys[0]: [1, 2, 3], keys[2]: [6]}
        assert session.sequence_state(keys[1]) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_prefill_beyond_server_limits_follows_failure_policy(
    clock, monkeypatch, policy
):
    server, adapter = limited_server(max_feature_tokens=4, max_model_len=6)
    prefill = Mock(wraps=adapter.prefill)
    monkeypatch.setattr(adapter, "prefill", prefill)
    identity = dataclasses.replace(IDENTITY, max_num_batched_tokens=4, max_model_len=6)
    session = RemoteDraftSession(
        server.endpoint, identity, failure_policy=policy, clock=clock
    )
    try:
        session.connect()
        too_big = session.open_sequence(1)
        too_long = session.open_sequence(2)
        session.prefill(too_long, features([1, 2, 3, 4]), is_final=False)
        assert adapter.prefix(too_long) == [1, 2, 3, 4]
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="max_feature_tokens"):
                session.prefill(too_big, features([1] * 5), is_final=True)
            with pytest.raises(RemoteDraftError, match="max_model_len"):
                session.prefill(too_long, features([5, 6, 7]), is_final=True)
        else:
            session.prefill(too_big, features([1] * 5), is_final=True)
            session.prefill(too_long, features([5, 6, 7]), is_final=True)
        assert session.sequence_state(too_big) is SequenceState.TARGET_ONLY
        assert session.sequence_state(too_long) is SequenceState.TARGET_ONLY
        assert session.ping().active_sequences == 0
        prefill.assert_called_once()
        assert adapter.prefix(too_big) is None and adapter.prefix(too_long) is None
        assert server.registry.get(too_big) is None
        assert server.registry.get(too_long) is None
    finally:
        session.close()
        server.stop()


def test_stale_generation_desyncs_then_reopen_recovers(session, server, adapter):
    a = ready(session, 1)
    newer = SequenceKey(a.verifier_instance_id, a.sequence_id, a.generation + 1)
    replacement = transport_module.connect(server.endpoint, timeout=5)
    try:
        replacement.send_control(
            encode_message(
                Hello(
                    verifier_instance_id=IDENTITY.verifier_instance_id,
                    target_fingerprint=IDENTITY.target_fingerprint,
                    tokenizer_fingerprint=IDENTITY.tokenizer_fingerprint,
                    method=IDENTITY.method,
                    num_speculative_tokens=NUM_SPEC_TOKENS,
                    parallel_drafting=False,
                    supported_transports=("inline",),
                ),
                request_id=1,
            )
        )
        ack = decode_payload(decode_envelope(replacement.recv(timeout=5).body))
        assert isinstance(ack, HelloAck)
        replacement.send_control(
            encode_message(OpenSequence(newer), session_id=ack.session_id, request_id=2)
        )
        opened = decode_payload(decode_envelope(replacement.recv(timeout=5).body))
        assert isinstance(opened, SequenceAck) and opened.status is OK

        result = round_trip(session, batch(0, [a], recovery=[1]))
        assert result.row_statuses == (SpeculatorStatusCode.STALE_GENERATION,)
        assert session.sequence_state(a) is SequenceState.DESYNCED

        session.open_sequence(2)
        reopened = session.open_sequence(1)
        assert reopened.generation > newer.generation
        session.prefill(reopened, features([4, 5]), is_final=True)
        result = round_trip(session, batch(1, [reopened], recovery=[6]))
        assert result.row_statuses == (OK,)
        assert adapter.prefix(reopened) == [4, 5, 6]
        assert adapter.prefix(newer) is None
    finally:
        replacement.close()


def test_server_shutdown_degrades_to_target_only(session, server, adapter):
    a = ready(session, 1)
    adapter.release()
    server.stop()

    handle = session.dispatch(batch(0, [a], recovery=[1]))
    result = session.collect(handle)
    assert result.output.valid_lengths.tolist() == [0]
    assert result.row_statuses[0] is not OK
    assert not session.connected
    assert session.sequence_state(a) is SequenceState.INVALID

    later = session.open_sequence(2)
    assert session.sequence_state(later) is SequenceState.INVALID
    session.prefill(later, features([1]), is_final=True)
    result = round_trip(session, batch(1, [a, later], recovery=[1, 2]))
    assert result.output.valid_lengths.tolist() == [0, 0]
    session.close_sequences((a, later))
    assert session.sequence_state(later) is None


def test_server_shutdown_raises_under_error_policy(make_session, server, adapter):
    session = make_session(failure_policy="error")
    a = ready(session, 1)
    adapter.release()
    server.stop()
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    with pytest.raises(RemoteDraftError):
        session.collect(handle)


def test_open_queue_full_latches_target_only(clock):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=8,
            max_feature_tokens=64,
            max_sequences=1,
            max_model_len=64,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
    strict = RemoteDraftSession(
        server.endpoint, IDENTITY, failure_policy="error", clock=clock
    )
    try:
        session.connect()
        a = ready(session, 1)
        full = session.open_sequence(2)
        assert session.sequence_state(full) is SequenceState.TARGET_ONLY
        # The rejected sequence must degrade quietly, not wedge in OPENING:
        # prefill is a no-op and rounds skip the row with valid_length=0.
        session.prefill(full, features([1]), is_final=True)
        result = round_trip(session, batch(0, [a, full], recovery=[3, 4]))
        assert result.row_statuses == (OK, SpeculatorStatusCode.QUEUE_FULL)
        assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS, 0]

        strict.connect()
        with pytest.raises(RemoteDraftError, match="QUEUE_FULL"):
            strict.open_sequence(9)
    finally:
        session.close()
        strict.close()
        adapter.release()
        server.stop()


# ----------------------------------------------------------------------
# Reconnect
# ----------------------------------------------------------------------


def test_reconnect_completes_rounds_with_fresh_wire_state(session):
    a = ready(session, 1)
    assert round_trip(session, batch(0, [a], recovery=[5])).row_statuses == (OK,)
    old_epoch = session.session_epoch
    session.close()
    assert not session.connected

    session.connect()
    assert session.connected
    assert session.session_epoch != old_epoch
    b = session.open_sequence(1)
    assert b.generation == a.generation + 1
    session.prefill(b, features([1, 2]), is_final=True)
    # batch_id 0 repeats and the new server session numbers its responses
    # from 0 again; both must be accepted because that state is scoped to
    # the connection, or every post-reconnect round would look stale.
    result = round_trip(session, batch(0, [b], recovery=[7]))
    assert result.row_statuses == (OK,)
    assert result.output.token_ids.tolist() == [expected_row(7)]
    assert session.pending_frames == 0


def test_lost_connection_reconnects_with_backoff(make_session, server, clock):
    session = make_session()
    endpoint = server.endpoint
    old = ready(session, 1)
    server.stop()
    round_trip(session, batch(0, [old], recovery=[1]))
    assert not session.connected

    # No server yet: one attempt, then quiet until the backoff elapses.
    connects = Mock(wraps=session_module.transport_connect)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(session_module, "transport_connect", connects)
        assert session.sequence_state(session.open_sequence(2)) is SequenceState.INVALID
        assert session.sequence_state(session.open_sequence(3)) is SequenceState.INVALID
        assert connects.call_count == 1

        restarted = RemoteDraftServer(FakeDraftAdapter(vocab_size=VOCAB))
        restarted.start(endpoint)
        try:
            clock.advance(1.0)
            fresh = ready(session, 4)
            assert connects.call_count == 2
            assert session.connected
            result = round_trip(session, batch(0, [old, fresh], recovery=[1, 5]))
            assert result.row_statuses[1] is OK
            assert result.output.token_ids[1].tolist() == expected_row(5)
            assert session.sequence_state(old) is SequenceState.INVALID
        finally:
            session.close()
            restarted.stop()


def test_open_after_a_silent_restart_uses_the_new_server(make_session, server):
    # The session only learns that the old connection is dead when the
    # open fails on it; that request must still get the new connection.
    session = make_session()
    endpoint = server.endpoint
    old = ready(session, 1)
    server.stop()
    restarted = RemoteDraftServer(FakeDraftAdapter(vocab_size=VOCAB))
    restarted.start(endpoint)
    try:
        key = session.open_sequence(2)
        assert session.connected
        assert key.generation > old.generation
        assert session.sequence_state(key) is SequenceState.PREFILLING
    finally:
        session.close()
        restarted.stop()


def test_error_reply_to_an_earlier_request_is_ignored(protocol_peer, clock):
    def send_stale_error(peer):
        envelope, message, _ = peer.recv()
        assert isinstance(message, OpenSequence)
        peer.conn.send_control(
            encode_message(
                ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, "abandoned"),
                session_id="s",
                request_id=envelope.request_id - 1,
            )
        )
        peer.reply(envelope, SequenceAck(message.key))

    endpoint = protocol_peer(send_stale_error)
    session = RemoteDraftSession(endpoint, IDENTITY, clock=clock)
    try:
        session.connect()
        key = session.open_sequence(1)
        assert session.sequence_state(key) is SequenceState.PREFILLING
    finally:
        session.close()


def test_reconnect_fences_handles_from_the_old_connection(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    old_session_id = handle.session_id
    session.close()
    session.connect()
    assert session.session_id != old_session_id
    # Simulate a server restart that reuses an epoch. The random session ID
    # remains authoritative even when an integer epoch happens to collide.
    handle.session_epoch = session.session_epoch
    with pytest.raises(RemoteDraftError, match="belongs to session"):
        session.collect(handle)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_open_twice_is_an_error(session):
    session.open_sequence(1)
    with pytest.raises(RemoteDraftError, match="already open"):
        session.open_sequence(1)


def test_open_many_pipelines_generations_without_refreshing_deadline(
    session, clock, monkeypatch
):
    previous = session.open_sequence(1)
    session.close_sequences((previous,))
    deadline = session.preparation_deadline()
    send, wait = session._send, session._await
    events = []
    deadlines = []

    def write(*args, **kwargs):
        events.append("send")
        deadlines.append(kwargs["deadline"])
        result = send(*args, **kwargs)
        clock.advance(0.02)
        return result

    def read(matches, ack_deadline, **kwargs):
        events.append("wait")
        deadlines.append(ack_deadline)
        return wait(matches, ack_deadline, **kwargs)

    monkeypatch.setattr(session, "_send", write)
    monkeypatch.setattr(session, "_await", read)
    keys = session.open_sequences((1, 2), deadline=deadline)
    assert events == ["send", "send", "wait", "wait"]
    assert deadlines == [deadline] * 4
    assert keys[0].generation > previous.generation
    assert keys[1].generation > keys[0].generation
    assert session.sequence_state(previous) is None
    assert all(session.sequence_state(key) is SequenceState.PREFILLING for key in keys)


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("failure", ["record", "ownership", "after_open"])
def test_open_allocation_failure_reclaims_unreturned_keys(
    make_session, server, adapter, monkeypatch, policy, failure
):
    session = make_session(failure_policy=policy)
    sibling = ready(session, 99)
    conn = session._conn
    generation = session._next_generation
    stale = SequenceKey(IDENTITY.verifier_instance_id, 1, generation)
    cleaned = threading.Event()
    close = adapter.close_sequences

    def close_and_notify(keys):
        close(keys)
        if sibling in keys:
            cleaned.set()

    monkeypatch.setattr(adapter, "close_sequences", close_and_notify)
    with monkeypatch.context() as patch:
        if failure == "record":
            make_record = session_module._SequenceRecord

            def allocate(key, state):
                if key.sequence_id == 2:
                    raise MemoryError("injected open allocation failure")
                return make_record(key, state)

            patch.setattr(session_module, "_SequenceRecord", allocate)
        elif failure == "ownership":

            class Records(dict):
                def __setitem__(self, sequence_id, record):
                    if sequence_id == 2:
                        raise MemoryError("injected open allocation failure")
                    super().__setitem__(sequence_id, record)

            # Keep the failed dictionary visible after the patch context.
            session._records = Records(session._records)
        else:
            open_many = session._open_many

            def fail_after_open(*args, **kwargs):
                open_many(*args, **kwargs)
                raise MemoryError("injected open allocation failure")

            patch.setattr(session, "_open_many", fail_after_open)
        with pytest.raises(MemoryError, match="open allocation"):
            session.open_sequences((1, 2))
    assert not session.connected and conn.closed
    assert set(session._records) == {99}
    assert session.sequence_state(sibling) is SequenceState.INVALID
    assert not session._retiring
    assert cleaned.wait(2), "failed open retained connection-owned peer state"
    assert len(server.registry) == 0
    assert adapter.prefix(sibling) is None

    next_generation = session._next_generation
    session._records = dict(session._records)
    session.connect()
    session.close_sequences((sibling,))
    fresh = session.open_sequences((1, 2))
    assert fresh[0].generation >= next_generation > generation
    assert fresh[1].generation > fresh[0].generation
    session.close_sequences((stale,))
    session.prefill_many([(key, features([1]), True) for key in fresh])
    assert round_trip(session, batch(0, fresh, [4, 5])).row_statuses == (OK, OK)


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_open_many_partial_ack_disconnect_keeps_every_record_invalid(
    make_session, clock, monkeypatch, policy
):
    session = make_session(failure_policy=policy)
    wait = session._await
    acknowledged: list[SequenceAck] = []
    deadline = session.preparation_deadline()

    def disconnect(matches, ack_deadline, **kwargs):
        if acknowledged:
            clock.advance(0.1)
            raise TransportError("connection lost after first open ACK")
        reply = wait(matches, ack_deadline, **kwargs)
        acknowledged.append(reply[1])
        return reply

    monkeypatch.setattr(session, "_await", disconnect)
    expectation = (
        pytest.raises(RemoteDraftError, match="INTERNAL_ERROR")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expectation:
        session.open_sequences((1, 2), deadline=deadline)
    assert len(acknowledged) == 1
    assert not session.connected
    assert len(session._records) == 2
    assert all(
        record.state is SequenceState.INVALID for record in session._records.values()
    )


def test_open_many_reconnect_uses_remaining_preparation_budget(
    session, clock, monkeypatch
):
    session._invalidate("test reconnect")
    deadline = session.preparation_deadline()
    clock.advance(0.04)
    connect = session.connect
    budgets = []

    def reconnect(*, timeout_s):
        budgets.append(timeout_s)
        connect(timeout_s=timeout_s)
        clock.advance(0.06)

    monkeypatch.setattr(session, "connect", reconnect)
    keys = session.open_sequences((1, 2), deadline=deadline)
    assert budgets == [pytest.approx(0.06)]
    assert all(session.sequence_state(key) is SequenceState.TARGET_ONLY for key in keys)
    assert all(
        session._records[key.sequence_id].last_status is SpeculatorStatusCode.TIMEOUT
        for key in keys
    )


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_open_many_applies_every_ack_before_enforcing_failure_policy(
    make_session, adapter, monkeypatch, policy
):
    session = make_session(failure_policy=policy)
    open_sequence = adapter.open_sequence

    def reject_second(key):
        if key.sequence_id == 2:
            raise AdapterError(SpeculatorStatusCode.QUEUE_FULL, "test admission limit")
        return open_sequence(key)

    monkeypatch.setattr(adapter, "open_sequence", reject_second)
    expectation = (
        pytest.raises(RemoteDraftError, match="QUEUE_FULL")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expectation:
        session.open_sequences((1, 2, 3))
    keys = tuple(session._records[i].key for i in (1, 2, 3))
    assert [session.sequence_state(key) for key in keys] == [
        SequenceState.PREFILLING,
        SequenceState.TARGET_ONLY,
        SequenceState.PREFILLING,
    ]
    session.prefill_many(
        [(keys[0], features([1]), True), (keys[2], features([3]), True)]
    )
    result = round_trip(session, batch(0, keys, recovery=[4, 5, 6]))
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS, 0, NUM_SPEC_TOKENS]


def test_acknowledged_close_forgets_the_sequence(session, adapter):
    for sequence_id in range(50):
        previous = ready(session, sequence_id)
        session.close_sequences((previous,))
    assert not session._records
    reopened = ready(session, 1)
    assert reopened.generation > previous.generation
    result = round_trip(session, batch(0, [reopened], recovery=[4]))
    assert result.row_statuses == (OK,)


@pytest.mark.parametrize("failure_policy", ["target_only", "error"])
def test_unsent_close_timeout_disconnects_and_reclaims_server_requests(
    make_session, server, adapter, clock, monkeypatch, failure_policy
):
    session = make_session(failure_policy=failure_policy)
    a, b = ready(session, 1), ready(session, 2)
    conn = session._conn
    cleaned = threading.Event()
    close = adapter.close_sequences

    def close_sequences(keys):
        close(keys)
        if set(keys) == {a, b}:
            cleaned.set()

    monkeypatch.setattr(adapter, "close_sequences", close_sequences)
    # The first clock read creates the close budget; it expires before writing.
    clock.after_read = 0.101
    session.close_sequences((a,))
    assert not session.connected
    assert conn.closed
    assert session.sequence_state(a) is None
    assert session.sequence_state(b) is SequenceState.INVALID
    assert cleaned.wait(2), "disconnect did not release server-owned requests"
    assert len(server.registry) == 0
    assert adapter.prefix(a) is None and adapter.prefix(b) is None
    session.connect()
    fresh = ready(session, a.sequence_id)
    assert fresh.generation > a.generation
    assert round_trip(session, batch(0, [fresh], [4])).row_statuses == (OK,)


@pytest.mark.parametrize("failure_policy", ["target_only", "error"])
def test_sent_close_ack_timeout_keeps_other_requests_usable(
    protocol_peer, clock, failure_policy
):
    closed = threading.Event()

    def withhold_close_ack(peer):
        for _ in range(4):
            envelope, message, _ = peer.recv()
            assert isinstance(message, (OpenSequence, PrefillChunk))
            peer.reply(envelope, SequenceAck(message.key))
        _, close, _ = peer.recv()
        assert isinstance(close, CloseSequence) and len(close.keys) == 1
        closed.set()
        clock.advance(0.101)
        envelope, proposal, _ = peer.recv()
        assert proposal.keys[0] != close.keys[0]
        peer.reply(
            envelope,
            ProposalResponse(
                batch_id=proposal.batch_id, sequence_number=0, result_slot=0
            ),
            frames=(
                tensor_to_frame(0, tokens([expected_row(4)], torch.int64)),
                tensor_to_frame(1, tokens([NUM_SPEC_TOKENS])),
                tensor_to_frame(2, tokens([int(OK)])),
            ),
        )

    endpoint = protocol_peer(withhold_close_ack)
    session = RemoteDraftSession(
        endpoint, IDENTITY, failure_policy=failure_policy, clock=clock
    )
    try:
        session.connect()
        a, b = ready(session, 1), ready(session, 2)
        session.close_sequences((a,))
        assert closed.wait(2)
        assert session.connected
        assert session.sequence_state(a) is None
        assert session.sequence_state(b) is SequenceState.READY
        result = round_trip(session, batch(0, [b], [4]))
        assert result.row_statuses == (OK,)
        assert result.output.token_ids.tolist() == [expected_row(4)]
    finally:
        session.close()


def test_unacknowledged_close_keeps_bumping_the_generation(session):
    first = ready(session, 1)
    session.close()
    session.close_sequences((first,))
    session.connect()
    second = session.open_sequence(1)
    assert second.generation == first.generation + 1


def test_disconnected_sequence_churn_does_not_retain_request_ids(
    make_session, server, clock
):
    session = make_session()
    first = ready(session, 1)
    endpoint = server.endpoint
    server.stop()
    round_trip(session, batch(0, [first], recovery=[4]))
    session.close_sequences((first,))

    # The fake clock holds reconnect backoff steady during the outage.
    for sequence_id in range(10_000):
        key = session.open_sequence(sequence_id)
        assert session.sequence_state(key) is SequenceState.INVALID
        session.close_sequences((key,))
    assert not session._records
    assert session.pending_frames == 0

    restarted = RemoteDraftServer(FakeDraftAdapter(vocab_size=VOCAB))
    restarted.start(endpoint)
    try:
        clock.advance(1.0)
        fresh = ready(session, first.sequence_id)
        assert fresh.generation > key.generation
        session.close_sequences((first,))
        assert session.sequence_state(fresh) is SequenceState.READY
        result = round_trip(session, batch(0, [fresh], recovery=[5]))
        assert result.row_statuses == (OK,)
    finally:
        session.close()
        restarted.stop()


def test_unacknowledged_close_fences_server_state_without_retaining_history(
    server, adapter, clock, protocol_peer
):
    def defer_close(peer):
        upstream = transport_module.connect(server.endpoint, timeout=5)
        try:
            upstream.send_control(encode_message(peer.hello, request_id=1))
            ack = decode_payload(decode_envelope(upstream.recv(timeout=5).body))
            assert isinstance(ack, HelloAck)
            with contextlib.suppress(transport_module.ConnectionClosed):
                while True:
                    envelope, message, frames = peer.recv()
                    if isinstance(message, CloseSequence):
                        # Keep the old state on the real server until a newer OPEN.
                        clock.advance(0.101)
                        continue
                    for frame in frames:
                        upstream.send_data(frame)
                    upstream.send_control(
                        encode_message(
                            message,
                            session_id=ack.session_id,
                            request_id=envelope.request_id,
                        )
                    )
                    response = upstream.recv(timeout=5)
                    peer.reply(envelope, decode_payload(decode_envelope(response.body)))
        finally:
            upstream.close()

    endpoint = protocol_peer(defer_close)
    session = RemoteDraftSession(endpoint, IDENTITY, clock=clock)
    try:
        session.connect()
        first = ready(session, 1)
        session.close_sequences((first,))
        assert adapter.prefix(first) == [1, 2, 3]
        second = ready(session, first.sequence_id, prompt=(4, 5))
        assert second.generation > first.generation
        assert adapter.prefix(first) is None
        assert adapter.prefix(second) == [4, 5]
        session.close_sequences((first,))
        assert session.sequence_state(second) is SequenceState.READY
    finally:
        session.close()


def test_close_is_idempotent(session, server, adapter):
    a = ready(session, 1)
    b = ready(session, 2)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, 99)
    unknown = SequenceKey("verifier-a", 42, 0)
    session.close_sequences((a, stale, unknown))
    session.close_sequences((a, b))
    session.close_sequences((b,))
    assert session.sequence_state(a) is None
    assert session.sequence_state(b) is None
    assert len(server.registry) == 0
    assert adapter.prefix(a) is None


@pytest.mark.parametrize(
    "num_requests,num_tokens,send_buffer", [(256, 1, None), (32, 64, 4096)]
)
@pytest.mark.timeout(15)
def test_prefill_many_drains_acks_under_socket_backpressure(
    tmp_path_factory, adapter, num_requests, num_tokens, send_buffer
):
    server = RemoteDraftServer(adapter)
    server.start(f"unix://{tmp_path_factory.mktemp('pf')}/prefill.sock")
    identity = dataclasses.replace(IDENTITY, max_num_seqs=num_requests)
    session = RemoteDraftSession(server.endpoint, identity, request_timeout_ms=2000)
    try:
        session.connect()
        keys = [session.open_sequence(i) for i in range(num_requests)]
        if send_buffer is not None:
            peer = next(iter(server._connections))
            for conn in (session._conn, peer):
                assert conn is not None
                conn._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer)
        prompt = [1] * num_tokens
        session.prefill_many([(key, features(prompt), True) for key in keys])
        assert session.connected
        assert all(session.sequence_state(key) is SequenceState.READY for key in keys)
        assert all(adapter.prefix(key) == prompt for key in keys)
        assert session.ping().active_sequences == num_requests
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize("send_buffer", [None, 8192])
@pytest.mark.parametrize("flush_before_request", [False, True])
@pytest.mark.timeout(20)
def test_retirement_acks_cannot_block_a_later_inline_request(
    tmp_path_factory, adapter, send_buffer, flush_before_request
):
    """Retirement writes can finish before the peer's ACK writes finish."""
    num_tokens = 131072
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=256,
            max_feature_tokens=num_tokens,
            max_sequences=256,
            max_model_len=num_tokens + 1,
        ),
    )
    server.start(f"unix://{tmp_path_factory.mktemp('ra')}/retirement.sock")
    identity = dataclasses.replace(
        IDENTITY,
        max_num_seqs=256,
        max_num_batched_tokens=num_tokens,
        max_model_len=num_tokens + 1,
    )
    session = RemoteDraftSession(server.endpoint, identity, request_timeout_ms=1000)
    try:
        session.connect()
        keys = [session.open_sequence(i) for i in range(256)]
        if send_buffer is not None:
            peer = next(iter(server._connections))
            for conn in (session._conn, peer):
                assert conn is not None
                conn._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_buffer)
        retired, healthy = tuple(keys[:-1]), keys[-1]
        deadline = session.preparation_deadline() if flush_before_request else 0.0
        session.fail_sequences(retired, detail="feature OOM", deadline=deadline)
        assert bool(session._retiring) is not flush_before_request

        session.prefill(healthy, features([1] * num_tokens), is_final=True)
        assert session.connected
        assert session.sequence_state(healthy) is SequenceState.READY
        assert all(
            session.sequence_state(key) is SequenceState.TARGET_ONLY for key in retired
        )
        assert adapter.prefix(healthy) == [1] * num_tokens
        assert session.ping().active_sequences == 1
        result = round_trip(session, batch(0, [healthy], [4]))
        assert result.row_statuses == (OK,)
        assert result.output.token_ids.tolist() == [expected_row(4)]
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("is_final", [True, False])
def test_prefill_acknowledgements_cannot_revive_a_lost_connection(
    protocol_peer, policy, is_final
):
    def disconnect_after_one_ack(peer):
        for _ in range(2):
            envelope, message, _ = peer.recv()
            assert isinstance(message, OpenSequence)
            peer.reply(envelope, SequenceAck(message.key))
        requests = [peer.recv() for _ in range(2)]
        assert all(isinstance(message, PrefillChunk) for _, message, _ in requests)
        envelope, chunk, _ = requests[0]
        peer.reply(envelope, SequenceAck(chunk.key))
        # The fixture closes after this ACK; the second request gets EOF.

    endpoint = protocol_peer(disconnect_after_one_ack)
    session = RemoteDraftSession(endpoint, IDENTITY, failure_policy=policy)
    try:
        session.connect()
        keys = (session.open_sequence(1), session.open_sequence(2))
        expected = (
            pytest.raises(RemoteDraftError, match="prefill failed")
            if policy == "error"
            else contextlib.nullcontext()
        )
        with expected:
            session.prefill_many([(key, features([1]), is_final) for key in keys])
        assert not session.connected
        assert all(session.sequence_state(key) is SequenceState.INVALID for key in keys)
        if policy == "target_only":
            session.prefill_many([(key, features([2]), True) for key in keys])
            result = round_trip(session, batch(0, keys, recovery=[2, 3]))
            assert result.output.valid_lengths.tolist() == [0, 0]
            assert all(status is not OK for status in result.row_statuses)
    finally:
        session.close()


def test_prefill_many_pipelines_chunks_and_isolates_failed_rows(protocol_peer):
    def acknowledge_after_all_chunks(peer):
        for _ in range(3):
            envelope, message, _ = peer.recv()
            assert isinstance(message, OpenSequence)
            peer.reply(envelope, SequenceAck(message.key))
        # Withhold all replies until the whole batch reaches the wire.
        requests = [peer.recv() for _ in range(3)]
        for (envelope, chunk, frames), values, final in zip(
            requests, ([1, 2], [3], [4, 5]), (True, True, False)
        ):
            assert isinstance(chunk, PrefillChunk)
            assert chunk.offset == 0
            assert chunk.num_tokens == len(values)
            assert chunk.is_final is final
            assert transport_module.frame_to_ints(frames[0]) == values
            status = (
                SpeculatorStatusCode.SEQUENCE_DESYNC
                if chunk.key.sequence_id == 2
                else OK
            )
            peer.reply(envelope, SequenceAck(chunk.key, status))

    endpoint = protocol_peer(acknowledge_after_all_chunks)
    session = RemoteDraftSession(endpoint, IDENTITY)
    try:
        session.connect()
        a, b, c = (session.open_sequence(i) for i in (1, 2, 3))
        session.prefill_many(
            [
                (a, features([1, 2]), True),
                (b, features([3]), True),
                (c, features([4, 5]), False),
            ]
        )
        assert session.sequence_state(a) is SequenceState.READY
        assert session.sequence_state(b) is SequenceState.DESYNCED
        assert session.sequence_state(c) is SequenceState.PREFILLING
    finally:
        session.close()


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("sent_before_failure", [0, 1])
@pytest.mark.parametrize("timeout", [False, True])
def test_prefill_send_failure_reports_cause_for_unsent_rows(
    make_session, monkeypatch, policy, sent_before_failure, timeout
):
    session = make_session(failure_policy=policy)
    keys = tuple(session.open_sequence(i) for i in range(3))
    conn = session._conn
    assert conn is not None
    send_data = conn.send_data
    sent = 0
    error = TransportTimeout if timeout else TransportError

    def fail_send(*args, **kwargs):
        nonlocal sent
        if sent == sent_before_failure:
            raise error("injected send failure")
        send_data(*args, **kwargs)
        sent += 1

    monkeypatch.setattr(conn, "send_data", fail_send)
    report = Mock()
    monkeypatch.setattr(session_module.logger, "warning", report)
    expected = (
        pytest.raises(RemoteDraftError, match="prefill failed")
        if policy == "error"
        else contextlib.nullcontext()
    )
    with expected as raised:
        session.prefill_many([(key, features([1]), True) for key in keys])

    if policy == "error":
        assert raised is not None
        detail = str(raised.value)
    else:
        detail = next(
            call.args[1]
            for call in report.call_args_list
            if call.args[0] == "Remote draft falling back to target-only: %s"
        )
    if timeout:
        assert detail.count("TIMEOUT") == len(keys) - sent_before_failure
        assert "INTERNAL_ERROR" not in detail
    else:
        assert not session.connected
        assert all(session.sequence_state(key) is SequenceState.INVALID for key in keys)
        assert "TIMEOUT" not in detail
        assert detail.count("INTERNAL_ERROR injected send failure") == len(keys)


def test_prefill_rejects_stale_key_ready_sequence_and_duplicate_chunks(session):
    a = ready(session, 1)
    with pytest.raises(RemoteDraftError, match="cannot accept prefill"):
        session.prefill(a, features([1]), is_final=True)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, 7)
    with pytest.raises(ValueError, match="not current"):
        session.prefill(stale, features([1]), is_final=True)
    b = session.open_sequence(2)
    with pytest.raises(ValueError, match="one prefill chunk per sequence"):
        session.prefill_many([(b, features([6]), True), (b, features([7]), True)])


def test_ping_round_trip(session):
    ready(session, 1)
    pong = session.ping()
    assert pong.active_sequences == 1


# ----------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(6, dtype=torch.int32).reshape(2, 3),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        torch.empty(0, 4, dtype=torch.int64),
        torch.tensor(7, dtype=torch.int64),
    ],
    ids=["int32-2d", "bfloat16", "empty", "scalar"],
)
def test_tensor_frame_roundtrip(tensor):
    frame = tensor_to_frame(11, tensor)
    assert frame.slot == 11
    restored = frame_to_tensor(frame)
    assert restored.dtype == tensor.dtype
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)


@pytest.mark.parametrize(
    "kind", [transport_module.FRAME_CONTROL, transport_module.FRAME_DATA]
)
def test_client_handshake_rejects_header_before_receiving_body(monkeypatch, kind):
    raw, sock = socket.socketpair()
    conn = transport_module.FramedConnection(sock)
    session = RemoteDraftSession("tcp://127.0.0.1:1", IDENTITY)
    monkeypatch.setattr(session_module, "transport_connect", lambda *a, **kw: conn)
    size = (
        transport_module.MAX_HANDSHAKE_BYTES + 1
        if kind == transport_module.FRAME_CONTROL
        else 1
    )
    try:
        raw.sendall(transport_module._HEADER.pack(kind, size))
        with pytest.raises(RemoteDraftError, match="HELLO handshake failed.*exceeds"):
            session.connect()
        assert conn.closed
        assert len(conn._buffer) == 5
        assert session.pending_frames == session._inbound_bytes == 0
    finally:
        session.close()
        raw.close()


@pytest.mark.parametrize("overflow", ["bytes", "frames", "duplicate", "header"])
def test_client_bounds_pending_results_across_received_frames(protocol_peer, overflow):
    identity = dataclasses.replace(IDENTITY, max_num_seqs=1, num_speculative_tokens=16)

    def flood(peer):
        _, message, _ = peer.recv()
        assert isinstance(message, Ping)
        if overflow == "header":
            maximum = transport_module.check_data_frame_size(
                (1 << 63) - 1, "int64", (1, identity.num_speculative_tokens)
            )
            peer.conn._sock.sendall(
                transport_module._HEADER.pack(transport_module.FRAME_DATA, maximum + 1)
            )
            return
        values = [1] * 16 if overflow == "bytes" else [1]
        with contextlib.suppress(TransportError):
            for i in range(8):
                peer.conn.send_data(
                    transport_module.ints_to_frame(
                        0 if overflow == "duplicate" else i,
                        values,
                        dtype="int64",
                        shape=(1, len(values)),
                    )
                )

    endpoint = protocol_peer(flood)
    session = RemoteDraftSession(endpoint, identity)
    try:
        session.connect()
        with pytest.raises(RemoteDraftError, match="ping failed.*(exceed|duplicate)"):
            session.ping()
        assert not session.connected
        assert session.pending_frames == 0
    finally:
        session.close()


def test_consuming_results_replenishes_the_receive_budget(session):
    a = ready(session, 1)
    for round_id in range(3):
        assert round_trip(session, batch(round_id, [a], recovery=[1])).row_statuses == (
            OK,
        )
        assert session.pending_frames == session._inbound_bytes == 0
        assert not session._inbound_sizes
        assert session._conn._max_data_bytes == session._max_result_frame_bytes
