# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone draft server driven over the raw wire protocol: handshake
gating, idempotent sequence commands, batch ordering, and error replies."""

import contextlib
import dataclasses
import errno
import runpy
import socket
import threading
import time
import zlib
from unittest.mock import Mock

import msgspec
import pytest
import torch

from vllm.v1.spec_decode.remote import server as server_module
from vllm.v1.spec_decode.remote import transport as transport_module
from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.cuda_ipc import CudaIpcDescriptor
from vllm.v1.spec_decode.remote.protocol import (
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    FeatureSlot,
    Hello,
    HelloAck,
    OpenSequence,
    Ping,
    Pong,
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
    _MAX_INPUT_FRAMES,
    DEFAULT_LIMITS,
    FakeDraftAdapter,
    RemoteDraftServer,
    SequenceRegistry,
    StandaloneProposalBatch,
    StandaloneProposalResult,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_DATA,
    ConnectionClosed,
    DataFrame,
    DeviceFeature,
    Listener,
    connect,
    decode_data_frame,
    frame_checksum,
    frame_to_ints,
    ints_to_frame,
)

VERIFIER = "verifier-a"
NUM_SPEC_TOKENS = 4
VOCAB = 50
OK = SpeculatorStatusCode.OK


def test_module_entrypoint_reuses_canonical_adapter_exception_class(monkeypatch):
    canonical_main = Mock()
    monkeypatch.setattr(server_module, "main", canonical_main)
    # runpy reproduces `python -m`; the warning is expected because this
    # test already imported the module to replace its canonical entrypoint.
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module(server_module.__name__, run_name="__main__")
    canonical_main.assert_called_once_with()


def key(sequence_id: int, generation: int = 0, verifier: str = VERIFIER):
    return SequenceKey(
        verifier_instance_id=verifier,
        sequence_id=sequence_id,
        generation=generation,
    )


@pytest.mark.parametrize("fence", ["cleanup", "shutdown"])
@pytest.mark.parametrize("while_waiting", [False, True])
def test_fenced_teardown_quarantines_receiver_without_retiring_owners(
    monkeypatch, fence, while_waiting
):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter)
    conn, receiver = Mock(), Mock()
    server._connections.add(conn)
    owned = key(1)
    close_adapter = Mock(wraps=adapter.close_sequences)
    monkeypatch.setattr(adapter, "close_sequences", close_adapter)
    turn = server.adapter_turn

    def fenced_turn(**kwargs):
        setattr(server, f"_{fence}_failed", True)
        return turn(**kwargs)

    def run(session):
        session.session_id = "closing-session"
        session.ipc_receiver = receiver
        server.registry.open(owned, owner=session.session_id)
        adapter.open_sequence(owned)
        if while_waiting:
            monkeypatch.setattr(server, "adapter_turn", fenced_turn)
        else:
            setattr(server, f"_{fence}_failed", True)

    monkeypatch.setattr(server_module._ServerSession, "run", run)
    server._serve_connection(conn)
    receiver.quarantine.assert_called_once_with()
    receiver.close.assert_not_called()
    close_adapter.assert_not_called()
    assert server.registry.get(owned) is not None
    assert adapter.prefix(owned) == []
    conn.close.assert_called_once_with()
    assert conn not in server._connections


@pytest.mark.parametrize("stage", ["prefill", "propose"])
def test_fatal_feature_import_latches_health_and_quarantines_receiver(
    monkeypatch, stage
):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter)
    client = None
    receiver = Mock()

    def fail_import(session, frames, **kwargs):
        session.ipc_receiver = receiver
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    try:
        client = RawClient(server.start("tcp://127.0.0.1:0"))
        client.hello()
        client.ready(key(1))
        monkeypatch.setattr(
            server_module._ServerSession, "_resolve_features", fail_import
        )
        if stage == "prefill":
            assert client.open(key(2)).status is OK
            reply = client.prefill(key(2), [4], offset=0)
        else:
            reply = client.propose(0, [key(1)], [4])
        assert reply.status is SpeculatorStatusCode.INTERNAL_ERROR
        assert not client.call(Ping(nonce=42)).healthy
        assert server.wait(0)
        assert not server.healthy
    finally:
        if client is not None:
            client.conn.close()
        server.stop()
    receiver.quarantine.assert_called_once_with()
    receiver.close.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["ipc", "adapter"])
def test_cleanup_failure_releases_connection_and_preserves_failed_owners(
    monkeypatch, failure_stage
):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter)
    conn = Mock()
    server._connections.add(conn)
    owned, other = key(1), key(2)
    receiver = Mock()
    failure = RuntimeError(f"injected {failure_stage} cleanup failure")
    if failure_stage == "ipc":
        receiver.close.side_effect = failure
    close_sequences = adapter.close_sequences

    def close_with_failure(keys):
        close_sequences(keys)
        if failure_stage == "adapter":
            raise failure

    close_adapter = Mock(side_effect=close_with_failure)
    monkeypatch.setattr(adapter, "close_sequences", close_adapter)
    report_error = Mock()
    monkeypatch.setattr(server_module.logger, "exception", report_error)

    def run(session):
        session.session_id = "closing-session"
        session.ipc_receiver = receiver
        for sequence, owner in ((owned, session.session_id), (other, "other-session")):
            assert server.registry.open(sequence, owner=owner).status == OK
            adapter.open_sequence(sequence)

    monkeypatch.setattr(server_module._ServerSession, "run", run)
    server._serve_connection(conn)

    receiver.close.assert_called_once_with()
    close_adapter.assert_called_once_with((owned,))
    assert (server.registry.get(owned) is not None) == (failure_stage == "adapter")
    assert adapter.prefix(owned) is None
    assert server.registry.get(other) is not None
    assert adapter.prefix(other) == []
    conn.close.assert_called_once_with()
    assert conn not in server._connections
    report_error.assert_called_once()
    assert server._cleanup_failed == (failure_stage == "adapter")


@pytest.mark.parametrize(
    ("operation", "retired_count"),
    [
        ("close", 0),
        ("close", 1),
        ("discard", 0),
        ("discard", 1),
        ("cleanup", 0),
        ("cleanup", 1),
        ("cleanup", 2),
        ("replace", 0),
        ("replace", 1),
    ],
)
def test_retirement_failure_preserves_owners_and_fences_adapter(
    operation, retired_count
):
    """Failed device retirement retains owners, closes the peer, and fences work."""
    retirements = []

    class FailingRetirementAdapter(FakeDraftAdapter):
        def close_sequences(self, keys):
            retirements.append(keys)
            assert all(server.registry.get(sequence) is not None for sequence in keys)
            super().close_sequences(keys[:retired_count])
            raise MemoryError("injected block return failure")

        def advance_and_propose(self, batch):
            raise RuntimeError("injected adapter failure")

    adapter = FailingRetirementAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter)
    endpoint = server.start("tcp://127.0.0.1:0")
    first, second = RawClient(endpoint), RawClient(endpoint)
    owned, sibling, other = key(1), key(2), key(3)
    expected = (owned,) if operation == "replace" else (owned, sibling)
    try:
        assert isinstance(first.hello(), HelloAck)
        first.ready(owned)
        first.ready(sibling)
        assert isinstance(second.hello(), HelloAck)
        second.ready(other)

        with pytest.raises(ConnectionClosed):
            if operation == "close":
                first.send(CloseSequence((owned, sibling, other)))
                first.recv()
            elif operation == "discard":
                first.propose(0, [owned, sibling, other], recovery=[4, 5, 6])
            elif operation == "cleanup":
                first.conn._sock.shutdown(socket.SHUT_WR)
                first.recv()
            else:
                first.open(key(1, generation=1))
        assert server.wait(timeout=2)
        assert retirements == [expected]
        for sequence, owner in (
            (owned, first.session_id),
            (sibling, first.session_id),
            (other, second.session_id),
        ):
            entry = server.registry.get(sequence)
            assert entry is not None and entry.owner == owner
            assert adapter.prefix(sequence) == (
                None if sequence in expected[:retired_count] else [1, 2, 3]
            )
        assert server.registry.get(key(1, generation=1)) is None
        assert adapter.prefix(key(1, generation=1)) is None

        # Another live session cannot retry retirement or enter the failed adapter.
        with pytest.raises(ConnectionClosed):
            second.open(key(4))
        server.stop()
        with pytest.raises(server_module.SequenceCleanupFailed, match="restart"):
            server.start("tcp://127.0.0.1:0")
        assert retirements == [expected]
        assert len(server.registry) == 3
        assert server.queue_depth == 0
    finally:
        first.close()
        second.close()
        server.stop()


def checksum(*frames) -> int:
    value = 0
    for frame in frames:
        value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


class RawClient:
    """Minimal verifier that speaks the protocol directly."""

    def __init__(self, endpoint: str):
        self.conn = connect(endpoint, timeout=5.0)
        self.session_id = ""
        self.request_id = 0
        self.frames: dict[int, DataFrame] = {}
        self.next_slot = 100

    def send(self, message, *, frames=()):
        for frame in frames:
            self.conn.send_data(frame)
        self.request_id += 1
        self.conn.send_control(
            encode_message(
                message, session_id=self.session_id, request_id=self.request_id
            )
        )
        return self.request_id

    def recv(self, timeout: float = 5.0):
        while True:
            frame = self.conn.recv(timeout=timeout)
            if frame.kind == FRAME_DATA:
                data = decode_data_frame(frame.body)
                self.frames[data.slot] = data
                continue
            envelope = decode_envelope(frame.body)
            return envelope, decode_payload(envelope)

    def call(self, message, *, frames=()):
        request_id = self.send(message, frames=frames)
        envelope, reply = self.recv()
        assert envelope.request_id == request_id
        return reply

    def hello(self, **overrides) -> HelloAck:
        fields = dict(
            verifier_instance_id=VERIFIER,
            target_fingerprint="tf",
            tokenizer_fingerprint="kf",
            method="eagle3",
            num_speculative_tokens=NUM_SPEC_TOKENS,
            parallel_drafting=False,
            supported_transports=("inline",),
        )
        fields.update(overrides)
        reply = self.call(Hello(**fields))
        if isinstance(reply, HelloAck):
            self.session_id = reply.session_id
        return reply

    def open(self, k: SequenceKey) -> SequenceAck:
        return self.call(OpenSequence(key=k))

    def prefill(
        self,
        k: SequenceKey,
        tokens,
        *,
        offset: int,
        is_final=True,
        checksum_override=None,
    ) -> SequenceAck:
        slot = self.reserve(1)
        frame = ints_to_frame(slot, tokens)
        chunk = PrefillChunk(
            key=k,
            offset=offset,
            num_tokens=len(tokens),
            is_final=is_final,
            feature_slot=slot,
            checksum=checksum(frame)
            if checksum_override is None
            else checksum_override,
        )
        return self.call(chunk, frames=[frame])

    def ready(self, k: SequenceKey, tokens=(1, 2, 3)) -> None:
        assert self.open(k).status is OK
        assert self.prefill(k, list(tokens), offset=0).status is OK

    def propose(
        self,
        batch_id: int,
        keys,
        recovery,
        accepted=None,
        *,
        num_speculative_tokens=None,
    ):
        accepted = accepted or [0] * len(keys)
        base = self.reserve(3)
        frames = [
            ints_to_frame(base, accepted),
            ints_to_frame(base + 1, recovery),
            ints_to_frame(base + 2, recovery),
        ]
        message = AdvanceAndPropose(
            batch_id=batch_id,
            keys=tuple(keys),
            accepted_counts_slot=base,
            recovery_tokens_slot=base + 1,
            feature_slot=base + 2,
            num_speculative_tokens=num_speculative_tokens,
        )
        return self.call(message, frames=frames)

    def propose_with_frames(self, batch_id, keys, accepted, recovery, feature):
        base = self.reserve(3)
        frames = [
            DataFrame(base, accepted.dtype, accepted.shape, accepted.payload),
            DataFrame(base + 1, recovery.dtype, recovery.shape, recovery.payload),
            DataFrame(base + 2, feature.dtype, feature.shape, feature.payload),
        ]
        return self.call(
            AdvanceAndPropose(
                batch_id=batch_id,
                keys=tuple(keys),
                accepted_counts_slot=base,
                recovery_tokens_slot=base + 1,
                feature_slot=base + 2,
            ),
            frames=frames,
        )

    def propose_ragged(self, batch_id, keys, query_lens, token_ids, *, lens=True):
        base = self.reserve(4)
        frames = [
            ints_to_frame(base, [0] * len(keys)),
            ints_to_frame(base + 1, [1] * len(keys)),
            ints_to_frame(base + 3, token_ids),
        ]
        if lens:
            frames.append(ints_to_frame(base + 2, query_lens))
        return self.call(
            AdvanceAndPropose(
                batch_id=batch_id,
                keys=tuple(keys),
                accepted_counts_slot=base,
                recovery_tokens_slot=base + 1,
                feature_slot=base + 3,
                query_lens_slot=base + 2,
            ),
            frames=frames,
        )

    def result(self, response: ProposalResponse):
        base = response.result_slot
        assert base is not None
        tokens = self.frames.pop(base)
        valid = self.frames.pop(base + 1)
        statuses = self.frames.pop(base + 2)
        rows = tokens.shape[0]
        flat = frame_to_ints(tokens)
        return (
            [
                flat[i * NUM_SPEC_TOKENS : (i + 1) * NUM_SPEC_TOKENS]
                for i in range(rows)
            ],
            frame_to_ints(valid),
            [SpeculatorStatusCode(s) for s in frame_to_ints(statuses)],
        )

    def reserve(self, count: int) -> int:
        slot = self.next_slot
        self.next_slot += count
        return slot

    def close(self):
        self.conn.close()


def expected_row(last_token: int):
    return [(last_token + j + 1) % VOCAB for j in range(NUM_SPEC_TOKENS)]


class MalformedResultAdapter(FakeDraftAdapter):
    def __init__(self, mode: str):
        super().__init__(vocab_size=VOCAB)
        self.mode = mode

    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult:
        result = super().advance_and_propose(batch)
        if self.mode == "short_payload":
            valid_lengths = DataFrame(0, "int32", (len(batch.keys),), b"")
        else:
            valid_lengths = ints_to_frame(
                0, [batch.num_speculative_tokens + 1] * len(batch.keys)
            )
        return StandaloneProposalResult(
            result.token_ids, valid_lengths, result.row_statuses
        )


@pytest.fixture
def adapter():
    return FakeDraftAdapter(vocab_size=VOCAB)


@pytest.fixture
def server(adapter):
    server = RemoteDraftServer(
        adapter, server_id="srv", target_fingerprint="tf", tokenizer_fingerprint="kf"
    )
    server.start("tcp://127.0.0.1:0")
    yield server
    adapter.release()
    server.stop()


@pytest.fixture
def client(server):
    client = RawClient(server.endpoint)
    yield client
    client.close()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ----------------------------------------------------------------------
# Handshake
# ----------------------------------------------------------------------


def test_hello_ack_reports_adapter_capabilities(server, adapter, client):
    ack = client.hello()
    assert isinstance(ack, HelloAck)
    assert ack.server_id == "srv"
    assert ack.session_epoch >= 1
    assert ack.selected_transport == "inline"
    assert ack.capabilities == adapter.capabilities()
    assert ack.feature_schema == adapter.feature_schema()
    assert ack.limits == DEFAULT_LIMITS
    assert ack.feature_schema.slots[0].kind == TargetFeatureKind.TOKEN_IDS.value


@pytest.mark.parametrize("operation", ["prefill", "proposal"])
@pytest.mark.parametrize("import_outcome", ["ok", "failure", "expired"])
def test_cuda_ipc_ack_requires_completed_import_before_deadline(
    tmp_path, monkeypatch, operation, import_outcome
):
    observed: list[DataFrame | DeviceFeature | None] = []
    execution_deadlines: list[float | None] = []

    class CudaAdapter(FakeDraftAdapter):
        def cuda_ipc_device(self):
            return torch.device("cuda:0")

        def prefill(self, key, inputs):
            observed.extend(inputs.features)

        def advance_and_propose(self, batch):
            observed.extend(batch.features)
            execution_deadlines.append(batch.execution_deadline)
            return super().advance_and_propose(
                dataclasses.replace(batch, features=(None,))
            )

    adapter = CudaAdapter(
        vocab_size=VOCAB,
        feature_schema=TargetFeatureSchema(
            1, (FeatureSlot("last_hidden_states", "float32", (2,)),)
        ),
    )
    received = torch.tensor([[1.0, 2.0]])
    receiver = Mock()
    copy_budgets = []
    copy_deadlines = []

    def receive(*args, deadline):
        copy_budgets.append(deadline - time.monotonic())
        copy_deadlines.append(deadline)
        if import_outcome == "failure":
            raise RuntimeError("copy completion unknown")
        if import_outcome == "expired":
            # Completion may be observed after the host resumes past its budget.
            assert wait_until(lambda: time.monotonic() >= deadline)
        return received

    receiver.receive.side_effect = receive
    monkeypatch.setattr(server_module, "CudaIpcReceiver", Mock(return_value=receiver))
    server = RemoteDraftServer(adapter)
    server.start(f"unix://{tmp_path}/draft.sock")
    client = RawClient(server.endpoint)
    try:
        ack = client.hello(supported_transports=("cuda_ipc",))
        assert isinstance(ack, HelloAck)
        assert ack.selected_transport == "cuda_ipc"
        sequence = key(1)
        assert client.open(sequence).status is OK
        descriptor = CudaIpcDescriptor(
            buffer_id=1,
            generation=1,
            device_uuid="GPU-00000000-0000-0000-0000-000000000000",
            memory_handle=b"m" * 64,
            ready_event_handle=b"e" * 64,
            nbytes=8,
            capacity_bytes=8,
        )
        feature = DataFrame(100, "float32", (1, 2), b"", descriptor)
        message = PrefillChunk(
            sequence,
            0,
            1,
            False,
            100,
            frame_checksum([feature]),
            execution_timeout_ms=20 if import_outcome == "expired" else 1000,
            execution_group_id=1,
        )
        frames = (feature,)
        if operation == "proposal":
            inline = msgspec.structs.replace(
                feature, payload=received.numpy().tobytes(), cuda_ipc=None
            )
            initial = msgspec.structs.replace(
                message,
                is_final=True,
                checksum=frame_checksum([inline]),
                execution_timeout_ms=None,
            )
            assert client.call(initial, frames=(inline,)).status is OK
            observed.clear()
            message = AdvanceAndPropose(
                batch_id=0,
                keys=(sequence,),
                accepted_counts_slot=101,
                recovery_tokens_slot=102,
                feature_slot=100,
                execution_timeout_ms=message.execution_timeout_ms,
            )
            frames = (feature, ints_to_frame(101, [0]), ints_to_frame(102, [4]))
        if import_outcome == "failure":
            with pytest.raises(ConnectionClosed):
                client.call(message, frames=frames)
            assert observed == []
        elif import_outcome == "expired":
            reply = client.call(message, frames=frames)
            assert reply.status is SpeculatorStatusCode.TIMEOUT
            assert observed == []
            assert server.registry.get(sequence) is None
            assert client.call(Ping()).healthy
        else:
            reply = client.call(message, frames=frames)
            assert reply.status is OK
            assert len(observed) == 1 and isinstance(observed[0], DeviceFeature)
            assert observed[0].tensor is received
        assert receiver.receive.call_count == 1
        assert receiver.receive.call_args.args == (descriptor, "float32", (1, 2))
        assert len(copy_budgets) == 1 and 0 < copy_budgets[0] <= 1
        if import_outcome == "ok" and operation == "proposal":
            assert len(execution_deadlines) == 1 and execution_deadlines[0] is not None
            assert copy_deadlines[0] <= execution_deadlines[0]
        if import_outcome == "ok" and operation == "prefill":
            feature = msgspec.structs.replace(
                feature, cuda_ipc=msgspec.structs.replace(descriptor, generation=2)
            )
            final = msgspec.structs.replace(
                message, offset=1, is_final=True, checksum=frame_checksum([feature])
            )
            assert client.call(final, frames=(feature,)).status is OK
            assert len(observed) == 2 and isinstance(observed[1], DeviceFeature)
            assert observed[1].tensor is received
            assert copy_deadlines[1] <= copy_deadlines[0]
    finally:
        client.close()
        server.stop()
    receiver.close.assert_called_once_with()


def test_duplicate_wire_slots_close_connection(client):
    client.hello()
    frame = ints_to_frame(5, [1])
    client.conn.send_data(frame)
    client.conn.send_data(frame)
    _, reply = client.recv()
    assert isinstance(reply, ErrorReply)
    assert "duplicate data slot" in reply.detail
    with pytest.raises(ConnectionClosed):
        client.recv()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"target_fingerprint": "other"}, "target fingerprint"),
        ({"tokenizer_fingerprint": "other"}, "tokenizer fingerprint"),
        ({"method": "ngram"}, "method 'ngram'"),
        ({"supported_transports": ("cuda_ipc",)}, "no common transport"),
        ({"max_num_seqs": DEFAULT_LIMITS.max_batch_size + 1}, "max_num_seqs"),
        (
            {"max_num_batched_tokens": DEFAULT_LIMITS.max_feature_tokens + 1},
            "max_num_batched_tokens",
        ),
        ({"max_model_len": DEFAULT_LIMITS.max_model_len + 1}, "max_model_len"),
    ],
)
def test_hello_rejections_close_the_connection(client, overrides, reason):
    reply = client.hello(**overrides)
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.UNSUPPORTED_FEATURE
    assert reason in reply.detail
    with pytest.raises(ConnectionClosed):
        client.recv(timeout=5.0)


@pytest.mark.parametrize("per_sequence", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_hello_rejects_inline_features_that_exceed_frame_bytes(
    monkeypatch, per_sequence, declared
):
    monkeypatch.setattr(transport_module, "MAX_FRAME_BYTES", 1024)
    schema = TargetFeatureSchema(
        1,
        (
            FeatureSlot(
                "aux_hidden_states", "bfloat16", (3, 4), per_sequence=per_sequence
            ),
        ),
    )
    adapter = FakeDraftAdapter(feature_schema=schema)
    max_sequences, max_tokens = (64, 1) if per_sequence else (1, 64)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(max_sequences, max_tokens, 64, 64),
    )
    server.start("tcp://127.0.0.1:0")
    try:
        client = RawClient(server.endpoint)
        try:
            declarations = (
                {"max_num_seqs": max_sequences, "max_num_batched_tokens": max_tokens}
                if declared
                else {}
            )
            reply = client.hello(**declarations)
            assert isinstance(reply, ErrorReply)
            assert reply.status is SpeculatorStatusCode.UNSUPPORTED_FEATURE
            assert "inline feature slot 0" in reply.detail
            assert "1024" in reply.detail
            with pytest.raises(ConnectionClosed):
                client.recv(timeout=5.0)
        finally:
            client.close()
        # Server capacity can be larger than this verifier's declared budget.
        client = RawClient(server.endpoint)
        try:
            assert isinstance(
                client.hello(max_num_seqs=1, max_num_batched_tokens=1), HelloAck
            )
        finally:
            client.close()
    finally:
        server.stop()


def test_first_message_must_be_hello(client):
    reply = client.call(Ping(nonce=1))
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert "expected hello" in reply.detail


def test_hello_rejects_missing_or_mismatched_draft_fingerprint():
    # The check must be fail-closed: when the server declares a draft
    # checkpoint fingerprint, an empty client value is a mismatch too.
    adapter = FakeDraftAdapter(vocab_size=VOCAB, draft_checkpoint_fingerprint="d1")
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    try:
        for offered in ("", "other"):
            client = RawClient(server.endpoint)
            reply = client.hello(draft_checkpoint_fingerprint=offered)
            assert isinstance(reply, ErrorReply)
            assert "draft checkpoint fingerprint" in reply.detail
            client.close()
        client = RawClient(server.endpoint)
        assert isinstance(client.hello(draft_checkpoint_fingerprint="d1"), HelloAck)
        client.close()
    finally:
        adapter.release()
        server.stop()


def test_each_session_gets_a_fresh_epoch(server):
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    try:
        epochs = {first.hello().session_epoch, second.hello().session_epoch}
        assert len(epochs) == 2
    finally:
        first.close()
        second.close()


# ----------------------------------------------------------------------
# Sequence commands
# ----------------------------------------------------------------------


def test_open_is_idempotent_and_generation_aware(client, adapter, server):
    client.hello()
    assert client.open(key(1)).status is OK
    assert client.open(key(1)).status is OK
    assert adapter.prefix(key(1)) == []

    assert client.open(key(1, generation=1)).status is OK
    assert adapter.prefix(key(1)) is None
    assert adapter.prefix(key(1, generation=1)) == []
    assert len(server.registry) == 1

    stale = client.open(key(1))
    assert stale.status is SpeculatorStatusCode.STALE_GENERATION


def test_open_rejects_foreign_verifier_key(client):
    client.hello()
    ack = client.open(key(1, verifier="someone-else"))
    assert ack.status is SpeculatorStatusCode.PROTOCOL_ERROR


def test_prefill_idempotency_rules(client, adapter):
    client.hello()
    k = key(2)
    client.open(k)
    assert client.prefill(k, [1, 2, 3], offset=0, is_final=False).status is OK
    assert adapter.prefix(k) == [1, 2, 3]

    duplicate = client.prefill(k, [1, 2, 3], offset=0, is_final=False)
    assert duplicate.status is OK
    assert adapter.prefix(k) == [1, 2, 3]

    changed_finality = client.prefill(k, [1, 2, 3], offset=0, is_final=True)
    assert changed_finality.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "finality" in changed_finality.detail

    mismatch = client.prefill(k, [9, 9, 9], offset=0, is_final=False)
    assert mismatch.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "different content" in mismatch.detail

    gap = client.prefill(k, [4], offset=5, is_final=False)
    assert gap.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "expected offset 3" in gap.detail

    bad_header = client.prefill(k, [4], offset=3, checksum_override=1)
    assert bad_header.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "checksum" in bad_header.detail

    assert client.prefill(k, [4], offset=3, is_final=True).status is OK
    assert adapter.prefix(k) == [1, 2, 3, 4]
    late = client.prefill(k, [5], offset=4, is_final=True)
    assert late.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "after final" in late.detail


def test_prefill_unknown_sequence_or_missing_frame(client):
    client.hello()
    unknown = client.prefill(key(3), [1], offset=0)
    assert unknown.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "unknown sequence" in unknown.detail

    client.open(key(3))
    missing = client.call(
        PrefillChunk(
            key=key(3),
            offset=0,
            num_tokens=1,
            is_final=True,
            feature_slot=999,
            checksum=0,
        )
    )
    assert missing.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert "missing" in missing.detail


def test_close_is_idempotent_and_keeps_newer_generation(client, adapter, server):
    client.hello()
    client.ready(key(4))
    for _ in range(2):
        ack = client.call(CloseSequence(keys=(key(4),)))
        assert ack.status is OK
    assert adapter.prefix(key(4)) is None
    assert len(server.registry) == 0

    client.ready(key(4, generation=1))
    assert client.call(CloseSequence(keys=(key(4),))).status is OK
    assert adapter.prefix(key(4, generation=1)) == [1, 2, 3]
    assert len(server.registry) == 1


def test_close_acks_every_key(client):
    client.hello()
    client.ready(key(5))
    client.ready(key(6))
    request_id = client.send(CloseSequence(keys=(key(5), key(6), key(7))))
    acked = set()
    for _ in range(3):
        envelope, ack = client.recv()
        assert envelope.request_id == request_id
        assert isinstance(ack, SequenceAck) and ack.status is OK
        acked.add(ack.key)
    assert acked == {key(5), key(6), key(7)}


def test_max_sequences_is_global_and_allows_identity_replacement():
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=4,
            max_feature_tokens=16,
            max_sequences=1,
            max_model_len=32,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    first = RawClient(server.endpoint)
    replacement = RawClient(server.endpoint)
    other = RawClient(server.endpoint)
    try:
        first.hello()
        assert first.open(key(8)).status is OK

        replacement.hello()
        assert replacement.open(key(8, generation=1)).status is OK
        assert first.open(key(8)).status is SpeculatorStatusCode.STALE_GENERATION
        assert first.open(key(9)).status is SpeculatorStatusCode.QUEUE_FULL

        other.hello(verifier_instance_id="other")
        assert (
            other.open(key(8, verifier="other")).status
            is SpeculatorStatusCode.QUEUE_FULL
        )
        assert len(server.registry) == 1
    finally:
        first.close()
        replacement.close()
        other.close()
        adapter.release()
        server.stop()


# ----------------------------------------------------------------------
# Proposal rounds
# ----------------------------------------------------------------------


def test_proposal_follows_fake_formula(client, adapter):
    client.hello()
    client.ready(key(10), tokens=[1, 2])
    client.ready(key(11), tokens=[3])
    response = client.propose(0, [key(10), key(11)], recovery=[5, 47], accepted=[0, 2])
    assert isinstance(response, ProposalResponse)
    assert response.status is OK and response.sequence_number == 0
    tokens, valid, statuses = client.result(response)
    assert tokens == [expected_row(5), expected_row(47)]
    assert valid == [NUM_SPEC_TOKENS, NUM_SPEC_TOKENS]
    assert statuses == [OK, OK]
    assert adapter.prefix(key(10)) == [1, 2, 5]
    assert adapter.prefix(key(11)) == [3, 47]


def test_proposal_marks_unready_and_stale_rows(client):
    client.hello()
    client.ready(key(20))
    client.open(key(21))
    client.ready(key(22, generation=1))
    response = client.propose(
        0, [key(20), key(21), key(22, generation=0)], recovery=[7, 8, 9]
    )
    assert response.status is OK
    tokens, valid, statuses = client.result(response)
    assert statuses == [
        OK,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.STALE_GENERATION,
    ]
    assert valid == [NUM_SPEC_TOKENS, 0, 0]
    assert tokens[0] == expected_row(7)


def test_batch_ids_must_increase_and_sequence_numbers_do(client):
    client.hello()
    client.ready(key(30))
    first = client.propose(5, [key(30)], recovery=[1])
    repeat = client.propose(5, [key(30)], recovery=[1])
    older = client.propose(4, [key(30)], recovery=[1])
    newer = client.propose(6, [key(30)], recovery=[1])
    assert first.status is OK
    assert repeat.status is SpeculatorStatusCode.ROUND_MISMATCH
    assert repeat.result_slot is None
    assert older.status is SpeculatorStatusCode.ROUND_MISMATCH
    assert newer.status is OK
    assert [r.sequence_number for r in (first, repeat, older, newer)] == [0, 1, 2, 3]


def test_proposal_without_input_frames_is_protocol_error(client):
    client.hello()
    client.ready(key(40))
    response = client.call(
        AdvanceAndPropose(
            batch_id=0,
            keys=(key(40),),
            accepted_counts_slot=900,
            recovery_tokens_slot=901,
            feature_slot=902,
        )
    )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR


@pytest.mark.parametrize(
    ("accepted", "recovery"),
    [
        (
            ints_to_frame(0, [0], shape=(1, 1)),
            ints_to_frame(0, [1]),
        ),
        (
            ints_to_frame(0, [0]),
            ints_to_frame(0, [1], shape=(1, 1)),
        ),
        (
            ints_to_frame(0, [0]),
            DataFrame(0, "float32", (1,), b"\x00" * 4),
        ),
        (
            ints_to_frame(0, [0]),
            DataFrame(0, "int64", (1,), b""),
        ),
    ],
    ids=["accepted-shape", "recovery-shape", "recovery-dtype", "recovery-bytes"],
)
def test_proposal_rejects_malformed_integer_frames(client, adapter, accepted, recovery):
    client.hello()
    k = key(42)
    client.ready(k)
    response = client.propose_with_frames(
        0, [k], accepted, recovery, ints_to_frame(0, [1])
    )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert adapter.proposed_batches == []


@pytest.mark.parametrize("operation", ["prefill", "propose"])
@pytest.mark.parametrize("payload", [b"", b"\x00" * 8], ids=["short", "long"])
def test_feature_payload_rejected_before_adapter(
    client, adapter, server, monkeypatch, operation, payload
):
    client.hello()
    k = key(44)
    if operation == "prefill":
        assert client.open(k).status is OK
        method = "prefill"
        expected_prefix = []
    else:
        client.ready(k)
        method = "advance_and_propose"
        expected_prefix = [1, 2, 3]
    adapter_call = Mock(wraps=getattr(adapter, method))
    monkeypatch.setattr(adapter, method, adapter_call)
    feature = DataFrame(client.reserve(1), "int32", (1,), payload)
    if operation == "prefill":
        response = client.call(
            PrefillChunk(
                key=k,
                offset=0,
                num_tokens=1,
                is_final=True,
                feature_slot=feature.slot,
                checksum=checksum(feature),
            ),
            frames=[feature],
        )
    else:
        response = client.propose_with_frames(
            0, [k], ints_to_frame(0, [0]), ints_to_frame(0, [4]), feature
        )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR
    adapter_call.assert_not_called()
    assert adapter.prefix(k) == expected_prefix
    assert server.registry.get(k).prefix_length == len(expected_prefix)


@pytest.mark.parametrize("mode", ["short_payload", "out_of_range"])
def test_malformed_adapter_result_discards_sequence_state(mode):
    adapter = MalformedResultAdapter(mode)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    k = key(43)
    try:
        client.hello()
        client.ready(k)
        response = client.propose(0, [k], recovery=[2])
        assert response.status is SpeculatorStatusCode.INTERNAL_ERROR
        assert response.result_slot is None
        assert server.registry.get(k) is None
        assert adapter.prefix(k) is None
    finally:
        client.close()
        adapter.release()
        server.stop()


def test_rejected_batch_frames_are_not_read_by_a_later_round(client):
    # A rejected batch must not leave its frames behind to be misread by a
    # later round that reuses the same slot numbers.
    client.hello()
    client.ready(key(41))
    client.propose(3, [key(41)], recovery=[1])
    client.next_slot -= 3
    rejected = client.propose(3, [key(41)], recovery=[30])
    assert rejected.status is SpeculatorStatusCode.ROUND_MISMATCH
    client.next_slot -= 3
    fresh = client.propose(4, [key(41)], recovery=[20])
    assert fresh.status is OK
    tokens, _, _ = client.result(fresh)
    assert tokens == [expected_row(20)]


@pytest.mark.parametrize(
    ("query_lens", "token_ids", "lens", "detail"),
    [
        ([1, 1], [1, 2], False, "query_lens frame is missing"),
        ([1], [1], True, "query_lens frame is missing or does not match"),
        ([-1, 2], [1], True, "negative query_lens"),
        ([1, 2], [1, 2], True, "expected 3 rows"),
    ],
)
def test_proposal_rejects_inconsistent_query_lens(
    client, adapter, query_lens, token_ids, lens, detail
):
    client.hello()
    client.ready(key(81))
    client.ready(key(82))
    response = client.propose_ragged(
        0, [key(81), key(82)], query_lens, token_ids, lens=lens
    )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert adapter.proposed_batches == []


def test_proposal_routes_ragged_token_features(client, adapter):
    client.hello()
    client.ready(key(83))
    # Never prefilled: the registry rejects the row but its segment still
    # occupies rows in the frame.
    assert client.open(key(84)).status is OK
    client.ready(key(85))
    response = client.propose_ragged(
        0, [key(83), key(84), key(85)], [2, 2, 0], [10, 11, 20, 21]
    )
    _, _, statuses = client.result(response)
    assert statuses == [OK, SpeculatorStatusCode.SEQUENCE_DESYNC, OK]
    assert adapter.round_tokens == {key(83): [10, 11], key(85): []}


def test_accepted_counts_out_of_range_rejected(client, adapter):
    # Out-of-range accepted counts would corrupt prefix accounting via
    # ``prefix_length += accepted + 1``; the server must not trust them.
    client.hello()
    client.ready(key(80))
    negative = client.propose(0, [key(80)], recovery=[1], accepted=[-1])
    too_many = client.propose(
        1, [key(80)], recovery=[1], accepted=[NUM_SPEC_TOKENS + 1]
    )
    assert negative.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert too_many.status is SpeculatorStatusCode.PROTOCOL_ERROR
    good = client.propose(2, [key(80)], recovery=[5])
    assert good.status is OK
    assert adapter.proposed_batches == [2]
    assert adapter.prefix(key(80)) == [1, 2, 3, 5]


def test_prefill_and_rounds_respect_advertised_limits():
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=4,
            max_feature_tokens=4,
            max_sequences=8,
            max_model_len=6,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        client.hello()
        k = key(90)
        assert client.open(k).status is OK
        too_long = client.prefill(k, [1] * 5, offset=0, is_final=False)
        assert too_long.status is SpeculatorStatusCode.PROTOCOL_ERROR
        assert "max_feature_tokens" in too_long.detail
        assert client.prefill(k, [1, 2, 3, 4], offset=0, is_final=False).status is OK
        overflow = client.prefill(k, [5, 6, 7], offset=4, is_final=True)
        assert overflow.status is SpeculatorStatusCode.OUT_OF_MEMORY
        assert "max_model_len" in overflow.detail
        assert client.prefill(k, [5, 6], offset=4, is_final=True).status is OK

        # The prefix sits at max_model_len: even accepted=0 grows it by the
        # bonus token, so the row must come back OUT_OF_MEMORY and retire.
        response = client.propose(0, [k], recovery=[9])
        assert response.status is OK
        _, valid, statuses = client.result(response)
        assert statuses == [SpeculatorStatusCode.OUT_OF_MEMORY]
        assert valid == [0]
        assert adapter.proposed_batches == []
        assert adapter.prefix(k) is None
        assert server.registry.get(k) is None
    finally:
        client.close()
        adapter.release()
        server.stop()


def test_failed_row_is_refused_by_later_rounds(client, adapter, monkeypatch):
    # The adapter did not advance a failed row. A verifier that keeps
    # sending it (e.g. after abandoning that batch) must get a desync
    # instead of silently extending stale draft state.
    client.hello()
    client.ready(key(70))
    propose = adapter.advance_and_propose

    def fail_rows(batch):
        result = propose(dataclasses.replace(batch, active_rows=()))
        return dataclasses.replace(
            result,
            row_statuses=(SpeculatorStatusCode.INTERNAL_ERROR,) * len(batch.keys),
        )

    monkeypatch.setattr(adapter, "advance_and_propose", fail_rows)
    _, valid, statuses = client.result(client.propose(0, [key(70)], recovery=[1]))
    assert statuses == [SpeculatorStatusCode.INTERNAL_ERROR]
    assert valid == [0]

    monkeypatch.undo()
    _, valid, statuses = client.result(client.propose(1, [key(70)], recovery=[2]))
    assert statuses == [SpeculatorStatusCode.SEQUENCE_DESYNC]
    assert valid == [0]
    assert adapter.prefix(key(70)) is None


def test_cancel_is_accepted_without_reply(client):
    client.hello()
    client.send(CancelBatch(batch_id=123))
    assert client.call(Ping(nonce=4)).nonce == 4


# ----------------------------------------------------------------------
# Session-level errors and teardown
# ----------------------------------------------------------------------


def test_malformed_control_frame_gets_error_reply_and_keeps_session(client):
    client.hello()
    client.conn.send_control(b"\xc1garbage")
    _, reply = client.recv()
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert client.call(Ping(nonce=2)).nonce == 2


def test_session_id_mismatch_rejected(client):
    client.hello()
    client.session_id = "bogus"
    reply = client.call(Ping())
    assert isinstance(reply, ErrorReply)
    assert "session id" in reply.detail


def test_unexpected_message_type_rejected(client):
    client.hello()
    reply = client.call(Pong(nonce=1))
    assert isinstance(reply, ErrorReply)
    assert "unexpected pong" in reply.detail


def test_frames_left_behind_by_a_message_are_dropped(client):
    client.hello()
    k = key(53)
    assert client.open(k).status is OK
    slot = client.reserve(1)
    stray = ints_to_frame(slot, [1, 2])
    assert isinstance(client.call(Ping(nonce=1), frames=[stray]), Pong)
    chunk = PrefillChunk(
        key=k,
        offset=0,
        num_tokens=2,
        is_final=True,
        feature_slot=slot,
        checksum=checksum(stray),
    )
    reply = client.call(chunk)
    assert reply.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert "missing" in reply.detail


def test_data_frames_without_a_control_message_are_bounded(client, adapter):
    client.hello()
    limit = _MAX_INPUT_FRAMES + len(adapter.feature_schema().slots)
    for slot in range(limit + 1):
        client.conn.send_data(ints_to_frame(slot, [1]))
    _, reply = client.recv()
    assert isinstance(reply, ErrorReply)
    assert "without a control message" in reply.detail
    with pytest.raises(ConnectionClosed):
        client.recv(timeout=5.0)


def test_ping_answers_while_the_adapter_is_busy(server, adapter, client):
    client.hello()
    client.ready(key(52))
    adapter.hold_batches.add(0)
    held = threading.Thread(target=client.propose, args=(0, [key(52)], [1]))
    held.start()
    prober = RawClient(server.endpoint)
    try:
        prober.hello(verifier_instance_id="verifier-b")
        assert wait_until(lambda: server.queue_depth == 1)
        started = time.monotonic()
        pong = prober.call(Ping(nonce=5))
        assert isinstance(pong, Pong)
        assert pong.queue_depth == 1
        assert time.monotonic() - started < 2.0
    finally:
        adapter.release()
        held.join()
        prober.close()
    assert server.queue_depth == 0


@pytest.mark.parametrize(
    "failure",
    ["queue_full", "adapter", "malformed_result", "draft_budget", "feature_import"],
)
def test_slow_error_reply_does_not_block_another_session_cleanup(
    server, adapter, client, monkeypatch, failure
):
    client.hello()
    failed_key = key(80)
    if failure == "feature_import":
        assert client.open(failed_key).status is OK
    elif failure != "queue_full":
        client.ready(failed_key)
    other = RawClient(server.endpoint)
    other.hello(verifier_instance_id="verifier-b")
    other_key = key(81, verifier="verifier-b")
    other.ready(other_key)
    if failure == "queue_full":
        server.limits = RemoteServerLimits(
            max_batch_size=DEFAULT_LIMITS.max_batch_size,
            max_feature_tokens=DEFAULT_LIMITS.max_feature_tokens,
            max_sequences=1,
            max_model_len=DEFAULT_LIMITS.max_model_len,
        )
    elif failure == "feature_import":
        resolve = server_module._ServerSession._resolve_features

        def fail_import(session, frames, **kwargs):
            if session.session_id == client.session_id:
                raise RuntimeError("injected feature import failure")
            return resolve(session, frames, **kwargs)

        monkeypatch.setattr(
            server_module._ServerSession, "_resolve_features", fail_import
        )
    else:
        propose = adapter.advance_and_propose

        def fail_proposal(batch):
            if failure == "adapter":
                raise RuntimeError("injected adapter failure")
            if failure == "draft_budget":
                return propose(
                    dataclasses.replace(batch, effective_num_speculative_tokens=None)
                )
            result = propose(batch)
            return dataclasses.replace(result, valid_lengths=ints_to_frame(0, []))

        monkeypatch.setattr(adapter, "advance_and_propose", fail_proposal)

    sending = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()
    replies = []
    errors = []
    sendall = socket.socket.sendall
    client_address = client.conn._sock.getsockname()

    def slow_error_reply(sock, data, *args, **kwargs):
        if sock.getpeername() == client_address:
            sending.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("error reply was not released")
        return sendall(sock, data, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "sendall", slow_error_reply)

    def reject():
        try:
            if failure == "queue_full":
                reply = client.open(failed_key)
            elif failure == "feature_import":
                reply = client.prefill(failed_key, [1, 2, 3], offset=0)
            else:
                reply = client.propose(
                    0,
                    [failed_key],
                    recovery=[1],
                    num_speculative_tokens=0 if failure == "draft_budget" else None,
                )
            replies.append(reply)
        except Exception as error:
            errors.append(error)

    def cleanup():
        try:
            reply = other.call(CloseSequence(keys=(other_key,)))
            assert reply.status is OK
            cleaned.set()
        except Exception as error:
            errors.append(error)

    rejected = threading.Thread(target=reject)
    closing = threading.Thread(target=cleanup)
    try:
        rejected.start()
        assert sending.wait(timeout=5.0)
        if failure != "queue_full":
            assert server.registry.get(failed_key) is None
            assert adapter.prefix(failed_key) is None
        closing.start()
        assert cleaned.wait(timeout=2.0)
        assert server.registry.get(other_key) is None
        assert adapter.prefix(other_key) is None
    finally:
        release.set()
        rejected.join(timeout=5.0)
        if closing.ident is not None:
            closing.join(timeout=5.0)
        other.close()
    assert not rejected.is_alive() and not closing.is_alive()
    assert not errors
    expected = (
        SpeculatorStatusCode.QUEUE_FULL
        if failure == "queue_full"
        else SpeculatorStatusCode.INTERNAL_ERROR
    )
    assert len(replies) == 1 and replies[0].status is expected


def test_warmup_runs_before_the_endpoint_accepts():
    seen: list[str | None] = []

    class WarmAdapter(FakeDraftAdapter):
        def warmup(self):
            seen.append(server.endpoint)

    server = RemoteDraftServer(WarmAdapter(vocab_size=VOCAB))
    server.start("tcp://127.0.0.1:0")
    try:
        assert seen == [None]
    finally:
        server.stop()


@pytest.mark.parametrize("fault", ["recoverable", "fatal", "completion_unknown"])
def test_failed_warmup_preserves_error_and_only_fatal_context_blocks_retry(fault):
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        CudaRuntimeError,
    )

    error = {
        "recoverable": ValueError("recoverable warmup failure"),
        "fatal": CudaRuntimeError("opaque startup failure", cuda_error_code=226),
        "completion_unknown": CompletionUnknownError("warmup completion unknown"),
    }[fault]

    class WarmAdapter(FakeDraftAdapter):
        def __init__(self):
            super().__init__(vocab_size=VOCAB)
            self.attempts = 0

        def warmup(self):
            self.attempts += 1
            assert server.endpoint is None
            if self.attempts == 1:
                raise error

    adapter = WarmAdapter()
    server = RemoteDraftServer(adapter)
    with pytest.raises(type(error)) as caught:
        server.start("tcp://127.0.0.1:0")
    assert caught.value is error
    assert server.endpoint is None
    assert server.healthy is (fault == "recoverable")
    if fault != "recoverable":
        with pytest.raises(RuntimeError, match="process restart"):
            server.start("tcp://127.0.0.1:0")
        assert adapter.attempts == 1
    else:
        server.start("tcp://127.0.0.1:0")
        try:
            assert server.endpoint is not None
            assert adapter.attempts == 2
        finally:
            server.stop()


@pytest.mark.parametrize(
    ("endpoint", "warns"), [("tcp://127.0.0.1:0", False), ("tcp://0.0.0.0:0", True)]
)
def test_listener_reachable_from_other_hosts_warns(
    adapter, monkeypatch, endpoint, warns
):
    warning = Mock()
    monkeypatch.setattr(server_module.logger, "warning", warning)
    server = RemoteDraftServer(adapter, allow_remote_connections=warns)
    server.start(endpoint)
    try:
        messages = [c.args[0] for c in warning.call_args_list]
        assert any("no authentication" in m for m in messages) is warns
    finally:
        server.stop()


def test_ping_reports_active_sequences(client):
    client.hello()
    client.ready(key(50))
    client.ready(key(51))
    pong = client.call(Ping(nonce=77))
    assert isinstance(pong, Pong)
    assert pong.nonce == 77
    assert pong.active_sequences == 2


@pytest.mark.parametrize("stage", ["open_sequence", "prefill", "advance_and_propose"])
@pytest.mark.parametrize("fault", ["oom", "fatal", "completion_unknown"])
def test_adapter_fault_health_fences_context_fatal_errors(stage, fault):
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        CudaRuntimeError,
    )

    class FaultAdapter(FakeDraftAdapter):
        def __init__(self):
            super().__init__(vocab_size=VOCAB)
            self.armed = False
            self.calls = 0
            self.close_calls = []

        def fail(self, operation):
            if self.armed and operation == stage:
                self.calls += 1
                self.armed = False
                raise {
                    "oom": torch.OutOfMemoryError("CUDA out of memory"),
                    "fatal": CudaRuntimeError("executor failed", cuda_error_code=700),
                    "completion_unknown": CompletionUnknownError(
                        "upload completion unknown"
                    ),
                }[fault]

        def open_sequence(self, key):
            self.fail("open_sequence")
            return super().open_sequence(key)

        def prefill(self, key, inputs):
            self.fail("prefill")
            return super().prefill(key, inputs)

        def close_sequences(self, keys):
            self.close_calls.append(keys)
            return super().close_sequences(keys)

        def advance_and_propose(self, batch):
            self.fail("advance_and_propose")
            return super().advance_and_propose(batch)

    adapter = FaultAdapter()
    server = RemoteDraftServer(
        adapter, server_id="srv", target_fingerprint="tf", tokenizer_fingerprint="kf"
    )
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        client.hello()
        failed_key, sibling = key(90), key(91)
        client.ready(sibling)
        if stage != "open_sequence":
            assert client.open(failed_key).status is OK
        if stage == "advance_and_propose":
            assert client.prefill(failed_key, [1], offset=0).status is OK
        adapter.armed = True
        if stage == "open_sequence":
            response = client.open(failed_key)
        elif stage == "prefill":
            response = client.prefill(failed_key, [1], offset=0)
        else:
            response = client.propose(0, [failed_key], [2])
        assert response.status is SpeculatorStatusCode.INTERNAL_ERROR
        fatal = fault != "oom"
        assert client.call(Ping(nonce=23)).healthy is not fatal
        assert server.wait(0) is fatal
        if fatal:
            assert client.open(key(92)).status is SpeculatorStatusCode.INTERNAL_ERROR
            assert (
                client.prefill(sibling, [4], offset=3).status
                is SpeculatorStatusCode.INTERNAL_ERROR
            )
            assert (
                client.propose(1, [sibling], [4]).status
                is SpeculatorStatusCode.INTERNAL_ERROR
            )
        else:
            client.ready(key(92))
            assert client.propose(1, [key(92)], [4]).status is OK
        assert adapter.calls == 1
        if fatal:
            assert adapter.close_calls == []
            assert server.registry.get(sibling) is not None
    finally:
        client.close()
        server.stop()
    if fatal:
        assert adapter.close_calls == []
        assert server.registry.get(sibling) is not None


@pytest.mark.parametrize(
    "detail",
    [
        "device-side assert triggered",
        "hardware stack error",
        "tensor memory not completely freed",
    ],
)
def test_fatal_adapter_error_wakes_main_and_joins_workers(monkeypatch, detail):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter)
    monkeypatch.setattr(server_module, "RemoteDraftServer", lambda *a, **kw: server)
    started = threading.Event()
    start = server.start

    def start_and_notify(endpoint):
        resolved = start(endpoint)
        started.set()
        return resolved

    monkeypatch.setattr(server, "start", start_and_notify)
    errors = []

    def run_main():
        try:
            server_module.main(["--endpoint", "tcp://127.0.0.1:0"])
        except Exception as error:
            errors.append(error)

    owner = threading.Thread(target=run_main)
    owner.start()
    client = None
    try:
        assert started.wait(5.0)
        client = RawClient(server.endpoint)
        client.hello()
        client.ready(key(93))
        close = Mock(wraps=adapter.close_sequences)
        monkeypatch.setattr(adapter, "close_sequences", close)
        monkeypatch.setattr(
            adapter,
            "open_sequence",
            Mock(side_effect=RuntimeError(f"CUDA error: {detail}")),
        )
        with contextlib.suppress(ConnectionClosed):
            client.open(key(94))
        owner.join(timeout=5.0)
        assert not owner.is_alive()
        assert len(errors) == 1 and "restart required" in str(errors[0])
        assert not server.healthy
        assert not server._workers and not server._connections
        assert server.registry.get(key(93)) is not None
        assert server.registry.get(key(94)) is not None
        close.assert_not_called()
    finally:
        if client is not None:
            client.close()
        server.stop()
        owner.join(timeout=5.0)


def test_disconnect_releases_sequences(server, adapter, client):
    client.hello()
    client.ready(key(60))
    client.ready(key(61))
    assert len(server.registry) == 2
    client.close()
    assert wait_until(lambda: len(server.registry) == 0)
    assert adapter.prefix(key(60)) is None


def test_late_disconnect_only_releases_entries_the_session_still_owns(server, adapter):
    # A reconnected verifier re-opens its sequences under new generations;
    # when the old connection finally goes away, its cleanup must not take
    # the replacement entries with it.
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    try:
        first.hello()
        first.ready(key(70))
        first.ready(key(71))
        second.hello()
        second.ready(key(70, generation=1), tokens=(7, 8))
        assert len(server.registry) == 2

        first.close()
        assert wait_until(lambda: len(server.registry) == 1)
        response = second.propose(0, [key(70, generation=1)], recovery=[9])
        assert response.status is OK
        _, _, statuses = second.result(response)
        assert statuses == [OK]
        assert adapter.prefix(key(70, generation=1)) == [7, 8, 9]
        assert adapter.prefix(key(71)) is None
    finally:
        first.close()
        second.close()


def test_live_sessions_cannot_mutate_each_others_sequences(server, adapter):
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    k = key(72)
    try:
        first.hello()
        first.ready(k)
        second.hello()

        assert second.open(k).status is SpeculatorStatusCode.SEQUENCE_DESYNC
        assert (
            second.prefill(k, [9], offset=3).status
            is SpeculatorStatusCode.SEQUENCE_DESYNC
        )
        response = second.propose(0, [k], recovery=[9])
        _, valid, statuses = second.result(response)
        assert valid == [0]
        assert statuses == [SpeculatorStatusCode.SEQUENCE_DESYNC]
        assert second.call(CloseSequence(keys=(k,))).status is OK

        response = first.propose(0, [k], recovery=[4])
        _, _, statuses = first.result(response)
        assert statuses == [OK]
        assert adapter.prefix(k) == [1, 2, 3, 4]
    finally:
        first.close()
        second.close()


@pytest.mark.timeout(10)
def test_server_closes_connection_when_verifier_stops_reading(server, client):
    client.hello(num_speculative_tokens=1 << 18)
    k = key(1)
    client.ready(k)
    client.conn._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    conn = next(iter(server._connections))
    conn._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    server.send_timeout_s = 0.05
    base = client.reserve(3)
    client.send(
        AdvanceAndPropose(
            batch_id=0,
            keys=(k,),
            accepted_counts_slot=base,
            recovery_tokens_slot=base + 1,
            feature_slot=base + 2,
        ),
        frames=[
            ints_to_frame(base, [0]),
            ints_to_frame(base + 1, [4]),
            ints_to_frame(base + 2, [4]),
        ],
    )
    assert wait_until(lambda: conn.closed)


def test_stop_disconnects_clients(server, client):
    client.hello()
    server.stop()
    with pytest.raises(ConnectionClosed):
        client.recv(timeout=5.0)


@pytest.mark.parametrize("failed_owner", ["listener", 0, 1])
def test_stop_attempts_all_owners_after_socket_close_error(
    adapter, monkeypatch, failed_owner
):
    accepted = []
    cleanup_error = OSError(errno.EIO, "injected cleanup failure")

    class FailingSocket(socket.socket):
        fail_close = False

        def accept(self):
            conn, address = super().accept()
            accepted.append(conn)
            return conn, address

        def close(self):
            super().close()
            if self.fail_close:
                self.fail_close = False
                raise cleanup_error

    monkeypatch.setattr(socket, "socket", FailingSocket)
    server = RemoteDraftServer(adapter)
    endpoint = server.start("tcp://127.0.0.1:0")
    clients = [RawClient(endpoint) for _ in range(3)]
    try:
        for client in clients:
            assert isinstance(client.hello(), HelloAck)
        acceptor, workers = server._acceptor, tuple(server._workers)
        assert len(accepted) == len(workers) == 3
        failed_socket = (
            server._listener._sock
            if failed_owner == "listener"
            else accepted[failed_owner]
        )
        failed_socket.fail_close = True
        with pytest.raises(server_module.ServerShutdownTimeout) as failure:
            server.stop()
        assert cleanup_error in failure.value.__cause__.exceptions
        assert server._shutdown_failed
        assert server.wait(0)
        assert not acceptor.is_alive()
        assert all(not worker.is_alive() for worker in workers)
        assert all(sock.fileno() == -1 for sock in accepted)
        server.stop()
        with pytest.raises(server_module.ServerShutdownTimeout, match="restart"):
            server.start(endpoint)
    finally:
        for client in clients:
            client.close()
        server.stop()


@pytest.mark.parametrize("error_number", [errno.EMFILE, errno.ENFILE, errno.EBADF])
@pytest.mark.parametrize("recovers", [False, True])
def test_accept_failure_recovers_or_notifies_server_owner(
    adapter, monkeypatch, error_number, recovers
):
    failures = 0

    class FailingSocket(socket.socket):
        def accept(self):
            nonlocal failures
            if not recovers or failures < 2:
                failures += 1
                raise OSError(error_number, "injected accept failure")
            return super().accept()

    monkeypatch.setattr(socket, "socket", FailingSocket)
    server = RemoteDraftServer(adapter)
    endpoint = server.start("tcp://127.0.0.1:0")
    client = None
    try:
        if recovers and error_number != errno.EBADF:
            client = RawClient(endpoint)
            assert isinstance(client.hello(), HelloAck)
            assert failures == 2
            assert not server._shutdown_failed
            assert not server.wait(0)
        else:
            assert server.wait(2)
            assert server._shutdown_failed
            assert server._lifecycle_error.errno == error_number
    finally:
        if client is not None:
            client.close()
        server.stop()


@pytest.mark.parametrize("failure", ["before", "after", "uncertain"])
def test_acceptor_start_failure_rolls_back_unix_listener(
    adapter, monkeypatch, tmp_path, failure
):
    class FailingThread(threading.Thread):
        def start(self):
            if self.name == "remote-draft-acceptor":
                if failure == "after":
                    super().start()
                if failure == "before":
                    raise RuntimeError("injected thread start failure")
                raise MemoryError("injected thread start failure")
            return super().start()

    server = RemoteDraftServer(adapter, shutdown_timeout_s=0.2)
    path = tmp_path / "draft.sock"
    endpoint = f"unix://{path}"
    with monkeypatch.context() as patch:
        patch.setattr(threading, "Thread", FailingThread)
        with pytest.raises((RuntimeError, MemoryError), match="injected"):
            server.start(endpoint)
    assert not path.exists()
    if failure == "uncertain":
        assert server._shutdown_failed
        with pytest.raises(server_module.ServerShutdownTimeout):
            server.stop()
        with pytest.raises(server_module.ServerShutdownTimeout):
            server.start(endpoint)
    else:
        assert server.endpoint is None
        server.stop()
        server.start(endpoint)
        server.stop()


def test_normal_stop_racing_accept_closure_does_not_fence_server(adapter, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class ClosingSocket(socket.socket):
        def accept(self):
            entered.set()
            assert release.wait(2)
            raise ConnectionClosed("listener closed before accepting")

    monkeypatch.setattr(socket, "socket", ClosingSocket)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    errors = []

    def stop():
        try:
            server.stop()
        except BaseException as error:
            errors.append(error)

    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(2)
        stopper.start()
        assert server.wait(2)
        release.set()
        stopper.join(2)
        assert not stopper.is_alive()
        assert not errors
        assert not server._shutdown_failed
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(2)
        server.stop()


def test_stop_closes_connection_accepted_before_registration(adapter, monkeypatch):
    accepted = threading.Event()
    register = threading.Event()
    joining_acceptor = threading.Event()
    original_accept = Listener.accept

    def accept_then_pause(listener, timeout=None):
        conn = original_accept(listener, timeout)
        accepted.set()
        assert register.wait(5.0)
        return conn

    monkeypatch.setattr(Listener, "accept", accept_then_pause)
    server = RemoteDraftServer(adapter)
    endpoint = server.start("tcp://127.0.0.1:0")
    acceptor = server._acceptor
    assert acceptor is not None
    original_join = acceptor.join

    def observe_acceptor_join(timeout=None):
        joining_acceptor.set()
        original_join(timeout)

    monkeypatch.setattr(acceptor, "join", observe_acceptor_join)
    conn = connect(endpoint, timeout=5.0)
    stopper = threading.Thread(target=server.stop, daemon=True)
    try:
        assert accepted.wait(5.0)
        stopper.start()
        assert joining_acceptor.wait(5.0)
        register.set()
        stopper.join(timeout=5.0)
        assert not stopper.is_alive()
        assert not acceptor.is_alive()
        assert not server._workers
        assert not server._connections
        with pytest.raises(ConnectionClosed):
            conn.recv(timeout=5.0)
    finally:
        register.set()
        conn.close()
        if stopper.ident is not None:
            stopper.join(timeout=5.0)
        server.stop()


# ----------------------------------------------------------------------
# Registry unit rules
# ----------------------------------------------------------------------


def test_registry_open_generations():
    registry = SequenceRegistry()
    assert registry.open(key(1), owner="s1").created
    assert not registry.open(key(1), owner="s1").created
    outcome = registry.open(key(1, generation=2), owner="s1")
    assert outcome.created and outcome.replaced == key(1)
    assert registry.open(key(1, generation=1), owner="s1").status is (
        SpeculatorStatusCode.STALE_GENERATION
    )
    assert registry.get(key(1)) is None
    assert registry.get(key(1, generation=2)) is not None


def test_registry_prefill_and_rounds():
    registry = SequenceRegistry()
    registry.open(key(1), owner="s1")
    assert (
        registry.begin_round(key(1), 0, owner="s1")[0]
        is SpeculatorStatusCode.SEQUENCE_DESYNC
    )
    assert registry.prefill(
        key(1),
        owner="s1",
        offset=0,
        num_tokens=3,
        checksum=9,
        is_final=False,
    ) == (OK, "", True)
    assert (
        registry.prefill(
            key(1),
            owner="s1",
            offset=0,
            num_tokens=3,
            checksum=9,
            is_final=False,
        )[2]
        is False
    )
    assert (
        registry.prefill(
            key(1),
            owner="s1",
            offset=3,
            num_tokens=1,
            checksum=1,
            is_final=True,
        )[0]
        is OK
    )
    assert registry.begin_round(key(1), 0, owner="s1") == (OK, "")
    assert (
        registry.begin_round(key(1), 0, owner="s1")[0]
        is SpeculatorStatusCode.ROUND_MISMATCH
    )
    registry.advance(key(1), 2, owner="s1")
    entry = registry.get(key(1))
    assert entry is not None
    assert entry.prefix_length == 4 + 3


def test_registry_counts_track_open_sequences():
    registry = SequenceRegistry()
    registry.open(key(1), owner="s1")
    registry.open(key(1, generation=1), owner="s1")
    registry.open(key(2), owner="s1")
    registry.open(key(3, verifier="other"), owner="s2")
    assert registry.count(VERIFIER) == 2
    assert registry.count("other") == 1
    registry.close(key(2), owner="s1")
    registry.close_all("s2")
    assert registry.count(VERIFIER) == 1
    assert registry.count("other") == 0


def test_registry_close_all_releases_only_owned_entries():
    registry = SequenceRegistry()
    registry.open(key(1), owner="s1")
    registry.open(key(2), owner="s1")
    registry.open(key(3, verifier="other"), owner="s2")
    assert set(registry.close_all("s1")) == {key(1), key(2)}
    assert len(registry) == 1
    assert registry.close_all("s1") == ()
    assert registry.close(key(3, verifier="other"), owner="s2") == key(
        3, verifier="other"
    )
    assert registry.close(key(3, verifier="other"), owner="s2") is None


def test_registry_new_generation_transfers_ownership():
    registry = SequenceRegistry()
    registry.open(key(1), owner="old")
    assert registry.open(key(1, generation=1), owner="new").created
    assert registry.close_all("old") == ()
    same_generation = registry.open(key(1, generation=1), owner="newer")
    assert same_generation.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert registry.close_all("newer") == ()
    assert set(registry.close_all("new")) == {key(1, generation=1)}
    assert len(registry) == 0


def test_registry_rejects_other_owner_and_future_close():
    registry = SequenceRegistry()
    current = key(1, generation=1)
    registry.open(current, owner="s1")
    outcome = registry.open(current, owner="s2")
    assert outcome.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert registry.close(key(1, generation=2), owner="s1") is None
    assert registry.close(current, owner="s2") is None
    assert registry.get(current) is not None


def test_handshake_rejects_data_header_without_receiving_body(adapter):
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    conn = connect(server.endpoint, timeout=1)
    try:
        conn._sock.sendall(transport_module._HEADER.pack(FRAME_DATA, 100))
        with pytest.raises(ConnectionClosed):
            conn.recv(timeout=1)
    finally:
        conn.close()
        server.stop()


def test_negotiated_schema_rejects_large_header_before_body(adapter):
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        assert isinstance(
            client.hello(max_num_seqs=1, max_num_batched_tokens=1), HelloAck
        )
        client.conn._sock.sendall(transport_module._HEADER.pack(FRAME_DATA, 1000))
        with pytest.raises(ConnectionClosed):
            client.conn.recv(timeout=1)
    finally:
        client.close()
        server.stop()


def test_pending_receive_budget_counts_bytes_before_frame_count(adapter):
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        assert isinstance(
            client.hello(max_num_seqs=1, max_num_batched_tokens=256), HelloAck
        )
        # Each frame fits the token schema, but two exceed the declared batch
        # bytes even though there are fewer than the five allowed input frames.
        client.conn.send_data(ints_to_frame(1, list(range(256))))
        client.conn._sock.sendall(transport_module._HEADER.pack(FRAME_DATA, 1024))
        with pytest.raises(ConnectionClosed):
            client.conn.recv(timeout=1)
    finally:
        client.close()
        server.stop()


def test_handshake_rejects_legal_frames_exceeding_total_server_budget(adapter):
    server = RemoteDraftServer(adapter, max_pending_bytes=100)
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        reply = client.hello(max_num_seqs=1, max_num_batched_tokens=1)
        assert isinstance(reply, ErrorReply)
        assert "max_pending_bytes=100" in reply.detail
    finally:
        client.close()
        server.stop()


@pytest.mark.parametrize("saturated", ["connections", "handshakes"])
def test_connection_admission_is_bounded_and_recovers(adapter, saturated):
    server = RemoteDraftServer(
        adapter,
        max_connections=1 if saturated == "connections" else 2,
        max_handshakes=1,
    )
    endpoint = server.start("tcp://127.0.0.1:0")
    first = RawClient(endpoint)
    rejected = None
    replacement = None
    try:
        if saturated == "connections":
            assert isinstance(first.hello(), HelloAck)
        assert wait_until(lambda: len(server._connections) == 1)
        rejected = connect(endpoint, timeout=1)
        with pytest.raises(ConnectionClosed):
            rejected.recv(timeout=1)
        assert len(server._connections) == len(server._workers) == 1
        first.close()
        assert wait_until(lambda: not server._connections)
        replacement = RawClient(endpoint)
        assert isinstance(replacement.hello(), HelloAck)
    finally:
        first.close()
        if rejected is not None:
            rejected.close()
        if replacement is not None:
            replacement.close()
        server.stop()


@pytest.mark.parametrize("partial", [b"\x01", b"\x02\x00\x00\x00\x08x"])
def test_started_frame_stall_expires_but_idle_session_survives(adapter, partial):
    server = RemoteDraftServer(adapter, partial_frame_timeout_s=0.03)
    endpoint = server.start("tcp://127.0.0.1:0")
    idle = RawClient(endpoint)
    stalled = RawClient(endpoint)
    try:
        assert isinstance(idle.hello(), HelloAck)
        assert isinstance(stalled.hello(verifier_instance_id="stalled"), HelloAck)
        stalled.conn._sock.sendall(partial)
        with pytest.raises(ConnectionClosed):
            stalled.conn.recv(timeout=1)
        assert isinstance(idle.call(Ping(nonce=9)), Pong)
    finally:
        stalled.close()
        idle.close()
        server.stop()


def test_adapter_waiter_limit_reserves_cleanup_progress(adapter):
    server = RemoteDraftServer(adapter, max_adapter_waiters=1)
    entered = threading.Event()
    release = threading.Event()
    finished = []

    def hold():
        with server.adapter_turn():
            entered.set()
            assert release.wait(5)

    def queued(cleanup):
        with server.adapter_turn(cleanup=cleanup):
            finished.append(cleanup)

    threads = [threading.Thread(target=hold)]
    threads[0].start()
    try:
        assert entered.wait(1)
        threads.append(threading.Thread(target=queued, args=(False,)))
        threads[-1].start()
        assert wait_until(lambda: server.queue_depth == 2)
        with pytest.raises(server_module.AdapterQueueFull), server.adapter_turn():
            pytest.fail("saturated foreground work was admitted")
        threads.append(threading.Thread(target=queued, args=(True,)))
        threads[-1].start()
        assert wait_until(lambda: server.queue_depth == 3)
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
    assert sorted(finished) == [False, True]
    assert server.queue_depth == 0


@pytest.mark.parametrize("operation", ["prefill", "proposal"])
@pytest.mark.parametrize("budget_ms", [0, 20, 5000, None])
def test_execution_budget_expires_before_queued_adapter_work(operation, budget_ms):
    """Expired queued work retires own-KV without a new forward or context update."""

    class CountingAdapter(FakeDraftAdapter):
        def __init__(self):
            super().__init__(vocab_size=VOCAB)
            self.prefill_calls = 0

        def prefill(self, key, inputs):
            self.prefill_calls += 1
            return super().prefill(key, inputs)

    adapter = CountingAdapter()
    server = RemoteDraftServer(
        adapter,
        server_id="srv",
        target_fingerprint="tf",
        tokenizer_fingerprint="kf",
        ipc_copy_timeout_s=0.02,
    )
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    assert isinstance(client.hello(), HelloAck)
    client.ready(key(1))
    waiting = RawClient(server.endpoint)
    sequence = key(2, verifier="waiting")
    try:
        assert isinstance(waiting.hello(verifier_instance_id="waiting"), HelloAck)
        if operation == "prefill":
            assert waiting.open(sequence).status is OK
        else:
            waiting.ready(sequence)
        entry = server.registry.get(sequence)
        assert entry is not None
        original_offset = entry.applied_offset
        prefill_calls = adapter.prefill_calls
        adapter.hold_batches.add(0)
        client.send(
            AdvanceAndPropose(
                batch_id=0,
                keys=(key(1),),
                accepted_counts_slot=100,
                recovery_tokens_slot=101,
                feature_slot=102,
            ),
            frames=(
                ints_to_frame(100, [0]),
                ints_to_frame(101, [4]),
                ints_to_frame(102, [4]),
            ),
        )
        assert wait_until(lambda: server.queue_depth == 1)
        frame = ints_to_frame(102, [4])
        message: PrefillChunk | AdvanceAndPropose
        frames: tuple[DataFrame, ...]
        if operation == "prefill":
            message = PrefillChunk(
                key=sequence,
                offset=0,
                num_tokens=1,
                is_final=True,
                feature_slot=102,
                checksum=checksum(frame),
                execution_timeout_ms=budget_ms,
            )
            frames = (frame,)
        else:
            message = AdvanceAndPropose(
                batch_id=1,
                keys=(sequence,),
                accepted_counts_slot=100,
                recovery_tokens_slot=101,
                feature_slot=102,
                execution_timeout_ms=budget_ms,
            )
            frames = (ints_to_frame(100, [0]), ints_to_frame(101, [4]), frame)
        waiting.send(message, frames=frames)
        assert wait_until(lambda: server.queue_depth == 2)
        # Expire short execution budgets and the IPC-only copy timeout;
        # inline features must not inherit the latter.
        deadline = time.monotonic() + server.ipc_copy_timeout_s
        assert wait_until(lambda: time.monotonic() >= deadline)
        adapter.release()
        assert client.recv()[1].status is OK
        reply = waiting.recv()[1]
        expired = budget_ms in (0, 20)
        assert reply.status is (SpeculatorStatusCode.TIMEOUT if expired else OK)
        if expired:
            assert server.registry.get(sequence) is None
            assert adapter.prefix(sequence) is None
            assert entry.applied_offset == original_offset
            assert entry.last_batch_id == -1
            assert adapter.prefill_calls == prefill_calls
            assert adapter.proposed_batches == [0]
            # Missing context cannot silently resume under the retired key.
            response = waiting.propose(2, (sequence,), [4])
            assert waiting.result(response)[2] == [SpeculatorStatusCode.SEQUENCE_DESYNC]
        else:
            assert server.registry.get(sequence) is entry
            prefix = adapter.prefix(sequence)
            assert prefix is not None and prefix[-1] == 4
        assert waiting.call(Ping(nonce=1)).healthy
        replacement = key(3, verifier="waiting")
        waiting.ready(replacement)
        assert waiting.propose(3, (replacement,), [4]).status is OK
    finally:
        adapter.release()
        waiting.close()
        client.close()
        server.stop()


def test_shutdown_discards_queued_proposal_and_retires_both_sessions():
    entered = threading.Event()
    release = threading.Event()
    proposals = []
    stop_errors = []

    class GatedAdapter(FakeDraftAdapter):
        def advance_and_propose(self, batch):
            proposals.append(batch.keys)
            if len(proposals) == 1:
                entered.set()
                assert release.wait(5)
            return super().advance_and_propose(batch)

    adapter = GatedAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(adapter, shutdown_timeout_s=1)
    endpoint = server.start("tcp://127.0.0.1:0")
    clients = [RawClient(endpoint), RawClient(endpoint)]
    keys = [key(1), key(2, verifier="verifier-b")]

    def stop():
        try:
            server.stop()
        except Exception as exc:
            stop_errors.append(exc)

    stopper = threading.Thread(target=stop)
    try:
        for client, sequence in zip(clients, keys):
            assert isinstance(
                client.hello(verifier_instance_id=sequence.verifier_instance_id),
                HelloAck,
            )
            client.ready(sequence)
        for index, (client, sequence) in enumerate(zip(clients, keys)):
            client.send(
                AdvanceAndPropose(
                    batch_id=0,
                    keys=(sequence,),
                    accepted_counts_slot=100,
                    recovery_tokens_slot=101,
                    feature_slot=102,
                ),
                frames=(
                    ints_to_frame(100, [0]),
                    ints_to_frame(101, [4]),
                    ints_to_frame(102, [4]),
                ),
            )
            if index == 0:
                assert entered.wait(1)
        assert wait_until(lambda: server.queue_depth == 2)
        stopper.start()
        for client in clients:
            with pytest.raises(ConnectionClosed):
                client.recv(timeout=1)
        release.set()
        stopper.join(timeout=2)
        assert not stopper.is_alive()
        assert not stop_errors
        assert proposals == [(keys[0],)]
        assert not server._shutdown_failed
        assert not server._draining
        assert len(server.registry) == 0
        assert all(adapter.prefix(sequence) is None for sequence in keys)
        assert not server._connections
        assert server.queue_depth == 0
    finally:
        release.set()
        for client in clients:
            client.close()
        if stopper.ident is not None:
            stopper.join(timeout=2)
            assert not stopper.is_alive()
        server.stop()


def test_shutdown_grace_retains_live_adapter_and_forbids_restart(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class BlockedAdapter(FakeDraftAdapter):
        def advance_and_propose(self, batch):
            entered.set()
            assert release.wait(5)
            return super().advance_and_propose(batch)

    adapter = BlockedAdapter(vocab_size=VOCAB)
    close = Mock(wraps=adapter.close_sequences)
    monkeypatch.setattr(adapter, "close_sequences", close)
    server = RemoteDraftServer(adapter, shutdown_timeout_s=0.3)
    endpoint = server.start("tcp://127.0.0.1:0")
    client = RawClient(endpoint)
    try:
        assert isinstance(client.hello(), HelloAck)
        client.ready(key(1))
        # Stop while the actual socket worker owns mutable adapter state.
        client.send(
            AdvanceAndPropose(
                batch_id=0,
                keys=(key(1),),
                accepted_counts_slot=100,
                recovery_tokens_slot=101,
                feature_slot=102,
            ),
            frames=(
                ints_to_frame(100, [0]),
                ints_to_frame(101, [4]),
                ints_to_frame(102, [4]),
            ),
        )
        assert entered.wait(1)
        started = time.monotonic()
        with pytest.raises(server_module.ServerShutdownTimeout, match="process"):
            server.stop()
        assert time.monotonic() - started < 1.5
        assert any(worker.is_alive() for worker in server._workers)
        assert server.adapter is adapter
        close.assert_not_called()
        with pytest.raises(server_module.ServerShutdownTimeout):
            server.start("tcp://127.0.0.1:0")
    finally:
        release.set()
        client.close()
        for worker in server._workers:
            worker.join(timeout=2)
            assert not worker.is_alive()
        server.stop()
    close.assert_not_called()
    assert len(server.registry) == 1


def test_external_binding_requires_explicit_opt_in_before_warmup(adapter, monkeypatch):
    warmup = Mock()
    monkeypatch.setattr(adapter, "warmup", warmup)
    server = RemoteDraftServer(adapter)
    with pytest.raises(ValueError, match="allow_remote_connections"):
        server.start("tcp://0.0.0.0:0")
    warmup.assert_not_called()
    assert server.endpoint is None
