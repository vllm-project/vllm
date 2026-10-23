# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request, RequestStatus


def test_hash_publication_reserved_before_reset_cannot_publish_after_reset():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from vllm.utils.hashing import sha256
    from vllm.utils.session_prefix import Reservation
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.session_prefix import SessionHashCache

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    incoming = EngineCoreRequest(
        request_id="inflight",
        prompt_token_ids=list(range(33)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session",
    )
    native = Request.from_engine_core_request(incoming, hasher)
    for close in (False, True):
        cache = SessionHashCache(1 << 20, 4)
        entered, release = threading.Event(), threading.Event()
        previous = threading.getprofile()

        def observe(frame, event, arg, cache=cache, entered=entered, release=release):
            if (
                frame.f_code is Reservation.commit.__code__
                and event == "call"
                and frame.f_locals["self"].store is cache.store
            ):
                entered.set()
                assert release.wait(10)

        threading.setprofile(observe)
        try:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(cache.publish, incoming, native, cache.store.epoch)
                try:
                    assert entered.wait(10), "real commit was not reached"
                    assert cache.store.bytes_used > 0
                    cache.store.clear(close=close)
                finally:
                    release.set()
                future.result(timeout=10)
        finally:
            release.set()
            threading.setprofile(previous)
        assert cache.store.bytes_used == 0
        assert cache.prefix(incoming) is None


def test_session_hash_reuse_preserves_native_chain_and_decode():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.session_prefix import SessionHashCache

    init_none_hash(sha256)
    for block_size in (2, 4, 16):
        hasher = get_request_block_hasher(block_size, sha256)
        cache = SessionHashCache(1 << 20, block_size)
        for turn, ids in enumerate(
            (list(range(33)), list(range(35)), [99] + list(range(34)))
        ):
            incoming = EngineCoreRequest(
                request_id=str(turn),
                prompt_token_ids=ids,
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=3),
                pooling_params=None,
                arrival_time=0.0,
                lora_request=None,
                cache_salt="salt",
                data_parallel_rank=None,
                session_id="session",
            )
            expected = Request.from_engine_core_request(incoming, hasher)
            initial = cache.prefix(incoming)
            actual = Request.from_engine_core_request(incoming, hasher, initial)
            assert actual.block_hashes == expected.block_hashes
            if turn == 1:
                assert initial is not None and initial[1]
            cache.publish(incoming, actual, cache.store.epoch)
            for token in range(100, 118):
                actual.append_output_token_ids(token)
                expected.append_output_token_ids(token)
                assert actual.block_hashes == expected.block_hashes
        incoming.cache_salt = "different"
        assert cache.prefix(incoming) is None
        assert cache.stats["reused_blocks"] > 0
        cache.store.clear()
        assert cache.store.bytes_used == 0


def test_initial_hashes_reject_invalid_block_count():
    import pytest

    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    init_none_hash(sha256)
    with pytest.raises(ValueError, match="initial prompt block hashes"):
        Request(
            "invalid",
            [1],
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(16, sha256),
            initial_block_hashes=(16, [b"invalid"]),
        )


def test_session_hash_reset_close_and_eviction_preserve_native_results():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.session_prefix import SessionHashCache

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    incoming = EngineCoreRequest(
        request_id="lifecycle",
        prompt_token_ids=list(range(33)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt="salt",
        data_parallel_rank=None,
        session_id="session",
    )
    native = Request.from_engine_core_request(incoming, hasher)
    for close in (False, True):
        cache = SessionHashCache(1 << 20, 4)
        cache.publish(incoming, native, cache.store.epoch)
        initial = cache.prefix(incoming)
        assert initial is not None
        epoch = cache.store.epoch
        cache.store.clear(close=close)
        cache.publish(incoming, native, epoch)
        assert cache.store.bytes_used == 0  # Stale work cannot repopulate.
        actual = Request.from_engine_core_request(incoming, hasher, initial)
        assert actual.block_hashes == native.block_hashes  # Existing reader is valid.
        assert cache.prefix(incoming) is None
        if not close:
            cache.publish(incoming, actual, cache.store.epoch)
            assert cache.prefix(incoming) is not None

    probe = SessionHashCache(1 << 20, 4)
    probe.publish(incoming, native, probe.store.epoch)
    capacity = probe.store.bytes_used + 1
    cache = SessionHashCache(capacity, 4)
    cache.publish(incoming, native, cache.store.epoch)
    with cache.store.borrow(("salt", "session")) as snapshot:
        assert snapshot is not None
        incoming.session_id = "another"
        cache.publish(incoming, native, cache.store.epoch)
        assert cache.prefix(incoming) is None  # Pinned retired reader consumes budget.
        assert 0 < cache.store.bytes_used <= capacity
        assert snapshot.hashes == native.block_hashes
    assert cache.store.bytes_used == 0
    incoming.session_id = "session"
    cache.publish(incoming, native, cache.store.epoch)
    initial = cache.prefix(incoming)
    assert initial is not None
    assert (
        Request.from_engine_core_request(incoming, hasher, initial).block_hashes
        == native.block_hashes
    )


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    """Test that Request preserves session ID from EngineCoreRequest."""
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_request_copies_kv_hints_from_engine_core_request():
    """Test that Request preserves KV hints from EngineCoreRequest."""
    kv_hints = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="msg-1",
        actions=[
            KvHintAction(
                action_id="action-1",
                action_type="example.action",
                action_version="1.0",
                payload={},
            )
        ],
    )
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        kv_hints=kv_hints,
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.kv_hints == kv_hints
