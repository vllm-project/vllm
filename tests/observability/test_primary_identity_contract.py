from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest


def _load_primary_usdt():
    path = Path(__file__).parents[2] / "vllm" / "primary_usdt.py"
    spec = importlib.util.spec_from_file_location("g2_primary_usdt_identity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_primary_usdt()


def test_a02_engine_identity_is_nonzero_128_bit_and_restart_distinct():
    identities = {M.new_engine_instance_id() for _ in range(64)}
    assert len(identities) == 64
    for high, low in identities:
        assert (high, low) != (0, 0)
        assert 0 <= high <= M.UINT64_MAX
        assert 0 <= low <= M.UINT64_MAX


def test_a05_zero_is_valid_and_unavailable_sentinel_is_never_allocated():
    counter = M.SourceOwnedCounter()
    assert [counter.next(), counter.next(), counter.next()] == [0, 1, 2]
    counter = M.SourceOwnedCounter(M.UINT64_MAX - 1)
    assert counter.next() == M.UINT64_MAX - 1
    with pytest.raises(OverflowError):
        counter.next()


def test_a03_submission_mapping_is_source_owned_and_immutable():
    engine = M.new_engine_instance_id()
    attempt = M.new_submission_attempt(engine)
    request_hash = M.primary_request_id_hash("request-A")
    M.bind_submission_attempt(engine, request_hash, attempt)
    assert M.submission_attempt_for_request(engine, request_hash) == attempt
    with pytest.raises(M.PrimaryUSDTError, match="multiple submissions"):
        M.bind_submission_attempt(engine, request_hash, attempt + 1)


def test_a04_terminal_closure_dedup_and_output_ordinal_state_machine():
    engine = M.new_engine_instance_id()
    request_hash = M.primary_request_id_hash("request-terminal")
    assert M.next_output_ordinal(engine, request_hash) == 0
    assert M.next_output_ordinal(engine, request_hash) == 1
    assert M.mark_terminal_once(engine, request_hash)
    assert not M.mark_terminal_once(engine, request_hash)
    assert M.terminal_was_marked(engine, request_hash)
    assert M.mark_frontend_closed_once(engine, request_hash)
    assert not M.mark_frontend_closed_once(engine, request_hash)


def test_a16_request_hash_is_stable_nonzero_and_not_case_name_identity():
    assert M.primary_request_id_hash("request-A") == M.primary_request_id_hash(
        "request-A"
    )
    assert M.primary_request_id_hash("request-A") != 0
    assert M.primary_request_id_hash("request-A") != M.primary_request_id_hash(
        "request-B"
    )
    assert M.primary_request_id_hash(None) == 0


def test_a03_async_pre_admission_exception_closes_submission():
    backend = M.CaptureBackend()
    M._GLOBAL_EMITTER = M.PrimaryUSDTEmitter(backend, enabled=True)

    class Dummy:
        _primary_engine_instance_id = (7, 9)

        @M.primary_async_submission
        async def fail(self):
            raise ValueError("rejected")

    with pytest.raises(ValueError, match="rejected"):
        asyncio.run(Dummy().fail())
    probes = [name for name, _ in backend.records]
    assert any(name.startswith("request_arrival_v2__") for name in probes)
    assert any(name.startswith("request_terminal_v2__") for name in probes)
    assert not any(name.startswith("request_cleanup_v3__") for name in probes)
