# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AsyncLLM disagg DP routing (ROCm MoRI-IO path).

Covers the pure ``_pick_dp_rank_for_request`` (blake2s over transfer_id or
request_id, dp_rank_hint short-circuit) and the lone mutation
``_ensure_disagg_transfer_id`` (stamps the shared transfer_id). Both only read
``parallel_config`` / ``params``, so we use lightweight stand-ins.
"""

import hashlib
from types import SimpleNamespace

import pytest

from vllm.v1.engine.async_llm import AsyncLLM

pick = AsyncLLM._pick_dp_rank_for_request
ensure = AsyncLLM._ensure_disagg_transfer_id


def _self(dp_size: int, dp_local: int | None = None):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size,
                data_parallel_size_local=dp_local,
            )
        )
    )


def _expected(key: str, dp_size: int) -> int:
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dp_size


@pytest.mark.parametrize("dp_size", [0, 1])
def test_single_or_no_dp_returns_none(dp_size):
    # DP disabled (size <= 1) => no rank pinning.
    assert pick(_self(dp_size), "req-1", SimpleNamespace(extra_args=None)) is None


def test_no_extra_args_hashes_request_id():
    # Non-disagg request (no extra_args): hash the request_id directly.
    dp = 16
    rid = "cmpl-abcdef-0"
    got = pick(_self(dp), rid, SimpleNamespace(extra_args=None))
    assert got == _expected(rid, dp)
    assert 0 <= got < dp


def test_deterministic_same_request_id():
    dp = 16
    rid = "cmpl-stable-7"
    a = pick(_self(dp), rid, SimpleNamespace(extra_args=None))
    b = pick(_self(dp), rid, SimpleNamespace(extra_args=None))
    assert a == b


def test_all_results_in_range_across_many_ids():
    dp = 8
    for i in range(1000):
        r = pick(_self(dp), f"req-{i}", SimpleNamespace(extra_args=None))
        assert 0 <= r < dp


def test_explicit_dp_rank_hint_wins():
    dp = 16
    params = SimpleNamespace(extra_args={"kv_transfer_params": {"dp_rank_hint": 5}})
    assert pick(_self(dp), "req-x", params) == 5


@pytest.mark.parametrize("bad_hint", [-1, 16, 999, "3", None])
def test_invalid_hint_falls_back_to_hash(bad_hint):
    dp = 16
    params = SimpleNamespace(
        extra_args={"kv_transfer_params": {"dp_rank_hint": bad_hint}}
    )
    got = pick(_self(dp), "req-y", params)
    # Out-of-range / non-int hints are ignored; with no stored transfer_id the
    # read-only pick falls back to hashing the request_id.
    assert got == _expected("req-y", dp)


def test_existing_transfer_id_is_used_for_hash():
    dp = 16
    tid = "transfer-zzz"
    params = SimpleNamespace(extra_args={"kv_transfer_params": {"transfer_id": tid}})
    assert pick(_self(dp), "req-ignored", params) == _expected(tid, dp)


def test_pick_is_read_only():
    # pick must not mutate params: no kv_transfer_params/transfer_id created.
    dp = 16
    extra: dict = {}
    pick(_self(dp), "req-ro", SimpleNamespace(extra_args=extra))
    assert extra == {}

    ktp: dict = {}
    pick(_self(dp), "req-ro2", SimpleNamespace(extra_args={"kv_transfer_params": ktp}))
    assert ktp == {}


# --- _ensure_disagg_transfer_id: the single intentional mutation -------------


def test_ensure_synthesizes_transfer_id_when_absent():
    rid = "cmpl-needs-tid-0"
    ktp: dict = {}
    params = SimpleNamespace(extra_args={"kv_transfer_params": ktp})
    ensure(_self(16), rid, params)
    # transfer_id synthesized from request_id and stored so the connector's
    # decode-side hash routing keys off the SAME string (no sidecar-<rid> split).
    assert ktp["transfer_id"] == rid


def test_ensure_creates_kv_transfer_params_when_extra_args_empty():
    rid = "cmpl-empty-extra-0"
    extra: dict = {}
    ensure(_self(16), rid, SimpleNamespace(extra_args=extra))
    assert extra["kv_transfer_params"]["transfer_id"] == rid


def test_ensure_preserves_existing_transfer_id():
    tid = "sidecar-supplied"
    ktp = {"transfer_id": tid}
    ensure(_self(16), "req-x", SimpleNamespace(extra_args={"kv_transfer_params": ktp}))
    assert ktp["transfer_id"] == tid


def test_ensure_skips_when_hint_present():
    # Explicit sidecar pin => no shared key to synthesize.
    ktp = {"dp_rank_hint": 5}
    ensure(_self(16), "req-x", SimpleNamespace(extra_args={"kv_transfer_params": ktp}))
    assert "transfer_id" not in ktp


def test_ensure_noop_without_extra_args():
    params = SimpleNamespace(extra_args=None)
    ensure(_self(16), "req-x", params)
    assert params.extra_args is None


def test_ensure_then_pick_converge_on_request_id():
    # End-to-end: caller runs ensure, then pick; both legs hash request_id.
    dp = 16
    rid = "cmpl-converge-0"
    params = SimpleNamespace(extra_args={"kv_transfer_params": {}})
    ensure(_self(dp), rid, params)
    got = pick(_self(dp), rid, params)
    assert params.extra_args["kv_transfer_params"]["transfer_id"] == rid
    assert got == _expected(rid, dp)


def test_local_dp_size_caps_rank_space():
    # Global DP=16 but 8 local ranks per node => rank must land in [0, 8).
    dp_global, dp_local = 16, 8
    for i in range(500):
        r = pick(
            _self(dp_global, dp_local), f"req-{i}", SimpleNamespace(extra_args=None)
        )
        assert 0 <= r < dp_local
