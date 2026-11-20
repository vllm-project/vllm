# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sizing of the DeepEP low-latency per-rank dispatch buffer.

Hermetic: deep_ep is replaced by a fake whose RDMA size hint is linear in the
number of tokens (~9.6 MB/token, which is what DeepEP reports for hidden=4096,
288 experts, 8 ranks).
"""

import os
import sys
import types

import pytest

from vllm.distributed.device_communicators.all2all import (
    DEEPEP_LL_MAX_RDMA_BYTES,
    DeepEPLLAll2AllManager,
    _user_nvshmem_qp_depth,
)

BYTES_PER_TOKEN = 9_584_640
HIDDEN, NUM_RANKS, NUM_EXPERTS = 4096, 8, 288


def _size_hint(num_tokens: int, hidden: int, num_ranks: int, num_experts: int) -> int:
    return num_tokens * BYTES_PER_TOKEN


@pytest.fixture(autouse=True)
def fake_deep_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = types.SimpleNamespace(get_low_latency_rdma_size_hint=_size_hint)
    monkeypatch.setitem(sys.modules, "deep_ep", types.SimpleNamespace(Buffer=buffer))
    monkeypatch.delenv("NVSHMEM_QP_DEPTH", raising=False)
    _user_nvshmem_qp_depth.cache_clear()


def _max_tokens(max_num_batched_tokens: int) -> int:
    return DeepEPLLAll2AllManager.max_dispatch_tokens_per_rank(
        max_num_batched_tokens, HIDDEN, NUM_RANKS, NUM_EXPERTS
    )


@pytest.mark.parametrize(
    "max_num_batched_tokens, expected_tokens",
    [(64, 64), (256, 256), (512, 256), (4096, 256)],
)
def test_default_caps_tokens(
    monkeypatch: pytest.MonkeyPatch, max_num_batched_tokens: int, expected_tokens: int
) -> None:
    assert _max_tokens(max_num_batched_tokens) == expected_tokens
    # deep_ep.Buffer writes its default QP depth into the environment. Later
    # MoE layers must not mistake it for a user override and size differently.
    monkeypatch.setenv("NVSHMEM_QP_DEPTH", "1024")
    assert _max_tokens(max_num_batched_tokens) == expected_tokens


@pytest.mark.parametrize(
    "qp_depth, max_num_batched_tokens, expected_tokens",
    [
        ("1024", 4096, 511),
        ("4096", 1024, 1024),
        # Bounded by QP depth to 4095 tokens, then halved until the RDMA
        # buffer fits DeepEP's 32 GiB limit.
        ("8192", 16384, 2047),
    ],
)
def test_user_qp_depth(
    monkeypatch: pytest.MonkeyPatch,
    qp_depth: str,
    max_num_batched_tokens: int,
    expected_tokens: int,
) -> None:
    monkeypatch.setenv("NVSHMEM_QP_DEPTH", qp_depth)
    num_tokens = _max_tokens(max_num_batched_tokens)
    assert num_tokens == expected_tokens
    assert num_tokens * BYTES_PER_TOKEN < DEEPEP_LL_MAX_RDMA_BYTES
    assert os.environ["NVSHMEM_QP_DEPTH"] == qp_depth


def test_qp_depth_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVSHMEM_QP_DEPTH", "2")
    with pytest.raises(ValueError, match="NVSHMEM_QP_DEPTH=2 is too small"):
        _max_tokens(4096)
