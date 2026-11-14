# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP ranks must sample alike: collective sampler choice and config check (CPU-only,
collectives faked)."""

import pytest
import torch
import torch.distributed as dist

from vllm.v1.worker.gpu.sample import sampler as sampler_mod
from vllm.v1.worker.gpu.sample import tp_consistency


def test_no_tp_group_is_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tp_consistency, "_tp_cpu_group", lambda: None)
    assert tp_consistency.all_tp_ranks_agree(True) is True
    assert tp_consistency.all_tp_ranks_agree(False) is False

    def must_not_build():
        raise AssertionError("config built without TP ranks to compare")

    tp_consistency.check_sampling_config(must_not_build)


@pytest.fixture
def two_ranks(monkeypatch: pytest.MonkeyPatch):
    """Rank 0's view of a 2-rank TP group whose other rank reports `peer`."""
    peer: dict = {}
    monkeypatch.setattr(tp_consistency, "_tp_cpu_group", lambda: object())
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    def all_reduce(t, op, group):
        assert op == dist.ReduceOp.MIN
        t.copy_(torch.minimum(t, torch.tensor([int(peer["flag"])], dtype=t.dtype)))

    def all_gather_object(out, obj, group):
        out[0], out[1] = obj, peer["config"]

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    return peer


@pytest.mark.parametrize(
    "local, peer, expected",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_all_tp_ranks_agree(two_ranks, local, peer, expected):
    two_ranks["flag"] = peer
    assert tp_consistency.all_tp_ranks_agree(local) is expected


def test_same_config_passes(two_ranks):
    two_ranks["config"] = {"seed": 0, "use_flashinfer": True}
    tp_consistency.check_sampling_config(lambda: {"seed": 0, "use_flashinfer": True})


def test_different_seed_raises(two_ranks):
    two_ranks["config"] = {"seed": 1, "use_flashinfer": True}
    with pytest.raises(RuntimeError, match=r"seed per TP rank = \[0, 1\]"):
        tp_consistency.check_sampling_config(
            lambda: {"seed": 0, "use_flashinfer": True}
        )


def test_fused_sampler_needs_every_rank(two_ranks):
    two_ranks["flag"] = False  # the other rank cannot use FlashInfer
    assert sampler_mod._agreed_support(True, lambda: True, "FlashInfer") is False
    two_ranks["flag"] = True
    assert sampler_mod._agreed_support(True, lambda: True, "FlashInfer") is True

    # Ineligible here (e.g. sampling masks requested): never asks the backend.
    def not_asked() -> bool:
        raise AssertionError("backend queried for an ineligible sampler")

    assert sampler_mod._agreed_support(False, not_asked, "FlashInfer") is False


def test_local_error_raised_after_collective(two_ranks, monkeypatch):
    two_ranks["flag"] = True
    votes: list[bool] = []
    agree = tp_consistency.all_tp_ranks_agree

    def recording(local: bool) -> bool:
        votes.append(local)
        return agree(local)

    monkeypatch.setattr(sampler_mod, "all_tp_ranks_agree", recording)

    def unavailable_but_requested():
        raise RuntimeError("FlashInfer top-p/top-k sampling unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        sampler_mod._agreed_support(True, unavailable_but_requested, "FlashInfer")
    assert votes == [False]  # the other ranks still got this rank's vote
