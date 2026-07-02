# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-fair KV-read routing across MoRIIO heterogeneous P/D configs.

Dependency-light (no GPU / ROCm / mori): builds bare ``MoRIIOConnectorWorker``
instances via ``object.__new__`` and drives the REAL read-source routing
decision (``_resolve_read_source`` / ``_next_flex_tp_rank``) directly -- no
routing logic is re-implemented here, so a future change to those functions is
what these tests exercise.

Scope -- the CONNECTOR (decode-side) read routing. The connector never selects
the prefill instance; the proxy does and hands the connector a ``remote_host`` +
``remote_dp_rank`` per request. The connector's only fairness lever is WHICH
prefill (dp, tp) rank each read targets, and the RFC's deployments collapse to
three real connector behaviours:

    symmetric TP  (TP prefill + TP decode)  -> decode tp_k reads prefill tp_k
    flexible      (TP prefill + DP decode)  -> round-robin over prefill tp0..N-1
    owner DP      (DP prefill, any decode)  -> read the owner rank, tp0

These assume the proxy delivers a FAIR request stream (each prefill instance +
owner dp-rank an equal share) and verify the connector never re-introduces a
bottleneck. That proxy contract is checked separately against the real
``flat_interleaved_dp_route`` in the toy-proxy tests (PR #46115).

A node runs one of two modes (8 GPUs each):
  * TP8   -> (dp_size, tp_size) = (1, 8); MLA latent KV REPLICATED on all ranks.
  * DP8EP -> (dp_size, tp_size) = (8, 1); KV PARTITIONED, one owner rank/request.
"""

from collections import Counter
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ReqMeta,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)

MODE_DIMS = {"TP8": (1, 8), "DP8EP": (8, 1)}  # (dp_size, tp_size), 8 GPUs/node


@dataclass(frozen=True)
class PDConfig:
    name: str
    p_mode: str
    d_mode: str

    @property
    def p_dp(self) -> int:
        return MODE_DIMS[self.p_mode][0]

    @property
    def p_tp(self) -> int:
        return MODE_DIMS[self.p_mode][1]

    @property
    def d_dp(self) -> int:
        return MODE_DIMS[self.d_mode][0]

    @property
    def d_tp(self) -> int:
        return MODE_DIMS[self.d_mode][1]

    @property
    def n_prefill_gpus(self) -> int:
        return self.p_dp * self.p_tp


CONFIGS = [
    PDConfig("1P_TP8:1D_TP8", "TP8", "TP8"),
    PDConfig("2P_TP8:1D_DP8EP", "TP8", "DP8EP"),
    PDConfig("2P_TP8:2D_TP8", "TP8", "TP8"),
    PDConfig("2P_DP8EP:3D_DP8EP", "DP8EP", "DP8EP"),
    PDConfig("2P_DP8EP:4D_TP8", "DP8EP", "TP8"),
]
CONFIG_IDS = [c.name for c in CONFIGS]


def make_decode_worker(*, world_size: int, tp_rank: int, dp_rank: int):
    w = object.__new__(MoRIIOConnectorWorker)
    w.world_size = world_size
    w.tp_rank = tp_rank
    w.dp_rank = dp_rank
    w.use_mla = True
    return w


def make_meta(*, p_tp: int, p_dp: int, remote_dp_rank: int, host: str = "phost0"):
    return ReqMeta(
        transfer_id="t",
        local_block_ids=[1],
        remote_block_ids=[2],
        remote_host=host,
        remote_port=1234,
        remote_handshake_port=6301,
        remote_notify_port=61005,
        remote_engine_id=f"{host}:6301",
        tp_size=p_tp,
        remote_dp_size=p_dp,
        remote_dp_rank=remote_dp_rank,
    )


def build_decode_workers(cfg: PDConfig) -> list:
    """Decode workers that issue reads for one prefill instance. A TP decode
    instance reads from every tp rank; a DP decode instance from each dp-rank
    worker. Reused across requests so per-worker round-robin state advances."""
    if cfg.d_tp > 1:
        return [
            make_decode_worker(world_size=cfg.d_tp, tp_rank=r, dp_rank=0)
            for r in range(cfg.d_tp)
        ]
    return [
        make_decode_worker(world_size=1, tp_rank=0, dp_rank=d)
        for d in range(cfg.d_dp)
    ]


def prefill_target_multiset(cfg: PDConfig, rounds: int) -> Counter:
    """Drive the REAL _resolve_read_source over a fair input stream; tally which
    prefill GPU each read targets. The only modelled assumption is the proxy
    contract -- each owner dp-rank delivered equally (``for owner_dp in
    range(p_dp)``); no proxy algorithm is reproduced."""
    workers = build_decode_workers(cfg)
    hits: Counter = Counter()
    for _ in range(rounds):
        for owner_dp in range(cfg.p_dp):
            for worker in workers:
                meta = make_meta(
                    p_tp=cfg.p_tp, p_dp=cfg.p_dp, remote_dp_rank=owner_dp
                )
                chosen_tp, _flexible = worker._resolve_read_source(meta)
                target = get_port_offset(owner_dp, chosen_tp, cfg.p_tp)
                assert 0 <= target < cfg.n_prefill_gpus
                hits[target] += 1
    return hits


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_kv_read_load_is_hardware_fair(cfg: PDConfig) -> None:
    # rounds a multiple of p_tp so the round-robin closes on an exact cycle.
    hits = prefill_target_multiset(cfg, rounds=16)
    gpus = set(range(cfg.n_prefill_gpus))
    assert set(hits) == gpus, f"unused prefill GPUs: {sorted(gpus - set(hits))}"
    assert max(hits.values()) == min(hits.values()), dict(sorted(hits.items()))


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_flexible_gate_fires_only_for_tp_prefill_dp_decode(cfg: PDConfig) -> None:
    flags = set()
    for worker in build_decode_workers(cfg):
        meta = make_meta(p_tp=cfg.p_tp, p_dp=cfg.p_dp, remote_dp_rank=0)
        _chosen, flexible = worker._resolve_read_source(meta)
        flags.add(flexible)
    is_mirror = cfg.d_tp == 1 and cfg.p_dp == 1 and cfg.p_tp > 1
    assert flags == {is_mirror}


def test_symmetric_tp_is_a_bijection() -> None:
    targets = []
    for tp_rank in range(8):
        worker = make_decode_worker(world_size=8, tp_rank=tp_rank, dp_rank=0)
        chosen_tp, flexible = worker._resolve_read_source(
            make_meta(p_tp=8, p_dp=1, remote_dp_rank=0)
        )
        assert not flexible
        targets.append(chosen_tp)
    assert sorted(targets) == list(range(8))


def test_owner_dp_read_is_faithful_and_covers_every_rank() -> None:
    worker = make_decode_worker(world_size=8, tp_rank=3, dp_rank=0)
    targets = []
    for owner_dp in range(8):
        chosen_tp, flexible = worker._resolve_read_source(
            make_meta(p_tp=1, p_dp=8, remote_dp_rank=owner_dp)
        )
        assert not flexible
        assert chosen_tp == 0  # p_tp == 1, only tp0 exists
        targets.append(get_port_offset(owner_dp, chosen_tp, 1))
    assert sorted(targets) == list(range(8))


def test_flexible_round_robin_is_deterministic_uniform_and_staggered() -> None:
    w0 = make_decode_worker(world_size=1, tp_rank=0, dp_rank=0)
    seq = [w0._next_flex_tp_rank(8) for _ in range(64)]
    assert seq[:8] == list(range(8))
    assert Counter(seq) == Counter({t: 8 for t in range(8)})

    first_pick = [
        make_decode_worker(
            world_size=1, tp_rank=0, dp_rank=d
        )._next_flex_tp_rank(8)
        for d in range(8)
    ]
    assert sorted(first_pick) == list(range(8))


def test_read_blocks_for_req_threads_chosen_tp() -> None:
    # The resolved (chosen_tp, flexible) must reach _read_blocks, which keys the
    # session AND the notify port off that single value -- so a read and its
    # completion notify address the same prefill rank.
    worker = make_decode_worker(world_size=1, tp_rank=0, dp_rank=3)
    worker._read_blocks = MagicMock()
    worker._read_blocks_for_req("r", make_meta(p_tp=8, p_dp=1, remote_dp_rank=0))
    kw = worker._read_blocks.call_args.kwargs
    assert kw["flexible"] is True
    assert kw["chosen_tp"] == 3  # first flexible pick = dp_rank seed
    assert get_port_offset(0, kw["chosen_tp"], 8) == 3
