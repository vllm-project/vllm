# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-fair KV-read routing across MoRIIO heterogeneous P/D configs.

Dependency-light (no GPU / ROCm / mori): builds bare ``MoRIIOConnectorWorker``
instances via ``object.__new__`` and drives the REAL read-source routing
decision (``_resolve_read_source`` and friends) directly — no routing logic is
re-implemented here, so a future change to the routing functions is what these
tests exercise, not a stale copy.

Scope — the CONNECTOR (decode-side) read routing. The connector never selects
the prefill instance; the proxy does and hands the connector a ``remote_host`` +
``remote_dp_rank`` per request. So the connector's only fairness lever is WHICH
prefill (dp, tp) rank each read targets, and the RFC's five deployments collapse
to three real connector behaviours:

    symmetric TP  (TP prefill + TP decode)  -> decode tp_k reads prefill tp_k
                                               1P_TP8:1D_TP8, 2P_TP8:2D_TP8
    flexible      (TP prefill + DP decode)  -> round-robin over prefill tp0..N-1
                                               2P_TP8:1D_DP8EP  (the mirror)
    owner DP      (DP prefill, any decode)  -> read the owner rank, tp0
                                               2P_DP8EP:3D_DP8EP, 2P_DP8EP:4D_TP8

These tests assume the proxy delivers a FAIR request stream (each prefill
instance + owner dp-rank an equal share) and verify the connector never then
re-introduces a bottleneck. That proxy CONTRACT is verified separately against
the real ``flat_interleaved_dp_route`` in the toy-proxy tests (PR #46115).

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

# (dp_size, tp_size) per node mode; 8 GPUs per node either way.
MODE_DIMS = {"TP8": (1, 8), "DP8EP": (8, 1)}


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
    def n_prefill_gpus_per_instance(self) -> int:
        return self.p_dp * self.p_tp


# xP:yD ratios are invisible to the connector (the proxy owns instance choice),
# so the RFC's five deployments are keyed here by their (prefill, decode) modes.
CONFIGS = [
    PDConfig("1P_TP8:1D_TP8", "TP8", "TP8"),
    PDConfig("2P_TP8:1D_DP8EP", "TP8", "DP8EP"),
    PDConfig("2P_TP8:2D_TP8", "TP8", "TP8"),
    PDConfig("2P_DP8EP:3D_DP8EP", "DP8EP", "DP8EP"),
    PDConfig("2P_DP8EP:4D_TP8", "DP8EP", "TP8"),
]
CONFIG_IDS = [c.name for c in CONFIGS]


def make_decode_worker(*, world_size: int, tp_rank: int, dp_rank: int):
    """A bare worker with only the attributes the routing decision reads."""
    w = object.__new__(MoRIIOConnectorWorker)
    w.world_size = world_size
    w.tp_rank = tp_rank
    w.dp_rank = dp_rank
    w.use_mla = True
    return w


def make_meta(*, p_tp: int, p_dp: int, remote_dp_rank: int, host: str,
              remote_hosts: list[str] | None = None) -> ReqMeta:
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
        remote_hosts=remote_hosts,
    )


def build_decode_workers(cfg: PDConfig) -> list:
    """The decode workers that issue reads for one prefill instance.

    A TP decode instance reads from every one of its tp ranks (each shard needs
    the KV); a DP decode instance reads from each independent dp-rank worker.
    Workers are created ONCE and reused so per-worker round-robin state advances
    exactly as it would in a live decode engine.
    """
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
    """Drive the REAL connector routing over a fair input stream and tally which
    prefill GPU (within one instance) each read targets.

    The only modelled assumption is the proxy CONTRACT — each owner dp-rank is
    delivered an equal number of times (``for owner_dp in range(p_dp)``). No
    proxy routing algorithm is reproduced. Everything else comes from the real
    ``_resolve_read_source``.
    """
    workers = build_decode_workers(cfg)
    hits: Counter = Counter()
    for _ in range(rounds):
        for owner_dp in range(cfg.p_dp):
            for worker in workers:
                meta = make_meta(
                    p_tp=cfg.p_tp,
                    p_dp=cfg.p_dp,
                    remote_dp_rank=owner_dp,
                    host="phost0",
                )
                chosen_tp, _flexible, host = worker._resolve_read_source(meta)
                assert host == "phost0"  # single instance -> instance host
                target = get_port_offset(owner_dp, chosen_tp, cfg.p_tp)
                assert 0 <= target < cfg.n_prefill_gpus_per_instance
                hits[target] += 1
    return hits


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_kv_read_load_is_hardware_fair(cfg: PDConfig) -> None:
    # rounds a multiple of p_tp so the round-robin (flexible mirror) closes on an
    # exact cycle -> perfectly uniform, asserted as max == min.
    hits = prefill_target_multiset(cfg, rounds=16)

    gpus = set(range(cfg.n_prefill_gpus_per_instance))
    # 1) Coverage: no prefill GPU is starved.
    assert set(hits) == gpus, f"unused prefill GPUs: {sorted(gpus - set(hits))}"
    # 2) Balance: every prefill GPU serves the exact same number of reads.
    assert max(hits.values()) == min(hits.values()), dict(sorted(hits.items()))


@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
def test_flexible_gate_fires_only_for_tp_prefill_dp_decode(cfg: PDConfig) -> None:
    flags = set()
    for worker in build_decode_workers(cfg):
        meta = make_meta(p_tp=cfg.p_tp, p_dp=cfg.p_dp, remote_dp_rank=0, host="h")
        _chosen, flexible, _host = worker._resolve_read_source(meta)
        flags.add(flexible)
    # Flexible (round-robin over prefill TP ranks) is valid ONLY when decode is
    # TP1 (DP mode) AND prefill is pure TP with >1 rank + MLA — the mirror.
    is_mirror = cfg.d_tp == 1 and cfg.p_dp == 1 and cfg.p_tp > 1
    assert flags == {is_mirror}


def test_symmetric_tp_is_a_bijection() -> None:
    # TP prefill + TP decode: decode tp_k reads prefill tp_k, so the 8 decode
    # ranks cover the 8 prefill ranks one-to-one (none doubled, none idle).
    targets = []
    for tp_rank in range(8):
        worker = make_decode_worker(world_size=8, tp_rank=tp_rank, dp_rank=0)
        meta = make_meta(p_tp=8, p_dp=1, remote_dp_rank=0, host="phost0")
        chosen_tp, flexible, _host = worker._resolve_read_source(meta)
        assert not flexible
        targets.append(chosen_tp)
    assert sorted(targets) == list(range(8))


def test_owner_dp_read_is_faithful_and_covers_every_rank() -> None:
    # DP prefill: KV is partitioned, so each read MUST hit the owner dp-rank
    # (tp0). The connector must not collapse to a fixed rank; enumerating the
    # owner-rank domain must bijectively cover the instance's 8 GPUs.
    targets = []
    worker = make_decode_worker(world_size=8, tp_rank=3, dp_rank=0)
    for owner_dp in range(8):
        meta = make_meta(p_tp=1, p_dp=8, remote_dp_rank=owner_dp, host="phost0")
        chosen_tp, flexible, _host = worker._resolve_read_source(meta)
        assert not flexible
        assert chosen_tp == 0  # p_tp == 1, only tp0 exists
        targets.append(get_port_offset(owner_dp, chosen_tp, 1))
    assert sorted(targets) == list(range(8))


def test_flexible_round_robin_is_deterministic_uniform_and_staggered() -> None:
    # One decode rank round-robins evenly over prefill tp0..7.
    w0 = make_decode_worker(world_size=1, tp_rank=0, dp_rank=0)
    seq = [w0._next_flex_tp_rank(8) for _ in range(64)]
    assert seq[:8] == list(range(8))  # deterministic 0,1,..,7
    assert Counter(seq) == Counter({t: 8 for t in range(8)})  # exactly uniform

    # Distinct decode DP ranks are phase-staggered by dp_rank, so at read index 0
    # they target DIFFERENT prefill ranks (no synchronized hot rank).
    first_pick = [
        make_decode_worker(
            world_size=1, tp_rank=0, dp_rank=d
        )._next_flex_tp_rank(8)
        for d in range(8)
    ]
    assert sorted(first_pick) == list(range(8))


def test_read_source_threads_one_tp_through_handshake_and_read() -> None:
    # The chosen prefill rank must reach the handshake dial, the session key, and
    # the notify port identically — drift reads one rank but notifies another,
    # leaking that rank's prefill buffer (MR overflow). Verify _read_blocks_for_req
    # threads the SAME chosen_tp into both downstream calls.
    worker = make_decode_worker(world_size=1, tp_rank=0, dp_rank=3)
    worker._ensure_remote_dp_tp_handshaked = MagicMock()
    worker._read_blocks = MagicMock()
    meta = make_meta(p_tp=8, p_dp=1, remote_dp_rank=0, host="phost0")

    worker._read_blocks_for_req("req0", meta)

    handshake_tp = worker._ensure_remote_dp_tp_handshaked.call_args.args[1]
    read_kwargs = worker._read_blocks.call_args.kwargs
    assert read_kwargs["chosen_tp"] == handshake_tp
    assert read_kwargs["flexible"] is True
    # Notify port offset is computed from the same chosen_tp.
    assert get_port_offset(0, read_kwargs["chosen_tp"], 8) == handshake_tp


def test_multi_node_tp_prefill_spreads_reads_across_nodes() -> None:
    # A single TP16 prefill instance split across 2 nodes: flexible reads must
    # spread across BOTH nodes' NICs, not pile onto node 0. tp0..7 -> node 0,
    # tp8..15 -> node 1 (get_port_offset / _pick_remote_rank_host ordering).
    hosts = ["phost0", "phost1"]
    worker = make_decode_worker(world_size=1, tp_rank=0, dp_rank=0)
    node_hits: Counter = Counter()
    for _ in range(160):
        meta = make_meta(
            p_tp=16, p_dp=1, remote_dp_rank=0, host=hosts[0], remote_hosts=hosts
        )
        chosen_tp, flexible, host = worker._resolve_read_source(meta)
        assert flexible
        node_hits[host] += 1
    assert set(node_hits) == set(hosts)
    assert node_hits["phost0"] == node_hits["phost1"]
