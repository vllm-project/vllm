# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the DCP "front" query order (`cutedsl.dsa.overlap`
QUERY_AFTER_FRONT): MlaQuery forked after IndexFront and launched without PDL.

- Policy (no GPU): the table names DCP16 at 6-8 rows only.
- Wiring (no GPU): `PreparedDsa.run_dcp` runs the indexer inputs (IndexFront)
  before forking MlaQuery for the listed shapes, after it for every other shape, and
  the sparse core waits for the query either way.

The kernel's bits with and without PDL: `test_specialized_decode_dsa.py`
(`test_query_without_pdl`).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import overlap
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
    prepared as dsa_prepared,
)


@pytest.mark.parametrize("dcp", [1, 2, 4, 8, 16])
def test_query_after_front_table(dcp):
    for batch in range(1, 321):
        want = dcp == 16 and 6 <= batch <= 8
        assert overlap.query_after_front(batch, dcp) is want, (dcp, batch)


@pytest.mark.parametrize("after", [False, True], ids=["fork-first", "front-first"])
def test_run_dcp_query_order(after, monkeypatch):
    batch, q = 8, 8
    calls: list = []
    dsa = dsa_prepared.PreparedDsa.__new__(dsa_prepared.PreparedDsa)
    dsa.box = SimpleNamespace(mailbox=lambda b: SimpleNamespace(format="latent"))
    dsa.parity, dsa.heads = 1, 6
    dsa.kernels = {(batch, q): ("query", "index", "value")}
    dsa.query_stage = SimpleNamespace(views=(None, None, None, None, "uv"))
    dsa.after_front = {(batch, q)} if after else set()

    def run_index(kernel, projected, index_slots, q):
        calls.append(("index", kernel, index_slots))

    def run_query(projected, cache, slots, q, publish=()):
        calls.append(("query", slots, publish))
        return "join"

    def dcp_core(cache, topk, metadata, batch, projected, join, q):
        calls.append(("core", join))
        return "core"

    monkeypatch.setattr(dsa, "run_index", run_index)
    monkeypatch.setattr(dsa, "run_query", run_query)
    monkeypatch.setattr(dsa, "dcp_core", dcp_core)
    lengths = torch.arange(1, 2)
    monkeypatch.setattr(
        dsa_prepared, "pad_requests", lambda reqs, table, lens: (None, lengths, None)
    )
    monkeypatch.setattr(dsa_prepared, "publish_args", lambda box, p: ("publish", p))

    def run_value(value, box, p, publish, heads, core, lens, uv, *rest):
        calls.append(("value", value, core, uv))
        return "out"

    monkeypatch.setattr(dsa_prepared, "run_value", run_value)
    metadata = SimpleNamespace(block_table=None, seq_lens=None)
    projected = torch.empty(batch, 1)
    out = dsa.run_dcp(projected, "out", "cache", "slots", "islots", metadata, q)
    assert out == "out"
    index = ("index", "index", "islots")
    query = ("query", "slots", ("publish", 1))
    first = [index, query] if after else [query, index]
    assert calls == first + [("core", "join"), ("value", "value", "core", "uv")]
