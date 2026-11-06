# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU tests of the exact one-round DCP decode top-k merge: the packed
layout, sizing and gating, the construction gate, the host-side call checks
and the packed-mode contract helper. The merge itself (select, rescue, CUDA
graphs, late peers) runs over a real DCP group in
``tests/distributed/test_dcp_topk_merge.py``.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl


def _has_deep_select() -> bool:
    try:
        import vllm._deepselect_C  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not has_cutedsl()
    or not _has_deep_select(),
    reason="The DCP top-k merge needs SM100, CuTe DSL and DeepSelect",
)

TOPK = 2048


class _FakeDCPGroup:
    """``all_gather(dim=0)`` of ``world_size`` copies of the input."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        return input_.repeat(self.world_size, *([1] * (input_.dim() - 1)))


def _local_case(lengths: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One rank's bf16 logits (rows 1024-byte aligned, as DeepSelect reads
    them) and row ends."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    width = -(-max(max(lengths), 1) // 512) * 512
    logits = torch.randn((len(lengths), width), generator=gen, device="cuda")
    ends = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return logits.to(torch.bfloat16), ends


# --- Sizing and gating ---------------------------------------------------


def test_local_topk_sizing():
    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_merge_local_topk,
        packed_row_words,
    )

    assert dcp_merge_local_topk(2048, 8) == 384
    assert dcp_merge_local_topk(2048, 4) == 768
    assert dcp_merge_local_topk(2048, 16) == 192
    assert dcp_merge_local_topk(2048, 2) == 1536
    assert dcp_merge_local_topk(2048, 1) == 0
    assert packed_row_words(384) == 396


@pytest.mark.parametrize(
    ("dcp", "pcp", "backend", "max_model_len", "expected"),
    [
        (8, 1, "auto", 262_144, 384),
        (8, 1, "deep_select", 1 << 20, 384),
        (1, 1, "auto", 262_144, 0),  # no DCP
        (8, 2, "auto", 262_144, 0),  # PCP keeps the generic merge
        (8, 1, "persistent", 262_144, 0),  # explicit top-k backend
        (8, 1, "auto", (1 << 20) + 8, 0),  # local index past 17 bits
        (16, 1, "auto", 1 << 20, 192),
        (32, 1, "auto", 1 << 20, 0),  # the saturation mask holds 16 ranks
    ],
)
def test_local_topk_config_gating(dcp, pcp, backend, max_model_len, expected):
    from types import SimpleNamespace

    from vllm.v1.attention.ops.dcp_topk_merge import (
        dcp_topk_merge_local_topk,
    )

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
        ),
        kernel_config=SimpleNamespace(sparse_indexer_topk_backend=backend),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )
    assert dcp_topk_merge_local_topk(config, TOPK) == expected


def test_init_fails_closed(monkeypatch):
    """Without symmetric memory over the DCP group (on any rank), under
    ubatching or past 16 DCP ranks the construction gate keeps the generic
    merge (N = 0) instead of a lossy one-round merge. The ranks agree (MIN)
    before the collective allocation, which then never starts; the result is
    computed once per top-k (later layers run no collectives)."""
    from types import SimpleNamespace

    import vllm.distributed as distributed
    import vllm.v1.attention.ops.cp_common as cp_common
    import vllm.v1.attention.ops.dcp as dcp
    from vllm.v1.attention.ops import dcp_topk_merge as gate
    from vllm.v1.attention.ops.dcp_topk_merge import kernels

    def config(ubatching: bool = False, dcp_size: int = 8) -> SimpleNamespace:
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp_size,
                prefill_context_parallel_size=1,
                use_ubatching=ubatching,
            ),
            kernel_config=SimpleNamespace(
                sparse_indexer_topk_backend="auto", enable_jit_warmup=False
            ),
            model_config=SimpleNamespace(max_model_len=262_144),
        )

    collectives = []

    def all_reduce(tensor, op=None, group=None):
        # A peer without symmetric memory: the MIN is 0.
        assert op == torch.distributed.ReduceOp.MIN
        collectives.append(int(tensor[0]))
        tensor.fill_(0)

    def allocate(*args, **kwargs):
        raise AssertionError("allocated after the ranks disagreed")

    monkeypatch.setattr(gate, "_MERGE_MAX_ROWS", {})
    monkeypatch.setattr(gate, "_MERGE_INIT", {})
    monkeypatch.setattr(
        distributed, "get_dcp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(dcp, "get_dcp_workspace_max_num_tokens", lambda cfg: 64)
    monkeypatch.setattr(cp_common, "direct_cp_enabled", lambda *args: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(kernels, "init_dcp_topk_rescue_workspace", allocate)
    assert gate.init_dcp_topk_merge(config(ubatching=True), TOPK) == 0
    assert collectives == []
    gate._MERGE_INIT.clear()
    assert gate.init_dcp_topk_merge(config(dcp_size=32), TOPK) == 0
    assert collectives == []
    gate._MERGE_INIT.clear()
    assert gate.init_dcp_topk_merge(config(), TOPK) == 0
    assert collectives == [1]
    assert gate.init_dcp_topk_merge(config(), TOPK) == 0  # cached
    assert collectives == [1]
    assert gate.dcp_topk_merge_max_rows(TOPK, 384) == 0
    assert not gate.dcp_topk_merge_fits(1, TOPK, 384)


# --- Packed layout ---------------------------------------------------------


def test_pack_layout_bit_exact():
    """The B' word layout, bit for bit, from a hand-computed reference."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack_indices,
    )

    n = 64
    width = 140_000
    scores = torch.tensor(
        [1.5, -1.5, 0.0, -0.0, float("inf"), float("-inf"), 3.0e-39, -2.0],
        dtype=torch.bfloat16,
    )
    local = torch.tensor([0, 1, 65535, 65536, 70000, 131071, 5, 98304])
    logits = torch.zeros((1, width), dtype=torch.bfloat16)
    logits[0, local] = scores
    idx = torch.full((1, n), -1, dtype=torch.int32)
    idx[0, : len(local)] = local.int()
    idx[0, 40] = 7  # a candidate after a gap; logits[7] = 0
    out = torch.full((1, n + n // 32), 123, dtype=torch.int32, device="cuda")
    dcp_topk_pack_indices(logits.cuda(), idx.cuda(), out)

    def word(score_bits: int, loc: int) -> int:
        ordered = (
            (score_bits | 0x8000) if score_bits < 0x8000 else (~score_bits & 0xFFFF)
        )
        return ((ordered << 16) | (0xFFFF - (loc & 0xFFFF))) & 0xFFFFFFFF

    bits = logits.view(torch.int16).int() & 0xFFFF
    expected = [0] * (n + n // 32)
    for j, loc in enumerate(idx[0].tolist()):
        if loc >= 0:
            expected[j] = word(int(bits[0, loc]), loc)
            if loc >> 16:
                expected[n + j // 32] |= 1 << (j % 32)
    got = [v & 0xFFFFFFFF for v in out[0].tolist()]
    assert got == expected


@pytest.mark.parametrize("with_row_ends", [False, True])
def test_pack_drops_invalid_ids_and_nan(with_row_ends):
    """Ids outside ``[0, min(row_end, S, 2**17))`` and NaN scores pack as
    empty slots; +-inf scores pack as ordinary candidates."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack_indices,
    )

    n, width, row_end = 64, 140_000, 100_000
    logits = torch.zeros((1, width), dtype=torch.bfloat16)
    bits = logits.view(torch.int16)
    bits[0, 1] = 0x7FC0  # +NaN
    bits[0, 2] = -64  # 0xFFC0, -NaN
    bits[0, 65535] = -1  # 0xFFFF, -NaN: would pack to the empty word 0
    bits[0, 3] = 0x7F81  # signaling NaN
    logits[0, 4] = float("inf")
    logits[0, 5] = float("-inf")
    logits[0, 99_999] = 2.0
    logits[0, 120_000] = 3.0
    local = [0, 1, 2, 65535, 3, 4, 5, 99_999, 120_000, 131_072, 139_999]
    local += [width, 1 << 30, 2**31 - 1, -5, -(2**31)]
    idx = torch.full((1, n), -1, dtype=torch.int32)
    idx[0, : len(local)] = torch.tensor(local, dtype=torch.int32)
    out = torch.full((1, n + n // 32), 123, dtype=torch.int32, device="cuda")
    ends = torch.tensor([row_end], dtype=torch.int32, device="cuda")
    dcp_topk_pack_indices(
        logits.cuda(), idx.cuda(), out, ends if with_row_ends else None
    )

    bound = row_end if with_row_ends else 1 << 17
    u16 = bits.int() & 0xFFFF
    expected = [0] * (n + n // 32)
    for j, loc in enumerate(local):
        if not 0 <= loc < bound or (int(u16[0, loc]) & 0x7FFF) > 0x7F80:
            continue
        b = int(u16[0, loc])
        ordered = (b | 0x8000) if b < 0x8000 else (~b & 0xFFFF)
        expected[j] = ((ordered << 16) | (0xFFFF - (loc & 0xFFFF))) & 0xFFFFFFFF
        if loc >> 16:
            expected[n + j // 32] |= 1 << (j % 32)
    got = [v & 0xFFFFFFFF for v in out[0].tolist()]
    assert got == expected
    # 0, inf, -inf, 99_999; 120_000 only without row_ends.
    assert sum(1 for w in got[:n] if w) == (4 if with_row_ends else 5)


def test_pack_max_local_len_matches_narrow_logits():
    """DeepGEMM's logits are as wide as the global max_model_len. Bounding
    them by max_local_len (which moves DeepSelect off its <= 6-row cluster
    variant at >= 512K columns) packs the same candidates as logits of the
    local width."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_pack,
    )

    torch.manual_seed(0)
    width, world, local_topk = 1 << 20, 8, 384
    bound = width // world
    ends = torch.tensor([bound, 100_000, 300, 70_000, 5], dtype=torch.int32).cuda()
    logits = torch.randn(ends.shape[0], width, device="cuda").to(torch.bfloat16)

    def candidates(packed: torch.Tensor) -> list[set[tuple[int, int]]]:
        # (packed word, local-index bit 16) per filled slot.
        words = packed[:, :local_topk].tolist()
        planes = packed[:, local_topk:].tolist()
        return [
            {(w, (p[j // 32] >> (j % 32)) & 1) for j, w in enumerate(ws) if w}
            for ws, p in zip(words, planes)
        ]

    full = dcp_topk_pack(logits, ends, local_topk, max_local_len=bound)
    narrow = dcp_topk_pack(logits[:, :bound].contiguous(), ends, local_topk)
    got = candidates(full)
    assert [len(c) for c in got] == ends.clamp(max=local_topk).tolist()
    assert got == candidates(narrow)


# --- Memory profile, host checks and the packed-mode contract -------------


def test_merge_requires_rescue_workspace(monkeypatch):
    """No lossy path: without the rescue workspace the entry points raise,
    and packed mode needs the logits (a saturated rank re-reads its row)."""
    import vllm.distributed as distributed
    from vllm.v1.attention.ops.dcp_topk_merge import kernels
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge,
        dcp_topk_merge_rows,
        dcp_topk_pack,
    )

    n = 384
    logits, ends = _local_case([1_250, 625], seed=3)
    sent = dcp_topk_pack(logits, ends, n)
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: _FakeDCPGroup(8))
    monkeypatch.setattr(kernels, "_RESCUE_WORKSPACES", {})
    out = torch.empty((2, TOPK), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="rescue workspace"):
        dcp_topk_merge(logits, ends, out, n)
    rows = dict(
        local_topk=n,
        topk=TOPK,
        req_id=torch.zeros(2, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        block_size=64,
        stride_rows=64,
        dcp_rank=0,
        rows_out=torch.empty((2, TOPK), dtype=torch.int32, device="cuda"),
        valid_out=torch.empty(2, dtype=torch.int32, device="cuda"),
    )
    with pytest.raises(ValueError, match="logits"):
        dcp_topk_merge_rows(packed=sent, **rows)
    with pytest.raises(RuntimeError, match="rescue workspace"):
        dcp_topk_merge_rows(packed=sent, logits=logits, row_ends=ends, **rows)


def _host_only_workspace(world: int, max_rows: int, n: int):
    """A rescue workspace for the host-side checks only: local buffers, no
    peers. Every call below is refused before a kernel launches."""
    from vllm.v1.attention.ops.dcp_topk_merge import (
        kernels,
        packed_row_words,
        rescue_slots,
    )

    ws = kernels.DCPTopkRescueWorkspace.__new__(kernels.DCPTopkRescueWorkspace)
    ws.device_group = None
    ws.world_size = world
    ws.rank = 0
    recv = torch.zeros(
        (max_rows, rescue_slots(TOPK, n), packed_row_words(TOPK)),
        dtype=torch.int32,
        device="cuda",
    )
    flags = torch.zeros((world, max_rows), dtype=torch.int32, device="cuda")
    no_peers = torch.zeros(world, dtype=torch.int64, device="cuda")
    ws._init(recv, flags, no_peers, no_peers, max_rows, TOPK, n)
    return ws


def test_exact_call_checks(monkeypatch):
    """The kernels index the rescue workspace (and peers' symmetric memory)
    by row, rank and slot without bounds checks, and CuTe accepts mismatched
    tensors silently: the host refuses every call the workspace or the
    compiled layouts do not cover."""
    import vllm.distributed as distributed
    from vllm.v1.attention.ops.dcp_topk_merge import kernels, packed_row_words
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        dcp_topk_merge_rows,
        dcp_topk_select,
    )

    world, max_rows, n = 8, 4, 384
    ws = _host_only_workspace(world, max_rows, n)
    monkeypatch.setattr(kernels, "_RESCUE_WORKSPACES", {(TOPK, n): ws})
    lg, e = _local_case([1_250] * 6, seed=4)
    gathered = torch.zeros(
        (world, 6, packed_row_words(n)), dtype=torch.int32, device="cuda"
    )
    out = torch.empty((6, TOPK), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="exceeds its rescue workspace"):
        dcp_topk_select(gathered, n, out, exact=ws.args(lg, e))
    g4, out4, lg4, e4 = gathered[:, :4].contiguous(), out[:4], lg[:4], e[:4]
    bad_args = [
        (ws.args(lg4.float(), e4), "logits must be bf16"),
        (ws.args(lg4[:2], e4), "logits must be bf16"),
        (ws.args(lg4, e4.long()), "row_ends must be"),
        (ws.args(lg4, e4[:2]), "row_ends must be"),
        (ws.args(lg4, e[:8:2]), "row_ends must be"),
    ]
    for exact, match in bad_args:
        with pytest.raises(ValueError, match=match):
            dcp_topk_select(g4, n, out4, exact=exact)
    with pytest.raises(ValueError, match="does not match its rescue workspace"):
        dcp_topk_select(g4, n, out4[:, :1024], exact=ws.args(lg4, e4))
    with pytest.raises(ValueError, match="does not match its rescue workspace"):
        dcp_topk_select(g4[:4].contiguous(), n, out4, exact=ws.args(lg4, e4))
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: _FakeDCPGroup(world))
    rows = dict(
        packed=g4[0],
        logits=lg4,
        row_ends=e4,
        local_topk=n,
        topk=TOPK,
        req_id=torch.zeros(4, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        block_size=64,
        stride_rows=64,
        dcp_rank=0,
        rows_out=torch.empty((4, TOPK), dtype=torch.int32, device="cuda"),
        valid_out=torch.empty(4, dtype=torch.int32, device="cuda"),
    )
    with pytest.raises(ValueError, match="workspace's rank"):
        dcp_topk_merge_rows(**{**rows, "dcp_rank": 1})
    with pytest.raises(ValueError, match="rows_out must be"):
        dcp_topk_merge_rows(**{**rows, "rows_out": rows["rows_out"][:, :n]})
    with pytest.raises(ValueError, match="packed must be"):
        dcp_topk_merge_rows(**{**rows, "packed": g4[0, :, :n]})


def test_packed_contract_validator():
    """``assert_packed_merge_contract`` accepts the stock producer (DeepSelect
    + pack, including short and empty rows) and catches a dropped word, a
    non-top word and a word whose score is not its logit."""
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        assert_packed_merge_contract,
        dcp_topk_pack,
    )

    n = 384
    lg, e = _local_case([12_500, 3_750, 250, 0], seed=6)
    packed = dcp_topk_pack(lg, e, n)
    assert_packed_merge_contract(packed, lg, e, n)

    words = packed[0, :n]
    j = int(torch.nonzero(words).flatten()[0])
    dropped = packed.clone()
    dropped[0, j] = 0
    with pytest.raises(AssertionError, match="words"):
        assert_packed_merge_contract(dropped, lg, e, n)

    # Replace a sent word by the row's lowest-scoring token.
    low = int(lg[0, : int(e[0])].float().argmin())
    bits = int(lg[0, low : low + 1].view(torch.int16)[0]) & 0xFFFF
    ordered = (~bits & 0xFFFF) if bits & 0x8000 else bits | 0x8000
    non_top = packed.clone()
    word = (ordered << 16) | (0xFFFF - low)
    non_top[0, j] = word - (1 << 32) if word >= 1 << 31 else word
    with pytest.raises(AssertionError, match="top-"):
        assert_packed_merge_contract(non_top, lg, e, n)

    wrong_score = packed.clone()
    wrong_score[0, j] ^= 1 << 16
    with pytest.raises(AssertionError, match="logit"):
        assert_packed_merge_contract(wrong_score, lg, e, n)
