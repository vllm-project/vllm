# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QRingBufferCapture._build_q_step_state row mapping.

Under continuous batching, a request's query-row count in a forward step is
its *scheduled* token count, which need not equal its store op's chunk-aligned
token count. The plan must therefore attribute query rows to ops through
``attn_metadata.slot_mapping`` (row -> GPU KV slot) instead of positionally.
These tests build synthetic steps and assert the exact row set captured for
every op, including the misalignment cases that used to corrupt the capture:

* a chunk-unaligned prompt tail shifting the next request's rows,
* batch row order differing from connector-metadata order,
* mixed STORE/RETRIEVE steps,
* ops whose tokens were partially computed in an earlier step.
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.qringbuffer import QRingBuffer, QRingBufferCapture

BLOCK_SIZE = 16
HIDDEN_DIM = 8
NUM_LAYERS = 2


@dataclass
class _FakeOp:
    """Minimal LoadStoreOp stand-in: block_ids pre-sliced to [start, end)."""

    block_ids: list
    start: int = 0
    end: int = 0
    token_ids: list = field(default_factory=list)


@dataclass
class _FakeRequestMeta:
    request_id: str
    direction: str
    op: _FakeOp
    cache_salt: str = ""


@dataclass
class _FakeConnectorMetadata:
    requests: list


class _Request:
    """A synthetic request: its GPU blocks, scheduled rows, and store op."""

    def __init__(
        self,
        request_id: str,
        num_scheduled_tokens: int,
        first_gpu_block: int,
        store_tokens: int | None = None,
        direction: str = "STORE",
        computed_offset: int = 0,
    ):
        """
        Args:
            num_scheduled_tokens: query rows this request contributes.
            first_gpu_block: first GPU block id (blocks are contiguous).
            store_tokens: the store op's [0, store_tokens) range; defaults to
                the scheduled count aligned down to BLOCK_SIZE.
            computed_offset: tokens computed in *earlier* steps (this step's
                rows start at this token position).
        """
        self.request_id = request_id
        self.num_scheduled = num_scheduled_tokens
        self.computed_offset = computed_offset
        if store_tokens is None:
            store_tokens = (
                (computed_offset + num_scheduled_tokens) // BLOCK_SIZE
            ) * BLOCK_SIZE
        self.store_tokens = store_tokens
        n_blocks = max(
            (computed_offset + num_scheduled_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (store_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
        )
        self.gpu_blocks = list(range(first_gpu_block, first_gpu_block + n_blocks))
        self.direction = direction

    def meta(self) -> _FakeRequestMeta:
        op_blocks = self.gpu_blocks[: self.store_tokens // BLOCK_SIZE]
        return _FakeRequestMeta(
            request_id=self.request_id,
            direction=self.direction,
            op=_FakeOp(block_ids=[op_blocks], start=0, end=self.store_tokens),
        )

    def row_slots(self) -> list[int]:
        """GPU KV slot written by each of this request's rows, in row order."""
        slots = []
        for i in range(self.num_scheduled):
            t = self.computed_offset + i
            slots.append(self.gpu_blocks[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE)
        return slots


def _make_capture(num_ring_blocks: int = 64) -> QRingBufferCapture:
    ring = QRingBuffer(
        num_layers=NUM_LAYERS,
        num_blocks=num_ring_blocks,
        block_size=BLOCK_SIZE,
        hidden_dim=HIDDEN_DIM,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    ring_adapter = SimpleNamespace(
        q_ring=ring,
        scatter_q_layer=lambda layer_index, query, ring_slots: ring.scatter(
            layer_index, query, ring_slots
        ),
    )
    worker_adapter = SimpleNamespace(is_kv_writer=True)
    return QRingBufferCapture(worker_adapter, ring_adapter)  # type: ignore[arg-type]


def _build_step(requests: list[_Request], row_order: list[int] | None = None):
    """Assemble (query, metadata, attn_metadata) for one forward step.

    Row value convention: row r of the query tensor is filled with the
    constant float(global token tag), so ring contents identify tokens.
    """
    order = row_order if row_order is not None else list(range(len(requests)))
    row_slots: list[int] = []
    row_tags: list[float] = []
    for idx in order:
        req = requests[idx]
        row_slots.extend(req.row_slots())
        for i in range(req.num_scheduled):
            row_tags.append(
                float(hash((req.request_id, req.computed_offset + i)) % 100003)
            )
    num_rows = len(row_slots)
    query = torch.empty((num_rows, HIDDEN_DIM), dtype=torch.float32)
    for r, tag in enumerate(row_tags):
        query[r].fill_(tag)
    metadata = _FakeConnectorMetadata(requests=[r.meta() for r in requests])
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor(row_slots, dtype=torch.int64)
    )
    return query, metadata, attn_metadata


def _captured_tokens(capture, state, query, req: _Request) -> torch.Tensor | None:
    """Read back the ring rows captured for ``req``, in op-token order."""
    ring = capture.q_ring_adapter.q_ring
    store = next((s for s in state.stores if s.request_id == req.request_id), None)
    if store is None:
        return None
    # Scatter layer 0 the way save_q_layer would.
    ring.scatter(0, query, state.ring_slots)
    rows = []
    for i in range(store.op.end - store.op.start):
        slot = store.ring_block_ids[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
        rows.append(ring._layer_tensors[0][slot // BLOCK_SIZE, slot % BLOCK_SIZE])
    return torch.stack(rows)


def _expected_tokens(req: _Request, query, requests, row_order=None) -> torch.Tensor:
    """The query rows that truly belong to req's op token range, in token order."""
    order = row_order if row_order is not None else list(range(len(requests)))
    base = 0
    for idx in order:
        if requests[idx] is req:
            op_rows = [
                i
                for i in range(req.num_scheduled)
                if req.computed_offset + i < req.store_tokens
            ]
            return query[base : base + req.num_scheduled][op_rows]
        base += requests[idx].num_scheduled
    raise AssertionError("request not in step")


def test_aligned_two_request_step():
    """Baseline: two aligned prefills capture every row correctly."""
    a = _Request("A", 512, first_gpu_block=0)
    b = _Request("B", 256, first_gpu_block=100)
    capture = _make_capture()
    query, md, am = _build_step([a, b])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None and len(state.stores) == 2
    for req in (a, b):
        got = _captured_tokens(capture, state, query, req)
        want = _expected_tokens(req, query, [a, b])
        assert torch.equal(got, want)


def test_unaligned_tail_does_not_shift_next_request():
    """The reported bug: A has a chunk-unaligned tail (300 rows, op covers
    [0, 288)); B's rows must still map to B's own op, not to A's tail."""
    a = _Request("A", 300, first_gpu_block=0)  # op end = 288
    b = _Request("B", 256, first_gpu_block=100)
    capture = _make_capture()
    query, md, am = _build_step([a, b])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None and len(state.stores) == 2
    got_b = _captured_tokens(capture, state, query, b)
    want_b = _expected_tokens(b, query, [a, b])
    assert torch.equal(got_b, want_b)
    # A captures only its aligned prefix [0, 288)
    got_a = _captured_tokens(capture, state, query, a)
    assert got_a.shape[0] == 288
    assert torch.equal(got_a, query[:288])


def test_batch_row_order_differs_from_metadata_order():
    """Rows laid out B-then-A while metadata lists A-then-B."""
    a = _Request("A", 256, first_gpu_block=0)
    b = _Request("B", 512, first_gpu_block=100)
    capture = _make_capture()
    query, md, am = _build_step([a, b], row_order=[1, 0])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None and len(state.stores) == 2
    got_a = _captured_tokens(capture, state, query, a)
    # A's rows are the last 256 in the batch
    assert torch.equal(got_a, query[512:768])
    got_b = _captured_tokens(capture, state, query, b)
    assert torch.equal(got_b, query[:512])


def test_mixed_store_retrieve_step_still_captures():
    """A RETRIEVE request in the step no longer kills the whole capture."""
    a = _Request("A", 256, first_gpu_block=0)
    r = _Request("R", 0, first_gpu_block=200, store_tokens=256, direction="RETRIEVE")
    capture = _make_capture()
    query, md, am = _build_step([a, r])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None
    assert [s.request_id for s in state.stores] == ["A"]
    got_a = _captured_tokens(capture, state, query, a)
    assert torch.equal(got_a, query[:256])


def test_op_tokens_from_earlier_step_skips_only_that_op():
    """A store op whose leading tokens were computed in a previous chunked
    prefill step is skipped; other requests still capture, no ring leak."""
    # A: 256-token op but only 200 rows this step (56 computed earlier).
    a = _Request("A", 200, first_gpu_block=0, store_tokens=256, computed_offset=56)
    b = _Request("B", 256, first_gpu_block=100)
    capture = _make_capture()
    ring = capture.q_ring_adapter.q_ring
    free_before = ring.num_free_blocks()
    query, md, am = _build_step([a, b])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None
    assert [s.request_id for s in state.stores] == ["B"]
    got_b = _captured_tokens(capture, state, query, b)
    assert torch.equal(got_b, query[200:456])
    used = sum(len(s.ring_block_ids) for s in state.stores)
    assert ring.num_free_blocks() == free_before - used  # nothing leaked


def test_ring_exhaustion_skips_op_without_leak():
    a = _Request("A", 512, first_gpu_block=0)
    b = _Request("B", 256, first_gpu_block=100)
    # Ring only fits B after A takes 32 blocks? Give 34 blocks: A takes 32,
    # B needs 16 -> only 2 left -> B skipped.
    capture = _make_capture(num_ring_blocks=34)
    ring = capture.q_ring_adapter.q_ring
    query, md, am = _build_step([a, b])
    state = capture._build_q_step_state(query, md, am)
    assert state is not None
    assert [s.request_id for s in state.stores] == ["A"]
    assert ring.num_free_blocks() == 34 - 32


def test_slot_mapping_padding_rows_are_dropped():
    """Padded rows (slot -1 or beyond slot_mapping) never reach the ring."""
    a = _Request("A", 256, first_gpu_block=0)
    capture = _make_capture()
    query, md, am = _build_step([a])
    # Add 8 padding rows: 4 covered by slot_mapping = -1, 4 beyond it.
    pad = torch.full((8, HIDDEN_DIM), -1.0)
    query = torch.cat([query, pad])
    am.slot_mapping = torch.cat(
        [am.slot_mapping, torch.full((4,), -1, dtype=torch.int64)]
    )
    state = capture._build_q_step_state(query, md, am)
    assert state is not None
    assert int((state.ring_slots >= 0).sum()) == 256
    got_a = _captured_tokens(capture, state, query, a)
    assert torch.equal(got_a, query[:256])


def test_missing_slot_mapping_disables_step():
    a = _Request("A", 256, first_gpu_block=0)
    capture = _make_capture()
    query, md, _ = _build_step([a])
    assert capture._build_q_step_state(query, md, None) is None
    assert capture._build_q_step_state(query, md, SimpleNamespace(foo=1)) is None


def test_no_store_ops_returns_none():
    r = _Request("R", 0, first_gpu_block=0, store_tokens=256, direction="RETRIEVE")
    capture = _make_capture()
    query, md, am = _build_step([r])
    assert capture._build_q_step_state(query, md, am) is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_randomized_batches_capture_exact_rows(seed):
    """Randomized fuzz: arbitrary mixes of aligned/unaligned/retrieve
    requests in arbitrary row order; every accepted op captures exactly its
    own rows."""
    g = torch.Generator().manual_seed(seed)

    def randint(lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=g).item())

    requests = []
    next_block = 0
    for i in range(randint(2, 7)):
        direction = "STORE" if randint(0, 4) else "RETRIEVE"
        if direction == "RETRIEVE":
            req = _Request(
                f"R{i}", 0, next_block, store_tokens=BLOCK_SIZE, direction="RETRIEVE"
            )
        else:
            n = randint(1, 40) * BLOCK_SIZE + randint(0, BLOCK_SIZE)
            req = _Request(f"S{i}", n, next_block)
        requests.append(req)
        next_block += len(req.gpu_blocks) + 1
    order = torch.randperm(len(requests), generator=g).tolist()
    capture = _make_capture(num_ring_blocks=4096)
    query, md, am = _build_step(requests, row_order=order)
    state = capture._build_q_step_state(query, md, am)
    stored_ids = set() if state is None else {s.request_id for s in state.stores}
    for req in requests:
        if req.direction != "STORE" or req.store_tokens == 0:
            assert req.request_id not in stored_ids
            continue
        assert req.request_id in stored_ids
        got = _captured_tokens(capture, state, query, req)
        # expected: this request's rows [0, store_tokens) in batch order
        base = 0
        for idx in order:
            if requests[idx] is req:
                break
            base += requests[idx].num_scheduled
        want = query[base : base + req.store_tokens]
        assert torch.equal(got, want)
