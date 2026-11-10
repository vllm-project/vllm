# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA row attention (`qsa_row_attention`, a CuTe DSL kernel) against an
FP32 reference and the stock Triton kernels at Qwen3.8-Flash-Next's TP1 shape
(24 query heads, 2 KV heads of 256 dims, 2,051 selection columns, 1,600-token
pages in a cache whose blocks hold two layers), and the numerics also at its TP4
shape (6 query heads and 1 KV head a rank, 256 rows): full, partial and empty
selections, padding rows, page and table edges, positions past a row's count,
the operand checks, CUDA graph replay, and the executor's swap of the QSA
layers' implementation (`runtime/qsa.py`), which takes the row attention in
launches of `MIN_CTAS` CTAs and more and the stock kernels below, and whose
output without stock's zero fill is bitwise stock's in padding rows and past
the step's rows (with the stock kernels, in every row)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="the QSA row attention requires SM90+",
)

if current_platform.is_cuda():
    # The model module first: it imports the QSA module (which imports it back).
    from vllm.models.qwen4_exp.nvidia import model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAImpl
    from vllm.models.qwen4_exp.nvidia.specialized_decode.qsa import (
        qsa_row_attention,
        row_attention_unsupported,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.qsa import (
        MIN_CTAS,
        DecodeQsaImpl,
        QsaRows,
    )

HEADS = 24
KV_HEADS = 2
DIM = 256
PAGE = 1600
TOPK = 2051
WIDTH = 164
LAYERS = 2  # layers sharing a cache block (block stride = LAYERS * PAGE tokens)
BLOCKS = 8
CONTEXT = 5 * PAGE + 77  # tokens of every request's context
ROWS = 128  # the TP1 step at 32 requests: 256 CTAs
TP4_ROWS = 256  # the TP4 step at 64 requests: 256 CTAs of one KV head


class Case:
    """Operands of one call in the served layout, and its FP32 reference."""

    def __init__(
        self, rows: int, seed: int, heads: int = HEADS, kv_heads: int = KV_HEADS
    ) -> None:
        self.heads, self.kv_heads = heads, kv_heads
        gen = torch.Generator(device="cuda").manual_seed(seed)
        cpu = torch.Generator().manual_seed(seed)
        storage = torch.randn(
            BLOCKS, LAYERS, PAGE, kv_heads, 2 * DIM, generator=gen, device="cuda"
        ).bfloat16()
        # The layer's view [blocks, heads, tokens, 512], as the model binds it.
        self.kv_cache = storage[:, 1].permute(0, 2, 1, 3)
        self.k, self.v = self.kv_cache.transpose(1, 2).split(DIM, dim=-1)
        self.requests = max(rows // 4, 1)
        pages = -(-CONTEXT // PAGE)
        table = torch.full((self.requests, WIDTH), -1, dtype=torch.int32)
        for r in range(self.requests):
            table[r, :pages] = torch.randperm(BLOCKS, generator=cpu)[:pages].int()
        self.table = table.cuda()
        indices = torch.full((rows, TOPK + 1), -1, dtype=torch.int32)
        t2r = torch.arange(rows, dtype=torch.int32) // 4 % self.requests
        counts = [TOPK, 2048, 1000, 37, 1, 0, 33, 64]
        for row in range(rows):
            count = counts[row % len(counts)]
            tokens = torch.randperm(CONTEXT, generator=cpu)[:count].int()
            indices[row, :count] = tokens
            indices[row, TOPK] = count
        self.indices = indices
        self.t2r = t2r
        self.q = torch.randn(rows, heads, DIM, generator=gen, device="cuda").bfloat16()
        self.gate = torch.randn(
            rows, heads * DIM, generator=gen, device="cuda"
        ).bfloat16()

    def edges(self) -> None:
        """Rows 0-6 exercise the guards (when the case has them)."""
        rows = self.indices.shape[0]
        if rows < 8:
            return
        # Page edges and the context's last token.
        self.indices[0, :8] = torch.tensor(
            [0, PAGE - 1, PAGE, 2 * PAGE - 1, 2 * PAGE, CONTEXT - 1, 3 * PAGE, 1]
        )
        # A -1 inside the counted columns, a logical page past the table, and
        # pages whose table entry is -1 or past the cache.
        self.indices[1, :4] = torch.tensor([-1, WIDTH * PAGE + 3, 5 * PAGE + 70, 9])
        self.table[self.t2r[1].item(), 5] = BLOCKS + 3
        # A padding row: request -1 and count 0; a request -1 with positions.
        self.t2r[2] = -1
        self.indices[2] = -1
        self.indices[2, TOPK] = 0
        self.t2r[3] = -1
        # A count past the selection width and a negative count.
        self.indices[4, TOPK] = 5000
        self.indices[5, TOPK] = -7
        self.indices[5, :3] = torch.tensor([3, 4, 5])
        # Valid positions past the count (33) are not attended.
        self.indices[6, 33:41] = torch.tensor(
            [0, 1, 2, PAGE, PAGE + 1, 2 * PAGE + 5, CONTEXT - 2, 7]
        )

    def operands(self) -> tuple:
        """The device operands of the call."""
        return (
            self.q,
            self.k,
            self.v,
            self.indices.cuda(),
            self.table,
            self.t2r.cuda(),
        )

    def reference(self) -> torch.Tensor:
        """FP32 softmax attention over each row's valid positions (stock's
        guards), rounded to BF16, times sigmoid(gate) in FP32, rounded."""
        rows = self.q.shape[0]
        heads_n, group = self.heads, self.heads // self.kv_heads
        out = torch.zeros(rows, heads_n, DIM, dtype=torch.bfloat16, device="cuda")
        table = self.table.cpu()
        for row in range(rows):
            request = int(self.t2r[row])
            count = min(max(int(self.indices[row, TOPK]), 0), TOPK)
            if not 0 <= request < self.requests:
                continue
            tokens = self.indices[row, :count].long()
            tokens = tokens[tokens >= 0]
            tokens = tokens[tokens // PAGE < WIDTH]
            blocks = table[request, tokens // PAGE].long()
            keep = (blocks >= 0) & (blocks < BLOCKS)
            tokens, blocks = tokens[keep], blocks[keep]
            if tokens.numel() == 0:
                continue
            offsets = (tokens % PAGE).cuda()
            blocks = blocks.cuda()
            for kv in range(self.kv_heads):
                keys = self.k[blocks, offsets, kv].float()
                values = self.v[blocks, offsets, kv].float()
                heads = slice(kv * group, (kv + 1) * group)
                scores = self.q[row, heads].float() @ keys.T * DIM**-0.5
                attended = torch.softmax(scores, dim=-1) @ values
                gate = self.gate[row].view(heads_n, DIM)[heads].float()
                out[row, heads] = (
                    attended.bfloat16().float() * torch.sigmoid(gate)
                ).bfloat16()
        return out


def _errors(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Max abs and relative Frobenius error of `out` against `ref`."""
    d = out.float() - ref.float()
    return d.abs().max().item(), (d.norm() / ref.float().norm()).item()


@pytest.mark.parametrize(
    "rows,heads,kv_heads",
    [(8, HEADS, KV_HEADS), (ROWS, HEADS, KV_HEADS), (TP4_ROWS, HEADS // 4, 1)],
)
def test_row_attention_matches_fp32_and_stock(rows, heads, kv_heads):
    case = Case(rows, seed=rows, heads=heads, kv_heads=kv_heads)
    case.edges()
    q, k, v, indices, table, t2r = case.operands()
    out = torch.full_like(q, float("nan"))
    assert not row_attention_unsupported(q, k, v, indices, table, out, case.gate)
    qsa_row_attention(q, k, v, indices, table, t2r, out, output_gate=case.gate)
    stock = torch.empty_like(q)
    qsa_sparse_paged_attention(
        q, k, v, indices, table, t2r, False, stock, output_gate=case.gate
    )
    ref = case.reference()
    assert torch.isfinite(out).all()
    row_abs, row_rel = _errors(out, ref)
    stock_abs, stock_rel = _errors(stock, ref)
    # Stock's own error against FP32 is the bound (relative error about
    # 2.5e-3, max abs about 1e-3 at these shapes).
    assert row_rel <= 1.5 * stock_rel + 1e-4, (row_rel, stock_rel)
    assert row_abs <= 2 * stock_abs + 2**-12, (row_abs, stock_abs)
    # Rows without a usable position are zero, as stock writes them.
    empty = [r for r in range(rows) if not ref[r].float().abs().sum()]
    assert empty and torch.equal(out[empty], stock[empty])
    assert not out[empty].float().abs().sum()


def test_replay_is_stable():
    """A captured launch replays bit for bit, and writes every output."""
    case = Case(ROWS, seed=3)
    q, k, v, indices, table, t2r = case.operands()
    eager = torch.empty_like(q)
    qsa_row_attention(q, k, v, indices, table, t2r, eager, output_gate=case.gate)
    out = torch.full_like(q, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_row_attention(q, k, v, indices, table, t2r, out, output_gate=case.gate)
    for _ in range(5):
        out.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, eager)


class _StockImpl:
    """The stock implementation's interface, recording its forwards."""

    head_size = DIM

    def __init__(self) -> None:
        self.forwards = 0

    def do_kv_cache_update(self, *args) -> None:
        """Unused."""

    def forward(self, layer, query, kv_cache, attn_metadata, output, **kwargs):
        """Count the call and zero the output."""
        self.forwards += 1
        return output.zero_()


def test_row_attention_only_in_large_launches():
    """`QsaRows.active` gives every layer a `DecodeQsaImpl`, which takes
    the row attention only for steps whose launch has at least `MIN_CTAS`
    CTAs (rows times the layer's KV heads: 128 rows with two KV heads at TP1,
    256 with one at TP4), and restores the layer's own on exit."""
    assert MIN_CTAS == 2 * ROWS
    stocks = [_StockImpl(), _StockImpl()]
    layers = []
    for kv_heads, stock in zip((2, 1), stocks):
        layer = torch.nn.Module()
        layer.impl = stock
        layer.num_kv_heads = kv_heads
        layers.append(layer)
    rows = QsaRows(layers)  # type: ignore[arg-type]
    for size, swapped in (
        (1, ()),
        (32, ()),
        (ROWS - 1, ()),
        (ROWS, (0,)),
        (2 * ROWS - 1, (0,)),
        (2 * ROWS, (0, 1)),
    ):
        with rows.active(size):
            for index, (layer, stock) in enumerate(zip(layers, stocks)):
                assert isinstance(layer.impl, DecodeQsaImpl)
                assert layer.impl.stock is stock
                assert layer.impl.row == (index in swapped)
        assert all(layer.impl is stock for layer, stock in zip(layers, stocks))


def test_unsupported_names_output_and_gate_layouts():
    """`row_attention_unsupported` rejects an output or gate the kernel's
    epilogue cannot store or load, so `DecodeQsaImpl` keeps the stock
    kernels for them."""
    case = Case(8, seed=7)
    q, k, v, indices, table, _ = case.operands()
    out = torch.empty_like(q)
    assert not row_attention_unsupported(q, k, v, indices, table, out, case.gate)
    wide = torch.empty(8, HEADS, DIM + 1, dtype=torch.bfloat16, device="cuda")
    flat = torch.empty(out.numel() + 1, dtype=torch.bfloat16, device="cuda")
    shifted = flat[1:].view_as(out)  # 2-byte aligned
    for bad in (out.float(), out[:4], wide[..., :DIM], shifted):
        assert "a BF16 output" in row_attention_unsupported(
            q, k, v, indices, table, bad, case.gate
        )
    for bad in (case.gate.half(), case.gate[:4], case.gate.t().contiguous().t()):
        assert "gate logits" in row_attention_unsupported(
            q, k, v, indices, table, out, bad
        )


def test_row_impl_writes_every_output():
    """`DecodeQsaImpl.forward` with the row attention runs the kernel on a
    BF16 cache without stock's zero fill of the step's rows: its output does
    not depend on what the
    buffer held (the 0.0 buffer is the former zero fill), padding rows
    (request -1, count 0 or no usable position) and the rows past the step's
    are bitwise what the stock forward leaves, and an FP8 cache goes to the
    stock implementation."""
    case = Case(ROWS, seed=5)
    case.edges()
    q, k, v, indices, table, t2r = case.operands()
    stock = _StockImpl()
    impl = DecodeQsaImpl(stock, row=True)  # type: ignore[arg-type]
    layer = SimpleNamespace(topk_indices_buffer=indices)
    rows = ROWS - 3
    metadata = SimpleNamespace(num_actual_tokens=rows, block_table=table)
    # Six more rows in the output than the step has (a padded capture).
    query = torch.cat([q, q[:3]])
    kwargs = dict(token_to_req=t2r, use_prefill_config=False, output_gate=case.gate)
    outputs = []
    for fill in (float("nan"), 0.0, -1.0):
        output = torch.full_like(query, fill)
        impl.forward(layer, query, case.kv_cache, metadata, output, **kwargs)
        outputs.append(output.view(torch.int16))
    assert stock.forwards == 0
    # Bitwise comparisons: int16 views (NaN != NaN, -0.0 == 0.0 as values).
    assert all(torch.equal(outputs[0], other) for other in outputs[1:])
    expected = torch.empty_like(q[:rows])
    qsa_row_attention(
        q[:rows],
        k,
        v,
        indices[:rows],
        table,
        t2r[:rows],
        expected,
        output_gate=case.gate[:rows],
    )
    assert torch.equal(outputs[0][:rows], expected.view(torch.int16))
    # The stock forward (zero fill, then the Triton kernels) on the same step.
    reference = torch.full_like(query, float("nan"))
    Qwen4ExpQSAImpl.forward(
        SimpleNamespace(head_size=DIM, kv_cache_dtype="auto"),  # type: ignore[arg-type]
        layer,
        query,
        case.kv_cache,
        metadata,
        reference,
        **kwargs,
    )
    reference = reference.view(torch.int16)
    ref = case.reference()
    padding = [
        row
        for row in range(rows)
        if int(t2r[row]) < 0 or not ref[row].float().abs().sum()
    ]
    assert 2 in padding and 3 in padding
    for part in (padding, slice(rows, None)):
        assert torch.equal(outputs[0][part], reference[part])
        assert not outputs[0][part].any()  # +0.0 everywhere
    fp8 = case.kv_cache.view(torch.uint8)[..., : 2 * DIM]
    impl.forward(layer, query, fp8, metadata, output, **kwargs)
    assert stock.forwards == 1


@pytest.mark.parametrize("rows", [8, 64, 300])
def test_stock_kernels_without_the_fill(rows):
    """`DecodeQsaImpl.forward` below `MIN_CTAS` runs the stock kernels
    without stock's zero fill of the step's rows, yet whatever the buffer held
    (NaN, -1 or the former zero fill) its output is bitwise the stock
    forward's (the fill, then the kernels) in every row: padding rows (request
    -1, count 0 or no usable position) and the rows past the step's included.
    On GB300 8 and 64 rows take the split-K kernel and its merge and 300 rows
    one split; other GPUs may split all three (stock's split choice)."""
    case = Case(rows, seed=11)
    case.edges()
    q, k, v, indices, table, t2r = case.operands()
    stock = _StockImpl()
    impl = DecodeQsaImpl(stock, row=False)  # type: ignore[arg-type]
    layer = SimpleNamespace(topk_indices_buffer=indices)
    step = rows - 3
    metadata = SimpleNamespace(num_actual_tokens=step, block_table=table)
    # Six more rows in the output than the step has (a padded capture).
    query = torch.cat([q, q[:3]])
    kwargs = dict(token_to_req=t2r, use_prefill_config=False, output_gate=case.gate)
    reference = torch.full_like(query, float("nan"))
    Qwen4ExpQSAImpl.forward(
        SimpleNamespace(head_size=DIM, kv_cache_dtype="auto"),  # type: ignore[arg-type]
        layer,
        query,
        case.kv_cache,
        metadata,
        reference,
        **kwargs,
    )
    # Bitwise comparisons: int16 views (NaN != NaN, -0.0 == 0.0 as values).
    reference = reference.view(torch.int16)
    for fill in (float("nan"), -1.0, 0.0):
        output = torch.full_like(query, fill)
        impl.forward(layer, query, case.kv_cache, metadata, output, **kwargs)
        assert torch.equal(output.view(torch.int16), reference)
    assert stock.forwards == 0
    assert not reference[2:4].any() and not reference[step:].any()
