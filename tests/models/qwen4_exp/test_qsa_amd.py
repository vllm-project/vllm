# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.amd import indexer_qsa
from vllm.models.qwen4_exp.amd import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd.indexer_qsa import (
    QSAIndexer,
    apply_qsa_rmsnorm,
    apply_qsa_rope,
)
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.amd.ops.qsa_prepare import (
    qsa_prepare,
    supports_fused_pre_indexer,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD QSA requires ROCm",
)

requires_qsa_kernels = pytest.mark.skipif(
    not HAS_TRITON,
    reason="AMD QSA kernels require Triton",
)


def test_ple_ngram_embedding_custom_op_uses_resident_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.ple"
    layer = ple_layer_module.Qwen4ExpPLELayer.__new__(ple_layer_module.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = torch.nn.Module()
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(8, 3)
    context = SimpleNamespace(no_compile_layers={layer_name: layer})
    monkeypatch.setattr(ple_layer_module, "get_forward_context", lambda: context)

    ngram_ids = torch.tensor([[0, 1], [2, 3]])
    output = torch.empty(2, 6)
    ple_layer_module.qwen4_exp_amd_ple_ngram_embedding(
        ngram_ids,
        output,
        layer_name,
    )

    expected = layer.ple_embedding.ngram_embedding(ngram_ids).flatten(-2)
    torch.testing.assert_close(output, expected)


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values.float()).to(
            q.dtype
        )
    return output


def test_qsa_rope_uses_platform_dispatch() -> None:
    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    positions = torch.tensor([0, 1])
    calls = []

    def apply_rotary_emb(
        rotary_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((rotary_input, cos, sin))
        return rotary_input + 1

    rotary_emb = SimpleNamespace(
        rotary_dim=2,
        apply_rotary_emb=apply_rotary_emb,
        _match_cos_sin_cache_dtype=lambda _: torch.zeros(2, 4),
    )

    output = apply_qsa_rope(rotary_emb, positions, tensor)

    assert len(calls) == 1
    torch.testing.assert_close(output[..., :2], tensor[..., :2] + 1)
    torch.testing.assert_close(output[..., 2:], tensor[..., 2:])


def test_qsa_rmsnorm_uses_portable_implementation(default_vllm_config) -> None:
    norm = GemmaRMSNorm(4, eps=1e-6)
    norm.weight.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    output = apply_qsa_rmsnorm(norm, tensor)

    torch.testing.assert_close(output, norm.forward_native(tensor))


def _make_attn(*, use_fused: bool) -> SimpleNamespace:
    """The main-attention fields the fused QSA prepare reads from its owner."""
    return SimpleNamespace(
        use_fused_qsa_prepare=use_fused,
        kv_cache=torch.zeros(1, 2, 16, 8),
        q_norm=GemmaRMSNorm(4, eps=1e-6),
        k_norm=GemmaRMSNorm(4, eps=1e-6),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )


def _make_indexer(*, use_fused: bool, num_tokens: int = 2) -> QSAIndexer:
    """A QSAIndexer whose unfused path runs for real, with no projection weights.

    Both branches have to be reachable: a fused-path test is only meaningful if
    the unfused one it replaces would otherwise have executed.
    """
    heads, head_dim = 2, 4
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads = heads
    indexer.index_kv_heads = 1
    indexer.index_head_dim = head_dim
    indexer.compress_ratio = 2
    indexer.use_fused_pre_indexer = use_fused
    indexer.skip_topk = True
    indexer.index_qk_proj = lambda hidden_states: (
        torch.zeros(hidden_states.shape[0], (heads + 1) * head_dim),
        None,
    )
    indexer.q_layernorm = GemmaRMSNorm(head_dim, eps=1e-6)
    indexer.k_layernorm = GemmaRMSNorm(head_dim, eps=1e-6)
    indexer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.zeros(16, head_dim),
        rotary_dim=head_dim // 2,
        apply_rotary_emb=lambda tensor, cos, sin: tensor,
        _match_cos_sin_cache_dtype=lambda _: torch.zeros(16, head_dim),
    )
    # The raw cache's full width carries the MRoPE position tail that the fused
    # kernel writes; key_cache is only the leading key columns.
    indexer.raw_key_cache = SimpleNamespace(
        kv_cache=torch.zeros(1, 4, 1, head_dim + 12),
        key_cache=torch.zeros(1, 4, 1, head_dim),
        cache_rope_positions=True,
        rope_position_offset=head_dim,
    )
    indexer.compressed_key_cache = SimpleNamespace(
        kv_cache=torch.zeros(1, 2, 1, head_dim)
    )
    raw_metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        slot_mapping=torch.arange(num_tokens),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        logical_positions=torch.arange(num_tokens),
        token_to_req=torch.zeros(num_tokens, dtype=torch.int32),
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        slot_mapping=torch.arange(num_tokens),
        k_work_metadata=torch.zeros(1, 2, dtype=torch.int32),
    )
    indexer._metadata = lambda: (raw_metadata, compressed_metadata)
    indexer.test_metadata = (raw_metadata, compressed_metadata)
    return indexer


@pytest.mark.usefixtures("default_vllm_config")
def test_qsa_fused_prepare_replaces_the_unfused_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = _make_indexer(use_fused=True)
    attn = _make_attn(use_fused=True)
    calls = []
    main_outputs = (torch.empty(0), torch.empty(0))

    def fake_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return main_outputs

    monkeypatch.setattr(indexer_qsa, "qsa_prepare", fake_prepare)
    monkeypatch.setattr(
        QSAIndexer,
        "_update_and_compress",
        lambda *args, **kwargs: pytest.fail("fused path ran the unfused chain"),
    )
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)
    qkv = torch.zeros(4, 16)
    slot_mapping = torch.arange(4)
    gate_out = torch.empty(4, 1, 4)

    actual, actual_main = QSAIndexer.forward(
        indexer,
        torch.zeros(2, 8),
        torch.tensor([7, 8]),
        rows,
        attn=attn,
        qkv=qkv,
        slot_mapping=slot_mapping,
        gate_out=gate_out,
    )

    assert actual is rows
    assert actual_main is main_outputs
    assert len(calls) == 1
    args, kwargs = calls[0]
    passed = inspect.signature(qsa_prepare).bind(*args, **kwargs).arguments
    _, compressed_metadata = indexer.test_metadata
    # The kernel addresses the raw cache at its full width and writes the MRoPE
    # position tail itself, so it needs kv_cache, not the key_cache view.
    assert passed["state_cache"] is indexer.raw_key_cache.kv_cache
    assert passed["compressed_cache"] is indexer.compressed_key_cache.kv_cache
    assert passed["k_work_metadata"] is compressed_metadata.k_work_metadata
    assert passed["compress_ratio"] == indexer.compress_ratio
    assert passed["rope_pos_offset"] == indexer.index_head_dim
    # Main attention: only the real tokens, the vLLM cache viewed per token,
    # and the gate landing in the caller's buffer.
    assert passed["main_qkv"].shape[0] == 2
    assert passed["main_qkv"].data_ptr() == qkv.data_ptr()
    assert passed["main_slot_mapping"].shape == (2,)
    assert passed["main_kv_cache"].shape == (1, 16, 2, 8)
    assert passed["main_kv_cache"].data_ptr() == attn.kv_cache.data_ptr()
    assert passed["main_gate_out"].shape[0] == 2
    assert passed["main_gate_out"].data_ptr() == gate_out.data_ptr()
    assert passed["main_q_norm_weight"] is attn.q_norm.weight
    assert passed["main_k_norm_weight"] is attn.k_norm.weight


@pytest.mark.usefixtures("default_vllm_config")
def test_qsa_unsupported_config_keeps_the_unfused_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = _make_indexer(use_fused=False)
    monkeypatch.setattr(
        indexer_qsa,
        "qsa_prepare",
        lambda *args, **kwargs: pytest.fail("unfused path ran the fused kernel"),
    )
    updates = []
    monkeypatch.setattr(
        QSAIndexer,
        "_update_and_compress",
        lambda *args, **kwargs: updates.append(args),
    )
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)

    actual, main_outputs = QSAIndexer.forward(
        indexer,
        torch.zeros(2, 8),
        torch.tensor([7, 8]),
        rows,
        attn=_make_attn(use_fused=False),
    )

    assert actual is rows
    assert main_outputs is None
    assert len(updates) == 1


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, True, id="qwen3_8_flash_next"),
        pytest.param({"is_neox_style": False}, False, id="not-neox"),
        pytest.param({"rotary_dim": 128}, False, id="rotary-dim"),
        pytest.param({"mrope_interleaved": False}, False, id="mrope-not-interleaved"),
        pytest.param({"mrope_section": (11, 11, 10, 0)}, False, id="mrope-section-len"),
    ],
)
def test_fused_pre_indexer_gate_is_config_shaped(overrides, expected) -> None:
    # Qwen3.8-Flash-Next's indexer geometry, which both vendors satisfy.
    rotary = SimpleNamespace(
        rotary_dim=64,
        is_neox_style=True,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
    )
    for name, value in overrides.items():
        setattr(rotary, name, value)

    assert (
        supports_fused_pre_indexer(
            rotary, head_dim=128, num_kv_heads=1, compress_ratio=4
        )
        is expected
    )


def test_qsa_selection_uses_portable_topk_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 2
    token_topk = 8
    compress_ratio = 4
    block_topk = token_topk // compress_ratio
    blocks = torch.empty((rows, block_topk), dtype=torch.int32)
    visible_blocks = torch.tensor([4, 6], dtype=torch.int32)
    logits = torch.empty((rows, 8), dtype=torch.float32)
    selection_output = torch.empty(
        (rows, token_topk + compress_ratio - 1), dtype=torch.int32
    )
    topk_call: dict[str, Any] = {}

    monkeypatch.setattr(
        qsa_ops,
        "qsa_mqa_paged",
        lambda *args, **kwargs: (logits, visible_blocks),
    )
    monkeypatch.setattr(qsa_ops.current_platform, "is_cuda", lambda: False)

    def top_k_per_row_decode(
        topk_logits,
        next_n,
        seq_lens,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    ) -> None:
        topk_call.update(
            logits=topk_logits,
            next_n=next_n,
            seq_lens=seq_lens,
            raw_topk_indices=raw_topk_indices,
            num_rows=num_rows,
            strides=(stride0, stride1),
            topk_tokens=topk_tokens,
        )
        raw_topk_indices.zero_()

    def expand_qsa_block_indices(*args) -> None:
        args[-1].fill_(-1)

    monkeypatch.setattr(qsa_ops.ops, "top_k_per_row_decode", top_k_per_row_decode)
    monkeypatch.setattr(
        qsa_ops,
        "expand_qsa_block_indices_cuda",
        expand_qsa_block_indices,
    )

    output = qsa_ops.qsa_select_paged_tokens(
        torch.empty((rows, 1, 1)),
        torch.empty((1, 4, 1, 1)),
        torch.empty((1, 2), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows, dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        token_topk,
        compress_ratio,
        selection_output,
    )

    assert output is selection_output
    assert torch.all(output == -1)
    assert topk_call["logits"] is logits
    assert topk_call["next_n"] == 1
    assert topk_call["seq_lens"] is visible_blocks
    assert topk_call["raw_topk_indices"].shape == blocks.shape
    assert topk_call["num_rows"] == rows
    assert topk_call["strides"] == (logits.stride(0), logits.stride(1))
    assert topk_call["topk_tokens"] == block_topk


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_mqa_paged_matches_reference_on_visible_columns(
    dtype: torch.dtype,
) -> None:
    # A prefill chunk, a decode request, and rows with nothing visible, so one
    # row block spans three requests. Top-k reads only [0, visible_blocks).
    torch.manual_seed(0)
    page_size, compress_ratio, table_width, num_pages = 16, 4, 24, 80
    positions = [range(1160, 1203), range(159, 162), range(3)]
    lengths = torch.tensor([1203, 162, 3], device="cuda", dtype=torch.int32)
    token_to_req = torch.tensor(
        [r for r, p in enumerate(positions) for _ in p],
        device="cuda",
        dtype=torch.int32,
    )
    query_positions = torch.tensor(
        [i for p in positions for i in p], device="cuda", dtype=torch.int64
    )
    q = torch.randn(token_to_req.numel(), 2, 32, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_pages, page_size, 1, 32, device="cuda", dtype=dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
        : 3 * table_width
    ].view(3, table_width)

    logits, visible_blocks = qsa_ops.qsa_mqa_paged(
        q, k_cache, page_table, token_to_req, query_positions, lengths, compress_ratio
    )

    expected_visible = torch.minimum(
        (query_positions + 1) // compress_ratio,
        lengths[token_to_req.long()] // compress_ratio,
    ).to(torch.int32)
    torch.testing.assert_close(visible_blocks, expected_visible, rtol=0, atol=0)
    columns = torch.arange(logits.shape[1], device="cuda")
    logits = logits.masked_fill(columns >= expected_visible[:, None], -torch.inf)
    expected = _qsa_mqa_paged_reference(
        q, k_cache, page_table, token_to_req, expected_visible
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("num_requests", [1, 5, 32, 33])
def test_qsa_mqa_paged_decode_rows_match_reference(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, num_requests: int
) -> None:
    # One decode row per request, with lengths that end mid-page. Up to 32 rows
    # are scored row by row on gfx942, 33 use the tiled kernel.
    spy = _force_rows_kernel(monkeypatch)
    torch.manual_seed(num_requests)
    page_size, compress_ratio, table_width = 16, 4, 24
    num_pages = num_requests * table_width + 8
    lengths = torch.randint(1, 1500, (num_requests,), device="cuda", dtype=torch.int32)
    token_to_req = torch.arange(num_requests, device="cuda", dtype=torch.int32)
    query_positions = (lengths - 1).to(torch.int64)
    q = torch.randn(num_requests, 2, 32, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_pages, page_size, 1, 32, device="cuda", dtype=dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
        : num_requests * table_width
    ].view(num_requests, table_width)

    logits, visible_blocks = qsa_ops.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        lengths,
        compress_ratio,
        single_row_decode=True,
    )

    assert spy.launches == (num_requests <= 32)
    expected_visible = (lengths // compress_ratio).to(torch.int32)
    torch.testing.assert_close(visible_blocks, expected_visible, rtol=0, atol=0)
    columns = torch.arange(logits.shape[1], device="cuda")
    logits = logits.masked_fill(columns >= expected_visible[:, None], -torch.inf)
    expected = _qsa_mqa_paged_reference(
        q, k_cache, page_table, token_to_req, expected_visible
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)


class _KernelSpy:
    """Counts launches of a Triton kernel."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


def _force_rows_kernel(monkeypatch: pytest.MonkeyPatch) -> _KernelSpy:
    # The per-row path is gfx942 only, but the kernel runs on any ROCm arch.
    monkeypatch.setattr(qsa_ops, "_on_gfx942", lambda: True)
    spy = _KernelSpy(qsa_ops._qsa_mqa_paged_rows_kernel)
    monkeypatch.setattr(qsa_ops, "_qsa_mqa_paged_rows_kernel", spy)
    return spy


def _decode_select_case(num_requests: int, dtype: torch.dtype):
    torch.manual_seed(num_requests)
    page_size, compress_ratio, table_width = 16, 4, 24
    num_pages = num_requests * table_width + 8
    lengths = torch.randint(8, 1500, (num_requests,), device="cuda", dtype=torch.int32)
    return (
        torch.randn(num_requests, 2, 32, device="cuda", dtype=dtype),
        torch.randn(num_pages, page_size, 1, 32, device="cuda", dtype=dtype),
        torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
            : num_requests * table_width
        ].view(num_requests, table_width),
        torch.arange(num_requests, device="cuda", dtype=torch.int32),
        (lengths - 1).to(torch.int64),
        lengths,
        compress_ratio,
    )


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_qsa_select_decode_rows_ignores_unwritten_logits(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    # Neither kernel writes logits past visible_blocks, so top-k must not read
    # them: fill every fresh logits buffer with NaN and compare both paths.
    spy = _force_rows_kernel(monkeypatch)
    *case, compress_ratio = _decode_select_case(7, dtype)
    token_topk = 16
    expected = qsa_ops.qsa_select_paged_tokens(
        *case, token_topk, compress_ratio, single_row_decode=False
    )

    real_empty = torch.empty

    def empty_nan(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        if out.dtype == torch.float32 and out.dim() == 2 and out.is_cuda:
            out.fill_(float("nan"))
        return out

    monkeypatch.setattr(torch, "empty", empty_nan)
    spy.launches = 0
    actual = qsa_ops.qsa_select_paged_tokens(
        *case, token_topk, compress_ratio, single_row_decode=True
    )
    assert spy.launches == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_qsa_kernels
def test_qsa_select_decode_rows_replays_in_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _force_rows_kernel(monkeypatch)
    *case, compress_ratio = _decode_select_case(8, torch.bfloat16)
    q, k_cache, page_table, token_to_req, positions, lengths = case
    token_topk = 16

    def select():
        return qsa_ops.qsa_select_paged_tokens(
            q,
            k_cache,
            page_table,
            token_to_req,
            positions,
            lengths,
            token_topk,
            compress_ratio,
            single_row_decode=True,
        )

    select()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select()

    # Replay on new inputs written in place, as the runner does between steps.
    torch.manual_seed(1)
    q.copy_(torch.randn_like(q))
    new_lengths = torch.randint(
        8, 1500, lengths.shape, device="cuda", dtype=lengths.dtype
    )
    lengths.copy_(new_lengths)
    positions.copy_((new_lengths - 1).to(torch.int64))
    graph.replay()
    torch.cuda.synchronize()
    expected = qsa_ops.qsa_select_paged_tokens(
        q,
        k_cache,
        page_table,
        token_to_req,
        positions,
        lengths,
        token_topk,
        compress_ratio,
        single_row_decode=False,
    )
    # Warmup and capture launched it; the replay does not go through Python.
    assert spy.launches == 2
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_prefills", "decode_query_len", "expected"),
    [(0, 1, True), (1, 1, False), (0, 4, False), (0, 0, False)],
    ids=["decode", "mixed", "spec-decode", "empty"],
)
def test_qsa_indexer_flags_single_row_decode(
    monkeypatch: pytest.MonkeyPatch,
    num_prefills: int,
    decode_query_len: int,
    expected: bool,
) -> None:
    seen: dict[str, Any] = {}

    def select(*args, **kwargs):
        seen.update(kwargs)
        return args[-1]

    monkeypatch.setattr(qsa_ops, "qsa_select_paged_tokens", select)
    indexer = SimpleNamespace(
        compressed_key_cache=SimpleNamespace(kv_cache=None),
        token_topk=8,
        compress_ratio=4,
    )
    metadata = SimpleNamespace(
        block_table=None,
        token_to_req=None,
        logical_positions=None,
        seq_lens=None,
        num_prefills=num_prefills,
        decode_query_len=decode_query_len,
    )
    QSAIndexer._select(indexer, None, metadata, None)  # type: ignore[arg-type]
    assert seen["single_row_decode"] is expected


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("num_rows", "num_query_heads", "num_kv_heads", "page_size"),
    [
        pytest.param(1, 24, 2, 1792, id="tp1_split64"),
        pytest.param(16, 12, 1, 1792, id="tp2_split32"),
        pytest.param(32, 6, 1, 1024, id="tp4_split8"),
        pytest.param(257, 6, 1, 1024, id="tp4_split4"),
        pytest.param(513, 6, 1, 1024, id="tp4_split1"),
    ],
)
def test_qsa_sparse_paged_attention_matches_reference(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> None:
    torch.manual_seed(2)
    head_dim = 256
    num_requests = 2
    num_selected_pages = 64
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    request_row_counts = torch.tensor(
        [rows_per_request, num_rows - rows_per_request],
        device="cuda",
        dtype=torch.int32,
    )

    context_length = num_pages_per_request * page_size - 1
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_length - request_row_counts[token_to_req.long()] + rows_within_request
    ).to(torch.int64)
    sequence_lengths = torch.full(
        (num_requests,), context_length, device="cuda", dtype=torch.int32
    )
    logical_indices = qsa_ops.expand_qsa_block_indices_cuda(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer_compress_ratio,
        indexer_budget,
    )
    assert logical_indices.shape == (num_rows, selection_width)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
