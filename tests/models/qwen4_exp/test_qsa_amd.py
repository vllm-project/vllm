# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import itertools
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.warmup import qwen4_exp_qsa_warmup
from vllm.models.qwen4_exp.amd import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.amd import ple_layer as ple_layer_module
from vllm.models.qwen4_exp.amd import qsa as qsa_owner
from vllm.models.qwen4_exp.amd import qsa_indexer
from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.amd.ops import qsa_flydsl_prefill as fly
from vllm.models.qwen4_exp.amd.ops.qsa_prepare import (
    qsa_prepare,
    supports_fused_pre_indexer,
)
from vllm.models.qwen4_exp.amd.qsa_indexer import (
    QSAIndexer,
    apply_qsa_rmsnorm,
    apply_qsa_rope,
)
from vllm.models.qwen4_exp.common.qsa import QSAPrefillLayout, host_prefill_layout
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import CommonAttentionMetadata

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
        kv_cache_dtype="auto",
        q_norm=GemmaRMSNorm(4, eps=1e-6),
        k_norm=GemmaRMSNorm(4, eps=1e-6),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )


def _make_indexer(
    monkeypatch: pytest.MonkeyPatch, *, use_fused: bool, num_tokens: int = 2
) -> QSAIndexer:
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
    indexer.indexer_dtype = torch.bfloat16
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
    monkeypatch.setattr(
        indexer, "_metadata", lambda: (raw_metadata, compressed_metadata)
    )
    indexer.test_metadata = (raw_metadata, compressed_metadata)
    return indexer


@pytest.mark.usefixtures("default_vllm_config")
def test_qsa_fused_prepare_replaces_the_unfused_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexer = _make_indexer(monkeypatch, use_fused=True)
    attn = _make_attn(use_fused=True)
    calls = []
    main_outputs = (torch.empty(0), torch.empty(0))

    def fake_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return main_outputs

    monkeypatch.setattr(qsa_indexer, "qsa_prepare", fake_prepare)
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
    indexer = _make_indexer(monkeypatch, use_fused=False)
    monkeypatch.setattr(
        qsa_indexer,
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


@requires_qsa_kernels
@pytest.mark.parametrize(
    "num_rows,table_width",
    [
        pytest.param(1, 64, id="one_row"),
        pytest.param(32, 64, id="decode32"),
        pytest.param(128, 168, id="decode128_wide_table"),
        pytest.param(1024, 8, id="prefill_chunk"),
    ],
)
def test_qsa_mqa_paged_scores_visible_columns(num_rows: int, table_width: int) -> None:
    """The scores of each row's visible columns (the ones the top-k reads)
    match a reference whatever the page table's width; a padding row and
    unmapped pages are handled."""
    torch.manual_seed(3)
    heads, head_dim, page_size, compress_ratio = 4, 128, 64, 4
    num_requests = 3
    pages = num_requests * table_width + 1
    k_cache = torch.randn(
        pages, page_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    page_table = (
        torch.randperm(pages - 1, device="cuda")[: num_requests * table_width] + 1
    ).to(torch.int32)
    page_table = page_table.view(num_requests, table_width).contiguous()
    page_table[2, table_width // 2 :] = -1  # a request whose late pages are unmapped
    q = torch.randn(num_rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    token_to_req = torch.arange(num_rows, device="cuda", dtype=torch.int32) % 3
    token_to_req[-1] = -1  # a padding row
    capacity = table_width * page_size * compress_ratio
    sequence_lengths = torch.tensor(
        [capacity, capacity // 3, capacity], device="cuda", dtype=torch.int32
    )
    query_positions = torch.randint(
        0, capacity, (num_rows,), device="cuda", dtype=torch.int32
    )

    logits, visible = qsa_ops.qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )

    request = token_to_req.clamp_min(0).long()
    expected_visible = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths[request] // compress_ratio,
    )
    expected_visible = torch.where(token_to_req >= 0, expected_visible, 0)
    torch.testing.assert_close(visible, expected_visible.to(torch.int32))
    table = page_table[request].long()
    keys = k_cache[table.clamp_min(0), :, 0, :].flatten(1, 2).float()
    mapped = (table >= 0).repeat_interleave(page_size, dim=1)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys).relu().sum(-1)
    reference = torch.where(mapped, scores / math.sqrt(head_dim), -torch.inf)
    for row in range(num_rows):
        n = int(visible[row])
        torch.testing.assert_close(
            logits[row, :n], reference[row, :n], rtol=1e-4, atol=1e-4
        )


# Qwen3.8-Flash-Next's indexer as served: 4 heads x 128, compressed pages of
# 1568 // 4 = 392 keys, a block top-k of 2048 // 4 = 512.
_INDEXER_HEADS, _INDEXER_DIM, _COMPRESSED_PAGE, _RATIO = 4, 128, 392, 4
_TOKEN_TOPK = 2048


def _served_scoring_inputs(
    requests: list[tuple[int, int]], table_width: int, device: str = "cuda"
) -> SimpleNamespace:
    """Scorer inputs in a served step's layout. ``requests`` lists (rows,
    sequence length) in batch order, and a request's rows are its last
    positions. Each request maps ``table_width`` shuffled pages except one
    (-1), the third, fourth or fifth by request; a padding row (request -1,
    position -1) ends the batch."""
    generator = torch.Generator(device=device).manual_seed(len(requests))
    num_pages = len(requests) * table_width + 1
    pages = torch.randperm(num_pages, generator=generator, device=device)
    page_table = pages[: len(requests) * table_width].view(-1, table_width)
    page_table = page_table.to(torch.int32)
    request_ids = torch.arange(len(requests), device=device)
    page_table[request_ids, 2 + request_ids % 3] = -1
    token_to_req = torch.cat(
        [torch.full((rows,), r) for r, (rows, _) in enumerate(requests)]
        + [torch.tensor([-1])]
    )
    positions = torch.cat(
        [torch.arange(length - rows, length) for rows, length in requests]
        + [torch.tensor([-1])]
    )

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device).to(
            torch.bfloat16
        )

    return SimpleNamespace(
        q=randn(token_to_req.numel(), _INDEXER_HEADS, _INDEXER_DIM),
        k_cache=randn(num_pages, _COMPRESSED_PAGE, 1, _INDEXER_DIM),
        page_table=page_table,
        token_to_req=token_to_req.to(device=device, dtype=torch.int32),
        positions=positions.to(device),
        seq_lens=torch.tensor(
            [length for _, length in requests], dtype=torch.int32, device=device
        ),
    )


def _served_scores_fp64(
    inputs: SimpleNamespace, columns: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``visible`` per row from its definition, the fp64 scores [rows,
    columns] (-inf at unmapped pages, NaN past ``visible``), and the error
    bound of an fp32 kernel.

    A kernel that forms the bf16 products exactly in fp32 and sums each
    head's dot product in any order is within gamma(head_dim - 1) *
    sum_d |q k| of the exact dot (Higham, Accuracy and Stability, 3.1);
    the ReLU is exact, and summing the heads and dividing by an fp32
    sqrt(head_dim) add four roundings. Hence |score - fp64| <=
    (head_dim + 8) * 2**-24 * sum_h sum_d |q k| / sqrt(head_dim), whatever
    the summation order: the bound holds for the per-row kernel and the MFMA
    kernel alike, and an error from a wrong key, row, head or page exceeds it
    by orders of magnitude."""
    request = inputs.token_to_req.long()
    valid_request = request >= 0
    safe_request = request.clamp_min(0)
    visible = torch.minimum(
        (inputs.positions + 1) // _RATIO,
        torch.where(valid_request, inputs.seq_lens[safe_request], 0) // _RATIO,
    )
    rows = request.numel()
    scores = torch.full(
        (rows, columns), torch.nan, dtype=torch.float64, device=request.device
    )
    bound = torch.zeros_like(scores)
    column = torch.arange(columns, device=request.device)
    page = inputs.page_table.long()[:, column // _COMPRESSED_PAGE]
    for r in request[valid_request].unique().tolist():
        selected = (request == r).nonzero().flatten()
        keys = inputs.k_cache[
            page[r].clamp_min(0), column % _COMPRESSED_PAGE, 0
        ].double()
        q = inputs.q[selected].double()
        row_scores = sum((q[:, h] @ keys.T).relu() for h in range(q.shape[1]))
        row_scores /= math.sqrt(_INDEXER_DIM)
        magnitude = q.abs().sum(1) @ keys.abs().T
        row_scores[:, page[r] < 0] = -torch.inf
        seen = column[None, :] < visible[selected, None]
        scores[selected] = torch.where(seen, row_scores, torch.nan)
        bound[selected] = (
            (_INDEXER_DIM + 8) * 2.0**-24 * magnitude / math.sqrt(_INDEXER_DIM)
        )
    return visible.to(torch.int32), scores, bound


@requires_qsa_kernels
@pytest.mark.parametrize(
    "requests,table_width",
    [
        # the smallest call the MFMA kernel takes
        pytest.param([(767, 8192)], 6, id="gate768"),
        # a 128K prompt's last chunk in the sub-chunks a 128 MiB logits
        # workspace holds at 32832 columns; visible up to 32800 blocks
        pytest.param([(1021, 131200)], 84, id="chunk1022_128k"),
        # two prompts in one 16384-row step; a tile straddles them
        pytest.param([(8100, 8100), (8283, 8283)], 6, id="chunk16384"),
        # decode rows in the same call: a tile spans many requests
        pytest.param(
            [(1, 9000 - i) for i in range(40)] + [(800, 9000)], 6, id="decode40"
        ),
    ],
)
def test_qsa_mfma_scorer_matches_fp64(
    requests: list[tuple[int, int]],
    table_width: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefill calls of ``qsa_mqa_paged`` take the MFMA kernel, and its scores
    and the per-row kernel's stay within the fp32 rounding bound of fp64 on
    every visible column. The top-512 block sets the MFMA scores select equal
    the fp64 sets on >= 99.9 % of the blocks, and every block they differ in
    scores within twice the row's bound of the 512th fp64 score (a tie that
    rounding may break either way)."""
    inputs = _served_scoring_inputs(requests, table_width)
    max_seq_len = max(length for _, length in requests)
    columns = min(
        round_up(cdiv(max_seq_len, _RATIO), 64), table_width * _COMPRESSED_PAGE
    )
    visible, reference, bound = _served_scores_fp64(inputs, columns)
    launches: list[int] = []
    launch = qsa_ops.launch_qsa_mqa_paged_mfma

    def recorded(q: torch.Tensor, *args: Any) -> None:
        launches.append(q.shape[0])
        launch(q, *args)

    monkeypatch.setattr(qsa_ops, "launch_qsa_mqa_paged_mfma", recorded)
    seen = ~reference.isnan()
    finite = reference.isfinite()
    scored = {}
    for allow_mfma in (False, True):
        logits, kernel_visible = qsa_ops.qsa_mqa_paged(
            inputs.q,
            inputs.k_cache,
            inputs.page_table,
            inputs.token_to_req,
            inputs.positions,
            inputs.seq_lens,
            _RATIO,
            num_columns=columns,
            allow_mfma=allow_mfma,
        )
        torch.testing.assert_close(kernel_visible, visible, rtol=0, atol=0)
        logits = logits.double().where(seen, torch.nan)
        assert torch.equal(logits.isneginf(), reference.isneginf())
        error = (logits - reference).abs()[finite]
        assert bool((error <= bound[finite]).all()), (allow_mfma, error.max())
        scored[allow_mfma] = logits
    assert launches == [inputs.q.shape[0]]

    block_topk = _TOKEN_TOPK // _RATIO
    full = visible > block_topk
    reference = reference[full].masked_fill(~seen[full], -torch.inf)
    mfma = scored[True][full].masked_fill(~seen[full], -torch.inf)
    reference_sets = torch.zeros_like(reference, dtype=torch.bool).scatter_(
        1, reference.topk(block_topk).indices, True
    )
    mfma_sets = torch.zeros_like(reference_sets).scatter_(
        1, mfma.topk(block_topk).indices, True
    )
    assert int(full.sum()) > 0
    agreement = (reference_sets & mfma_sets).sum() / (full.sum() * block_topk)
    assert agreement >= 0.999, float(agreement)
    boundary = reference.topk(block_topk).values[:, -1:]
    band = 2 * bound[full].amax(dim=1, keepdim=True)
    differing = reference_sets ^ mfma_sets
    assert not bool((differing & ((reference - boundary).abs() > band)).any())


def test_mfma_scorer_gate() -> None:
    """Only bf16 or fp16 calls of at least 768 rows, cache and query alike."""

    def applies(rows: int, q_dtype: torch.dtype, k_dtype: torch.dtype) -> bool:
        q = torch.empty((rows, 4, 128), dtype=q_dtype, device="meta")
        k_cache = torch.empty((1, 392, 1, 128), dtype=k_dtype, device="meta")
        return qsa_ops.mfma_scorer_applies(q, k_cache)

    bf16, fp16, fp8 = torch.bfloat16, torch.float16, current_platform.fp8_dtype()
    assert applies(768, bf16, bf16) and applies(16384, fp16, fp16)
    assert not applies(767, bf16, bf16)
    assert not applies(768, fp8, fp8) and not applies(768, bf16, fp16)


@pytest.mark.parametrize("capturing", [False, True])
def test_qsa_selection_trims_width_and_splits_decode_rows(
    monkeypatch: pytest.MonkeyPatch, capturing: bool
) -> None:
    """Eager steps score at the width ``max_seq_len`` bounds, captured ones at
    the page table's; decode rows and prefill rows never share a chunk, and
    only prefill chunks may take the MFMA kernel."""
    table_width, compress_ratio, token_topk = 168, 4, 2048
    capacity = table_width * _COMPRESSED_PAGE
    calls: list[tuple[int, int, int | None, bool]] = []

    def qsa_mqa_paged(q, *args, num_columns=None, allow_mfma=False, **kwargs):
        calls.append((args[2].data_ptr(), q.shape[0], num_columns, allow_mfma))
        rows = q.shape[0]
        return torch.empty(rows, num_columns), torch.empty(rows, dtype=torch.int32)

    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", qsa_mqa_paged)
    monkeypatch.setattr(qsa_ops.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(qsa_ops.ops, "top_k_per_row_decode", lambda *args: None)
    monkeypatch.setattr(qsa_ops, "expand_qsa_block_indices_cuda", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    rows, num_decode_tokens = 3000, 60
    token_to_req = torch.zeros(rows, dtype=torch.int32)

    def select(**kwargs: Any) -> list[tuple[int, int, int | None, bool]]:
        calls.clear()
        qsa_ops.qsa_select_paged_tokens(
            torch.empty((rows, 1, 1)),
            torch.empty((1, _COMPRESSED_PAGE, 1, 1)),
            torch.empty((1, table_width), dtype=torch.int32),
            token_to_req,
            torch.zeros(rows, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32),
            token_topk,
            compress_ratio,
            **kwargs,
        )
        return [(start - base, n, w, mfma) for start, n, w, mfma in calls]

    base = token_to_req.data_ptr()
    # 128 MiB of fp32 logits: 509 rows of 65856 columns, 1022 of 32832
    width, chunk = (capacity, 509) if capturing else (32832, 1022)
    expected = [(0, 60, width, False)] + [
        (start * token_to_req.element_size(), min(chunk, rows - start), width, True)
        for start in range(60, rows, chunk)
    ]
    assert select(max_seq_len=131200, num_decode_tokens=num_decode_tokens) == expected
    # without the decode count every row keeps the per-row kernel
    assert not any(mfma for *_, mfma in select(max_seq_len=131200))
    # without max_seq_len the full width, as before
    assert {w for _, _, w, _ in select()} == {capacity}


@requires_qsa_kernels
def test_qsa_selection_scores_prefill_chunks_with_mfma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 128K mixed step (8 decode rows, a 1500-row chunk) through the real
    selection: the trimmed width selects what the full width selects, the
    decode rows and the last 479 rows (below the gate) select exactly
    what the per-row kernel selects, and the 1022 rows of the MFMA chunk
    select >= 99.9 % of the same tokens."""
    requests = [(1, 131200 - i) for i in range(8)] + [(1500, 131200)]
    inputs = _served_scoring_inputs(requests, table_width=84)
    launches: list[tuple[int, int]] = []
    launch = qsa_ops.launch_qsa_mqa_paged_mfma
    base = inputs.token_to_req.data_ptr()

    def recorded(q, k_cache, page_table, token_to_req, *args: Any) -> None:
        launches.append((q.shape[0], (token_to_req.data_ptr() - base) // 4))
        launch(q, k_cache, page_table, token_to_req, *args)

    monkeypatch.setattr(qsa_ops, "launch_qsa_mqa_paged_mfma", recorded)

    def select(**kwargs: Any) -> torch.Tensor:
        out = qsa_ops.qsa_select_paged_tokens(
            inputs.q,
            inputs.k_cache,
            inputs.page_table,
            inputs.token_to_req,
            inputs.positions,
            inputs.seq_lens,
            _TOKEN_TOPK,
            _RATIO,
            **kwargs,
        )
        return out.sort(dim=1).values

    full_width = select()
    per_row = select(max_seq_len=131200)
    assert not launches
    mixed = select(max_seq_len=131200, num_decode_tokens=8)
    assert launches == [(1022, 8)]
    assert torch.equal(full_width, per_row)
    mfma_rows = slice(8, 8 + 1022)
    outside = torch.ones(mixed.shape[0], dtype=torch.bool, device=mixed.device)
    outside[mfma_rows] = False
    assert torch.equal(mixed[outside], per_row[outside])

    def tokens(rows: torch.Tensor) -> torch.Tensor:
        # column 0 collects the -1 padding
        selected = rows.new_zeros((rows.shape[0], 131200 + 1), dtype=torch.bool)
        return selected.scatter_(1, rows.long() + 1, True)[:, 1:]

    reference_tokens = tokens(per_row[mfma_rows])
    same = (tokens(mixed[mfma_rows]) & reference_tokens).sum()
    assert same >= 0.999 * reference_tokens.sum()


@requires_qsa_kernels
def test_qsa_mfma_scorer_warmup_covers_served_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the warmup, MFMA launches at every decode-prefix offset and at
    both score widths compile nothing: the selection slices the request map
    and positions at the decode count, so their alignment varies by step."""
    from triton import knobs

    from vllm.models.qwen4_exp.amd.ops import qsa_mfma_scorer

    # A page-table width no other test uses: its kernel variant is new here.
    table_width = 13
    inputs = _served_scoring_inputs([(3, 4000), (2, 4100), (900, 4200)], table_width)
    compiles: list[object] = []
    monkeypatch.setattr(
        knobs.runtime,
        "jit_post_compile_hook",
        lambda *args, **kwargs: compiles.append(kwargs.get("fn")),
    )
    widths = qsa_mfma_scorer.warmup_qsa_mqa_paged_mfma(
        inputs.k_cache,
        inputs.page_table,
        num_heads=_INDEXER_HEADS,
        head_dim=_INDEXER_DIM,
        compress_ratio=_RATIO,
        max_num_batched_tokens=8192,
    )
    capacity = table_width * _COMPRESSED_PAGE
    assert widths == (64, capacity) and compiles
    compiles.clear()
    for offset in range(4):
        for columns in (1088, capacity):
            qsa_ops.qsa_mqa_paged(
                inputs.q[offset:],
                inputs.k_cache,
                inputs.page_table,
                inputs.token_to_req[offset:],
                inputs.positions[offset:],
                inputs.seq_lens,
                _RATIO,
                num_columns=columns,
                allow_mfma=True,
            )
    assert compiles == []


@requires_qsa_kernels
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize(
    "num_rows,num_query_heads,num_kv_heads,page_size,flydsl",
    [
        pytest.param(1, 24, 2, 1792, None, id="tp1_split64"),
        pytest.param(16, 12, 1, 1792, None, id="tp2_split32"),
        pytest.param(32, 6, 1, 1024, None, id="tp4_split8"),
        pytest.param(257, 6, 1, 1024, None, id="tp4_split4"),
        pytest.param(513, 6, 1, 1024, None, id="tp4_split1"),
        # AITER's FlyDSL sparse prefill takes the prefilling requests' rows:
        # both requests (direct kernels), or the second one (union tiles from
        # 1536 rows) after rows left to the Triton kernel.
        pytest.param(1024, 12, 1, 1792, "both", id="tp2_flydsl_direct"),
        pytest.param(4096, 12, 1, 1792, "last", id="tp2_flydsl_union"),
    ],
)
def test_qsa_sparse_paged_attention_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
    flydsl: str | None,
    use_fp8: bool,
) -> None:
    if flydsl is not None and (use_fp8 or not fly.flydsl_prefill_available()):
        pytest.skip("AITER's FlyDSL sparse prefill takes bf16 caches on gfx950")
    torch.manual_seed(2)
    head_dim = 256
    num_requests = 2
    num_selected_pages = 64
    num_pages_per_request = num_selected_pages + 1
    rows_per_request = math.ceil(num_rows / num_requests)
    page_padding = 0
    if flydsl is not None:
        # The op takes causal selections only: the selected pages end before
        # the first query row. Its cache pages sit further apart than they
        # hold, as a layer's pages do in a served pool.
        num_pages_per_request += cdiv(rows_per_request, page_size)
        page_padding = 4
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size + page_padding,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )[:, :page_size]
    if use_fp8:
        fp8_dtype = current_platform.fp8_dtype()
        # The kernel's contract is that the cache already holds values in
        # the attention dtype's numeric space; quantize the stored cache and
        # have the reference consume the stored (quantized) values upcast
        # to float, not the original bf16 — the fp8 cast is lossy.
        kv_stored = kv_cache.to(fp8_dtype)
        k_cache, v_cache = kv_stored.split(head_dim, dim=-1)
        ref_k, ref_v = k_cache.float(), v_cache.float()
    else:
        k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
        ref_k, ref_v = k_cache, v_cache
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
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

    prefill_layout = None
    flydsl_rows: list[int] = []
    if flydsl is not None:
        rows = (rows_per_request, num_rows - rows_per_request)
        first = 0 if flydsl == "both" else 1
        prefill_layout = QSAPrefillLayout(
            first,
            sum(rows[:first]),
            rows[first:],
            tuple(context_length - n for n in rows[first:]),
        )
        op = fly._flydsl_op()

        def counted(q, *args, **kwargs):
            flydsl_rows.append(q.shape[0])
            return op(q, *args, **kwargs)

        monkeypatch.setattr(fly, "_flydsl_op", lambda: counted)
    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        prefill_layout=prefill_layout,
    )
    if prefill_layout is not None:
        assert flydsl_rows == [prefill_layout.num_rows]
    expected = _qsa_sparse_paged_attention_reference(
        q,
        ref_k,
        ref_v,
        logical_indices,
        block_table,
        token_to_req,
        q.shape[-1] ** -0.5,
    )

    # Tight enough with the op that one wrong key in a row fails
    tolerance = 2e-2 if flydsl is None else 2e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@requires_qsa_kernels
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3"])
def test_qsa_impl_constructs_with_fp8_kv(kv_cache_dtype: str) -> None:
    """QSA accepts FP8 KV cache dtypes without probing FlashAttention."""
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAImpl

    impl = Qwen4ExpQSAImpl(
        num_heads=24,
        head_size=128,
        scale=0.088,
        num_kv_heads=2,
        kv_cache_dtype=kv_cache_dtype,
    )
    assert impl.kv_cache_dtype == kv_cache_dtype


# AITER's FlyDSL sparse prefill (gfx950): the host layout, the gate, the cache
# read in place, the hook and the warmup. The op itself runs in
# test_qsa_sparse_paged_attention_matches_reference.
_PAGE = 8  # tokens per page of the synthetic caches
_D = fly.HEAD_DIM
_WIDTH = 4 * 512 + 3  # the served selection width
_LAYOUTS = {
    # (first_req, first_row, query_lens, prefix_lens)
    "one short request": (0, 0, (5,), (0,)),
    "one token past a page": (1, 3, (1,), (_PAGE,)),
    "queries across pages": (2, 6, (_PAGE + 2, 3), (_PAGE - 1, 2 * _PAGE)),
    "ragged requests": (1, 3, (2 * _PAGE - 1, 1, 17), (0, 3 * _PAGE, 5)),
}


def _metadata(
    requests, *, pad_tokens=0, host_seq=True, prefilling=True, device_seq=None
) -> CommonAttentionMetadata:
    """A batch of (prefix, query, is_prefilling) requests. The device seq_lens
    are zeros unless given, so that nothing can read them."""
    query = [q for _, q, _ in requests]
    seq = [p + q for p, q, _ in requests]
    starts = torch.tensor([0, *itertools.accumulate(query)], dtype=torch.int32)
    tokens = sum(query) + pad_tokens
    return CommonAttentionMetadata(
        query_start_loc=starts.clone(),
        query_start_loc_cpu=starts,
        seq_lens=torch.zeros(len(seq), dtype=torch.int32)
        if device_seq is None
        else device_seq,
        num_reqs=len(requests),
        num_actual_tokens=tokens,
        max_query_len=max(query),
        max_seq_len=max(seq),
        block_table_tensor=torch.zeros(len(seq), 1, dtype=torch.int32),
        slot_mapping=torch.zeros(tokens, dtype=torch.int64),
        is_prefilling=torch.tensor([r[2] for r in requests]) if prefilling else None,
        seq_lens_cpu_upper_bound=torch.tensor(seq, dtype=torch.int32)
        if host_seq
        else None,
    )


def _interleaved_cache(num_pages, heads=1, per_page=None, base_rows=0):
    """K and V as the QSA owner views its cache: the two halves of one K|V
    record per token. With ``per_page`` the pages sit that many records apart,
    ``base_rows`` records into a larger allocation (a layer's pages among every
    layer's)."""
    g = torch.Generator().manual_seed(0)
    if per_page is None:
        raw = torch.randn(num_pages, heads, _PAGE, 2 * _D, generator=g)
    else:
        assert heads == 1
        flat = torch.randn(base_rows + num_pages * per_page, 2 * _D, generator=g)
        # bf16 before the view: a cast after it would pack the pages together
        raw = flat.to(torch.bfloat16).as_strided(
            (num_pages, 1, _PAGE, 2 * _D),
            (per_page * 2 * _D, 2 * _D, 2 * _D, 1),
            base_rows * 2 * _D,
        )
    k, v = raw.to(torch.bfloat16).transpose(1, 2).split(_D, dim=-1)
    return canonicalize_singleton_dim_strides(k), canonicalize_singleton_dim_strides(v)


def _planes(num_pages, heads, per_page, base_rows=0):
    """K and V in separate token pools viewed as pages ``per_page`` rows apart."""
    g = torch.Generator().manual_seed(0)
    shape = (num_pages, _PAGE, heads, _D)
    stride = (per_page * heads * _D, heads * _D, _D, 1)
    return tuple(
        torch.randn(base_rows + num_pages * per_page, heads, _D, generator=g)
        .to(torch.bfloat16)
        .as_strided(shape, stride, base_rows * heads * _D)
        for _ in range(2)
    )


def _block_table(lengths, num_pages=40, seed=0):
    """int32 [requests, width]: each request's pages, drawn without repeats."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_pages, generator=g).tolist()
    pages = [cdiv(n, _PAGE) for n in lengths]
    table = torch.zeros(len(pages), max(pages) + 1, dtype=torch.int32)
    for r, start in enumerate(itertools.accumulate([0, *pages[:-1]])):
        table[r, : pages[r]] = torch.tensor(perm[start : start + pages[r]])
    return table


def _layout_table(layout, seed=0):
    """A block table for the layout's requests, after a few short requests."""
    own = [q + p for q, p in zip(layout.query_lens, layout.prefix_lens)]
    return _block_table([3] * layout.first_req + own, seed=seed)


def _reference_kv(k_cache, v_cache, block_table, layout):
    """K and V of every key of the layout's requests, one token at a time."""
    tokens = [
        (int(block_table[layout.first_req + i, t // _PAGE]), t % _PAGE)
        for i, (q, p) in enumerate(zip(layout.query_lens, layout.prefix_lens))
        for t in range(q + p)
    ]
    return tuple(
        torch.stack([cache[page, offset] for page, offset in tokens])
        for cache in (k_cache, v_cache)
    )


_HOST_LAYOUTS = {
    # requests (prefix, query, prefilling) -> the layout, or None
    "one prompt": ([(0, 4096, True)], (0, 0, (4096,), (0,))),
    "a chunk of a long prompt": ([(8192, 8192, True)], (0, 0, (8192,), (8192,))),
    "decodes, then prefills": (
        [(70, 3, False), (90, 3, False), (0, 1000, True), (4096, 2048, True)],
        (2, 6, (1000, 2048), (0, 4096)),
    ),
    "a prefilling short extend starts the tail": (
        [(10, 3, False), (500, 5, True), (0, 300, True)],
        (1, 3, (5, 300), (500, 0)),
    ),
    "decodes only": ([(100, 3, False), (50, 3, False)], None),
    "a decode after a prefill": ([(0, 100, True), (10, 3, False)], None),
}


@pytest.mark.parametrize(
    "requests,expected",
    [pytest.param(r, e, id=name) for name, (r, e) in _HOST_LAYOUTS.items()],
)
def test_qsa_host_prefill_layout(requests, expected) -> None:
    layout = host_prefill_layout(_metadata(requests))
    assert layout == (None if expected is None else QSAPrefillLayout(*expected))


def test_qsa_host_prefill_layout_reads_host_data_only() -> None:
    junk = torch.full((3,), 99, dtype=torch.int32)
    batch = [(70, 3, False), (0, 1000, True), (4096, 2048, True)]
    layout = host_prefill_layout(_metadata(batch, device_seq=junk))
    assert layout == QSAPrefillLayout(1, 3, (1000, 2048), (0, 4096))


def _sequence_shorter_than_its_query() -> CommonAttentionMetadata:
    metadata = _metadata([(0, 100, True), (0, 50, True)])
    assert metadata.seq_lens_cpu_upper_bound is not None
    metadata.seq_lens_cpu_upper_bound[1] = 40
    return metadata


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"pad_tokens": 4}, id="rows past the requests"),
        pytest.param({"host_seq": False}, id="no host seq_lens"),
        pytest.param({"prefilling": False}, id="no is_prefilling"),
        pytest.param(_sequence_shorter_than_its_query, id="sequence below query"),
        pytest.param(
            lambda: _metadata([(0, 0, True), (0, 50, True)]),
            id="empty prefilling request",
        ),
    ],
)
def test_qsa_host_prefill_layout_declines_what_the_host_cannot_give(
    metadata,
) -> None:
    batch = [(70, 3, False), (0, 1000, True)]
    assert host_prefill_layout(_metadata(batch)) is not None
    if isinstance(metadata, dict):
        metadata = _metadata(batch, **metadata)
    else:
        metadata = metadata()
    assert host_prefill_layout(metadata) is None


@pytest.mark.parametrize(
    "available,dtype,prompt,collects",
    [
        (True, torch.bfloat16, 765, True),  # 768 rows with the decodes
        (True, torch.bfloat16, 764, False),
        (False, torch.bfloat16, 1000, False),
        (True, torch.uint8, 1000, False),  # fp8 or Octave storage
    ],
)
def test_qsa_rocm_builder_collects_layouts_when_the_op_runs(
    monkeypatch: pytest.MonkeyPatch, available, dtype, prompt, collects
) -> None:
    monkeypatch.setattr(fly, "flydsl_prefill_available", lambda: available)
    spec = SimpleNamespace(dtype=dtype, kv_quant_mode=SimpleNamespace(is_octave=False))
    builder = qsa_owner.Qwen4ExpROCmQSAMetadataBuilder(
        spec, [], None, torch.device("cpu")
    )
    metadata = builder.build(0, _metadata([(70, 3, False), (0, prompt, True)]))
    expected = QSAPrefillLayout(1, 3, (prompt,), (0,)) if collects else None
    assert metadata.prefill_layout == expected
    # the layout survives a block-table swap (hybrid cache groups)
    swapped = builder.update_block_table(
        metadata, torch.zeros(2, 3, dtype=torch.int32), metadata.slot_mapping
    )
    assert swapped.prefill_layout == expected


def test_qsa_flydsl_op_needs_an_aiter_that_supports_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aiter_op = object()

    def importer(platforms):
        def import_module(name):
            if platforms is None:
                raise ImportError(f"No module named {name!r}")
            if name.endswith("qsa_sparse_prefill"):
                return SimpleNamespace(flydsl_qsa_sparse_prefill=aiter_op)
            return SimpleNamespace(PLATFORMS=platforms)

        return import_module

    try:
        for platforms, expected in [
            (None, None),
            ({("gfx942", 80): 1}, None),
            ({("gfx942", 80): 1, ("gfx950", 256): 2}, aiter_op),
        ]:
            fly._flydsl_op.cache_clear()
            monkeypatch.setattr(
                fly, "importlib", SimpleNamespace(import_module=importer(platforms))
            )
            assert fly._flydsl_op() is expected
    finally:
        fly._flydsl_op.cache_clear()


def test_qsa_flydsl_prefill_runs_where_vllm_uses_aiter_on_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    on = {"aiter": True, "gfx950": True, "op": True}
    aiter_ops = SimpleNamespace(is_enabled=lambda: on["aiter"])
    monkeypatch.setattr(fly, "rocm_aiter_ops", aiter_ops)
    monkeypatch.setattr(fly, "_on_gfx950", lambda: on["gfx950"])
    monkeypatch.setattr(fly, "_flydsl_op", lambda: object() if on["op"] else None)
    for values in itertools.product((False, True), repeat=3):
        on.update(zip(on, values))
        assert fly.flydsl_prefill_available() is all(values)


def test_qsa_flydsl_prefill_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """bf16 caches, at least 768 rows, heads and a selection width the op
    takes, a contiguous selection and a contiguous 16-byte aligned output, on a
    device and AITER that run the op, outside graph capture."""
    available, capturing = [True], [False]
    monkeypatch.setattr(fly, "flydsl_prefill_available", lambda: available[0])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])

    def applies(rows=768, heads=12, head_dim=_D, width=_WIDTH, dtype=None, **kw):
        q = torch.empty(rows, heads, head_dim, dtype=torch.bfloat16)
        k_cache = torch.empty(2, _PAGE, 1, head_dim, dtype=dtype or torch.bfloat16)
        indices = kw.get("indices", torch.empty(rows, width, dtype=torch.int32))
        out = kw.get("out", torch.empty_like(q))
        return fly.flydsl_prefill_applies(rows, q, k_cache, indices, out)

    assert applies() and applies(heads=16)
    assert applies(width=4 * 256 + 3) and applies(width=4 * 1024 + 3)
    assert not applies(rows=767)
    assert not applies(dtype=current_platform.fp8_dtype())
    assert not applies(head_dim=128) and not applies(heads=17)
    assert not applies(width=_WIDTH - 1) and not applies(width=4 * 1280 + 3)
    wide = torch.empty(768, _WIDTH + 1, dtype=torch.int32)
    assert not applies(indices=wide[:, :_WIDTH])
    assert not applies(out=torch.empty(768, 24, _D, dtype=torch.bfloat16)[:, ::2])
    shifted = torch.empty(768 * 12 * _D + 1, dtype=torch.bfloat16)[1:]
    assert not applies(out=shifted.view(768, 12, _D))
    capturing[0] = True
    assert not applies()
    capturing[0], available[0] = False, False
    assert not applies()


_IN_PLACE_CACHES = {
    "interleaved K|V records": lambda: _interleaved_cache(
        40, per_page=_PAGE + 4, base_rows=8
    ),
    "interleaved, pages back to back": lambda: _interleaved_cache(
        40, per_page=_PAGE, base_rows=8
    ),
    "two heads in separate planes": lambda: _planes(40, 2, _PAGE, base_rows=8),
}


@pytest.mark.parametrize("cache", _IN_PLACE_CACHES.values(), ids=_IN_PLACE_CACHES)
@pytest.mark.parametrize("layout", _LAYOUTS.values(), ids=_LAYOUTS)
def test_qsa_flydsl_reads_the_cache_in_place(layout, cache) -> None:
    layout = QSAPrefillLayout(*layout)
    k_cache, v_cache = cache()
    table = _layout_table(layout)
    expected = _reference_kv(k_cache, v_cache, table, layout)
    # the requests before the layout's are not read
    table[: layout.first_req] = 10**6
    pools = fly._pools(k_cache, v_cache)
    assert pools is not None
    k, v, per_page = pools
    slots = fly.token_slots(layout, table, _PAGE, per_page)
    lengths = [q + p for q, p in zip(layout.query_lens, layout.prefix_lens)]
    assert slots.dtype == torch.int32 and slots.shape == (sum(lengths),)
    # what the op does with a slot: row ``slot`` of each pool
    assert torch.equal(k[slots.long()], expected[0])
    assert torch.equal(v[slots.long()], expected[1])
    # in place: each pool starts where its cache starts, a token row apart
    assert k.data_ptr() == k_cache.data_ptr() and v.data_ptr() == v_cache.data_ptr()
    assert k.stride(0) == v.stride(0) == k_cache.stride(1)
    # the op reads four-token blocks at four consecutive rows from a multiple
    # of four
    for start, n in zip(itertools.accumulate([0, *lengths]), lengths):
        for block in slots[start : start + n].long().split(4):
            assert block[0] % 4 == 0
            assert torch.equal(block, block[0] + torch.arange(len(block)))


def test_qsa_flydsl_slots_copy_the_lengths_once_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies = []

    def h2d(data, device, dtype):
        copies.append(list(data))
        return torch.tensor(data, dtype=dtype, device=device)

    monkeypatch.setattr(fly, "async_tensor_h2d", h2d)
    layout = QSAPrefillLayout(*_LAYOUTS["queries across pages"])
    table = _layout_table(layout)
    first = fly.token_slots(layout, table, _PAGE, _PAGE + 4)
    assert fly.token_slots(layout, table, _PAGE, _PAGE + 4) is first
    assert copies == [[q + p for q, p in zip(layout.query_lens, layout.prefix_lens)]]
    # another block table (another KV cache group) gets its own slots
    assert (
        fly.token_slots(layout, _layout_table(layout, 5), _PAGE, _PAGE + 4) is not first
    )
    assert len(copies) == 2


def _paged(page_stride):
    """K and V halves of 2 D records, pages ``page_stride`` elements apart."""
    storage = torch.zeros(41 * page_stride, dtype=torch.bfloat16)
    shape, stride = (40, _PAGE, 1, _D), (page_stride, 2 * _D, _D, 1)
    return storage.as_strided(shape, stride), storage.as_strided(shape, stride, _D)


_NOT_IN_PLACE = {
    "two heads whose vectors are not adjacent (TP1)": lambda: _interleaved_cache(40, 2),
    "pages 10 rows apart": lambda: _paged(10 * 2 * _D),
    "pages 9.5 rows apart": lambda: _paged(19 * _D),
    "views off a 16-byte boundary": lambda: tuple(
        c.as_strided(c.shape, c.stride(), c.storage_offset() + 1)
        for c in _interleaved_cache(40, per_page=_PAGE + 4, base_rows=8)
    ),
}


@pytest.mark.parametrize("cache", _NOT_IN_PLACE.values(), ids=_NOT_IN_PLACE)
def test_qsa_flydsl_leaves_a_cache_it_cannot_read_in_place_to_triton(cache) -> None:
    assert fly._pools(*cache()) is None


def test_qsa_flydsl_reads_the_served_cache_in_place() -> None:
    # K and V of one QSA layer in the served pool (Qwen3.8-Flash-Next, TP2,
    # bf16): 1568-token pages 10,235,904 elements apart, a 512-element K|V
    # record per token
    pool = torch.empty(10175 * 10235904, dtype=torch.bfloat16, device="meta")
    shape, stride = (10175, 1568, 1, 256), (10235904, 512, 256, 1)
    k, v = pool.as_strided(shape, stride), pool.as_strided(shape, stride, 256)
    pools = fly._pools(k, v)
    assert pools is not None and pools[2] == 19992
    for view, cache in zip(pools[:2], (k, v)):
        assert view.shape == (10174 * 19992 + 1568, 1, 256)
        assert view.stride() == (512, 256, 1)
        assert view.storage_offset() == cache.storage_offset()


class _FakeOp:
    """Stands for flydsl_qsa_sparse_prefill: records each call and writes ones."""

    def __init__(self):
        self.calls: list[SimpleNamespace] = []

    def __call__(self, q, k, v, indices, *, query_lens, prefix_lens, **kwargs):
        self.calls.append(
            SimpleNamespace(
                q=q,
                k=k,
                v=v,
                indices=indices,
                query_lens=query_lens,
                prefix_lens=prefix_lens,
                **kwargs,
            )
        )
        if kwargs.get("out") is not None:
            kwargs["out"].fill_(1)


@pytest.fixture
def flydsl_op(monkeypatch: pytest.MonkeyPatch) -> _FakeOp:
    """A fake op where vLLM uses AITER on gfx950, outside graph capture."""
    op = _FakeOp()
    monkeypatch.setattr(fly, "rocm_aiter_ops", SimpleNamespace(is_enabled=lambda: True))
    monkeypatch.setattr(fly, "_on_gfx950", lambda: True)
    monkeypatch.setattr(fly, "_flydsl_op", lambda: op)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return op


def _step_inputs(layout, heads=12):
    rows = layout.first_row + layout.num_rows
    g = torch.Generator().manual_seed(1)
    q = torch.randn(rows, heads, _D, generator=g).to(torch.bfloat16)
    indices = torch.randint(0, 50, (rows, _WIDTH), generator=g, dtype=torch.int32)
    return q, indices, torch.zeros(rows, heads, _D, dtype=torch.bfloat16)


@pytest.mark.parametrize("strided_q", [False, True])
def test_qsa_flydsl_takes_the_prefill_rows_with_the_hosts_lengths(
    monkeypatch: pytest.MonkeyPatch, flydsl_op: _FakeOp, strided_q: bool
) -> None:
    monkeypatch.setattr(fly, "FLYDSL_PREFILL_MIN_ROWS", 1)
    layout = QSAPrefillLayout(*_LAYOUTS["ragged requests"])
    k_cache, v_cache = _interleaved_cache(40, per_page=_PAGE + 4, base_rows=8)
    table = _layout_table(layout)
    q, indices, out = _step_inputs(layout)
    if strided_q:
        wide = torch.zeros(*q.shape[:2], 2 * _D, dtype=q.dtype)
        wide[..., :_D] = q
        q = wide[..., :_D]
    first = layout.first_row
    args = (q, k_cache, v_cache, indices, table, out, layout)
    assert fly.try_flydsl_prefill(*args) == first
    (call,) = flydsl_op.calls
    assert call.q.is_contiguous() and torch.equal(call.q, q[first:])
    assert torch.equal(call.indices, indices[first:])
    expected = _reference_kv(k_cache, v_cache, table, layout)
    assert torch.equal(call.k[call.kv_slots.long()], expected[0])
    assert torch.equal(call.v[call.kv_slots.long()], expected[1])
    assert (call.query_lens, call.prefix_lens) == (
        layout.query_lens,
        layout.prefix_lens,
    )
    assert call.softmax_scale == _D**-0.5
    # the op wrote the prefill rows of the output and nothing else
    assert call.out.data_ptr() == out[first:].data_ptr()
    assert not out[:first].any() and out[first:].eq(1).all()


def test_qsa_flydsl_leaves_a_step_it_declines_untouched(
    monkeypatch: pytest.MonkeyPatch, flydsl_op: _FakeOp
) -> None:
    layout = QSAPrefillLayout(*_LAYOUTS["ragged requests"])
    k_cache, v_cache = _interleaved_cache(40, per_page=_PAGE + 4, base_rows=8)
    table = _layout_table(layout)
    q, indices, out = _step_inputs(layout)
    monkeypatch.setattr(fly, "FLYDSL_PREFILL_MIN_ROWS", layout.num_rows + 1)
    assert (
        fly.try_flydsl_prefill(q, k_cache, v_cache, indices, table, out, layout) is None
    )
    monkeypatch.setattr(fly, "FLYDSL_PREFILL_MIN_ROWS", layout.num_rows)
    # rows that do not match the layout (a sliced micro-batch)
    sliced = (q[:-1], k_cache, v_cache, indices[:-1], table, out[:-1], layout)
    assert fly.try_flydsl_prefill(*sliced) is None
    # a cache the op cannot read in place
    k_heads, v_heads = _interleaved_cache(40, 2)
    args = (q, k_heads, v_heads, indices, table, out, layout)
    assert fly.try_flydsl_prefill(*args) is None
    assert not flydsl_op.calls and not out.any()
    args = (q, k_cache, v_cache, indices, table, out, layout)
    assert fly.try_flydsl_prefill(*args) == layout.first_row


class _Launches:
    """Stands for a Triton kernel: records the grid of each launch."""

    def __init__(self):
        self.grids: list[tuple] = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.grids.append(grid)


@requires_qsa_kernels
@pytest.mark.parametrize(
    "taken,launched",
    [("no layout", [12]), (None, [12]), (3, [3]), (0, [])],
    ids=["no layout", "declined", "decodes left to Triton", "no rows left"],
)
def test_qsa_sparse_paged_attention_runs_triton_on_the_rows_the_op_leaves(
    monkeypatch: pytest.MonkeyPatch, taken, launched
) -> None:
    launches = _Launches()
    monkeypatch.setattr(qsa_ops, "_qsa_sparse_paged_gqa_splitk_kernel", launches)
    monkeypatch.setattr(qsa_ops, "_qsa_merge_splitk_kernel", _Launches())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    layout = None if taken == "no layout" else QSAPrefillLayout(1, 3, (9,), (0,))
    seen = []

    def try_flydsl_prefill(q, k, v, indices, table, out, prefill_layout):
        seen.append(prefill_layout)
        return taken

    monkeypatch.setattr(fly, "try_flydsl_prefill", try_flydsl_prefill)
    k_cache, v_cache = _interleaved_cache(4)
    q = torch.zeros(12, 12, _D, dtype=torch.bfloat16)
    indices = torch.zeros(12, _WIDTH, dtype=torch.int32)
    table, token_to_req = (
        torch.zeros(3, 2, dtype=torch.int32),
        torch.zeros(12, dtype=torch.int32),
    )
    out = torch.zeros_like(q)
    result = qsa_ops.qsa_sparse_paged_attention(
        q, k_cache, v_cache, indices, table, token_to_req, out, prefill_layout=layout
    )
    assert result is out and seen == ([] if layout is None else [layout])
    # the Triton kernel's rows (the entry checks every input against q's rows)
    assert [grid[0] for grid in launches.grids] == launched


def test_qsa_owner_passes_the_metadata_layout_to_the_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        qsa_ops,
        "qsa_sparse_paged_attention",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    impl = qsa_owner.Qwen4ExpQSAImpl.__new__(qsa_owner.Qwen4ExpQSAImpl)
    impl.octave_fmt, impl.head_size, impl.kv_cache_dtype = None, _D, "auto"
    rows = 12
    layer = SimpleNamespace(
        topk_indices_buffer=torch.zeros(rows, _WIDTH, dtype=torch.int32),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )
    layout = QSAPrefillLayout(1, 3, (9,), (0,))
    for prefill_layout in (None, layout):
        metadata = SimpleNamespace(
            num_actual_tokens=rows,
            block_table=torch.zeros(3, 2, dtype=torch.int32),
            prefill_layout=prefill_layout,
        )
        impl.forward(
            layer,
            torch.zeros(rows, 12, _D, dtype=torch.bfloat16),
            torch.zeros(4, 1, _PAGE, 2 * _D, dtype=torch.bfloat16),
            metadata,
            torch.zeros(rows, 12, _D, dtype=torch.bfloat16),
            token_to_req=torch.zeros(rows, dtype=torch.int32),
        )
    assert calls == [{"prefill_layout": None}, {"prefill_layout": layout}]


def _assert_the_ops_selection(row: list[int], v: int, blocks: int) -> None:
    """The op's contract for a row whose causal window holds v tokens:
    columns [0, 4 min(v // 4, T)) hold that many distinct complete blocks
    j < v // 4 of four tokens, the next v % 4 columns the tail tokens, the
    rest -1."""
    whole = min(v // 4, blocks)
    starts = row[: 4 * whole : 4]
    assert len(set(starts)) == whole
    for j, start in enumerate(starts):
        assert start % 4 == 0 and start // 4 < v // 4
        assert row[4 * j : 4 * j + 4] == [start + i for i in range(4)]
    tail = [4 * (v // 4) + i for i in range(v % 4)]
    assert row[4 * whole : 4 * whole + v % 4] == tail
    assert set(row[4 * whole + v % 4 :]) <= {-1}


def test_qsa_flydsl_warmup_calls_through_the_layers_cache(
    monkeypatch: pytest.MonkeyPatch, flydsl_op: _FakeOp
) -> None:
    synced = []
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device=None: synced.append(device)
    )
    k_cache, v_cache = _interleaved_cache(3, per_page=_PAGE + 4, base_rows=8)
    layouts = ((768,), (4096,), (2048, 2048))
    assert fly.warmup_flydsl_qsa_prefill(12, k_cache, v_cache, _WIDTH) == layouts
    assert [call.query_lens for call in flydsl_op.calls] == list(layouts)
    for call in flydsl_op.calls:
        assert call.prefix_lens == (0,) * len(call.query_lens)
        assert call.q.shape == (sum(call.query_lens), 12, _D)
        # the layer's own cache, read in place at its record pitch: the row
        # stride and pool span the served steps compile for
        assert call.k.data_ptr() == k_cache.data_ptr()
        assert call.v.data_ptr() == v_cache.data_ptr()
        assert call.k.stride(0) == call.v.stride(0) == 2 * _D
        assert call.k.shape[0] == 2 * (_PAGE + 4) + _PAGE
        # every key on block 0, the one block an extensible cache has mapped
        positions = torch.cat([torch.arange(n) for n in call.query_lens])
        assert torch.equal(call.kv_slots.long(), positions % _PAGE)
        # a selection the op takes: the first rows, the rows around 4 T = 2048
        # tokens (from where a row selects its last T blocks) and each last row
        ends = set(itertools.accumulate(call.query_lens))
        for row, position in enumerate(positions.tolist()):
            if position < 8 or 2044 <= position < 2060 or row + 1 in ends:
                indices = call.indices[row].tolist()
                _assert_the_ops_selection(indices, position + 1, 512)
    assert synced == [k_cache.device]


@pytest.mark.parametrize(
    "heads,width,fp8,op",
    [
        pytest.param(12, _WIDTH, False, False, id="no op"),
        pytest.param(12, _WIDTH - 1, False, True, id="odd width"),
        pytest.param(17, _WIDTH, False, True, id="17 heads per KV head"),
        pytest.param(12, _WIDTH, True, True, id="fp8 cache"),
    ],
)
def test_qsa_flydsl_warmup_skips_what_the_op_does_not_take(
    monkeypatch: pytest.MonkeyPatch, flydsl_op: _FakeOp, heads, width, fp8, op
) -> None:
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: None)
    if not op:
        monkeypatch.setattr(fly, "_flydsl_op", lambda: None)
    k_cache, v_cache = _interleaved_cache(3, per_page=_PAGE + 4, base_rows=8)
    if fp8:
        k_cache, v_cache = (
            c.to(current_platform.fp8_dtype()) for c in (k_cache, v_cache)
        )
    assert fly.warmup_flydsl_qsa_prefill(heads, k_cache, v_cache, width) == ()
    assert not flydsl_op.calls


@pytest.mark.parametrize(
    "dtype,available,warms",
    [
        (torch.bfloat16, True, True),
        (torch.bfloat16, False, False),
        # fp8 and Octave storage: an Octave cache does not split into K and V
        (torch.uint8, True, False),
    ],
)
def test_qsa_server_warmup_compiles_flydsl_on_the_owners_cache(
    monkeypatch: pytest.MonkeyPatch, dtype, available, warms
) -> None:
    layer = qsa_owner.Qwen4ExpQSAAttention.__new__(qsa_owner.Qwen4ExpQSAAttention)
    nn.Module.__init__(layer)
    layer.num_heads = 12
    layer.kv_cache = torch.empty(0, dtype=dtype)
    layer.indexer = SimpleNamespace(output_width=_WIDTH)
    views = []

    def kv_cache_views(kv_cache):
        views.append(kv_cache)
        return "k", "v"

    layer.impl = SimpleNamespace(kv_cache_views=kv_cache_views)
    model = nn.Module()
    model.add_module("layer", layer)
    calls = []

    def warmup(*args):
        calls.append(args)
        return ()

    monkeypatch.setattr(fly, "flydsl_prefill_available", lambda: available)
    monkeypatch.setattr(fly, "warmup_flydsl_qsa_prefill", warmup)
    monkeypatch.setattr(qwen4_exp_qsa_warmup, "_amd_qsa_triton_warmup", lambda w: None)
    qwen4_exp_qsa_warmup.qwen4_exp_qsa_triton_warmup(
        SimpleNamespace(get_model=lambda: model)
    )
    assert views == ([layer.kv_cache] if warms else [])
    assert calls == ([(12, "k", "v", _WIDTH)] if warms else [])
