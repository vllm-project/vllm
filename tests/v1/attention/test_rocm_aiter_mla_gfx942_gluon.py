# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.v1.attention.backends.mla import rocm_aiter_mla
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLAHelper,
    AiterMLAImpl,
    _prepare_gluon_gfx942_graph_inputs,
)

NUM_HEADS = 12
KV_LORA_RANK = 4
QK_ROPE_HEAD_DIM = 2
QLEN = 4
NUM_REQS = 2


def _impl():
    return SimpleNamespace(
        num_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        scale=0.125,
    )


def _metadata(
    *,
    qlen: int = QLEN,
    num_reqs: int = NUM_REQS,
    output_dtype: torch.dtype = torch.bfloat16,
):
    return SimpleNamespace(
        decode=SimpleNamespace(
            max_qo_len=qlen,
            paged_kv_indptr=torch.arange(0, num_reqs + 1, dtype=torch.int32),
            paged_kv_indices=torch.arange(num_reqs, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(num_reqs, dtype=torch.int32),
            qo_indptr=torch.arange(
                0,
                (num_reqs + 1) * qlen,
                step=qlen,
                dtype=torch.int32,
            ),
            attn_out_dtype=output_dtype,
            use_gluon_decode=False,
            has_persistent_metadata=False,
        ),
        work_meta_data=None,
    )


def _inputs(
    *,
    qlen: int = QLEN,
    num_reqs: int = NUM_REQS,
    dtype: torch.dtype = torch.bfloat16,
):
    num_tokens = qlen * num_reqs
    q_nope = (
        torch.arange(
            num_tokens * NUM_HEADS * KV_LORA_RANK,
            dtype=torch.float32,
        )
        .reshape(num_tokens, NUM_HEADS, KV_LORA_RANK)
        .to(dtype)
    )
    q_pe = (
        torch.arange(
            num_tokens * NUM_HEADS * QK_ROPE_HEAD_DIM,
            dtype=torch.float32,
        )
        .reshape(num_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM)
        .to(dtype)
    )
    kv_cache = (
        torch.arange(
            3 * 2 * (KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            dtype=torch.float32,
        )
        .reshape(3, 2, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        .to(dtype)
    )
    return q_nope, q_pe, kv_cache


@pytest.mark.parametrize(
    ("num_heads", "qlen", "num_reqs", "expected"),
    [
        (12, 1, 1, False),
        (12, 2, 1, True),
        (12, 3, 2, False),
        (12, 4, 1, True),
        (12, 4, 16, True),
        (12, 4, 17, False),
        (12, 8, 1, True),
        (12, 9, 1, False),
        (8, 4, 1, False),
        (16, 4, 1, False),
    ],
)
def test_gfx942_gluon_graph_shape_gate(num_heads, qlen, num_reqs, expected):
    assert AiterMLAHelper.use_gluon_gfx942_graph(num_heads, qlen, num_reqs) is expected


def test_existing_single_token_gluon_requires_gfx950(monkeypatch):
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: False)
    assert not AiterMLAHelper.use_gluon_decode(NUM_HEADS, 1)

    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    assert AiterMLAHelper.use_gluon_decode(NUM_HEADS, 1)


def test_nondivisor_head_padding_and_unpadding():
    q = torch.arange(2 * NUM_HEADS * 3).reshape(2, NUM_HEADS, 3)
    padded = AiterMLAHelper.get_mla_padded_q(NUM_HEADS, q)

    assert padded.shape == (2, 16, 3)
    torch.testing.assert_close(padded[:, :NUM_HEADS], q)
    assert torch.count_nonzero(padded[:, NUM_HEADS:]) == 0

    output = torch.arange(2 * 16 * 3).reshape(2, 16, 3)
    unpadded = AiterMLAHelper.get_mla_unpadded_o(NUM_HEADS, output)
    torch.testing.assert_close(unpadded, output[:, :NUM_HEADS])


def test_gfx942_gluon_graph_normalizes_q_and_flattens_cache():
    q_nope, q_pe, kv_cache = _inputs()

    prepared = _prepare_gluon_gfx942_graph_inputs(
        (q_nope, q_pe),
        kv_cache,
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
    )

    assert prepared is not None
    graph_q_nope, graph_q_pe, graph_kv = prepared
    assert graph_q_nope.shape == (
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        KV_LORA_RANK,
    )
    assert graph_q_pe.shape == (
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        QK_ROPE_HEAD_DIM,
    )
    assert graph_kv.shape == (
        kv_cache.shape[0] * kv_cache.shape[1],
        kv_cache.shape[-1],
    )
    assert graph_q_nope.data_ptr() == q_nope.data_ptr()
    assert graph_q_pe.data_ptr() == q_pe.data_ptr()
    assert graph_kv.data_ptr() == kv_cache.data_ptr()


def test_gfx942_gluon_graph_passes_ragged_metadata_directly(monkeypatch):
    q_nope, q_pe, kv_cache = _inputs()
    metadata = _metadata()
    graph = mock.Mock(side_effect=lambda *args: args[3].fill_(7))

    monkeypatch.setattr(rocm_aiter_mla, "_on_gfx942", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_get_mla_gluon_gfx942_graph",
        lambda: graph,
    )

    output, output_lse = AiterMLAImpl.forward_mqa(
        _impl(),
        (q_nope, q_pe),
        kv_cache,
        metadata,
        layer=None,
    )

    assert output_lse is None
    assert output.shape == (
        NUM_REQS * QLEN,
        NUM_HEADS,
        KV_LORA_RANK,
    )
    assert torch.all(output == 7)
    graph_q_nope, graph_q_pe, graph_kv, graph_o, page_table, seq_info, scale = (
        graph.call_args.args
    )
    assert graph_q_nope.shape == (
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        KV_LORA_RANK,
    )
    assert graph_q_pe.shape == (
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        QK_ROPE_HEAD_DIM,
    )
    assert graph_kv.shape == (
        kv_cache.shape[0] * kv_cache.shape[1],
        kv_cache.shape[-1],
    )
    assert graph_o.shape == (
        NUM_REQS,
        QLEN,
        NUM_HEADS,
        KV_LORA_RANK,
    )
    assert page_table is metadata.decode.paged_kv_indices
    assert seq_info is metadata.decode.paged_kv_indptr
    assert scale == _impl().scale


@pytest.mark.parametrize(
    "unsupported",
    ["architecture", "symbol", "dtype", "shape"],
)
def test_gfx942_gluon_graph_unsupported_cases_fall_back(monkeypatch, unsupported):
    dtype = torch.float16 if unsupported == "dtype" else torch.bfloat16
    q_nope, q_pe, kv_cache = _inputs(dtype=dtype)
    if unsupported == "shape":
        q_pe = q_pe[..., :-1]

    graph = mock.Mock()
    asm = mock.Mock()
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_on_gfx942",
        lambda: unsupported != "architecture",
    )
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_get_mla_gluon_gfx942_graph",
        lambda: None if unsupported == "symbol" else graph,
    )
    monkeypatch.setattr(rocm_aiter_mla.rocm_aiter_ops, "mla_decode_fwd", asm)

    AiterMLAImpl.forward_mqa(
        _impl(),
        (q_nope, q_pe),
        kv_cache,
        _metadata(),
        layer=SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    graph.assert_not_called()
    asm.assert_called_once()


def test_gfx942_gluon_graph_runtime_errors_propagate(monkeypatch):
    q_nope, q_pe, kv_cache = _inputs()
    asm = mock.Mock()

    monkeypatch.setattr(rocm_aiter_mla, "_on_gfx942", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_get_mla_gluon_gfx942_graph",
        lambda: mock.Mock(side_effect=RuntimeError("kernel failed")),
    )
    monkeypatch.setattr(rocm_aiter_mla.rocm_aiter_ops, "mla_decode_fwd", asm)

    with pytest.raises(RuntimeError, match="kernel failed"):
        AiterMLAImpl.forward_mqa(
            _impl(),
            (q_nope, q_pe),
            kv_cache,
            _metadata(),
            layer=SimpleNamespace(_q_scale=None, _k_scale=None),
        )

    asm.assert_not_called()
