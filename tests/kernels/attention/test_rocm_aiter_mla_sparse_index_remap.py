# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse MLA index remap test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse MLA index remap test requires a supported AITER "
        "installation.",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod

NUM_HEADS = 16
KV_LORA_RANK = 512
HEAD_SIZE = 576
NUM_TOKENS = 2
TOPK_TOKENS = 4


def _make_metadata():
    return SimpleNamespace(
        num_actual_tokens=NUM_TOKENS,
        req_id_per_token=torch.zeros(NUM_TOKENS, dtype=torch.int32, device="cpu"),
        block_table=torch.arange(16, dtype=torch.int32, device="cpu").view(2, 8),
        paged_kv_indptr=torch.zeros(NUM_TOKENS + 1, dtype=torch.int32, device="cpu"),
        paged_kv_indices=torch.zeros(
            NUM_TOKENS * TOPK_TOKENS, dtype=torch.int32, device="cpu"
        ),
        block_size=1,
        topk_tokens=TOPK_TOKENS,
    )


def _patch_init_deps(monkeypatch, *, speculative: bool):
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper,
        "check_num_heads_validity",
        staticmethod(lambda num_heads: None),
    )
    monkeypatch.setattr(
        sparse_mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            speculative_config=SimpleNamespace() if speculative else None,
        ),
    )
    monkeypatch.setattr(
        sparse_mod,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: tuple(
                torch.zeros(shape, dtype=dtype, device="cpu") for shape, dtype in specs
            )
        ),
    )


def _build_impl(indexer, topk_indices_buffer):
    return sparse_mod.ROCMAiterMLASparseImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=topk_indices_buffer,
        indexer=indexer,
        kv_lora_rank=KV_LORA_RANK,
    )


@pytest.mark.parametrize(
    "has_indexer,speculative,expected",
    [
        (True, False, True),
        (False, False, False),
        # Spec decode permutes the buffer, so skip layers must remap too.
        (False, True, True),
        (True, True, True),
    ],
)
def test_needs_index_remap_gate(monkeypatch, has_indexer, speculative, expected):
    _patch_init_deps(monkeypatch, speculative=speculative)
    buffer = torch.zeros(NUM_TOKENS, TOPK_TOKENS, dtype=torch.int32, device="cpu")
    indexer = SimpleNamespace(topk_indices_buffer=buffer) if has_indexer else None

    impl = _build_impl(indexer, buffer)

    assert impl.needs_index_remap is expected
    assert impl.topk_indices_buffer is buffer


@pytest.mark.parametrize("needs_index_remap", [True, False])
def test_forward_mqa_remaps_only_when_layer_owns_indexer(
    monkeypatch, needs_index_remap
):
    _patch_init_deps(monkeypatch, speculative=needs_index_remap)
    buffer = torch.zeros(NUM_TOKENS, TOPK_TOKENS, dtype=torch.int32, device="cpu")
    impl = _build_impl(None, buffer)
    assert impl.needs_index_remap is needs_index_remap

    remap_calls: list[tuple] = []
    monkeypatch.setattr(
        sparse_mod,
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: remap_calls.append(args),
    )
    monkeypatch.setattr(
        sparse_mod.AiterMLAHelper,
        "get_mla_padded_q",
        staticmethod(lambda num_heads, q: q),
    )
    attn_out = torch.zeros(NUM_TOKENS, NUM_HEADS, KV_LORA_RANK, device="cpu")
    monkeypatch.setattr(
        type(impl),
        "_forward_mla",
        lambda self, layer, q, kv_cache, attn_metadata: attn_out,
    )

    out, lse = impl.forward_mqa(
        torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, device="cpu"),
        torch.zeros(4, 1, HEAD_SIZE, device="cpu"),
        _make_metadata(),
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )

    assert out is attn_out
    assert lse is None
    assert len(remap_calls) == (1 if needs_index_remap else 0)
