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


@pytest.fixture(autouse=True)
def _reset_index_epochs():
    # The counter is keyed by id(), and a freed test buffer can hand its id to
    # the next one, so a leftover count would leak across tests.
    sparse_mod._INDEX_EPOCHS.clear()
    yield
    sparse_mod._INDEX_EPOCHS.clear()


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
        remapped_buf=None,
        remapped_epoch=-1,
    )


def _patch_init_deps(monkeypatch, *, speculative: bool = False):
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


def _new_buffer():
    return torch.zeros(NUM_TOKENS, TOPK_TOKENS, dtype=torch.int32, device="cpu")


def _patch_forward_deps(monkeypatch, impl, remap_calls):
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
    return attn_out


def _run_forward(impl, metadata):
    return impl.forward_mqa(
        torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, device="cpu"),
        torch.zeros(4, 1, HEAD_SIZE, device="cpu"),
        metadata,
        SimpleNamespace(_q_scale=None, _k_scale=None),
    )


@pytest.mark.parametrize("has_indexer", [True, False])
def test_owns_indexer_is_structural(monkeypatch, has_indexer):
    """owns_indexer reflects the layer, not the serving config."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    indexer = SimpleNamespace(topk_indices_buffer=buffer) if has_indexer else None

    impl = _build_impl(indexer, buffer)

    assert impl.owns_indexer is has_indexer
    assert impl.topk_indices_buffer is buffer


def test_forward_mqa_remaps_once_per_metadata(monkeypatch):
    """Skip layers sharing a metadata share its remap; the first one pays.

    All layers read one top-k buffer, so the reuse has to key off the metadata
    that owns ``paged_kv_indices``, not off the buffer being distinct.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    first = _build_impl(None, buffer)
    second = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    attn_out = _patch_forward_deps(monkeypatch, first, remap_calls)
    metadata = _make_metadata()

    out, lse = _run_forward(first, metadata)
    assert out is attn_out
    assert lse is None
    assert len(remap_calls) == 1
    assert metadata.remapped_buf is buffer

    _run_forward(second, metadata)
    assert len(remap_calls) == 1


def test_forward_mqa_remaps_for_each_metadata(monkeypatch):
    """Skip layers on a second metadata must remap its own paged_kv_indices.

    Regression: gating on ``indexer is not None`` skipped every layer in the
    non-indexer metadata group, leaving its ``paged_kv_indices`` unwritten.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    indexer_layer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_layer = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, indexer_layer, remap_calls)

    indexer_metadata = _make_metadata()
    _run_forward(indexer_layer, indexer_metadata)
    assert len(remap_calls) == 1
    assert indexer_metadata.remapped_buf is buffer

    # A distinct metadata instance carries its own, still-unwritten indices.
    skip_metadata = _make_metadata()
    _run_forward(skip_layer, skip_metadata)
    assert len(remap_calls) == 2
    assert skip_metadata.remapped_buf is buffer


def test_forward_mqa_remaps_after_a_later_indexer_write(monkeypatch):
    """A skip metadata must be refreshed once per indexer write, not once ever.

    Regression for cross-layer index sharing (``index_topk_freq``). With
    freq=4 on a 78-layer model the 57 skip layers share one metadata while the
    21 indexer layers share another, so the skip metadata has to pick up every
    new selection. Recording only the buffer left layers 4 onwards attending
    to the first selection of the pass, which collapsed RULER niah_single_2 to
    0.026 on GLM-5.3 and took GPQA from 0.884 to 0.672.
    """
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    # Construction order mirrors the layer stack: indexer, three skips, indexer.
    _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    first_skip = _build_impl(None, buffer)
    reuse_skip = _build_impl(None, buffer)
    _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    next_skip = _build_impl(None, buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, first_skip, remap_calls)
    skip_metadata = _make_metadata()

    _run_forward(first_skip, skip_metadata)
    assert len(remap_calls) == 1

    # Same selection, same metadata: reuse.
    _run_forward(reuse_skip, skip_metadata)
    assert len(remap_calls) == 1

    # A later indexer layer has since rewritten the buffer in place. The object
    # is unchanged, so only the epoch can reveal it.
    _run_forward(next_skip, skip_metadata)
    assert len(remap_calls) == 2
    assert skip_metadata.remapped_epoch == next_skip.index_epoch


def test_index_epoch_tracks_the_preceding_indexer_write(monkeypatch):
    """Layers between two indexer layers share the earlier one's epoch."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    first_indexer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_a = _build_impl(None, buffer)
    skip_b = _build_impl(None, buffer)
    second_indexer = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)
    skip_c = _build_impl(None, buffer)

    assert skip_a.index_epoch == first_indexer.index_epoch
    assert skip_b.index_epoch == first_indexer.index_epoch
    assert second_indexer.index_epoch == first_indexer.index_epoch + 1
    assert skip_c.index_epoch == second_indexer.index_epoch


def test_forward_mqa_remaps_when_buffer_differs(monkeypatch):
    """A layer reading a different buffer cannot reuse the metadata's remap."""
    _patch_init_deps(monkeypatch)
    remapped_buffer = _new_buffer()
    other_buffer = _new_buffer()
    layer = _build_impl(None, other_buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, layer, remap_calls)
    metadata = _make_metadata()
    metadata.remapped_buf = remapped_buffer

    _run_forward(layer, metadata)
    assert len(remap_calls) == 1
    assert metadata.remapped_buf is other_buffer


def test_forward_mqa_indexer_layer_always_remaps(monkeypatch):
    """The indexer rewrites its buffer in place, so its remap is never stale."""
    _patch_init_deps(monkeypatch)
    buffer = _new_buffer()
    impl = _build_impl(SimpleNamespace(topk_indices_buffer=buffer), buffer)

    remap_calls: list[tuple] = []
    _patch_forward_deps(monkeypatch, impl, remap_calls)
    metadata = _make_metadata()
    metadata.remapped_buf = buffer
    metadata.remapped_epoch = impl.index_epoch

    _run_forward(impl, metadata)
    assert len(remap_calls) == 1
