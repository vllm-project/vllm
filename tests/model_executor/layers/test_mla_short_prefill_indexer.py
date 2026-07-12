# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

INDEXER_LAYER = "model.layers.0.self_attn.indexer.k_cache"
MLA_LAYER = "model.layers.0.self_attn.attn"


def make_indexer_metadata(
    *,
    max_seq_len: int = 2048,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    num_prefills: int = 1,
    num_prefill_tokens: int = 1,
    slot_mapping: torch.Tensor | None = None,
) -> DeepseekV32IndexerMetadata:
    if slot_mapping is None:
        slot_mapping = torch.zeros(num_prefill_tokens, dtype=torch.long)
    return DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=max_seq_len,
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=SimpleNamespace(chunks=[]) if num_prefills else None,
    )


def make_mla_metadata(
    *,
    use_mha: bool = True,
    num_decode_tokens: int = 0,
    has_prefill: bool = True,
):
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        prefill=SimpleNamespace(use_mha=use_mha) if has_prefill else None,
    )


@pytest.mark.parametrize(
    "batch_kind",
    [
        "short",
        "threshold_mismatch",
        "force_mqa",
        "no_prefill",
        "missing_mla",
        "mla_decode",
        "negative",
        "long",
        "very_long",
        "mixed",
        "decode",
        "capture",
        "full",
    ],
)
def test_short_prefill_updates_k_cache_before_scoring_decision(
    monkeypatch: pytest.MonkeyPatch,
    batch_kind: str,
):
    slot_mapping = torch.tensor([63, 64, 127, 128, -1])
    mla_num_decode_tokens = {"mla_decode": 32, "mixed": 1, "decode": 5}.get(
        batch_kind, 0
    )
    runtime_mode = (
        CUDAGraphMode.FULL if batch_kind == "full" else CUDAGraphMode.PIECEWISE
    )
    should_check_mla = batch_kind != "full"
    should_query_capture = batch_kind in ("short", "threshold_mismatch", "capture")
    should_skip = batch_kind in ("short", "threshold_mismatch")
    num_decodes = int(batch_kind in ("threshold_mismatch", "mixed", "decode"))
    num_decode_tokens = {
        "threshold_mismatch": 3,
        "mixed": 1,
        "decode": 5,
    }.get(batch_kind, 0)
    num_prefills = 0 if batch_kind in ("threshold_mismatch", "decode") else 2
    num_prefill_tokens = (
        0 if batch_kind in ("threshold_mismatch", "decode") else 5 - num_decode_tokens
    )
    max_seq_len = {"negative": -1, "long": 2049, "very_long": 8195}.get(
        batch_kind, 2048
    )
    if batch_kind == "threshold_mismatch":
        # With MTP=3 the indexer threshold is four. A main MLA backend whose
        # threshold is one (for example FlashMLA under DCP) still routes this
        # three-token extend through dense prefill attention.
        slot_mapping = slot_mapping[:3]
    indexer_metadata = make_indexer_metadata(
        max_seq_len=max_seq_len,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        slot_mapping=slot_mapping,
    )
    if indexer_metadata.num_decodes:
        indexer_metadata.decode = object()
    mla_metadata = make_mla_metadata(
        use_mha=batch_kind not in ("force_mqa", "negative", "long", "very_long"),
        num_decode_tokens=mla_num_decode_tokens,
        has_prefill=batch_kind not in ("no_prefill", "decode"),
    )

    resolved_layer_names: list[str] = []
    metadata_gets: list[str] = []
    capture_queries: list[None] = []
    observed: dict[str, object] = {}

    class TrackingMetadata(dict):
        def get(self, key, default=None):
            metadata_gets.append(key)
            return super().get(key, default)

    metadata_by_layer: dict[str, object] = {INDEXER_LAYER: indexer_metadata}
    if batch_kind != "missing_mla":
        metadata_by_layer[MLA_LAYER] = mla_metadata
    attn_metadata = TrackingMetadata(metadata_by_layer)
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=attn_metadata,
            cudagraph_runtime_mode=runtime_mode,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )

    resolve_layer_name = sparse_indexer._resolve_layer_name

    def record_layer_name_resolution(layer_name):
        resolved_layer_name = resolve_layer_name(layer_name)
        resolved_layer_names.append(resolved_layer_name)
        return resolved_layer_name

    def record_capture_query():
        capture_queries.append(None)
        return batch_kind == "capture"

    monkeypatch.setattr(
        sparse_indexer, "_resolve_layer_name", record_layer_name_resolution
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", record_capture_query)

    def record_cache_update(k, kv_cache, slots, block_size, scale_fmt):
        observed.update(
            k=k.clone(),
            kv_cache=kv_cache,
            slots=slots,
            block_size=block_size,
            scale_fmt=scale_fmt,
        )

    monkeypatch.setattr(
        sparse_indexer.ops, "indexer_k_quant_and_cache", record_cache_update
    )

    class ScoringReached(Exception):
        pass

    def scoring_workspace():
        if should_skip:
            pytest.fail("short dense-MHA prefill must not enter indexer scoring")
        raise ScoringReached

    def scoring_decode(*args):
        raise ScoringReached

    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", scoring_workspace)
    monkeypatch.setattr(
        sparse_indexer,
        "kv_cache_as_quant_view",
        scoring_decode,
    )

    hidden_states = torch.full((7, 1), float("inf"))
    k = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    kv_cache = torch.empty(1)
    topk_indices = torch.full((7, 2048), 17, dtype=torch.int32)

    def run_indexer():
        return sparse_indexer.sparse_attn_indexer(
            hidden_states,
            INDEXER_LAYER,
            kv_cache,
            torch.full((7, 1), float("inf")),
            None,
            k,
            torch.full((7, 1), float("inf")),
            128,
            "ue8m0",
            2048,
            4,
            4096,
            4096,
            topk_indices,
            False,
            False,
            MLA_LAYER,
        )

    if should_skip:
        assert run_indexer() is topk_indices
        assert torch.all(topk_indices == 17)
    else:
        with pytest.raises(ScoringReached):
            run_indexer()
        assert torch.all(topk_indices == -1)

    torch.testing.assert_close(observed["k"], k[: slot_mapping.numel()])
    assert observed["kv_cache"] is kv_cache
    assert observed["slots"] is slot_mapping
    assert observed["block_size"] == 128
    assert observed["scale_fmt"] == "ue8m0"
    expected_resolutions = [INDEXER_LAYER]
    if should_check_mla:
        expected_resolutions.append(MLA_LAYER)
    assert resolved_layer_names == expected_resolutions
    assert metadata_gets == ([MLA_LAYER] if should_check_mla else [])
    assert len(capture_queries) == int(should_query_capture)


def test_empty_slot_mapping_does_not_launch_k_cache_kernel(
    monkeypatch: pytest.MonkeyPatch,
):
    indexer_metadata = make_indexer_metadata(
        num_prefills=0,
        num_prefill_tokens=0,
        slot_mapping=torch.empty(0, dtype=torch.long),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={INDEXER_LAYER: indexer_metadata},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(
        sparse_indexer.ops,
        "indexer_k_quant_and_cache",
        lambda *args: pytest.fail("empty slot mapping must not launch a zero grid"),
    )

    topk_indices = torch.full((2, 2048), 17, dtype=torch.int32)
    result = sparse_indexer.sparse_attn_indexer(
        torch.empty(2, 1),
        INDEXER_LAYER,
        torch.empty(1),
        torch.empty(2, 1),
        None,
        torch.empty(2, 4),
        torch.empty(2, 1),
        128,
        "ue8m0",
        2048,
        4,
        4096,
        4096,
        topk_indices,
        False,
        False,
        "",
    )

    assert result is topk_indices
    assert torch.all(topk_indices == -1)
