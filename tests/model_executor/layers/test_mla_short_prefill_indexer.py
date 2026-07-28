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
        max_seq_len=2048,
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=SimpleNamespace(chunks=[]) if num_prefills else None,
    )


def make_mla_metadata(*, use_dense_mha: bool = True, num_decode_tokens: int = 0):
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        prefill=SimpleNamespace(use_dense_mha=use_dense_mha),
    )


@pytest.mark.parametrize(
    "batch_kind",
    ["short", "threshold_mismatch", "force_mqa", "mla_decode", "capture", "full"],
)
def test_short_prefill_updates_k_cache_before_scoring_decision(
    monkeypatch: pytest.MonkeyPatch,
    batch_kind: str,
):
    slot_mapping = torch.tensor([63, 64, 127, 128, -1])
    mla_num_decode_tokens = 1 if batch_kind == "mla_decode" else 0
    runtime_mode = (
        CUDAGraphMode.FULL if batch_kind == "full" else CUDAGraphMode.PIECEWISE
    )
    should_skip = batch_kind in ("short", "threshold_mismatch")
    num_decodes = int(batch_kind == "threshold_mismatch")
    num_decode_tokens = 3 if batch_kind == "threshold_mismatch" else 0
    num_prefills = 0 if batch_kind == "threshold_mismatch" else 2
    num_prefill_tokens = 0 if batch_kind == "threshold_mismatch" else 5
    if batch_kind == "threshold_mismatch":
        # With MTP=3 the indexer threshold is four. A main MLA backend whose
        # threshold is one (for example FlashMLA under DCP) still routes this
        # three-token extend through dense prefill attention.
        slot_mapping = slot_mapping[:3]
    indexer_metadata = make_indexer_metadata(
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        slot_mapping=slot_mapping,
    )
    if indexer_metadata.num_decodes:
        indexer_metadata.decode = object()
    mla_metadata = make_mla_metadata(
        use_dense_mha=batch_kind != "force_mqa",
        num_decode_tokens=mla_num_decode_tokens,
    )

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                INDEXER_LAYER: indexer_metadata,
                MLA_LAYER: mla_metadata,
            },
            cudagraph_runtime_mode=runtime_mode,
        ),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: batch_kind == "capture",
    )

    def record_cache_update(k, kv_cache, slots, block_size, scale_fmt):
        observed.update(k=k.clone(), slots=slots)

    monkeypatch.setattr(
        sparse_indexer.ops, "indexer_k_quant_and_cache", record_cache_update
    )

    class ScoringReached(Exception):
        pass

    def scoring_trigger():
        if should_skip:
            pytest.fail("short dense-MHA prefill must not enter indexer scoring")
        raise ScoringReached

    def scoring_decode(*args):
        raise ScoringReached

    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", scoring_trigger)
    monkeypatch.setattr(
        sparse_indexer,
        "kv_cache_as_quant_view",
        scoring_decode,
    )

    hidden_states = torch.full((7, 1), float("inf"))
    k = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    topk_indices = torch.full((7, 2048), 17, dtype=torch.int32)

    def run_indexer():
        return sparse_indexer.sparse_attn_indexer(
            hidden_states,
            INDEXER_LAYER,
            torch.empty(1),
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

    # K cache is always updated before the scoring decision.
    torch.testing.assert_close(observed["k"], k[: slot_mapping.numel()])
    assert observed["slots"] is slot_mapping
