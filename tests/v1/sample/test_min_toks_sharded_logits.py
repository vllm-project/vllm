# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MinTokensLogitsProcessor with TP-sharded (vocab-parallel) logits.

The processor receives rank-local logits when the LM head output is sharded;
its stop-token index tensors hold global ids, which must be translated to the
local shard (and out-of-shard ids dropped) instead of being used as-is.

CPU-only: no model weights, no distributed init — the TP rank is simulated by
monkeypatching ``get_tensor_model_parallel_rank``.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate

VOCAB = 100
# pad_vocab_size(100, 64) = 128: the LM head shards the *padded* vocab,
# so each rank owns 32 columns at TP=4.
TP = 4
LOCAL = ((VOCAB + 63) // 64) * 64 // TP  # 32 columns per rank
STOP_ID = 40  # owned by rank 1 (shard [32, 64), local column 8)
DEVICE = "cpu"


def make_processor(monkeypatch: pytest.MonkeyPatch, rank: int, tp: int = TP):
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_rank", lambda: rank
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: VOCAB),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )
    return MinTokensLogitsProcessor(vllm_config, torch.device(DEVICE), False)


def add_request(processor: MinTokensLogitsProcessor, stop_ids: list[int], **kwargs):
    batch_update = BatchUpdate(
        batch_size=1,
        removed=[],
        added=[
            (
                0,
                SamplingParams(min_tokens=1, stop_token_ids=stop_ids, **kwargs),
                None,
                [],
            )
        ],
        moved=[],
    )
    processor.update_state(batch_update)


def test_sharded_out_of_shard_stop_ids_dropped(monkeypatch):
    """On a rank that does not own the stop id, masking must be a no-op.

    Fails on main with IndexError (global id used as local column).
    """
    processor = make_processor(monkeypatch, rank=0)  # shard [0, 25), STOP_ID=30 outside
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, LOCAL)
    expected = logits.clone()
    out = processor.apply(logits)
    assert torch.equal(out, expected)


def test_sharded_owned_stop_id_masked(monkeypatch):
    """On the owning rank, the stop id must be masked at its local column."""
    processor = make_processor(monkeypatch, rank=1)  # shard [25, 50), local col 5
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, LOCAL)
    out = processor.apply(logits)
    assert out[0, STOP_ID - LOCAL].item() == float("-inf")
    # everything else untouched
    mask = torch.ones(LOCAL, dtype=torch.bool)
    mask[STOP_ID - LOCAL] = False
    assert torch.equal(out[0, mask], logits[0, mask])


def test_unsharded_path_unchanged(monkeypatch):
    """tp=1 keeps the identity mapping: byte-identical to a plain index_put_."""
    processor = make_processor(monkeypatch, rank=0, tp=1)
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, VOCAB)
    expected = logits.clone()
    expected[0, STOP_ID] = float("-inf")
    assert torch.equal(processor.apply(logits), expected)


def test_sharded_restore_path(monkeypatch):
    """Structured-output restore semantics hold per shard: when masking would
    leave the whole (local) row at -inf, the stop-token logits are restored."""
    processor = make_processor(monkeypatch, rank=1)
    add_request(
        processor,
        [STOP_ID],
        structured_outputs=StructuredOutputsParams(json={"type": "object"}),
    )
    # Start from an all -inf row: masking STOP_ID would leave the whole local
    # row at -inf, so the restore path must put the original logit back.
    logits = torch.full((1, LOCAL), float("-inf"))
    original = logits[0, STOP_ID - LOCAL].item()
    out = processor.apply(logits)
    assert out[0, STOP_ID - LOCAL].item() == original


def test_sharded_spec_decode_path(monkeypatch):
    """apply_with_spec_decode must also operate in local coordinates."""
    processor = make_processor(monkeypatch, rank=1)
    add_request(processor, [STOP_ID])
    logits = torch.randn(3, LOCAL)  # 3 draft rows for 1 request
    out = processor.apply_with_spec_decode(logits, num_draft_tokens=[3])
    assert out[0, STOP_ID - LOCAL].item() == float("-inf")
    assert torch.equal(out[1:], logits[1:])


# ---------------------------------------------------------------------------
# Full-vocabulary logits with TP > 1: the default LM-head path all-gathers
# (and trims to the original vocab) before sampling, so the processor must
# keep masking in global coordinates even when TP > 1. Regression tests for
# the case where TP size alone was used to decide the coordinate system.
# ---------------------------------------------------------------------------


def test_full_vocab_logits_masked_with_tp_gt_1(monkeypatch):
    """TP > 1 + full-vocabulary logits must mask in global coordinates.

    Regression: with the coordinate system inferred from TP size alone,
    rank 0 (shard [0, 32)) dropped the global id 40 even though the
    gathered logits contain the full vocabulary, so the stop token was
    never masked.
    """
    processor = make_processor(monkeypatch, rank=0)
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, VOCAB)
    logits[0, STOP_ID] = 10.0
    out = processor.apply(logits)
    assert out[0, STOP_ID].item() == float("-inf")
    assert out.argmax(-1).item() != STOP_ID
    mask = torch.ones(VOCAB, dtype=torch.bool)
    mask[STOP_ID] = False
    assert torch.equal(out[0, mask], logits[0, mask])


def test_full_vocab_logits_masked_on_any_rank(monkeypatch):
    """With full-vocabulary logits the TP rank must not affect masking."""
    processor = make_processor(monkeypatch, rank=3)  # shard [96, 128)
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, VOCAB)
    out = processor.apply(logits)
    assert out[0, STOP_ID].item() == float("-inf")


def test_full_vocab_restore_path_with_tp_gt_1(monkeypatch):
    """Structured-output restore semantics hold in global coordinates."""
    processor = make_processor(monkeypatch, rank=0)
    add_request(
        processor,
        [STOP_ID],
        structured_outputs=StructuredOutputsParams(json={"type": "object"}),
    )
    logits = torch.full((1, VOCAB), float("-inf"))
    out = processor.apply(logits)
    assert out[0, STOP_ID].item() == float("-inf")


def test_full_vocab_spec_decode_path_with_tp_gt_1(monkeypatch):
    """apply_with_spec_decode masks in global coordinates for full logits."""
    processor = make_processor(monkeypatch, rank=0)
    add_request(processor, [STOP_ID])
    logits = torch.randn(3, VOCAB)
    out = processor.apply_with_spec_decode(logits, num_draft_tokens=[3])
    assert out[0, STOP_ID].item() == float("-inf")
    assert torch.equal(out[1:], logits[1:])


# ---------------------------------------------------------------------------
# Constructor contract: the processor must be constructible from a bare
# VllmConfig (model_config/parallel_config=None, as done by some existing
# tests); without a vocab size it falls back to global-only indexing.
# ---------------------------------------------------------------------------


def test_constructs_without_model_config():
    """A config without model_config/parallel_config falls back to global-only
    indexing (same contract as constructing with a bare ``VllmConfig()``,
    whose model_config and parallel_config are both None)."""
    vllm_config = SimpleNamespace(model_config=None, parallel_config=None)
    processor = MinTokensLogitsProcessor(vllm_config, torch.device(DEVICE), False)
    add_request(processor, [STOP_ID])
    logits = torch.randn(1, VOCAB)
    expected = logits.clone()
    expected[0, STOP_ID] = float("-inf")
    assert torch.equal(processor.apply(logits), expected)
