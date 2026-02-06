# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID


def test_dflash_runtime_mode_defaults_to_shared_eagle():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={})
    )
    assert proposer._get_dflash_runtime_mode_from_config() == "shared_eagle"


def test_dflash_runtime_mode_accepts_block_drafting():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={"runtime_mode": "block_drafting"})
    )
    assert proposer._get_dflash_runtime_mode_from_config() == "block_drafting"


def test_dflash_runtime_mode_rejects_invalid_value():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={"runtime_mode": "invalid"})
    )
    with pytest.raises(ValueError, match="Expected one of"):
        proposer._get_dflash_runtime_mode_from_config()


def test_resolve_mask_token_id_prefers_dflash_config():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vocab_size=1000,
            mask_token_id=3,
            dflash_config={"mask_token_id": 17},
        )
    )
    assert proposer._resolve_mask_token_id_from_config() == 17


def test_resolve_mask_token_id_uses_hf_fallbacks():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vocab_size=1000, pad_token_id=9, dflash_config={})
    )
    assert proposer._resolve_mask_token_id_from_config() == 9


def test_resolve_mask_token_id_raises_out_of_vocab():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vocab_size=10, dflash_config={"mask_token_id": 11})
    )
    with pytest.raises(ValueError, match="out of vocab bounds"):
        proposer._resolve_mask_token_id_from_config()


def test_maybe_combine_target_hidden_states_passthrough_without_hook():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.model = SimpleNamespace()
    target_hidden_states = torch.zeros(2, 4, dtype=torch.float32)

    out = proposer._maybe_combine_target_hidden_states(target_hidden_states)
    assert out is target_hidden_states


def test_maybe_combine_target_hidden_states_with_model_hook():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.hidden_size = 4
    proposer.model = SimpleNamespace(
        combine_hidden_states=lambda hs: hs + 1,
    )
    target_hidden_states = torch.zeros(2, 4, dtype=torch.float32)

    out = proposer._maybe_combine_target_hidden_states(target_hidden_states)
    assert torch.equal(out, torch.ones(2, 4, dtype=torch.float32))


def test_maybe_combine_target_hidden_states_raises_on_hidden_size_mismatch():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.hidden_size = 4
    proposer.model = SimpleNamespace(
        combine_hidden_states=lambda _hs: torch.zeros(2, 3, dtype=torch.float32),
    )
    target_hidden_states = torch.zeros(2, 4, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="combined hidden size mismatch"):
        proposer._maybe_combine_target_hidden_states(target_hidden_states)


def _make_propose_inputs(batch_size: int):
    num_tokens = 2 * batch_size
    return {
        "target_token_ids": torch.zeros(num_tokens, dtype=torch.int32),
        "target_positions": torch.arange(num_tokens, dtype=torch.int64),
        "target_hidden_states": torch.zeros(num_tokens, 4, dtype=torch.float32),
        "next_token_ids": torch.zeros(batch_size, dtype=torch.int32),
        "token_indices_to_sample": None,
        "common_attn_metadata": SimpleNamespace(batch_size=lambda: batch_size),
        "sampling_metadata": SimpleNamespace(),
    }


def test_dflash_proposer_dispatches_to_shared_eagle():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.runtime_mode = "shared_eagle"
    proposer.model = SimpleNamespace()
    proposer.num_speculative_tokens = 2
    inputs = _make_propose_inputs(batch_size=2)
    sentinel = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    with patch.object(
        DFlashProposer, "_propose_shared_eagle", return_value=sentinel
    ) as mock_shared:
        out = proposer.propose(**inputs)

    mock_shared.assert_called_once()
    assert torch.equal(out, sentinel)


def test_dflash_proposer_dispatches_to_block_drafting_for_bs1():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.runtime_mode = "block_drafting"
    proposer.model = SimpleNamespace()
    proposer.num_speculative_tokens = 2
    inputs = _make_propose_inputs(batch_size=1)
    sentinel = torch.tensor([[7, 8]], dtype=torch.int32)

    with patch.object(
        DFlashProposer, "_propose_block_drafting", return_value=sentinel
    ) as mock_block:
        out = proposer.propose(**inputs)

    mock_block.assert_called_once()
    assert torch.equal(out, sentinel)


def test_dflash_proposer_block_drafting_falls_back_to_shared_for_bs_gt1():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.runtime_mode = "block_drafting"
    proposer.model = SimpleNamespace()
    proposer.num_speculative_tokens = 2
    inputs = _make_propose_inputs(batch_size=2)
    sentinel = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    with (
        patch.object(
            DFlashProposer, "_propose_shared_eagle", return_value=sentinel
        ) as mock_shared,
        patch.object(DFlashProposer, "_propose_block_drafting") as mock_block,
    ):
        out = proposer.propose(**inputs)

    mock_block.assert_not_called()
    mock_shared.assert_called_once()
    assert torch.equal(out, sentinel)


def test_dflash_proposer_raises_on_invalid_draft_shape():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.runtime_mode = "shared_eagle"
    proposer.model = SimpleNamespace()
    proposer.num_speculative_tokens = 2
    inputs = _make_propose_inputs(batch_size=2)
    invalid = torch.tensor([[1], [2]], dtype=torch.int32)

    with (
        patch.object(DFlashProposer, "_propose_shared_eagle", return_value=invalid),
        pytest.raises(RuntimeError, match="unexpected draft token shape"),
    ):
        proposer.propose(**inputs)


def test_block_drafting_requires_resolved_mask_token_id():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.mask_token_id = None

    with pytest.raises(ValueError, match="requires a resolved mask token id"):
        proposer._propose_block_drafting(
            target_positions=torch.tensor([0], dtype=torch.int64),
            target_hidden_states=torch.zeros(1, 4, dtype=torch.float32),
            next_token_ids=torch.tensor([1], dtype=torch.int32),
            common_attn_metadata=SimpleNamespace(batch_size=lambda: 1),
        )


def test_block_drafting_rejects_batch_size_gt1():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.mask_token_id = 0

    with pytest.raises(NotImplementedError, match="batch size 1 only"):
        proposer._propose_block_drafting(
            target_positions=torch.tensor([0], dtype=torch.int64),
            target_hidden_states=torch.zeros(1, 4, dtype=torch.float32),
            next_token_ids=torch.tensor([1, 2], dtype=torch.int32),
            common_attn_metadata=SimpleNamespace(batch_size=lambda: 2),
        )


def _make_block_drafting_proposer_for_errors() -> DFlashProposer:
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.mask_token_id = 0
    proposer.num_speculative_tokens = 3
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.indexer_layer_names = []
    return proposer


def test_block_drafting_raises_when_query_positions_exceed_max_model_len():
    proposer = _make_block_drafting_proposer_for_errors()
    proposer.max_model_len = 4

    with pytest.raises(RuntimeError, match="query positions exceed max_model_len"):
        proposer._propose_block_drafting(
            target_positions=torch.tensor([3], dtype=torch.int64),
            target_hidden_states=torch.zeros(1, 4, dtype=torch.float32),
            next_token_ids=torch.tensor([1], dtype=torch.int32),
            common_attn_metadata=SimpleNamespace(batch_size=lambda: 1),
        )


def test_block_drafting_raises_when_block_table_is_too_small():
    proposer = _make_block_drafting_proposer_for_errors()
    proposer.max_model_len = 128
    proposer.runner = object()
    proposer.attn_metadata_builder = None
    builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
    common_attn_metadata = SimpleNamespace(
        batch_size=lambda: 1,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int64),
    )

    with (
        patch.object(proposer, "_get_attention_metadata_builder", return_value=builder),
        pytest.raises(RuntimeError, match="needs more block_table entries"),
    ):
        proposer._propose_block_drafting(
            target_positions=torch.tensor([0], dtype=torch.int64),
            target_hidden_states=torch.zeros(1, 4, dtype=torch.float32),
            next_token_ids=torch.tensor([1], dtype=torch.int32),
            common_attn_metadata=common_attn_metadata,
        )


def test_dflash_get_slot_mapping_pads_with_padding_slot_id():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer._slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)
    proposer.attn_layer_names = ["layer_a", "layer_b"]
    proposer.indexer_layer_names = ["indexer_layer"]

    result = proposer._get_slot_mapping(
        num_tokens=6,
        slot_mapping=torch.tensor([10, 11, 12], dtype=torch.int64),
    )

    expected = torch.tensor(
        [10, 11, 12, PADDING_SLOT_ID, PADDING_SLOT_ID, PADDING_SLOT_ID],
        dtype=torch.int64,
    )
    assert set(result.keys()) == {"layer_a", "layer_b", "indexer_layer"}
    for view in result.values():
        assert torch.equal(view, expected)


def test_dflash_set_inputs_first_pass_bs_gt1_updates_positions_and_indices():
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(uses_mrope=False)
    )
    proposer.input_ids = torch.zeros(16, dtype=torch.int32)
    proposer.positions = torch.zeros(16, dtype=torch.int64)
    proposer.hidden_states = torch.zeros(16, 4, dtype=torch.float32)

    cad = SimpleNamespace(query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32))
    target_token_ids = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32)
    target_positions = torch.tensor([7, 8, 9, 6, 7], dtype=torch.int64)
    target_hidden_states = torch.arange(20, dtype=torch.float32).view(5, 4)
    next_token_ids = torch.tensor([100, 200], dtype=torch.int32)

    num_tokens, token_indices_to_sample, output_cad = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    assert num_tokens == 5
    assert output_cad is cad
    assert torch.equal(token_indices_to_sample, torch.tensor([2, 4], dtype=torch.int32))
    assert torch.equal(
        proposer.input_ids[:5],
        torch.tensor([11, 12, 100, 21, 200], dtype=torch.int32),
    )
    assert torch.equal(proposer.positions[:5], target_positions)
    assert torch.equal(proposer.hidden_states[:5], target_hidden_states)
