# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft seed positions per RoPE flavor, and the 1D slot coordinate.

The both-XD-RoPE config allocates ``xdrope_positions`` instead of the 1D
``positions`` buffer, so the draft loop's seed must read from it (#54555).
And because XD-RoPE decode positions are the absolute token index on every
dim, the slot coordinate is dim 0 of that buffer, the same extraction the
M-RoPE branch uses.
"""

import torch

import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def _proposer(uses_mrope: bool, uses_xdrope_dim: int, draft_xdrope_dim: int):
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.uses_mrope = uses_mrope
    proposer.uses_xdrope_dim = uses_xdrope_dim
    proposer.draft_uses_xdrope_dim = draft_xdrope_dim
    proposer.mrope_positions = (
        torch.zeros((3, 64), dtype=torch.int64) if uses_mrope else None
    )
    proposer.xdrope_positions = (
        torch.zeros((uses_xdrope_dim, 64), dtype=torch.int64)
        if uses_xdrope_dim > 0 and draft_xdrope_dim > 0
        else None
    )
    proposer.positions = (
        None
        if (uses_mrope or (uses_xdrope_dim > 0 and draft_xdrope_dim > 0))
        else torch.zeros(64, dtype=torch.int64)
    )
    return proposer


def test_seed_reads_xdrope_buffer_for_both_xdrope() -> None:
    proposer = _proposer(uses_mrope=False, uses_xdrope_dim=4, draft_xdrope_dim=4)
    proposer.xdrope_positions[0] = torch.arange(64)
    idx = torch.tensor([3, 7])
    seeded = proposer._draft_seed_positions(idx)
    assert torch.equal(seeded[0], torch.tensor([3, 7]))
    assert proposer.positions is None  # the 1D buffer must not be touched


def test_seed_reads_mrope_buffer_for_mrope() -> None:
    proposer = _proposer(uses_mrope=True, uses_xdrope_dim=0, draft_xdrope_dim=0)
    proposer.mrope_positions[0] = torch.arange(64)
    idx = torch.tensor([3, 7])
    seeded = proposer._draft_seed_positions(idx)
    assert torch.equal(seeded[0], torch.tensor([3, 7]))


def test_seed_reads_1d_buffer_for_plain() -> None:
    proposer = _proposer(uses_mrope=False, uses_xdrope_dim=0, draft_xdrope_dim=0)
    proposer.positions = torch.arange(64)
    idx = torch.tensor([3, 7])
    seeded = proposer._draft_seed_positions(idx)
    assert torch.equal(seeded, torch.tensor([3, 7]))


def test_multidim_flag_covers_both_rope_flavors() -> None:
    assert _proposer(True, 0, 0)._uses_multidim_positions() is True
    assert _proposer(False, 4, 4)._uses_multidim_positions() is True
    assert _proposer(False, 0, 0)._uses_multidim_positions() is False
    # Target xdrope with a non-xdrope draft falls back to the 1D path.
    assert _proposer(False, 4, 0)._uses_multidim_positions() is False


def test_slot_coordinate_is_dim0_for_both_xdrope(monkeypatch) -> None:
    """`_update_positions_dependent_metadata` feeds dim 0 to the slot kernel."""
    proposer = _proposer(uses_mrope=False, uses_xdrope_dim=4, draft_xdrope_dim=4)
    proposer.max_model_len = 64
    proposer._slot_mapping_buffer = torch.zeros(2, dtype=torch.int64)
    proposer.xdrope_positions[:, 40] = torch.tensor([40, 40, 40, 40])

    captured = {}
    monkeypatch.setattr(
        llm_base_proposer,
        "eagle_step_update_slot_mapping_and_metadata",
        lambda **kwargs: captured.update(kwargs),
    )

    class _Meta:
        block_table_tensor = torch.zeros((2, 4), dtype=torch.int64)
        seq_lens = torch.tensor([41, 41])
        slot_mapping = None
        max_seq_len = 41
        _seq_lens_cpu = None
        _num_computed_tokens_cpu = None
        seq_lens_cpu_upper_bound = None

    positions = proposer.xdrope_positions[:, 40].reshape(4, 1)
    proposer._update_positions_dependent_metadata(
        positions, _Meta(), batch_size=1, input_batch_size=1, block_size=16
    )
    assert torch.equal(captured["positions_1d"], torch.tensor([40]))
