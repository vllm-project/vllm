# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Design C: the Qwen3.5 MTP draft must run as a standalone (first==last) draft.

Under target PP>1 the drafter executes on the *last* global PP rank, where
`get_pp_group().is_first_rank == False`. Without a standalone flag the MTP
forward takes the `else` branch and demands `intermediate_tensors` that the
proposer never supplies (brick 60). With `draft_pipeline_parallel_size == 1`
(Design C) the forward must behave as first==last: embed -> fc -> layer -> norm,
never reading/returning IntermediateTensors.

CPU-only: stubs the submodules and mocks the PP group to exercise the branch
routing without constructing the full model (validated end-to-end at E3).
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

HIDDEN = 4


def _stub_predictor(standalone: bool) -> Qwen3_5MultiTokenPredictor:
    """A minimally-stubbed predictor that exercises forward()'s branch routing
    without the heavy real submodules (attention backends, weights, etc.)."""
    m = object.__new__(Qwen3_5MultiTokenPredictor)
    torch.nn.Module.__init__(m)
    m.standalone_draft = standalone
    m.num_mtp_layers = 1
    m.embed_tokens = lambda ids: torch.zeros(ids.shape[0], HIDDEN)
    m.pre_fc_norm_embedding = lambda x: x
    m.pre_fc_norm_hidden = lambda x: x
    # real fc maps [2*hidden] -> [hidden]; emulate by halving the last dim.
    m.fc = lambda x: x[..., :HIDDEN]
    m.layers = [lambda positions, hidden_states, residual: (hidden_states, residual)]
    m.norm = lambda h, r: (h, None)
    return m


def _last_rank_of_pp2():
    """Mock the global PP group as the last rank of a pp=2 group."""
    grp = type("G", (), {"is_first_rank": False, "is_last_rank": True})()
    return patch(
        "vllm.model_executor.models.qwen3_5_mtp.get_pp_group", lambda: grp
    )


def test_standalone_draft_uses_embed_path_on_last_rank():
    """Design C: with the standalone flag, the last-rank draft embeds its own
    input and returns a hidden-state tensor — no intermediate_tensors needed."""
    m = _stub_predictor(standalone=True)
    n = 3
    input_ids = torch.zeros(n, dtype=torch.long)
    positions = torch.zeros(n, dtype=torch.long)
    hidden_states = torch.zeros(n, HIDDEN)

    with _last_rank_of_pp2():
        out = m.forward(input_ids, positions, hidden_states)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (n, HIDDEN)


def test_non_standalone_last_rank_still_requires_intermediate_tensors():
    """Control: without the flag, the last rank of pp=2 (is_first_rank=False)
    takes the else branch and asserts on the missing intermediate_tensors —
    i.e. exactly the broken Design-B-shaped behavior the flag fixes."""
    m = _stub_predictor(standalone=False)
    n = 3
    input_ids = torch.zeros(n, dtype=torch.long)
    positions = torch.zeros(n, dtype=torch.long)
    hidden_states = torch.zeros(n, HIDDEN)

    with _last_rank_of_pp2(), pytest.raises(AssertionError):
        m.forward(input_ids, positions, hidden_states)
