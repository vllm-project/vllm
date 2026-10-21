# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash skips the draft forward on steps scheduled with K=0, but still stores
the context K/V so the draft has the whole context when K rises again."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

NUM_REQS, NUM_TARGET_TOKENS, NUM_STEPS = 2, 6, 3


class _DraftForward(Exception):
    pass


def _speculator(cls, monkeypatch, dp_size: int = 1):
    spec = cls.__new__(cls)
    for name in (
        "input_buffers",
        "block_tables",
        "context_positions",
        "_context_slot_mappings",
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
        "temperature",
        "seeds",
        "dcp_size",
        "parallel_drafting_token_id",
        "max_num_reqs",
        "max_num_tokens",
        "sample_from_anchor",
        "_store_hidden_states",
        "prepare_context_anchor",
        "_precompute_context_kv",
        "_generate_draft",
        "query_cudagraph_manager",
        "dp_rank",
    ):
        setattr(spec, name, MagicMock())
    spec.num_query_per_req = NUM_STEPS
    spec.num_speculative_steps = NUM_STEPS
    spec.max_model_len = 1024
    spec.pcp_manager = None
    spec.draft_kv_cache_group_id = 0
    spec.draft_kv_cache_group_ids = [0]
    spec.dp_size = dp_size
    spec.draft_tokens = torch.zeros(8, NUM_STEPS, dtype=torch.int64)
    monkeypatch.setattr(dflash, "prepare_dflash_inputs", MagicMock())

    def draft_forward(*args, **kwargs):
        raise _DraftForward

    monkeypatch.setattr(dflash, "dispatch_cg_and_sync_dp", draft_forward)
    return spec


def _propose(spec, **kwargs):
    input_batch = SimpleNamespace(
        num_reqs=NUM_REQS,
        num_tokens=NUM_TARGET_TOKENS,
        seq_lens_cpu_upper_bound=torch.tensor([10, 12]),
        query_start_loc=MagicMock(),
    )
    hidden = torch.zeros(NUM_TARGET_TOKENS, 4)
    num = torch.zeros(NUM_REQS, dtype=torch.int32)
    return spec.propose(
        input_batch, {}, {}, hidden, None, num, num, num, num, num, num, **kwargs
    )


@pytest.mark.parametrize("cls", [DFlashSpeculator, DSparkSpeculator])
def test_k0_step_stores_context_and_skips_the_draft(cls, monkeypatch):
    spec = _speculator(cls, monkeypatch)

    draft_tokens = _propose(spec, num_speculative_tokens=0)

    assert draft_tokens.shape == (NUM_REQS, NUM_STEPS)
    spec._store_hidden_states.assert_called_once()
    spec.prepare_context_anchor.assert_called_once()
    spec._precompute_context_kv.assert_called_once_with(0, NUM_TARGET_TOKENS)
    spec._generate_draft.assert_not_called()


@pytest.mark.parametrize(
    ("k", "dp_size", "dummy_run", "is_profile"),
    [
        (None, 1, False, False),  # no per-step schedule
        (1, 1, False, False),
        (0, 2, False, False),  # DP ranks must run the synchronized step
        (0, 1, True, False),
        (0, 1, False, True),
    ],
)
def test_draft_runs_unless_a_real_k0_step_without_dp(
    monkeypatch, k, dp_size, dummy_run, is_profile
):
    spec = _speculator(DFlashSpeculator, monkeypatch, dp_size=dp_size)

    with pytest.raises(_DraftForward):
        _propose(
            spec,
            num_speculative_tokens=k,
            dummy_run=dummy_run,
            is_profile=is_profile,
        )
