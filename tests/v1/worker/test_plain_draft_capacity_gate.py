# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity gate for independent draft_model speculation on Model Runner V2.

An independent draft keeps its own max_model_len. A batch whose seq len plus
the drafter query window exceeds that limit must skip the draft forward.
Resizing the draft rotary cache to the target length is the EAGLE/DFlash
path, not this one.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
    StandaloneARSpeculator,
)

pytestmark = pytest.mark.cpu_test

_DRAFT_LIMIT = 100
_NUM_SPEC = 3
_TARGET_MAX_MODEL_LEN = 4096


def _batch(seq_lens: list[int], *, num_reqs: int | None = None) -> SimpleNamespace:
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(seq_lens) if num_reqs is None else num_reqs,
        seq_lens_cpu_upper_bound=seq,
        idx_mapping=torch.arange(len(seq_lens), dtype=torch.int32),
    )


def _make_spec(
    *,
    draft_limit: int = _DRAFT_LIMIT,
    num_speculative_steps: int = _NUM_SPEC,
    dp_size: int = 1,
) -> SimpleNamespace:
    spec = SimpleNamespace(
        model=object(),
        # Target length. The gate must not treat this as the draft capacity.
        max_model_len=_TARGET_MAX_MODEL_LEN,
        dp_size=dp_size,
        num_speculative_steps=num_speculative_steps,
        effective_drafter_max_model_len=draft_limit,
        draft_tokens=torch.full((8, num_speculative_steps), 9, dtype=torch.int64),
        prefill_calls=0,
        decode_calls=0,
    )
    spec._input_fits_in_drafter = MethodType(
        StandaloneARSpeculator._input_fits_in_drafter, spec
    )
    spec._input_fits_in_drafter_across_dp = MethodType(
        StandaloneARSpeculator._input_fits_in_drafter_across_dp, spec
    )

    def _copy_request_inputs(*args, **kwargs):
        return None

    def _prefill(*args, **kwargs):
        spec.prefill_calls += 1

    def _multi_step_decode(*args, **kwargs):
        spec.decode_calls += 1

    spec._copy_request_inputs = _copy_request_inputs
    spec._prefill = _prefill
    spec._multi_step_decode = _multi_step_decode
    return spec


def _propose(spec: SimpleNamespace, batch: SimpleNamespace, **kwargs) -> torch.Tensor:
    num_reqs = batch.num_reqs
    return StandaloneARSpeculator.propose(
        spec,
        batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.empty(0),
        aux_hidden_states=None,
        num_sampled=torch.zeros(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=torch.zeros(8, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, 8, dtype=torch.int32),
        temperature=torch.zeros(8),
        seeds=torch.zeros(8, dtype=torch.int64),
        **kwargs,
    )


def test_input_fits_in_drafter_boundary():
    # Standalone AR drafters reserve num_spec query tokens, not the DFlash bonus.
    # 97 + 3 == 100 fits; 98 + 3 == 101 does not. The check is batch-wide.
    spec = SimpleNamespace(
        num_speculative_steps=_NUM_SPEC,
        effective_drafter_max_model_len=_DRAFT_LIMIT,
        max_model_len=_TARGET_MAX_MODEL_LEN,
    )
    fits = StandaloneARSpeculator._input_fits_in_drafter
    assert fits(spec, _batch([97]))
    assert fits(spec, _batch([10, 97]))
    assert not fits(spec, _batch([98]))
    assert not fits(spec, _batch([10, 98]))
    # Padded tail must not count. A stale huge value past num_reqs is ignored.
    assert fits(spec, _batch([97, 10_000], num_reqs=1))


@pytest.mark.parametrize("dummy_run", [False, True])
def test_propose_skips_when_batch_exceeds_draft_capacity(dummy_run: bool):
    spec = _make_spec()
    out = _propose(spec, _batch([10, 98]), dummy_run=dummy_run)

    assert spec.prefill_calls == 0
    assert spec.decode_calls == 0
    assert out.shape == (2, _NUM_SPEC)
    assert torch.count_nonzero(out) == 0
    # Rows outside the batch keep their previous contents.
    assert torch.all(spec.draft_tokens[2:] == 9)


@pytest.mark.parametrize("dummy_run", [False, True])
def test_propose_drafts_on_the_draft_limit_boundary(dummy_run: bool):
    spec = _make_spec()
    out = _propose(spec, _batch([10, 97]), dummy_run=dummy_run)

    assert spec.prefill_calls == 1
    assert spec.decode_calls == 1
    assert out.shape == (2, _NUM_SPEC)
    assert torch.all(out == 9)


def test_dp_disagreement_skips_on_every_rank(monkeypatch: pytest.MonkeyPatch):
    # This rank fits (97 + 3 == 100). Another rank does not, so the MIN
    # reduction forces every rank to skip rather than split a collective.
    spec = _make_spec(dp_size=2)

    def _all_reduce(flag, op=None, group=None):
        flag.fill_(0)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    out = _propose(spec, _batch([97]))
    assert spec.prefill_calls == 0
    assert torch.count_nonzero(out) == 0
