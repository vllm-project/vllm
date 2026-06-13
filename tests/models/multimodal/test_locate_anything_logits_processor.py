# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.locate_anything import (
    LocateAnythingSlowGrammarLogitsProcessor,
    LocateAnythingSlowLogitsProcessor,
    LocateAnythingTokenIds,
)
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.locate_anything import LocateAnythingConfig

# CPU-only unit tests; core_model puts them in the per-PR CI run that
# sweeps tests/models/multimodal with `-m core_model`.
pytestmark = pytest.mark.core_model


def _token_ids() -> LocateAnythingTokenIds:
    return LocateAnythingTokenIds.from_config(LocateAnythingConfig())


def _logits(vocab_size: int = 152681) -> torch.Tensor:
    return torch.zeros(vocab_size)


def _finite_ids(logits: torch.Tensor) -> set[int]:
    return set(torch.isfinite(logits).nonzero().flatten().tolist())


def test_box_start_allows_none_or_first_coordinate():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    out = proc([ids.box_start], _logits())

    allowed = _finite_ids(out)
    assert ids.none in allowed
    assert ids.coord_start in allowed
    assert ids.coord_end in allowed
    assert ids.box_end not in allowed


def test_none_box_must_close_immediately():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    out = proc([ids.box_start, ids.none], _logits())

    assert _finite_ids(out) == {ids.box_end}


def test_single_coordinate_cannot_close():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    # After exactly one coordinate the grammar must force a second coordinate:
    # neither <box_end> nor <none> may be emitted. Guards `num_coords < 2`.
    out = proc([ids.box_start, ids.coord_start + 1], _logits())

    allowed = _finite_ids(out)
    assert ids.coord_start in allowed
    assert ids.coord_end in allowed
    assert ids.box_end not in allowed
    assert ids.none not in allowed


def test_three_coordinates_cannot_close():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    # After three coordinates the grammar must force a fourth: a 3-coord box is
    # invalid (only 2-coord points or 4-coord boxes close). Guards `num_coords
    # < 4`.
    out = proc(
        [
            ids.box_start,
            ids.coord_start + 1,
            ids.coord_start + 2,
            ids.coord_start + 3,
        ],
        _logits(),
    )

    allowed = _finite_ids(out)
    assert ids.coord_start in allowed
    assert ids.coord_end in allowed
    assert ids.box_end not in allowed
    assert ids.none not in allowed


def test_point_can_close_after_two_coordinates():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    out = proc(
        [ids.box_start, ids.coord_start + 1, ids.coord_start + 2],
        _logits(),
    )

    allowed = _finite_ids(out)
    assert ids.box_end in allowed
    assert ids.coord_start in allowed
    assert ids.coord_end in allowed
    assert ids.none not in allowed


def test_bbox_must_close_after_four_coordinates():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)

    out = proc(
        [
            ids.box_start,
            ids.coord_start + 1,
            ids.coord_start + 2,
            ids.coord_start + 3,
            ids.coord_start + 4,
        ],
        _logits(),
    )

    assert _finite_ids(out) == {ids.box_end}


def test_outside_box_leaves_logits_unchanged():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)
    logits = _logits()

    out = proc([ids.ref_start, ids.ref_end], logits.clone())

    assert torch.equal(out, logits)


def test_closed_box_deactivates_grammar():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)
    logits = _logits()

    # Once a box is closed (a <box_end> appears after the last <box_start>) the
    # grammar must deactivate and leave logits untouched, rather than re-arming.
    out = proc(
        [ids.box_start, ids.coord_start + 1, ids.coord_start + 2, ids.box_end],
        logits.clone(),
    )

    assert torch.equal(out, logits)


def test_allowed_positions_preserve_original_logits():
    ids = _token_ids()
    proc = LocateAnythingSlowLogitsProcessor(ids)
    torch.manual_seed(0)
    logits = torch.randn(152681)

    # A none-box must close immediately: only <box_end> is allowed. Its logit
    # must be the *original* value (not zeroed) — a zero-logits fixture would
    # not catch a regression that overwrites allowed values with a constant.
    out = proc([ids.box_start, ids.none], logits)

    assert _finite_ids(out) == {ids.box_end}
    assert out[ids.box_end] == logits[ids.box_end]


def test_sampling_params_must_preserve_special_tokens():
    with pytest.raises(ValueError, match="skip_special_tokens=False"):
        LocateAnythingSlowGrammarLogitsProcessor.validate_params(SamplingParams())
