# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.utils import (
    capture_eagle_vocab_map,
    validate_eagle_vocab_map,
)

pytestmark = pytest.mark.skip_global_cleanup


def _mask_for(target_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[target_ids] = True
    return t2d


def test_capture_records_d2t_and_skips_t2d():
    vocab_map: dict[str, torch.Tensor] = {}
    d2t = torch.zeros(2, dtype=torch.int64)
    t2d = torch.zeros(4, dtype=torch.bool)

    assert not capture_eagle_vocab_map("d2t", d2t, vocab_map)
    assert capture_eagle_vocab_map("model.t2d", t2d, vocab_map)
    assert vocab_map["d2t"] is d2t
    assert vocab_map["t2d"] is t2d


def test_matching_d2t_and_t2d_pass():
    draft_ids = torch.arange(8)
    target_ids = draft_ids + 20
    vocab_map = {
        "d2t": target_ids - draft_ids,
        "t2d": _mask_for(target_ids, 40),
    }
    validate_eagle_vocab_map(vocab_map)


def test_d2t_alone_is_not_checked():
    validate_eagle_vocab_map({"d2t": torch.tensor([1, 2])})


def test_bf16_round_trip_d2t_is_rejected():
    draft_ids = torch.arange(32)
    # Offsets past bf16's exact integer range, so the round trip moves ids.
    target_ids = draft_ids * 1000 + 3
    d2t = target_ids - draft_ids
    rounded = d2t.to(torch.bfloat16).to(torch.int64)
    assert not torch.equal(rounded, d2t)

    vocab_map = {
        "d2t": rounded,
        "t2d": _mask_for(target_ids, int(target_ids.max()) + 1),
    }
    with pytest.raises(ValueError, match="d2t does not match t2d"):
        validate_eagle_vocab_map(vocab_map)


def test_one_displaced_draft_id_is_rejected():
    draft_ids = torch.arange(4)
    target_ids = draft_ids + 10
    d2t = target_ids - draft_ids
    # Land on an in-range target id that t2d does not mark.
    d2t[1] += 5
    vocab_map = {"d2t": d2t, "t2d": _mask_for(target_ids, 20)}
    with pytest.raises(ValueError, match="1 of 4"):
        validate_eagle_vocab_map(vocab_map)


def test_extra_t2d_marks_are_allowed():
    draft_ids = torch.arange(4)
    target_ids = draft_ids + 10
    t2d = _mask_for(target_ids, 20)
    t2d[0] = True
    validate_eagle_vocab_map(
        {"d2t": target_ids - draft_ids, "t2d": t2d},
    )


def test_duplicate_target_is_rejected():
    # draft ids 0 and 1 both land on target 10.
    d2t = torch.tensor([10, 9, 10, 10])
    targets = torch.arange(4) + d2t
    vocab_map = {"d2t": d2t, "t2d": _mask_for(torch.unique(targets), 20)}
    with pytest.raises(ValueError, match="shares a target"):
        validate_eagle_vocab_map(vocab_map)


def test_non_bool_t2d_is_skipped():
    validate_eagle_vocab_map(
        {
            "d2t": torch.tensor([1, 2]),
            "t2d": torch.ones(4, dtype=torch.int64),
        }
    )
