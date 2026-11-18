# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpeculatorOutput structural invariants."""

import pytest
import torch

from vllm.v1.spec_decode.proposal import SpeculatorOutput


def test_batch_size_mismatch_rejected():
    with pytest.raises(ValueError, match="batch size mismatch"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3, dtype=torch.long),
            valid_lengths=torch.zeros(3, dtype=torch.int32),
        )


def test_wrong_rank_rejected():
    with pytest.raises(ValueError, match="2-D"):
        SpeculatorOutput(
            token_ids=torch.zeros(6, dtype=torch.long),
            valid_lengths=torch.zeros(6, dtype=torch.int32),
        )


def test_non_integer_dtype_rejected():
    with pytest.raises(ValueError, match="integer dtype"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3),
            valid_lengths=torch.zeros(2, dtype=torch.int32),
        )
