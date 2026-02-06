# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.eagle import EagleProposer


def test_dflash_proposer_rejects_batch_size_gt_one():
    proposer = DFlashProposer.__new__(DFlashProposer)
    common_attn_metadata = SimpleNamespace(batch_size=lambda: 2)
    with pytest.raises(NotImplementedError, match="batch size 1 only"):
        proposer.propose(common_attn_metadata=common_attn_metadata)


def test_dflash_proposer_delegates_to_eagle_for_bs1():
    proposer = DFlashProposer.__new__(DFlashProposer)
    common_attn_metadata = SimpleNamespace(batch_size=lambda: 1)
    sentinel = torch.tensor([[1]], dtype=torch.int32)

    with patch.object(EagleProposer, "propose", return_value=sentinel) as mock_propose:
        out = proposer.propose(common_attn_metadata=common_attn_metadata)

    mock_propose.assert_called_once()
    assert torch.equal(out, sentinel)
