# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Import test utilities from v1/core (relative import works in pytest)
import sys
from pathlib import Path

import torch

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.request import RequestStatus

test_dir = Path(__file__).parent / "core"
sys.path.insert(0, str(test_dir))
from utils import create_requests  # noqa: E402


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_get_num_encoder_tokens_no_mask():
    """Test get_num_encoder_tokens returns length when is_embed=None."""
    # Create request with multimodal input without mask
    mm_positions = [[PlaceholderRange(offset=0, length=100)]]
    requests = create_requests(
        num_requests=1,
        num_tokens=10,
        mm_positions=mm_positions,
    )
    request = requests[0]

    # Should return placeholder length (backward compatible)
    assert request.get_num_encoder_tokens(input_id=0) == 100


def test_get_num_encoder_tokens_sparse_mask():
    """Test get_num_encoder_tokens returns embedding count with sparse mask."""
    # Create sparse mask: 100 positions, only 8 embeddings (8% density)
    is_embed = torch.zeros(100, dtype=torch.bool)
    is_embed[:8] = True

    mm_positions = [[PlaceholderRange(offset=0, length=100, is_embed=is_embed)]]
    requests = create_requests(
        num_requests=1,
        num_tokens=10,
        mm_positions=mm_positions,
    )
    request = requests[0]

    # Should return embedding count (8), not placeholder length (100)
    assert request.get_num_encoder_tokens(input_id=0) == 8


def test_get_num_encoder_tokens_dense_mask():
    """Test get_num_encoder_tokens with all-True mask equals length."""
    # Create dense mask: all positions are embeddings
    is_embed = torch.ones(100, dtype=torch.bool)

    mm_positions = [[PlaceholderRange(offset=0, length=100, is_embed=is_embed)]]
    requests = create_requests(
        num_requests=1,
        num_tokens=10,
        mm_positions=mm_positions,
    )
    request = requests[0]

    # Should return embedding count (100), same as length
    assert request.get_num_encoder_tokens(input_id=0) == 100
