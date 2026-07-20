# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm.v1.structured_output import utils


@patch("vllm.v1.structured_output.utils.xgr")
def test_apply_grammar_bitmask_buffer_reuse(mock_xgr):
    """
    Verify that the _STAGING_BUFFER is allocated once and reused across steps
    to prevent performance regression (Issue #49013).
    """
    # 1. Ensure clean state
    utils._STAGING_BUFFER = None

    # 2. Setup mock objects matching expected vLLM structures
    mock_scheduler_output = MagicMock()
    mock_scheduler_output.scheduled_spec_decode_tokens = {}

    mock_grammar_output = MagicMock()
    # Mocking a small bitmask (e.g., 1 request, 100 bitmask width)
    mock_grammar_output.grammar_bitmask = np.zeros((1, 100), dtype=np.int32)
    mock_grammar_output.structured_output_request_ids = ["req_1"]

    mock_input_batch = MagicMock()
    mock_input_batch.req_ids = ["req_1"]

    # Mock logits tensor on CPU
    logits = torch.randn((1, 32000))

    # --- Test Case A: Initial Allocation ---
    utils.apply_grammar_bitmask(
        mock_scheduler_output, mock_grammar_output, mock_input_batch, logits
    )

    assert utils._STAGING_BUFFER is not None
    first_buffer_id = id(utils._STAGING_BUFFER)

    # --- Test Case B: Buffer Reuse (Same shape) ---
    utils.apply_grammar_bitmask(
        mock_scheduler_output, mock_grammar_output, mock_input_batch, logits
    )

    second_buffer_id = id(utils._STAGING_BUFFER)
    assert first_buffer_id == second_buffer_id, (
        "Buffer was reallocated instead of reused!"
    )

    # --- Test Case C: Dynamic Resizing (Larger shape) ---
    # Increase batch size to 2
    logits_large = torch.randn((2, 32000))
    mock_input_batch.req_ids = ["req_1", "req_2"]
    mock_grammar_output.structured_output_request_ids = ["req_1", "req_2"]
    mock_grammar_output.grammar_bitmask = np.zeros(
        (2, 100), dtype=np.int32
    )  # UPDATE: Match mask shape to 2 requests

    utils.apply_grammar_bitmask(
        mock_scheduler_output, mock_grammar_output, mock_input_batch, logits_large
    )

    third_buffer_id = id(utils._STAGING_BUFFER)
    assert third_buffer_id != first_buffer_id, (
        "Buffer should have reallocated for a larger batch size!"
    )
