# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm.v1.structured_output import utils


@patch("vllm.v1.structured_output.utils.xgr")
def test_apply_grammar_bitmask_buffer_reuse(mock_xgr):
    """
    Verify that the staging buffer is allocated once and reused across steps
    to prevent performance regression, utilizing an instance-level cache (Issue #49013).
    """
    # 1. Initialize an empty local cache dictionary to simulate instance-level storage
    cache: dict = {}

    # 2. Setup mock objects matching expected vLLM structures
    mock_scheduler_output = MagicMock()
    mock_scheduler_output.scheduled_spec_decode_tokens = {}

    mock_grammar_output = MagicMock()
    mock_grammar_output.grammar_bitmask = np.zeros((1, 100), dtype=np.int32)
    mock_grammar_output.structured_output_request_ids = ["req_1"]

    mock_input_batch = MagicMock()
    mock_input_batch.req_ids = ["req_1"]

    logits = torch.randn((1, 32000))

    # --- Test Case A: Initial Allocation ---
    start_alloc = time.perf_counter()
    utils.apply_grammar_bitmask(
        mock_scheduler_output, mock_grammar_output, mock_input_batch, logits, cache
    )
    end_alloc = time.perf_counter()
    alloc_time_ms = (end_alloc - start_alloc) * 1000

    assert cache.get("buffer") is not None
    first_buffer_id = id(cache["buffer"])

    # --- Test Case B: Buffer Reuse (Same shape) ---
    start_reuse = time.perf_counter()
    utils.apply_grammar_bitmask(
        mock_scheduler_output, mock_grammar_output, mock_input_batch, logits, cache
    )
    end_reuse = time.perf_counter()
    reuse_time_ms = (end_reuse - start_reuse) * 1000

    second_buffer_id = id(cache["buffer"])
    assert first_buffer_id == second_buffer_id, (
        "Buffer was reallocated instead of reused!"
    )

    # --- Test Case C: Dynamic Resizing (Larger shape) ---
    logits_large = torch.randn((2, 32000))
    mock_input_batch.req_ids = ["req_1", "req_2"]
    mock_grammar_output.structured_output_request_ids = ["req_1", "req_2"]
    mock_grammar_output.grammar_bitmask = np.zeros((2, 100), dtype=np.int32)

    start_resize = time.perf_counter()
    utils.apply_grammar_bitmask(
        mock_scheduler_output,
        mock_grammar_output,
        mock_input_batch,
        logits_large,
        cache,
    )
    end_resize = time.perf_counter()
    resize_time_ms = (end_resize - start_resize) * 1000

    third_buffer_id = id(cache["buffer"])
    assert third_buffer_id != first_buffer_id, (
        "Buffer should have reallocated for a larger batch size!"
    )

    # --- Benchmark Output ---
    print("\n" + "=" * 50)
    print("🚀 [BENCHMARK RESULTS] Staging Buffer Cache Performance")
    print("=" * 50)
    print(f"Initial Allocation Time : {alloc_time_ms:.4f} ms")
    print(f"Buffer Reuse Time       : {reuse_time_ms:.4f} ms")
    print(f"Dynamic Resizing Time   : {resize_time_ms:.4f} ms")
    print("=" * 50)
