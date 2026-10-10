# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar

from .utils import create_requests, create_scheduler


def test_traversal_past_unschedulable_head():
    """Verify that requests parked in skipped_waiting statuses are not stranded
    behind an unschedulable queue head when nothing is running."""
    BLOCK_SIZE = 16
    NUM_BLOCKS = 10

    scheduler = create_scheduler(
        max_num_seqs=100,
        max_num_batched_tokens=1000,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
    )

    # Priority policy ensures predictable queue order
    scheduler.vllm_config.scheduler_config.policy = "priority"

    # 1. Parked request (low priority)
    req_grammar = create_requests(1, num_tokens=BLOCK_SIZE, req_ids=["req_grammar"])[0]
    req_grammar.priority = 1
    req_grammar.structured_output_request = Mock()
    req_grammar.structured_output_request.grammar = None  # Compiling
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.skipped_waiting.add_request(req_grammar)
    scheduler.requests[req_grammar.request_id] = req_grammar

    # 2. Blocker request (high priority, too large to fit in 10 blocks)
    req_blocker = create_requests(
        1, num_tokens=BLOCK_SIZE * 15, req_ids=["req_blocker"]
    )[0]
    req_blocker.priority = 0
    req_blocker.status = RequestStatus.WAITING
    scheduler.waiting.add_request(req_blocker)

    # First schedule(): req_blocker is head, but too large (15 > 10 blocks).
    # Since nothing is running, scheduler MUST continue scanning and check req_grammar.
    # req_grammar is checked, but grammar is still None, so it remains parked.
    scheduler.schedule()
    assert len(scheduler.running) == 0
    assert req_grammar.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert req_blocker.status == RequestStatus.WAITING

    # Now simulate grammar compilation finishing
    req_grammar.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)

    # Second schedule(): req_blocker is still head and too large.
    # Scheduler MUST scan past it, see req_grammar, promote it
    # to WAITING, and schedule it!
    scheduler.schedule()
    assert req_grammar.status == RequestStatus.RUNNING
    assert req_blocker.status == RequestStatus.WAITING
    assert req_grammar in scheduler.running


@patch("time.time")
def test_blocked_waiting_timeout(mock_time):
    """Verify that requests parked in skipped_waiting for too long are aborted."""
    mock_time.return_value = 0.0

    scheduler = create_scheduler(
        max_num_seqs=100,
        max_num_batched_tokens=1000,
        block_size=16,
        num_blocks=10,
    )

    # Create a parked request
    req_grammar = create_requests(1, num_tokens=16, req_ids=["req_grammar"])[0]
    req_grammar.structured_output_request = Mock()
    req_grammar.structured_output_request.grammar = None
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.skipped_waiting.add_request(req_grammar)
    scheduler.requests[req_grammar.request_id] = req_grammar

    # First schedule(): marks blocked_since to 0.0
    scheduler.schedule()
    assert getattr(req_grammar, "blocked_since", None) == 0.0
    assert req_grammar.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    # Fast forward just below timeout
    mock_time.return_value = 59.9
    scheduler.schedule()
    assert req_grammar.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    # Fast forward past timeout
    mock_time.return_value = 60.1
    scheduler.schedule()

    # Request should be aborted
    assert req_grammar.status == RequestStatus.FINISHED_ABORTED
    assert len(scheduler.skipped_waiting) == 0


@patch("time.time")
def test_blocked_waiting_timeout_reset_and_exclusion(mock_time):
    """Verify WAITING_FOR_STREAMING_REQ exclusion, promotion-before-timeout,
    and resetting blocked_since on promotion."""
    mock_time.return_value = 0.0

    scheduler = create_scheduler(
        max_num_seqs=100,
        max_num_batched_tokens=1000,
        block_size=16,
        num_blocks=10,
    )

    # 1. Test WAITING_FOR_STREAMING_REQ has no timeout
    req_streaming = create_requests(1, num_tokens=16, req_ids=["req_streaming"])[0]
    req_streaming.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.skipped_waiting.add_request(req_streaming)
    scheduler.requests[req_streaming.request_id] = req_streaming

    scheduler.schedule()
    # blocked_since should be explicitly cleared or not set
    assert getattr(req_streaming, "blocked_since", None) is None

    mock_time.return_value = 100.0
    scheduler.schedule()
    # Still not aborted
    assert req_streaming.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    # 2. Test promotion before timeout (promoted at t=60.1)
    req_grammar = create_requests(1, num_tokens=16, req_ids=["req_grammar"])[0]
    req_grammar.structured_output_request = Mock()
    req_grammar.structured_output_request.grammar = None
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.skipped_waiting.add_request(req_grammar)
    scheduler.requests[req_grammar.request_id] = req_grammar

    mock_time.return_value = 100.0
    scheduler.schedule()
    assert req_grammar.blocked_since == 100.0

    # Advance past 60s, but make grammar ready
    mock_time.return_value = 160.1
    req_grammar.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    scheduler.schedule()

    # Promotion succeeded: status -> RUNNING (or WAITING if capacity limits it)
    assert req_grammar.status == RequestStatus.RUNNING
    assert req_grammar.blocked_since is None
