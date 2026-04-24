# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.cohere.utils import handle_thinking_tokens


class DummyRequest:
    def __init__(self, request_id, output_token_ids):
        self.request_id = request_id
        self.output_token_ids = output_token_ids


def make_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    return logger


def test_start_thinking_token_sets_correct_index():
    logger = make_logger()

    request = DummyRequest(
        request_id="req1",
        output_token_ids=[10, 11, 12, 255010, 5],
    )

    new_token_ids = [12, 255010, 5]  # START_THINKING at index 1
    start_thinking_token_id = 255010
    end_thinking_token_id = 255011

    requests_to_start_thinking_idx: dict[str, int | None] = {}
    requests_with_remaining_budget: dict[str, int] = {}

    handle_thinking_tokens(
        request=request,
        new_token_ids=new_token_ids,
        start_thinking_token_id=start_thinking_token_id,
        end_thinking_token_id=end_thinking_token_id,
        requests_to_start_thinking_idx=requests_to_start_thinking_idx,
        requests_with_remaining_budget=requests_with_remaining_budget,
        logger=logger,
    )

    # previous_length = 5 - 3 = 2
    # start index = 1 + 1 + 2 = 4
    assert requests_to_start_thinking_idx["req1"] == 4

    logger.info.assert_called_once()


def test_end_thinking_token_cleans_up_state():
    logger = make_logger()

    request = DummyRequest(
        request_id="req2",
        output_token_ids=[1, 2, 3, 4, 5, 6, 7],
    )

    new_token_ids = [77, 255011]  # END_THINKING at index 1
    start_thinking_token_id = 255010
    end_thinking_token_id = 255011

    requests_to_start_thinking_idx: dict[str, int | None] = {"req2": None}
    requests_with_remaining_budget: dict[str, int] = {}

    handle_thinking_tokens(
        request=request,
        new_token_ids=new_token_ids,
        start_thinking_token_id=start_thinking_token_id,
        end_thinking_token_id=end_thinking_token_id,
        requests_to_start_thinking_idx=requests_to_start_thinking_idx,
        requests_with_remaining_budget=requests_with_remaining_budget,
        logger=logger,
    )

    assert "req2" not in requests_to_start_thinking_idx
    assert "req2" not in requests_with_remaining_budget

    logger.warning.assert_called_once()


def test_no_thinking_tokens_no_side_effects():
    logger = make_logger()

    request = DummyRequest(
        request_id="req4",
        output_token_ids=[1, 2, 3],
    )

    new_token_ids = [8, 9]
    start_thinking_token_id = 255010
    end_thinking_token_id = 255011

    requests_to_start_thinking_idx: dict[str, int | None] = {}
    requests_with_remaining_budget: dict[str, int] = {}

    handle_thinking_tokens(
        request=request,
        new_token_ids=new_token_ids,
        start_thinking_token_id=start_thinking_token_id,
        end_thinking_token_id=end_thinking_token_id,
        requests_to_start_thinking_idx=requests_to_start_thinking_idx,
        requests_with_remaining_budget=requests_with_remaining_budget,
        logger=logger,
    )

    assert requests_to_start_thinking_idx == {}
    assert requests_with_remaining_budget == {}

    logger.info.assert_not_called()
    logger.warning.assert_not_called()


def test_start_end_thinking_simultaneous_evictions():
    logger = make_logger()

    request = DummyRequest(
        request_id="req1",
        output_token_ids=[10, 11, 12, 255010, 255011],
    )

    new_token_ids = [12, 255010, 255011]  # START_THINKING at index 1
    start_thinking_token_id = 255010
    end_thinking_token_id = 255011

    requests_to_start_thinking_idx: dict[str, int | None] = {}
    requests_with_remaining_budget: dict[str, int] = {}

    handle_thinking_tokens(
        request=request,
        new_token_ids=new_token_ids,
        start_thinking_token_id=start_thinking_token_id,
        end_thinking_token_id=end_thinking_token_id,
        requests_to_start_thinking_idx=requests_to_start_thinking_idx,
        requests_with_remaining_budget=requests_with_remaining_budget,
        logger=logger,
    )

    assert "req1" not in requests_to_start_thinking_idx
    assert "req1" not in requests_with_remaining_budget

    logger.info.assert_called_once()
    logger.warning.assert_called_once()


if __name__ == "__main__":
    test_start_thinking_token_sets_correct_index()
    test_end_thinking_token_cleans_up_state()
    test_no_thinking_tokens_no_side_effects()
    test_start_end_thinking_simultaneous_evictions()

    print("All tests passed ✅")
