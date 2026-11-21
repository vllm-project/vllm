# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import importlib
import queue
from pathlib import Path

import pytest


@pytest.fixture
def multi_turn_module(monkeypatch: pytest.MonkeyPatch):
    benchmark_dir = Path(__file__).parents[2] / "benchmarks" / "multi_turn"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    return importlib.import_module("benchmark_serving_multi_turn")


class _StopEvent:
    def is_set(self) -> bool:
        return False


@pytest.mark.benchmark
def test_client_drains_active_conversations_after_task_queue_exhaustion(
    multi_turn_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All active conversations finish after the task queue sends its sentinel."""
    requests = []

    async def fake_send_turn(
        _session,
        _client_id,
        conv_id,
        _messages,
        current_turn,
        _tokenizer,
        _req_args,
        _verbose,
        _verify_output,
    ):
        requests.append((conv_id, current_turn))
        return multi_turn_module.RequestStats(
            ttft_ms=0.0,
            tpot_ms=0.0,
            latency_ms=0.0,
            start_time_ms=0.0,
            input_num_turns=current_turn,
            input_num_tokens=0,
            output_num_tokens=0,
            output_num_chunks=0,
            output_num_first_chunk_tokens=0,
            approx_cached_percent=0.0,
            conversation_id=conv_id,
            client_id=0,
        )

    monkeypatch.setattr(multi_turn_module, "send_turn", fake_send_turn)

    args = multi_turn_module.ClientArgs(
        seed=0,
        max_num_requests=None,
        skip_first_turn=False,
        max_turns=None,
        max_active_conversations=2,
        verbose=False,
        print_content=False,
        verify_output=False,
        conversation_sampling=multi_turn_module.ConversationSampling.ROUND_ROBIN,
        request_rate=0,
        max_retries=0,
    )
    req_args = multi_turn_module.RequestArgs(
        chat_url="http://localhost",
        model="test",
        stream=False,
        limit_min_tokens=-1,
        limit_max_tokens=-1,
        timeout_sec=1,
        send_conversation_id=False,
        headers={},
    )
    conversations = {
        "conversation-1": [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"},
        ],
        "conversation-2": [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"},
        ],
    }
    task_queue = queue.Queue()
    for item in conversations.items():
        task_queue.put(item)
    task_queue.put((multi_turn_module.TERM_SIGNAL, multi_turn_module.TERM_SIGNAL))

    result_queue = queue.Queue()
    conv_queue = queue.Queue()

    asyncio.run(
        multi_turn_module.client_main(
            args,
            req_args,
            0,
            tokenizer=None,
            stop_event=_StopEvent(),
            task_queue=task_queue,
            result_queue=result_queue,
            conv_queue=conv_queue,
        )
    )

    assert len(requests) == 4
    assert {conv_id for conv_id, _ in requests} == set(conversations)
    assert result_queue.qsize() == 4
    assert conv_queue.qsize() == 3
