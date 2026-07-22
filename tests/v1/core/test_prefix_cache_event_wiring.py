# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    EventBatch,
    PrefixCacheEventUploaderFactory,
)
from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.skip_global_cleanup


def test_scheduler_wires_prefix_cache_uploader_through_output_and_shutdown(
    monkeypatch,
):
    class RecordingUploader:
        def __init__(self):
            self.batches: list[EventBatch] = []
            self.shutdown_calls = 0

        def publish(self, batch: EventBatch) -> None:
            self.batches.append(batch)

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    uploader = RecordingUploader()
    factory_calls = []

    def create(cls, config, data_parallel_rank=0, initial_snapshot=None):
        factory_calls.append((config, data_parallel_rank, initial_snapshot))
        return uploader

    monkeypatch.setattr(
        PrefixCacheEventUploaderFactory,
        "create",
        classmethod(create),
    )
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(enable_prefix_caching=True)
    try:
        assert scheduler.prefix_cache_event_uploader is uploader
        assert len(factory_calls) == 1
        assert factory_calls[0][1] == scheduler.parallel_config.data_parallel_index

        scheduler.kv_cache_manager.take_events = Mock(return_value=[AllBlocksCleared()])
        request = create_requests(num_requests=1, req_ids=["req-0"])[0]
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

        assert len(uploader.batches) == 1
        assert isinstance(uploader.batches[0].events[0], AllBlocksCleared)
    finally:
        scheduler.shutdown()

    assert uploader.shutdown_calls == 1
