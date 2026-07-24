# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import MagicMock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor


def _make_request(request_id: str = "test-req-42") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_processor(*, disable_randomization: bool) -> InputProcessor:
    proc = MagicMock(spec=InputProcessor)
    proc._disable_request_id_randomization = disable_randomization
    proc.assign_request_id = lambda req: InputProcessor.assign_request_id(proc, req)
    return proc


class TestAssignRequestId:
    def test_randomization_enabled_by_default(self):
        proc = _make_processor(disable_randomization=False)
        req = _make_request("my-req")
        proc.assign_request_id(req)

        assert req.external_req_id == "my-req"
        assert req.request_id.startswith("my-req-")
        assert len(req.request_id) > len("my-req-")

    def test_randomization_disabled_for_kv_transfer(self):
        proc = _make_processor(disable_randomization=True)
        req = _make_request("my-req")
        proc.assign_request_id(req)

        assert req.external_req_id == "my-req"
        assert req.request_id == "my-req"

    def test_raises_if_external_req_id_already_set(self):
        proc = _make_processor(disable_randomization=False)
        req = _make_request("my-req")
        req.external_req_id = "already-set"

        with pytest.raises(ValueError, match="external_req_id"):
            proc.assign_request_id(req)

    def test_randomized_ids_are_unique(self):
        proc = _make_processor(disable_randomization=False)
        ids = set()
        for _ in range(100):
            req = _make_request("same-id")
            proc.assign_request_id(req)
            ids.add(req.request_id)

        assert len(ids) == 100
