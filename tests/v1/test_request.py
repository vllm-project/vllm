# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.kv_checkpointing import (
    KV_CHECKPOINT_RESTORE_ID_ARG,
    KV_CHECKPOINT_SAVE_ID_ARG,
)
from vllm.v1.request import Request, RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_parses_kv_checkpoint_args():
    sampling_params = SamplingParams(
        max_tokens=8,
        extra_args={
            KV_CHECKPOINT_RESTORE_ID_ARG: "ckpt_restore",
            KV_CHECKPOINT_SAVE_ID_ARG: "ckpt_save",
        },
    )
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    assert request.kv_checkpoint_restore_id == "ckpt_restore"
    assert request.kv_checkpoint_save_id == "ckpt_save"


def test_request_rejects_non_string_kv_checkpoint_args():
    sampling_params = SamplingParams(
        max_tokens=8,
        extra_args={KV_CHECKPOINT_RESTORE_ID_ARG: 123},
    )
    with pytest.raises(ValueError):
        Request(
            request_id="req",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
        )
