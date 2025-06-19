# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.request import RequestGenerationState, RequestStatus
from typing import Optional


def check_stop(request: RequestGenerationState, max_model_len: int) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.params.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.params.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.params.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False


def check_stop_v1(
    request: RequestGenerationState, 
    max_model_len: int
) -> tuple[bool, Optional[RequestStatus], Optional[int]]:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.params.max_tokens):
        return True, RequestStatus.FINISHED_LENGTH_CAPPED, None

    sampling_params = request.params.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.params.eos_token_id):
        return True, RequestStatus.FINISHED_STOPPED, None

    if last_token_id in (sampling_params.stop_token_ids or ()):
        return True, RequestStatus.FINISHED_STOPPED, last_token_id
    return False, None, None
