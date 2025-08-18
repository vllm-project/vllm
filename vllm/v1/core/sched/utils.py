# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.v1.request import Request, RequestStatus


def check_stop(request: Request,
               max_model_len: int,
               pooler_output: Optional[torch.Tensor] = None) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False


def maybe_update_thinking_state(
    request: Request,
    new_token_id: int,
    think_start_token_id: Optional[int] = None,
    think_end_token_id: Optional[int] = None,
) -> None:
    """
    Update thinking state of the request based on new token ID.
    """
    if think_start_token_id is not None and new_token_id == \
        think_start_token_id:
        request.thinking_state = True

    if think_end_token_id is not None and new_token_id == \
        think_end_token_id:
        request.thinking_state = False
