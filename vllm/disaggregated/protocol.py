# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import msgspec

from vllm import SamplingParams
from vllm.outputs import RequestOutput

# NOTE FOR DEVELOPERS:
# DO NOT USE PICKLE FOR THESE CLASSES. IN A MULTI NODE
# SETUP WE WILL USE TCP. WE CANNOT USE PICKLE OTHERWISE
# WE RISK REMOTE CODE EXECUTION FROM UNSTRUSTED USERS.


class PDRequestType:
    GENERATION = b'\x00'
    ABORT = b'\x01'


class PDGenerationRequest(msgspec.Struct):
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    # TODO: support multimodal inputs.


class PDAbortRequest(msgspec.Struct):
    request_id: str


class PDResponseType:
    GENERATION = b'\x00'
    FAILURE = b'\x01'


class PDGenerationResponse(msgspec.Struct):
    request_id: str
    text: str
    token_ids: list[int]
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    # TODO: support full protocol.
    logprobs = None

    @classmethod
    def from_request_output(
            self, request_output: RequestOutput) -> "PDGenerationResponse":
        assert len(request_output.outputs) == 1, "Only support N=1 right now."
        out = request_output.outputs[0]
        return PDGenerationResponse(
            request_id=request_output.request_id,
            text=out.text,
            token_ids=out.token_ids,
            finish_reason=out.finish_reason,
            stop_reason=out.stop_reason,
        )
