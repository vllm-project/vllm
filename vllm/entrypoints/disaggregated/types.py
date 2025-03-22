# SPDX-License-Identifier: Apache-2.0

import msgspec
from typing import List, Optional
from vllm import SamplingParams

# NOTE FOR DEVELOPERS:
# DO NOT USE PICKLE FOR THESE CLASSES. IN A MULTI NODE
# SETUP WE WILL USE TCP. WE CANNOT USE PICKLE OTHERWISE
# WE RISK REMOTE CODE EXECUTION FROM UNSTRUSTED USERS.

class PDRequest(msgspec.Struct,
              array_like=True,  # type: ignore[call-arg]
              omit_defaults=True,  # type: ignore[call-arg]
              gc=False):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    # TODO: support multimodal inputs.

class PDResponse(msgspec.Struct,
              array_like=True,  # type: ignore[call-arg]
              omit_defaults=True,  # type: ignore[call-arg]
              gc=False):  # type: ignore[call-arg]
    request_id: str
    success: bool
    text: str
    token_ids: List[int]
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    logprobs = None # TODO
