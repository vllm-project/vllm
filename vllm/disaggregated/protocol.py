# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

import msgspec

from vllm import SamplingParams

class RemotePrefillRequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """The request data of one remote prefill output of a request.
    Args:
        engine_id: The unique ID of the sending engine.
        request_id: The unique ID of the request.
        prompt_token_ids: The token IDs of the prompt.
        sampling_params: The sampling parameters.
        block_ids: The block IDs of the request.
    """
    engine_id: str
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    block_ids: List[int]


class RemotePrefillParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """Remote prefill parameters for text generation."""
    decode_engine_id: Optional[str] = None


class RemoteDecodeParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """Remote decode parameters for text generation."""
    decode_engine_id: str
    decode_block_ids: List[int]
