from dataclasses import dataclass
from typing import Callable, Optional, List
from enum import Enum

import msgspec

from vllm.sampling_params import SamplingParams


class RemotePrefillRequest(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    """The request data of one remote prefill output of a request.

    Args:
        engine_id: The unique ID of the engine.
        request_id: The unique ID of the request.
        prompt_token_ids: The token IDs of the prompt.
        sampling_params: The sampling parameters.
        block_ids: The block IDs of the request.
        computed_block_ids: The computed block IDs of the request.
    """
    engine_id: str
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    block_ids: List[int]
    computed_block_ids: List[int]

class MemoryOpType(str, Enum):
    WRITE = "WRITE"
    READ = "READ"


class MemoryTransferRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """The request data of one memory transfer output of a request.

    Args:
        request_id: The unique ID of the request.
    """
    request_id: str
    local_block_ids: List[int]
    staging_block_ids: List[int]
    remote_block_ids: List[int]
    remote_engine_id: str
    notify_msg: str
    op_type: MemoryOpType


RemotePrefillRequestCallback = Callable[[RemotePrefillRequest], None]


@dataclass
class RemotePrefillParams:
    """Remote prefill parameters for text generation."""
    is_remote_prefill: bool = False
    is_remote_decode: bool = False
    decode_block_ids: Optional[List[int]] = None
    decode_computed_block_ids: Optional[List[int]] = None
    decode_engine_id: Optional[str] = None
    remote_prefill_request_callback: Optional[RemotePrefillRequestCallback] = None