from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Union

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

# Success string used for RPC instructions.
VLLM_RPC_SUCCESS_STR = "SUCCESS"


@dataclass
class RPCGenerateRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


@dataclass
class RPCAbortRequest:
    request_id: str


class RPCUtilityRequest(Enum):
    DO_LOG_STATS = 1
    CHECK_HEALTH = 2


class RPCStartupRequest(Enum):
    IS_SERVER_READY = 1
    GET_MODEL_CONFIG = 2
    GET_DECODING_CONFIG = 3
    GET_PARALLEL_CONFIG = 4
    GET_SCHEDULER_CONFIG = 5
    GET_LORA_CONFIG = 6
    GET_TRACING_ENABLED = 7
    CLIENT_IS_READY = 8


RPC_REQUEST_TYPE = Union[RPCGenerateRequest, RPCAbortRequest,
                         RPCUtilityRequest, RPCStartupRequest]
