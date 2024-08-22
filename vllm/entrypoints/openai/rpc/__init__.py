from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Union

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

# Success string used for RPC instructions.
VLLM_RPC_SUCCESS_STR = "SUCCESS"

# Minimum value of ZMQ.SOCKET_LIMIT to run mp.
VLLM_RPC_SOCKET_LIMIT_CUTOFF = 2000

# HWM is set to Infinity.
VLLM_RPC_ZMQ_HWM = 0


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
    IS_SERVER_READY = 1
    GET_MODEL_CONFIG = 2
    GET_DECODING_CONFIG = 3
    GET_PARALLEL_CONFIG = 4
    GET_SCHEDULER_CONFIG = 5
    GET_LORA_CONFIG = 6
    DO_LOG_STATS = 7
    IS_SERVER_HEALTHY = 8
    IS_TRACING_ENABLED = 9
    START_PROFILE = 10
    STOP_PROFILE = 11


RPC_REQUEST_TYPE = Union[RPCGenerateRequest, RPCAbortRequest,
                         RPCUtilityRequest]
