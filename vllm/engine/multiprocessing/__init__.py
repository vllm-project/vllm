from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Optional, Union

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

VLLM_RPC_SUCCESS_STR = "SUCCESS"

IPC_INPUT_EXT = "_input_socket"
IPC_OUTPUT_EXT = "_output_socket"
IPC_HEALTH_EXT = "_health_socket"
IPC_DATA_EXT = "_data_socket"


class MQEngineDeadError(RuntimeError):
    pass


@dataclass
class RPCGenerateRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


@dataclass
class RPCError:
    request_id: Optional[str]
    is_engine_errored: bool
    exception: BaseException


@dataclass
class RPCAbortRequest:
    request_id: str


class RPCHealthRequest:
    pass


class RPCStartupRequest(Enum):
    IS_SERVER_READY = 1
    GET_MODEL_CONFIG = 2
    GET_DECODING_CONFIG = 3
    GET_PARALLEL_CONFIG = 4
    GET_SCHEDULER_CONFIG = 5
    GET_LORA_CONFIG = 6
    GET_TRACING_ENABLED = 7
    CLIENT_IS_READY = 8


RPC_REQUEST_T = Union[RPCGenerateRequest, RPCAbortRequest, RPCHealthRequest,
                      RPCStartupRequest]

REQUEST_OUTPUTS_T = Union[List[RequestOutput], RPCError]

ENGINE_DEAD_ERROR = MQEngineDeadError(
    "Engine loop is not running. Inspect the output to find "
    "the stacktrace of the error that caused the engine loop "
    "to stop (MQEngineDeadError).")
