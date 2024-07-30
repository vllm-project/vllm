from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

VLLM_GENERATE_RPC_PATH = "tcp://localhost:5570"
VLLM_GET_DATA_RPC_PATH = "tcp://localhost:5571"
VLLM_IS_READY_RPC_PATH = "tcp://localhost:5572"


@dataclass
class GenerateRequest:
    inputs: PromptInputs
    sampling_params: SamplingParams
    request_id: str
    lora_request: Optional[LoRARequest] = None
    trace_headers: Optional[Mapping[str, str]] = None
    prompt_adapter_request: Optional[PromptAdapterRequest] = None


class GetDataRequest(Enum):
    MODEL_CONFIG = 1
