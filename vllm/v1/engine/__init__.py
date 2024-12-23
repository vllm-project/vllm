import os
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
from vllm.sampling_params import SamplingParams
from vllm.utils import kill_process_tree

@dataclass
class BackgroundProcHandle:
    proc: BaseProcess
    input_path: str
    output_path: str

    def shutdown(self):
        # Shutdown the process if needed.
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)

            if self.proc.is_alive():
                kill_process_tree(self.proc.pid)

        # Remove zmq ipc socket files
        ipc_sockets = [self.output_path, self.input_path]
        for ipc_socket in ipc_sockets:
            socket_file = ipc_socket.replace("ipc://", "")
            if os and os.path.exists(socket_file):
                os.remove(socket_file)


@dataclass
class EngineRequest:
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    mm_inputs: Optional[List[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[List[str]]
    mm_placeholders: Optional[MultiModalPlaceholderDict]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


@dataclass
class EngineAbortRequest:
    request_ids: List[str]


@dataclass
class EngineProfileRequest:
    is_start: bool


EngineRequestUnion = Union[EngineRequest, EngineAbortRequest,
                           EngineProfileRequest]


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout and using an int enum for finish/stop reason

    # [num_reqs]
    outputs: List[EngineCoreOutput]
