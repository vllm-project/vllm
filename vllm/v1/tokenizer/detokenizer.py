import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import msgspec
import zmq
from msgspec import msgpack

from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.detokenizer_utils import (
    convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer

IPC_PATH_INPUT = "/tmp/prototype-input"
IPC_PATH_NEW_REQUEST = "/tmp/prototype-new_request"
IPC_POLLING_TIMEOUT_MS = 5000

logger = init_logger(__name__)


class DetokenizerNewRequest(msgspec.Struct):

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind


@dataclass
class DetokenizerInputData:

    request_id: str
    new_token_ids: List[int]
    finished: List[bool]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class DetokenizerInputs(msgspec.Struct):

    # [num_reqs]
    data: List[DetokenizerInputData]


@dataclass
class DetokenizerRequestState:

    # Generated text
    output_text: str

    # Prompt information
    prompt: Optional[str]
    num_prompt_tokens: int

    # Prompt/generation tokens
    tokens: List[str]
    token_ids: List[int]

    # Metadata for incremental detokenization
    prefix_offset: int
    read_offset: int

    # Parameters for detokenization
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    @classmethod
    def from_new_request(cls, tokenizer, new_request: DetokenizerNewRequest):
        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=new_request.prompt_token_ids,
            skip_special_tokens=new_request.skip_special_tokens,
        )

        return cls(
            req_id=new_request.request_id,
            output_text="",
            prompt=new_request.prompt,
            num_prompt_tokens=len(new_request.prompt_token_ids),
            tokens=tokens,
            token_ids=new_request.prompt_token_ids,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=new_request.skip_special_tokens,
            spaces_between_special_tokens=new_request.
            spaces_between_special_tokens,
            output_kind=new_request.output_kind,
        )


class Detokenizer:

    def __init__(self, tokenizer_name: str, output_socket_path: str):
        # FIXME(woosuk): Currently, the detokenizer is just a hacky prototype.
        # For example, it does not terminate properly. We need to improve this.

        # Serialization protocol.
        self.msgpack_encoder = msgpack.Encoder()

        # Setup IPC to the Detokenizer background process.
        self.zmq_context = zmq.Context()
        self.new_request_socket = self.zmq_context.socket(zmq.PULL)
        self.new_request_socket.bind(f"icp://{IPC_PATH_NEW_REQUEST}")
        self.input_socket = self.zmq_context.socket(zmq.PULL)
        self.input_socket.bind(f"icp://{IPC_PATH_INPUT}")

        # Start the background process.
        self.detokenizer = DetokenizerProc(tokenizer_name, output_socket_path)
        self.detokenizer.start()

    def add_request(self, new_request: DetokenizerNewRequest):
        self.new_request_socket.send(self.msgpack_encoder.encode(new_request),
                                     flags=zmq.NOBLOCK)

    def send(self, inputs: DetokenizerInputs) -> None:
        self.input_socket.send(self.msgpack_encoder.encode(inputs),
                               flags=zmq.NOBLOCK)

    def terminate(self) -> None:
        self.input_socket.send(b"", flags=zmq.NOBLOCK)
        self.detokenizer.join()
        logger.info("Detokenizer shutdown is complete.")


class DetokenizerProc(multiprocessing.Process):

    def __init__(self, tokenizer_name: str, output_socket_path: str):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.output_socket_path = output_socket_path

        # Cache the request output and update it incrementally to avoid
        # creating a new RequestOutput every step.
        # Request id -> RequestOutput
        self.request_outputs: Dict[str, RequestOutput] = {}

        # Request id -> DetokenizerRequestState
        self.request_states: Dict[str, DetokenizerRequestState] = {}

    def run(self):
        # Initialize these objects after the process is forked since they
        # are not picklable.
        self.msgpack_encoder = msgpack.Encoder()
        self.msgpack_new_request_decoder = msgpack.Decoder(
            DetokenizerNewRequest)
        self.msgpack_input_decoder = msgpack.Decoder(DetokenizerInputs)
        self.tokenizer = get_tokenizer(self.tokenizer_name)

        # Ipc objects.
        self.zmq_context = zmq.Context()
        self.poller = zmq.Poller()

        # Gets new requests from the LLMEngine.
        self.new_request_socket = self.zmq_context.socket(zmq.PULL)
        self.new_request_socket.connect(f"{IPC_PATH_NEW_REQUEST}")
        self.poller.register(self.new_request_socket, zmq.POLLIN)

        # Gets new input from the LLMEngine.
        self.input_socket = self.zmq_context.socket(zmq.PULL)
        self.input_socket.connect(f"{IPC_PATH_INPUT}")
        self.poller.register(self.input_socket, zmq.POLLIN)

        # Sends RequestOutputs to the EngineClient.
        self.output_socket = self.zmq_context.socket(zmq.PUSH)
        self.output_socket.bind(self.output_socket_path)

        # Busy loop.
        self._should_terminate = False
        while not self._should_terminate:
            events = dict(self.poller.poll(timeout=IPC_POLLING_TIMEOUT_MS))
            if len(events) == 0:
                logger.debug("Detokenizer is waiting for work.")

            if self.new_request_socket in events:
                assert events[self.new_request_socket] == zmq.POLLIN
                self._handle_new_request()

            if self.input_socket in events:
                assert events[self.input_socket] == zmq.POLLIN
                self._handle_new_input()

    def free(self, request_id: str) -> None:
        del self.request_states[request_id]

    def detokenize(self, request_id: str, new_token_ids: List[int]) -> str:
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        req_state = self.request_states[request_id]
        decoded_text = ""
        for new_token_id in new_token_ids:
            req_state.token_ids.append(new_token_id)
            (new_tokens, new_decoded_token_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 tokenizer=self.tokenizer,
                 all_input_ids=req_state.token_ids,
                 prev_tokens=req_state.tokens,
                 prefix_offset=req_state.prefix_offset,
                 read_offset=req_state.read_offset,
                 skip_special_tokens=req_state.skip_special_tokens,
                 spaces_between_special_tokens=req_state.
                 spaces_between_special_tokens,
             )

            req_state.tokens.extend(new_tokens)
            req_state.prefix_offset = prefix_offset
            req_state.read_offset = read_offset
            req_state.output_text += new_decoded_token_text
            decoded_text += new_decoded_token_text
        return decoded_text

    def _make_request_output(
        self,
        request_id: str,
        new_output_text: str,
        finished: bool,
        finish_reason: Optional[str],
        stop_reason: Optional[str],
    ) -> RequestOutput:
        assert request_id in self.request_states
        request_state = self.request_states[request_id]

        # Get cached RequestOutput or make it.
        req_output = self.request_outputs.get(request_id)
        if req_output is None:
            # TODO: Support `n` > 1.
            completion_output = CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,  # TODO
                finish_reason=None,
                stop_reason=None,
                lora_request=None,
            )
            req_output = RequestOutput(
                request_id=request_id,
                prompt=request_state.prompt,
                prompt_token_ids=request_state.prompt_token_ids,
                prompt_logprobs=None,  # TODO
                outputs=[completion_output],
                finished=False,
                metrics=None,
                lora_request=None,
                encoder_prompt=None,
                encoder_prompt_token_ids=None,
            )
            self.request_outputs[request_id] = req_output

        completion_output = req_output.outputs[0]
        if request_state.output_kind == RequestOutputKind.CUMULATIVE:
            completion_output.text += new_output_text
            completion_output.token_ids = request_state.token_ids
        elif request_state.output_kind == RequestOutputKind.DELTA:
            completion_output.text = new_output_text
            num_prev_tokens = len(completion_output.token_ids)
            completion_output.token_ids = request_state.token_ids[
                num_prev_tokens:]
        elif request_state.output_kind == RequestOutputKind.FINAL_ONLY:
            if finished:
                completion_output.text = request_state.output_text
                completion_output.token_ids = request_state.token_ids
            else:
                completion_output.text = ""
                completion_output.token_ids = []

        if finished:
            completion_output.finish_reason = finish_reason
            completion_output.stop_reason = stop_reason
            req_output.finished = finished
        return req_output

    def _handle_new_request(self):
        # TODO(robertgshaw2): this should be nonblocking
        message = self.new_request_socket.recv()
        if message == b"" or self._should_terminate:
            # Terminate signal.
            self._should_terminate = True
            return

        # Deserialize.
        new_request = self.msgpack_new_request_decoder.decode(message)

        # Add to request_state tracker.
        request_id = new_request.request_id
        request_state = DetokenizerRequestState.from_new_request(
            self.tokenizer, new_request)
        self.request_states[request_id] = request_state

    def _handle_new_input(self):
        # TODO(robertgshaw2): this should be nonblocking
        message = self.input_socket.recv()
        if message == b"" or self._should_terminate:
            self._should_terminate = True
            return

        # Deserialize.
        inputs = self.msgpack_input_decoder.decode(message).data

        # Update request states and create RequestOutput objects.
        request_outputs = [
            self._make_request_output(
                request_id=input.request_id,
                new_output_text=self.detokenize(input.request_id,
                                                input.new_token_ids),
                finished=input.finished,
                finish_reason=input.finish_reason,
                stop_reason=input.stop_reason,
            ) for input in inputs
        ]

        self.send_request_output(request_outputs)
