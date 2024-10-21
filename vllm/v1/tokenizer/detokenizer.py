import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional

import msgspec
import zmq
from msgspec import msgpack

from vllm.transformers_utils.detokenizer_utils import (
    convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import get_open_port


class DetokenizerInputs(msgspec.Struct):

    # [num_reqs]
    req_ids: List[str]
    # A request's prompt token ids is sent to the detokenizer only when
    # the request is first detokenized. Otherwise, an empty list is sent.
    prompt_token_ids: List[List[int]]
    new_token_ids: List[List[int]]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]

    # [num_free_reqs]
    free_req_ids: List[str]


class DetokenizerOutputs(msgspec.Struct):

    # [num_reqs]
    req_ids: List[str]
    detokenized_texts: List[str]
    # NOTE(woosuk): The number of the output token ids of each request
    # at the time of detokenization. The detokenizer returns this to the engine
    # because the request state (including the output token ids) is
    # asynchronously updated in the engine, while RequestOutput requires the
    # output token ids to be consistent with the detokenized text.
    num_output_token_ids: List[int]


class Detokenizer:

    def __init__(self, tokenizer_name: str):
        # FIXME(woosuk): Currently, the detokenizer is just a hacky prototype.
        # For example, it does not terminate properly. We need to improve this.
        self.push_port = get_open_port()
        self.pull_port = get_open_port()
        self.detokenizer = DetokenizerProc(tokenizer_name, self.push_port,
                                           self.pull_port)
        self.detokenizer.start()

        self.zmq_context = zmq.Context()
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{self.push_port}")
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://localhost:{self.pull_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.pull_socket, zmq.POLLIN)
        self.msgpack_encoder = msgpack.Encoder()
        self.msgpack_decoder = msgpack.Decoder(DetokenizerOutputs)

    def send(self, inputs: DetokenizerInputs) -> None:
        self.push_socket.send(self.msgpack_encoder.encode(inputs),
                              flags=zmq.NOBLOCK)

    def recv(self) -> Optional[DetokenizerOutputs]:
        socks = dict(self.poller.poll(timeout=0))
        if self.pull_socket in socks and socks[self.pull_socket] == zmq.POLLIN:
            msg = self.pull_socket.recv()
            return self.msgpack_decoder.decode(msg)
        return None

    def terminate(self) -> None:
        self.push_socket.send(b"", flags=zmq.NOBLOCK)
        self.detokenizer.join()


class DetokenizerProc(multiprocessing.Process):

    def __init__(
        self,
        tokenizer_name: str,
        pull_port: int,
        push_port: int,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        # NOTE: The pull_port of the detokenizer should be the same as the
        # push_port of the engine. Vice versa.
        self.pull_port = pull_port
        self.push_port = push_port

    def run(self):
        # Initialize these objects after the process is forked since they are
        # not picklable.
        self.msgpack_encoder = msgpack.Encoder()
        self.msgpack_decoder = msgpack.Decoder(DetokenizerInputs)
        self.tokenizer = get_tokenizer(self.tokenizer_name)
        # req_id -> RequestState
        self.request_states: Dict[str, RequestState] = {}

        self.zmq_context = zmq.Context()
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{self.pull_port}")
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://*:{self.push_port}")

        while True:
            message = self.pull_socket.recv()
            if message == b"":
                # Terminate signal.
                break
            inputs = self.msgpack_decoder.decode(message)

            for req_id in inputs.free_req_ids:
                self.free(req_id)

            detokenized_texts: List[str] = []
            num_output_token_ids: List[int] = []
            num_reqs = len(inputs.req_ids)
            for i in range(num_reqs):
                req_id = inputs.req_ids[i]
                if req_id not in self.request_states:
                    self.add_request(
                        request_id=req_id,
                        prompt_token_ids=inputs.prompt_token_ids[i],
                        skip_special_tokens=inputs.skip_special_tokens[i],
                        spaces_between_special_tokens=inputs.
                        spaces_between_special_tokens[i],
                    )
                new_str = self.detokenize(req_id, inputs.new_token_ids[i])
                detokenized_texts.append(new_str)
                req_state = self.request_states[req_id]
                num_output_token_ids.append(
                    len(req_state.token_ids) - req_state.num_prompt_tokens)

            detokenized = DetokenizerOutputs(
                req_ids=inputs.req_ids,
                detokenized_texts=detokenized_texts,
                num_output_token_ids=num_output_token_ids,
            )
            self.push_socket.send(self.msgpack_encoder.encode(detokenized),
                                  flags=zmq.NOBLOCK)

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ) -> None:
        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=self.tokenizer,
            prompt_ids=prompt_token_ids,
            skip_special_tokens=skip_special_tokens,
        )
        self.request_states[request_id] = RequestState(
            req_id=request_id,
            token_ids=prompt_token_ids,
            tokens=tokens,
            num_prompt_tokens=len(prompt_token_ids),
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

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


@dataclass
class RequestState:

    req_id: str

    token_ids: List[int]
    tokens: List[str]
    num_prompt_tokens: int

    prefix_offset: int
    read_offset: int

    skip_special_tokens: bool
    spaces_between_special_tokens: bool

    output_text: str = ""
