import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import msgspec
import zmq

from .tokenizer import get_tokenizer
from .detokenizer_utils import (convert_prompt_ids_to_tokens,
                                detokenize_incrementally)


class RequestData(msgspec.Struct):

    # [num_reqs]
    request_ids: List[str]
    prompt_token_ids: List[List[int]]
    new_token_ids: List[List[int]]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]

    # [num_free_reqs]
    free_request_ids: List[str]


class DetokenizedData(msgspec.Struct):

    # [num_reqs]
    request_ids: List[str]
    detokenized_texts: List[str]


@dataclass
class RequestState:

    req_id: str

    token_ids: List[int]
    tokens: List[str]

    prefix_offset: int
    read_offset: int

    skip_special_tokens: bool
    spaces_between_special_tokens: bool

    output_text: str = ""


class Detokenizer(multiprocessing.Process):

    def __init__(
        self,
        tokenizer_name: str,
        port1: int,
        port2: int,
    ):
        super().__init__()
        self.port1 = port1
        self.port2 = port2
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(RequestData)

        self.tokenizer = get_tokenizer(tokenizer_name)
        self.requests: Dict[str, RequestState] = {}

    def run(self):
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{self.port1}")
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://*:{self.port2}")

        while True:
            message = self.pull_socket.recv()
            if message == b"":
                # Terminate signal.
                break
            data = self.decoder.decode(message)

            for req_id in data.free_request_ids:
                self.free(req_id)

            req_ids: List[str] = []
            detokenized_texts: List[str] = []
            num_reqs = len(data.request_ids)
            for i in range(num_reqs):
                req_id = data.request_ids[i]
                req_ids.append(req_id)
                if req_id not in self.requests:
                    self.add_request(
                        request_id=req_id,
                        prompt_token_ids=data.prompt_token_ids[i],
                        skip_special_tokens=data.skip_special_tokens[i],
                        spaces_between_special_tokens=data.
                        spaces_between_special_tokens[i],
                    )
                new_str = self.detokenize(req_id, data.new_token_ids[i])
                detokenized_texts.append(new_str)

            detokenized = DetokenizedData(
                request_ids=req_ids,
                detokenized_texts=detokenized_texts,
            )
            self.push_socket.send(self.encoder.encode(detokenized),
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
        self.requests[request_id] = RequestState(
            req_id=request_id,
            token_ids=prompt_token_ids,
            tokens=tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    def free(self, request_id: str) -> None:
        del self.requests[request_id]

    def detokenize(self, request_id: str, new_token_ids: List[int]) -> str:
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        req_state = self.requests[request_id]
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
