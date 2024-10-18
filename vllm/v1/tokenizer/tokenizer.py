import multiprocessing
from typing import List

import msgspec
import zmq
from msgspec import msgpack

from vllm.transformers_utils.tokenizer import get_tokenizer


class TokenizerInputs(msgspec.Struct):

    # [num_reqs]
    req_ids: List[str]
    prompts: List[str]


class TokenizerOutputs(msgspec.Struct):

    # [num_reqs]
    req_ids: List[str]
    token_ids: List[List[int]]


class Tokenizer(multiprocessing.Process):

    def __init__(
        self,
        tokenizer_id: str,
        port1: int,
        port2: int,
        **tokenizer_config,
    ):
        super().__init__()
        self.port1 = port1
        self.port2 = port2
        self.msgspec_encoder = msgpack.Encoder()
        self.msgspec_decoder = msgpack.Decoder(TokenizerInputs)

        self.tokenizer = get_tokenizer(tokenizer_id, **tokenizer_config)

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
            inputs = self.msgspec_decoder.decode(message)

            token_ids = self.tokenize(inputs.prompts)
            outputs = TokenizerOutputs(
                req_ids=inputs.req_ids,
                token_ids=token_ids,
            )
            self.push_socket.send(self.msgspec_encoder.encode(outputs),
                                    flags=zmq.NOBLOCK)

    def tokenize(self, prompts: List[str]) -> List[int]:
        return self.tokenizer.encode(prompts)
