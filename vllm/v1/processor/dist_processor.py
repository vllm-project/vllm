import multiprocessing
import pickle
from abc import ABC, abstractmethod
from typing import Optional, Type

import msgspec
import zmq
from msgspec import msgpack

from vllm.utils import get_open_port


class ProcessorInputs:
    pass


class ProcessorOutputs:
    pass


class ProcessorImpl(ABC):

    @abstractmethod
    def process_inputs(self, inputs: ProcessorInputs) -> ProcessorOutputs:
        raise NotImplementedError


class Processor:

    def __init__(
        self,
        proc_cls: Type[ProcessorImpl],
        input_cls: Type[ProcessorInputs],
        output_cls: Type[ProcessorOutputs],
        *args,
        **kwargs,
    ):
        self.push_port = get_open_port()
        self.pull_port = get_open_port()
        self.worker = ProcessorWorker(self.push_port, self.pull_port, proc_cls,
                                      input_cls, output_cls, *args, **kwargs)
        self.worker.start()

        self.zmq_context = zmq.Context()
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{self.push_port}")
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://localhost:{self.pull_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.pull_socket, zmq.POLLIN)

        if issubclass(input_cls, msgspec.Struct):
            self.input_encoder = msgpack.Encoder()
        else:
            self.input_encoder = PickleEncoder()
        if issubclass(output_cls, msgspec.Struct):
            self.output_decoder = msgpack.Decoder(output_cls)
        else:
            self.output_decoder = PickleDecoder()

    def send(self, inputs: ProcessorInputs) -> None:
        self.push_socket.send(self.input_encoder.encode(inputs),
                              flags=zmq.NOBLOCK)

    def recv(self) -> Optional[ProcessorOutputs]:
        socks = dict(self.poller.poll(timeout=0))
        if self.pull_socket in socks and socks[self.pull_socket] == zmq.POLLIN:
            msg = self.pull_socket.recv()
            return self.output_decoder.decode(msg)
        return None

    def terminate(self) -> None:
        self.push_socket.send(self.worker.TERMINATE_SIGNAL, flags=zmq.NOBLOCK)
        self.worker.join()


class ProcessorWorker(multiprocessing.Process):

    TERMINATE_SIGNAL = b""

    def __init__(
        self,
        pull_port: int,
        push_port: int,
        proc_cls: Type[ProcessorImpl],
        input_cls: Type[ProcessorInputs],
        output_cls: Type[ProcessorOutputs],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pull_port = pull_port
        self.push_port = push_port
        self.proc_cls = proc_cls
        self.input_cls = input_cls
        self.output_cls = output_cls
        self.args = args
        self.kwargs = kwargs

    def run(self):
        # Initialize these objects after the process is forked since they may
        # not be picklable.
        self.zmq_context = zmq.Context()
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{self.pull_port}")
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://*:{self.push_port}")

        if issubclass(self.input_cls, msgspec.Struct):
            self.input_decoder = msgpack.Decoder(self.input_cls)
        else:
            self.input_decoder = PickleDecoder()
        if issubclass(self.output_cls, msgspec.Struct):
            self.output_encoder = msgpack.Encoder()
        else:
            self.output_encoder = PickleEncoder()

        # Initialize the processor.
        self.processor = self.proc_cls(*self.args, **self.kwargs)

        while True:
            message = self.pull_socket.recv()
            if message == self.TERMINATE_SIGNAL:
                # Terminate signal.
                break
            inputs = self.input_decoder.decode(message)
            outputs = self.processor.process_inputs(inputs)
            self.push_socket.send(self.output_encoder.encode(outputs),
                                  flags=zmq.NOBLOCK)


class PickleEncoder:

    def encode(self, obj) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


class PickleDecoder:

    def decode(self, data: bytes):
        return pickle.loads(data)
