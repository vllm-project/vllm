import multiprocessing
from typing import Optional, Type

import msgspec
import zmq
from msgspec import msgpack

from vllm.utils import get_open_port


class ProcessorInputs(msgspec.Struct):
    pass


class ProcessorOutputs(msgspec.Struct):
    pass


class Processor:

    def __init__(
        self,
        proc_cls: Type["ProcessorProc"],
        input_cls: Type[ProcessorInputs],
        output_cls: Type[ProcessorOutputs],
        *args,
        **kwargs,
    ):
        self.push_port = get_open_port()
        self.pull_port = get_open_port()
        self.worker = proc_cls(self.push_port, self.pull_port, input_cls,
                               output_cls, *args, **kwargs)
        self.worker.start()

        self.zmq_context = zmq.Context()
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{self.push_port}")
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://localhost:{self.pull_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.pull_socket, zmq.POLLIN)
        self.msgpack_encoder = msgpack.Encoder()
        self.msgpack_decoder = msgpack.Decoder(output_cls)

    def send(self, inputs: ProcessorInputs) -> None:
        self.push_socket.send(self.msgpack_encoder.encode(inputs),
                              flags=zmq.NOBLOCK)

    def recv(self) -> Optional[ProcessorOutputs]:
        socks = dict(self.poller.poll(timeout=0))
        if self.pull_socket in socks and socks[self.pull_socket] == zmq.POLLIN:
            msg = self.pull_socket.recv()
            return self.msgpack_decoder.decode(msg)
        return None

    def terminate(self) -> None:
        self.push_socket.send(self.worker.TERMINATE_SIGNAL, flags=zmq.NOBLOCK)
        self.worker.join()


class ProcessorProc(multiprocessing.Process):

    TERMINATE_SIGNAL = b""

    def __init__(
        self,
        pull_port: int,
        push_port: int,
        input_cls: Type[ProcessorInputs],
        output_cls: Type[ProcessorOutputs],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pull_port = pull_port
        self.push_port = push_port
        self.input_cls = input_cls
        self.output_cls = output_cls
        self.args = args
        self.kwargs = kwargs

    def run(self):
        # Initialize these objects after the process is forked since they may
        # not be picklable.
        self.msgpack_encoder = msgpack.Encoder()
        self.msgpack_decoder = msgpack.Decoder(self.input_cls)
        self.zmq_context = zmq.Context()
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://*:{self.pull_port}")
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://*:{self.push_port}")

        self.init_states(*self.args, **self.kwargs)

        while True:
            message = self.pull_socket.recv()
            if message == self.TERMINATE_SIGNAL:
                # Terminate signal.
                break
            inputs = self.msgpack_decoder.decode(message)
            outputs = self.process_inputs(inputs)
            self.push_socket.send(self.msgpack_encoder.encode(outputs),
                                  flags=zmq.NOBLOCK)

    def init_states(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def process_inputs(self, inputs: ProcessorInputs) -> ProcessorOutputs:
        raise NotImplementedError
