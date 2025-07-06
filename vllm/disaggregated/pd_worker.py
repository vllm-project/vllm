# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import msgspec
import zmq
import zmq.asyncio

from vllm.disaggregated.protocol import (PDAbortRequest, PDGenerationRequest,
                                         PDGenerationResponse, PDRequestType,
                                         PDResponseType)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


class PDWorker:

    def __init__(
        self,
        engine: EngineClient,
        worker_addr: str,
        controller_addr: str,
    ):
        """
        PDWorker
            * Wrapper around AsyncLLM to handle converting PRRequests
              to PDResponse and sending back to the PDConroller.
            * Leverages ZMQ for communication with PDConroller. We may
              expand this in the future.
        """
        # Engine.
        self.engine = engine

        # ZMQ IPC.
        self.worker_addr = f"ipc://{worker_addr}"
        self.controller_addr = f"ipc://{controller_addr}"
        self.ctx = zmq.asyncio.Context()
        self.from_controller = self.ctx.socket(zmq.constants.PULL)
        self.from_controller.bind(self.worker_addr)
        self.to_controller = self.ctx.socket(zmq.constants.PUSH)
        self.to_controller.connect(self.controller_addr)
        self.decode_generation = msgspec.msgpack.Decoder(PDGenerationRequest)
        self.decode_abort = msgspec.msgpack.Decoder(PDAbortRequest)
        self.encoder = msgspec.msgpack.Encoder()

        # Active Requests.
        self.running_requests: set[asyncio.Task] = set()

    def shutdown(self):
        if hasattr(self, "ctx"):
            self.ctx.destroy()

        if hasattr(self, "running_requests"):
            for running_request in self.running_requests:
                running_request.cancel()

        if hasattr(self, "controller_addr"):
            ipc_paths = [self.worker_addr, self.controller_addr]
            for ipc_path in ipc_paths:
                socket_path = ipc_path.replace("ipc://", "")
                if os.path.exists(socket_path):
                    os.remove(socket_path)

    async def run_busy_loop(self):
        """
        main loop:
            1) wait for a request from the PDConroller
            2) handle the request
        """
        logger.info("PDWorker is ready To handle requests.")

        poller = zmq.asyncio.Poller()
        poller.register(self.from_controller, zmq.POLLIN)

        while True:
            # 1) Get request from the Connector.
            req_type, req_data = await self.from_controller.recv_multipart()

            # 2) Handle the request.
            await self._handle_request(req_type, req_data)

    async def _handle_request(self, req_type: bytes, req_data: bytes):
        """
        request handler:
            1) parse the request type
            2) call the appropriate handler for the request type
        """
        if req_type == PDRequestType.GENERATION:
            req = self.decode_generation.decode(req_data)
            await self._generation_handler(req)
        elif req_type == PDRequestType.ABORT:
            req = self.decode_abort.decode(req_data)
            await self._abort_handler(req)
        else:
            raise Exception(f"Unknown Request Type: {req_type}.")

    async def _generation_handler(self, req: PDGenerationRequest):
        """
        Handle a PDGenerationRequest by launching a task.
        """
        task = asyncio.create_task(self._generate(req))
        self.running_requests.add(task)
        task.add_done_callback(self.running_requests.discard)

    async def _abort_handler(self, req: PDGenerationRequest):
        """
        Handle a PDAbortRequest by cancelling the running task.
        The _generate coro aborts in the Engine.
        """
        # Convert running_requests set() into a dict(), keyed
        # by request_id. Cancel the task when an abort comes in.
        # Then update the _generate coroutine to handle a
        # cancel error by aborting in the Engine.
        pass

    async def _generate(self, req: PDGenerationRequest):
        """
        Handle a single PDGenerationRequest:
            * 1) submit request to AsyncLLM
            * 2) iterate the RequestOutputs
            * 3) convert RequestOutput --> PDResponse
            * 4) serialize and send to PDConroller
        """
        request_id = req.request_id

        # 1) Submit request to Engine.
        generator = self.engine.generate(
            prompt={"prompt_token_ids": req.prompt_token_ids},
            sampling_params=req.sampling_params,
            request_id=request_id)

        # 2) Iterate RequestOutputs.
        async for request_output in generator:
            # 3) Convert RequestOutput --> PDResponse.
            response = PDGenerationResponse.from_request_output(request_output)

            # 4) Serialize and send to PDConroller.
            response_bytes = self.encoder.encode(response)
            msg = (PDResponseType.GENERATION, response_bytes)
            await self.to_controller.send_multipart(msg, copy=False)
