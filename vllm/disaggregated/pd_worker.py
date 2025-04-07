# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import msgspec
import zmq
import zmq.asyncio

from vllm.disaggregated.protocol import (RemotePrefillRequest,
                                         RemoteDecodeParams)
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

logger = init_logger(__name__)


class PrefillWorker:

    def __init__(
        self,
        engine: EngineClient,
        prefill_addr: str,
        decode_addr: str,
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
        # TODO: switch to ROUTER<>DEALER with service discovery.
        self.prefill_worker_addr = f"ipc://{prefill_addr}"
        self.decode_worker_addr = f"ipc://{decode_addr}"
        self.ctx = zmq.asyncio.Context()
        self.from_decode = self.ctx.socket(zmq.constants.PULL)
        self.from_decode.bind(self.prefill_worker_addr)
        self.to_decode = self.ctx.socket(zmq.constants.PUSH)
        self.to_decode.connect(self.decode_worker_addr)
        self.decoder = msgspec.msgpack.Decoder(RemotePrefillRequest)
        self.encoder = msgspec.msgpack.Encoder()

        # Active Requests.
        self.running_requests: set[asyncio.Task] = set()

    def shutdown(self):
        if hasattr(self, "ctx"):
            self.ctx.destroy()

        if hasattr(self, "running_requests"):
            for running_request in self.running_requests:
                running_request.cancel()

        if hasattr(self, "prefill_worker_addr"):
            ipc_paths = [self.prefill_worker_addr,
                         self.decode_worker_addr]
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

        while True:
            # 1) Get request from the Connector.
            data = await self.from_decode.recv()
            req = self.decoder.decode(data)

            # 2) Handle the request.
            task = asyncio.create_task(self._generate(req))
            self.running_requests.add(task)
            task.add_done_callback(self.running_requests.discard)
        

    async def _generate(self, req: RemotePrefillRequest):
        """Handle a single RemotePrefillRequest"""
        request_id = req.request_id

        # 1) Update Params to be Prefill Only.
        req.sampling_params.max_tokens = 1
        req.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

        # 2) Submit RemoteDecode request to the Engine.
        generator = self.engine.generate(
            prompt={"prompt_token_ids": req.prompt_token_ids},
            sampling_params=req.sampling_params,
            request_id=request_id,
            remote_decode_params=RemoteDecodeParams(
                decode_engine_id=req.engine_id,
                decode_block_ids=req.block_ids,
            )
        )

        # 3) Process Prefill Request.
        async for _ in generator:
            pass
