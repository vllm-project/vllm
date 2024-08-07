from contextlib import contextmanager
from typing import Any, AsyncGenerator, Mapping, Optional

import cloudpickle
import zmq
import zmq.asyncio

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.entrypoints.openai.rpc import (RPC_REQUEST_TYPE,
                                         VLLM_RPC_HEALTHY_STR,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs


class AsyncEngineRPCClient:

    def __init__(self, port: int):
        self.context = zmq.asyncio.Context()
        self.path = f"tcp://localhost:{port}"

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        # Wait until server is ready.
        await self.wait_for_server()

        # Get the configs.
        self.model_config = await self._get_model_config_rpc()
        self.decoding_config = await self._get_decoding_config_rpc()
        self.tracing_flag = await self._is_tracing_enabled_rpc()

        # Create the tokenizer group.
        # TODO: refactor OAI server to avoid needing this info.
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=(await self._get_scheduler_config_rpc()),
            parallel_config=(await self._get_parallel_config_rpc()),
            enable_lora=bool(await self._get_lora_config_rpc()),
        )

    def close(self):
        """Destroy the ZeroMQ Context."""
        self.context.destroy()

    @contextmanager
    def socket(self):
        # Ensure client sockets are always closed after use

        # Connect to RPC socket for Request-Reply pattern,
        # Note that we use DEALER to enable asynchronous communication
        # to enable streaming.
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.path)
            yield socket
        finally:
            socket.close()

    async def _send_get_data_rpc_request(self, request: RPCUtilityRequest,
                                         expected_type: Any,
                                         error_message: str) -> Any:
        """Send an RPC request that is expecting data back."""

        with self.socket() as socket:

            # Ping RPCServer with a request.
            await socket.send(cloudpickle.dumps(request))

            # Await the data from the Server.
            data = cloudpickle.loads(await socket.recv())

        if not isinstance(data, expected_type):
            # LoRAConfig can be None.
            if expected_type == LoRAConfig and data is None:
                pass
            else:
                raise ValueError(error_message)

        return data

    async def _send_one_way_rpc_request(self, request: RPC_REQUEST_TYPE,
                                        error_message: str):
        """Send one-way RPC request to trigger an action."""
        with self.socket() as socket:
            # Ping RPC Server with request.
            await socket.send(cloudpickle.dumps(request))

            # Await acknowledgement from RPCServer.
            response = cloudpickle.loads(await socket.recv())

        if not isinstance(response, str) or response != VLLM_RPC_SUCCESS_STR:
            raise ValueError(error_message)

        return response

    async def get_tokenizer(self, lora_request: LoRARequest):
        return await self.tokenizer.get_lora_tokenizer_async(lora_request)

    async def get_decoding_config(self) -> DecodingConfig:
        return self.decoding_config

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def is_tracing_enabled(self) -> bool:
        return self.tracing_flag

    async def wait_for_server(self):
        """Wait for the RPCServer to start up."""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.IS_SERVER_READY,
            error_message="Unable to start RPC Server.")

    async def _get_model_config_rpc(self) -> ModelConfig:
        """Get the ModelConfig object from the RPC Server"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_MODEL_CONFIG,
            expected_type=ModelConfig,
            error_message="Could not get ModelConfig from RPC Server")

    async def _get_decoding_config_rpc(self) -> DecodingConfig:
        """Get DecodingConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_DECODING_CONFIG,
            expected_type=DecodingConfig,
            error_message="Could not get DecodingConfig from RPC Server")

    async def _get_parallel_config_rpc(self) -> ParallelConfig:
        """Get ParallelConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_PARALLEL_CONFIG,
            expected_type=ParallelConfig,
            error_message="Could not get ParallelConfig from RPC Server")

    async def _get_scheduler_config_rpc(self) -> SchedulerConfig:
        """Get SchedulerConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_SCHEDULER_CONFIG,
            expected_type=SchedulerConfig,
            error_message="Could not get SchedulerConfig from RPC Server")

    async def _get_lora_config_rpc(self):
        """Get LoRAConfig from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.GET_LORA_CONFIG,
            expected_type=LoRAConfig,
            error_message="Could not get LoRAConfig from RPC Server")

    async def _is_tracing_enabled_rpc(self) -> ParallelConfig:
        """Get is_tracing_enabled flag from the RPCServer"""

        return await self._send_get_data_rpc_request(
            RPCUtilityRequest.IS_TRACING_ENABLED,
            expected_type=bool,
            error_message="Could not get is_tracing_enabled flag from RPC "
            "Server")

    async def abort(self, request_id: str):
        """Send an ABORT_REQUEST signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCAbortRequest(request_id),
            error_message=f"RPCAbortRequest {request_id} failed")

    async def do_log_stats(self):
        """Send a DO_LOG_STATS signal to the RPC Server"""

        await self._send_one_way_rpc_request(
            request=RPCUtilityRequest.DO_LOG_STATS,
            error_message="RPCRequest DO_LOG_STATS failed.")

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncGenerator[RequestOutput, None]:
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        finished = False
        try:
            with self.socket() as socket:

                # Send RPCGenerateRequest to the RPCServer.
                await socket.send_multipart([
                    cloudpickle.dumps(
                        RPCGenerateRequest(
                            inputs=inputs,
                            sampling_params=sampling_params,
                            request_id=request_id,
                            lora_request=lora_request,
                            trace_headers=trace_headers,
                            prompt_adapter_request=prompt_adapter_request))
                ])

                # Stream back the results from the RPC Server.
                while not finished:
                    message = await socket.recv()
                    request_output = cloudpickle.loads(message)

                    if isinstance(request_output, Exception):
                        raise request_output

                    finished = request_output.finished
                    yield request_output
        finally:
            if not finished:
                await self.abort(request_id)

    async def check_health(self) -> None:
        """Raise if unhealthy"""

        with self.socket() as socket:

            # Ping RPCServer with CHECK_HEALTH request.
            await socket.send(cloudpickle.dumps(RPCUtilityRequest.CHECK_HEALTH)
                              )

            # Await the reply from the server.
            # TODO: do we need an internal timeout here?
            # Or do we expect the external probe to timeout and let this chill?
            health_message = cloudpickle.loads(await socket.recv())

        if isinstance(health_message, Exception):
            raise health_message

        if health_message != VLLM_RPC_HEALTHY_STR:
            raise ValueError("Expected healthy response from backend but got "
                             "f{health_message}")

    async def encode(self, *args,
                     **kwargs) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        raise NotImplementedError(
            "Embeddings not supported with multiprocessing backend")
