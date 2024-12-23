# Copyright 2033-2024 The vLLM team.
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Inspired by https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py

import asyncio
import zmq
import zmq.asyncio

from typing import AsyncGenerator, Dict, List, Mapping, Optional, Type, Union

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine import EngineAbortRequest
from vllm.v1.engine.core import MPEngineCoreClient
from vllm.v1.engine.detokenizer import MPDetokenizerClient
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.utils import zmq_socket_ctx, make_zmq_socket

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:
        assert start_engine_loop

        self.log_requests = log_requests
        self.log_stats = log_stats
        self.stat_loggers = stat_loggers
        self.model_config = vllm_config.model_config

        # RequestId -> OutputQueue.
        self.rid_to_queue: Dict[str, asyncio.Queue[RequestOutput]] = {}

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (in process).
        self.processor = Processor(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            tokenizer=self.tokenizer,
            input_registry=input_registry,
        )

        # IPC paths.
        from_engine_core_path = get_open_zmq_ipc_path()
        to_engine_core_path = get_open_zmq_ipc_path()
        self.to_detokenizer_path = get_open_zmq_ipc_path()
        self.from_detokenizer_path = get_open_zmq_ipc_path()

        # Detokenizer IPC.
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.to_detokenizer = make_zmq_socket(
            self.ctx, self.to_detokenizer_path, zmq.PULL)

        # Detokenizer (background process).
        self.detokenizer_client = MPDetokenizerClient(
            output_path=self.from_detokenizer_path,
            input_path=self.to_detokenizer_path,
            from_engine_core_path=from_engine_core_path,
            to_engine_core_path=to_engine_core_path,
            tokenizer_name=vllm_config.model_config.tokenizer,
            tokenizer_mode=vllm_config.model_config.tokenizer_mode,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
            revision=vllm_config.model_config.tokenizer_revision,
        )

        # EngineCore (background process).
        self.engine_core_client = MPEngineCoreClient(
            input_path=to_engine_core_path,
            output_path=from_engine_core_path,
            vllm_config=vllm_config,
            executor_class=executor_class,
            usage_context=usage_context,
        )

        self.output_handler: Optional[asyncio.Task] = None

    def __del__(self):
        self.shutdown()

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            vllm_config = engine_args.create_engine_config(usage_context)
        else:
            vllm_config = engine_config

        executor_class = cls._get_executor_cls(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        if ctx := getattr(self, "ctx", None):
            ctx.destroy(linger=0)
        
        if output_handler := getattr(self, "output_hander", None):
            output_handler.cancel()

        if engine_core_client := getattr(self, "engine_core_client", None):
            engine_core_client.shutdown()

        if detokenizer_client := getattr(self, "detokenizer_client", None):
            detokenizer_client.shutdown()

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig) -> Type[Executor]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        else:
            assert (distributed_executor_backend is None)
            from vllm.v1.executor.uniproc_executor import UniprocExecutor
            executor_class = UniprocExecutor
        return executor_class

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> asyncio.Queue[RequestOutput]:
        """Add new request to the AsyncLLM."""

        # 1) Convert Input --> EngineRequest.
        engine_request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)

        # 2) Create Queue (output_handler() pushes, generate() pulls).
        self.rid_to_queue[request_id] = asyncio.Queue()

        # 3) Send to Detokenizer (which forwards to EngineCore).
        # Note: we forward the request rather than sending to each
        # process separately to avoid race conditions in Detokenizer.
        await self.to_detokenizer.send_pyobj(engine_request)

        return self.rid_to_queue[request_id]

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Make an output queue for the Request.
            # 2) Processing the Input (e.g. Tokenizer).
            * 3) Adding the Request to Detokenizer + EngineCore.

        The output_handler() loop runs in a background task, pulling
        from Detokenizer and pushing to the per request queue.

        The generate() pulls from the per request queue and yeilds
        to the caller which iterates the AsyncGenerator.
        """

        try:
            # Start output_handler on first request.
            if not self.output_handler:
                loop = asyncio.get_event_loop()
                self.output_handler = loop.create_task(
                    self.output_handler_loop())

            # Add to Detokenizer and EngineCore and makes queue
            # to which the output_handler will push RequestOutputs.
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            while True:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() if q.qsize() > 0 else await q.get()

                # Note: both Detokenizer and EngineCore handle their
                # own request cleanup based on finished.
                if out.finished:
                    del self.rid_to_queue[request_id]
                    yield out
                    break

                yield out

        # Client request cancellation is handled through calling
        # task.cancel() on generate(). Calling self.abort() forwards the 
        # cancellation to the EngineCore and Detokenizer.
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise


    async def output_handler_loop(self):
        """Background loop: pulls from Detokenizer and push to Queues."""

        with zmq_socket_ctx(self.from_detokenizer_path, zmq.PULL) as socket:
            while True:
                # Note: use socket directly to avoid calling await multiple
                # times, which causes too much task switching at high QPS.
                outputs: List[RequestOutput] = []
                outputs = await socket.recv_pyobj()

                for out in outputs:
                    # Note: it is possible that a request was aborted
                    # due to client cancellation while EngineCoreOutputs
                    # are still flowing, so we just ignore.
                    if out.request_id in self.rid_to_queue:
                        self.rid_to_queue[out.request_id].put_nowait(out)
            

    async def abort(self, request_id: str):
        """Abort request if the client cancels the request."""

        # Send abort to Detokenizer (which will fwd to EngineCore).
        await self.detokenizer_client.input_socket.send_pyobj(
            EngineAbortRequest([request_id]))

        # Remove from request output queues.
        if request_id in self.rid_to_queue:
            del self.rid_to_queue[request_id]

        if self.log_requests:
            logger.info("Aborted %s.", request_id)

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        raise ValueError("Not Supported on V1 yet.")

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.tokenizer.get_lora_tokenizer(lora_request)

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        logger.debug("Called do_log_stats.")

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        await self.engine_core_client.profile_async(True)

    async def stop_profile(self) -> None:
        await self.engine_core_client.profile_async(False)

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return Exception()  # TODO: implement
