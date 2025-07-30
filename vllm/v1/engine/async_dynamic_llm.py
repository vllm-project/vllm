# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Any, Optional, Union

import numpy as np

from argparse import Namespace
import vllm.envs as envs
from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.inputs import PromptType
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.output_processor import (OutputProcessor,
                                             RequestOutputCollector)
from vllm.v1.metrics.loggers import StatLoggerFactory
from vllm.entrypoints.openai.api_server import init_app_state


logger = init_logger(__name__)

class AsyncDynamicLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_index: int = 0,
    ) -> None:
        self.vllm_config= vllm_config
        self.start_engine_loop= start_engine_loop
        self.usage_context= usage_context
        self.stat_loggers= stat_loggers
        self.disable_log_requests= disable_log_requests
        self.disable_log_stats= disable_log_stats
        self.client_addresses=client_addresses
        self.client_index= client_index
        self.engine = None


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
        return EngineDeadError()

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        return self.engine.generate(prompt, sampling_params, request_id, lora_request,
                           trace_headers, priority)


    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """
        ...

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        return self.vllm_config


    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...


    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        ...


    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        ...

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    async def is_tracing_enabled(self) -> bool:
        ...

    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[list[SamplerOutput]] = None,
    ) -> None:
        ...

    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        """Reset the prefix cache"""
        ...

    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Wake up the engine"""
        ...

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    async def scale_elastic_ep(self,
                               new_data_parallel_size: int,
                               drain_timeout: int = 300) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def load_model(self, model: str, state) -> None:
        """Load the model"""
        self.vllm_config.model_config.with_model(model)
        self.engine = AsyncLLM.from_vllm_config(
                    vllm_config=self.vllm_config,
                    usage_context=self.usage_context,
                    disable_log_requests=self.disable_log_requests,
                    disable_log_stats=self.disable_log_stats,
                    client_addresses=self.client_addresses,
                    client_index=self.client_index)

        # Don't keep the dummy data in memory
        await self.engine.reset_mm_cache()

        # Update app state
        state.args.model = model
        await init_app_state(self.engine, self.vllm_config, state, state.args)


    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ()

    def shutdown(self):
        pass
