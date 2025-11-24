# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.streaming.engine.streaming_output_processor import StreamingOutputProcessor

logger = init_logger(__name__)


class StreamingAsyncLLM(AsyncLLM):
    output_processor_cls: type[OutputProcessor] = StreamingOutputProcessor

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        close_session: bool = False,
    ) -> AsyncGenerator[RequestOutput, None]:
        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and sampling_params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        try:
            self._run_output_handler()

            tokenization_kwargs: dict[str, Any] = {}
            truncate_prompt_tokens = sampling_params.truncate_prompt_tokens

            _validate_truncation_size(
                self.model_config.max_model_len,
                truncate_prompt_tokens,
                tokenization_kwargs,
            )

            q = await self.add_request(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
                close_session=close_session,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                tokenization_kwargs=tokenization_kwargs,
                data_parallel_rank=data_parallel_rank,
            )

            finished = False
            while not finished:
                out = q.get_nowait() or await q.get()
                finished = out.finished
                if (
                    len(out.outputs) > 0
                    and out.outputs[0].stop_reason == "close_session"
                ):
                    return
                yield out

        except asyncio.CancelledError:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise
        except GeneratorExit:
            if self.log_requests:
                logger.info(
                    "Request %s generator completed normally, session remains alive.",
                    request_id,
                )
            # For streaming sessions, generator completion is normal
            return

        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError("pooling models are not supported for streaming")
