import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Mapping, Optional, Union

from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import (CompletionOutput, EmbeddingRequestOutput,
                          RequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import collect_from_async_generator, random_uuid

logger = init_logger(__name__)


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        ...

    @property
    @abstractmethod
    def errored(self) -> bool:
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    async def beam_search(
        self,
        prompt: Union[str, List[int]],
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty

        tokenizer = await self.get_tokenizer(lora_request=None)
        if isinstance(prompt, str):
            tokenized_prompt = tokenizer.encode(prompt)
            prompt_text = prompt
        else:
            tokenized_prompt = prompt
            prompt_text = None
        tokenized_length = len(tokenized_prompt)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        all_beams = [
            BeamSearchSequence(tokens=tokenized_prompt,
                               logprobs=[],
                               cum_logprob=0)
        ]
        completed = []

        for _ in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            tasks = []

            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)

            output = [x[0] for x in output]

            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        new_beam = BeamSearchSequence(
                            tokens=current_beam.tokens + [token_id],
                            logprobs=current_beam.logprobs + [logprobs],
                            cum_logprob=current_beam.cum_logprob +
                            logprob_obj.logprob)

                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(new_beam)
                        else:
                            new_beams.append(new_beam)

            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            beam.text = tokenizer.decode(beam.tokens[tokenized_length:])

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                ) for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=tokenized_prompt,
            prompt_logprobs=None)

        yield beam_search_output

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[EmbeddingRequestOutput, None]:
        """Generate outputs for a request from an embedding model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """

    @abstractmethod
    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_decoding_config(self) -> DecodingConfig:
        ...
        """Get the decoding configuration of the vLLM engine."""

    @abstractmethod
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool:
        ...

    @abstractmethod
    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...
