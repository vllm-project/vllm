# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.data import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    vllm_config: VllmConfig
    model_config: ModelConfig
    input_processor: InputProcessor
    io_processor: IOProcessor | None

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(
                prompt,
                request_id=request_id,
            )

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []

        for _ in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
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
                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens +
                                    [token_id] if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id))
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs))

            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]
            # Exit early if all beams have completed (hit EOS)
            if len(all_beams) == 0:
                break

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

    @abstractmethod
    def encode(
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
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def get_tokenizer(self) -> TokenizerLike:
        """Get the tokenizer"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

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
        """Stop profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    @abstractmethod
    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight requests
                to finish before pausing. When ``False`` (default), aborts in-flight
                requests immediately.
            clear_cache: Whether to clear KV and prefix caches after draining.
        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError
