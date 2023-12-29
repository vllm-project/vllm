from vllm.engine.llava_engine import LLaVAEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine, AsyncStream, AsyncEngineDeadError
import asyncio
import time
from typing import (List, Optional, Type, AsyncIterator)
from PIL import Image
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class _AsyncLLaVAEngine(LLaVAEngine, _AsyncLLMEngine):

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.

        This rewriting of the function is to sent the runner_method to model 
        runner then knowing that it is a llava model. It won't be  needed in the 
        future when we merge the execute_llava_model function to the 
        execute_model.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        # Execute the model.
        output = (await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            runner_method="execute_llava_model",
        )) if not scheduler_outputs.is_empty() else []

        return self._process_model_outputs(output, scheduler_outputs)


class AsyncLLaVAEngine(AsyncLLMEngine):

    _engine_class: Type[_AsyncLLaVAEngine] = _AsyncLLaVAEngine

    async def add_request(
            self,
            request_id: str,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: Optional[float] = None,
            images: Optional[List[Image.Image]] = None) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}."
                        f"images: {0 if images is None else len(images)}")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            images=images)

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        images: Optional[List[Image.Image]] = None
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            images: A list of PIL images for the prompt. It supports multiple
                images, although most llava models are trained with only one image.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_request(request_id,
                                            prompt,
                                            sampling_params,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time,
                                            images=images)

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e
