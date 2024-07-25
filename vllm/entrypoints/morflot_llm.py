import asyncio
from asyncio import Queue as queue
from asyncio import QueueEmpty
from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm import envs
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import (PromptInputs, PromptStrictInputs, TextPrompt,
                         TextTokensPrompt, TokensPrompt,
                         parse_and_batch_prompt)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, async_rpd_trace, deprecate_kwargs

logger = init_logger(__name__)


class MorflotLLM:

    def __init__(
        self,
        engine_args: EngineArgs,
        input_queue: queue = None,
        sampling_params: SamplingParams = SamplingParams(),
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

        self.input_queue = input_queue
        self.sampling_params = sampling_params
        self.finish = False
        self.result_queues = {}
        self.need_restart = False

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
        result_queue: queue,
    ) -> None:
        self.result_queues[request_id] = result_queue
        if isinstance(inputs, list):
            inputs = TextTokensPrompt(prompt_token_ids=inputs)
        self.llm_engine.add_request(request_id, inputs, params)

    async def _poll_requests(self):
        while True:
            if not self.llm_engine.has_unfinished_requests():
                #broadcast_tensor_dict({}, src=0)
                logger.info("No unfinished requests. Waiting...")
                (request_id, prompt, sampling_params,
                 result_queue) = await self.input_queue.get()
                if self.need_restart:
                    for worker in self.llm_engine.model_executor.workers:
                        worker.execute_method("start_worker_execution_loop")
                    self.need_restart = False

            else:
                try:
                    (request_id, prompt, sampling_params,
                     result_queue) = self.input_queue.get_nowait()
                except QueueEmpty:
                    break
            self._add_request(prompt, sampling_params, request_id,
                              result_queue)

    #@async_rpd_trace()
    async def run_engine(self):
        steps_before_yield = envs.VLLM_ENGINE_STEPS_BEFORE_YIELD
        request_stats = {}
        while True:
            await self._poll_requests()
            step_outputs = self.llm_engine.step()
            if not self.llm_engine.has_unfinished_requests():
                logger.info("Broadcast stop")
                broadcast_tensor_dict({}, src=0)
                self.need_restart = True
                steps_before_yield = 0
            for output in step_outputs:
                assert len(output.outputs) == 1
                output_len = len(output.outputs[0].text)
                result_queue = self.result_queues[output.request_id]
                stats = None
                if output_len >= 0 and (output.request_id
                                        not in request_stats):
                    request_stats[output.request_id] = output_len
                    result = output.outputs[0].text
                    steps_before_yield = min(
                        steps_before_yield, envs.VLLM_ENGINE_STEPS_FIRST_TOKEN)
                else:
                    result = output.outputs[0].text[
                        request_stats[output.request_id]:output_len]
                if output.finished:
                    # signal end of stream with None
                    stats = {
                        "prompt": len(output.prompt_token_ids),
                        "tokens": len(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                        "stop_reason": output.outputs[0].stop_reason,
                    }
                    del request_stats[output.request_id]
                    del self.result_queues[output.request_id]
                    steps_before_yield = min(
                        steps_before_yield,
                        envs.VLLM_ENGINE_STEPS_COMPLETED_REQUEST)
                else:
                    request_stats[output.request_id] = output_len
                result_queue.put_nowait((output.request_id, result, stats))
            steps_before_yield -= 1
            if steps_before_yield <= 0:
                logger.info(
                    f"Engine yield. Requests: {self.llm_engine.get_num_unfinished_requests()}"
                )
                steps_before_yield = envs.VLLM_ENGINE_STEPS_BEFORE_YIELD
                await asyncio.sleep(0)
