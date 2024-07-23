import multiprocessing as mp
import queue
from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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
from vllm.utils import Counter, deprecate_kwargs

logger = init_logger(__name__)


class QueueLLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", "squeezellm", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine_args`)
    
    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = False
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 32768,
        disable_custom_all_reduce: bool = False,
        *,
        input_queue: mp.Queue = None,
        first_token_queue: mp.Queue = None,
        result_queue: mp.Queue = None,
        sampling_params: SamplingParams = SamplingParams(),
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

        self.input_queue = input_queue
        self.first_token_queue = first_token_queue
        self.result_queue = result_queue
        self.sampling_params = sampling_params
        self.finish = False


    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def start(self, use_tqdm: bool = True):
        self._pull_tokens_from_input_queue(block=True)
        self._run_engine(use_tqdm=use_tqdm)

    def _pull_tokens_from_input_queue(self, block: bool = True):
        try:
            input = self.input_queue.get() if block else self.input_queue.get_nowait()
            if input is None:
                self.finish = True
            for sample_id, token_ids in input:
                inputs = self._convert_v1_inputs(
                    prompts=None,
                    prompt_token_ids=token_ids,
                    multi_modal_data=None,
                )

                self._validate_and_add_requests(
                    inputs=inputs,
                    params=self.sampling_params,
                    request_id=sample_id,
                )
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Unexpected exception during pulling tokens: {e}")


    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
        multi_modal_data: Optional[MultiModalData],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        inputs: List[PromptInputs] = []
        for i in range(num_requests):
            if prompts is not None:
                if prompt_token_ids is not None:
                    item = TextTokensPrompt(
                        prompt=prompts[i],
                        prompt_token_ids=prompt_token_ids[i])
                else:
                    item = TextPrompt(prompt=prompts[i])
            else:
                if prompt_token_ids is not None:
                    item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
                else:
                    raise AssertionError

            if multi_modal_data is not None:
                item["multi_modal_data"] = multi_modal_data

            inputs.append(item)

        return inputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        request_id: str,
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                request_id=request_id,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        request_id: str = None,
    ) -> None:
        self.llm_engine.add_request(request_id,
                                    inputs,
                                    params,
                                    lora_request=None)

    def _run_engine(
            self, *, use_tqdm: bool=False
    ):
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=f"Generation Speed: {0:.2f} toks/s",
            )
        # Run the engine.
        total_toks = 0
        request_stats = {}
        while not self.finish and self.llm_engine.has_unfinished_requests():
            self._pull_tokens_from_input_queue(block=False)
            logger.info(f"Performing engine step. Requests: {self.llm_engine.get_num_unfinished_requests()}")
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                output_len = len(output.outputs[0].token_ids)
                if output_len > 0 and (output.request_id not in request_stats):
                    if self.first_token_queue is not None:
                        self.first_token_queue.put((output.request_id, output.outputs[0].token_ids))
                    request_stats[output.request_id] = output_len
                if request_stats[output.request_id] < output_len:
                    self.result_queue.put_nowait((output.request_id, output.outputs[0].token_ids[request_stats[output.request_id]: output_len]))
                    if output.finished:
                        # signal end of stream with None
                        self.result_queue.put_nowait((output.request_id, None))
                        del request_stats[output.request_id]
                        if use_tqdm:
                            if isinstance(output, RequestOutput):
                                # Calculate tokens only for RequestOutput
                                total_toks += sum(
                                    len(stp.token_ids) for stp in output.outputs)
                                spd = total_toks / pbar.format_dict["elapsed"]
                                pbar.postfix = f"Generation Speed: {spd:.2f} toks/s"
                            pbar.update(1)
                    else:
                        request_stats[output.request_id] = output_len
        if use_tqdm:
            pbar.close()
