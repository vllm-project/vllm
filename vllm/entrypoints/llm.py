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
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs

logger = init_logger(__name__)


class LLM:
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
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
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

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            self.llm_engine.tokenizer.tokenizer = tokenizer
        else:
            self.llm_engine.tokenizer.tokenizer = get_cached_tokenizer(
                tokenizer)

    @overload  # LEGACY: single (prompt + optional token ids)
    def generate(
        self,
        prompts: str,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[str] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def generate(
        self,
        prompts: None,
        sampling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload
    def generate(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs("prompts",
                      "prompt_token_ids",
                      is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
                      additional_message="Please use the 'inputs' parameter "
                      "instead.")
    def generate(
        self,
        prompts: Union[Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: A list of inputs to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters. 
                When it is a single value, it is applied to every prompt. 
                When it is a list, the list must have the same length as the 
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for generation models "
                "(XForCausalLM).")

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(
                Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=sampling_params,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    @overload  # LEGACY: single (prompt + optional token ids)
    def encode(
        self,
        prompts: str,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def encode(
        self,
        prompts: List[str],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[str] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[List[str]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def encode(
        self,
        prompts: None,
        pooling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload
    def encode(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @deprecate_kwargs("prompts",
                      "prompt_token_ids",
                      is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
                      additional_message="Please use the 'inputs' parameter "
                      "instead.")
    def encode(
        self,
        prompts: Union[Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: The inputs to the LLM. You may pass a sequence of inputs for
                batch inference. See :class:`~vllm.inputs.PromptStrictInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            generated embeddings in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if not self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.encode() is only supported for embedding models (XModel)."
            )

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(
                Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
                prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, EmbeddingRequestOutput)

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
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

            inputs.append(item)

        return inputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptStrictInputs, Sequence[PromptStrictInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
    ) -> None:
        if isinstance(inputs, (str, dict)):
            # Convert a single prompt to a list.
            inputs = [inputs]

        num_requests = len(inputs)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    inputs,
                                    params,
                                    lora_request=lora_request)

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
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
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            total_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            spd = total_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = f"Generation Speed: {spd:.2f} toks/s"
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
