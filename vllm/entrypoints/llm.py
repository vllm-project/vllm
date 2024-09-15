from contextlib import contextmanager
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import PromptInputs, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    GuidedDecodingRequest, get_local_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs, is_list_of

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
            we support "awq", "gptq", and "fp8" (experimental).
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
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
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
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False for decoder-only models and True
        for encoder/decoder models, since encoder/decoder models
        do not currently support CUDAGraph.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = (
            "image_token_id",
            "image_feature_size",
            "image_input_shape",
            "image_input_type",
        )
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore.")
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
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

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
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompts",
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'inputs' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptInputs, Sequence[PromptInputs]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None
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
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).")

        if prompt_token_ids is not None:
            inputs = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> List[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A single conversation represented as a list of messages.
                Each message is a dictionary with 'role' and 'content' keys.
            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            add_generation_prompt: If True, adds a generation template
                to each message.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """

        tokenizer = self.get_tokenizer()
        model_config = self.llm_engine.get_model_config()

        conversation, mm_data = parse_chat_messages(messages, model_config,
                                                    tokenizer)

        prompt: Union[str, List[int]]
        if isinstance(tokenizer, MistralTokenizer):
            prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
            )

        inputs: PromptInputs
        if is_list_of(prompt, int):
            inputs = TokensPrompt(prompt_token_ids=prompt)
        else:
            inputs = TextPrompt(prompt=prompt)

        if mm_data is not None:
            inputs["multi_modal_data"] = mm_data

        return self.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

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
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        /,  # We may enable `inputs` keyword after removing the old API
        *,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompts",
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'inputs' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptInputs, Sequence[PromptInputs]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            inputs: The inputs to the LLM. You may pass a sequence of inputs for
                batch inference. See :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

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
            inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, EmbeddingRequestOutput)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

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
            item: PromptInputs

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            inputs.append(item)

        return inputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
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

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_processor(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            self._add_request(
                request_inputs,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            inputs,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

    def _add_guided_processor(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options:
            if guided_options.guided_decoding_backend is None:
                decoding_config = self.llm_engine.get_decoding_config()
                guided_options.guided_decoding_backend = (
                    decoding_config.guided_decoding_backend)
            guided_logits_processor = get_local_guided_decoding_logits_processor(  #noqa
                guided_options.guided_decoding_backend, guided_options,
                self.get_tokenizer())
            if guided_logits_processor:
                if params.logits_processors is None:
                    params.logits_processors = []
                params.logits_processors.append(guided_logits_processor)
        return params

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
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))

    def _is_encoder_decoder_model(self):
        return self.llm_engine.is_encoder_decoder_model()

    def _is_embedding_model(self):
        return self.llm_engine.is_embedding_model()
