import itertools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Union, cast, overload)

from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (GuidedDecodingParams, RequestOutputKind,
                                  SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs, is_list_of

logger = init_logger(__name__)


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
    tokens: List[int]
    cum_logprob: float = 0.0
    text: Optional[str] = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: List[BeamSearchSequence]


class BeamSearchInstance:

    def __init__(self, prompt_tokens: List[int]):
        self.beams: List[BeamSearchSequence] = [
            BeamSearchSequence(tokens=prompt_tokens)
        ]
        self.completed: List[BeamSearchSequence] = []


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
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
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
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
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
            mm_processor_kwargs=mm_processor_kwargs,
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
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
        priority: Optional[List[int]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

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
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

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
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    def beam_search(
        self,
        prompts: List[Union[str, List[int]]],
        beam_width: int,
        max_tokens: int,
        ignore_eos: bool = False,
        temperature: float = 0.0,
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            beam_width: The number of beams to keep at each step.
            max_tokens: The max number of tokens to generate for each prompt.
            temperature: The temperature to use for generation.
        
        TODO: how does beam search work together with length penalty, frequency
        penalty, and stopping criteria, etc.?
        """

        tokenizer = self.get_tokenizer()
        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: List[BeamSearchInstance] = []

        for prompt in prompts:
            prompt_tokens = prompt if isinstance(
                prompt, list) else tokenizer.encode(prompt)
            instances.append(BeamSearchInstance(prompt_tokens))

        for _ in range(max_tokens):
            all_beams: List[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: List[Tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(prompts_batch,
                                   sampling_params=beam_search_params,
                                   use_tqdm=False)

            for (start, end), instance in zip(instance_start_and_end,
                                              instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob)

                            if token_id == tokenizer.eos_token_id and \
                                not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
                instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def chat(
        self,
        messages: Union[List[ChatCompletionMessageParam],
                        List[List[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation. 
                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.
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
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be `True`
                if `add_generation_prompt` is also `True`.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """
        list_of_messages: List[List[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is List[List[...]]
            list_of_messages = messages
        else:
            # messages is List[...]
            list_of_messages = [messages]

        prompts: List[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            tokenizer = self.get_tokenizer()
            model_config = self.llm_engine.get_model_config()

            conversation, mm_data = parse_chat_messages(
                msgs, model_config, tokenizer)

            prompt_data: Union[str, List[int]]
            if isinstance(tokenizer, MistralTokenizer):
                prompt_data = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )
            else:
                prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )

            prompt: Union[TokensPrompt, TextPrompt]
            if is_list_of(prompt_data, int):
                prompt = TokensPrompt(prompt_token_ids=prompt_data)
            else:
                prompt = TextPrompt(prompt=prompt_data)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            prompts.append(prompt)

        return self.generate(
            prompts,
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
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
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
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
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
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
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

        parsed_prompts: List[PromptType] = []
        for i in range(num_requests):
            item: PromptType

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            parsed_prompts.append(item)

        return parsed_prompts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _add_guided_params(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options is None:
            return params

        if params.guided_decoding is not None:
            raise ValueError("Cannot set both guided_options_request and"
                             "params.guided_decoding.")

        params.guided_decoding = GuidedDecodingParams(
            json=guided_options.guided_json,
            regex=guided_options.guided_regex,
            choice=guided_options.guided_choice,
            grammar=guided_options.guided_grammar,
            json_object=guided_options.guided_json_object,
            backend=guided_options.guided_decoding_backend,
            whitespace_pattern=guided_options.guided_whitespace_pattern)
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
