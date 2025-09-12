# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.beam_search import (BeamSearchInstance, BeamSearchOutput,
                              BeamSearchSequence,
                              create_sort_beams_key_function)
from vllm.config import (CompilationConfig, ModelDType, TokenizerMode,
                         is_init_field)
from vllm.engine.arg_utils import (ConvertOption, EngineArgs, HfOverrides,
                                   PoolerConfig, RunnerOption)
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.score_utils import (ScoreContentPartParam,
                                          ScoreMultiModalParam,
                                          _cosine_similarity,
                                          _validate_score_input_lens,
                                          compress_token_type_ids,
                                          get_score_prompt)
# yapf: enable
from vllm.entrypoints.utils import (_validate_truncation_size,
                                    log_non_default_args)
from vllm.inputs import (DataPrompt, PromptType, SingletonPrompt, TextPrompt,
                         TokensPrompt)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (BeamSearchParams, RequestOutputKind,
                                  SamplingParams)
from vllm.tasks import PoolingTask
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, Device, as_iter, is_list_of
from vllm.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Metric

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


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
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
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
        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,
            this is set to None and vllm can automatically infer the kv cache
            size based on gpu_memory_utilization. However, users may want to
            manually specify the kv cache memory size. kv_cache_memory_bytes
            allows more fine-grain control of how much memory gets used when
            compared with using gpu_memory_memory_utilization. Note that
            kv_cache_memory_bytes (when not-None) ignores
            gpu_memory_utilization
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Noting that `best_of` is only supported in V0. Otherwise, too small
            values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See
            [ParallelConfig][vllm.config.ParallelConfig].
        disable_async_output_proc: Disable async output processing.
            This may result in lower performance.
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `huggingface-cli login` (stored in `~/.huggingface`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor. Overrides for the
            multi-modal processor obtained from `AutoProcessor.from_pretrained`.
            The available overrides depend on the model that is being run.
            For example, for Phi-3-Vision: `{"num_crops": 4}`.
        override_pooler_config: Initialize non-default pooling config or
            override default pooling config for the pooling model.
            e.g. `PoolerConfig(pooling_type="mean", normalize=False)`.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the level of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        **kwargs: Arguments for [`EngineArgs`][vllm.EngineArgs].

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the [AsyncLLMEngine][vllm.AsyncLLMEngine] class instead.
    """

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: Optional[str] = None,
        tokenizer_mode: TokenizerMode = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: Optional[QuantizationMethods] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        override_pooler_config: Optional[PoolerConfig] = None,
        kv_cache_memory_bytes: Optional[int] = None,
        compilation_config: Optional[Union[int, dict[str, Any],
                                           CompilationConfig]] = None,
        logits_processors: Optional[list[Union[str,
                                               type[LogitsProcessor]]]] = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            logits_processors=logits_processors,
            **kwargs,
        )

        log_non_default_args(engine_args)

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

        if envs.VLLM_USE_V1:
            supported_tasks = self.llm_engine \
                .get_supported_tasks()  # type: ignore
        else:
            supported_tasks = self.llm_engine.model_config.supported_tasks

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(self.llm_engine.vllm_config,
                                             io_processor_plugin)

    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group().get_lora_tokenizer(
            lora_request)

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group()

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = (
                self.llm_engine.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        priority: Optional[list[int]] = None,
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using `prompts` and `prompt_token_ids` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the `inputs` parameter.
        """
        model_config = self.llm_engine.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model.")

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        # Add any modality specific loras to the corresponding prompts
        lora_request = self._get_modality_specific_lora_reqs(
            prompts, lora_request)

        self._validate_and_add_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def _get_modality_specific_lora_reqs(
            self, prompts: Union[PromptType, Sequence[PromptType]],
            lora_request: Optional[Union[list[LoRARequest], LoRARequest]]):
        # Grab the lora config off the vllm config on the engine,
        # since this is the same for both v0 & v1.
        lora_config = self.llm_engine.vllm_config.lora_config

        # If there's no lora config / default_mm_loras, or the model
        # isn't multimodal, leave the lora as is.
        if (lora_config is None
                or not self.llm_engine.model_config.is_multimodal_model
                or (lora_config and lora_config.default_mm_loras is None)):
            return lora_request

        if not isinstance(prompts, Sequence):
            prompts = [prompts]

        optional_loras = ([lora_request] * len(prompts)
                          if not isinstance(lora_request, Sequence) else
                          lora_request)

        return [
            self._resolve_single_prompt_mm_lora(
                prompt,
                opt_lora_req,
                lora_config.default_mm_loras,
            ) for prompt, opt_lora_req in zip(prompts, optional_loras)
        ]

    def _resolve_single_prompt_mm_lora(self, prompt: PromptType,
                                       lora_request: Optional[LoRARequest],
                                       default_mm_loras: Optional[dict[str,
                                                                       str]]):
        if (not default_mm_loras or not isinstance(prompt, dict)
                or "multi_modal_data" not in prompt):
            return lora_request

        prompt = cast(Union[TextPrompt, TokensPrompt], prompt)

        intersection = set(prompt["multi_modal_data"].keys()) \
            .intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request
        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be"
                " used by a single prompt consuming several modalities; "
                " currently we only support one lora per request; as such,"
                " lora(s) registered with modalities: %s"
                " will be skipped", intersection)
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt")
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """
        executor = self.llm_engine.model_executor
        return executor.apply_model(func)

    def _get_beam_search_lora_requests(
        self,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]],
        prompts: list[Union[TokensPrompt, TextPrompt]],
    ) -> list[Optional[LoRARequest]]:
        """Get the optional lora request corresponding to each prompt."""
        if isinstance(lora_request,
                      Sequence) and len(lora_request) != len(prompts):
            raise ValueError(
                "Lora request list should be the same length as the prompts")

        if lora_request is None or isinstance(lora_request, LoRARequest):
            return [lora_request] * len(prompts)

        raise TypeError(f"Invalid lora_request type {type(lora_request)}")

    def beam_search(
        self,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: BeamSearchParams,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        use_tqdm: bool = False,
        concurrency_limit: Optional[int] = None,
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.
        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        lora_requests = self._get_beam_search_lora_requests(
            lora_request, prompts)

        tokenizer = self.get_tokenizer()
        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id,
            length_penalty,
        )

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar.")
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(prompts)

        def create_tokens_prompt_from_beam(
                beam: BeamSearchSequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {
                "prompt_token_ids": beam.tokens
            }
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs[
                    "mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, prompts):
            # Add multimodal processor kwargs & data
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt[
                    "mm_processor_kwargs"]

            if "prompt_token_ids" in prompt:
                prompt = cast(TokensPrompt, prompt)  # Needed for mypy
                prompt_tokens = prompt["prompt_token_ids"]
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"])

            instances.append(
                BeamSearchInstance(
                    prompt_tokens,
                    lora_request=lora_req,
                    logprobs=None,
                    **mm_kwargs,
                ), )

        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start:prompt_start +
                                        concurrency_limit]

            token_iter = range(max_tokens)
            if use_tqdm:
                token_iter = tqdm(token_iter,
                                  desc="Beam search",
                                  unit="token",
                                  unit_scale=False)
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress.")
            for _ in token_iter:
                all_beams: list[BeamSearchSequence] = list(
                    sum((instance.beams for instance in instances_batch), []))
                pos = [0] + list(
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch))
                instance_start_and_end: list[tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:]))

                if len(all_beams) == 0:
                    break

                # create corresponding batch entries for prompt & optional lora
                prompts_batch, lora_req_batch = zip(
                    *[(create_tokens_prompt_from_beam(beam), beam.lora_request)
                      for beam in all_beams])

                # only runs for one step
                # we don't need to use tqdm here
                output = self.generate(prompts_batch,
                                       sampling_params=beam_search_params,
                                       use_tqdm=False,
                                       lora_request=lora_req_batch)

                for (start, end), instance in zip(instance_start_and_end,
                                                  instances_batch):
                    instance_new_beams = []
                    for i in range(start, end):
                        current_beam = all_beams[i]
                        result = output[i]

                        if result.outputs[0].logprobs is not None:
                            # if `result.outputs[0].logprobs` is None, it means
                            # the sequence is completed because of the
                            # max-model-len or abortion. we don't need to add
                            # it to the new beams.
                            logprobs = result.outputs[0].logprobs[0]
                            for token_id, logprob_obj in logprobs.items():
                                new_beam = BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    lora_request=current_beam.lora_request,
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs)

                                if token_id == tokenizer.eos_token_id and \
                                    not ignore_eos:
                                    instance.completed.append(new_beam)
                                else:
                                    instance_new_beams.append(new_beam)
                    sorted_beams = sorted(instance_new_beams,
                                          key=sort_beams_key,
                                          reverse=True)
                    instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=sort_beams_key,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def preprocess_chat(
        self,
        messages: Union[list[ChatCompletionMessageParam],
                        list[list[ChatCompletionMessageParam]]],
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[TokensPrompt]:
        """
        Generate prompt for a chat conversation. The pre-processed
        prompt can then be used as input for the other LLM methods.

        Refer to `chat` for a complete description of the arguments.
        Returns:
            A list of `TokensPrompts` objects containing the tokenized
            prompt after chat template interpolation, and the
            pre-processed multi-modal inputs.
        """
        list_of_messages: list[list[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is list[list[...]]
            list_of_messages = cast(list[list[ChatCompletionMessageParam]],
                                    messages)
        else:
            # messages is list[...]
            list_of_messages = [
                cast(list[ChatCompletionMessageParam], messages)
            ]

        tokenizer = self.get_tokenizer(lora_request)
        model_config = self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            model_config=model_config,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        prompts: list[TokensPrompt] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data, mm_uuids = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format=resolved_content_format,
            )

            if isinstance(tokenizer, MistralTokenizer):
                prompt_token_ids = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    **_chat_template_kwargs,
                )
            else:
                prompt_str = apply_hf_chat_template(
                    tokenizer=tokenizer,
                    conversation=conversation,
                    model_config=model_config,
                    **_chat_template_kwargs,
                )
                # Special tokens are already included in chat templates so
                # should not be added by the tokenizer in this case.
                prompt_token_ids = tokenizer.encode(prompt_str,
                                                    add_special_tokens=False)

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            if mm_uuids is not None:
                prompt["multi_modal_uuids"] = mm_uuids

            if mm_processor_kwargs is not None:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs

            prompts.append(prompt)

        return prompts

    def chat(
        self,
        messages: Union[list[ChatCompletionMessageParam],
                        list[list[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

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
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
                If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

                - "string" will render the content as a string.
                  Example: `"Who are you?"`
                - "openai" will render the content as a list of dictionaries,
                  similar to OpenAI schema.
                  Example: `[{"type": "text", "text": "Who are you?"}]`

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                `True` if `add_generation_prompt` is also `True`.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """

        prompts = self.preprocess_chat(
            messages=messages,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    def encode(
        self,
        prompts: Union[PromptType, Sequence[PromptType], DataPrompt],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        pooling_task: PoolingTask = "encode",
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: overrides tokenization_kwargs set in
                pooling_params

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.

        Note:
            Using `prompts` and `prompt_token_ids` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the `inputs` parameter.
        """
        if pooling_task is None:
            if "embed" in self.supported_tasks:
                pooling_task = "embed"
            else:
                pooling_task = "encode"

            logger.warning_once(
                "`LLM.encode` is currently using `pooling_task = %s`.\n"
                "Please use one of the more specific methods or set the "
                "task directly when using `LLM.encode`:\n"
                "  - For embeddings, use `LLM.embed(...)` "
                "or `pooling_task=\"embed\"`.\n"
                "  - For classification logits, use `LLM.classify(...)` "
                "or `pooling_task=\"classify\"`.\n"
                "  - For rewards, use `LLM.reward(...)` "
                "or `pooling_task=\"reward\"`\n"
                "  - For similarity scores, use `LLM.score(...)`.",
                pooling_task)

        model_config = self.llm_engine.model_config
        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model.")

        if pooling_task not in self.supported_tasks:
            raise ValueError(
                f"pooling_task must be one of {self.supported_tasks}.")

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        for param in as_iter(pooling_params):
            param.verify(pooling_task, model_config)
            # for backwards compatibility
            if truncate_prompt_tokens is not None:
                param.truncate_prompt_tokens = truncate_prompt_tokens

        io_processor_prompt = False
        if isinstance(prompts, dict) and "data" in prompts:
            io_processor_prompt = True
            if self.io_processor is None:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details.")

            # Validate the request data is valid for the loaded plugin
            validated_prompt = self.io_processor.parse_request(prompts)

            # obtain the actual model prompts from the pre-processor
            prompts = self.io_processor.pre_process(prompt=validated_prompt)

        self._validate_and_add_requests(
            prompts=prompts,
            params=pooling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)

        model_outputs = self.engine_class.validate_outputs(
            outputs, PoolingRequestOutput)

        if io_processor_prompt:
            # get the post-processed model outputs
            assert self.io_processor is not None
            processed_outputs = self.io_processor.post_process(
                model_output=model_outputs)

            return [
                PoolingRequestOutput[Any](request_id="",
                                          outputs=processed_outputs,
                                          prompt_token_ids=[],
                                          finished=True)
            ]
        else:
            return model_outputs

    def embed(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "embed" not in self.supported_tasks:
            raise ValueError(
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`.")

        items = self.encode(
            prompts,
            truncate_prompt_tokens=truncate_prompt_tokens,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "classify" not in self.supported_tasks:
            raise ValueError(
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`.")

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="classify",
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def reward(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[PoolingRequestOutput]:
        """
        Generate rewards for each prompt.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        return self.encode(
            prompts,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            truncate_prompt_tokens=truncate_prompt_tokens,
            pooling_task="encode",
        )

    def _embedding_score(
        self,
        tokenizer: AnyTokenizer,
        text_1: list[Union[str, TextPrompt, TokensPrompt]],
        text_2: list[Union[str, TextPrompt, TokensPrompt]],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[PoolingParams] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[ScoringRequestOutput]:

        encoded_output: list[PoolingRequestOutput] = self.encode(
            text_1 + text_2,
            truncate_prompt_tokens=truncate_prompt_tokens,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="embed",
        )

        encoded_output_1: list[PoolingRequestOutput] = encoded_output[
            0:len(text_1)]
        encoded_output_2: list[PoolingRequestOutput] = encoded_output[
            len(text_1):]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        scores = _cosine_similarity(tokenizer=tokenizer,
                                    embed_1=encoded_output_1,
                                    embed_2=encoded_output_2)

        items = self.engine_class.validate_outputs(scores,
                                                   PoolingRequestOutput)
        return [ScoringRequestOutput.from_base(item) for item in items]

    def _cross_encoding_score(
        self,
        tokenizer: AnyTokenizer,
        data_1: Union[list[str], list[ScoreContentPartParam]],
        data_2: Union[list[str], list[ScoreContentPartParam]],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[PoolingParams] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[ScoringRequestOutput]:
        model_config = self.llm_engine.model_config

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "Score API is not supported for Mistral tokenizer")

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        if pooling_params is None:
            pooling_params = PoolingParams(task="score")

        model_config = self.llm_engine.model_config
        pooling_params.verify("score", model_config)
        pooling_params_list = list[PoolingParams]()

        tokenization_kwargs: dict[str, Any] = {}

        _validate_truncation_size(model_config.max_model_len,
                                  truncate_prompt_tokens, tokenization_kwargs)

        prompts = list[PromptType]()

        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]

        model_config = self.llm_engine.model_config

        for q, d in input_pairs:
            _, engine_prompt = get_score_prompt(
                model_config=model_config,
                data_1=q,
                data_2=d,
                tokenizer=tokenizer,
                tokenization_kwargs=tokenization_kwargs,
            )

            if (token_type_ids := engine_prompt.pop("token_type_ids", None)):
                params = pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                params.extra_kwargs = {"compressed_token_type_ids": compressed}
                pooling_params_list.append(params)
            else:
                pooling_params_list.append(pooling_params)

            prompts.append(engine_prompt)

        self._validate_and_add_requests(
            prompts=prompts,
            params=pooling_params_list,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        items = self.engine_class.validate_outputs(outputs,
                                                   PoolingRequestOutput)

        return [ScoringRequestOutput.from_base(item) for item in items]

    def score(
        self,
        data_1: Union[SingletonPrompt, Sequence[SingletonPrompt],
                      ScoreMultiModalParam],
        data_2: Union[SingletonPrompt, Sequence[SingletonPrompt],
                      ScoreMultiModalParam],
        /,
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        pooling_params: Optional[PoolingParams] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs `<text,text_pair>` or
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """
        model_config = self.llm_engine.model_config
        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model.")

        supported_tasks = self.supported_tasks
        if all(t not in supported_tasks for t in ("embed", "classify")):
            raise ValueError("Score API is not supported by this model. "
                             "Try converting the model using "
                             "`--convert embed` or `--convert classify`.")

        if (model_config.is_cross_encoder
                and getattr(model_config.hf_config, "num_labels", 0) != 1):
            raise ValueError("Score API is only enabled for num_labels == 1.")

        # the tokenizer for models such as
        # "cross-encoder/ms-marco-MiniLM-L-6-v2" doesn't support passing
        # lists of tokens to the `text` and `text_pair` kwargs
        tokenizer = self.get_tokenizer()

        if not model_config.is_multimodal_model:

            def check_data_type(data: Union[SingletonPrompt,
                                            Sequence[SingletonPrompt],
                                            ScoreMultiModalParam]):
                if isinstance(data, dict) and "content" in data:
                    raise ValueError("ScoreMultiModalParam is not supported "
                                     f"for {model_config.architecture}")

            check_data_type(data_1)
            check_data_type(data_2)

            def ensure_str(prompt: SingletonPrompt):
                if isinstance(prompt, dict):
                    if "multi_modal_data" in prompt:
                        raise ValueError("Multi-modal prompt is not "
                                         "supported for scoring")
                    elif "prompt_token_ids" in prompt:
                        prompt = tokenizer.decode(
                            cast(TokensPrompt, prompt)["prompt_token_ids"])
                    elif "prompt" in prompt:
                        prompt = cast(TextPrompt, prompt)["prompt"]
                assert type(prompt) is str
                return prompt

            if isinstance(data_1, (str, dict)):
                # Convert a single prompt to a list.
                data_1 = [data_1]  # type: ignore[list-item]

            data_1 = [ensure_str(t) for t in data_1]

            if isinstance(data_2, (str, dict)):
                # Convert a single prompt to a list.
                data_2 = [data_2]  # type: ignore[list-item]

            data_2 = [ensure_str(t) for t in data_2]

        if isinstance(data_1, dict) and "content" in data_1:
            data_1 = data_1.get("content")  # type: ignore[assignment]
        elif isinstance(data_1, str):
            data_1 = [data_1]

        if isinstance(data_2, dict) and "content" in data_2:
            data_2 = data_2.get("content")  # type: ignore[assignment]
        elif isinstance(data_2, str):
            data_2 = [data_2]

        _validate_score_input_lens(data_1, data_2)  # type: ignore[arg-type]

        if model_config.is_cross_encoder:
            return self._cross_encoding_score(
                tokenizer,
                data_1,  # type: ignore[arg-type]
                data_2,  # type: ignore[arg-type]
                truncate_prompt_tokens,
                use_tqdm,
                pooling_params,
                lora_request)
        else:
            return self._embedding_score(
                tokenizer,
                data_1,  # type: ignore[arg-type]
                data_2,  # type: ignore[arg-type]
                truncate_prompt_tokens,
                use_tqdm,
                pooling_params,
                lora_request)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.llm_engine.reset_prefix_cache(device)

    def sleep(self, level: int = 1):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache. The content of kv cache
                is forgotten. Level 1 sleep is good for sleeping and waking
                up the engine to run the same model again. The model weights
                are backed up in CPU memory. Please make sure there's enough
                CPU memory to store the model weights. Level 2 sleep will
                discard both the model weights and the kv cache. The content
                of both the model weights and kv cache is forgotten. Level 2
                sleep is good for sleeping and waking up the engine to run a
                different model or update the model, where previous model
                weights are not needed. It reduces CPU memory pressure.
        """
        self.reset_prefix_cache()
        self.llm_engine.sleep(level=level)

    def wake_up(self, tags: Optional[list[str]] = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                engine is used again.
        """
        self.llm_engine.wake_up(tags)

    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A ``MetricSnapshot`` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
        assert isinstance(self.llm_engine, V1LLMEngine)
        return self.llm_engine.get_metrics()

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        priority: Optional[list[int]] = None,
    ) -> None:
        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, Sequence) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      Sequence) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, Sequence) else (params, ):
            if isinstance(sp, SamplingParams):
                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        it = prompts
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            it = tqdm_func(it, desc="Adding requests")

        model_config = self.llm_engine.model_config

        for i, prompt in enumerate(it):

            param = params[i] if isinstance(params, Sequence) else params

            tokenization_kwargs: dict[str, Any] = {}
            _validate_truncation_size(model_config.max_model_len,
                                      param.truncate_prompt_tokens,
                                      tokenization_kwargs)

            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        lora_request: Optional[LoRARequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

    def _run_engine(
        self,
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
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
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
