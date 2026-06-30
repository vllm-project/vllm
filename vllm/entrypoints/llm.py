# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import overload

from vllm.config import (
    AttentionConfig,
    CompilationConfig,
    PoolerConfig,
    ProfilerConfig,
    StructuredOutputsConfig,
    is_init_field,
)
from vllm.config.compilation import CompilationMode
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    load_chat_template,
)
from vllm.entrypoints.generate.beam_search.offline import BeamSearchOfflineMixin
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.serve.utils.api_utils import log_non_default_args
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine import PauseMode
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.sample.logits_processor import LogitsProcessor

from .offline_utils import _O, _R, OfflineInferenceMixin

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Metric

logger = init_logger(__name__)


class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin):
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
        allowed_media_domains: If set, only media URLs that belong to this
            domain can be used for multi-modal inputs.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `dtype` attribute of the Transformers model's config. However,
            if the `dtype` in the config is `float32`, we will use `float16` instead.
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
        chat_template: The chat template to apply.
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
            compared with using gpu_memory_utilization. Note that
            kv_cache_memory_bytes (when not-None) ignores
            gpu_memory_utilization
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        offload_group_size: Prefetch offloading: Group every N layers
            together. Offload last `offload_num_in_group` layers of each group.
            Default is 0 (disabled).
        offload_num_in_group: Prefetch offloading: Number of layers to
            offload per group. Default is 1.
        offload_prefetch_step: Prefetch offloading: Number of layers to
            prefetch ahead. Higher values hide more latency but use more GPU
            memory. Default is 1.
        offload_params: Prefetch offloading: Set of parameter name segments
            to selectively offload. Only parameters whose names contain one of
            these segments will be offloaded (e.g., {"gate_up_proj", "down_proj"}
            for MLP weights, or {"w13_weight", "w2_weight"} for MoE expert
            weights). If None or empty, all parameters are offloaded.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        enable_return_routed_experts: Whether to return routed experts.
        disable_custom_all_reduce: See
            [ParallelConfig][vllm.config.ParallelConfig].
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `hf auth login` (stored in `~/.cache/huggingface/token`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor. Overrides for the
            multi-modal processor obtained from `AutoProcessor.from_pretrained`.
            The available overrides depend on the model that is being run.
            For example, for Phi-3-Vision: `{"num_crops": 4}`.
        pooler_config: Initialize non-default pooling config for the pooling model,
            e.g., `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the mode of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        attention_config: Configuration for attention mechanisms. Can be a
            dictionary or an AttentionConfig instance. If a dictionary, it will
            be converted to an AttentionConfig. Allows specifying the attention
            backend and other attention-related settings.
        spec_method: Top-level alias for `speculative_config["method"]`.
        spec_model: Top-level alias for `speculative_config["model"]`.
        spec_tokens: Top-level alias for
            `speculative_config["num_speculative_tokens"]`.
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
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template: Path | str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.92,
        cpu_offload_gb: float = 0,
        offload_group_size: int = 0,
        offload_num_in_group: int = 1,
        offload_prefetch_step: int = 1,
        offload_params: set[str] | None = None,
        enforce_eager: bool = False,
        enable_return_routed_experts: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        pooler_config: PoolerConfig | None = None,
        structured_outputs_config: dict[str, Any]
        | StructuredOutputsConfig
        | None = None,
        profiler_config: dict[str, Any] | ProfilerConfig | None = None,
        attention_config: dict[str, Any] | AttentionConfig | None = None,
        kv_cache_memory_bytes: int | None = None,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None,
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
        spec_method: str | None = None,
        spec_model: str | None = None,
        spec_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""

        if "swap_space" in kwargs:
            kwargs.pop("swap_space")
            import warnings

            warnings.warn(
                "The 'swap_space' parameter is deprecated and ignored. "
                "It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict
        ):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                # Consider re-raising a more specific vLLM error or ValueError
                # to provide better context to the user.
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        if hf_overrides is None:
            hf_overrides = {}

        def _make_config(value: Any, cls: type[_R]) -> _R:
            """Convert dict/None/instance to a config instance."""
            if value is None:
                return cls()
            if isinstance(value, dict):
                return cls(**{k: v for k, v in value.items() if is_init_field(cls, k)})  # type: ignore[arg-type]
            return value

        if isinstance(compilation_config, int):
            compilation_config_instance = CompilationConfig(
                mode=CompilationMode(compilation_config)
            )
        else:
            compilation_config_instance = _make_config(
                compilation_config, CompilationConfig
            )

        structured_outputs_instance = _make_config(
            structured_outputs_config, StructuredOutputsConfig
        )
        profiler_config_instance = _make_config(profiler_config, ProfilerConfig)
        attention_config_instance = _make_config(attention_config, AttentionConfig)

        # warn about single-process data parallel usage.
        _dp_size = int(kwargs.get("data_parallel_size", 1))
        _distributed_executor_backend = kwargs.get("distributed_executor_backend")
        if (
            _dp_size > 1
            and not _distributed_executor_backend == "external_launcher"
            and not current_platform.is_tpu()
        ):
            raise ValueError(
                f"LLM(data_parallel_size={_dp_size}) is not supported for single-"
                "process usage and may hang. Please use "
                "the explicit multi-process data-parallel example at "
                "'examples/features/data_parallel/data_parallel_offline.py'."
            )

        engine_args = EngineArgs(
            model=model,
            runner=runner,
            convert=convert,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            cpu_offload_gb=cpu_offload_gb,
            offload_group_size=offload_group_size,
            offload_num_in_group=offload_num_in_group,
            offload_prefetch_step=offload_prefetch_step,
            offload_params=offload_params or set(),
            enforce_eager=enforce_eager,
            enable_return_routed_experts=enable_return_routed_experts,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            pooler_config=pooler_config,
            structured_outputs_config=structured_outputs_instance,
            profiler_config=profiler_config_instance,
            attention_config=attention_config_instance,
            compilation_config=compilation_config_instance,
            quantization_config=quantization_config,
            logits_processors=logits_processors,
            spec_method=spec_method,
            spec_model=spec_model,
            spec_tokens=spec_tokens,
            **kwargs,
        )

        log_non_default_args(engine_args)

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        self.supported_tasks = supported_tasks

        self.runner_type = self.model_config.runner_type
        self.renderer = self.llm_engine.renderer
        self.chat_template = load_chat_template(chat_template)
        self.input_processor = self.llm_engine.input_processor

        # The renderer thread pool is only consumed by the async renderer
        # path; the synchronous `LLM` entrypoint runs multimodal
        # preprocessing serially. Warn so the setting is not a silent
        # no-op. See vllm-project/vllm#42901.
        if self.model_config.renderer_num_workers > 1:
            logger.warning_once(
                "`renderer_num_workers=%d` was set, but the offline `LLM` "
                "entrypoint uses the synchronous renderer path and runs "
                "multimodal preprocessing serially across prompts. The "
                "renderer thread pool is only consumed by the async "
                "renderer path used by `vllm serve` / `AsyncLLM`, so this "
                "setting has no effect here.",
                self.model_config.renderer_num_workers,
            )

        PoolingOfflineMixin.__init__(self)

        # Cache for __repr__ to avoid repeated collective_rpc calls
        self._cached_repr: str | None = None

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLM":
        """Create an LLM instance from EngineArgs."""
        return cls(**vars(engine_args))

    def get_tokenizer(self) -> TokenizerLike:
        return self.llm_engine.get_tokenizer()

    def get_world_size(self, include_dp: bool = True) -> int:
        """Get the world size from the parallel config.

        Args:
            include_dp: If True (default), returns the world size including
                data parallelism (TP * PP * DP). If False, returns the world
                size without data parallelism (TP * PP).

        Returns:
            The world size (tensor_parallel_size * pipeline_parallel_size),
            optionally multiplied by data_parallel_size if include_dp is True.
        """
        parallel_config = self.llm_engine.vllm_config.parallel_config
        if include_dp:
            return parallel_config.world_size_across_dp
        return parallel_config.world_size

    def reset_mm_cache(self) -> None:
        self.renderer.clear_mm_cache()
        self.llm_engine.reset_mm_cache()

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
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
                If provided, must be a list of integers matching the length
                of `prompts`, where each priority value corresponds to the prompt
                at the same index.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.
        """
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def enqueue(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Enqueue prompts for generation without waiting for completion.

        This method adds requests to the engine queue but does not start
        processing them. Use wait_for_completion() to process the queued
        requests and get results.

        Args:
            prompts: The prompts to the LLM. See generate() for details.
            sampling_params: The sampling parameters for text generation.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            use_tqdm: If True, shows a tqdm progress bar while adding requests.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError("LLM.enqueue() is only supported for generative models.")

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._add_completion_requests(
            prompts=prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    @overload
    def wait_for_completion(
        self,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[RequestOutput | PoolingRequestOutput]: ...

    @overload
    def wait_for_completion(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]: ...

    def wait_for_completion(
        self,
        output_type: type[Any] | tuple[type[Any], ...] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[Any]:
        """Wait for all enqueued requests to complete and return results.

        This method processes all requests currently in the engine queue
        and returns their outputs. Use after enqueue() to get results.

        Args:
            output_type: The expected output type(s). If not provided, accepts
                both RequestOutput and PoolingRequestOutput.
            use_tqdm: If True, shows a tqdm progress bar.

        Returns:
            A list of output objects for all completed requests.
        """
        if output_type is None:
            output_type = (RequestOutput, PoolingRequestOutput)

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
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

        !!! warning
            To reduce the overhead of data transfer, avoid returning large
            arrays or tensors from this method. If you must return them,
            make sure you move them to CPU first to avoid taking up additional
            VRAM!
        """
        return self.llm_engine.apply_model(func)

    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the [generate][vllm.LLM.generate] method to generate
        the responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A sequence of conversations or a single conversation.

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
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of `RequestOutput` objects containing the generated
            responses in the same order as the input messages.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_chat(
            messages=messages,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def enqueue_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Enqueue chat conversations for generation without waiting.

        This method renders chat conversations and adds the resulting requests
        to the engine queue. Use wait_for_completion() to get results. To
        guarantee that all requests are queued before scheduling starts, pause
        scheduling with sleep(level=0) before calling this method and resume it
        with wake_up(tags=["scheduling"]) afterward.

        Args:
            messages: A sequence of conversations or a single conversation.
                Each conversation is represented as a list of messages.
            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar while rendering
                conversations.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
            chat_template: The template to use for structuring the chat.
            chat_template_content_format: The format to render message content.
            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one.
            tools: Tools to make available to the model, if any.
            chat_template_kwargs: Additional kwargs to pass to the chat
                template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
            mm_processor_kwargs: Overrides for `processor.__call__`.

        Returns:
            A list of request IDs for the enqueued requests.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.enqueue_chat() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._add_chat_requests(
            messages=messages,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def start_profile(self, profile_prefix: str | None = None) -> None:
        """Start profiling with optional custom trace prefix.

        Args:
            profile_prefix: Optional prefix for the trace file names. If provided,
                           trace files will be named as "<prefix>_dp<X>_pp<Y>_tp<Z>".
                           If not provided, default naming will be used.
        """
        self.llm_engine.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level.
                - Level 0: Pause scheduling but continue accepting requests.
                           Requests are queued but not processed.
                - Level 1: Offload model weights to CPU, discard KV cache.
                           The content of kv cache is forgotten. Good for
                           sleeping and waking up the engine to run the same
                           model again. Please make sure there's enough CPU
                           memory to store the model weights.
                - Level 2: Discard all GPU memory (weights + KV cache).
                           Good for sleeping and waking up the engine to run
                           a different model or update the model, where
                           previous model weights are not needed. It reduces
                           CPU memory pressure.
            mode: How to handle any existing requests, can be "abort", "wait",
                or "keep".
        """
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None):
        """
        Wake up the engine from sleep mode. See the [sleep][vllm.LLM.sleep]
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the engine memory
                for specific memory allocations. Values must be in
                `("weights", "kv_cache", "scheduling")`. If None, all memory
                is reallocated. wake_up should be called with all tags
                (or None) before the engine is used again.
                Use tags=["scheduling"] to resume from level 0 sleep.
        """
        self.llm_engine.wake_up(tags)

    def get_metrics(self) -> list["Metric"]:
        """Return a snapshot of aggregated metrics from Prometheus.

        Returns:
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        return self.llm_engine.get_metrics()

    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def start_weight_update(self) -> None:
        """Start a new weight update."""
        self.llm_engine.collective_rpc("start_weight_update")

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )
        options_dict = (
            request.get("options", {}) if isinstance(request, dict) else request.options
        )

        self.llm_engine.collective_rpc(
            "update_weights",
            kwargs={"update_info": update_info_dict, "options": options_dict},
        )

    def finish_weight_update(self) -> None:
        """Finish the current weight update."""
        self.llm_engine.collective_rpc("finish_weight_update")

    def __repr__(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # Cache the result to avoid repeated collective_rpc calls
        if self._cached_repr is None:
            results = self.llm_engine.collective_rpc("get_model_inspection")
            # In distributed settings, we get results from all workers
            # Just return the first one (they should all be the same)
            if results:
                self._cached_repr = results[0]
            else:
                self._cached_repr = f"LLM(model={self.model_config.model!r})"
        return self._cached_repr
