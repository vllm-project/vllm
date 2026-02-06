# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar

from vllm.beam_search import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)
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
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreData,
    ScoreMultiModalParam,
    _cosine_similarity,
    compress_token_type_ids,
    compute_maxsim_score,
    get_score_prompt,
    validate_score_input,
)
from vllm.entrypoints.utils import log_non_default_args
from vllm.inputs.data import (
    DataPrompt,
    PromptType,
    SingletonPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.inputs import DecoderOnlyDictPrompt, DictPrompt
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq,
    get_prompt_components,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import BeamSearchParams, RequestOutputKind, SamplingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils.collection_utils import as_iter, is_list_of
from vllm.utils.counter import Counter
from vllm.v1.engine.llm_engine import LLMEngine
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
        enable_return_routed_experts: Whether to return routed experts.
        disable_custom_all_reduce: See
            [ParallelConfig][vllm.config.ParallelConfig].
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
        pooler_config: Initialize non-default pooling config for the pooling model,
            e.g., `PoolerConfig(seq_pooling_type="MEAN", use_activation=False)`.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the mode of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        attention_config: Configuration for attention mechanisms. Can be a
            dictionary or an AttentionConfig instance. If a dictionary, it will
            be converted to an AttentionConfig. Allows specifying the attention
            backend and other attention-related settings.
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
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
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
        logits_processors: list[str | type[LogitsProcessor]] | None = None,
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
                "'examples/offline_inference/data_parallel.py'."
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
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
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
            logits_processors=logits_processors,
            **kwargs,
        )

        log_non_default_args(engine_args)

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)
        self.supported_tasks = supported_tasks

        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor
        self.io_processor = self.llm_engine.io_processor

        # Cache for __repr__ to avoid repeated collective_rpc calls
        self._cached_repr: str | None = None

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
        self.input_processor.clear_mm_cache()
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
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
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

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.
        """
        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        prompts_seq = prompt_to_seq(prompts)

        self._validate_and_add_requests(
            prompts=prompts_seq,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=self._get_modality_specific_lora_reqs(
                prompts_seq, lora_request
            ),
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def _get_modality_specific_lora_reqs(
        self,
        prompts: Sequence[PromptType],
        lora_request: list[LoRARequest] | LoRARequest | None,
    ):
        # Grab the lora config off the vllm config on the engine,
        # since this is the same for both v0 & v1.
        lora_config = self.llm_engine.vllm_config.lora_config

        # If there's no lora config / default_mm_loras, or the model
        # isn't multimodal, leave the lora as is.
        if (
            lora_config is None
            or not self.model_config.is_multimodal_model
            or (lora_config and lora_config.default_mm_loras is None)
        ):
            return lora_request

        optional_loras = (
            [lora_request] * len(prompts)
            if not isinstance(lora_request, Sequence)
            else lora_request
        )

        return [
            self._resolve_single_prompt_mm_lora(
                prompt,
                opt_lora_req,
                lora_config.default_mm_loras,
            )
            for prompt, opt_lora_req in zip(prompts, optional_loras)
        ]

    def _resolve_single_prompt_mm_lora(
        self,
        prompt: PromptType | DictPrompt,
        lora_request: LoRARequest | None,
        default_mm_loras: dict[str, str] | None,
    ):
        if (
            not default_mm_loras
            or not isinstance(prompt, dict)
            or not (mm_data := prompt.get("multi_modal_data") or {})
        ):
            return lora_request

        intersection = set(
            mm_data.keys()  # type: ignore
        ).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request
        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be"
                " used by a single prompt consuming several modalities; "
                " currently we only support one lora per request; as such,"
                " lora(s) registered with modalities: %s"
                " will be skipped",
                intersection,
            )
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
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

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

    def _get_beam_search_lora_requests(
        self,
        lora_request: list[LoRARequest] | LoRARequest | None,
        prompts: list[TokensPrompt | TextPrompt],
    ) -> list[LoRARequest | None]:
        """Get the optional lora request corresponding to each prompt."""
        if isinstance(lora_request, Sequence) and len(lora_request) != len(prompts):
            raise ValueError(
                "Lora request list should be the same length as the prompts"
            )

        if lora_request is None or isinstance(lora_request, LoRARequest):
            return [lora_request] * len(prompts)

        raise TypeError(f"Invalid lora_request type {type(lora_request)}")

    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
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

        lora_requests = self._get_beam_search_lora_requests(lora_request, prompts)

        tokenizer = self.get_tokenizer()
        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id,
            length_penalty,
        )

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(prompts)

        def create_tokens_prompt_from_beam(beam: BeamSearchSequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {"prompt_token_ids": beam.tokens}
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs["mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
            skip_clone=True,  # Internal beam search, safe to skip clone
        )
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, prompts):
            # Add multimodal processor kwargs & data
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt["mm_processor_kwargs"]

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
                ),
            )

        for prompt_start in range(0, len(prompts), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

            token_iter = range(max_tokens)
            if use_tqdm:
                token_iter = tqdm(
                    token_iter, desc="Beam search", unit="token", unit_scale=False
                )
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            for _ in token_iter:
                all_beams: list[BeamSearchSequence] = list(
                    sum((instance.beams for instance in instances_batch), [])
                )
                pos = [0] + list(
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch
                    )
                )
                instance_start_and_end: list[tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:])
                )

                if len(all_beams) == 0:
                    break

                # create corresponding batch entries for prompt & optional lora
                prompts_batch, lora_req_batch = zip(
                    *[
                        (create_tokens_prompt_from_beam(beam), beam.lora_request)
                        for beam in all_beams
                    ]
                )

                # only runs for one step
                # we don't need to use tqdm here
                output = self.generate(
                    prompts_batch,
                    sampling_params=beam_search_params,
                    use_tqdm=False,
                    lora_request=lora_req_batch,
                )

                for (start, end), instance in zip(
                    instance_start_and_end, instances_batch
                ):
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
                                    logprobs=current_beam.logprobs + [logprobs],
                                    lora_request=current_beam.lora_request,
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                    multi_modal_data=current_beam.multi_modal_data,
                                    mm_processor_kwargs=current_beam.mm_processor_kwargs,
                                )

                                if (
                                    token_id == tokenizer.eos_token_id
                                    and not ignore_eos
                                ):
                                    instance.completed.append(new_beam)
                                else:
                                    instance_new_beams.append(new_beam)
                    sorted_beams = sorted(
                        instance_new_beams, key=sort_beams_key, reverse=True
                    )
                    instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def _get_cmpl_tok_params(self, tokenization_kwargs: dict[str, Any] | None):
        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            # For Whisper, special tokens should be provided by the user based
            # on the task and language of their request. Also needed to avoid
            # appending an EOS token to the prompt which disrupts generation.
            add_special_tokens=not model_config.is_encoder_decoder,
        ).with_kwargs(tokenization_kwargs)

    def _preprocess_completion(
        self,
        prompts: Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[DictPrompt]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `TokensPrompts` objects containing the tokenized prompt
            after chat template interpolation, and the raw multi-modal inputs.
        """
        renderer = self.llm_engine.renderer
        model_config = self.model_config

        # Some MM and encoder-decoder models have non-default `add_special_tokens`
        # TODO: Move multi-modal processor into tokenization
        is_mm_or_enc_dec = (
            model_config.is_multimodal_model or model_config.is_encoder_decoder
        )
        tok_params = self._get_cmpl_tok_params(tokenization_kwargs)

        engine_prompts = list[DictPrompt]()
        for prompt in prompts:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            in_prompt = renderer.render_completion(parsed_prompt)

            engine_prompts.append(
                renderer.tokenize_prompt(in_prompt, tok_params)
                if not is_mm_or_enc_dec and "encoder_prompt" not in in_prompt
                else in_prompt
            )

        return engine_prompts

    def _get_chat_tok_params(self, tokenization_kwargs: dict[str, Any] | None):
        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        ).with_kwargs(tokenization_kwargs)

    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[DecoderOnlyDictPrompt]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `TokensPrompts` objects containing the tokenized prompt
            after chat template interpolation, and the raw multi-modal inputs.
        """
        renderer = self.llm_engine.renderer

        chat_params = ChatParams(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenize=isinstance(renderer.tokenizer, MistralTokenizer),
                ),
            ),
        )
        tok_params = self._get_chat_tok_params(tokenization_kwargs)

        engine_prompts = list[DecoderOnlyDictPrompt]()
        for conversation in conversations:
            _, in_prompt = renderer.render_messages(conversation, chat_params)
            if mm_processor_kwargs is not None:
                in_prompt["mm_processor_kwargs"] = mm_processor_kwargs

            engine_prompts.append(renderer.tokenize_prompt(in_prompt, tok_params))

        return engine_prompts

    def chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: LoRARequest | None = None,
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
        prompts = self._preprocess_chat(
            conversation_to_seq(messages),
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
        )

    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        truncate_prompt_tokens: int | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        pooling_task: PoolingTask | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
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
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        if pooling_task is None:
            raise ValueError(
                "pooling_task required for `LLM.encode`\n"
                "Please use one of the more specific methods or set the "
                "pooling_task when using `LLM.encode`:\n"
                "  - For embeddings, use `LLM.embed(...)` "
                'or `pooling_task="embed"`.\n'
                "  - For classification logits, use `LLM.classify(...)` "
                'or `pooling_task="classify"`.\n'
                "  - For similarity scores, use `LLM.score(...)`.\n"
                "  - For rewards, use `LLM.reward(...)` "
                'or `pooling_task="token_classify"`\n'
                "  - For token classification, "
                'use `pooling_task="token_classify"`\n'
                '  - For multi-vector retrieval, use `pooling_task="token_embed"`'
            )

        model_config = self.model_config
        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        if truncate_prompt_tokens is not None:
            warnings.warn(
                "The `truncate_prompt_tokens` parameter in `LLM.encode()` "
                "is deprecated and will be removed in v0.16. "
                "Please pass it via `tokenization_kwargs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            tokenization_kwargs = merge_kwargs(
                tokenization_kwargs,
                dict(truncate_prompt_tokens=truncate_prompt_tokens),
            )

        io_processor_prompt = False
        if isinstance(prompts, dict) and "data" in prompts:
            io_processor_prompt = True
            if self.io_processor is None:
                raise ValueError(
                    "No IOProcessor plugin installed. Please refer "
                    "to the documentation and to the "
                    "'prithvi_geospatial_mae_io_processor' "
                    "offline inference example for more details."
                )

            # Validate the request data is valid for the loaded plugin
            validated_prompt = self.io_processor.parse_request(prompts)

            # obtain the actual model prompts from the pre-processor
            prompts = self.io_processor.pre_process(prompt=validated_prompt)

        if io_processor_prompt:
            assert self.io_processor is not None
            if is_list_of(pooling_params, PoolingParams):
                validated_pooling_params: list[PoolingParams] = []
                for param in as_iter(pooling_params):
                    validated_pooling_params.append(
                        self.io_processor.validate_or_generate_params(param)
                    )
                pooling_params = validated_pooling_params
            else:
                assert not isinstance(pooling_params, Sequence)
                pooling_params = self.io_processor.validate_or_generate_params(
                    pooling_params
                )

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        for param in as_iter(pooling_params):
            if param.task is None:
                param.task = pooling_task
            elif param.task != pooling_task:
                msg = f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                raise ValueError(msg)

        self._validate_and_add_requests(
            prompts=prompt_to_seq(prompts),
            params=pooling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)

        model_outputs = self.engine_class.validate_outputs(
            outputs, PoolingRequestOutput
        )

        if io_processor_prompt:
            # get the post-processed model outputs
            assert self.io_processor is not None
            processed_outputs = self.io_processor.post_process(
                model_output=model_outputs
            )

            return [
                PoolingRequestOutput[Any](
                    request_id="",
                    outputs=processed_outputs,
                    num_cached_tokens=getattr(
                        processed_outputs, "num_cached_tokens", 0
                    ),
                    prompt_token_ids=[],
                    finished=True,
                )
            ]
        else:
            return model_outputs

    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        truncate_prompt_tokens: int | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
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
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "embed" not in self.supported_tasks:
            raise ValueError(
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`."
            )

        if truncate_prompt_tokens is not None:
            tokenization_kwargs = merge_kwargs(
                tokenization_kwargs,
                dict(truncate_prompt_tokens=truncate_prompt_tokens),
            )

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
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
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if "classify" not in self.supported_tasks:
            raise ValueError(
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`."
            )

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="classify",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def reward(
        self,
        prompts: PromptType | Sequence[PromptType],
        /,
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        truncate_prompt_tokens: int | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """
        Generate rewards for each prompt.

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
            tokenization_kwargs: Overrides for `tokenizer.encode`.

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
            pooling_task="token_classify",
            tokenization_kwargs=tokenization_kwargs,
        )

    def _embedding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
    ) -> list[ScoringRequestOutput]:
        tokenizer = self.get_tokenizer()

        input_texts: list[str] = []
        for text in data_1 + data_2:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Embedding scores currently do not support multimodal input."
                )
            input_texts.append(text)

        encoded_output = self.encode(
            input_texts,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        encoded_output_1 = encoded_output[0 : len(data_1)]
        encoded_output_2 = encoded_output[len(data_1) :]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        scores = _cosine_similarity(
            tokenizer=tokenizer,
            embed_1=encoded_output_1,
            embed_2=encoded_output_2,
        )

        items = self.engine_class.validate_outputs(scores, PoolingRequestOutput)
        return [ScoringRequestOutput.from_base(item) for item in items]

    def _late_interaction_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
    ) -> list[ScoringRequestOutput]:
        """
        Late interaction scoring (ColBERT MaxSim).

        Encodes queries and documents into per-token embeddings, then computes
        MaxSim: sum over query tokens of max similarity to any document token.
        """
        from vllm.outputs import PoolingOutput

        tokenizer = self.get_tokenizer()

        # Extract text from ScoreData
        text_1: list[str] = []
        for text in data_1:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Late interaction scores currently do not support multimodal input."
                )
            text_1.append(text)

        text_2: list[str] = []
        for text in data_2:
            if not isinstance(text, str):
                raise NotImplementedError(
                    "Late interaction scores currently do not support multimodal input."
                )
            text_2.append(text)

        encoded_output: list[PoolingRequestOutput] = self.encode(
            text_1 + text_2,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            pooling_params=pooling_params,
            pooling_task="token_embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        encoded_output_1: list[PoolingRequestOutput] = encoded_output[0 : len(text_1)]
        encoded_output_2: list[PoolingRequestOutput] = encoded_output[len(text_1) :]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        # Compute MaxSim scores
        scores: list[PoolingRequestOutput] = []
        padding: list[int] = []
        if (pad_token_id := tokenizer.pad_token_id) is not None:
            padding = [pad_token_id]

        for emb_1, emb_2 in zip(encoded_output_1, encoded_output_2):
            # emb_1.outputs.data: [query_len, dim]
            # emb_2.outputs.data: [doc_len, dim]
            q_emb = emb_1.outputs.data
            d_emb = emb_2.outputs.data

            maxsim_score = compute_maxsim_score(q_emb, d_emb)

            tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

            scores.append(
                PoolingRequestOutput(
                    request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                    outputs=PoolingOutput(data=maxsim_score),
                    prompt_token_ids=tokens,
                    num_cached_tokens=emb_1.num_cached_tokens + emb_2.num_cached_tokens,
                    finished=True,
                )
            )

        items = self.engine_class.validate_outputs(scores, PoolingRequestOutput)
        return [ScoringRequestOutput.from_base(item) for item in items]

    def _cross_encoding_score(
        self,
        data_1: list[ScoreData],
        data_2: list[ScoreData],
        *,
        use_tqdm: bool | Callable[..., tqdm],
        pooling_params: PoolingParams | None,
        lora_request: list[LoRARequest] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any],
        score_template: str | None,
    ) -> list[ScoringRequestOutput]:
        model_config = self.model_config
        tokenizer = self.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError("Score API is not supported for Mistral tokenizer")

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        if pooling_params is None:
            pooling_params = PoolingParams(task="score")
        elif pooling_params.task is None:
            pooling_params.task = "score"

        pooling_params_list = list[PoolingParams]()

        prompts = list[PromptType]()

        input_pairs = [(t1, t2) for t1, t2 in zip(data_1, data_2)]

        for q, d in input_pairs:
            _, engine_prompt = get_score_prompt(
                model_config=model_config,
                data_1=q,
                data_2=d,
                tokenizer=tokenizer,
                tokenization_kwargs=tokenization_kwargs,
                score_template=score_template,
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):
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
        items = self.engine_class.validate_outputs(outputs, PoolingRequestOutput)

        return [ScoringRequestOutput.from_base(item) for item in items]

    def score(
        self,
        data_1: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        data_2: SingletonPrompt
        | Sequence[SingletonPrompt]
        | ScoreMultiModalParam
        | list[ScoreMultiModalParam],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        chat_template: str | None = None,
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
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """
        model_config = self.model_config

        runner_type = model_config.runner_type
        if runner_type != "pooling":
            raise ValueError(
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        supported_tasks = self.supported_tasks
        # Late interaction models (e.g., ColBERT) use token_embed for scoring
        is_late_interaction = model_config.is_late_interaction
        if not is_late_interaction and all(
            t not in supported_tasks for t in ("embed", "classify")
        ):
            raise ValueError(
                "Score API is not supported by this model. "
                "Try converting the model using "
                "`--convert embed` or `--convert classify`."
            )

        if (
            model_config.is_cross_encoder
            and getattr(model_config.hf_config, "num_labels", 0) != 1
        ):
            raise ValueError("Score API is only enabled for num_labels == 1.")

        if not model_config.is_cross_encoder and chat_template is not None:
            raise ValueError(
                "chat_template is only supported for cross-encoder models."
            )

        is_multimodal_model = model_config.is_multimodal_model
        architecture = model_config.architecture

        score_data_1, score_data_2 = validate_score_input(
            data_1,  # type: ignore[arg-type]
            data_2,  # type: ignore[arg-type]
            is_multimodal_model=is_multimodal_model,
            architecture=architecture,
        )

        tok_params = self._get_cmpl_tok_params(tokenization_kwargs)
        encode_kwargs = tok_params.get_encode_kwargs()

        if model_config.is_cross_encoder:
            return self._cross_encoding_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
                score_template=chat_template,
            )
        elif is_late_interaction:
            return self._late_interaction_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
            )
        else:
            return self._embedding_score(
                score_data_1,
                score_data_2,
                use_tqdm=use_tqdm,
                pooling_params=pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=encode_kwargs,
            )

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.llm_engine.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

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

    def wake_up(self, tags: list[str] | None = None):
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
            A `MetricSnapshot` instance capturing the current state
            of all aggregated metrics from Prometheus.

        Note:
            This method is only available with the V1 LLM engine.
        """
        return self.llm_engine.get_metrics()

    def _validate_and_add_requests(
        self,
        prompts: Sequence[PromptType],
        params: SamplingParams
        | Sequence[SamplingParams]
        | PoolingParams
        | Sequence[PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest | None] | LoRARequest | None,
        tokenization_kwargs: dict[str, Any] | None = None,
        priority: list[int] | None = None,
    ) -> None:
        num_requests = len(prompts)

        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({params}) "
                    f"and lora_request ({len(params)}) must be the same."
                )

            engine_params = params
        else:
            engine_params = [params] * num_requests

        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            engine_lora_requests: Sequence[LoRARequest | None] = lora_request
        else:
            engine_lora_requests = [lora_request] * num_requests

        if priority is not None:
            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )
        else:
            priority = [0] * num_requests

        if any(param.truncate_prompt_tokens is not None for param in engine_params):
            # TODO: Remove this after deprecating `param.truncate_prompt_tokens`
            # Then, move the code from the `else` block to the top and let
            # `self._preprocess_completion` handle prompt normalization
            engine_prompts = [
                engine_prompt
                for prompt, param in zip(prompts, engine_params)
                for engine_prompt in self._preprocess_completion(
                    [prompt],
                    tokenization_kwargs=merge_kwargs(
                        tokenization_kwargs,
                        dict(truncate_prompt_tokens=param.truncate_prompt_tokens),
                    ),
                )
            ]
        else:
            engine_prompts = self._preprocess_completion(
                prompts,
                tokenization_kwargs=tokenization_kwargs,
            )

        for sp in engine_params:
            if isinstance(sp, SamplingParams):
                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        it = engine_prompts
        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            it = tqdm_func(it, desc="Adding requests")

        added_request_ids: list[str] = []

        try:
            for i, prompt in enumerate(it):
                request_id = self._add_request(
                    prompt,
                    engine_params[i],
                    lora_request=engine_lora_requests[i],
                    tokenization_kwargs=tokenization_kwargs,
                    priority=priority[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

    def _add_request(
        self,
        prompt: PromptType | DictPrompt,
        params: SamplingParams | PoolingParams,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> str:
        prompt_text, _, _ = get_prompt_components(prompt)
        request_id = str(next(self.request_counter))

        if params.truncate_prompt_tokens is not None:
            params_type = type(params).__name__
            warnings.warn(
                f"The `truncate_prompt_tokens` parameter in `{params_type}` "
                "is deprecated and will be removed in v0.16. "
                "Please pass it via `tokenization_kwargs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            tokenization_kwargs = merge_kwargs(
                tokenization_kwargs,
                dict(truncate_prompt_tokens=params.truncate_prompt_tokens),
            )

        tok_params = self._get_cmpl_tok_params(tokenization_kwargs)

        tokenization_kwargs = tok_params.get_encode_kwargs()
        engine_request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            supported_tasks=self.supported_tasks,
        )

        self.llm_engine.add_request(
            request_id,
            engine_request,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            prompt_text=prompt_text,
        )
        return engine_request.request_id

    def _run_engine(
        self, *, use_tqdm: bool | Callable[..., tqdm] = True
    ) -> list[RequestOutput | PoolingRequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[RequestOutput | PoolingRequestOutput] = []
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
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
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

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """
        Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info
        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

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
