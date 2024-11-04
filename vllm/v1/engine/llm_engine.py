import time
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type,
                    Union)

from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs,
                         EncoderDecoderLLMInputs, InputRegistry, PromptType)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer_group import (
    BaseTokenizerGroup, init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.tokenizer.detokenizer import Detokenizer, DetokenizerInputs
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class LLMEngine:

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:

        # TODO: remove the local variables and use self.* throughout the class.
        model_config = self.model_config = vllm_config.model_config
        cache_config = self.cache_config = vllm_config.cache_config
        lora_config = self.lora_config = vllm_config.lora_config
        parallel_config = self.parallel_config = vllm_config.parallel_config
        scheduler_config = self.scheduler_config = vllm_config.scheduler_config
        device_config = self.device_config = vllm_config.device_config
        speculative_config = self.speculative_config = vllm_config.speculative_config  # noqa
        load_config = self.load_config = vllm_config.load_config
        decoding_config = self.decoding_config = vllm_config.decoding_config or DecodingConfig(  # noqa
        )
        prompt_adapter_config = self.prompt_adapter_config = vllm_config.prompt_adapter_config  # noqa
        observability_config = self.observability_config = vllm_config.observability_config or ObservabilityConfig(  # noqa
        )

        # Override the configs for V1.
        # FIXME
        if usage_context == UsageContext.LLM_CLASS:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            scheduler_config.max_num_seqs = 1024
            scheduler_config.max_num_batched_tokens = 2048

        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "override_neuron_config=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, "
            "num_scheduler_steps=%d, enable_prefix_caching=%s, "
            "use_async_output_proc=%s, mm_processor_kwargs=%s)",
            VLLM_VERSION,
            model_config.model,
            speculative_config,
            model_config.tokenizer,
            model_config.skip_tokenizer_init,
            model_config.tokenizer_mode,
            model_config.revision,
            model_config.override_neuron_config,
            model_config.rope_scaling,
            model_config.rope_theta,
            model_config.tokenizer_revision,
            model_config.trust_remote_code,
            model_config.dtype,
            model_config.max_model_len,
            load_config.download_dir,
            load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization,
            model_config.enforce_eager,
            cache_config.cache_dtype,
            model_config.quantization_param_path,
            device_config.device,
            decoding_config,
            observability_config,
            model_config.seed,
            model_config.served_model_name,
            scheduler_config.num_scheduler_steps,
            cache_config.enable_prefix_caching,
            model_config.use_async_output_proc,
            model_config.mm_processor_kwargs,
        )

        self.log_stats = log_stats

        assert not self.model_config.skip_tokenizer_init
        self.tokenizer = self._init_tokenizer()
        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()
        self.detokenizer = Detokenizer(self.model_config.tokenizer)

        self.generation_config_fields = _load_generation_config_dict(
            model_config)
        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer)
        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            model_config)

        # Request id -> Request
        self.requests: Dict[str, Request] = {}
        # NOTE(woosuk): Now that the detokenizer works asynchronously, we need
        # to keep track of how many steps each request has been lagged behind
        # in terms of detokenization.
        # Request id -> how many detokenizer steps the request should wait for.
        self.num_lagged_steps: Dict[str, int] = {}
        # OPTIMIZATION: Cache the request output and update it incrementally.
        # This is used to avoid creating a new RequestOutput object every step.
        # Request id -> RequestOutput
        self.request_outputs: Dict[str, RequestOutput] = {}

        self.model_executor = executor_class(vllm_config=vllm_config)
        assert self.model_config.task != "embedding"
        self._initialize_kv_caches()

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

    def _initialize_kv_caches(self) -> None:
        num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks(
        )

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = 0
        self.model_executor.initialize_cache(num_gpu_blocks)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        executor_class = cls._get_executor_cls(engine_config)
        # Create the LLM engine.
        engine = cls(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
        return engine

    def _init_tokenizer(self) -> BaseTokenizerGroup:
        return init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            parallel_config=self.parallel_config,
            enable_lora=bool(self.lora_config))

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)
        if self.lora_config:
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)
        if self.prompt_adapter_config:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: Union[DecoderOnlyInputs, EncoderDecoderLLMInputs],
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        assert prompt_adapter_request is None
        assert trace_headers is None
        self._validate_model_inputs(processed_inputs)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        # TODO(woosuk): Support embedding mode.
        assert isinstance(params, SamplingParams)
        sampling_params = params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)

        # TODO(woosuk): Check max_logprobs
        # TODO(woosuk): Support encoder-decoder models.
        req = Request(request_id, processed_inputs, params, eos_token_id,
                      arrival_time)
        self.requests[request_id] = req
        self.num_lagged_steps[request_id] = 0
        self.scheduler.add_request(req)

    def stop_remote_worker_execution_loop(self) -> None:
        raise NotImplementedError("TP not implemented yet.")

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        assert priority == 0, "vLLM V1 does not support priority at the moment."

        preprocessed_inputs = self.input_preprocessor.preprocess(
            prompt,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
        )

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.scheduler.finish_requests(request_id,
                                       RequestStatus.FINISHED_ABORTED)
        self._free_request(request_id)

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return len(self.requests)

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return len(self.requests) > 0

    def step(self) -> List[RequestOutput]:
        # NOTE(woosuk): This method may return an empty list when the
        # detokenizer is still processing the outputs. This should not be
        # considered as the end of the generation process.
        # FIXME(woosuk): Currently, the step method is inefficient because it
        # creates RequestOutput objects for all running requests, while they
        # may not be needed unless the output is streamed to the client.
        if self.scheduler.has_unfinished_requests():
            scheduler_output = self.scheduler.schedule()
            output = self.model_executor.execute_model(scheduler_output)
            sampled = self.scheduler.update_from_output(
                scheduler_output, output)
            self.send_to_detokenizer(sampled)
        req_outputs = self.recv_from_detokenizer()
        return req_outputs

    def send_to_detokenizer(self, sampled: List[Tuple[Request, int]]) -> None:
        inputs = DetokenizerInputs(
            req_ids=[],
            prompt_token_ids=[],
            new_token_ids=[],
            skip_special_tokens=[],
            spaces_between_special_tokens=[],
            free_req_ids=[],  # TODO(woosuk): Implement freeing.
        )
        for req, num_tokens in sampled:
            inputs.req_ids.append(req.request_id)
            if len(req.output_token_ids) == num_tokens:
                # The request is first detokenized.
                inputs.prompt_token_ids.append(req.prompt_token_ids)
            else:
                # The prompt token ids are already cached in the detokenizer.
                inputs.prompt_token_ids.append([])
            inputs.new_token_ids.append(req.output_token_ids[-num_tokens:])
            inputs.skip_special_tokens.append(
                req.sampling_params.skip_special_tokens)
            inputs.spaces_between_special_tokens.append(
                req.sampling_params.spaces_between_special_tokens)

            # Update the number of lagged steps.
            self.num_lagged_steps[req.request_id] += 1
        self.detokenizer.send(inputs)

    def recv_from_detokenizer(self) -> List[RequestOutput]:
        detokenizer_output = self.detokenizer.recv()
        if detokenizer_output is None:
            return []

        req_outputs: List[RequestOutput] = []
        num_reqs = len(detokenizer_output.req_ids)
        for i in range(num_reqs):
            req_id = detokenizer_output.req_ids[i]
            if req_id not in self.requests:
                # The request has been aborted while the detokenizer was
                # processing the outputs.
                continue

            req = self.requests[req_id]
            req.output_text += detokenizer_output.detokenized_texts[i]

            self.num_lagged_steps[req_id] -= 1
            finished = (self.num_lagged_steps[req_id] == 0
                        and req.is_finished())
            req_output = self._make_request_output(
                req, detokenizer_output.num_output_token_ids[i],
                detokenizer_output.detokenized_texts[i], finished)
            req_outputs.append(req_output)

            if finished:
                self._free_request(req_id)
        return req_outputs

    def terminate_detokenizer(self) -> None:
        self.detokenizer.terminate()

    def _make_request_output(
        self,
        request: Request,
        num_output_tokens: int,
        new_output_text: str,
        finished: bool,
    ) -> RequestOutput:
        req_output = self.request_outputs.get(request.request_id)
        if req_output is None:
            # TODO: Support `n` > 1.
            completion_output = CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,  # TODO
                finish_reason=None,
                stop_reason=None,
                lora_request=None,
            )
            req_output = RequestOutput(
                request_id=request.request_id,
                prompt=request.prompt,
                prompt_token_ids=request.prompt_token_ids,
                prompt_logprobs=None,  # TODO
                outputs=[completion_output],
                finished=False,
                metrics=None,
                lora_request=None,
                encoder_prompt=None,
                encoder_prompt_token_ids=None,
            )
            self.request_outputs[request.request_id] = req_output

        completion_output = req_output.outputs[0]
        if request.sampling_params.output_kind == RequestOutputKind.CUMULATIVE:
            completion_output.text += new_output_text
            completion_output.token_ids = (
                request.output_token_ids[:num_output_tokens])
        elif request.sampling_params.output_kind == RequestOutputKind.DELTA:
            completion_output.text = new_output_text
            num_prev_tokens = len(completion_output.token_ids)
            completion_output.token_ids = request.output_token_ids[
                num_prev_tokens:num_output_tokens]
        elif (request.sampling_params.output_kind ==
              RequestOutputKind.FINAL_ONLY):
            if finished:
                completion_output.text = request.output_text
                completion_output.token_ids = request.output_token_ids
            else:
                completion_output.text = ""
                completion_output.token_ids = []

        if finished:
            completion_output.finish_reason = request.get_finished_reason()
            completion_output.stop_reason = request.stop_reason
            req_output.finished = finished
        return req_output

    def _free_request(self, request_id: str) -> None:
        self.requests.pop(request_id, None)
        self.num_lagged_steps.pop(request_id, None)
        self.request_outputs.pop(request_id, None)

    def check_health(self) -> None:
        if self.tokenizer:
            self.tokenizer.check_health()
        self.model_executor.check_health()

    def _validate_model_inputs(self, inputs: Union[DecoderOnlyInputs,
                                                   EncoderDecoderLLMInputs]):
        prompt_ids = inputs.get("prompt_token_ids")
        if prompt_ids is None or len(prompt_ids) == 0:
            raise ValueError("Prompt cannot be empty")

        if self.model_config.is_multimodal_model:
            max_prompt_len = self.model_config.max_model_len

            if len(prompt_ids) > max_prompt_len:
                raise ValueError(
                    f"The prompt (total length {len(prompt_ids)}) is too long "
                    f"to fit into the model (context length {max_prompt_len}). "
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_parallel_config(self) -> ParallelConfig:
        """Gets the parallel configuration."""
        return self.parallel_config

    def get_decoding_config(self) -> DecodingConfig:
        """Gets the decoding configuration."""
        return self.decoding_config

    def get_scheduler_config(self) -> SchedulerConfig:
        """Gets the scheduler configuration."""
        return self.scheduler_config

    def get_lora_config(self) -> LoRAConfig:
        """Gets the LoRA configuration."""
        return self.lora_config

    @classmethod
    def _get_executor_cls(cls, engine_config: VllmConfig):
        return GPUExecutor

    def is_tracing_enabled(self) -> bool:
        return False

    def do_log_stats(self, *args, **kwargs) -> None:
        pass

    def is_encoder_decoder_model(self) -> bool:
        return False

    def start_profile(self) -> None:
        pass

    def stop_profile(self) -> None:
        pass

    def get_tokenizer_group(self, *args, **kwargs):
        return self.tokenizer


def _load_generation_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    config = try_get_generation_config(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if config is None:
        return {}

    return config.to_diff_dict()
