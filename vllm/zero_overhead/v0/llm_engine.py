from functools import partial
import os
import queue
import threading
import traceback
from typing import Callable, Dict, List, Mapping, Optional, Type, Union
from zlib import ZLIB_VERSION
import torch
from vllm import envs
from vllm.config import DecodingConfig, ObservabilityConfig, VllmConfig
from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.engine.llm_engine import _LOCAL_LOGGING_INTERVAL_SEC, LLMEngine, SchedulerContext, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.logger import init_logger
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import INPUT_REGISTRY
from vllm.inputs.parse import is_token_prompt, split_enc_dec_inputs
from vllm.inputs.data import ProcessorInputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.inputs.registry import InputRegistry
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.registry import MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest, ParallelSampleSequenceGroup, SequenceGroup, SequenceGroupBase, SequenceGroupMetadata
from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.version import __version__ as VLLM_VERSION
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled
from vllm.utils import resolve_obj_by_qualname, weak_bind, Counter
from vllm.zero_overhead.v0.sampler import SampleRecorder, get_last_sampler
from vllm.zero_overhead.v0.sequence import ZeroOverheadSequence
from vllm.zero_overhead.v0.stop_check import ZeroOverheadStopChecker
from vllm.zero_overhead.v0.tokenizer import ZeroOverheadDetokenizer
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.zero_overhead.v0.utils import is_zero_no_thread

logger = init_logger(__name__)

class ZeroOverheadEngine(LLMEngine):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
    ) -> None:
        if envs.VLLM_USE_V1:
            raise ValueError(
                "Using V0 LLMEngine, but envs.VLLM_USE_V1=True. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config  # noqa
        self.load_config = vllm_config.load_config
        self.decoding_config = vllm_config.decoding_config or DecodingConfig(  # noqa
        )
        self.prompt_adapter_config = vllm_config.prompt_adapter_config  # noqa
        self.observability_config = vllm_config.observability_config or ObservabilityConfig(  # noqa
        )

        logger.info(
            "Initializing a V0 LLM engine (v%s) with config: %s, "
            "use_cached_outputs=%s, ",
            VLLM_VERSION,
            vllm_config,
            use_cached_outputs,
        )

        self.log_stats = log_stats
        self.use_cached_outputs = use_cached_outputs

        if not self.model_config.skip_tokenizer_init:
            self.tokenizer = self._init_tokenizer()
            self.detokenizer = ZeroOverheadDetokenizer(self.tokenizer)
            tokenizer_group = self.get_tokenizer_group()
        else:
            self.tokenizer = None
            self.detokenizer = None
            tokenizer_group = None

        # Ensure that the function doesn't contain a reference to self,
        # to avoid engine GC issues
        def get_tokenizer_for_seq(sequence: ZeroOverheadSequence) -> AnyTokenizer:
            assert tokenizer_group, ("tokenizer_group cannot be None, "
                                     "make sure skip_tokenizer_init is False")
            return tokenizer_group.get_lora_tokenizer(sequence.lora_request)

        self.seq_counter = Counter()
        self.generation_config_fields = (
            self.model_config.try_get_generation_config())

        self.input_preprocessor = InputPreprocessor(self.model_config,
                                                    self.tokenizer,
                                                    mm_registry)

        self.input_registry = input_registry
        self.input_processor = input_registry.create_input_processor(
            self.model_config)

        self.model_executor = executor_class(vllm_config=vllm_config, )

        if self.model_config.runner_type != "pooling":
            self._initialize_kv_caches()

        # If usage stat is enabled, collect relevant info.
        if is_usage_stats_enabled():
            from vllm.model_executor.model_loader import (
                get_architecture_class_name)
            usage_message.report_usage(
                get_architecture_class_name(self.model_config),
                usage_context,
                extra_kvs={
                    # Common configuration
                    "dtype":
                    str(self.model_config.dtype),
                    "tensor_parallel_size":
                    self.parallel_config.tensor_parallel_size,
                    "block_size":
                    self.cache_config.block_size,
                    "gpu_memory_utilization":
                    self.cache_config.gpu_memory_utilization,

                    # Quantization
                    "quantization":
                    self.model_config.quantization,
                    "kv_cache_dtype":
                    str(self.cache_config.cache_dtype),

                    # Feature flags
                    "enable_lora":
                    bool(self.lora_config),
                    "enable_prompt_adapter":
                    bool(self.prompt_adapter_config),
                    "enable_prefix_caching":
                    self.cache_config.enable_prefix_caching,
                    "enforce_eager":
                    self.model_config.enforce_eager,
                    "disable_custom_all_reduce":
                    self.parallel_config.disable_custom_all_reduce,
                })

        if self.tokenizer:
            # Ping the tokenizer to ensure liveness if it runs in a
            # different process.
            self.tokenizer.ping()

        self.cached_scheduler_outputs = [
            SchedulerOutputState()
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.
                             multi_step_stream_outputs)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs,
                        ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.parallel_config.pipeline_parallel_size)
            ]
        else:
            self.async_callbacks = []

        # Currently used by AsyncLLMEngine to ensure quick append
        # of request outputs to asyncio queues
        self.process_request_outputs_callback: Optional[Callable] = None

        # Create the scheduler.
        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        if isinstance(self.vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                self.vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = self.vllm_config.scheduler_config.scheduler_cls
        self.scheduler = [
            Scheduler(
                self.scheduler_config, self.cache_config, self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None)
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

        # Metric Logging.
        if self.log_stats:
            if stat_loggers is not None:
                self.stat_loggers = stat_loggers
            else:
                # Lazy import for prometheus multiprocessing.
                # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
                # before prometheus_client is imported.
                # See https://prometheus.github.io/client_python/multiprocess/
                from vllm.engine.metrics import (LoggingStatLogger,
                                                 PrometheusStatLogger)

                self.stat_loggers = {
                    "logging":
                    LoggingStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        vllm_config=vllm_config),
                    "prometheus":
                    PrometheusStatLogger(
                        local_interval=_LOCAL_LOGGING_INTERVAL_SEC,
                        labels=dict(
                            model_name=self.model_config.served_model_name),
                        vllm_config=vllm_config),
                }
                self.stat_loggers["prometheus"].info("cache_config",
                                                     self.cache_config)

        self.tracer = None
        if self.observability_config.otlp_traces_endpoint:
            self.tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)

        # Create sequence output processor, e.g. for beam search or
        # speculative decoding.
        self.output_processor = (
            SequenceGroupOutputProcessor.create_output_processor(
                self.scheduler_config,
                self.detokenizer,
                self.scheduler,
                self.seq_counter,
                get_tokenizer_for_seq,
                stop_checker=ZeroOverheadStopChecker(
                    self.scheduler_config.max_model_len,
                    get_tokenizer_for_seq,
                ),
            ))
        self.tree_decoding = os.environ.get('VLLM_TREE_DECODING') == '1'

        self.seq_id_to_seq_group: Dict[str, SequenceGroupBase] = {}

        # Flag to set when an input fails to process and the engine should run
        # the next step without re-scheduling.
        self._skip_scheduling_next_step = False
        self.async_d2h = None
        self.last_record = None
        self.async_event = torch.cuda.Event(enable_timing=False)
        self.thread_running = False
        self.q_recorder = queue.Queue()
        if not is_zero_no_thread():
            self.zero_thread = threading.Thread(target=self.thread_zero_overhead)
            self.thread_running = True
            self.sem_m2s = threading.Semaphore(0) # main to scheduler thread
            self.zero_thread.start()
    
    def __del__(self):
        self.finish_thread()
        return super().__del__()

    def finish_thread(self):
        if self.thread_running:
            self.thread_running = False
            self.sem_m2s.release()
    
    def thread_zero_overhead(self):
        logger.info('zero overhead thread start!')
        try:
            while True:
                self.sem_m2s.acquire()
                if not self.thread_running:
                    break

                virtual_engine = 0
                # Clear outputs for each new scheduler iteration
                
                # Schedule iteration
                (seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc
                    ) = self.scheduler[virtual_engine].schedule()
                if self.last_record is not None:
                    last_sampler = self.last_record[1]
                    self.async_d2h = last_sampler.sampled_token_ids_tensor.to('cpu', non_blocking=True)
                    self.async_event.record()
                    self.q_recorder.put(self.last_record)
                else:
                    self.q_recorder.put(None)
                if len(seq_group_metadata_list) == 0:
                    self.last_record = None
                    continue

                finished_requests_ids = self.scheduler[
                    virtual_engine].get_and_reset_finished_requests_ids()

                assert seq_group_metadata_list is not None
                assert scheduler_outputs is not None
                last_sampled_token_ids = \
                    self._get_last_sampled_token_ids(virtual_engine)

                execute_model_req = ExecuteModelRequest(
                    seq_group_metadata_list=seq_group_metadata_list,
                    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                    blocks_to_copy=scheduler_outputs.blocks_to_copy,
                    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                    running_queue_size=scheduler_outputs.running_queue_size,
                    finished_requests_ids=finished_requests_ids,
                    # We use ExecuteModelRequest to pass the last sampled_token_ids
                    # to each of the non-last PP stages for in-place prepare_input.
                    last_sampled_token_ids=last_sampled_token_ids)
                outputs = self.model_executor.execute_model(
                    execute_model_req=execute_model_req)

                if len(outputs) == 1:
                    self._advance_to_next_step(
                        outputs[0], seq_group_metadata_list,
                        scheduler_outputs.scheduled_seq_groups)
                scheduler_outputs.scheduled_seq_groups = [item for item in scheduler_outputs.scheduled_seq_groups] #deep copy
                last_sampler = get_last_sampler()    
                self.last_record = [outputs, last_sampler, seq_group_metadata_list, scheduler_outputs]

        except Exception as e:
            print(f"thread_zero_overhead error : {e}")
            traceback.print_exc()

    def zero_overhead_step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        if not self.thread_running:
            self.zero_thread.join()
            self.thread_running = True
            self.zero_thread = threading.Thread(target=self.thread_zero_overhead)
            self.zero_thread.start()
        self.sem_m2s.release()
        recode_output = self.q_recorder.get()
        if recode_output is None: # None is for the first step
            return None
        virtual_engine = 0
        ctx = self.scheduler_contexts[virtual_engine]
        ctx.request_outputs.clear()
        outputs, last_sampler, seq_group_metadata_list, scheduler_outputs = recode_output
        ctx.seq_group_metadata_list = seq_group_metadata_list
        ctx.scheduler_outputs = scheduler_outputs
        self.async_event.synchronize()
        self._fix_last_step(
            outputs, last_sampler, seq_group_metadata_list,
            scheduler_outputs.scheduled_seq_groups)

        # is_first_step_output is True only when the num_steps of all
        # the sequences are 1. When the num_steps > 1,
        # multi_step_model_runner does the first-step output append.
        is_first_step_output: bool = False if not seq_group_metadata_list \
            else seq_group_metadata_list[0].state.num_steps == 1

        # Add results to the output_queue
        ctx.append_output(outputs=outputs,
                            seq_group_metadata_list=seq_group_metadata_list,
                            scheduler_outputs=scheduler_outputs,
                            is_async=True,
                            is_last_step=True,
                            is_first_step_output=is_first_step_output)

        # Check if need to run the usual non-async path
        #if not allow_async_output_proc:
        self._process_model_outputs(ctx=ctx)

        #profile.ProfRangeAutoPush('has_unfinish')
        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()
        return ctx.request_outputs
    
    
    def _fix_last_step(
            self, output: List[SamplerOutput],
            last_sampler: SampleRecorder,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            scheduled_seq_groups: List[ScheduledSequenceGroup]) -> None:
        
        #sample_out_list = output[0].sampler_out_tenosr.cpu().tolist()
        sample_out_list = self.async_d2h.tolist()
        sample_out_ids = last_sampler.seq_id.tolist()
        for seq_group_metadata, sequence_group_outputs, scheduled_seq_group in \
            zip(seq_group_metadata_list, output[0], scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                continue

            if seq_group_metadata.do_sample:
                sample = sequence_group_outputs.samples[0]

                assert len(seq_group.seqs) == 1
                seq : ZeroOverheadSequence = seq_group.seqs[0]
                for token_id, seq_id in zip(sample_out_list, sample_out_ids):
                    if seq.seq_id == seq_id:
                        if type(token_id) is list:
                            sample.output_token = token_id[0]
                        else:
                            sample.output_token = token_id
                        seq.fix_last_token_id(sample.output_token)
                        break

    def no_thread_step(self):
        virtual_engine = 0
        # Clear outputs for each new scheduler iteration
        
        # Schedule iteration
        (seq_group_metadata_list, scheduler_outputs,
            allow_async_output_proc
            ) = self.scheduler[virtual_engine].schedule()
        if self.last_record is not None:
            last_sampler = self.last_record[1]
            self.async_d2h = last_sampler.sampled_token_ids_tensor.to('cpu', non_blocking=True)
            self.async_event.record()
            self.q_recorder.put(self.last_record)
        else:
            self.q_recorder.put(None)
        if len(seq_group_metadata_list) == 0:
            self.last_record = None
        else:
            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()

            assert seq_group_metadata_list is not None
            assert scheduler_outputs is not None
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)
            outputs = self.model_executor.execute_model(
                execute_model_req=execute_model_req)

            if len(outputs) == 1:
                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)
            scheduler_outputs.scheduled_seq_groups = [item for item in scheduler_outputs.scheduled_seq_groups] #deep copy
            last_sampler = get_last_sampler()    
            self.last_record = [outputs, last_sampler, seq_group_metadata_list, scheduler_outputs]

        recode_output = self.q_recorder.get()
        if recode_output is None: # None is for the first step
            return None
        virtual_engine = 0
        ctx = self.scheduler_contexts[virtual_engine]
        ctx.request_outputs.clear()
        outputs, last_sampler, seq_group_metadata_list, scheduler_outputs = recode_output
        ctx.seq_group_metadata_list = seq_group_metadata_list
        ctx.scheduler_outputs = scheduler_outputs
        self.async_event.synchronize()
        self._fix_last_step(
            outputs, last_sampler, seq_group_metadata_list,
            scheduler_outputs.scheduled_seq_groups)

        # is_first_step_output is True only when the num_steps of all
        # the sequences are 1. When the num_steps > 1,
        # multi_step_model_runner does the first-step output append.
        is_first_step_output: bool = False if not seq_group_metadata_list \
            else seq_group_metadata_list[0].state.num_steps == 1

        # Add results to the output_queue
        ctx.append_output(outputs=outputs,
                            seq_group_metadata_list=seq_group_metadata_list,
                            scheduler_outputs=scheduler_outputs,
                            is_async=True,
                            is_last_step=True,
                            is_first_step_output=is_first_step_output)

        # Check if need to run the usual non-async path
        #if not allow_async_output_proc:
        self._process_model_outputs(ctx=ctx)

        #profile.ProfRangeAutoPush('has_unfinish')
        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()
        return ctx.request_outputs
                    
    def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        if is_zero_no_thread():
            out = self.no_thread_step()
            if out is None: #the first step need launch twice
                out = self.no_thread_step()
        else:
            out = self.zero_overhead_step()
            if out is None: #the first step need launch twice
                out = self.zero_overhead_step()
        return out
    
    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: ProcessorInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> Optional[SequenceGroup]:
        """Add a processed request to the engine's request pool.
        return the created sequence group.
        """
        if isinstance(params, SamplingParams) and params.n > 1:
            ParallelSampleSequenceGroup.add_request(
                request_id,
                self,
                params,
                processed_inputs=processed_inputs,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )
            return None

        self._validate_model_inputs(processed_inputs, lora_request)
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        seq = ZeroOverheadSequence(seq_id, decoder_inputs, block_size, eos_token_id,
                       lora_request, prompt_adapter_request)

        encoder_seq = (None if encoder_inputs is None else ZeroOverheadSequence(
            seq_id, encoder_inputs, block_size, eos_token_id, lora_request,
            prompt_adapter_request))

        # Create a SequenceGroup based on SamplingParams or PoolingParams
        if isinstance(params, SamplingParams):
            seq_group = self._create_sequence_group_with_sampling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        elif isinstance(params, PoolingParams):
            seq_group = self._create_sequence_group_with_pooling(
                request_id,
                seq,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                encoder_seq=encoder_seq,
                priority=priority)
        else:
            raise ValueError(
                "Either SamplingParams or PoolingParams must be provided.")

        # Add the sequence group to the scheduler with least unfinished seqs.
        costs = [
            scheduler.get_num_unfinished_seq_groups()
            for scheduler in self.scheduler
        ]
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(seq_group)

        return seq_group