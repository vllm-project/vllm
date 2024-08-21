from typing import List, Optional

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig)
from vllm.sequence import SequenceGroupMetadata
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)


class TargetModelRunner(ModelRunner):
    """Specialized model runner for speculative decoding target model.
    In speculative decoding, the log probabilities selected finally may not
    be the same ones as selected by the target model sampling. This means
    that the time spent in the log probability calculation of the target model
    is time wasted, since we calculate log probabilities after deciding which
    tokens are accepted. For this reason disabling log probabilities in the
    target model will make decode faster. The model runner sets the
    SamplingMetadata parameters according to whether log probabilities are
    requested or not. 
    """

    def __init__(self,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 scheduler_config: SchedulerConfig,
                 device_config: DeviceConfig,
                 cache_config: CacheConfig,
                 load_config: LoadConfig,
                 lora_config: Optional[LoRAConfig],
                 kv_cache_dtype: Optional[str] = "auto",
                 is_driver_worker: bool = False,
                 prompt_adapter_config: Optional[PromptAdapterConfig] = None,
                 return_hidden_states: bool = False,
                 observability_config: Optional[ObservabilityConfig] = None):
        # An internal boolean member variable to indicate if token log
        # probabilities are needed or not.
        self.disable_logprobs = True
        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            prompt_adapter_config=prompt_adapter_config,
            return_hidden_states=return_hidden_states,
            observability_config=observability_config,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForGPUWithSamplingMetadata:
        model_input: ModelInputForGPUWithSamplingMetadata = super(
        ).prepare_model_input(seq_group_metadata_list, virtual_engine,
                              finished_requests_ids)
        # If token log probabilities is disabled then skip generating sampler
        # CPU output. We directly serialize the GPU sampled_token_id tensors
        # as needed. If log probabilities is enabled then synchronize all the
        # sampling related tensors which includes the logprobs tensors.
        model_input.sampling_metadata.skip_sampler_cpu_output = (
            self.disable_logprobs)
        return model_input
