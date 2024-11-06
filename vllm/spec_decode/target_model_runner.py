from typing import List, Optional

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.sequence import SequenceGroupMetadata

if current_platform.is_cuda_alike():
    from vllm.worker.model_runner import (
        ModelInputForGPUWithSamplingMetadata as ModelInputCls)  # yapf: disable
    from vllm.worker.model_runner import ModelRunner as ModelRunnerCls
elif current_platform.is_neuron():
    from vllm.worker.neuron_model_runner import (
        ModelInputForNeuron as ModelInputCls)  # yapf: disable
    from vllm.worker.neuron_model_runner import (
        NeuronModelRunner as ModelRunnerCls)  # yapf: disable
elif current_platform.is_hpu():
    from vllm.worker.hpu_model_runner import HPUModelRunner as ModelRunnerCls
    from vllm.worker.hpu_model_runner import (
        ModelInputForHPUWithSamplingMetadata as ModelInputCls)  # yapf: disable
elif current_platform.is_openvino():
    from vllm.worker.openvino_model_runner import ModelInput as ModelInputCls
    from vllm.worker.openvino_model_runner import (
        OpenVINOModelRunner as ModelRunnerCls)  # yapf: disable
elif current_platform.is_cpu():
    from vllm.worker.cpu_model_runner import CPUModelRunner as ModelRunnerCls
    from vllm.worker.cpu_model_runner import (
        ModelInputForCPUWithSamplingMetadata as ModelInputCls)  # yapf: disable
elif current_platform.is_tpu():
    from vllm.worker.tpu_model_runner import ModelInputForTPU as ModelInputCls
    from vllm.worker.tpu_model_runner import TPUModelRunner as ModelRunnerCls
elif current_platform.is_xpu():
    from vllm.worker.xpu_model_runner import (
        ModelInputForXPUWithSamplingMetadata as ModelInputCls)  # yapf: disable
    from vllm.worker.xpu_model_runner import XPUModelRunner as ModelRunnerCls
else:
    raise ValueError(f"Unsupported platform: {current_platform}")


class TargetModelRunner(ModelRunnerCls):
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
    ):
        # An internal boolean member variable to indicate if token log
        # probabilities are needed or not.
        self.disable_logprobs = True
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            return_hidden_states=return_hidden_states,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputCls:
        model_input: ModelInputCls = super().prepare_model_input(
            seq_group_metadata_list, virtual_engine, finished_requests_ids)
        # If token log probabilities is disabled then skip generating sampler
        # CPU output. We directly serialize the GPU sampled_token_id tensors
        # as needed. If log probabilities is enabled then synchronize all the
        # sampling related tensors which includes the logprobs tensors.
        model_input.sampling_metadata.skip_sampler_cpu_output = (
            self.disable_logprobs)
        return model_input
