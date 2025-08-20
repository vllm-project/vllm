# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch

from vllm.config import EPDDisaggConfig, VllmConfig
from vllm.logger import init_logger
from vllm.separated_encode.ec_transfer.connector.redis import (
    RedisECConnector)
from vllm.separated_encode.ec_transfer.connector.template import (
    ECConnectorTemplate)
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class DisaggEncodeGPURunnerWrapper(GPUModelRunner):
    """
    GPU model runner wrapper for disaggregated Vision/Encoder processing.
    
    This class extends GPUModelRunner to support encoder-only processing in
    Encode-Prefill-Decode (EPD) disaggregation. It handles multimodal encoder
    execution and transfers the resulting encoder caches to remote instances
    for further processing.
    
    This wrapper focuses on encoder processing and does not initialize KV cache
    since it doesn't perform language model inference.
    
    Attributes:
        epd_disagg_config: Configuration for EPD disaggregation
        ec_connector: Connector for encoder cache transfer
        instance_type: Type of processing instance this runner handles
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        self.epd_disagg_config: EPDDisaggConfig
        self.ec_connector: ECConnectorTemplate
        self.epd_disagg_config: EPDDisaggConfig = vllm_config.epd_disagg_config
        self.instance_type: Literal["NoEPD", "prefill+decode", "prefill",
                                    "encode"]

        assert self.epd_disagg_config.instance_type != "NoEPD",\
            "Can't use Encode instance without EPD disaggregation"

        self.instance_type = vllm_config.epd_disagg_config.instance_type
        self.ec_connector = RedisECConnector(
            vllm_config=vllm_config,
            intra_instance_type="model-runner",
            preallocate_callback=None,
            injection_callback=None,
            redis_host="localhost",  # replace with ec_transfer_config later
            redis_port=6379,         # replace with ec_transfer_config later
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Updates internal request states based on scheduler output.
        
        Manages the lifecycle of requests by removing finished
        requests and adding newly scheduled requests. This method maintains
        the requests needed for encoder processing.
        """
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        
        for (req_id, input_id) in scheduler_output.free_encoder_input_ids:
            self.encoder_cache[req_id].pop(input_id)
            if not self.encoder_cache[req_id]:
                self.encoder_cache.pop(req_id)

        for new_req_data in scheduler_output.scheduled_new_reqs:
            self.requests[new_req_data.req_id] = CachedRequestState(
                req_id=new_req_data.req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=None,
                pooling_params=None,
                generator=None,
                block_ids=[],
                num_computed_tokens=0,
                output_token_ids=[],
            )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """
        Executes encoder processing and schedules results' transfer.
        
        This method handles the core encoder execution workflow by updating 
        internal request states, executing multimodal encoders for scheduled 
        inputs, and transferring computed encoder caches to remote instances 
        via a connector, while providing transfer status information to the 
        scheduler. 
        
        It also converts encoder outputs into CPU tensors and then to numpy 
        arrays to prepare data for the transfer.        
        """
        self._update_states(scheduler_output)
        self._execute_mm_encoder(scheduler_output)
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs

        for req_id, mm_input_ids in scheduled_encoder_inputs.items():
            for input_id in mm_input_ids:
                encoder_output = self.encoder_cache[req_id][input_id]\
                    .to("cpu", dtype = torch.float32).numpy()
                self.ec_connector.add_encoder_cache(req_id, input_id,
                                                    encoder_output)

        transfered_ids = self.ec_connector.get_transfered_ids()
        # logger.info(f"Arif: Transfered ids: {transfered_ids}")
        model_runner_output = ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            transfered_mm_data=transfered_ids
        )
        return model_runner_output

    # Don't initialize of KV cache on encode instance
    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None
