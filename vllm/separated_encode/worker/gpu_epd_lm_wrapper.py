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
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

class DisaggLModelGPURunnerWrapper(GPUModelRunner):
    """
    GPU model runner wrapper for disaggregated Language Model processing.
    
    This class extends GPUModelRunner to support Encode-Prefill-Decode (EPD) 
    disaggregation by handling remote encoder cache injection and management. 
    It integrates with encoder cache connectors to receive and process encoder 
    outputs from remote encoder instances.
    
    The runner maintains encoder cache state and coordinates with the scheduler
    to track successful encoder cache injections for multimodal processing.
    
    Attributes:
        epd_disagg_config : Configuration for EPD disaggregation
        ec_connector : Connector for encoder cache transfer
        injected_encoder_cache_ids : List of successfully  injected encoder 
            cache identifiers
        instance_type : Type of processing instance this runner handles
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        self.epd_disagg_config: EPDDisaggConfig
        self.ec_connector: ECConnectorTemplate
        self.epd_disagg_config: EPDDisaggConfig
        self.injected_encoder_cache_ids: list[tuple[str, int]]
        self.instance_type: Literal["NoEPD", "prefill+decode", "prefill",
                                    "encode"]
        self.epd_disagg_config = vllm_config.epd_disagg_config

        assert self.epd_disagg_config.instance_type != "NoEPD",\
            "Can't use LM instance without EPD disaggregation"
        
        self.instance_type = vllm_config.epd_disagg_config.instance_type
        self.ec_connector = RedisECConnector(
            vllm_config=vllm_config,
            intra_instance_type="model-runner",
            preallocate_callback=None,
            injection_callback=self.receive_encoder_cache,
            redis_host="localhost", # replace with ec_transfer_config later
            redis_port=6379,        # replace with ec_transfer_config later
        )
        self.injected_encoder_cache_ids = []

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """
        Executes the model and includes injected encoder cache information.
        
        Extends the base execute_model functionality to track and report
        encoder cache injections that occurred during model execution.
        The injected encoder cache IDs are included in the model output
        to inform the scheduler about successful cache injections.
        """
        model_runner_output = super().execute_model(scheduler_output,
                                                    intermediate_tensors)
        injected_encoder_cache_ids = None
        with self.encoder_cache_lock:
            injected_encoder_cache_ids = self.injected_encoder_cache_ids
            self.injected_encoder_cache_ids = []
        model_runner_output.injected_mm_data = injected_encoder_cache_ids
        return model_runner_output

    def receive_encoder_cache(self, request_id, input_id, encoder_cache):
        """
        Callback function for receiving encoder cache from remote instances.
        
        This method is invoked by the encoder cache connector when encoder
        cache data is received from remote encoder instances. It processes
        the received numpy array by converting it to a PyTorch tensor with
        the correct device placement and data type, then stores it in the
        local encoder cache dictionary.
        
        The method updates the injected encoder cache IDs list to inform the
        scheduler about successful cache injections.
        """
        with self.encoder_cache_lock:
            if request_id not in self.encoder_cache:
                self.encoder_cache[request_id] = {}
            encoder_cache = torch.from_numpy(encoder_cache)
            self.encoder_cache[request_id][input_id] = encoder_cache.to(
                device=self.device, dtype=self.dtype)
            self.injected_encoder_cache_ids.append((request_id, input_id))
