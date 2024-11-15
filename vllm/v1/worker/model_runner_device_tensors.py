"""
Define object that packs all the devices required by the model runner.
"""

from dataclasses import dataclass

import torch

@dataclass
class ModelRunnerDeviceSamplingTensors:
    """
    Device tensors related to model sampling
    """
    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor

    @staticmethod
    def make(max_num_reqs: int,
             device: torch.device) -> "ModelRunnerDeviceSamplingTensors":
        temperature = torch.empty((max_num_reqs, ),
                                  dtype=torch.float32,
                                  device=device)
        top_p = torch.empty((max_num_reqs, ),
                            dtype=torch.float32,
                            device=device)
        top_k = torch.empty((max_num_reqs, ),
                            dtype=torch.int32,
                            device=device)
        return ModelRunnerDeviceSamplingTensors(temperature, top_p, top_k)

@dataclass
class ModelRunnerDeviceTensors:
    """
    Device tensors to be maintained by the ModelRunner 
    """
    block_table: torch.Tensor
    input_positions: torch.Tensor
    input_tokens: torch.Tensor
    inputs_embeds: torch.Tensor
    slot_mapping: torch.Tensor
    sampling_tensors: ModelRunnerDeviceSamplingTensors 

    @staticmethod
    def make(max_num_reqs: int,
             max_num_tokens: int,
             max_num_blocks_per_req: int,
             input_embeds_hidden_size: int,
             input_embeds_dtype: torch.dtype,
             device: torch.device) -> "ModelRunnerDeviceTensors":
        block_table = torch.zeros((max_num_reqs,
                                  max_num_blocks_per_req),
                                  device=device,
                                  dtype=torch.int32)

        input_positions = torch.zeros(max_num_tokens,
                                dtype=torch.int64,
                                device=device)
        
        input_tokens = torch.empty(max_num_tokens,
                                   dtype=torch.int32,
                                   device=device)

        slot_mapping = torch.empty(max_num_tokens,
                                   dtype=torch.long,
                                   device=device)

        inputs_embeds = torch.zeros(
            (max_num_tokens, input_embeds_hidden_size),
             dtype=input_embeds_dtype,
             device=device)

        return ModelRunnerDeviceTensors(
                    block_table = block_table,
                    input_positions = input_positions,
                    input_tokens = input_tokens,
                    inputs_embeds = inputs_embeds,
                    slot_mapping = slot_mapping,
                    sampling_tensors = ModelRunnerDeviceSamplingTensors.make(max_num_reqs, device))