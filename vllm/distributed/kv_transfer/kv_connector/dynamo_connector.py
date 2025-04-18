# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class DynamoConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
        world_group,
    ):

        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.rank = rank

        if self.config.kv_connector != "DynamoNcclConnector":
            raise NotImplementedError("Only DynamoNcclConnector is supported by the DynamoConnector class")

        from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
            PyNcclPipe)
        from vllm.distributed.kv_transfer.kv_pipe.dynamo_nccl_pipe import (
            DynamoNcclDataPlane)
        
        logger.info(
            "Initializing DynamoNcclConnector under kv_transfer_config %s",
            self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_data_pipe: PyNcclPipe
        self.consumer_data_pipe: PyNcclPipe
        self.producer_signal_pipe: PyNcclPipe
        self.consumer_signal_pipe: PyNcclPipe

        self._broadcast_and_enhance_kv_config(rank, config, world_group)

        self.kv_group_rank = self._get_kv_group_rank(self.config.kv_rank, rank, self.config)
        self.tp_size = config.parallel_config.tensor_parallel_size

        # 2 pipes for every rank in the world
        if self.config.is_kv_producer:
            port_offset_base = rank + 1
        else:
            port_offset_base = rank // self.config.tensor_parallel_multiplier + 1


        self.local_kv_rank = rank % self.config.tensor_parallel_multiplier
        self.global_kv_rank = self._get_global_kv_rank(self.config.kv_rank, rank, self.config)

        self.data_pipe = PyNcclPipe(
            kv_group_rank=self.kv_group_rank,
            local_rank=local_rank,
            config=self.config,
            port_offset=port_offset_base,
        )

        self.data_plane = DynamoNcclDataPlane(
            data_pipe=self.data_pipe,
            port=self._get_data_plane_port(self.global_kv_rank),
        )

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        model_config = model_executable.model.config
        is_deepseek = "deepseek" in model_config.architectures[0].lower()
        if not is_deepseek:
            num_heads = int(model_config.num_key_value_heads / self.tp_size)
            hidden_size = model_config.hidden_size
            num_attention_heads = model_config.num_attention_heads
            head_size = int(hidden_size / num_attention_heads)
        else:
            num_heads = int(model_config.num_key_value_heads / self.tp_size)
            hidden_size = model_config.hidden_size
            num_attention_heads = model_config.num_attention_heads
            head_size = int(4.5 * hidden_size / num_attention_heads)

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            current_request_id = request_ids[idx]
            decode_hostname, decode_kv_rank = self.parse_request_id(current_request_id)
            decode_first_global_rank = self._get_global_kv_rank(decode_kv_rank, self.rank * self.config.tensor_parallel_multiplier, self.config)

            for target_rank in range(self.config.tensor_parallel_multiplier):

                keys, values = [], []

                for layer_id in range(start_layer, end_layer):
                    kv_cache = kv_caches[layer_id - start_layer]

                    current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                    num_heads_per_rank = num_heads // self.config.tensor_parallel_multiplier
                    head_start = target_rank * num_heads_per_rank
                    head_end = head_start + num_heads_per_rank

                    if not is_deepseek:
                        key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                        value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
                        keys.append(key_cache[current_slot_mapping, head_start:head_end].unsqueeze(0))
                        values.append(value_cache[current_slot_mapping, head_start:head_end].unsqueeze(0))
                    else:
                        key_cache = kv_cache
                        keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                        values.append(torch.empty(0))

                keys = torch.cat(keys, dim=0)
                values = torch.cat(values, dim=0)

                decode_global_rank = decode_first_global_rank + target_rank
                decode_port = self._get_data_plane_port(decode_global_rank)
                partial_hidden_or_intermediate_states = hidden_or_intermediate_states[start_pos:end_pos]
                self._send(decode_hostname, decode_port, current_request_id, keys, values,
                            partial_hidden_or_intermediate_states)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        start_pos_list = []

        model_config = model_executable.model.config
        is_deepseek = "deepseek" in model_config.architectures[0].lower()

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            current_request_id = request_ids[idx]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self._recv(current_request_id)
            keys: torch.Tensor = ret[0]
            values: torch.Tensor = ret[1]
            hidden: torch.Tensor = ret[2]

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                if not is_deepseek:
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                    ops.reshape_and_cache_flash(
                        keys[i - model_executable.model.start_layer].to(
                            key_cache.device),
                        values[i - model_executable.model.start_layer].to(
                            value_cache.device),
                        key_cache,
                        value_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale,
                        layer.self_attn.attn._v_scale,
                    )
                else:
                    key_cache = kv_cache
                    copy_from =keys[i - model_executable.model.start_layer].to(
                            key_cache.device)
                    kv_cache[slot_mapping[start_pos:end_pos]] = copy_from

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.data_pipe.close()
        # self.data_plane.close()

    @staticmethod
    def parse_request_id(request_id: str) -> Tuple[str, int]:
        # Regular expression to match the string hostname and integer decode_kv_rank
        pattern = r"___decode_hostname_(.*)___decode_kv_rank_(\d+)"
        
        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            decode_hostname = match.group(1)
            decode_rank = int(match.group(2))
            
            return decode_hostname, decode_rank
        raise ValueError(f"Request id {request_id} does not contain hostname and decode_kv_rank")

    def _send(self, hostname: str, port: int, request_id: str, keys: torch.Tensor, values: torch.Tensor, hidden: torch.Tensor):
        remote_address = f"{hostname}:{port}"
        self.data_plane.send_tensor(keys, f"{request_id}_keys", remote_address)
        self.data_plane.send_tensor(values, f"{request_id}_values", remote_address)
        self.data_plane.send_tensor(hidden, f"{request_id}_hidden", remote_address)

    def _recv(self, request_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = self.data_plane.recv_tensor(f"{request_id}_keys")
        values = self.data_plane.recv_tensor(f"{request_id}_values")
        hidden = self.data_plane.recv_tensor(f"{request_id}_hidden")
        return keys, values, hidden

    def _get_kv_group_rank(self, kv_rank: int, rank: int, config: KVTransferConfig) -> int:
        if kv_rank < config.kv_producers_parallel_size:
            return kv_rank
        
        kv_consumer_rank = kv_rank - config.kv_producers_parallel_size
        return config.kv_producers_parallel_size + kv_consumer_rank * config.tensor_parallel_multiplier + rank % config.tensor_parallel_multiplier
    

    def _get_global_kv_rank(self, kv_rank: int, rank: int, config: KVTransferConfig) -> int:
        if kv_rank <= config.kv_producers_parallel_size:
            return kv_rank * config.kv_producers_tensor_parallel_size + rank
        
        kv_consumer_rank = kv_rank - config.kv_producers_parallel_size
        return config.kv_producers_parallel_size * config.kv_producers_tensor_parallel_size + kv_consumer_rank * config.kv_consumers_tensor_parallel_size + rank


    def _get_data_plane_port(self, global_kv_rank: int) -> int:
        return self.config.kv_port + self.config.kv_producers_tensor_parallel_size + 1 + global_kv_rank

    def _broadcast_and_enhance_kv_config(self, rank: int, config: VllmConfig, world_group):
        if rank == 0:
            config_group = StatelessProcessGroup.create(
                host=self.config.kv_ip,
                port=self.config.kv_port,
                rank=self.config.kv_rank,
                world_size=self.config.kv_parallel_size,
            )
            parallel_configs = config_group.all_gather_obj({
                "kv_role": self.config.kv_role,
                "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
                "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
            })
            logger.debug("parallel_configs: %s", parallel_configs)
            kv_config_enhanced = {
                "kv_producers_tensor_parallel_size": None,
                "kv_consumers_tensor_parallel_size": None,
                "kv_producers_pipeline_parallel_size": None,
                "kv_consumers_pipeline_parallel_size": None,
                "kv_producers_parallel_size": 0,
            }
            for parallel_config in parallel_configs:
                kv_role = parallel_config["kv_role"]
                assert parallel_config["pipeline_parallel_size"] == 1, f"Only pipeline parallel size 1 is supported for kv transfer instances"
                
                if kv_role == "kv_producer":
                    kv_config_enhanced["kv_producers_parallel_size"] += 1
                if kv_config_enhanced[f"{kv_role}s_tensor_parallel_size"] is None:
                    kv_config_enhanced[f"{kv_role}s_tensor_parallel_size"] = parallel_config["tensor_parallel_size"]
                    kv_config_enhanced[f"{kv_role}s_pipeline_parallel_size"] = parallel_config["pipeline_parallel_size"]
                else:
                    assert kv_config_enhanced[f"{kv_role}s_tensor_parallel_size"] == parallel_config["tensor_parallel_size"], f"All kv {kv_role}s should have the same tensor parallel size"
                    assert kv_config_enhanced[f"{kv_role}s_pipeline_parallel_size"] == parallel_config["pipeline_parallel_size"], f"All kv {kv_role}s should have the same pipeline parallel size"
            world_group.broadcast_object(kv_config_enhanced)
        else:
            kv_config_enhanced = world_group.broadcast_object()
        logger.info("kv_config_enhanced: %s", kv_config_enhanced)

        self.config.kv_producers_tensor_parallel_size = kv_config_enhanced["kv_producers_tensor_parallel_size"]
        self.config.kv_consumers_tensor_parallel_size = kv_config_enhanced["kv_consumers_tensor_parallel_size"]
        self.config.kv_producers_pipeline_parallel_size = kv_config_enhanced["kv_producers_pipeline_parallel_size"]
        self.config.kv_consumers_pipeline_parallel_size = kv_config_enhanced["kv_consumers_pipeline_parallel_size"]
        self.config.kv_producers_parallel_size = kv_config_enhanced["kv_producers_parallel_size"]
