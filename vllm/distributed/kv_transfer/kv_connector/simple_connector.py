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

import vllm.envs as envs
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


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
        world_group,
    ):

        self.config = config.kv_transfer_config
        self.kv_group_rank = self._get_kv_group_rank(self.config.kv_rank, rank, self.config)
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE

        if self.config.kv_connector == "PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            logger.info(
                "Initializing PyNcclConfig under kv_transfer_config %s",
                self.config)
        elif self.config.kv_connector == "MooncakeConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_distributed_pipe = os.getenv(
                'MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_distributed_pipe:
                raise ValueError(
                    "To use MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                    MooncakePipe)
                logger.info(
                    "Initializing MooncakeConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]

        self._broadcast_and_enhance_kv_config(rank, config, world_group)

        self.kv_group_rank = self._get_kv_group_rank(self.config.kv_rank, rank, self.config)
        self.tp_size = config.parallel_config.tensor_parallel_size

        # 2 pipes for every rank in the world
        if self.config.is_kv_producer:
            port_offset_base = 2 * rank + 1
        else:
            port_offset_base = 2 * (rank // self.config.tensor_parallel_multiplier) + 1

        self.local_kv_rank = rank % self.config.tensor_parallel_multiplier
        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:

            if self.config.kv_connector == "PyNcclConnector":
                self.producer_data_pipe = PyNcclPipe(
                    kv_group_rank=self.kv_group_rank,
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.producer_signal_pipe = PyNcclPipe(
                    kv_group_rank=self.kv_group_rank,
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.producer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                # We only need to initialize MooncakePipe once
                self.producer_signal_pipe = self.producer_data_pipe

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size)

        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder
            if self.config.kv_connector == "PyNcclConnector":
                self.consumer_data_pipe = PyNcclPipe(
                    kv_group_rank=self.kv_group_rank,
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.consumer_signal_pipe = PyNcclPipe(
                    kv_group_rank=self.kv_group_rank,
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.consumer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.consumer_signal_pipe = self.consumer_data_pipe

            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    def select(self, source_rank: int, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        logger.info("Selecting KV caches and hidden states for source rank %d", source_rank)

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(source_rank, self.local_kv_rank, input_tokens, roi)

    def insert(self, kv_group_rank: int, target_rank: int, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        logger.info("Inserting KV caches and hidden states for kv_group_rank %d, target rank %d", kv_group_rank, target_rank)

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(kv_group_rank, target_rank, input_tokens, roi, key, value, hidden)

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
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim",
                                int(hidden_size // num_attention_heads))

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You have some decode requests while using "
                               "SimpleConnector. Their KVCache won't be sent.")
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            current_request_id = request_ids[idx]
            _, decode_kv_rank = self.parse_request_id(current_request_id)
            starting_kv_group_rank = self._get_kv_group_rank(decode_kv_rank, 0, self.config)

            for target_rank in range(self.config.tensor_parallel_multiplier):

                if self.is_deepseek_mla and self.use_mla_opt:
                    key_cache = kv_cache.reshape(-1, num_heads, head_size)
                    value_cache = kv_cache.reshape(-1, num_heads, head_size)
                else:
                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

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

                self.insert(starting_kv_group_rank, target_rank, current_tokens,
                            torch.ones_like(current_tokens,
                                            dtype=bool), keys, values,
                            hidden_or_intermediate_states[start_pos:end_pos])

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

        model_config = model_executable.model.config

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        model_config = model_executable.model.config
        is_deepseek = "deepseek" in model_config.architectures[0].lower()

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to --max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            current_request_id = request_ids[idx]
            prefill_rank, _ = self.parse_request_id(current_request_id)
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(prefill_rank, current_tokens,
                              torch.ones_like(current_tokens, dtype=bool))
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                if self.is_deepseek_mla and self.use_mla_opt:
                    layer.self_attn.attn = layer.self_attn.mla_attn
                    k_c_normed_k_pe = keys[
                        i - model_executable.model.start_layer].to(
                            kv_cache.device).squeeze(1)
                    k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
                    k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
                    ops.concat_and_cache_mla(
                        k_c_normed,
                        k_pe,
                        kv_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale,
                    )
                else:
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

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.warning(
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
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == "PyNcclConnector":
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass

    @staticmethod
    def parse_request_id(request_id):
        # Regular expression to match the ranks
        pattern = r"___prefill_kv_rank_(\d+)___decode_kv_rank_(\d+)"
        
        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        
        if match:
            # Extract the ranks
            prefill_rank = int(match.group(1))
            decode_rank = int(match.group(2))
            
            return prefill_rank, decode_rank
        else:
            return None, None

    

    def _get_kv_group_rank(self, kv_rank: int, rank: int, config: KVTransferConfig) -> int:
        if kv_rank < config.kv_producers_parallel_size:
            return kv_rank
        
        kv_consumer_rank = kv_rank - config.kv_producers_parallel_size
        return config.kv_producers_parallel_size + kv_consumer_rank * config.tensor_parallel_multiplier + rank % config.tensor_parallel_multiplier

    def _broadcast_and_enhance_kv_config(self, rank: int, config: VllmConfig, world_group):
        if rank == 0:
            if self.config.kv_connector == "PyNcclConnector":
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
                raise NotImplementedError("MooncakeConnector is not supported in Dynamo patch")
        else:
            kv_config_enhanced = world_group.broadcast_object()
        logger.info("kv_config_enhanced: %s", kv_config_enhanced)

        self.config.kv_producers_tensor_parallel_size = kv_config_enhanced["kv_producers_tensor_parallel_size"]
        self.config.kv_consumers_tensor_parallel_size = kv_config_enhanced["kv_consumers_tensor_parallel_size"]
        self.config.kv_producers_pipeline_parallel_size = kv_config_enhanced["kv_producers_pipeline_parallel_size"]
        self.config.kv_consumers_pipeline_parallel_size = kv_config_enhanced["kv_consumers_pipeline_parallel_size"]
        self.config.kv_producers_parallel_size = kv_config_enhanced["kv_producers_parallel_size"]
