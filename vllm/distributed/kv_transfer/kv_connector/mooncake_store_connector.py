# SPDX-License-Identifier: Apache-2.0
"""
MooncakeStore Connector for Distributed Machine Learning Inference

The MooncakeStoreConnector transfers KV caches between prefill vLLM workers
(KV cache producer) and decode vLLM workers (KV cache consumer) using a
database-style KVStore.
"""
import hashlib
from typing import TYPE_CHECKING, List, Tuple, Union
import time
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata
from vllm_hpu_extension.utils import VLLMKVCache

import habana_frameworks.torch as htorch

logger = init_logger(__name__)


class MooncakeStoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.local_rank = local_rank
        self.k_head_size = 64
        self.v_head_size = 512
        self.k_v_head_size = self.k_head_size + self.v_head_size
        self.block_size = 128
        self.local_offset_start = self.k_v_head_size // self.tp_size * local_rank
        self.local_offset_end = self.k_v_head_size // self.tp_size * (local_rank + 1)
        self.local_tp_rank = local_rank

        # Init kv_store
        if self.config.kv_connector == "MooncakeStoreConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_store = os.getenv('MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_store:
                raise ValueError(
                    "To use MooncakeStoreConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import (  # noqa: E501
                    MooncakeStore)
                logger.info(
                    "Initializing KVStoreConnector under kv_transfer_config %s",
                    self.config)
                self.kv_store = MooncakeStore(config)
        else:
            logger.error("Can not find %s", self.config.kv_connector)

        assert self.kv_store is not None
        if config.cache_config.cache_dtype == "fp8_inc":
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.bfloat16
        self.padding_k_tensor = torch.zeros((self.block_size, self.k_v_head_size), dtype=dtype, device="hpu")
        self.padding_v_tensor = torch.zeros((self.block_size, self.v_head_size), dtype=dtype, device="hpu")
        self.cache_k = VLLMKVCache()
        self.cache_v = VLLMKVCache()
        
        
    def close(self) -> None:
        """Close the buffer and release resources.
        This method is responsible for cleaning up resources related to the 
        connector when it is no longer needed.
        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        self.kv_store.close()

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

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            store_key_prefix = self.tensor_hash(current_tokens)
            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            kvcache_to_sent = torch.stack((keys, values), dim=0)
            store_kvcache_key = f"{store_key_prefix}_{self.local_tp_rank}"
            self.kv_store.put(store_kvcache_key, kvcache_to_sent)

            hidden_key = f"{store_key_prefix}_hidden_{self.local_tp_rank}"
            self.kv_store.put(hidden_key,
                              hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        bypass_model_exec = True
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_or_intermediate_states_for_one_req = []

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
                               "should be equal to max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            # get roi for current seq
            load_key_prefix = self.tensor_hash(current_tokens)
            load_kvcache_key = f"{load_key_prefix}_{self.local_tp_rank}"
            remote_kv = self.kv_store.get(load_kvcache_key)
            hidden_key = f"{load_key_prefix}_hidden_{self.local_tp_rank}"
            hidden = self.kv_store.get(hidden_key)

            if remote_kv is None or hidden is None:
                # didn't find any match.
                bypass_model_exec = False
                continue

            num_computed_tokens = current_tokens.shape[0]

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # call self.kv_store to get kv layer by layer
            for layer_id in range(start_layer, end_layer):
                layer = model_executable.model.layers[layer_id]
                # get kvcache object
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                # get remote kvcache

                remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][
                    layer_id]
                # use ops.reshape_and_cache_flash to put kv into kvcache
                ops.reshape_and_cache_flash(
                    remote_k.to(key_cache.device),
                    remote_v.to(value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
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


    def send_kv_caches_and_hidden_states_hpu(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens # shape: [batch_size, seq_len_padding_to_128]
        seq_lens = model_input.attn_metadata.seq_lens # 2D list
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_kv_heads = 1
        k_head_size = 64
        v_head_size = 512

        model_config = model_executable.config
        # not used for MLA since key and value size are not equal 
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # For each sequence in the batch, we will pack kv together, so we send
        # 0. current_tokens [seq_len]
        # 1. bool mask [seq_len]
        # 2. key [num_layers, seq_len, num_kv_heads, (k_head_size + v_head_size)], [61, seq_len, 1, 576]
        # 3. empty tensor
        # 4. hidden_or_intermediate_states [1, hidden_size]
        for idx, slen in enumerate(seq_lens):
            if slen == 1: # we think this is a padding sequence, so we skip it
                continue
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[idx][:slen]
            store_key_prefix = self.tensor_hash(current_tokens)
            logger.debug(f"send token len: {slen}, token: {current_tokens}")
            keys, values = [], []
            start = 0
            end = slen
            current_slot_mapping = model_input.attn_metadata.slot_mapping[idx][start:end]

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache = kv_cache[0].reshape(-1, num_kv_heads, self.k_v_head_size)
                # value_cache = kv_cache[1].reshape(-1, num_kv_heads, v_head_size)

                htorch.core.mark_step()
                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                # values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            # values = torch.cat(values, dim=0)
            # we pack kv together, only need send one tensor
            kvcache_to_sent = keys
            store_kvcache_key = f"{store_key_prefix}_{self.local_tp_rank}"
            self.kv_store.put(store_kvcache_key, kvcache_to_sent)
            
            logger.debug(f"put kv cache key: {store_kvcache_key}")
            
            hidden_key = f"{store_key_prefix}_hidden_{self.local_tp_rank}"
            self.kv_store.put(hidden_key,
                              hidden_or_intermediate_states[idx].unsqueeze(0).cpu())

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


    def recv_kv_caches_and_hidden_states_hpu(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        attn_metadata: object,
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForHPUWithSamplingMetadata"]:
        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens_tensor = model_input.attn_metadata.seq_lens_tensor
        seq_lens = seq_lens_tensor.tolist() #2D list

        hidden_or_intermediate_states_for_one_req = []
        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []
        num_kv_heads = 1
        start_block_idx = 0
        # write to cache

        # For each sequence in the batch, we patch kv tensor together, so we recv
        # 0. current_tokens [seq_len]
        # 1. bool mask [seq_len]
        # 2. key_values [num_layers, seq_len, num_kv_heads, (k_head_size + v_head_size)], [61, seq_len, 1, 576]
        # 3. empty tensor
        # 4. hidden_or_intermediate_states [1, hidden_size]
        for idx, slen in enumerate(seq_lens):
            start = time.time()

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[idx][:slen]
            num_tokens = slen
            num_blocks = (slen + 127) // 128
            padding_size = (128 - slen % 128) % 128
            end_block_idx = start_block_idx + num_blocks

            # we think this is a padding sequence, so we skip it. but we still need write kv cache
            if slen == 1:
                for i in range(model_executable.model.model.start_layer,
                               model_executable.model.model.end_layer):
                    current_layer_idx = i - model_executable.model.model.start_layer
                    kv_cache = kv_caches[current_layer_idx]
                    # key_cache, value_cache = kv_cache[0], kv_cache[1]
                    key_cache = kv_cache[0]
                    self.cache_k(self.padding_k_tensor.unsqueeze(0),
                            key_cache,
                            attn_metadata.block_indices[start_block_idx:end_block_idx],
                            attn_metadata.block_offsets,
                            )
                    # self.cache_v(self.padding_v_tensor.unsqueeze(0),
                    #         value_cache,
                    #         attn_metadata.block_indices[start_block_idx:end_block_idx],
                    #         attn_metadata.block_offsets,
                    #         )
                # the first one should never be padding, so we can append the first one.
                hidden_or_intermediate_states_for_one_req.append(hidden_or_intermediate_states_for_one_req[0])
                start_block_idx = end_block_idx
                continue

            # get roi for current seq
            load_key_prefix = self.tensor_hash(current_tokens)
            load_kvcache_key = f"{load_key_prefix}_{self.local_tp_rank}"
            remote_kv = self.kv_store.get(load_kvcache_key)
            hidden_key = f"{load_key_prefix}_hidden_{self.local_tp_rank}"
            hidden = self.kv_store.get(hidden_key)
            
            if remote_kv is None or hidden is None:
                # didn't find any match.
                logger.warning(f"Didn't find any match, load_key_prefix: {load_kvcache_key}")
                bypass_model_exec = False
                continue

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            # all gather here
            key_values = remote_kv.to("hpu")
            # torch.hpu.synchronize()
            # all_gather_start = time.time()
            # key_values = tensor_model_parallel_all_gather(key_values, -1)
            torch.hpu.synchronize()
            # end = time.time()
            # logger.debug(f"all gather time takes: {end - all_gather_start}")

            keys = key_values
            # values = key_values[..., self.k_head_size:]
            num_computed_tokens = current_tokens.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)
            cur = time.time()
            logger.debug(f"select + allgather time for this request: {cur - start}")

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            # if not all([(num_computed_tokens == num_tokens), hidden is not None
            #             ]):
            #     logger.warning(f"Cannot bypass, num_computed_tokens: {num_computed_tokens}, num_tokens: {num_tokens}, hidden: {hidden}")
            #     bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory layer by layer
            # for each layer, we need to pad the key and value to 128, so 
            # key shape should be [num_blocks, block_size, num_kv_heads(1,ommited), k_head_size]
            # value shape should be [num_blocks, block_size, num_kv_heads(1,ommited), v_head_size]

            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):
                current_layer_idx = i - model_executable.model.start_layer
                kv_cache = kv_caches[current_layer_idx]
                layer = model_executable.model.layers[i] # for kv scale

                key_cache, value_cache = kv_cache[0], kv_cache[1]

                # [num_layers, seq_len, num_kv_heads, k/v_head_size] -> [seq_len, k/v_head_size]
                key = keys[current_layer_idx].squeeze(-2)
                # value = values[current_layer_idx].squeeze(-2) 

                # [seq_len, k/v_head_size] ->(padding [seq_len % block_size, k/v_head_size]) ->
                # [num_blocks * block_size, k/v_head_size]
                key = torch.cat([key, self.padding_k_tensor[:padding_size]], dim=0)
                # value = torch.cat([value, self.padding_v_tensor[:padding_size]], dim=0)

                # [num_blocks, block_size, k/v_head_size]
                key = key.view(num_blocks, self.block_size, self.k_v_head_size)
                # value = value.view(num_blocks, self.block_size, self.v_head_size)

                # ====== D2D =======
                self.cache_k(key,
                        key_cache,
                        attn_metadata.block_indices[start_block_idx:end_block_idx],
                        attn_metadata.block_offsets,
                        )
                # self.cache_v(value,
                #         value_cache,
                #         attn_metadata.block_indices[start_block_idx:end_block_idx],
                #         attn_metadata.block_offsets,
                #         )
            start_block_idx = end_block_idx
            hidden_or_intermediate_states_for_one_req.append(hidden.to("hpu"))
            end = time.time()
            logger.debug(f"cache time: {end - cur}")
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
                hidden_or_intermediate_states_for_one_req, dim=0).to("hpu")

        return hidden_or_intermediate_states, bypass_model_exec, model_input



    @staticmethod
    def tensor_hash(tensor: torch.Tensor) -> int:
        """Calculate the hash value of the tensor."""
        tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
        hash_object = hashlib.blake2b(tensor_bytes)
        hash_hex = hash_object.hexdigest()
        return int(hash_hex[:16], 16)
