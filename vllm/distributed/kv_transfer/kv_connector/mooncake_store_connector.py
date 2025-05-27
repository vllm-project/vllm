# SPDX-License-Identifier: Apache-2.0
"""
MooncakeStore Connector for Distributed Machine Learning Inference
The MooncakeStoreConnector transfers KV caches between prefill vLLM workers
(KV cache producer) and decode vLLM workers (KV cache consumer) using a
database-style KVStore.
"""
import hashlib
from typing import TYPE_CHECKING, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import (
    model_aware_kv_ops_helper as kv_helper)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class MooncakeStoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.kv_transfer_config = config.kv_transfer_config
        self.kv_helper = kv_helper(config)
        self.local_tp_rank = local_rank

        # Init kv_store
        if self.kv_transfer_config.kv_connector == "MooncakeStoreConnector":
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
                    self.kv_transfer_config)
                self.kv_store = MooncakeStore(config)
        else:
            logger.error("Can not find %s",
                         self.kv_transfer_config.kv_connector)

        assert self.kv_store is not None

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
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            store_key_prefix = self.tensor_hash(current_tokens)
            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                key_cache, value_cache = self.kv_helper.get_kv_from_cache(
                    kv_cache, num_heads, head_size)
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
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
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

                # get remote kvcache
                remote_k, remote_v = remote_kv[0][layer_id], remote_kv[1][
                    layer_id]

                self.kv_helper.put_kv_to_cache(model_executable, remote_k,
                                               remote_v, layer, kv_cache,
                                               slot_mapping, start_pos,
                                               end_pos)

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

    @staticmethod
    def tensor_hash(tensor: torch.Tensor) -> int:
        """Calculate the hash value of the tensor."""
        tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
        hash_object = hashlib.blake2b(tensor_bytes)
        hash_hex = hash_object.hexdigest()
        return int(hash_hex[:16], 16)
