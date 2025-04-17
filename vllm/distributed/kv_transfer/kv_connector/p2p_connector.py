# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import (
    model_aware_kv_ops_helper as kv_helper)
from vllm.distributed.kv_transfer.kv_pipe.p2p_nccl_pipe import P2pNcclPipe
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class P2pConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.rank = rank
        self.config = config.kv_transfer_config
        self.kv_helper = kv_helper(config)

        assert self.config.kv_connector == "P2pConnector"

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.p2p_nccl_pipe = P2pNcclPipe(
            local_rank=local_rank,
            config=self.config,
            hostname="",
            port_offset=rank,
        )

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        # input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_size = self.kv_helper.get_model_args(model_executable)

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            # current_tokens = input_tokens_tensor[start_pos:end_pos]
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
            kvcache = torch.stack((keys, values), dim=0)

            request_id = request_ids[idx]
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self.rank)

            self.p2p_nccl_pipe.send_tensor(request_id + "kv", kvcache,
                                           remote_address)
            self.p2p_nccl_pipe.send_tensor(
                request_id + "hidden",
                hidden_or_intermediate_states[start_pos:end_pos],
                remote_address)

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
        request_ids = list(model_input.request_ids_to_seq_ids.keys())
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

            request_id = request_ids[idx]
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self.rank)

            kvcache = self.p2p_nccl_pipe.recv_tensor(request_id + "kv",
                                                     remote_address)
            hidden = self.p2p_nccl_pipe.recv_tensor(request_id + "hidden",
                                                    remote_address)

            if kvcache is None or hidden is None:
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
                remote_k, remote_v = kvcache[0][layer_id], kvcache[1][layer_id]

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
    def parse_request_id(request_id: str, is_prefill=True) -> Tuple[str, int]:
        logger.debug("parse_request_id, request_id: %s, is_prefill: %s",
                     request_id, is_prefill)
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            logger.debug("parse_request_id, request_id: %s, ip: %s, port: %s",
                         request_id, ip, str(port))
            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    def close(self) -> None:
        self.p2p_nccl_pipe.close()
