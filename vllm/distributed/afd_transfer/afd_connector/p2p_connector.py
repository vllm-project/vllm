# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
from typing import Any, Optional

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group
from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata
from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group, _split_tensor_dict, TensorMetadata, GroupCoordinator
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger
from vllm.config import VllmConfig
logger = init_logger(__name__)

class DefaultProcessGroupSwitcher:

    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)


class P2PAFDConnector(AFDConnectorBase):
    def __init__(self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self._need_recv_metadata: bool = True
        self._tensor_metadata_list: dict[int, TensorMetadata] = dict()
        self._current_afd_connector_metadata: AFDConnectorMetadata | None = None
        self.num_hidden_layers = self.config.model_config.hf_config.num_hidden_layers
        self.recv_attn_output_counter: int = 0
        self.recv_ffn_output_counter: int = 0
        self._a2e_tag_base: int = 0
        self._e2a_tag_base: int = 10000

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        #ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        #attn_ranks = [i for i in range(attn_size)]
        world_rank = self.rank if role == "attention" else self.rank + attn_size

        logger.info(
            f"world_size = {ffn_size + attn_size}, world_rank = {world_rank}")
        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.config.afd_config.afd_host}:{self.config.afd_config.afd_port}",
            world_size=ffn_size + attn_size,
            rank=world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )
        ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        attn_ranks = [i for i in range(attn_size)]
        logger.info(f"jcz init_afd_connector ffn_ranks:{ffn_ranks} attn_ranks:{attn_ranks}")
        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            # Create two independent groups:
            # a2e_group: for attention -> expert/ffn communication (send_attn, recv_attn)
            # e2a_group: for expert/ffn -> attention communication (send_ffn, recv_ffn)
            # The communication domain (rank range) is the same, but different group_name
            # creates independent groups
            logger.info(f"jcz begin init_afd_connector a2e_group")
            self.a2e_group = init_model_parallel_group(sub_group_ranks,
                                                 self.local_rank,
                                                 backend="nccl",
                                                 group_name="a2e")
            logger.info(f"jcz init_afd_connector a2e_group:{self.a2e_group} rank_in_group:{self.a2e_group.rank_in_group}")
            self.e2a_group = init_model_parallel_group(sub_group_ranks,
                                                 self.local_rank,
                                                 backend="nccl",
                                                 group_name="e2a")
            logger.info(f"jcz init_afd_connector e2a_group:{self.e2a_group} rank_in_group:{self.a2e_group.rank_in_group}")

        logger.info("p2p connector initialized")

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def _build_tensor_metadata_list(self,
                                    tensor_metadata: TensorMetadata,
                                    connector_metadata: AFDConnectorMetadata) -> None:
        tensor_metadata_list = {}
        num_of_stages = connector_metadata.num_of_stages
        for idx in range(num_of_stages):
            if idx == 0:
                tensor_metadata_list[0] = tensor_metadata
            else:
                new_size = list(tensor_metadata.size)
                new_size[0] = connector_metadata.afd_tokens_lens[idx]
                tensor_metadata_list[idx] = TensorMetadata(tensor_metadata.device, tensor_metadata.dtype, torch.Size(new_size))
        return tensor_metadata_list

    def _send_metadata(
        self,
        metadata: AFDConnectorMetadata,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: GroupCoordinator) -> None:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
        
        tensor_metadata = TensorMetadata(hidden_states.device.type, hidden_states.dtype, hidden_states.size())
        metadata_tuple = (metadata, tensor_metadata)
        process_group.send_object(metadata_tuple, dst=dst)
        self._tensor_metadata_list = self._build_tensor_metadata_list(tensor_metadata, metadata)
        logger.info(f"jcz _send_metadata tensor_metadata_list:{self._tensor_metadata_list}")
    
    def _recv_metadata(
        self,
        src: int,
        process_group: GroupCoordinator
    ) -> None:
        (self._current_afd_connector_metadata, tensor_metadata) = process_group.recv_object(src=src)
        num_of_stages = self._current_afd_connector_metadata.num_of_stages
        logger.info(f"jcz _recv_metadata num_of_stages:{num_of_stages} "
                    f"afd_tokens_lens:{self._current_afd_connector_metadata.afd_tokens_lens}")
        # assert num_of_stages == len(self._current_afd_connector_metadata.afd_tokens_lens), \
        #     f"num_of_stages:{num_of_stages} != afd_tokens_lens:{self._current_afd_connector_metadata.afd_tokens_lens}"
        
        self._tensor_metadata_list = self._build_tensor_metadata_list(tensor_metadata, self._current_afd_connector_metadata)
        logger.info(f"jcz _recv_metadata tensor_metadata_list:{self._tensor_metadata_list}")

    def _send_hidden_states(
        self, 
        hidden_states: torch.Tensor,
        dst: int,
        process_group: GroupCoordinator,
        tag: int = 0
    ) -> None:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
        assert not hidden_states.is_cpu, "Hidden states must be on GPU"
        # torch.distributed.isend(
        #     hidden_states, dst=process_group.ranks[dst], group=process_group.device_group
        # )
        torch.distributed.send(hidden_states, dst=process_group.ranks[dst], group=process_group.device_group)
    
    def _recv_hidden_states(
        self,
        src: int,
        stage_idx: int,
        process_group: GroupCoordinator,
        tensor_metadata: TensorMetadata,
        tag: int = 0
    ) -> tuple[torch.Tensor, list]:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []
        
        assert src < process_group.world_size, f"Invalid src rank ({src})"

        hidden_states = torch.empty(tensor_metadata.size,
                                    dtype=tensor_metadata.dtype,
                                    device=tensor_metadata.device)
        logger.info(f"jcz _recv_hidden_states stage_idx:{stage_idx} size:{tensor_metadata.size} "
                    f"dtype:{tensor_metadata.dtype} device:{tensor_metadata.device}")
        
        # work_list = []
        # work = torch.distributed.irecv(
        #     hidden_states, src=process_group.ranks[src], group=process_group.device_group
        # )
        # work_list.append(work)
        
        # return hidden_states, work_list
        torch.distributed.recv(hidden_states, src=process_group.ranks[src], group=process_group.device_group)
        return hidden_states, []


    def send_attn_output(
        self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata
    ):
        """
        This method will be called by the ATTN side.


        * To send the intermediate tensors generated by ATTN instances to FFN.
        """

        try:
            dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size
            if metadata.layer_idx == 0 and metadata.stage_idx == 0:
                logger.info(f"jcz send_attn_output begin sending metadata")
                self._send_metadata(metadata, hidden_states, dst, self.a2e_group)
                logger.info(f"jcz send_attn_output end sending metadata")
            self._current_afd_connector_metadata = metadata
            # torch.cuda.current_stream().synchronize()
            a2e_tag = self._a2e_tag_base + metadata.num_of_stages * metadata.layer_idx + metadata.stage_idx
            logger.info(f"jcz send_attn_output a2e_tag:{a2e_tag} hidden_states shape:{hidden_states.shape} "
                        f"layer_idx:{metadata.layer_idx} stage_idx:{metadata.stage_idx} num_of_stages:{metadata.num_of_stages}")
            self._send_hidden_states(hidden_states, dst, self.a2e_group, a2e_tag)
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_attn_output(self) -> torch.Tensor:
        """
        This method will be called by the FFN side.


        * To receive the intermediate tensors from ATTN.
        * And (Maybe) dispatch them from the receiver to other GPUs.
        """

        # Use a2e_group for attention -> expert/ffn communication
        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        if self._need_recv_metadata:
            self._recv_metadata(src, self.a2e_group)
            logger.info(f"jcz self._current_afd_connector_metadata.stage_idx:{self._current_afd_connector_metadata.stage_idx} "
                        f"self._current_afd_connector_metadata.num_of_stages:{self._current_afd_connector_metadata.num_of_stages}")
            logger.info("jcz set _need_recv_metadata to False")
            self._need_recv_metadata = False

        # Use async receive for tensor_dict
        stage_idx = self.recv_attn_output_counter % self._current_afd_connector_metadata.num_of_stages
        layer_idx = self.recv_attn_output_counter // self._current_afd_connector_metadata.num_of_stages
        a2e_tag = self._a2e_tag_base + self.recv_attn_output_counter
        hidden_states, work_list = self._recv_hidden_states(src,
                                                            stage_idx,
                                                            self.a2e_group,
                                                            self._tensor_metadata_list[stage_idx],
                                                            a2e_tag)
        logger.info(f"jcz recv_attn_output a2e_tag:{a2e_tag} hidden_states shape:{hidden_states.shape} "
                    f"layer_idx:{layer_idx} stage_idx:{stage_idx} num_of_stages:{self._current_afd_connector_metadata.num_of_stages}")
        self._current_afd_connector_metadata.recv_handle_list = work_list
        self._current_afd_connector_metadata.layer_idx = self.recv_attn_output_counter // self._current_afd_connector_metadata.num_of_stages
        return hidden_states, self._current_afd_connector_metadata

    # -------------------------------------------------------------------------
    #                                attn <- ffn
    # -------------------------------------------------------------------------
    def send_ffn_output(
        self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata
    ):
        """
        This method will be called by the FFN side.


        * To send the intermediate tensors generated by FFN instances back to
            the sender (this should be the same GPU as it comes from)
        """
        # Use async send instead of sync send
        # Use e2a_group for expert/ffn -> attention communication
        # torch.cuda.current_stream().synchronize()
        dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size

        self._send_hidden_states(hidden_states, dst, self.e2a_group)

        self.recv_attn_output_counter += 1
        if self.recv_attn_output_counter % \
            (self._current_afd_connector_metadata.num_of_stages * self.num_hidden_layers) == 0:
            self._need_recv_metadata = True
            self.recv_attn_output_counter = 0
            logger.info(f"jcz send_ffn_output recv_attn_output_counter: {self.recv_attn_output_counter} detected, "
                        f"self._current_afd_connector_metadata.num_of_stages:{self._current_afd_connector_metadata.num_of_stages} "
                        f"self.num_hidden_layers:{self.num_hidden_layers} "
                        f"reset _need_recv_metadata to True")

    def recv_ffn_output(self) -> torch.Tensor:
        """
        This method will be called by the ATTN side.


        * To receive the MOE output intermediate tensors.
        * And (Maybe) dispatch them from the receiver to other GPUs.
            (this should be the same GPU as it comes from)
        """
        # Use e2a_group for expert/ffn -> attention communication
        src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size
        stage_idx = self.recv_ffn_output_counter % self._current_afd_connector_metadata.num_of_stages
        logger.info(f"jcz recv_ffn_output self.recv_ffn_output_counter:{self.recv_ffn_output_counter} "
                    f"stage_idx:{stage_idx} num_of_stages:{self._current_afd_connector_metadata.num_of_stages}")
        hidden_states, work_list = self._recv_hidden_states(src,
                                                            stage_idx,
                                                            self.e2a_group,
                                                            self._tensor_metadata_list[stage_idx])
        self._current_afd_connector_metadata.recv_handle_list = work_list
        self.recv_ffn_output_counter = (self.recv_ffn_output_counter + 1) % self._current_afd_connector_metadata.num_of_stages
        logger.info(f"jcz recv_ffn_output src:{src} stage_idx:{stage_idx} "
                    f"hidden_states shape:{hidden_states.shape}")
        return hidden_states, self._current_afd_connector_metadata
