# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
import pickle

import torch
from torch.distributed.distributed_c10d import _get_default_group, _update_default_pg

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
    init_afd_process_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

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
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self._initialized: bool = False
        self._need_recv_metadata: bool = True
        self._tensor_metadata_list: dict[int, TensorMetadata] = {}
        self._current_afd_connector_metadata: AFDConnectorMetadata | None = None
        if getattr(self.config.model_config.hf_config, "text_config", None) is not None:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.text_config.num_hidden_layers
            )
        else:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.num_hidden_layers
            )

        self.recv_attn_output_counter: int = 0
        self.recv_ffn_output_counter: int = 0
        self.dp_metadata_list: dict[int, DPMetadata] = {}
        self.a2e_pynccl: PyNcclCommunicator | None = None
        self.e2a_pynccl: PyNcclCommunicator | None = None
        self.ffn_size: int = 0
        self.min_size: int = 0
        self.dst_list = []

    def close(self) -> None:
        """Close the connector and release resources."""
        # TODO: Implement proper resource clean up if needed.
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        logger.info("jcz init_afd_connector begin")
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(int, re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        self.world_rank = self.rank if role == "ffn" else self.rank + ffn_size
        self.ffn_size = ffn_size
        self.min_size = min(ffn_size, attn_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=(
                f"tcp://{self.config.afd_config.afd_host}"
                f":{self.config.afd_config.afd_port}"
            ),
            world_size=ffn_size + attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )
        logger.info(f"jcz afd_pg initialized world_rank:{self.world_rank}")
        # Construct rank lists for sub groups.
        # Each group contains one attention and one ffn rank.
        ffn_ranks = [i for i in range(ffn_size)]
        attn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        assert len(ffn_ranks) == len(attn_ranks), (
            "ffn_ranks and attn_ranks must have the same length"
        )
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                # ranks = [attn_ranks[i], ffn_ranks[i]]
                ranks = [ffn_ranks[i], attn_ranks[i]]
                sub_group_ranks.append(ranks)
            # Create two independent groups:
            # a2e_group: for attention -> expert/ffn communication (send_attn, recv_attn)
            # e2a_group: for expert/ffn -> attention communication (send_ffn, recv_ffn)
            # The communication domain (rank range) is the same, but different group_name
            # creates independent groups.
            logger.info("jcz before self.a2e_group")
            self.a2e_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend="nccl",
                group_name="a2e",
            )
            logger.info("jcz before self.e2a_group")
            self.e2a_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend="nccl",
                group_name="e2a",
            )
            logger.info("jcz before a2e_pynccl")
            self.a2e_pynccl = PyNcclCommunicator(
                group=self.a2e_group.cpu_group,
                device=self.local_rank,
            )
            logger.info("jcz before e2a_pynccl")
            self.e2a_pynccl = PyNcclCommunicator(
                group=self.e2a_group.cpu_group,
                device=self.local_rank,
            )
            logger.info("jcz after a2e_pynccl and e2a_pynccl")
        
        # All FFN and the first min_size Attention participate in p2p communication.
        # All FFN: world_rank in [0, ffn_size)
        # First min_size Attention: world_rank in [ffn_size, ffn_size + min_size)
        if self.is_vaild_rank_for_inequal_AF(self.world_rank):
            self.p2p_pg = init_afd_process_group(
                backend="nccl",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        # The first min_size Attention sends metadata to multiple FFNs (1-to-many mapping).
        # Each attn_i sends to all ffn_j where (j % min_size == i)
        if self.is_attn_top_min_size_rank(self.world_rank):
            local_attn_rank = self.world_rank - self.ffn_size
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size
        logger.info(
            f"[P2P] world_rank={self.world_rank}, p2p_rank={self.p2p_rank}, min_size={self.min_size}, "
            f"dst_list={self.dst_list}, p2p connector initialized"
        )

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def _build_tensor_metadata_list(
        self,
        tensor_metadata: TensorMetadata,
        connector_metadata: AFDConnectorMetadata,
    ) -> dict[int, TensorMetadata]:
        tensor_metadata_list = {}
        num_of_stages = connector_metadata.num_of_stages
        for idx in range(num_of_stages):
            if idx == 0:
                tensor_metadata_list[0] = tensor_metadata
            else:
                new_size = list(tensor_metadata.size)
                new_size[0] = connector_metadata.afd_tokens_lens[idx]
                tensor_metadata_list[idx] = TensorMetadata(
                    tensor_metadata.device,
                    tensor_metadata.dtype,
                    torch.Size(new_size),
                )
        return tensor_metadata_list

    def _send_metadata(
        self,
        metadata: AFDConnectorMetadata,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: GroupCoordinator,
    ) -> None:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"

        tensor_metadata = TensorMetadata(
            hidden_states.device.type, hidden_states.dtype, hidden_states.size()
        )
        metadata_tuple = (metadata, tensor_metadata)
        process_group.send_object(metadata_tuple, dst=dst)
        self._tensor_metadata_list = self._build_tensor_metadata_list(
            tensor_metadata, metadata
        )

    def _recv_metadata(
        self,
        src: int,
        process_group: GroupCoordinator,
    ) -> None:
        (self._current_afd_connector_metadata, tensor_metadata) = (
            process_group.recv_object(src=src)
        )
        self._tensor_metadata_list = self._build_tensor_metadata_list(
            tensor_metadata, self._current_afd_connector_metadata
        )
        if self.config.parallel_config.data_parallel_size > 1:
            logger.info(
                "jcz recv_metadata num_of_stages:{}".format(
                    self._current_afd_connector_metadata.num_of_stages
                )
            )
            for stage_idx in range(self._current_afd_connector_metadata.num_of_stages):
                num_tokens_per_ubatch = self._tensor_metadata_list[stage_idx].size[0]
                self.dp_metadata_list[stage_idx] = DPMetadata.make(
                    self.config.parallel_config,
                    num_tokens_per_ubatch,
                    torch.tensor(
                        [num_tokens_per_ubatch]
                        * self.config.parallel_config.data_parallel_size,
                        device="cpu",
                        dtype=torch.int32,
                    ),
                )
            logger.info(
                "jcz recv_metadata self.dp_metadata_list:{}".format(
                    self.dp_metadata_list
                )
            )

    def _send_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: GroupCoordinator,
    ) -> None:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []
        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"
        assert not hidden_states.is_cpu, "Hidden states must be on GPU"

        # Try to use PyNCCL first
        pynccl_comm = None
        if process_group == self.a2e_group:
            pynccl_comm = self.a2e_pynccl
        elif process_group == self.e2a_group:
            pynccl_comm = self.e2a_pynccl

        if pynccl_comm and not pynccl_comm.disabled:
            # PyNCCL uses rank in group
            logger.info(f"jcz send_hidden_states using pynccl dst:{dst}")
            pynccl_comm.send(hidden_states, dst)
        else:
            raise RuntimeError("PyNCCL communicator is required but not available.")

    def _recv_hidden_states(
        self,
        src: int,
        process_group: GroupCoordinator,
        tensor_metadata: TensorMetadata,
    ) -> torch.Tensor:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []
        assert src < process_group.world_size, f"Invalid src rank ({src})"

        hidden_states = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.dtype,
            device=tensor_metadata.device,
        )

        # Try to use PyNCCL first
        pynccl_comm = None
        if process_group == self.a2e_group:
            pynccl_comm = self.a2e_pynccl
        elif process_group == self.e2a_group:
            pynccl_comm = self.e2a_pynccl

        if pynccl_comm and not pynccl_comm.disabled:
            # PyNCCL uses rank in group
            logger.info(f"jcz recv_hidden_states using pynccl src:{src}")
            pynccl_comm.recv(hidden_states, src)
        else:
            raise RuntimeError("PyNCCL communicator is required but not available.")
        return hidden_states

    # -------------------------------------------------------------------------
    #                                attn -> ffn
    # -------------------------------------------------------------------------

    def send_attn_output(
        self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata
    ) -> None:
        """
        Called by ATTN side to send intermediate tensors
        generated by ATTN instances to FFN.
        """
        try:
            dst = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
            logger.info(f"jcz send_attn_output rank_in_group:{self.a2e_group.rank_in_group} dst:{dst} world_size:{self.a2e_group.world_size}")
            if metadata.layer_idx == 0 and metadata.stage_idx == 0:
                self._send_metadata(metadata, hidden_states, dst, self.a2e_group)
            self._current_afd_connector_metadata = metadata
            self._send_hidden_states(hidden_states, dst, self.a2e_group)
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_ffn_output(self) -> torch.Tensor:
        """
        Called by the ATTN side to receive MOE output intermediate tensors,
        possibly dispatching from the receiver to other GPUs.
        """
        src = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        stage_idx = (
            self.recv_ffn_output_counter
            % self._current_afd_connector_metadata.num_of_stages
        )
        hidden_states = self._recv_hidden_states(
            src,
            self.e2a_group,
            self._tensor_metadata_list[stage_idx],
        )
        self.recv_ffn_output_counter = (
            self.recv_ffn_output_counter + 1
        ) % self._current_afd_connector_metadata.num_of_stages
        return hidden_states

    # -------------------------------------------------------------------------
    #                                ffn -> attn
    # -------------------------------------------------------------------------

    def send_ffn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """
        Called by FFN side to send intermediate tensors generated by FFN
        instances back to the sender (should be the same GPU as source).
        """
        dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        self._send_hidden_states(hidden_states, dst, self.e2a_group)
        self.recv_attn_output_counter += 1
        if (
            self.recv_attn_output_counter
            % (
                self._current_afd_connector_metadata.num_of_stages
                * self.num_hidden_layers
            )
            == 0
        ):
            self._need_recv_metadata = True
            self.recv_attn_output_counter = 0

    def recv_attn_output(self) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """
        Called by the FFN side to receive intermediate tensors from ATTN.
        Handles receiving and possibly dispatching tensors.
        """
        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        logger.info(f"jcz recv_attn_output rank_in_group:{self.a2e_group.rank_in_group} src:{src} world_size:{self.a2e_group.world_size}")
        if self._need_recv_metadata:
            self._recv_metadata(src, self.a2e_group)
            self._need_recv_metadata = False

        stage_idx = (
            self.recv_attn_output_counter
            % self._current_afd_connector_metadata.num_of_stages
        )
        layer_idx = (
            self.recv_attn_output_counter
            // self._current_afd_connector_metadata.num_of_stages
        )
        hidden_states = self._recv_hidden_states(
            src,
            self.a2e_group,
            self._tensor_metadata_list[stage_idx],
        )
        self._current_afd_connector_metadata.layer_idx = layer_idx
        self._current_afd_connector_metadata.stage_idx = stage_idx
        return hidden_states, self._current_afd_connector_metadata

    def send_dp_metadata_list(self, data):
        for dst in self.dst_list:
            object_bytes = pickle.dumps(data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            object_tensor_gpu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="cuda")
            object_tensor_gpu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                        dtype=torch.long,
                                        device="cuda")
            logger.info(f"jcz send_dp_metadata_list dst:{dst} self.p2p_rank:{self.p2p_rank}")
            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_gpu, dst=dst, group=self.p2p_pg)
    
    def recv_dp_metadata_list(self):
        src = self.p2p_rank % self.min_size + self.ffn_size
        logger.info(f"jcz recv_dp_metadata_list src:{src} self.p2p_rank:{self.p2p_rank}")

        size_tensor = torch.empty(1, dtype=torch.long, device="cuda")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor_gpu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
        rank_object = torch.distributed.recv(object_tensor_gpu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_gpu.cpu()
        data = pickle.loads(object_tensor_cpu.numpy().tobytes())
        return data

    def is_vaild_rank_for_inequal_AF(self,rank):
        # Only support ffn rank < attn rank
        return ((rank >= self.ffn_size and rank < self.ffn_size + self.min_size) or rank < self.ffn_size)

    def is_attn_top_min_size_rank(self,rank):
        # Only support ffn rank < attn rank
        return (rank >= self.ffn_size and rank < self.ffn_size + self.min_size)