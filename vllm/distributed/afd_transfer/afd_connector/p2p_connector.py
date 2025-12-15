# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta

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
        self.num_hidden_layers: int = (
            self.config.model_config.hf_config.num_hidden_layers
        )
        self.recv_attn_output_counter: int = 0
        self.recv_ffn_output_counter: int = 0
        self.dp_metadata_list: dict[int, DPMetadata] = {}

    def close(self) -> None:
        """Close the connector and release resources."""
        # TODO: Implement proper resource clean up if needed.
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        attn_size, ffn_size = map(int, re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        world_rank = self.rank if role == "attention" else self.rank + attn_size
        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=(
                f"tcp://{self.config.afd_config.afd_host}"
                f":{self.config.afd_config.afd_port}"
            ),
            world_size=ffn_size + attn_size,
            rank=world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )

        # Construct rank lists for sub groups.
        # Each group contains one attention and one ffn rank.
        ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        attn_ranks = [i for i in range(attn_size)]
        assert len(ffn_ranks) == len(attn_ranks), (
            "ffn_ranks and attn_ranks must have the same length"
        )
        default_pg_switcher = DefaultProcessGroupSwitcher(_get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = [attn_ranks[i], ffn_ranks[i]]
                sub_group_ranks.append(ranks)
            # Create two independent groups:
            # a2e_group: for attention -> expert/ffn communication (send_attn, recv_attn)
            # e2a_group: for expert/ffn -> attention communication (send_ffn, recv_ffn)
            # The communication domain (rank range) is the same, but different group_name
            # creates independent groups.
            self.a2e_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend="nccl",
                group_name="a2e",
            )
            self.e2a_group = init_model_parallel_group(
                sub_group_ranks,
                self.local_rank,
                backend="nccl",
                group_name="e2a",
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
            logger.info("jcz recv_metadata num_of_stages:{}".format(self._current_afd_connector_metadata.num_of_stages))
            for stage_idx in range(self._current_afd_connector_metadata.num_of_stages):
                num_tokens_per_ubatch = self._tensor_metadata_list[stage_idx].size[0]
                self.dp_metadata_list[stage_idx] = DPMetadata.make(
                    self.config.parallel_config,
                    num_tokens_per_ubatch,
                    torch.tensor([num_tokens_per_ubatch] * self.config.parallel_config.data_parallel_size,
                                device="cpu", dtype=torch.int32),
                )
            logger.info("jcz recv_metadata self.dp_metadata_list:{}".format(self.dp_metadata_list))

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
        torch.distributed.send(
            hidden_states,
            dst=process_group.ranks[dst],
            group=process_group.device_group,
        )

    def _recv_hidden_states(
        self,
        src: int,
        process_group: GroupCoordinator,
        tensor_metadata: TensorMetadata,
    ) -> tuple[torch.Tensor, list]:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []
        assert src < process_group.world_size, f"Invalid src rank ({src})"

        hidden_states = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.dtype,
            device=tensor_metadata.device,
        )
        torch.distributed.recv(
            hidden_states,
            src=process_group.ranks[src],
            group=process_group.device_group,
        )
        return hidden_states, []

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
            dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size
            if metadata.layer_idx == 0 and metadata.stage_idx == 0:
                self._send_metadata(metadata, hidden_states, dst, self.a2e_group)
            self._current_afd_connector_metadata = metadata
            self._send_hidden_states(hidden_states, dst, self.a2e_group)
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_ffn_output(self) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """
        Called by the ATTN side to receive MOE output intermediate tensors,
        possibly dispatching from the receiver to other GPUs.
        """
        src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size
        stage_idx = (
            self.recv_ffn_output_counter
            % self._current_afd_connector_metadata.num_of_stages
        )
        hidden_states, work_list = self._recv_hidden_states(
            src,
            self.e2a_group,
            self._tensor_metadata_list[stage_idx],
        )
        self._current_afd_connector_metadata.recv_handle_list = work_list
        self.recv_ffn_output_counter = (
            self.recv_ffn_output_counter + 1
        ) % self._current_afd_connector_metadata.num_of_stages
        return hidden_states, self._current_afd_connector_metadata

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
        hidden_states, work_list = self._recv_hidden_states(
            src,
            self.a2e_group,
            self._tensor_metadata_list[stage_idx],
        )
        self._current_afd_connector_metadata.recv_handle_list = work_list
        self._current_afd_connector_metadata.layer_idx = layer_idx
        self._current_afd_connector_metadata.stage_idx = stage_idx
        return hidden_states, self._current_afd_connector_metadata
