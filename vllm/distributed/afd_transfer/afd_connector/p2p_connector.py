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
from vllm.utils.torch_utils import direct_register_custom_op
from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

logger = init_logger(__name__)

# -------------------------------------------------------------------------
# Custom Ops Registration for P2P Communication
# -------------------------------------------------------------------------

# Global registry to map integer IDs to PyNcclCommunicator objects
# because we cannot pass complex Python objects to custom ops.
_AFD_COMMUNICATORS: dict[int, PyNcclCommunicator] = {}
_AFD_COMM_ID_COUNTER = 0

# Side stream for shadow eager execution during graph capture
_AFD_SIDE_STREAM: torch.cuda.Stream | None = None

def _get_side_stream() -> torch.cuda.Stream:
    global _AFD_SIDE_STREAM
    if _AFD_SIDE_STREAM is None:
        _AFD_SIDE_STREAM = torch.cuda.Stream()
    return _AFD_SIDE_STREAM


def _register_comm(comm: PyNcclCommunicator) -> int:
    global _AFD_COMM_ID_COUNTER
    comm_id = _AFD_COMM_ID_COUNTER
    _AFD_COMMUNICATORS[comm_id] = comm
    _AFD_COMM_ID_COUNTER += 1
    return comm_id


def _unregister_comm(comm_id: int) -> None:
    _AFD_COMMUNICATORS.pop(comm_id, None)

# --- Send Op ---

def afd_p2p_send_impl(tensor: torch.Tensor, dst: int, comm_id: int) -> None:
    comm = _AFD_COMMUNICATORS.get(comm_id)
    if comm is None:
        raise RuntimeError(f"Communicator with ID {comm_id} not found/registered.")
    print("jcz before afd_p2p_send_impl", flush=True)
    comm.send(tensor, dst)
    print("jcz after afd_p2p_send_impl", flush=True)

def afd_p2p_send_fake(tensor: torch.Tensor, dst: int, comm_id: int) -> None:
    print("jcz afd_p2p_send_fake", flush=True)
    return None

direct_register_custom_op(
    op_name="afd_p2p_send",
    op_func=afd_p2p_send_impl,
    mutates_args=["tensor"],
    fake_impl=afd_p2p_send_fake,
)

# --- Recv Op ---

def afd_p2p_recv_impl(
    out: torch.Tensor,
    src: int,
    comm_id: int,
) -> None:
    comm = _AFD_COMMUNICATORS.get(comm_id)
    if comm is None:
        raise RuntimeError(f"Communicator with ID {comm_id} not found/registered.")
    print("jcz before afd_p2p_recv_impl", flush=True)
    comm.recv(out, src)
    print("jcz after afd_p2p_recv_impl", flush=True)


def afd_p2p_recv_fake(
    out: torch.Tensor,
    src: int,
    comm_id: int,
) -> None:
    return None

direct_register_custom_op(
    op_name="afd_p2p_recv",
    op_func=afd_p2p_recv_impl,
    mutates_args=["out"],
    fake_impl=afd_p2p_recv_fake,
)

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
        self._tensor_metadata_list: dict[int, TensorMetadata] = {}
        if getattr(self.config.model_config.hf_config, "text_config", None) is not None:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.text_config.num_hidden_layers
            )
        else:
            self.num_hidden_layers: int = (
                self.config.model_config.hf_config.num_hidden_layers
            )

        self.a2e_pynccl: PyNcclCommunicator | None = None
        self.e2a_pynccl: PyNcclCommunicator | None = None
        self.a2e_comm_id: int | None = None
        self.e2a_comm_id: int | None = None
        self.ffn_size: int = 0
        self.min_size: int = 0
        self.dst_list = []

    def close(self) -> None:
        """Close the connector and release resources."""
        if self.a2e_comm_id is not None:
            _unregister_comm(self.a2e_comm_id)
            self.a2e_comm_id = None
        if self.e2a_comm_id is not None:
            _unregister_comm(self.e2a_comm_id)
            self.e2a_comm_id = None

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
            self.a2e_comm_id = _register_comm(self.a2e_pynccl)
            logger.info("jcz before e2a_pynccl")
            self.e2a_pynccl = PyNcclCommunicator(
                group=self.e2a_group.cpu_group,
                device=self.local_rank,
            )
            self.e2a_comm_id = _register_comm(self.e2a_pynccl)
            logger.info("jcz after a2e_pynccl and e2a_pynccl")
        
        # All FFN and the first min_size Attention participate in p2p communication.
        # All FFN: world_rank in [0, ffn_size)
        # First min_size Attention: world_rank in [ffn_size, ffn_size + min_size)
        if self.is_vaild_rank_for_inequal_AF(self.world_rank):
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
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
        comm_id = None
        if process_group == self.a2e_group:
            comm_id = self.a2e_comm_id
        elif process_group == self.e2a_group:
            comm_id = self.e2a_comm_id

        if comm_id is not None:
            # PyNCCL uses rank in group
            torch.ops.vllm.afd_p2p_send(hidden_states, dst, comm_id)
        else:
            raise RuntimeError("PyNCCL communicator is required but not available.")

    def _recv_hidden_states(
        self,
        src: int,
        process_group: GroupCoordinator,
        tensor_metadata: TensorMetadata,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []
        assert src < process_group.world_size, f"Invalid src rank ({src})"

        # Try to use PyNCCL first
        comm_id = None
        if process_group == self.a2e_group:
            comm_id = self.a2e_comm_id
        elif process_group == self.e2a_group:
            comm_id = self.e2a_comm_id

        if comm_id is not None:
            # PyNCCL uses rank in group
            # Use ref_tensor to capture dynamic shapes (e.g. batch size) if provided
            size = list(tensor_metadata.size)
            if ref_tensor is not None:
                # Assume dimension 0 is the dynamic batch/seq_len dimension
                size[0] = ref_tensor.shape[0]

            hidden_states = torch.empty(
                tuple(size),
                dtype=tensor_metadata.dtype,
                device=tensor_metadata.device,
            )
            torch.ops.vllm.afd_p2p_recv(hidden_states, src, comm_id)
        else:
            raise RuntimeError("PyNCCL communicator is required but not available.")
        return hidden_states
    
    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict[int, DPMetadata],
    ) -> None:
        """Update the connector state based on the received DPMetadata list.
        This replaces the explicit metadata communication step.
        """
        self.dp_metadata_list = dp_metadata_list
        num_of_stages = len(dp_metadata_list)
        
        # Build tensor metadata list for each stage
        self._tensor_metadata_list = {}
        
        for stage_idx in range(num_of_stages):
            dp_metadata = dp_metadata_list[stage_idx]
            # In vLLM DPMetadata, num_tokens_across_dp_cpu holds the number of tokens across all dp ranks
            # For current rank, we need to get the number of tokens using parallel_config.data_parallel_rank
            dp_rank = self.config.parallel_config.data_parallel_rank
            num_tokens = dp_metadata.num_tokens_across_dp_cpu[dp_rank].item()
            
            self._tensor_metadata_list[stage_idx] = TensorMetadata(
                torch.device(f"cuda:{self.local_rank}"),
                # TODO(jcz): use dtype from dp_metadata
                self.config.model_config.dtype,
                torch.Size([num_tokens, self.config.model_config.hf_config.hidden_size]),
            )

    # -------------------------------------------------------------------------
    #                                attn -> ffn
    # -------------------------------------------------------------------------

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """
        Called by ATTN side to send intermediate tensors
        generated by ATTN instances to FFN.
        """
        try:
            dst = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
            self._send_hidden_states(hidden_states, dst, self.a2e_group)
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_ffn_output(self, ref_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """
        Called by the ATTN side to receive MOE output intermediate tensors,
        possibly dispatching from the receiver to other GPUs.
        """
        ubatch_idx = get_forward_context().afd_metadata.afd_stage_idx
        src = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        hidden_states = self._recv_hidden_states(
            src,
            self.e2a_group,
            self._tensor_metadata_list[ubatch_idx],
            ref_tensor=ref_tensor
        )
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

    def recv_attn_output(
        self, ubatch_idx: int = 0
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """
        Called by the FFN side to receive intermediate tensors from ATTN.
        Handles receiving and possibly dispatching tensors.
        """
        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        hidden_states = self._recv_hidden_states(
            src,
            self.a2e_group,
            self._tensor_metadata_list[ubatch_idx],
        )

        # TODO(jcz): remove this after.
        from types import SimpleNamespace
        metadata = SimpleNamespace(
            stage_idx=ubatch_idx,
            recv_handle_list=None,
        )
        return hidden_states, metadata

    def send_dp_metadata_list(self, data):
        self.update_state_from_dp_metadata(data)
        for dst in self.dst_list:
            object_bytes = pickle.dumps(data)
            # Use CPU tensor for Gloo backend
            object_tensor = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
            size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)
            
            logger.info(f"jcz send_dp_metadata_list dst:{dst} self.p2p_rank:{self.p2p_rank}")
            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor, dst=dst, group=self.p2p_pg)
    
    def recv_dp_metadata_list(self):
        src = self.p2p_rank % self.min_size + self.ffn_size
        logger.info(f"jcz recv_dp_metadata_list src:{src} self.p2p_rank:{self.p2p_rank}")

        # Use CPU tensor for Gloo backend
        size_tensor = torch.empty(1, dtype=torch.long)
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        
        object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
        rank_object = torch.distributed.recv(object_tensor, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        data = pickle.loads(object_tensor.numpy().tobytes())
        return data

    def is_vaild_rank_for_inequal_AF(self,rank):
        # Only support ffn rank < attn rank
        return ((rank >= self.ffn_size and rank < self.ffn_size + self.min_size) or rank < self.ffn_size)

    def is_attn_top_min_size_rank(self,rank):
        # Only support ffn rank < attn rank
        return (rank >= self.ffn_size and rank < self.ffn_size + self.min_size)