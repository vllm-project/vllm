# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StepMesh-based AFD connector implementation."""

import os
import subprocess
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.logger import init_logger

from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

import fserver_lib as ps  # isort: skip

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class StepMeshAFDConnector(AFDConnectorBase):
    """StepMesh-based implementation of AFD connector.

    This connector uses StepMesh parameter server for communication between
    attention workers and FFN servers.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        """Initialize StepMesh AFD connector.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFD configuration
        """
        self.afd_config = config.afd_config
        self.rank = rank
        self.local_rank = local_rank
        self.server_rank = self.afd_config.afd_server_rank
        self.num_recv_times = (self.afd_config.num_ffn_servers
                               if self.afd_config.afd_role == "attention" else
                               self.afd_config.num_attention_servers)
        parallel_config = config.parallel_config
        self.world_size = (parallel_config.tensor_parallel_size *
                           parallel_config.pipeline_parallel_size *
                           parallel_config.data_parallel_size)
        self._initialized = False
        self.num_stages = self.afd_config.num_afd_stages
        self.recv_counter = 0

        # Metadata tracking for new interface
        self._current_comm_handles = None
        self._current_metadata = None

        if self.afd_config.afd_role == "attention":
            self.events: deque = deque(maxlen=self.num_stages)
            self.max_num_tokens = (
                config.scheduler_config.max_num_batched_tokens //
                self.num_stages +
                config.scheduler_config.max_num_batched_tokens %
                self.num_stages)
            self.recv_buffer: list[list[torch.Tensor]] = [[
                torch.empty(
                    (
                        self.max_num_tokens,
                        config.model_config.hf_config.hidden_size,
                    ),
                    dtype=torch.bfloat16,
                    device=torch.device("cuda"),
                ).contiguous() for _ in range(self.num_recv_times)
            ] for _ in range(self.afd_config.num_afd_stages)]
            self.send_buffer: list[torch.Tensor] = [
                torch.empty(
                    (
                        self.max_num_tokens,
                        config.model_config.hf_config.hidden_size,
                    ),
                    dtype=torch.bfloat16,
                    device=torch.device("cuda"),
                ).contiguous() for _ in range(self.afd_config.num_afd_stages)
            ]
        else:
            self.max_num_tokens = (
                config.scheduler_config.max_num_batched_tokens //
                self.num_stages +
                config.scheduler_config.max_num_batched_tokens %
                self.num_stages) * self.num_recv_times
            self.ret_buffer = torch.empty(
                [
                    self.max_num_tokens,
                    config.model_config.hf_config.hidden_size,
                ],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ).contiguous()

        # StepMesh environment setup
        self._setup_stepmesh_env()

        if (self.afd_config.afd_role == "ffn"
                and self.afd_config.afd_server_rank == 0
                and self.local_rank == 0):
            self._start_scheduler_process()

    def _setup_stepmesh_env(self) -> None:
        """Setup StepMesh environment variables."""
        # Basic StepMesh configuration based on draft.diff
        if self.afd_config.afd_role == "attention":
            os.environ["DMLC_ROLE"] = "worker"
        elif self.afd_config.afd_role == "ffn":
            os.environ["DMLC_ROLE"] = "server"
        else:
            raise ValueError(f"Invalid AFD role: {self.afd_config.afd_role}")

        os.environ["DMLC_NUM_WORKER"] = str(
            self.afd_config.num_attention_servers)
        os.environ["DMLC_NUM_SERVER"] = str(self.afd_config.num_ffn_servers)

        os.environ["DMLC_ENABLE_RDMA"] = "ibverbs"
        os.environ["DMLC_INTERFACE"] = "auto"
        os.environ["STEPMESH_SPLIT_QP_LAG"] = os.environ.get(
            "STEPMESH_SPLIT_QP_LAG", "0")
        os.environ["STEPMESH_BIND_CPU_CORE"] = "1"

        os.environ["STEPMESH_GPU"] = os.environ.get("STEPMESH_GPU",
                                                    str(self.local_rank))

        os.environ["DMLC_PS_ROOT_PORT"] = str(self.afd_config.afd_port)
        os.environ["DMLC_PS_ROOT_URI"] = self.afd_config.afd_host
        os.environ["DMLC_NODE_HOST"] = str(self.afd_config.afd_host)
        os.environ["SCHEDULER_IP"] = str(self.afd_config.afd_host)

        os.environ["DMLC_NODE_RANK"] = str(self.afd_config.afd_server_rank)
        os.environ["DMLC_GROUP_SIZE"] = str(self.world_size)

        os.environ["PS_VERBOSE"] = os.environ.get("PS_VERBOSE", "2")

        logger.info(
            "StepMesh environment setup: role=%s, "
            "num_worker=%s, "
            "num_server=%s, "
            "port=%s, "
            "host=%s, "
            "node_rank=%s, "
            "gpu=%s, "
            "group_size=%s", os.environ.get('DMLC_ROLE'),
            os.environ.get('DMLC_NUM_WORKER'),
            os.environ.get('DMLC_NUM_SERVER'),
            os.environ.get('DMLC_PS_ROOT_PORT'),
            os.environ.get('DMLC_PS_ROOT_URI'),
            os.environ.get('DMLC_NODE_RANK'), os.environ.get('STEPMESH_GPU'),
            os.environ.get('DMLC_GROUP_SIZE'))

    def _start_scheduler_process(self) -> None:
        """Start scheduler process for FFN role.

        This method launches a separate subprocess to run the StepMesh scheduler
        when the current process is in FFN role.
        """
        try:
            logger.info("Starting scheduler subprocess for FFN role")
            # Use subprocess.Popen to start scheduler as a separate process
            self.scheduler_process = subprocess.Popen(
                [
                    "python",
                    "-c",
                    "import torch; import fserver_lib as ps; import os; "
                    'os.environ["DMLC_ROLE"] = "scheduler"; '
                    'os.environ["DMLC_INTERFACE"] = "brainpf_bond0"; '
                    "ps.init(); ps.stop()",
                ],
                env=os.environ.copy(),
            )
            logger.info("Scheduler subprocess started with PID: %s",
                        self.scheduler_process.pid)
        except Exception as e:
            logger.error("Failed to start scheduler subprocess: %s", e)
            raise RuntimeError(
                f"Failed to start scheduler subprocess: {e}") from e

    def init_afd_connector(self) -> None:
        """Initialize StepMesh connector."""
        if self._initialized:
            return
        try:
            logger.info("+++++Start init ps. %s", self.rank)
            ps.init()
            logger.info("----Finish init ps. %s", self.rank)

            self._initialized = True
            logger.info("StepMesh connector initialized successfully as %s",
                        os.environ.get('DMLC_ROLE'))

        except ImportError as e:
            raise ImportError(
                f"StepMesh is not available. Please install StepMesh to use "
                f"StepMesh AFD connector. Error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize StepMesh connector: {e}") from e

    @property
    def is_initialized(self) -> bool:
        """Check if the connector is initialized."""
        return self._initialized

    # AFD Communication Methods Implementation
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> Any:
        """Send attention output to FFN servers via StepMesh push_pull.

        Args:
            hidden_states: Attention output tensor
            metadata: AFD metadata containing layer_idx, stage_idx, seq_len info

        Returns:
            Any: Event handle for tracking this request
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        # Validate metadata consistency
        if not metadata.validate_tensor_shape(hidden_states.shape):
            raise ValueError("Tensor shape %s doesn't match metadata %s",
                             hidden_states.shape, metadata)

        if not metadata.is_single_sequence:
            raise ValueError("Attention side should have single sequence")

        seq_len = metadata.seq_lens[0]  # Single sequence for attention side
        stage_id = metadata.stage_idx

        # Create keys based on the pattern from your example
        node_rank_offset = int(self.rank * 1e6)
        recv_key = [stage_id + 1000]
        recv_buff = [t[:seq_len] for t in self.recv_buffer[stage_id]]
        send_buff = [self.send_buffer[stage_id][:seq_len]]

        if seq_len > self.max_num_tokens:
            raise ValueError("AFD seq_len[%s] exceeds max_num_tokens[%s]",
                             seq_len, self.max_num_tokens)

        send_buff[0].copy_(hidden_states[:seq_len])
        send_key = [stage_id + node_rank_offset]

        event = ps.push_pull(
            send_buff,
            send_key,
            recv_buff,
            recv_key,
        )
        self.events.append((event, metadata))

    def recv_attn_output(
        self,
        timeout_ms: Optional[float] = None,
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """Receive attention output from attention workers (FFN server side).

        Args:
            timeout_ms: Optional timeout in milliseconds

        Returns:
            tuple: (hidden_states, metadata) received from attention workers
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        try:
            # batches = self.signal.get_batch() # type: ignore
            batches = ps.get_batch()  # type: ignore

            # Extract tensors and build metadata
            recv_tensors = []
            seq_lens = []
            comm_handles = []

            for node_rank in range(self.num_recv_times):
                tensor = batches[node_rank][1][0]
                comm_id = batches[node_rank][0]

                recv_tensors.append(tensor)
                seq_lens.append(tensor.shape[0])
                comm_handles.append(comm_id)

            # Merge tensors
            merged_tensor = torch.cat(recv_tensors, dim=0)

            # Infer metadata from communication
            # TODO: Extract layer_idx and stage_idx from comm_id encoding
            inferred_metadata = AFDConnectorMetadata.create_ffn_metadata(
                layer_idx=-1,  # Extract from comm_id
                stage_idx=-1,  # Extract from comm_id
                seq_lens=seq_lens,
                dtype=merged_tensor.dtype,
                device=merged_tensor.device,
                request_id=f"ffn_batch_{time.time()}",
            )

            # Store handles for response
            self._current_comm_handles = comm_handles  # type: ignore
            self._current_metadata = inferred_metadata  # type: ignore

            return merged_tensor, inferred_metadata

        except Exception as e:
            logger.error("Failed to receive attention output: %s", e)
            raise RuntimeError(f"StepMesh recv_attn_output failed: {e}") from e

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """Send FFN computation result back to attention workers.

        Args:
            ffn_output: Computed FFN output
            metadata: AFD metadata containing seq_lens for s
                      plitting and routing info
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        self.ret_buffer[:ffn_output.shape[0]].copy_(ffn_output)

        try:
            # Use metadata.seq_lens for splitting
            split_indices = metadata.get_split_indices()
            if split_indices:
                split_outputs = torch.split(ffn_output,
                                            metadata.seq_lens,
                                            dim=0)
            else:
                split_outputs = [ffn_output]

            comm_handles = self._current_comm_handles
            ps.respond_vec(self.ret_buffer, split_outputs, comm_handles)

        except Exception as e:
            logger.error("Failed to send FFN output: %s", e)
            raise RuntimeError(f"StepMesh send_ffn_output failed: {e}") from e

    def recv_ffn_output(
        self,
        timeout_ms: Optional[float] = None,
    ) -> torch.Tensor:
        """Wait for FFN computation result from FFN servers.

        Args:
            handle: Event handle returned by send_attn_output

        Returns:
            torch.Tensor: FFN computation result
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        try:
            if len(self.events) > 0:
                event, metadata = self.events.popleft()
                ps.wait(event, timeout_ms=50000)
            # Get result from recv_buffer
            if metadata:
                stage_idx = metadata.stage_idx
                seq_len = metadata.seq_lens[
                    0]  # Single sequence for attention side
                if len(self.recv_buffer[stage_idx]) == 1:
                    return self.recv_buffer[stage_idx][0][:seq_len]
                else:
                    return torch.stack(
                        [t[:seq_len] for t in self.recv_buffer[stage_idx]],
                        dim=0,
                    ).sum(dim=0)
            else:
                raise ValueError("No metadata found for handle")

        except Exception as e:
            logger.error("Failed to wait for FFN output: %s", e)
            raise RuntimeError(f"StepMesh recv_ffn_output failed: {e}") from e

    def close(self) -> None:
        """Close the StepMesh connector and release resources."""
        if self._initialized:
            try:
                ps.finalize()
                self._initialized = False
                logger.info("StepMesh connector closed successfully")
            except Exception as e:
                logger.error("Failed to close StepMesh connector: %s", e)

        # Clean up scheduler subprocess if it exists
        if (hasattr(self, "scheduler_process")
                and self.scheduler_process is not None):
            try:
                if (self.scheduler_process.poll()
                        is None):  # Process is still running
                    logger.info("Terminating scheduler subprocess")
                    self.scheduler_process.terminate()
                    self.scheduler_process.wait(timeout=5)
                    logger.info("Scheduler subprocess terminated successfully")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Scheduler subprocess failed to terminate gracefully")
                self.scheduler_process.kill()
            except Exception as e:
                logger.error("Failed to terminate scheduler subprocess: %s", e)
