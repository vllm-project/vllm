# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import zmq
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger
from vllm.utils.network_utils import join_host_port, make_zmq_path, split_host_port

logger = init_logger(__name__)
NONE_INT = -150886311


@dataclass
class MooncakeTransferEngineConfig:
    prefill_url: str
    decode_url: str
    metadata_backend: str | None
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            prefill_url=config.get("prefill_url"),
            decode_url=config.get("decode_url"),
            metadata_backend=config.get("metadata_backend", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


class MooncakeTransferEngine:
    """Handles the transfer of data using mooncake_vllm_adaptor and ZeroMQ."""

    def __init__(self, kv_rank: int, local_rank: int):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        self.engine = TransferEngine()
        self.local_rank = local_rank

        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise
        prefill_host, base_prefill_port = split_host_port(self.config.prefill_url)
        decode_host, base_decode_port = split_host_port(self.config.decode_url)

        # Avoid ports conflict when running prefill and decode on the same node
        if prefill_host == decode_host and base_prefill_port == base_decode_port:
            base_decode_port = base_decode_port + 100

        prefill_port = base_prefill_port + self.local_rank
        decode_port = base_decode_port + self.local_rank
        self.prefill_url = join_host_port(prefill_host, prefill_port)
        self.decode_url = join_host_port(decode_host, decode_port)

        self.initialize(
            self.prefill_url if kv_rank == 0 else self.decode_url,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
            self.config.metadata_backend,
        )

        self.remote_url = self.decode_url if kv_rank == 0 else self.prefill_url

        # Initialize ZeroMQ context and sockets
        self.context = zmq.Context()  # type: ignore[attr-defined]
        self.sender_socket = self.context.socket(zmq.constants.PUSH)
        self.receiver_socket = self.context.socket(zmq.constants.PULL)
        self.sender_ack = self.context.socket(zmq.constants.PULL)
        self.receiver_ack = self.context.socket(zmq.constants.PUSH)

        self.buffer_cleaner = ThreadPoolExecutor(max_workers=1)
        self._setup_metadata_sockets(
            kv_rank, prefill_host, base_prefill_port, decode_host, base_decode_port
        )

    def _setup_metadata_sockets(
        self, kv_rank: int, p_host: str, p_port: int, d_host: str, d_port: int
    ) -> None:
        """Set up ZeroMQ sockets for sending and receiving data."""
        # Offsets < 8 are left for initialization in case tp and pp are enabled
        p_rank_offset = p_port + 8 + self.local_rank * 2
        d_rank_offset = d_port + 8 + self.local_rank * 2
        if kv_rank == 0:
            self.sender_socket.bind(make_zmq_path("tcp", p_host, p_rank_offset + 1))
            self.receiver_socket.connect(
                make_zmq_path("tcp", d_host, d_rank_offset + 1)
            )
            self.sender_ack.connect(make_zmq_path("tcp", d_host, d_rank_offset + 2))
            self.receiver_ack.bind(make_zmq_path("tcp", p_host, p_rank_offset + 2))
        else:
            self.receiver_socket.connect(
                make_zmq_path("tcp", p_host, p_rank_offset + 1)
            )
            self.sender_socket.bind(make_zmq_path("tcp", d_host, d_rank_offset + 1))
            self.receiver_ack.bind(make_zmq_path("tcp", d_host, d_rank_offset + 2))
            self.sender_ack.connect(make_zmq_path("tcp", p_host, p_rank_offset + 2))

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
        metadata_backend: str | None,
    ) -> None:
        """Initialize the mooncake instance."""
        if metadata_backend is None:
            self.engine.initialize(
                local_hostname, metadata_server, protocol, device_name
            )
        else:
            supported_backend = ["etcd", "redis"]
            metadata_backend = metadata_backend.lower()
            if metadata_backend not in supported_backend:
                raise ValueError(
                    "Mooncake Configuration error. `metadata_backend`"
                    f" should be one of {supported_backend}."
                )

            self.engine.initialize_ext(
                local_hostname, metadata_server, protocol, device_name, metadata_backend
            )

    def allocate_managed_buffer(self, length: int) -> int:
        """Allocate a managed buffer of the specified length."""
        ret = self.engine.allocate_managed_buffer(length)
        if ret <= 0:
            logger.error("Allocation Return Error")
            raise Exception("Allocation Return Error")
        return ret

    def free_managed_buffer(self, buffer: int, length: int) -> int:
        """Free a previously allocated managed buffer."""
        return self.engine.free_managed_buffer(buffer, length)

    def transfer_sync(self, buffer: int, peer_buffer_address: int, length: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transfer_sync_read(
            self.remote_url, buffer, peer_buffer_address, length
        )
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def write_bytes_to_buffer(self, buffer: int, user_data: bytes, length: int) -> int:
        """Write bytes to the allocated buffer."""
        return self.engine.write_bytes_to_buffer(buffer, user_data, length)

    def read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        """Read bytes from the allocated buffer."""
        return self.engine.read_bytes_from_buffer(buffer, length)

    def wait_for_ack(self, src_ptr: int, length: int) -> None:
        """Asynchronously wait for ACK from the receiver."""
        ack = self.sender_ack.recv()
        if ack != b"ACK":
            logger.error("Failed to receive ACK from the receiver")

        self.free_managed_buffer(src_ptr, length)

    def send_bytes(self, user_data: bytes) -> None:
        """Send bytes to the remote process."""
        length = len(user_data)
        src_ptr = self.allocate_managed_buffer(length)
        self.write_bytes_to_buffer(src_ptr, user_data, length)
        self.sender_socket.send_multipart(
            [struct.pack("!Q", src_ptr), struct.pack("!Q", length)]
        )
        self.buffer_cleaner.submit(self.wait_for_ack, src_ptr, length)

    def recv_bytes(self) -> bytes:
        """Receive bytes from the remote process."""
        data = self.receiver_socket.recv_multipart()
        src_ptr = struct.unpack("!Q", data[0])[0]
        length = struct.unpack("!Q", data[1])[0]
        dst_ptr = self.allocate_managed_buffer(length)
        self.transfer_sync(dst_ptr, src_ptr, length)
        ret = self.read_bytes_from_buffer(dst_ptr, length)

        # Buffer cleanup
        self.receiver_ack.send(b"ACK")
        self.free_managed_buffer(dst_ptr, length)

        return ret


class MooncakePipe(KVPipeBase):
    """MooncakeTransferEngine based Pipe implementation."""

    def __init__(
        self, local_rank: int, config: KVTransferConfig, device: str | None = None
    ):
        """Initialize the mooncake pipe and set related parameters."""
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        assert self.kv_rank is not None
        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
        else:
            self.device = self._select_device(device)

        self.transfer_engine = MooncakeTransferEngine(self.kv_rank, self.local_rank)
        self.transport_thread: ThreadPoolExecutor | None = None
        self.none_tensor = torch.tensor([NONE_INT], device=self.device)

    def _select_device(self, device: str) -> torch.device:
        """Select available device (CUDA or CPU)."""
        logger.info("Selecting device: %s", device)
        if device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def tensor_hash(self, tensor: torch.Tensor) -> int:
        """Calculate the hash value of the tensor."""
        return hash(tensor.data_ptr())

    def _send_impl(self, tensor: torch.Tensor) -> None:
        """Implement the tensor sending logic using safetensors."""
        self.transfer_engine.send_bytes(safetensors_save({"tensor": tensor}))

    def _recv_impl(self) -> torch.Tensor:
        """Implement the tensor receiving logic using safetensors."""
        data = self.transfer_engine.recv_bytes()
        return safetensors_load(data)["tensor"].to(self.device)

    def send_tensor(self, tensor: torch.Tensor | None) -> None:
        """Send tensor to the target process."""
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)
        tensor = tensor if tensor is not None else self.none_tensor
        assert len(tensor.shape) > 0
        self.transport_thread.submit(self._send_impl, tensor)

    def recv_tensor(self) -> torch.Tensor | None:
        """Receive tensor from other processes."""
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)
        tensor = self.transport_thread.submit(self._recv_impl).result()
        if tensor.numel() == 1 and tensor.item() == NONE_INT:
            return None
        else:
            return tensor

    def close(self) -> None:
        """Cleanup logic when closing the pipe."""
        self.transfer_engine.sender_socket.close()
        self.transfer_engine.receiver_socket.close()
        self.transfer_engine.sender_ack.close()
        self.transfer_engine.receiver_ack.close()
        self.transfer_engine.context.term()  # Terminate the ZMQ context
        logger.info("Closed the transfer engine and cleaned up resources.")
