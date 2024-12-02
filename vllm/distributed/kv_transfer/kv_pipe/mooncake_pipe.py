import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import mooncake_vllm_adaptor as mva
import torch
import zmq

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)
NONE_INT = -150886311


@dataclass
class MooncakeTransferEngineConfig:
    prefill_url: str
    decode_url: str
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeTransferEngineConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            prefill_url=config.get("prefill_url"),
            decode_url=config.get("decode_url"),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeTransferEngineConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeTransferEngineConfig.from_file(config_file_path)


class MooncakeTransferEngine:
    """Handles the transfer of data using mooncake_vllm_adaptor and ZeroMQ."""

    def __init__(self, rank_in_group: int):
        self.engine = mva.mooncake_vllm_adaptor()

        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        self.initialize(
            self.config.prefill_url if rank_in_group == 0 else
            self.config.decode_url, self.config.metadata_server,
            self.config.protocol, self.config.device_name)

        self.remote_url = (self.config.decode_url
                           if rank_in_group == 0 else self.config.prefill_url)

        # Initialize ZeroMQ context and sockets
        self.context = zmq.Context()  # type: ignore[attr-defined]
        self.sender_socket = self.context.socket(zmq.constants.PUSH)
        self.receiver_socket = self.context.socket(zmq.constants.PULL)
        self.sender_ack = self.context.socket(zmq.constants.PULL)
        self.receiver_ack = self.context.socket(zmq.constants.PUSH)

        host, port = self.remote_url.split(':')
        self.buffer_cleaner = ThreadPoolExecutor(max_workers=1)
        self._setup_sockets(rank_in_group, host, port)

    def _setup_sockets(self, rank_in_group: int, host: str, port: str) -> None:
        """Set up ZeroMQ sockets for sending and receiving data."""
        if rank_in_group == 0:
            self.sender_socket.bind(f"tcp://*:{int(port) + 1}")
            self.receiver_socket.connect(f"tcp://{host}:{int(port) + 2}")
            self.sender_ack.connect(f"tcp://{host}:{int(port) + 3}")
            self.receiver_ack.bind(f"tcp://*:{int(port) + 4}")
        else:
            self.receiver_socket.connect(f"tcp://{host}:{int(port) + 1}")
            self.sender_socket.bind(f"tcp://*:{int(port) + 2}")
            self.receiver_ack.bind(f"tcp://*:{int(port) + 3}")
            self.sender_ack.connect(f"tcp://{host}:{int(port) + 4}")

    def initialize(self, local_hostname: str, metadata_server: str,
                   protocol: str, device_name: str) -> None:
        """Initialize the mooncake instance."""
        self.engine.initialize(local_hostname, metadata_server, protocol,
                               device_name)

    def allocate_managed_buffer(self, length: int) -> int:
        """Allocate a managed buffer of the specified length."""
        ret = self.engine.allocateManagedBuffer(length)
        if ret <= 0:
            logger.error("Allocation Return Error")
            raise Exception("Allocation Return Error")
        return ret

    def free_managed_buffer(self, buffer: int, length: int) -> int:
        """Free a previously allocated managed buffer."""
        return self.engine.freeManagedBuffer(buffer, length)

    def transfer_sync(self, buffer: int, peer_buffer_address: int,
                      length: int) -> int:
        """Synchronously transfer data to the specified address."""
        ret = self.engine.transferSync(self.remote_url, buffer,
                                       peer_buffer_address, length)
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def write_bytes_to_buffer(self, buffer: int, user_data: bytes,
                              length: int) -> int:
        """Write bytes to the allocated buffer."""
        return self.engine.writeBytesToBuffer(buffer, user_data, length)

    def read_bytes_from_buffer(self, buffer: int, length: int) -> bytes:
        """Read bytes from the allocated buffer."""
        return self.engine.readBytesFromBuffer(buffer, length)

    def wait_for_ack(self, src_ptr: int, length: int) -> None:
        """Asynchronously wait for ACK from the receiver."""
        ack = self.sender_ack.recv_pyobj()
        if ack != b'ACK':
            logger.error("Failed to receive ACK from the receiver")

        self.free_managed_buffer(src_ptr, length)

    def send_bytes(self, user_data: bytes) -> None:
        """Send bytes to the remote process."""
        length = len(user_data)
        src_ptr = self.allocate_managed_buffer(length)
        self.write_bytes_to_buffer(src_ptr, user_data, length)
        self.sender_socket.send_pyobj((src_ptr, length))
        self.buffer_cleaner.submit(self.wait_for_ack, src_ptr, length)

    def recv_bytes(self) -> bytes:
        """Receive bytes from the remote process."""
        src_ptr, length = self.receiver_socket.recv_pyobj()
        dst_ptr = self.allocate_managed_buffer(length)
        self.transfer_sync(dst_ptr, src_ptr, length)
        ret = self.read_bytes_from_buffer(dst_ptr, length)

        # Buffer cleanup
        self.receiver_ack.send_pyobj(b'ACK')
        self.free_managed_buffer(dst_ptr, length)

        return ret


class MooncakePipe(KVPipeBase):
    """MooncakeTransferEngine based Pipe implementation."""

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None):
        """Initialize the mooncake pipe and set related parameters."""
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
        else:
            self.device = self._select_device(device)

        self.transfer_engine = MooncakeTransferEngine(self.kv_rank)
        self.transport_thread: Optional[ThreadPoolExecutor] = None
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
        """Implement the tensor sending logic."""
        value_bytes = pickle.dumps(tensor)
        self.transfer_engine.send_bytes(value_bytes)

    def _recv_impl(self) -> torch.Tensor:
        """Implement the tensor receiving logic."""
        data = self.transfer_engine.recv_bytes()
        return pickle.loads(data)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """Send tensor to the target process."""
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)
        tensor = tensor if tensor is not None else self.none_tensor
        assert (len(tensor.shape) > 0)
        self.transport_thread.submit(self._send_impl, tensor)

    def recv_tensor(self) -> Optional[torch.Tensor]:
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
        self.transfer_engine.context.term()  # Terminate the ZMQ context
        logger.info("Closed the transfer engine and cleaned up resources.")
