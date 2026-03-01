# Standard
import json
import os
import re
from dataclasses import dataclass

import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.backend.backend import Backend
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 134217728 # 128 MB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 128 MB


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e
        self.config = MooncakeStoreConfig.load_from_env()
        self.store = MooncakeDistributedStore()
        self.rank = parallel_config.rank

        local_hostname = get_ip()
        transfer_engine = global_te.get_transfer_engine(local_hostname, self.config.metadata_server, self.config.protocol, device_name=None)
        self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
        ret = self.store.setup(
                self.local_seg,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
                transfer_engine.get_engine(),
            )
        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(msg)
            raise RuntimeError(msg)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        try:
            config = ReplicateConfig()
            config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = True
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to put key {keys},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {keys},error:{e}")

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes, True)
            for value in res:
                if value < 0:
                    logger.error(f"Failed to get key {keys}, res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {keys}, error:{e}")


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "neuron"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
