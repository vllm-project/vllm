import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.config import CacheConfig
from vllm.core.evictor_v1 import EvictionPolicy
from vllm.logger import init_logger
from vllm.utils import Device, size_to_bytes
from vllm.worker.swap.interface import (DeviceStatus, SwapClientManagerBase,
                                        SwapDeviceClient, SwapDeviceManager,
                                        SwapSpaceManagerBase)

logger = init_logger(__name__)


class SwapSpaceManager(SwapSpaceManagerBase):
    """
    Stateful in the scheduler
    """

    def __init__(self) -> None:
        self.device_table: OrderedDict[int, SwapDeviceManager] = OrderedDict()
        # Can have holes when a device is detached
        self.current_device_id: int = 0

        # For swapin and swap out from to to disk
        self.disk_block_tables: Dict[int, Dict[int, BlockTable]] = {}

    def parse_and_add_swap_device(self, cache_config: CacheConfig):
        if not cache_config.enable_disk_swap:
            return
        assert cache_config.num_cpu_blocks is not None, \
            "Disk swap only works with CPU DRAM"
        block_size_bytes = int(cache_config.swap_space_bytes //
                               cache_config.num_cpu_blocks)
        for k, v in cache_config.disk_swap_config.items():
            # Does not fallthrough
            if isinstance(k, str) and k.lower() == "localdisk":
                from vllm.worker.swap.local_disk import (LocalDisk,
                                                         LocalDiskManager)
                assert isinstance(v, dict)
                size_bytes = (size_to_bytes(str(v["size"]))
                              if "size" in v else LocalDisk.DEFAULT_SIZE)
                path = (str(v["path"])
                        if "path" in v else LocalDisk.DEFAULT_PATH)
                # TODO: Skip the policy
                eviction_policy = EvictionPolicy.LRU
                self.add_swap_device(
                    LocalDiskManager(block_size=cache_config.block_size,
                                     num_blocks=int(size_bytes //
                                                    block_size_bytes),
                                     device_id=self.current_device_id,
                                     path=path,
                                     eviction_policy=eviction_policy))
            else:
                logger.warning("Unsupported swap device %s", str(k))
        return

    def add_swap_device(self, swap_device: SwapDeviceManager) -> DeviceStatus:
        status: DeviceStatus = swap_device.attach_device()
        if status == DeviceStatus.OK:
            assert self.current_device_id not in self.device_table
            self.device_table[self.current_device_id] = \
                swap_device
            self.current_device_id += 1
        return status

    def remove_swap_device(self,
                           swap_device: SwapDeviceManager) -> DeviceStatus:
        # NOTE: Blocked implementation
        dev_id = swap_device.dev_id
        if dev_id not in self.device_table:
            logger.warning("Remove unregistered swap device")
            return DeviceStatus.ERROR
        status = swap_device.detach_device()
        if status == DeviceStatus.OK:
            del self.device_table[dev_id]
        return status

    def get_num_free_blocks_for_all(self) -> int:
        num_free_blocks = 0
        for id, dev in self.device_table.items():
            num_free_blocks += dev.get_num_free_blocks()
        return num_free_blocks

    def _find_allocate(self,
                       block_hash: int,
                       num_hashed_tokens: int = 0) -> int:
        # TODO: Performance not ideal. 2 round trip
        for id, dev in self.device_table.items():
            if (dev.contains_block(block_hash, num_hashed_tokens)
                    or dev.can_allocate_block()):
                return dev.dev_id
        return -1

    def contains_block(self,
                       block_hash: int,
                       num_hashed_tokens: int = 0) -> bool:
        for _, dev in self.device_table.items():
            if dev.contains_block(block_hash, num_hashed_tokens):
                return True
        return False

    def can_allocate(self,
                     block_hash: int,
                     num_hashed_tokens: int = 0) -> bool:
        return self._find_allocate(block_hash, num_hashed_tokens) != -1

    def allocate(
            self,
            block_hash: int,
            num_hashed_tokens: int = 0,
            swap_dev_id: Optional[int] = None) -> Optional[PhysicalTokenBlock]:
        # TODO: Don't sacrifice performance for logic. 2 round trip
        # right now
        if swap_dev_id is not None:
            self.device_table[swap_dev_id].allocate(block_hash,
                                                    num_hashed_tokens)

        # TODO: Move to selector for more devices
        dev_id = self._find_allocate(block_hash, num_hashed_tokens)
        if dev_id < 0:
            logger.warning("Disk full. Cannot allocate block!")
            return None
        return self.device_table[dev_id].allocate(block_hash,
                                                  num_hashed_tokens)

    def free(self, block: PhysicalTokenBlock) -> None:
        assert block.device_id >= Device.SWAP
        dev_id: int = block.device_id - Device.SWAP
        assert dev_id in self.device_table, "Free a block from detached device"
        self.device_table[dev_id].free(block)

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        swap_device.update_hash(block_hash, block)

    def update_block_tables(self, seq_id: int,
                            disk_block_table: Dict[int, BlockTable]):
        for dev_id, block_table in disk_block_table.items():
            self.disk_block_tables.setdefault(dev_id, {}).setdefault(
                seq_id, BlockTable()).extend(block_table)

    def free_block_tables(self):
        # Free all the block tables. Invoked at the end of the scheduler
        for dev_id, block_tables in self.disk_block_tables.items():
            assert dev_id in self.device_table
            swap_device = self.device_table[dev_id]
            for _, block_table in block_tables.items():
                access_time = time.time()
                for block in block_table:
                    block.last_accessed = access_time
                    # TODO: Aggregate
                    swap_device.free(block)
            block_table.clear()
        self.disk_block_tables.clear()

    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int, block_id: int):
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        swap_device.add_rmap(block, seq_id, block_id)

    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_id: int):
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        swap_device.remove_rmap(block, seq_id, block_id)

    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        swap_device.remove_rmap_all(block)

    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        return swap_device.get_rmap(block)

    def n_rmap(self, block: PhysicalTokenBlock):
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        return swap_device.n_rmap(block)

    def move_swappable(self, block: PhysicalTokenBlock):
        dev_id = block.device_id - Device.SWAP
        assert dev_id in self.device_table, \
            "Update a block from detached device"
        swap_device = self.device_table[dev_id]
        return swap_device.move_swappable(block)


class SwapClientManager(SwapClientManagerBase):

    def __init__(self) -> None:
        self.device_table: OrderedDict[int, SwapDeviceClient] = OrderedDict()
        # Can have holes when a device is detached
        self.current_device_id: int = 0

    def parse_and_add_swap_device(self, cache_config: CacheConfig):
        if not cache_config.enable_disk_swap:
            return
        assert cache_config.num_cpu_blocks is not None, \
            "Disk swap only works with CPU DRAM"
        block_size_bytes = int(cache_config.swap_space_bytes //
                               cache_config.num_cpu_blocks)
        for k, v in cache_config.disk_swap_config.items():
            # Does not fallthrough
            if isinstance(k, str) and k.lower() == "localdisk":
                from vllm.worker.swap.local_disk import (LocalDisk,
                                                         LocalDiskClient)
                assert isinstance(v, dict)
                size_bytes = (size_to_bytes(str(v["size"]))
                              if "size" in v else LocalDisk.DEFAULT_SIZE)
                path = (str(v["path"])
                        if "path" in v else LocalDisk.DEFAULT_PATH)
                self.add_swap_device(
                    LocalDiskClient(block_size=cache_config.block_size,
                                    num_blocks=int(size_bytes //
                                                   block_size_bytes),
                                    device_id=self.current_device_id,
                                    path=path))
            else:
                logger.warning("Unsupported swap device %s", str(k))
        return

    def add_swap_device(self, swap_device: SwapDeviceClient) -> DeviceStatus:
        status: DeviceStatus = swap_device.attach_device()
        if status == DeviceStatus.OK:
            assert self.current_device_id not in self.device_table
            self.device_table[self.current_device_id] = \
                swap_device
            self.current_device_id += 1
        return status

    def remove_swap_device(self,
                           swap_device: SwapDeviceClient) -> DeviceStatus:
        # NOTE: Blocked implementation
        dev_id = swap_device.dev_id
        if dev_id not in self.device_table:
            logger.warning("Remove unregistered swap device")
            return DeviceStatus.ERROR
        status = swap_device.detach_device()
        if status == DeviceStatus.OK:
            del self.device_table[dev_id]
        return status

    def read_blocks(self, caches: List[torch.Tensor],
                    blocks_to_swap_in_from_dev: Dict[int, List[Tuple[int,
                                                                     int]]],
                    layers: List[int], head_range: Tuple[int, int]):
        # NOTE: Right now only support read to GPU
        # TODO: we will add another internal func for CPU
        for dev_id, blocks_to_swap_in in blocks_to_swap_in_from_dev.items():
            # Should be adjusted
            assert dev_id in self.device_table, "Swap blocks to nonexist device"
            self.device_table[dev_id].read_block_content(
                caches, blocks_to_swap_in, layers, head_range)

    def write_blocks(self, caches: List[torch.Tensor],
                     blocks_to_swap_out_to_dev: Dict[int, List[Tuple[int,
                                                                     int]]],
                     layers: List[int], head_range: Tuple[int, int]):
        # NOTE: Right now only support write from CPU
        # TODO: we will add another internal func for GPU
        for dev_id, blocks_to_swap_out in blocks_to_swap_out_to_dev.items():
            # Should be adjusted
            assert dev_id in self.device_table, \
                "Swap blocks from nonexist device"
            self.device_table[dev_id].write_block_content(
                caches, blocks_to_swap_out, layers, head_range)
