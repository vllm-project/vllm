from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from safetensors.torch import safe_open, save_file

from vllm.block import PhysicalTokenBlock
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.logger import init_logger
from vllm.utils import Device
from vllm.worker.swap.interface import (DeviceStatus, SwapDeviceBase,
                                        SwapDeviceClient, SwapDeviceManager)

logger = init_logger(__name__)


class LocalDisk(SwapDeviceBase):
    """
    The dummiest implementation of a local disk 
    """
    _GB = 1 << 30
    DEFAULT_SIZE: int = 16 * _GB
    DEFAULT_PATH: str = "/tmp/kv_cache"
    DEFAULT_POLICY: EvictionPolicy = EvictionPolicy.LRU

    def __init__(self,
                 block_size: int,
                 num_blocks: int,
                 device_id: int,
                 path: str = "/tmp/kv_cache"):
        self.device_id = device_id
        self.path = path + "/"
        self.block_size = block_size
        self.num_blocks = num_blocks

    def attach_device(self) -> DeviceStatus:
        Path(self.path).mkdir(parents=True, exist_ok=True)
        return DeviceStatus.OK

    @property
    def dev_id(self) -> int:
        return self.device_id

    @property
    def is_attached(self):
        return Path(self.path).is_dir()

    def detach_device(self) -> DeviceStatus:
        # Do nothing for now
        return DeviceStatus.OK


class LocalDiskClient(LocalDisk, SwapDeviceClient):
    """
    The stateless part in the worker
    """

    def __init__(self,
                 block_size: int,
                 num_blocks: int,
                 device_id: int,
                 path: str = "/tmp/kv_cache"):
        super().__init__(block_size, num_blocks, device_id, path)

    # Stateless part. Each worker should have one
    def read_block_content(self, caches: List[torch.Tensor],
                           blocks_to_swap_in: List[Tuple[int, int]],
                           layers: List[int], head_range: Tuple[int, int]):
        n_layers = len(caches)
        for layer in layers:
            assert layer < n_layers
            # TODO: Check whether we perform a copy or pass by ref here
            self.read_block_content_one_layer(caches[layer], blocks_to_swap_in,
                                              layer, head_range)

    def write_block_content(self, caches: List[torch.Tensor],
                            blocks_to_swap_in: List[Tuple[int, int]],
                            layers: List[int], head_range: Tuple[int, int]):
        n_layers = len(caches)
        for layer in layers:
            assert layer < n_layers
            # TODO: Check whether we perform a copy or pass by ref here
            self.write_block_content_one_layer(caches[layer],
                                               blocks_to_swap_in, layer,
                                               head_range)

    def _read_block_content_layers(self, cache: torch.Tensor,
                                   blocks_to_swap_in: List[Tuple[int, int]],
                                   layer_range: Tuple[int, int],
                                   head_range: Tuple[int, int]) -> None:
        # TODO: Support generic target device
        device = cache.get_device()

        for swap_in in blocks_to_swap_in:
            disk_block_number = swap_in[0]
            kv_path_name: str = (
                self.path +
                f"{layer_range[0]}_{layer_range[1]}_{disk_block_number}_{head_range[0]}_{head_range[1]}.pt"
            )
            device_str = f"cuda:{device}" if device >= 0 else "cpu"
            #logger.debug("Local disk read from %s", kv_path_name)
            with safe_open(
                    kv_path_name,
                    framework="pt",
                    device=device_str,
            ) as f:
                k: torch.Tensor = f.get_tensor('k')
                v: torch.Tensor = f.get_tensor('v')
                cache[0][swap_in[1]] = k
                cache[1][swap_in[1]] = v

    def read_block_content_one_layer(self, cache: torch.Tensor,
                                     blocks_to_swap_in: List[Tuple[int, int]],
                                     layer: int,
                                     head_range: Tuple[int, int]) -> None:
        # TODO: Support generic target device
        device = cache.get_device()

        for swap_in in blocks_to_swap_in:
            disk_block_number = swap_in[0]
            kv_path_name: str = (
                self.path +
                f"{layer}_{disk_block_number}_{head_range[0]}_{head_range[1]}.pt"
            )
            device_str = f"cuda:{device}" if device >= 0 else "cpu"
            #logger.debug("Local disk read from %s", kv_path_name)
            with safe_open(
                    kv_path_name,
                    framework="pt",
                    device=device_str,
            ) as f:
                k: torch.Tensor = f.get_tensor('k')
                v: torch.Tensor = f.get_tensor('v')
                cache[0][swap_in[1]] = k
                cache[1][swap_in[1]] = v

    def _write_block_content_layers(self, cache: torch.Tensor,
                                    blocks_to_swap_out: List[Tuple[int, int]],
                                    layer_range: Tuple[int, int],
                                    head_range: Tuple[int, int]) -> None:
        # TODO: Support generic target device

        for swap_out in blocks_to_swap_out:
            disk_block_number = swap_out[1]
            kv_path_name: str = (
                self.path +
                f"{layer_range[0]}_{layer_range[1]}_{disk_block_number}_{head_range[0]}_{head_range[1]}.pt"
            )
            #logger.debug("Local disk write to %s", kv_path_name)
            save_file(
                {
                    'k': cache[0][swap_out[0]].contiguous(),
                    'v': cache[1][swap_out[0]].contiguous(),
                }, kv_path_name)

    def write_block_content_one_layer(self, cache: torch.Tensor,
                                      blocks_to_swap_out: List[Tuple[int,
                                                                     int]],
                                      layer: int,
                                      head_range: Tuple[int, int]) -> None:
        # TODO: Support generic target device

        for swap_out in blocks_to_swap_out:
            disk_block_number = swap_out[1]
            kv_path_name: str = (
                self.path +
                f"{layer}_{disk_block_number}_{head_range[0]}_{head_range[1]}.pt"
            )
            #logger.debug("Local disk write to %s", kv_path_name)
            save_file(
                {
                    'k': cache[0][swap_out[0]].contiguous(),
                    'v': cache[1][swap_out[0]].contiguous(),
                }, kv_path_name)


class LocalDiskManager(LocalDisk, SwapDeviceManager):
    """
    The stateful part in the scheduler
    """

    def __init__(self,
                 block_size: int,
                 num_blocks: int,
                 device_id: int,
                 path: str = "/tmp/kv_cache",
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        super().__init__(block_size, num_blocks, device_id, path)

        self.current_num_blocks = 0
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

        self.evictor: Evictor = make_evictor(eviction_policy)

        self.swapper: Evictor = make_evictor(eviction_policy)
        # Reverse mapping: Block -> Seq ID and nth block in the sequence
        self.rmap: Dict[PhysicalTokenBlock, Set[Tuple[int, int]]] = {}

    def can_allocate_block(self) -> bool:
        return self.get_num_free_blocks() > 0

    def allocate_block(self, block_hash: int,
                       num_hashed_tokens: int) -> PhysicalTokenBlock:
        if self.current_num_blocks == self.num_blocks:
            block = self.evictor.evict()
            block.prev_block_hash = block.block_hash
            block.block_hash = block_hash
            block.prev_num_hashed_tokens = block.num_hashed_tokens
            block.num_hashed_tokens = num_hashed_tokens
            return block
        block = PhysicalTokenBlock(device=Device.SWAP,
                                   block_number=self.current_num_blocks,
                                   block_size=self.block_size,
                                   block_hash=block_hash,
                                   num_hashed_tokens=num_hashed_tokens,
                                   device_id=self.device_id +
                                   Device.SWAP)  # GPU CPU first
        self.current_num_blocks += 1
        return block

    def allocate(self,
                 block_hash: int,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        if block_hash in self.evictor:
            assert block_hash not in self.cached_blocks
            block = self.evictor.remove(block_hash)
            assert block.ref_count == 0
            self.cached_blocks[block_hash] = block
            block.ref_count += 1
            assert block.block_hash == block_hash
            return block
        if block_hash not in self.cached_blocks:
            self.cached_blocks[block_hash] = self.allocate_block(
                block_hash, num_hashed_tokens)
            block = self.cached_blocks[block_hash]
        else:
            block = self.cached_blocks[block_hash]
            block.is_evicted = False
        assert block.block_hash == block_hash
        block.ref_count += 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            assert block.block_hash not in self.evictor
            self.evictor.add(block)

            # Remove the block from the cached_blocks
            del self.cached_blocks[block.block_hash]

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self.current_num_blocks +
                self.evictor.num_blocks)

    def get_num_total_blocks(self) -> int:
        return self.num_blocks

    def contains_block(self, block_hash: int, num_hashed_tokens: int) -> bool:
        return block_hash in self.cached_blocks or block_hash in self.evictor

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        # Update the hash of block and the cached_blocks dictionary.
        assert not self.contains_block(block_hash, block.num_hashed_tokens)
        old_hash = block.block_hash
        block.block_hash = block_hash
        del self.cached_blocks[old_hash]
        self.cached_blocks[block_hash] = block

    def add_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                 block_id: int) -> None:
        # This rmap will be used when we swap out a mapped CC (only CC)
        self.rmap.setdefault(block, set()).add((seq_id, block_id))

    def remove_rmap(self, block: PhysicalTokenBlock, seq_id: int,
                    block_id: int) -> None:
        assert block in self.rmap
        self.rmap[block].remove((seq_id, block_id))
        if self.rmap[block] == set():
            del self.rmap[block]

    def remove_rmap_all(self, block: PhysicalTokenBlock) -> None:
        assert block in self.rmap
        del self.rmap[block]

    def get_rmap(self,
                 block: PhysicalTokenBlock) -> Optional[Set[Tuple[int, int]]]:
        # Return a ref
        return self.rmap[block] if block in self.rmap else None

    def n_rmap(self, block: PhysicalTokenBlock) -> int:
        return len(self.rmap[block]) if block in self.rmap else 0

    def move_swappable(self, block: PhysicalTokenBlock) -> None:
        # Do nothing for the ref count since the seq still owns it
        if block.ref_count == 0:
            raise ValueError(
                f"Cannot move free block {block} to swappable list."
                "They should go to the evictor!")
        # Block should not reside in the evictor
        assert block.block_hash not in self.evictor
        n_rmaps: int = len(self.rmap)
        # Can only have CC requests and Normal requests. Both of which
        # takes refs
        # TODO: Unify them in a general block manager system
        # TODO: take linux mm and implement based on it
        assert n_rmaps <= block.ref_count
        if n_rmaps == block.ref_count and block.block_hash not in self.swapper:
            # Only First time CCed request
            self.swapper.add(block)

        # They should still reside in cached_blocks
        # When they are popped out from the swapper they will be removed
        # from the HT
        assert block.block_hash in self.cached_blocks
