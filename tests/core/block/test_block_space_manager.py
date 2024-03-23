import random
import pytest
from typing import Optional, List
import random
from unittest.mock import MagicMock
import math

from vllm.core.block.block_space_manager import BlockSpaceManager
#from vllm.core.block.interfaces import NaiveBlockAllocator, NaiveBlock, BlockAllocator, Block
#from vllm.block2 import RefCounter
#from vllm.block2 import PrefixCachingBlock, PrefixCachingBlockAllocator

@pytest.mark.parametrize("block_size", [16])
def test_can_allocate(block_size: int):
    
    block_manager = BlockSpaceManager(
        block_size=block_size,
        num_gpu_blocks=1024,
        num_cpu_blocks=1024,
    )


