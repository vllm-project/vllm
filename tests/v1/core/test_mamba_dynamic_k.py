# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


def make_manager(num_speculative_blocks: int = 15) -> MambaManager:
    spec = MambaSpec(
        block_size=4,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=num_speculative_blocks,
        mamba_cache_mode="none",
    )
    pool = BlockPool(num_gpu_blocks=256, enable_caching=False, hash_block_size=4)
    return MambaManager(
        spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )


def blocks_needed(manager: MambaManager, override: int | None) -> int:
    return manager.get_num_blocks_to_allocate(
        request_id="req",
        num_tokens=4,
        new_computed_blocks=[],
        total_computed_tokens=4,
        num_local_computed_tokens=0,
        num_tokens_main_model=4,
        num_spec_override=override,
    )


def test_dynamic_override_sizes_mamba_admission():
    manager = make_manager()
    assert blocks_needed(manager, None) == 16
    assert blocks_needed(manager, 3) == 4


def test_zero_override_removes_speculative_admission():
    manager = make_manager()
    assert blocks_needed(manager, 0) == 1
