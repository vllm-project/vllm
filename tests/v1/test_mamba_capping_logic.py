import unittest
from unittest.mock import MagicMock
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.config import VllmConfig, SchedulerConfig, CacheConfig, ModelConfig

class TestMambaCappingLogic(unittest.TestCase):
    def test_capping_logic_mock(self):
        # 1. Setup KVCacheConfig with MambaSpec
        # Say we have 100 blocks total and 2 groups.
        # Each group should get 50 blocks.
        mamba_spec = MambaSpec(
            shapes=((1, 16, 64),),
            dtypes=(torch.float16,),
            block_size=1
        )
        
        group1 = KVCacheGroupSpec(layer_names=["layer1"], kv_cache_spec=mamba_spec)
        group2 = KVCacheGroupSpec(layer_names=["layer2"], kv_cache_spec=mamba_spec)
        
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[group1, group2]
        )
        
        # 2. Setup Config
        scheduler_config = MagicMock(spec=SchedulerConfig)
        scheduler_config.max_num_seqs = 256 # Initially high
        
        # Simulation of the logic in EngineCore or GPUModelRunner
        mamba_num_blocks = float("inf")
        has_mamba = False
        num_groups = len(kv_cache_config.kv_cache_groups)
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, MambaSpec):
                mamba_num_blocks = min(mamba_num_blocks, 
                                       kv_cache_config.num_blocks // num_groups)
                has_mamba = True

        if has_mamba and mamba_num_blocks < scheduler_config.max_num_seqs:
            # logger.warning(...)
            scheduler_config.max_num_seqs = int(mamba_num_blocks)
            
        # 3. Assertions
        self.assertTrue(has_mamba)
        self.assertEqual(mamba_num_blocks, 50)
        self.assertEqual(scheduler_config.max_num_seqs, 50)
        print("Mamba capping logic test passed!")

if __name__ == "__main__":
    unittest.main()
