# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.config.cache import CacheConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder

logger = init_logger(__name__)


class MinimalEagleProposer:
    
    def __init__(self):
        # M-RoPE related settings
        self.uses_mrope = True
        self.max_model_len = 8192
        self.block_size = 16
        
        # Tree structure
        self.tree_choices = [(0,), (0, 0), (0, 1), (1,)]
        self.cu_drafts_per_level = [1, 3, 4]
        self.child_drafts_per_level = [1, 2, 1] 
        
        # Device-related tensors
        self.input_ids = torch.zeros(1000, dtype=torch.int32)
        self.positions = torch.zeros((3, 1000), dtype=torch.int64)
        self.hidden_states = torch.zeros((1000, 4096), dtype=torch.float16)
        self.arange = torch.arange(10, dtype=torch.int32)
        
        # Tree offsets
        max_batch_size = 8
        self.tree_draft_pos_offsets = torch.arange(
            1, len(self.tree_choices) + 1, dtype=torch.int32
        ).repeat(max_batch_size, 1)
        
        # Mock components
        self.vllm_config = MockVllmConfig()
        self.model = MockModel()
        self.runner = MockRunner()
        self.attn_layer_names = ['layer1']
        self.use_cuda_graph = False

    def test_mrope_position_logic(self, batch_size, positions):
        """Test the core logic of M-RoPE position handling"""
        
        print(f"Input positions shape: {positions.shape}")
        print(f"Input positions: {positions}")
        
        # Validate input
        assert positions.dim() == 2, f"M-RoPE positions should be 2D, got {positions.dim()}D"
        assert positions.shape[0] == 3, f"M-RoPE positions should have 3 dims, got {positions.shape[0]}"
        assert positions.shape[1] == batch_size, f"Batch size mismatch: {positions.shape[1]} vs {batch_size}"
        
        total_num_drafts = self.cu_drafts_per_level[0]
        tree_depth = len(self.cu_drafts_per_level)
        
        # Test position updates for each level
        for level in range(tree_depth - 1):
            print(f"Testing level {level}")
            
            # M-RoPE position update
            draft_positions = positions + (level + 1)
            print(f"Level {level} draft_positions: {draft_positions}")
            
            # Check if exceeds maximum length
            exceeds_max_model_len = (positions[0] + total_num_drafts) >= self.max_model_len
            print(f"Level {level} exceeds_max_len: {exceeds_max_model_len}")
            
            # Position clamping
            draft_positions = torch.where(
                exceeds_max_model_len.unsqueeze(0),
                0,
                draft_positions,
            ).view(3, batch_size, -1)
            
            print(f"Level {level} clamped positions shape: {draft_positions.shape}")
            
            # Validate shape correctness
            assert draft_positions.shape == (3, batch_size, 1), \
                f"Draft positions shape error at level {level}: {draft_positions.shape}"
            
        print("M-RoPE position logic test passed")
        return True


class MockVllmConfig:
    def pad_for_cudagraph(self, x):
        return x


class MockModel:
    def compute_logits(self, hidden_states, *args):
        batch_size = hidden_states.shape[0]
        vocab_size = 32000
        return torch.randn(batch_size, vocab_size)
    
    def __call__(self, *args, **kwargs):
        positions = kwargs.get('positions')
        if positions is not None:
            if positions.dim() == 2:  # M-RoPE case
                num_tokens = positions.shape[1]
            else:  # Standard case
                num_tokens = positions.shape[0]
        else:
            num_tokens = 8
        
        hidden_size = 4096
        dummy_last_hidden = torch.randn(num_tokens, hidden_size)
        dummy_hidden = torch.randn(num_tokens, hidden_size)
        return dummy_last_hidden, dummy_hidden


class MockTreeAttentionMetadataBuilder(TreeAttentionMetadataBuilder):
    def __init__(self):
        pass
    
    def build_for_drafting(self, **kwargs):
        class MockTreeAttentionMetadata:
            def __init__(self):
                self.max_seq_len = 100
                self.seq_lens = torch.tensor([50, 60])
                self.block_table = torch.randint(0, 1000, (2, 100))
                self.slot_mapping = torch.zeros(100, dtype=torch.long)
                self.num_actual_tokens = 2
        
        return MockTreeAttentionMetadata()


class MockRunner:
    def __init__(self):
        class MockAttentionGroup:
            def __init__(self):
                self.metadata_builder = MockTreeAttentionMetadataBuilder()
        
        self.attn_groups = [[MockAttentionGroup()]]


def test_mrope_position_updates():
    """Test M-RoPE position update logic"""
    print("Testing M-RoPE position updates...")
    
    proposer = MinimalEagleProposer()
    
    # Test data
    batch_size = 2
    positions = torch.tensor([
        [50, 80],   # dim 0: sequence positions
        [25, 40],   # dim 1: image positions  
        [10, 15]    # dim 2: text positions
    ], dtype=torch.int64)
    
    # Test position update logic
    result = proposer.test_mrope_position_logic(batch_size, positions)
    assert result == True, "M-RoPE position logic test failed"
    
    print("M-RoPE position updates test passed")


def test_mrope_tree_propose_core():
    """Test the core M-RoPE part of tree_propose"""
    print("Testing tree_propose M-RoPE core logic...")
    
    proposer = MinimalEagleProposer()
    batch_size = 2
    
    # M-RoPE positions: (3, batch_size)
    positions = torch.tensor([
        [50, 80],   # dim 0: sequence positions
        [25, 40],   # dim 1: image positions  
        [10, 15]    # dim 2: text positions
    ], dtype=torch.int64)
    
    # Test precomputed draft token positions
    # Precompute the draft token positions. -> (3, B, L)
    flattened_draft_positions = (
        positions.view(3, batch_size, 1) +
        proposer.tree_draft_pos_offsets[:batch_size, :].unsqueeze(0)
    )
    
    print(f"Flattened draft positions shape: {flattened_draft_positions.shape}")
    print(f"Flattened draft positions: {flattened_draft_positions}")
    
    # Validate shape
    expected_shape = (3, batch_size, len(proposer.tree_choices))
    assert flattened_draft_positions.shape == expected_shape, \
        f"Shape mismatch: {flattened_draft_positions.shape} vs {expected_shape}"
    
    # Test query positions calculation
    level = 0
    query_len = proposer.cu_drafts_per_level[0]
    query_positions = flattened_draft_positions[:, :, level:level + query_len]
    
    print(f"Query positions shape: {query_positions.shape}")
    print(f"Query positions: {query_positions}")
    
    # Test block numbers calculation
    block_numbers = query_positions[0] // proposer.block_size
    print(f"Block numbers: {block_numbers}")
    
    print("tree_propose M-RoPE core logic test passed")


if __name__ == "__main__":
    print("Testing M-RoPE in tree_propose method")
    
    # Test position update logic
    test_mrope_position_updates()
    
    # Test core calculation logic
    test_mrope_tree_propose_core()
    
    print("All M-RoPE tree_propose tests passed")