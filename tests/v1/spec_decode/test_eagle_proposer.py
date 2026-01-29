import unittest
import torch
from collections import defaultdict
import numpy as np
from vllm.v1.spec_decode.eagle_proposer import EagleProposer

class TestEagleProposer(unittest.TestCase):
    
    def setUp(self):
        # This runs before each test
        self.proposer = EagleProposer()
        
    def test_get_prev_hidden_states_for_proposer(self):
        # Input values for the test case
        req_ids = [1, 2, 3]
        
        # Example tensors for the test
        hidden_state = torch.randn(10, 256)  # 5 tokens with hidden state size 256
        original_seq_start_locs = torch.tensor([0, 2, 4, 6, 8])
        proposer_seq_start_locs = torch.tensor([0, 2, 4, 3, 4])
        proposer_seq_lens = torch.tensor([2, 2, 2, 2, 2]) 
        hidden_state_from_prev_prefill = torch.tensor([False, False, False, False, False]) 
        is_prefill = torch.tensor([False, False, False, False, False]) 
        accepted_token_lengths = torch.tensor([1, 1, 1, 1, 1]) 

        # Call the method to be tested
        result = self.proposer._get_prev_hidden_states_for_proposer(
            req_ids,
            hidden_state,
            original_seq_start_locs,
            proposer_seq_start_locs,
            proposer_seq_lens,
            hidden_state_from_prev_prefill,
            is_prefill,
            accepted_token_lengths
        )
        # Assertions
        # Checking the shape of the returned tensor is consistent with the expected output
        total_tokens = proposer_seq_lens.sum().item()
        self.assertEqual(result.shape[0], total_tokens)  # The number of tokens in the result
        self.assertEqual(result.shape[1], hidden_state.shape[1])  # The hidden state size should match
        
    