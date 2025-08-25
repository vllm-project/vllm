# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from unittest.mock import Mock, patch

from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.config import VllmConfig

def create_mock_vllm_config(uses_mrope=True):
    config = Mock(spec=VllmConfig)

    # Model config
    config.model_config = Mock()
    config.model_config.dtype = torch.float16
    config.model_config.max_model_len = 2048
    config.model_config.uses_mrope = uses_mrope
    config.model_config.is_multimodal_model = False
    config.model_config.enforce_eager = False

    # Cache config
    config.cache_config = Mock()
    config.cache_config.block_size = 16

    # Scheduler config
    config.scheduler_config = Mock()
    config.scheduler_config.max_num_batched_tokens = 512
    config.scheduler_config.max_num_seqs = 8

    # Speculative config
    config.speculative_config = Mock()
    config.speculative_config.num_speculative_tokens = 3
    config.speculative_config.method = "eagle"
    # Tree: root + 2 children + 2 grandchildren = 5 nodes, num_drafts_per_level=[1,2,2]
    config.speculative_config.speculative_token_tree = "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]"

    # Draft model config
    draft_model_config = Mock()
    draft_model_config.get_hidden_size.return_value = 4096
    config.speculative_config.draft_model_config = draft_model_config

    # Compilation config
    config.compilation_config = Mock()
    config.compilation_config.level = Mock()
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]

    # Cudagraph pad passthrough
    config.pad_for_cudagraph = lambda x: x
    return config

class MockAttentionGroup:
    def __init__(self, batch_size, device):
        self.metadata_builder = Mock(spec=TreeAttentionMetadataBuilder)
        self.layer_names = ["layer1", "layer2"]
        self.batch_size = batch_size
        self.device = device
        self.call_count = 0

    def create_mock_metadata(self, common_attn_metadata, draft_index):
        # Use the query_len encoded in common_attn_metadata to make num_actual_tokens consistent
        num_tokens = common_attn_metadata.num_actual_tokens
        mock_attn_metadata = Mock()
        mock_attn_metadata.num_actual_tokens = int(num_tokens)
        mock_attn_metadata.max_seq_len = int(common_attn_metadata.max_seq_len)
        mock_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        mock_attn_metadata.block_table = torch.randint(
            0, 100, (self.batch_size, 20), device=self.device
        )
        # Initialize slot_mapping with correct length (num_tokens)
        mock_attn_metadata.slot_mapping = torch.randint(
            0, 1000, (num_tokens,), device=self.device
        )
        return mock_attn_metadata

def create_runner(batch_size, device):
    runner = Mock()
    grp = MockAttentionGroup(batch_size, device)
    grp.metadata_builder.build_for_drafting.side_effect = (
        lambda common_attn_metadata, draft_index:
            grp.create_mock_metadata(common_attn_metadata, draft_index)
    )
    runner.attn_groups = [[grp]]
    return runner

def create_common_attn_metadata(batch_size, device="cuda"):
    return CommonAttentionMetadata(
        query_start_loc=torch.arange(0, batch_size + 1, 1, device=device, dtype=torch.int32),
        seq_lens=torch.full((batch_size,), 50, device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(0, batch_size + 1, 1, dtype=torch.int32),
        seq_lens_cpu=torch.full((batch_size,), 50, dtype=torch.int32),
        num_computed_tokens_cpu=torch.full((batch_size,), 40, dtype=torch.int32),
        num_reqs=batch_size,
        num_actual_tokens=batch_size * 1,
        max_query_len=1,
        max_seq_len=100,
        block_table_tensor=torch.randint(0, 100, (batch_size, 20), device=device),
        slot_mapping=torch.randint(0, 1000, (batch_size * 1,), device=device),
        causal=True,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEagleProposerTreeMRoPE:
    def setup_method(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    @patch('vllm.v1.spec_decode.eagle.set_forward_context')
    @patch('vllm.v1.spec_decode.eagle.get_layers_from_vllm_config')
    @patch('vllm.v1.spec_decode.eagle.get_model')
    def test_propose_tree_mrope_tree_attention(self, mock_get_model, mock_get_layers, mock_set_forward_context):
        batch_size = 2
        mock_get_layers.return_value = {}

        config = create_mock_vllm_config(uses_mrope=True)
        runner = create_runner(batch_size, self.device)

        # Use normal constructor so __init__ sets everything (arange, tree_draft_pos_offsets, buffers)
        proposer = EagleProposer(config, self.device, runner)
        proposer.attn_layer_names = ["layer1", "layer2"]  # minimal requirement

        # Mock model
        mock_model = Mock()
        # Return shapes must match num_input_tokens; we won't rely on exact sizes here because we slice [:num_tokens]
        mock_model.return_value = (
            torch.randn(batch_size * 4, proposer.hidden_size, device=self.device),
            torch.randn(batch_size * 4, proposer.hidden_size, device=self.device),
        )
        mock_model.compute_logits.return_value = torch.randn(batch_size * 2, 50000, device=self.device)
        proposer.model = mock_model

        vocab_size = 50000
        logits = torch.randn(batch_size, vocab_size, device=self.device)
        # M-RoPE positions shape: (3, batch_size)
        positions = torch.randint(10, 100, (3, batch_size), device=self.device, dtype=torch.int64)
        hidden_states = torch.randn(batch_size, proposer.hidden_size, device=self.device)
        common_attn_metadata = create_common_attn_metadata(batch_size, self.device)

        result = proposer.propose_tree(
            batch_size=batch_size,
            logits=logits,
            positions=positions,
            hidden_states=hidden_states,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        for draft_tokens in result:
            assert draft_tokens.shape[0] == batch_size