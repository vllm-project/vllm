from unittest.mock import MagicMock, patch

import torch

from vllm.attention.backends.abstract import (
  AttentionType,
)
from vllm.v1.attention.backends.pallas import (
  PallasAttentionBackendImpl,
  PallasMetadata,
  NUM_KV_PAGES_PER_BLOCK,
  NUM_QUERIES_PER_BLOCK,
)



def test_ragged_paged_attention():
  num_heads = 4
  head_size = 128
  scale = 1.0
  num_kv_heads = 4
  sliding_window = 128
  logits_soft_cap = 50.0
  attn_impl = PallasAttentionBackendImpl(
      num_heads=num_heads,
      head_size=head_size,
      scale=scale,
      num_kv_heads=num_kv_heads,
      alibi_slopes=None,
      sliding_window=sliding_window,
      kv_cache_dtype="auto",
      logits_soft_cap=logits_soft_cap,
      attn_type=AttentionType.DECODER,
  )

  class FakeAttentionLayer:
    _k_scale_float: float
    _v_scale_float: float
  
  layer = FakeAttentionLayer()
  layer._k_scale_float = 1.0
  layer._v_scale_float = 1.0

  num_tokens = 16
  num_blocks = 1024
  block_size = 16
  query = torch.randn(num_tokens, num_heads * head_size)
  key = torch.randn(num_tokens, num_kv_heads * head_size)
  value = torch.randn(num_tokens, num_kv_heads * head_size)
  key_cache = torch.randn(num_blocks, block_size,
                            num_kv_heads * head_size)
  value_cache = torch.randn(num_blocks, block_size,
                            num_kv_heads * head_size)
  slot_mapping = torch.zeros(num_tokens, dtype=torch.int64)
  max_num_reqs = 8
  max_num_blocks_per_req = 8
  block_tables = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=torch.int32)
  context_lens = torch.ones((max_num_reqs,), dtype=torch.int32)
  query_lens = [1] * max_num_reqs
  query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                              dtype=torch.int32),
                                 dim=0,
                                 dtype=torch.int32)
  num_seqs = torch.tensor([max_num_reqs], dtype=torch.int32)
  attn_metadata = PallasMetadata(
      slot_mapping=slot_mapping,
      block_tables=block_tables,
      context_lens=context_lens,
      query_start_loc=query_start_loc,
      num_seqs=num_seqs,
  )

  with patch("torch.ops.xla.ragged_paged_attention") as mock_ragged_paged_attention:
    attn_impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=(key_cache, value_cache),
        attn_metadata=attn_metadata,
    )
    
    mock_ragged_paged_attention.assert_called_once_with(
        query.view(num_tokens, num_heads, head_size),
        key_cache,
        value_cache,
        attn_metadata.context_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.num_seqs,
        num_kv_pages_per_block=NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=NUM_QUERIES_PER_BLOCK,
        vmem_limit_bytes=attn_impl.vmem_limit_bytes,
        use_kernel=True,
        sm_scale=scale,
        sliding_window=sliding_window,
        soft_cap=logits_soft_cap,
    )
  

