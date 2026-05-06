# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for https://github.com/vllm-project/vllm/issues/34865

When multiple KV cache groups share the same MambaSpec (as in Nemotron
hybrid models), the metadata caching optimization reuses metadata from
an earlier group via update_block_table(). In 'all' mode with CUDA graphs,
update_block_table() must copy block_idx_last_scheduled_token and
block_idx_last_computed_token to the *current* builder's persistent
buffers, otherwise CUDA graph replay reads stale values from uninitialized
buffers.
"""

from types import SimpleNamespace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


class _ConcreteMambaBuilder(
    BaseMambaAttentionMetadataBuilder[BaseMambaAttentionMetadata]
):
    """Minimal concrete subclass for testing (base class is ABC)."""

    metadata_cls = BaseMambaAttentionMetadata


def _make_vllm_config(block_size, max_model_len, max_num_seqs):
    """Create a minimal mock VllmConfig with only the fields the builder
    accesses, avoiding any model download / HF config inspection."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_update_block_table_copies_block_idx_to_persistent_buffers():
    """update_block_table() must write block_idx tensors to the current
    builder's persistent buffers, not leave them pointing to a different
    builder's buffers."""

    block_size = 16
    max_model_len = 256
    num_reqs = 4
    device = torch.device("cpu")

    vllm_config = _make_vllm_config(block_size, max_model_len, num_reqs)

    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="all",
    )

    # Two builders simulating two KV cache groups with the same MambaSpec.
    builder_a = _ConcreteMambaBuilder(spec, ["layer0"], vllm_config, device)
    builder_b = _ConcreteMambaBuilder(spec, ["layer1"], vllm_config, device)

    # Sanity: each builder has its own persistent buffer.
    assert (
        builder_a.block_idx_last_scheduled_token.data_ptr()
        != builder_b.block_idx_last_scheduled_token.data_ptr()
    )

    # Construct decode-only metadata as if builder_a.build() produced it.
    max_blocks = max_model_len // block_size
    seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32, device=device)
    block_idx_vals = (seq_lens - 1) // block_size  # [3, 3, 3, 3]

    builder_a.block_idx_last_scheduled_token[:num_reqs].copy_(block_idx_vals)
    builder_a.block_idx_last_computed_token[:num_reqs].copy_(block_idx_vals)

    metadata_a = BaseMambaAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs,
        num_reqs=num_reqs,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        state_indices_tensor_d=builder_a.state_indices_tensor_d[:num_reqs],
        block_idx_last_scheduled_token=(
            builder_a.block_idx_last_scheduled_token[:num_reqs]
        ),
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=(
            builder_a.block_idx_last_computed_token[:num_reqs]
        ),
        seq_lens=seq_lens,
    )

    # Call update_block_table on builder_b (simulates the metadata caching
    # optimization reusing metadata from builder_a's group).
    blk_table = torch.randint(
        0, 100, (num_reqs, max_blocks), dtype=torch.int32, device=device
    )
    slot_mapping = torch.zeros(num_reqs, dtype=torch.int64, device=device)

    metadata_b = builder_b.update_block_table(metadata_a, blk_table, slot_mapping)

    # block_idx tensors must live in builder_b's persistent buffers.
    def shares_storage(tensor, buffer):
        return (
            tensor.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        )

    assert shares_storage(
        metadata_b.block_idx_last_scheduled_token,
        builder_b.block_idx_last_scheduled_token,
    ), "block_idx_last_scheduled_token not in builder_b's persistent buffer"

    assert shares_storage(
        metadata_b.block_idx_last_computed_token,
        builder_b.block_idx_last_computed_token,
    ), "block_idx_last_computed_token not in builder_b's persistent buffer"

    # Must NOT point to builder_a's buffers.
    assert not shares_storage(
        metadata_b.block_idx_last_scheduled_token,
        builder_a.block_idx_last_scheduled_token,
    ), "block_idx_last_scheduled_token still points to builder_a's buffer"

    # Values must be correct (copied from metadata_a).
    torch.testing.assert_close(
        metadata_b.block_idx_last_scheduled_token,
        block_idx_vals,
    )
    torch.testing.assert_close(
        metadata_b.block_idx_last_computed_token,
        block_idx_vals,
    )
