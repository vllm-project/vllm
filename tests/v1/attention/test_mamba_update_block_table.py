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


def _make_vllm_config(max_model_len, max_num_seqs, num_speculative_tokens=0):
    """Create a minimal mock VllmConfig with only the fields the builder
    accesses, avoiding any model download / HF config inspection."""
    speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )
        if num_speculative_tokens > 0
        else None
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
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

    vllm_config = _make_vllm_config(max_model_len, num_reqs)

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


def test_state_indices_tensor_d_includes_num_speculative_blocks():
    """Regression test for https://github.com/vllm-project/vllm/issues/39809
    bug 1: with mamba_cache_mode='all' and speculative decoding enabled,
    the cudagraph buffer for state_indices_tensor_d must allocate the same
    per-request column count as the runtime block table, which includes
    num_speculative_blocks trailing scratch columns."""

    block_size = 16
    max_model_len = 256
    max_num_seqs = 4
    num_speculative_tokens = 1
    num_speculative_blocks = 2
    device = torch.device("cpu")

    vllm_config = _make_vllm_config(
        max_model_len,
        max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
    )

    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="all",
        num_speculative_blocks=num_speculative_blocks,
    )

    builder = _ConcreteMambaBuilder(spec, ["layer0"], vllm_config, device)

    expected_cols = (max_model_len // block_size) + num_speculative_blocks
    assert builder.state_indices_tensor_d.shape == (max_num_seqs, expected_cols)


def test_block_idx_cudagraph_capture_padded_by_num_reqs():
    """Regression test for https://github.com/vllm-project/vllm/issues/39809
    bug 2: with mamba_cache_mode='all' and spec decode, _update_metadata_for
    _cudagraph_capture must slice block_idx_last_{scheduled,computed}_token
    by the request count (padded_bs == num_reqs), not by num_decode_tokens.
    Past num_decodes, the slice must be zero-filled."""

    block_size = 16
    max_model_len = 256
    max_num_seqs = 8
    num_speculative_tokens = 1
    device = torch.device("cpu")

    vllm_config = _make_vllm_config(
        max_model_len,
        max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
    )

    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="all",
        num_speculative_blocks=2,
    )

    builder = _ConcreteMambaBuilder(spec, ["layer0"], vllm_config, device)

    builder.block_idx_last_scheduled_token.fill_(-1)
    builder.block_idx_last_computed_token.fill_(-1)

    num_decodes = 2
    num_reqs = 3
    num_decode_tokens = num_decodes * (1 + num_speculative_tokens)
    seq_lens = torch.full((num_reqs,), 64, dtype=torch.int32, device=device)
    block_idx_vals = torch.tensor([3, 5], dtype=torch.int32, device=device)
    state_indices_d = torch.zeros(
        (num_decodes, builder.state_indices_tensor_d.shape[1]),
        dtype=torch.int32,
        device=device,
    )
    query_start_loc_d = torch.arange(
        num_decodes + 1, dtype=torch.int32, device=device
    ) * (1 + num_speculative_tokens)
    num_accepted_tokens = torch.ones(num_decodes, dtype=torch.int32, device=device)

    metadata = BaseMambaAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_reqs=num_reqs,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=state_indices_d,
        query_start_loc_d=query_start_loc_d,
        num_accepted_tokens=num_accepted_tokens,
        block_idx_last_scheduled_token=block_idx_vals,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=block_idx_vals,
        seq_lens=seq_lens,
    )

    out = builder._update_metadata_for_cudagraph_capture(metadata)

    assert out.block_idx_last_scheduled_token.shape == (num_reqs,)
    assert out.block_idx_last_computed_token.shape == (num_reqs,)
    torch.testing.assert_close(
        out.block_idx_last_scheduled_token[:num_decodes], block_idx_vals
    )
    torch.testing.assert_close(
        out.block_idx_last_computed_token[:num_decodes], block_idx_vals
    )
    assert torch.all(out.block_idx_last_scheduled_token[num_decodes:] == 0)
    assert torch.all(out.block_idx_last_computed_token[num_decodes:] == 0)
