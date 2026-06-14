# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant spec-decode metadata building."""

from types import SimpleNamespace

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import SpeculativeConfig
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
    _build_spec_decode_synth_seq_lens,
)

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _create_turboquant_builder(
    num_speculative_tokens: int = 0,
) -> TurboQuantMetadataBuilder:
    vllm_config = _create_vllm_config(num_speculative_tokens)
    return TurboQuantMetadataBuilder(
        kv_cache_spec=object(),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _create_vllm_config(num_speculative_tokens: int = 0) -> SimpleNamespace:
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    return SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def _build(
    builder: TurboQuantMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    num_accepted_tokens: list[int] | None = None,
) -> TurboQuantMetadata:
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        if num_accepted_tokens is None:
            num_accepted_tokens = [1] * batch_spec.batch_size
        kwargs["num_accepted_tokens"] = torch.tensor(
            num_accepted_tokens, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


def _assert_no_spec_metadata(meta: TurboQuantMetadata) -> None:
    assert meta.num_spec_decodes == 0
    assert meta.num_spec_decode_tokens == 0
    assert meta.spec_sequence_masks is None
    assert meta.spec_token_indx is None
    assert meta.non_spec_token_indx is None


def test_regular_decode_batch_has_no_spec_metadata():
    """No speculative config: keep the normal one-token decode split."""
    builder = _create_turboquant_builder(num_speculative_tokens=0)
    batch = BatchSpec(seq_lens=[40, 30, 20], query_lens=[1, 1, 1])

    meta = _build(builder, batch)

    assert meta.num_decodes == 3
    assert meta.num_decode_tokens == 3
    _assert_no_spec_metadata(meta)


def test_spec_config_with_no_draft_tokens_uses_regular_decode_split():
    """Spec config alone should not make a normal batch use spec routing."""
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[1, 1])

    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1])

    assert builder.reorder_batch_threshold == 3
    assert meta.num_decodes == 2
    assert meta.num_decode_tokens == 2
    _assert_no_spec_metadata(meta)


def test_no_draft_tokens_do_not_use_raised_spec_threshold():
    """A two-token prefill is not a decode when no request has draft tokens."""
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    # query_lens=[2, 1] would be classified as decode if the raised spec
    # threshold were used. TQ should keep threshold=1 for no-spec batches.
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[2, 1])

    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1])

    assert builder.reorder_batch_threshold == 3
    assert meta.num_decodes == 0
    assert meta.num_decode_tokens == 0
    _assert_no_spec_metadata(meta)


def test_pure_spec_batch_builds_flat_decode_token_order():
    """Two K+1 requests become six flattened spec decode tokens."""
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    # Each request verifies K+1=3 tokens; both requests are spec.
    spec_tokens_per_request = 3
    total_spec_tokens = 2 * spec_tokens_per_request
    batch = BatchSpec(seq_lens=[50, 30], query_lens=[3, 3])

    meta = _build(
        builder,
        batch,
        num_decode_draft_tokens=[2, 2],
        num_accepted_tokens=[1, 2],
    )

    assert meta.num_decodes == 0
    assert meta.num_decode_tokens == 0
    assert meta.num_spec_decodes == 2
    assert meta.num_spec_decode_tokens == total_spec_tokens

    # Both requests have draft tokens, so every query token is in spec verify.
    assert meta.spec_sequence_masks.tolist() == [True, True]

    # Each request contributes three verify tokens to the flattened spec batch.
    assert meta.spec_query_start_loc.tolist() == [0, 3, 6]

    # All flattened query tokens are spec tokens; no regular tokens remain.
    assert meta.spec_token_indx.tolist() == list(range(total_spec_tokens))
    assert meta.non_spec_token_indx.numel() == 0

    assert meta.non_spec_query_start_loc.tolist() == [0]

    # Accepted-token counts are filtered to spec requests only.
    assert meta.num_accepted_tokens.tolist() == [1, 2]


def test_mixed_spec_batch_splits_spec_tokens_from_regular_tokens():
    """Keep regular decode/prefill tokens separate from K+1 spec tokens."""
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    # Request 0: normal decode token.
    # Request 1: spec verify with K+1 = 3 query tokens.
    # Request 2: normal five-token prefill.
    batch = BatchSpec(seq_lens=[65, 20, 70], query_lens=[1, 3, 5])
    # Flatten query_lens=[1, 3, 5] into a token stream:
    #   request 0 -> index 0       (decode, non-spec)
    #   request 1 -> indices 1-3   (spec verify)
    #   request 2 -> indices 4-8   (prefill, non-spec)
    decode_token = [0]
    spec_tokens = [1, 2, 3]
    prefill_tokens = list(range(4, 9))
    non_spec_tokens = decode_token + prefill_tokens

    meta = _build(
        builder,
        batch,
        num_decode_draft_tokens=[-1, 2, -1],
        num_accepted_tokens=[1, 2, 1],
    )

    assert meta.num_decodes == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_spec_decodes == 1
    assert meta.num_spec_decode_tokens == len(spec_tokens)

    # Only request 1 is the speculative verify request.
    assert meta.spec_sequence_masks.tolist() == [False, True, False]

    # The spec subset contains one request with three query tokens.
    assert meta.spec_query_start_loc.tolist() == [0, 3]

    # Token positions in the original flattened query:
    #   request 0 decode: [0]
    #   request 1 verify: [1, 2, 3]
    #   request 2 prefill: [4, 5, 6, 7, 8]
    assert meta.non_spec_token_indx.tolist() == non_spec_tokens
    assert meta.spec_token_indx.tolist() == spec_tokens

    # Non-spec subset has query lengths [1, 5], so cumsum is [0, 1, 6].
    assert meta.non_spec_query_start_loc.tolist() == [0, 1, 6]
    # CPU copy is used by the scheduler; verify it stays in sync.
    assert meta.non_spec_query_start_loc_cpu.tolist() == [0, 1, 6]

    # Seq lens are filtered to the two non-spec requests: request 0 and 2.
    assert meta.non_spec_seq_lens_cpu.tolist() == [65, 70]

    # Accepted-token counts are also filtered to the spec request.
    assert meta.num_accepted_tokens.tolist() == [2]


def test_spec_decode_seq_lens_increase_within_each_request():
    """Earlier verify tokens must not see later speculative KV tokens."""
    final_seq_lens = torch.tensor([50, 30], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)

    synth_seq_lens = _build_spec_decode_synth_seq_lens(
        final_seq_lens,
        query_start_loc,
        output_size=6,
    )

    # The decode kernel uses query_pos = seq_len - 1 for each flattened row.
    # For a 3-token verify request with final length 50, visible lengths must
    # be 48, 49, 50. Repeating 50 would expose future KV to token 0 and 1.
    assert synth_seq_lens.tolist() == [48, 49, 50, 28, 29, 30]
