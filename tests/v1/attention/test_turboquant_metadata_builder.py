# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant attention metadata builder spec fields."""

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


def test_turboquant_build_without_spec_keeps_existing_split():
    builder = _create_turboquant_builder(num_speculative_tokens=0)
    batch = BatchSpec(seq_lens=[40, 30, 20], query_lens=[1, 1, 1])

    meta = _build(builder, batch)

    assert meta.num_decodes == 3
    assert meta.num_decode_tokens == 3
    assert meta.num_spec_decodes == 0
    assert meta.num_spec_decode_tokens == 0
    assert meta.spec_sequence_masks is None
    assert meta.spec_token_indx is None
    assert meta.non_spec_token_indx is None


def test_turboquant_build_with_spec_config_but_no_spec_sequence_is_inert():
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[1, 1])

    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1])

    assert builder.reorder_batch_threshold == 3
    assert meta.num_decodes == 2
    assert meta.num_decode_tokens == 2
    assert meta.num_spec_decodes == 0
    assert meta.spec_sequence_masks is None


def test_turboquant_no_spec_sequence_keeps_regular_decode_threshold():
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[2, 1])

    meta = _build(builder, batch, num_decode_draft_tokens=[-1, -1])

    assert builder.reorder_batch_threshold == 3
    assert meta.num_decodes == 0
    assert meta.num_decode_tokens == 0
    assert meta.num_spec_decodes == 0
    assert meta.spec_sequence_masks is None


def test_turboquant_build_pure_spec_sequence_fields():
    builder = _create_turboquant_builder(num_speculative_tokens=2)
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
    assert meta.num_spec_decode_tokens == 6
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.tolist() == [True, True]
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.tolist() == [0, 3, 6]
    assert meta.non_spec_query_start_loc is not None
    assert meta.non_spec_query_start_loc.tolist() == [0]
    assert meta.spec_token_indx is not None
    assert meta.spec_token_indx.tolist() == [0, 1, 2, 3, 4, 5]
    assert meta.non_spec_token_indx is not None
    assert meta.non_spec_token_indx.numel() == 0
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1, 2]


def test_turboquant_build_mixed_spec_token_partition():
    builder = _create_turboquant_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20, 70], query_lens=[1, 3, 5])

    meta = _build(
        builder,
        batch,
        num_decode_draft_tokens=[-1, 2, -1],
        num_accepted_tokens=[1, 2, 1],
    )

    # The spec request uses the raised spec threshold, but regular non-spec
    # requests still classify only one-token queries as decode.
    assert meta.num_decodes == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_spec_decodes == 1
    assert meta.num_spec_decode_tokens == 3
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.tolist() == [False, True, False]
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.tolist() == [0, 3]
    assert meta.non_spec_query_start_loc is not None
    assert meta.non_spec_query_start_loc.tolist() == [0, 1, 6]
    assert meta.non_spec_query_start_loc_cpu is not None
    assert meta.non_spec_query_start_loc_cpu.tolist() == [0, 1, 6]
    assert meta.non_spec_seq_lens_cpu is not None
    assert meta.non_spec_seq_lens_cpu.tolist() == [65, 70]
    assert meta.non_spec_token_indx is not None
    assert meta.non_spec_token_indx.tolist() == [0, 4, 5, 6, 7, 8]
    assert meta.spec_token_indx is not None
    assert meta.spec_token_indx.tolist() == [1, 2, 3]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [2]


def test_turboquant_spec_decode_seq_lens_increment_within_request():
    seq_lens = torch.tensor([50, 30], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)

    synth_seq_lens = _build_spec_decode_synth_seq_lens(
        seq_lens,
        query_start_loc,
        output_size=6,
    )

    # The decode kernel treats each flattened token as the last token in its
    # synthetic row. Repeating [50, 30] would expose future speculative KV.
    assert synth_seq_lens.tolist() == [48, 49, 50, 28, 29, 30]
