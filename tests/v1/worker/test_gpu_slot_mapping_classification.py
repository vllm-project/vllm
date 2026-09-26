# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for #56380 (problem 2): builder-managed KV cache groups
must be excluded from the generic position-indexed slot-mapping kernel.

``KpoolTailSpec`` rows hold one circular block per request. The generic
``_compute_slot_mappings_kernel`` computes ``pos // kernel_block_size``
against that one-column row and reads past it on chunked prefill, poisoning
the block ids used by every kpool consumer. The circular mapping is owned by
``KpoolTailMetadataBuilder``.
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    uses_generic_slot_mapping,
)

# Pure spec-classification tests: no accelerator, no dist env — skip the
# global teardown so the suite runs on CPU-only machines.
pytestmark = pytest.mark.skip_global_cleanup


def _kpool_tail_spec() -> KpoolTailSpec:
    return KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )


def _circular_buffer_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=640,
        shapes=((1, 8, 128), (1, 8, 128, 3)),
        dtypes=(torch.float32, torch.float32),
    )


@pytest.mark.parametrize(
    "spec_factory",
    [_kpool_tail_spec, _circular_buffer_spec],
    ids=["kpool_tail", "circular_buffer"],
)
def test_builder_managed_groups_excluded_from_generic_slot_mapping(spec_factory):
    # The bug: KpoolTailSpec used to pass the CircularBufferSpec-only check,
    # so the generic position-indexed kernel indexed its one-column
    # block-table row with pos // kpool on chunked prefill.
    assert not uses_generic_slot_mapping(spec_factory())


def test_position_indexed_groups_included():
    full = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    attention = AttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    assert uses_generic_slot_mapping(full)
    assert uses_generic_slot_mapping(attention)
    assert uses_generic_slot_mapping(_mamba_spec())


def test_uniform_type_wrapper_unwraps_to_member_spec():
    # A UniformTypeKVCacheSpecs collection must be classified by its member
    # specs: a kpool-tail-only group must stay excluded through the wrapper,
    # mirroring is_full_attention_spec's handling of the wrapper.
    spec = UniformTypeKVCacheSpecs(
        block_size=4,
        kv_cache_specs={
            "decoder.layers.0.kpool_tail": _kpool_tail_spec(),
            "decoder.layers.1.kpool_tail": _kpool_tail_spec(),
        },
    )
    assert not uses_generic_slot_mapping(spec)

    full_group = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "decoder.layers.0.self_attn": FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                head_size_v=128,
                dtype=torch.bfloat16,
            )
        },
    )
    assert uses_generic_slot_mapping(full_group)
