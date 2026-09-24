# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the shared transfer classification."""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.transfer_planning import (
    TransferClass,
    get_representative_spec,
    get_representative_spec_type,
    is_attention_spec,
    is_ssm_spec,
    transfer_class,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KpoolTailSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_sliding_window,
    is_full_attention_spec,
)

# Arguments mirror the specs the existing unit tests construct.
ATTENTION_KWARGS = {
    "block_size": 16,
    "num_kv_heads": 4,
    "head_size": 64,
    "dtype": torch.float16,
}
MLA_KWARGS = {
    "block_size": 16,
    "num_kv_heads": 1,
    "head_size": 64,
    "dtype": torch.float16,
}

SPEC_INSTANCES = [
    FullAttentionSpec(**ATTENTION_KWARGS),
    MLAAttentionSpec(**MLA_KWARGS),
    HiddenStateCacheSpec(**MLA_KWARGS),
    RSWASpec(**ATTENTION_KWARGS, rswa_window=128),
    ChunkedLocalAttentionSpec(**ATTENTION_KWARGS, attention_chunk_size=32),
    SlidingWindowSpec(**ATTENTION_KWARGS, sliding_window=128),
    CircularBufferSpec(**ATTENTION_KWARGS),
    SlidingWindowMLASpec(**MLA_KWARGS, sliding_window=128),
    KpoolTailSpec(**ATTENTION_KWARGS, sliding_window=128),
    MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.float16,)),
    EncoderOnlyAttentionSpec(**ATTENTION_KWARGS),
    CrossAttentionSpec(**ATTENTION_KWARGS),
    SinkFullAttentionSpec(**ATTENTION_KWARGS, sink_len=16),
    HiSparseHotSpec(block_size=16, page_size=2048, blocks_per_request=2),
    HiSparseResidentSpec(block_size=16, page_size=2048),
]


def _all_spec_classes() -> list[type[KVCacheSpec]]:
    """Every KVCacheSpec subclass in the tree, the wrapper excluded.

    The wrapper describes a group, so the table answers for it through the
    specs it wraps and the walk leaves it out.
    """
    found: list[type[KVCacheSpec]] = []
    pending = [KVCacheSpec]
    while pending:
        for spec_cls in pending.pop().__subclasses__():
            if spec_cls is UniformTypeKVCacheSpecs:
                continue
            found.append(spec_cls)
            pending.append(spec_cls)
    return sorted(found, key=lambda cls: cls.__name__)


ALL_SPEC_CLASSES = _all_spec_classes()


@pytest.mark.parametrize("spec_cls", ALL_SPEC_CLASSES, ids=lambda cls: cls.__name__)
def test_transfer_class_agrees_with_spec_hierarchy(spec_cls: type[KVCacheSpec]):
    """The table classifies every spec class exactly as the hierarchy does.

    ``ATTENTION`` is ``issubclass(spec_cls, AttentionSpec)``, ``SSM`` is
    ``issubclass(spec_cls, MambaSpec)`` and everything else is ``OTHER``, which
    is what makes it safe to replace the per-connector predicates.
    """
    if issubclass(spec_cls, AttentionSpec):
        expected = TransferClass.ATTENTION
    elif issubclass(spec_cls, MambaSpec):
        expected = TransferClass.SSM
    else:
        expected = TransferClass.OTHER

    assert transfer_class(spec_cls) is expected
    assert is_attention_spec(spec_cls) is (expected is TransferClass.ATTENTION)
    assert is_ssm_spec(spec_cls) is (expected is TransferClass.SSM)


@pytest.mark.parametrize(
    "spec_cls,expected_class",
    [
        # A circular buffer is attention state even though the kind table does
        # not name its class.
        (CircularBufferSpec, TransferClass.ATTENTION),
        (KpoolTailSpec, TransferClass.ATTENTION),
        (HiSparseHotSpec, TransferClass.OTHER),
        (HiSparseResidentSpec, TransferClass.OTHER),
        (MambaSpec, TransferClass.SSM),
        (MLAAttentionSpec, TransferClass.ATTENTION),
        (SlidingWindowMLASpec, TransferClass.ATTENTION),
        (HiddenStateCacheSpec, TransferClass.ATTENTION),
    ],
)
def test_named_spec_classes(spec_cls, expected_class):
    """Pin the classes the two deliberate answers in the table cover."""
    assert transfer_class(spec_cls) is expected_class


@pytest.mark.parametrize("spec", SPEC_INSTANCES, ids=lambda spec: type(spec).__name__)
def test_instance_classifies_like_its_class(spec: KVCacheSpec):
    """The API takes instances and classes and both answer the same."""
    assert transfer_class(spec) is transfer_class(type(spec))
    assert is_attention_spec(spec) is is_attention_spec(type(spec))
    assert is_ssm_spec(spec) is is_ssm_spec(type(spec))


def _group(specs: dict[str, KVCacheSpec]) -> UniformTypeKVCacheSpecs:
    """Build a group the way the framework builds one, so the layers are legal."""
    group = UniformTypeKVCacheSpecs.from_specs(specs)
    assert group is not None, "the layers must share one registered base spec"
    return group


def test_group_class_is_stable_across_layer_order():
    """A group of full attention and MLA layers is attention state either way.

    Layers merge into one group only when they share a registered base spec, and
    ``FullAttentionSpec`` is the base of ``MLAAttentionSpec`` as well, so this
    pair is a legal group whose concrete classes differ per layer.
    """
    full = FullAttentionSpec(**ATTENTION_KWARGS)
    mla = MLAAttentionSpec(**MLA_KWARGS)

    for specs in ({"full": full, "mla": mla}, {"mla": mla, "full": full}):
        group = _group(specs)
        assert transfer_class(group) is TransferClass.ATTENTION
        assert is_attention_spec(group) is True
        assert is_ssm_spec(group) is False


def test_representative_spec_answers_for_a_group():
    """The representative spec stands for the group, not for one layer.

    That holds because layers only merge into a group when they share one
    registered base spec, which the group builder checks.
    """
    full = FullAttentionSpec(**ATTENTION_KWARGS)
    group = _group({"layer_0": full})

    assert get_representative_spec(group) is full
    assert get_representative_spec_type(group) is FullAttentionSpec
    assert transfer_class(group) is TransferClass.ATTENTION


def test_group_with_mixed_transfer_classes_is_rejected():
    """A group that cannot move as one class fails closed."""
    full = FullAttentionSpec(**ATTENTION_KWARGS)
    mamba = MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.float16,))
    group = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"full": full, "mamba": mamba}
    )

    with pytest.raises(ValueError, match="one transfer class"):
        transfer_class(group)


def test_empty_wrapper_is_rejected():
    """A wrapper without layers has no representative and no class."""
    group = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={})

    with pytest.raises(ValueError, match="carries no layer spec"):
        get_representative_spec(group)
    with pytest.raises(ValueError, match="one transfer class"):
        transfer_class(group)


def test_wrapper_class_has_no_class_level_answer():
    """The wrapper class describes a group, so asking for its class raises."""
    with pytest.raises(ValueError, match="UniformTypeKVCacheSpecs"):
        transfer_class(UniformTypeKVCacheSpecs)
    with pytest.raises(ValueError, match="UniformTypeKVCacheSpecs"):
        is_attention_spec(UniformTypeKVCacheSpecs)


def test_wrapped_group_answers_the_scheduler_questions():
    """The wrapper-aware framework helpers see through the group spec.

    This is the input the sliding-window budget reads, so a wrapped group
    reports the window and the kind it actually has.
    """
    full_spec = FullAttentionSpec(**ATTENTION_KWARGS)
    sw_spec = SlidingWindowSpec(**ATTENTION_KWARGS, sliding_window=2048)

    full_wrapper = _group({"layer_0": full_spec})
    sw_wrapper = _group({"layer_1": sw_spec})

    assert get_kv_cache_spec_sliding_window(full_wrapper) is None
    assert is_full_attention_spec(full_wrapper) is True
    assert transfer_class(full_wrapper) is TransferClass.ATTENTION

    assert get_kv_cache_spec_sliding_window(sw_wrapper) == 2048
    assert is_full_attention_spec(sw_wrapper) is False
    assert transfer_class(sw_wrapper) is TransferClass.ATTENTION
