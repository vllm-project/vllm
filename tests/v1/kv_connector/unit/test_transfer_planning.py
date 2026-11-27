# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pytest
import torch

from tests.v1.kv_connector.unit.utils import make_kv_cache_group
from vllm.distributed.kv_transfer.kv_connector.v1.transfer_planning import (
    TransferClass,
    get_representative_spec_type,
    is_attention_spec,
    is_ssm_spec,
    transfer_class,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotPageSpec,
    HiSparsePageSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
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
    AttentionSpec(**ATTENTION_KWARGS),
    MLAAttentionSpec(**MLA_KWARGS),
    HiddenStateCacheSpec(**MLA_KWARGS),
    MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.float16,)),
    HiSparsePageSpec(block_size=16, page_size=2048),
    HiSparseHotPageSpec(block_size=16, page_size=2048),
]


def _all_spec_classes() -> list[type[KVCacheSpec]]:
    """Every KVCacheSpec subclass in the tree."""
    found: list[type[KVCacheSpec]] = []
    pending = [KVCacheSpec]
    while pending:
        for spec_cls in pending.pop().__subclasses__():
            found.append(spec_cls)
            pending.append(spec_cls)
    return sorted(found, key=lambda cls: cls.__name__)


ALL_SPEC_CLASSES = _all_spec_classes()


@pytest.mark.parametrize("spec_cls", ALL_SPEC_CLASSES, ids=lambda cls: cls.__name__)
def test_transfer_class_agrees_with_spec_hierarchy(spec_cls: type[KVCacheSpec]):
    """The classification matches the spec hierarchy for every spec class.

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
        (MambaSpec, TransferClass.SSM),
        (MLAAttentionSpec, TransferClass.ATTENTION),
        (HiddenStateCacheSpec, TransferClass.ATTENTION),
        (HiSparsePageSpec, TransferClass.OTHER),
        (HiSparseHotPageSpec, TransferClass.OTHER),
    ],
)
def test_named_spec_classes(spec_cls, expected_class):
    """Pin the classes the classification table covers."""
    assert transfer_class(spec_cls) is expected_class


@pytest.mark.parametrize("spec", SPEC_INSTANCES, ids=lambda spec: type(spec).__name__)
def test_instance_classifies_like_its_class(spec: KVCacheSpec):
    """The API takes instances and classes and both answer the same."""
    assert transfer_class(spec) is transfer_class(type(spec))
    assert is_attention_spec(spec) is is_attention_spec(type(spec))
    assert is_ssm_spec(spec) is is_ssm_spec(type(spec))


def test_group_classifies_by_its_specs():
    """A group answers through the specs its layers carry."""
    group = make_kv_cache_group(["layer0"], AttentionSpec(**ATTENTION_KWARGS))

    assert transfer_class(group) is TransferClass.ATTENTION
    assert is_attention_spec(group) is True
    assert is_ssm_spec(group) is False
    assert get_representative_spec_type(group) is AttentionSpec


def test_mla_group_is_attention_state():
    """An MLA layer carries attention state, not opaque state."""
    group = make_kv_cache_group(["layer0"], MLAAttentionSpec(**MLA_KWARGS))

    assert transfer_class(group) is TransferClass.ATTENTION
    assert is_attention_spec(group) is True


def test_mamba_group_is_ssm_state():
    """A Mamba group moves as SSM state."""
    group = make_kv_cache_group(
        ["layer0"],
        MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.float16,)),
    )

    assert transfer_class(group) is TransferClass.SSM
    assert is_ssm_spec(group) is True
    assert is_attention_spec(group) is False
