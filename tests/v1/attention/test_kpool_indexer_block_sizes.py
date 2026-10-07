# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for storage-block kernel-block selection in prepare_kernel_block_sizes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends.mla import indexer as indexer_mod
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.utils import prepare_kernel_block_sizes

pytestmark = pytest.mark.cpu_test

# ``index_kpool`` of zai-org/GLM-5.3-Flash: one indexer state pools 4
# compressed states (hence storage_block_size = page * 4).
INDEX_KPOOL = 4

# The hybrid KDA/mamba manager floors at TP8..TP1; 640 is the geometry of the
# original report ("the 640-token block table").
TP_FLOOR_MANAGER_BLOCKS = [640, 1152, 2176, 4352]


def _mock_rocm_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    # Platform predicates are mutually exclusive in production. Override both
    # so these ROCm policy tests do not inherit CUDA capabilities from the CI
    # host running them (same approach as test_deepseek_v4_rocm_adaptive).
    monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(indexer_mod.current_platform, "is_rocm", lambda: True)


def _kpool_storage_spec(manager_block: int) -> MLAAttentionSpec:
    """Real Glm5NextIndexerCache spec for the given cache block size."""
    with set_current_vllm_config(VllmConfig()):
        cache = Glm5NextIndexerCache(
            head_dim=128,
            dtype=torch.bfloat16,
            prefix="model.layers.0.indexer.k_cache_probe",
            cache_config=SimpleNamespace(block_size=manager_block),
            index_kpool=INDEX_KPOOL,
        )
        spec = cache.get_kv_cache_spec(VllmConfig())
    assert isinstance(spec, MLAAttentionSpec)
    return spec


def _prepare(
    spec: MLAAttentionSpec, backends: list[type[AttentionBackend]]
) -> list[int]:
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    attn_groups = [[SimpleNamespace(backend=backend) for backend in backends]]
    return prepare_kernel_block_sizes(kv_cache_config, attn_groups)


@pytest.mark.parametrize("manager_block", TP_FLOOR_MANAGER_BLOCKS)
def test_prepare_uses_storage_block_for_the_kpool_group(
    monkeypatch: pytest.MonkeyPatch, manager_block: int
):
    """The bug: on ROCm, kpool groups must get 128/256 - not the 640-4352
    manager block - as their kernel block size.

    Manager-granular selection is what made the block table address manager
    blocks while the kpool writer and the index-cache gather read pool pages,
    aliasing every cache column past the request's row.
    """
    _mock_rocm_platform(monkeypatch)
    spec = _kpool_storage_spec(manager_block)
    expected_storage = 256 if manager_block % 256 == 0 else 128
    assert spec.storage_block_size == expected_storage

    selected = _prepare(
        spec,
        [ROCMAiterMLASparseBackend, DeepseekV32IndexerBackend, KpoolTailBackend],
    )
    assert selected == [expected_storage]
    assert selected != [manager_block]
    # The hybrid block-table split stays integral.
    assert manager_block % selected[0] == 0


def test_prepare_falls_back_when_storage_block_is_not_supported():
    """No storage block, or a storage block the backends reject (e.g. the
    CUDA indexer's exact ``[64]`` vs a 128-token storage block): selection is
    `select_common_block_size`'s answer, unchanged from no-fix behavior.
    """

    class Fixed64Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [64]

        @staticmethod
        def get_name() -> str:
            return "FIXED64"

    # Storage block the group's backend does not accept -> backend vote (64).
    spec = _kpool_storage_spec(640)
    assert spec.storage_block_size == 128
    assert _prepare(spec, [Fixed64Backend]) == [64]

    # No storage block at all (any non-kpool MLA model) -> backend vote.
    plain_spec = MLAAttentionSpec(
        block_size=640,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    assert plain_spec.storage_block_size is None
    assert _prepare(plain_spec, [Fixed64Backend]) == [64]

    # ...and with backends that accept the manager block, the manager wins,
    # exactly as on main.
    class AcceptAllBackend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(1)]

        @staticmethod
        def get_name() -> str:
            return "ACCEPTALL"

    assert _prepare(plain_spec, [AcceptAllBackend]) == [640]
