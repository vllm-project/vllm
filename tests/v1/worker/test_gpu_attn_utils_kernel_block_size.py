# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.attn_utils as attn_utils
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.utils import AttentionGroup


def _backend(name: str, sizes: list[int]):
    class FakeBackend:
        @staticmethod
        def get_name() -> str:
            return name

        @staticmethod
        def full_cls_name() -> tuple[str, str]:
            return (__name__, name)

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
            return sizes

        @staticmethod
        def supports_batch_invariance() -> bool:
            return True

    return FakeBackend


# The target's backend only runs 128-token pages (like MiniMax-M3 sparse
# attention); the draft's backend also runs the 512-token manager block.
TARGET_BACKEND = _backend("TARGET", [128])
DRAFT_BACKEND = _backend("DRAFT", [128, 512])


@pytest.fixture
def init(monkeypatch):
    layers = {
        "target": SimpleNamespace(get_attn_backend=lambda: TARGET_BACKEND, num_heads=8),
        "draft": SimpleNamespace(get_attn_backend=lambda: DRAFT_BACKEND, num_heads=4),
    }
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda _config, _type, names=None: {
            n: layers[n] for n in (names if names is not None else layers)
        },
    )
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda _c: {})
    monkeypatch.setattr(
        attn_utils,
        "get_kv_sharing_fast_prefill_eligible_layers",
        lambda _c, _d: set(),
    )
    monkeypatch.setattr(attn_utils, "get_attn_cg_support", lambda _g, _c: None)
    monkeypatch.setattr(attn_utils, "get_num_ubatches", lambda _p: 1)
    monkeypatch.setattr(
        AttentionGroup,
        "create_metadata_builders",
        lambda self, vllm_config, device, kernel_block_size=None, **_: setattr(
            self, "kernel_block_size", kernel_block_size
        ),
    )

    def run(**kwargs):
        kv_cache_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["target", "draft"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=512,
                        num_kv_heads=1,
                        head_size=128,
                        dtype=torch.bfloat16,
                    ),
                )
            ],
        )
        vllm_config = SimpleNamespace(parallel_config=None)
        return attn_utils.init_attn_backend(
            kv_cache_config, vllm_config, torch.device("cpu"), **kwargs
        )

    return run


def test_draft_reuses_target_kernel_block_size(init):
    # The target runner sees every layer of the group, so the cache is laid
    # out in 128-token kernel blocks.
    groups, _, target_sizes = init(draft_layer_names={"draft"})
    assert target_sizes == [128]
    assert {g.kernel_block_size for g in groups[0]} == {128}

    # Seen alone, the draft's backend would pick 512 and read that cache with
    # the wrong page size.
    _, _, draft_only = init(active_layer_names={"draft"})
    assert draft_only == [512]

    # Given the target's sizes, the draft keeps the cache's 128-token pages.
    draft_groups, _, draft_sizes = init(
        active_layer_names={"draft"}, kernel_block_sizes=target_sizes
    )
    assert draft_sizes == [128]
    assert [g.kernel_block_size for g in draft_groups[0]] == [128]


def test_unset_kernel_block_size_falls_back_to_selection(init):
    _, _, sizes = init(active_layer_names={"draft"}, kernel_block_sizes=[None])
    assert sizes == [512]
