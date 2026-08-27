# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash FULL-replay metadata-rebuild skip must stay fail-closed.

Under FULL cudagraph replay the freshly built draft attention metadata is
discarded, so the rebuild exists only for its builder-state side effects;
skipping it is legal exactly when every builder declares build() free of
such side effects via ``supports_skip_draft_rebuild``.
"""

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    all_draft_builders_skip_safe,
)


def _group(*builders):
    return SimpleNamespace(metadata_builders=list(builders))


def _builder(skip_safe: bool):
    return SimpleNamespace(supports_skip_draft_rebuild=skip_safe)


def test_all_builders_skip_safe_requires_every_builder():
    assert all_draft_builders_skip_safe([[_group(_builder(True))]])
    assert all_draft_builders_skip_safe(
        [[_group(_builder(True), _builder(True))], [_group(_builder(True))]]
    )
    # One unsafe builder anywhere blocks the skip.
    assert not all_draft_builders_skip_safe(
        [[_group(_builder(True))], [_group(_builder(False))]]
    )


def test_no_builders_fails_closed():
    assert not all_draft_builders_skip_safe([])
    assert not all_draft_builders_skip_safe([[], []])
    assert not all_draft_builders_skip_safe([[_group()]])


def test_base_builder_defaults_to_unsafe():
    assert AttentionMetadataBuilder.supports_skip_draft_rebuild is False


def _triton_builder(rswa_window: int | None) -> TritonAttentionMetadataBuilder:
    """Real CPU construction with the smallest config __init__ reads."""
    model_config = SimpleNamespace(
        rswa_window=rswa_window,
        get_num_attention_heads=lambda parallel_config: 2,
        get_num_kv_heads=lambda parallel_config: 2,
        get_head_size=lambda: 64,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    return TritonAttentionMetadataBuilder(
        kv_cache_spec=SimpleNamespace(block_size=16),
        layer_names=[],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )


def test_triton_builder_skip_safe_iff_rswa_inactive():
    # build() restages persistent state only on the R-SWA branch, so the
    # constructed builder is skip-safe exactly when rswa_window is unset.
    assert _triton_builder(rswa_window=None).supports_skip_draft_rebuild
    assert not _triton_builder(rswa_window=1024).supports_skip_draft_rebuild
