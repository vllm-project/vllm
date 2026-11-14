# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


@pytest.fixture(scope="module")
def pipeline_module():
    return import_module("vllm.models.deepseek_v41.nvidia.pipeline")


@pytest.fixture
def flash_sharing_config():
    return DeepseekV41Config(
        text_config={
            "num_hidden_layers": 40,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 32,
        },
    )


@pytest.mark.parametrize("cut", range(1, 40))
def test_every_pipeline_cut_preserves_live_sharing_sources(
    pipeline_module, flash_sharing_config, cut
):
    """Main KV, indexer K, top-k and candidate state have different lifetimes."""
    config = flash_sharing_config
    (boundary,) = pipeline_module.get_pipeline_boundaries(config, [(0, cut), (cut, 40)])
    assert boundary.kv_source == next(
        (s for s, end in zip([2, 8, 14, 20], [8, 14, 20, 40]) if s < cut < end),
        None,
    )
    index_sources = config.index_source_layer_ids
    assert boundary.index_source == next(
        (s for s, end in zip(index_sources, index_sources[1:] + [40]) if s < cut < end),
        None,
    )
    assert boundary.candidate_source == (20 if 20 < cut <= 36 else None)
    # Only index sources 24-36 read another layer's indexer K cache (layer 20's).
    assert boundary.indexer_k_source == (20 if 20 < cut <= 36 else None)


def test_pipeline_state_survives_multiple_hops(pipeline_module, flash_sharing_config):
    boundaries = pipeline_module.get_pipeline_boundaries(
        flash_sharing_config, [(0, 10), (10, 12), (12, 25), (25, 40)]
    )
    assert boundaries == (
        pipeline_module.PipelineBoundary(8, 8),
        pipeline_module.PipelineBoundary(8, 8),
        pipeline_module.PipelineBoundary(20, 24, 20, 20),
    )


@pytest.mark.parametrize(
    "ranges",
    [[], [(0, 0), (0, 40)], [(0, 10), (11, 40)], [(0, 10), (9, 40)], [(0, 39)]],
)
def test_invalid_pipeline_partition_rejected(
    pipeline_module, flash_sharing_config, ranges
):
    with pytest.raises(ValueError, match="partition all layers"):
        pipeline_module.get_pipeline_boundaries(flash_sharing_config, ranges)


def test_pipeline_cut_validates_source_topology(pipeline_module, flash_sharing_config):
    flash_sharing_config.index_source_layer_ids.remove(8)
    with pytest.raises(ValueError, match="publish index keys"):
        pipeline_module.get_pipeline_boundaries(
            flash_sharing_config, [(0, 10), (10, 40)]
        )


@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        mhc_stream=None,
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix
