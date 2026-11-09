# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3.1 DSA wiring: config gates, weight renames, token sharding."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm path")


def test_indexer_backend_uses_64_token_kernel_blocks():
    from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerBackend

    # FP4 (68 B/token) and FP8 (132 B/token) packed caches both tile by 64.
    for spec in (None, SimpleNamespace(head_size=68), SimpleNamespace(head_size=132)):
        assert KimiK3IndexerBackend.get_supported_kernel_block_sizes(spec) == [64]


def _indexer_config(model_type, dsa_config):
    text = SimpleNamespace(model_type=model_type, dsa_config=dsa_config)
    return SimpleNamespace(
        attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda _: "mxfp4"),
        model_config=SimpleNamespace(hf_text_config=text, hf_config=text),
    )


@pytest.fixture
def cdna4_with_aiter(monkeypatch):
    import vllm.platforms.rocm as rocm
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm, "get_cdna_version", lambda: 4)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True))


@pytest.mark.parametrize("model_type", ["kimi_linear", "kimi_k3", "kimi_k31"])
def test_mxfp4_indexer_accepts_kimi_dsa(cdna4_with_aiter, model_type):
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    assert dsa_indexer_uses_fp4(_indexer_config(model_type, dsa_config={}))


@pytest.mark.parametrize(
    "model_type,dsa_config",
    [("kimi_k3", None), ("deepseek_v32", {}), ("glm_moe_dsa", {})],
)
def test_mxfp4_indexer_rejects_other_models(cdna4_with_aiter, model_type, dsa_config):
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    with pytest.raises(ValueError, match="only supported for"):
        dsa_indexer_uses_fp4(_indexer_config(model_type, dsa_config))


def test_rocm_sparse_indexer_refuses_dcp():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_aiter_sparse_attn_indexer,
    )

    empty = torch.empty(0)
    with pytest.raises(NotImplementedError, match="does not support DCP"):
        rocm_aiter_sparse_attn_indexer(
            empty, "layer", empty, empty, empty, empty, 128, None, 2048, 128,
            1024, 0, None, dcp_rank=1, dcp_world_size=2,
        )  # fmt: skip


@pytest.mark.parametrize(
    "checkpoint,param",
    [
        ("m.layers.3.self_attn.indexer.wq_b.weight", "dsa_indexer.wq_index.weight"),
        ("m.layers.3.self_attn.indexer.wk.weight", "dsa_indexer.wk_index.weight"),
        (
            "m.layers.3.self_attn.indexer.weights_proj.weight",
            "dsa_indexer.wv_index.weight",
        ),
        (
            "m.layers.3.self_attn.indexer.k_norm.weight",
            "dsa_indexer.k_index_norm.weight",
        ),
        ("m.layers.3.self_attn.indexer.k_norm.bias", "dsa_indexer.k_index_norm.bias"),
        ("m.layers.3.self_attn.indexer.other.weight", "dsa_indexer.other.weight"),
    ],
)
def test_dsa_indexer_checkpoint_rename(checkpoint, param):
    from vllm.models.kimi_k3.amd.linear import _DSA_RENAME_PARAMS_MAPPING

    # Same first-match loop as KimiLinearModel.load_weights.
    name = checkpoint
    for param_name, weight_name in _DSA_RENAME_PARAMS_MAPPING:
        if weight_name in name:
            name = name.replace(weight_name, param_name)
            break
    assert name == f"m.layers.3.self_attn.{param}"


class _FakeTokenGroup:
    """Stands in for the DCP group: all_gather runs every rank's projection."""

    def __init__(self, layer, world_size, rank):
        self.layer, self.world_size, self.rank_in_group = layer, world_size, rank

    def all_gather(self, local, dim):
        assert dim == 0
        x = self.layer._full_input
        parts = []
        for rank in range(self.world_size):
            self.rank_in_group = rank
            parts.append(self.layer._project_local(x)[0])
        return torch.cat(parts, dim=0)


@pytest.fixture
def routed_input_linear(default_vllm_config, dist_init):
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

    torch.manual_seed(0)
    layer = KimiRoutedInputLinear(
        256, 64, bias=False, params_dtype=torch.bfloat16, disable_tp=True
    ).cuda()
    layer.weight.data.copy_(torch.randn_like(layer.weight) / 16)
    return layer


@pytest.mark.parametrize("tokens", [64, 61, 3])
def test_routed_input_token_sharding_matches_replicated(
    routed_input_linear, monkeypatch, tokens
):
    layer = routed_input_linear
    monkeypatch.setattr(layer, "min_sharded_tokens", 1)
    x = torch.randn(tokens, 256, dtype=torch.bfloat16, device="cuda")
    layer._full_input = x
    layer._token_group = _FakeTokenGroup(layer, world_size=8, rank=0)

    out, bias = layer(x)

    assert bias is None
    assert out.shape == (tokens, 64)
    torch.testing.assert_close(out, F.linear(x, layer.weight))


def test_routed_input_small_batch_stays_replicated(routed_input_linear):
    layer = routed_input_linear
    x = torch.randn(16, 256, dtype=torch.bfloat16, device="cuda")

    class _Unused:
        world_size, rank_in_group = 8, 0

        def all_gather(self, *_):
            raise AssertionError("a batch below min_sharded_tokens must not gather")

    layer._token_group = _Unused()
    out, _ = layer(x)
    torch.testing.assert_close(out, F.linear(x, layer.weight))


def test_routed_input_sharding_off_outside_tp8_dcp8(default_vllm_config):
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

    layer = KimiRoutedInputLinear.__new__(KimiRoutedInputLinear)
    layer.input_size, layer.output_size = 7168, 3584
    # The default config is TP1/DCP1; sharding needs identical TP8/DCP8 groups.
    layer.enable_token_sharding()
    assert layer._token_group is None


def test_latent_experts_count_packed_fp4_width():
    from vllm.models.kimi_k3.amd.latent_moe_runner import KimiLatentAiterExperts

    experts = KimiLatentAiterExperts.__new__(KimiLatentAiterExperts)
    w1 = torch.empty(4, 512, 128, dtype=torch.uint8)
    w2 = torch.empty(4, 256, 128, dtype=torch.uint8)
    topk_ids = torch.zeros(3, 2, dtype=torch.int32)
    bf16 = torch.empty(3, 256, dtype=torch.bfloat16)
    packed = torch.empty(3, 128, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    assert experts.moe_problem_size(bf16, w1, w2, topk_ids) == (4, 3, 512, 256, 2)
    # Two FP4 values per byte: the prepared input still spans 256 columns.
    assert experts.moe_problem_size(packed, w1, w2, topk_ids) == (4, 3, 512, 256, 2)
