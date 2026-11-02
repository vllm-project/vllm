# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``CommonAttentionMetadata.tree_parents`` reaches the GDN spec kernels and
sends the batch past the chain-only fused MTP kernel. CPU only, kernels faked."""

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
from tests.v1.attention.test_gdn_metadata_builder import BLOCK_SIZE, DEVICE
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

PARENTS = torch.tensor([-1, 0, 0, 1, -1, 0, 1, 1], dtype=torch.int32)
NUM_TOKENS = 4


def _vllm_config(method: str, num_speculative_tokens: int = 3):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method=method, num_speculative_tokens=num_speculative_tokens
    )
    return vllm_config


class _Calls:
    def __init__(self):
        self.conv = 0
        self.recurrent = 0


def _layer(
    monkeypatch, calls: _Calls, metadata: GDNAttentionMetadata
) -> QwenGatedDeltaNetAttention:
    """A tiny GDN layer whose spec kernels are fakes that count their calls.

    Built with ``__new__``: the real constructor needs a TP group and weights,
    and these tests only exercise how ``_forward_core`` routes the rows.
    """

    def fake_conv(mixed_qkv_spec, *args, **kwargs):
        calls.conv += 1
        return mixed_qkv_spec + 100

    def fake_recurrent(**kwargs):
        calls.recurrent += 1
        return torch.full((1, kwargs["q"].shape[0], 2, 3), -1.0), None

    monkeypatch.setattr(gdn_layer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        gdn_layer_module, "fused_sigmoid_gating_delta_rule_update", fake_recurrent
    )
    monkeypatch.setattr(gdn_layer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        gdn_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.enable_packed_recurrent_decode = False
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=torch.zeros(5, 1, 4), bias=None)
    layer.kv_cache = (torch.zeros(2, 5, 3), torch.zeros(2, 2, 3, 3))
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.ones(2)
    layer.tp_size = 1
    layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim = (
        1,
        2,
        2,
        3,
    )
    layer.rearrange_mixed_qkv = lambda x: (None, None, None) if x is None else (x, x, x)
    layer.qkv_row_views = layer.rearrange_mixed_qkv
    return layer


def _spec_metadata(parents=None) -> GDNAttentionMetadata:
    """One spec-decode request of NUM_TOKENS rows and nothing else."""
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=NUM_TOKENS,
        num_actual_tokens=NUM_TOKENS,
        spec_query_start_loc=torch.tensor([0, NUM_TOKENS], dtype=torch.int32),
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        spec_tree_parents=parents,
    )


def _build(tree_parents):
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=["layer.0"],
        vllm_config=_vllm_config("ngram"),
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[40, 50], query_lens=[4, 4]), BLOCK_SIZE, DEVICE
    )
    common.tree_parents = tree_parents
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )


def test_builder_hands_the_spec_rows_their_parents():
    assert _build(None).spec_tree_parents is None
    metadata = _build(PARENTS)
    assert metadata.num_spec_decodes == 2
    assert torch.equal(metadata.spec_tree_parents.cpu(), PARENTS)


def test_layer_passes_the_parents_to_both_spec_kernels(monkeypatch):
    seen = {}
    parents = torch.tensor([-1, 0, 0, 2], dtype=torch.int32)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, _spec_metadata(parents))
    conv, recurrent = (
        gdn_layer_module.causal_conv1d_update,
        gdn_layer_module.fused_sigmoid_gating_delta_rule_update,
    )

    def record_conv(*args, **kwargs):
        seen["conv"] = kwargs.get("parents")
        return conv(*args, **kwargs)

    def record_recurrent(**kwargs):
        seen["recurrent"] = kwargs.get("parents")
        return recurrent(**kwargs)

    monkeypatch.setattr(gdn_layer_module, "causal_conv1d_update", record_conv)
    monkeypatch.setattr(
        gdn_layer_module, "fused_sigmoid_gating_delta_rule_update", record_recurrent
    )
    n = NUM_TOKENS
    b, a = torch.zeros(n, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(n, 5), b=b, a=a, core_attn_out=torch.zeros(n, 2, 3)
    )
    assert (calls.conv, calls.recurrent) == (1, 1)
    assert seen["conv"] is parents and seen["recurrent"] is parents


def test_fused_mtp_decode_is_skipped_for_a_tree(monkeypatch):
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    state_dtype = next(iter(gdn_layer_module.FUSED_GDN_STATE_DTYPES))
    layer.kv_cache = (torch.zeros(1), torch.zeros(1, dtype=state_dtype))
    layer.gdn_decode_kernel = "cuda"
    layer.num_v_heads = layer.num_k_heads = 2
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata()) is True
    tree = _spec_metadata(torch.tensor([-1, 0, 0, 1], dtype=torch.int32))
    assert layer._can_use_fused_gdn_mtp_decode(tree) is False
