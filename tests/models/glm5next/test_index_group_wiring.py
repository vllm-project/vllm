# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash hands its sparse MLA layers one shared index-group builder.

Without it each layer builds a private builder whose workspace has one row per
batched token, instead of one row per decode token like the DSA models.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.config import AttentionConfig, CacheConfig
from vllm.models.glm5next.common import attention as glm_attention
from vllm.models.glm5next.common import model as glm_model
from vllm.models.glm5next.common import mtp as glm_mtp
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

HIDDEN = 8
MAX_NUM_BATCHED_TOKENS = 32
MAX_NUM_SEQS = 4
INDEX_TOPK, INDEX_KPOOL = 128, 4
# round_up(index_topk + index_kpool - 1, 128)
TOPK_WIDTH = 256


pytestmark = pytest.mark.usefixtures("default_vllm_config", "dist_init")


def _config(**overrides) -> Glm5NextTextConfig:
    kwargs = dict(
        hidden_size=HIDDEN,
        intermediate_size=16,
        vocab_size=32,
        pad_token_id=0,
        num_hidden_layers=4,
        first_k_dense_replace=1,
        n_routed_experts=4,
        layer_types=["deepseek_sparse_attention", "linear_attention"] * 2,
        mhc=False,
        mla_use_nope=True,
        index_topk=INDEX_TOPK,
        index_kpool=INDEX_KPOOL,
    )
    return Glm5NextTextConfig(**(kwargs | overrides))


def _vllm_config(config, num_speculative_tokens: int | None = None):
    speculative_config = None
    if num_speculative_tokens is not None:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
            draft_model_config=SimpleNamespace(hf_text_config=config),
        )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        ),
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        speculative_config=speculative_config,
        use_v2_model_runner=False,
    )


class _Layer(nn.Module):
    """Decoder layer stand-in: the real ones need a GPU attention backend."""

    instances: list["_Layer"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        _Layer.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_layers():
    _Layer.instances = []


@pytest.mark.parametrize(
    ("num_speculative_tokens", "decode_rows"),
    [(None, MAX_NUM_SEQS), (3, MAX_NUM_SEQS * 4)],
)
def test_model_builds_one_decode_sized_builder_for_all_layers(
    monkeypatch, num_speculative_tokens, decode_rows
):
    config = _config()
    monkeypatch.setattr(glm_model, "Glm5NextDecoderLayer", _Layer)

    glm_model.Glm5NextModel(vllm_config=_vllm_config(config, num_speculative_tokens))

    buffer = _Layer.instances[0].kwargs["topk_indices_buffer"]
    builders = {id(layer.kwargs["index_group_builder"]) for layer in _Layer.instances}
    assert len(_Layer.instances) == config.num_hidden_layers
    assert len(builders) == 1
    builder = _Layer.instances[0].kwargs["index_group_builder"]
    assert isinstance(builder, SparseMLAIndexGroupBuilder)
    assert builder.logical_topk_indices is buffer
    assert builder.max_decode_rows == decode_rows


def test_mtp_layer_builds_its_own_decode_sized_builder(monkeypatch):
    config = _config()
    monkeypatch.setattr(glm_mtp, "Glm5NextDecoderLayer", _Layer)
    monkeypatch.setattr(glm_mtp, "SharedHead", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(glm_mtp.current_platform, "device_type", "cpu")

    glm_mtp.Glm5NextMultiTokenPredictorLayer(
        _vllm_config(config, num_speculative_tokens=3), "model.layers.3"
    )

    (layer,) = _Layer.instances
    builder = layer.kwargs["index_group_builder"]
    assert isinstance(builder, SparseMLAIndexGroupBuilder)
    assert builder.logical_topk_indices is layer.kwargs["topk_indices_buffer"]
    assert builder.max_decode_rows == MAX_NUM_SEQS * 4


def test_decoder_layer_hands_the_builder_to_the_mla_modules(monkeypatch):
    """The builder reaches the MLA layer's modules, from where the impl takes it."""
    config = _config(
        num_hidden_layers=2,
        layer_types=["deepseek_sparse_attention", "linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=32,
        qk_rope_head_dim=0,
        v_head_dim=32,
        q_lora_rank=16,
        kv_lora_rank=32,
        index_n_heads=2,
        index_head_dim=128,
        indexer_rope_interleave=True,
    )
    mla_modules = []

    class _Wrapper(nn.Module):
        """MLA wrapper stand-in: the real one needs a GPU attention backend."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            mla_modules.append(args[8])

    monkeypatch.setattr(glm_attention, "MultiHeadLatentAttentionWrapper", _Wrapper)
    # The top-k op needs DeepGEMM or AITER kernels and plays no part here.
    monkeypatch.setattr(
        glm_attention, "SparseAttnIndexerKpool", lambda *a, **k: nn.Module()
    )
    builder = SparseMLAIndexGroupBuilder(torch.empty(1, TOPK_WIDTH, dtype=torch.int32))

    glm_model.Glm5NextDecoderLayer(
        SimpleNamespace(
            cache_config=CacheConfig(),
            quant_config=None,
            model_config=SimpleNamespace(max_model_len=1024),
            attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=5),
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
            kernel_config=SimpleNamespace(enable_jit_warmup=False),
        ),
        config,
        layer_idx=0,
        topk_indices_buffer=builder.logical_topk_indices,
        index_group_builder=builder,
    )

    (modules,) = mla_modules
    assert modules.index_group_builder is builder
