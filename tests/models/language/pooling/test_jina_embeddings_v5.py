# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backbone validation for Jina Embeddings V5.

The V5 family ships two backbones under one `architectures` entry: `-small` is
a Qwen3 decoder, while `-nano` is a bidirectional EuroBERT encoder. Upstream
ships a separate `configuration_*.py` per repository, so the only signal
distinguishing them is `is_decoder`, which the encoder variant sets to False.
`HFConfigParser` parses encoder checkpoints into `EuroBertConfig` without
`auto_map`, `JinaEmbeddingsV5ModelConfig` sets `is_causal=False` for
`ModelConfig.attn_type`, and `JinaEmbeddingsV5Model` dispatches to
`JinaEmbeddingsV5EncoderModel` (`EuroBertModel`).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import EuroBertConfig, EuroBertModel, PreTrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    JinaEmbeddingsV5ModelConfig,
)
from vllm.model_executor.models.jina import JinaEmbeddingsV5EncoderModel
from vllm.model_executor.models.transformers import TransformersEmbeddingModel

from .embed_utils import check_embeddings_close


def _model_config(hf_config: PreTrainedConfig) -> ModelConfig:
    """Minimal stand-in for ModelConfig; only hf_config is read."""
    return cast(ModelConfig, SimpleNamespace(hf_config=hf_config))


@pytest.mark.cpu_test
def test_registered_for_the_architecture():
    """The handler only runs if it is wired to the architecture name."""
    assert MODELS_CONFIG_MAP["JinaEmbeddingsV5Model"] is JinaEmbeddingsV5ModelConfig


@pytest.mark.cpu_test
def test_encoder_backbone_enables_bidirectional_attention():
    """An encoder checkpoint (`is_decoder=False`) sets `is_causal=False` so
    `ModelConfig.attn_type` resolves to `encoder_only`.
    """
    hf_config = PreTrainedConfig(is_decoder=False)

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(hf_config))

    assert hf_config.is_causal is False


@pytest.mark.cpu_test
def test_supported_decoder_backbone_is_accepted():
    """The Qwen3-based variants must keep loading.

    `-small` omits `is_decoder` entirely, so an absent attribute has to be
    treated as a decoder. The first assertion pins that assumption: if
    PreTrainedConfig ever gains an `is_decoder=False` default, this fails here
    rather than misrouting `-small` to the encoder.
    """
    absent = PreTrainedConfig()
    assert not hasattr(absent, "is_decoder")

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(absent))
    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(
        _model_config(PreTrainedConfig(is_decoder=True))
    )


def _encoder_config_kwargs(vocab_size: int = 128) -> dict:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 3,
        "tie_word_embeddings": False,
    }


def _encoder_config_dict(vocab_size: int = 128) -> dict:
    return {
        **_encoder_config_kwargs(vocab_size=vocab_size),
        "architectures": ["JinaEmbeddingsV5Model"],
        "auto_map": {
            "AutoConfig": "configuration_jina_embeddings_v5.JinaEmbeddingsV5Config",
            "AutoModel": "modeling_jina_embeddings_v5.JinaEmbeddingsV5Model",
        },
        "model_type": "jina_embeddings_v5",
        "is_decoder": False,
        "dtype": "bfloat16",
    }


@pytest.mark.cpu_test
def test_encoder_config_uses_eurobert_without_auto_map(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_encoder_config_dict()))

    for trust_remote_code in (True, False):
        mc = ModelConfig(
            str(tmp_path),
            trust_remote_code=trust_remote_code,
            hf_overrides={"jina_task": "clustering"},
        )
        assert type(mc.hf_config) is EuroBertConfig
        assert not hasattr(mc.hf_config, "auto_map")
        assert mc.hf_config.is_causal is False
        assert mc.hf_config.jina_task == "clustering"
        assert mc.attn_type == "encoder_only"
        assert mc.pooler_config is not None
        assert mc.pooler_config.seq_pooling_type == "LAST"

    assert JinaEmbeddingsV5EncoderModel.default_seq_pooling_type == "LAST"


def test_encoder_backbone_merges_lora_with_last_pooling(
    vllm_runner, tmp_path: Path, monkeypatch
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import TokenizersBackend

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    adapter_ckpt = tmp_path / "with_adapter"
    merged_ckpt = tmp_path / "pre_merged"
    adapter_ckpt.mkdir()
    merged_ckpt.mkdir()

    vocab = {f"tok{i}": i for i in range(64)}
    vocab["<unk>"] = 0
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    for dest in (adapter_ckpt, merged_ckpt):
        TokenizersBackend(tokenizer_object=tok, unk_token="<unk>").save_pretrained(dest)

    torch.manual_seed(0)
    cfg = EuroBertConfig(**_encoder_config_kwargs(vocab_size=64), dtype=torch.bfloat16)
    EuroBertModel(cfg).to(torch.bfloat16).save_pretrained(adapter_ckpt)

    raw_cfg = _encoder_config_dict(vocab_size=64)
    (adapter_ckpt / "config.json").write_text(json.dumps(raw_cfg))
    (merged_ckpt / "config.json").write_text(json.dumps(raw_cfg))

    base_weights = load_file(adapter_ckpt / "model.safetensors")
    rank, alpha = 4, 8
    scaling = alpha / rank
    adapter_weights: dict[str, torch.Tensor] = {}
    merged_weights = dict(base_weights)
    target_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    torch.manual_seed(7)
    for name, weight in base_weights.items():
        if name.endswith(target_suffixes):
            out_dim, in_dim = weight.shape
            lora_a = torch.randn(rank, in_dim, dtype=torch.bfloat16) * 0.1
            lora_b = torch.randn(out_dim, rank, dtype=torch.bfloat16) * 0.1
            prefix = f"base_model.model.{name.removesuffix('.weight')}"
            adapter_weights[f"{prefix}.lora_A.weight"] = lora_a
            adapter_weights[f"{prefix}.lora_B.weight"] = lora_b
            merged_weights[name] = weight + (lora_b @ lora_a) * scaling

    adapter_dir = adapter_ckpt / "adapters" / "retrieval"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": alpha})
    )
    save_file(adapter_weights, adapter_dir / "adapter_model.safetensors")
    save_file(merged_weights, merged_ckpt / "model.safetensors")

    prompts = ["tok1 tok2 tok3 tok4", "tok5 tok6 tok7"]
    with vllm_runner(
        str(adapter_ckpt),
        runner="pooling",
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=True,
    ) as llm_adapter:
        mc = llm_adapter.llm.llm_engine.model_config
        assert mc.attn_type == "encoder_only"
        assert mc.pooler_config is not None
        assert mc.pooler_config.seq_pooling_type == "LAST"
        checks = llm_adapter.apply_model(
            lambda m: (
                isinstance(m, JinaEmbeddingsV5EncoderModel),
                isinstance(m, TransformersEmbeddingModel),
                sum(isinstance(sub, EncoderOnlyAttention) for sub in m.modules()),
            )
        )
        assert checks == [(True, True, 2)]
        adapter_embeds = llm_adapter.embed(prompts)

    with vllm_runner(
        str(merged_ckpt),
        runner="pooling",
        dtype="bfloat16",
        enforce_eager=True,
    ) as llm_merged:
        merged_embeds = llm_merged.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=merged_embeds,
        embeddings_1_lst=adapter_embeds,
        name_0="pre_merged",
        name_1="runtime_lora_merged",
        tol=1e-4,
    )

    with vllm_runner(
        str(adapter_ckpt),
        runner="pooling",
        dtype="bfloat16",
        enforce_eager=True,
        hf_overrides={"jina_task": "text-matching"},
    ) as llm_unmerged:
        unmerged_embeds = llm_unmerged.embed(prompts)

    delta = (
        (torch.tensor(adapter_embeds) - torch.tensor(unmerged_embeds))
        .abs()
        .max()
        .item()
    )
    assert delta > 1e-2
