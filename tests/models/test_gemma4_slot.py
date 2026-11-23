# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4ForSlotDecision's config hook and model state."""

import pytest
import torch
from transformers import Gemma4Config

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.gemma4_slot import Gemma4SlotDecisionModelState
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.cpu_test

OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "sliding_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}


def tiny_gemma4_config(
    moe: bool, overrides: dict | None = OVERRIDES, **config
) -> Gemma4Config:
    """A small Gemma 4 with the layout of bison-1, or groundhog-1 if `moe`."""
    text_config = {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        # Builds per_layer_config for the full layers, as in the checkpoints.
        "global_head_dim": 128,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        # Shorter than the test prompts, so the window matters.
        "sliding_window": 128,
        "final_logit_softcapping": 30.0,
        "hidden_size_per_layer_input": 0,
        "use_bidirectional_attention": "vision",
        # As in the checkpoints, far above the trained length.
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "dtype": "float32",
    }
    if moe:
        text_config.update(
            enable_moe_block=True,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=128,
            intermediate_size=128,
        )
    vision_config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
    }
    if overrides is not None:
        config["baseten_modeling_overrides"] = overrides
    return Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=text_config,
        vision_config=vision_config,
        audio_config=None,
        tie_word_embeddings=True,
        **config,
    )


def make_vllm_config(
    tmp_path, overrides=OVERRIDES, head_dtype=None, **engine_args
) -> VllmConfig:
    """The engine config of a checkpoint with only a config.json."""
    tiny_gemma4_config(False, overrides, head_dtype=head_dtype).save_pretrained(
        tmp_path
    )
    return EngineArgs(
        model=str(tmp_path), skip_tokenizer_init=True, **engine_args
    ).create_engine_config()


@pytest.mark.parametrize(
    "head_dtype,engine_args,expected",
    [
        (None, {}, ("float32", "float32")),
        ("model", {}, ("model", "model")),
        # A head dtype on the text config wins.
        (
            None,
            {"hf_overrides": {"text_config": {"head_dtype": "model"}}},
            ("float32", "model"),
        ),
    ],
)
def test_the_inner_language_model_gets_the_head_dtype(
    tmp_path, head_dtype, engine_args, expected
):
    config = make_vllm_config(tmp_path, head_dtype=head_dtype, **engine_args)
    model_config = config.model_config
    text_config = model_config.hf_config.get_text_config()
    inner = config.with_hf_config(text_config, ["Gemma4ForCausalLM"]).model_config
    dtypes = {"float32": torch.float32, "model": model_config.dtype}
    assert (model_config.head_dtype, inner.head_dtype) == tuple(
        dtypes[d] for d in expected
    )


@pytest.mark.parametrize(
    "model,moe", [("baselabs/bison-1", False), ("baselabs/groundhog-1", True)]
)
def test_published_configs(model, moe):
    config = EngineArgs(model=model, skip_tokenizer_init=True).create_engine_config()
    model_config = config.model_config
    assert model_config.architecture == "Gemma4ForSlotDecision"
    assert model_config.max_model_len == 131072
    assert model_config.head_dtype == torch.float32
    assert bool(model_config.hf_text_config.enable_moe_block) == moe
    assert config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert config.scheduler_config.scheduler_cls == (
        "vllm.v1.core.sched.prefix_lm_scheduler.PrefixLMScheduler"
    )
    assert config.cache_config.enable_prefix_caching


def test_stock_gemma4_is_not_routed(tmp_path):
    config = make_vllm_config(tmp_path, overrides=None)
    assert config.model_config.architecture == "Gemma4ForConditionalGeneration"


@pytest.mark.parametrize(
    "overrides,engine_args,match",
    [
        (
            {
                **OVERRIDES,
                "attention": {
                    "full_attention": "block_bidirectional",
                    "sliding_attention": "block_bidirectional",
                },
            },
            {},
            "unsupported baseten_modeling_overrides",
        ),
        (
            OVERRIDES,
            {"disable_hybrid_kv_cache_manager": True},
            "disable-hybrid-kv-cache-manager",
        ),
    ],
)
def test_config_rejects_unsupported_overrides_and_features(
    tmp_path, overrides, engine_args, match
):
    with pytest.raises(ValueError, match=match):
        make_vllm_config(tmp_path, overrides=overrides, **engine_args)


def test_config_keeps_the_hybrid_kv_cache_manager_on(tmp_path, monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: False)
    with pytest.raises(ValueError, match="Hybrid KV cache manager was explicitly"):
        make_vllm_config(tmp_path)


def test_model_state_keeps_prefills_out_of_full_cudagraphs(tmp_path):
    config = make_vllm_config(tmp_path, compilation_config={"cudagraph_mode": "FULL"})
    support = Gemma4SlotDecisionModelState(
        config, torch.nn.Module(), None, torch.device("cpu")
    ).get_additional_cg_support()
    assert (
        config.compilation_config.resolve_cudagraph_mode_and_sizes(*support)
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
