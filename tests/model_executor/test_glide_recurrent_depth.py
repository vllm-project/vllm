# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiDE recurrent depth and its isolation from the shared Qwen backbone."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from vllm.config import ModelConfig
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams

NUM_LAYERS = 6
BLOCKS = [[0, 1], [2, 4]]
MIDLOOP = {
    "conv": "A",
    "strategy": "euler",
    "blocks": BLOCKS,
    "n_gate": 0,
    "n_lora": 0,
    "passes": 3,
}


def checkpoint_config(
    model_dir: Path, midloop: dict, *, glide: bool = True
) -> Qwen3_5Config:
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration" if glide else "Qwen3_5Model"],
        text_config={
            "vocab_size": 256,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 32,
            "layer_types": ["full_attention"] * NUM_LAYERS,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [8, 8, 16],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": 128,
            "deepstack_visual_indexes": [],
        },
        image_token_id=248,
        video_token_id=249,
        vision_start_token_id=250,
        vision_end_token_id=251,
    )
    config.save_pretrained(model_dir)
    decision = {
        "format_version": 1,
        "attention_mode": "noncausal_full_attention",
        "codes": ["A", "B"],
        "token_ids": [32, 33],
        "midloop": midloop,
    }
    if glide:
        decision["prompt_format"] = "glide-v1"
    (model_dir / "decision_config.json").write_text(json.dumps(decision))
    return config


def model_config(model_dir: Path, midloop: dict, *, glide: bool = True) -> ModelConfig:
    checkpoint_config(model_dir, midloop, glide=glide)
    return ModelConfig(
        model=str(model_dir),
        skip_tokenizer_init=True,
        max_model_len=32,
        runner="pooling",
        dtype="float32",
    )


def test_midloop_is_applied_only_to_glide(tmp_path):
    config = model_config(tmp_path / "glide", MIDLOOP)
    assert config.hf_text_config.recurrent_depth == {
        "blocks": [[0, 1], [2, 4]],
        "passes": 3,
    }
    assert config.hf_text_config.is_causal is False
    for name, glide, passes in [("single-pass", True, 1), ("d3", False, 3)]:
        plain = model_config(
            tmp_path / name, {**MIDLOOP, "passes": passes}, glide=glide
        )
        assert not hasattr(plain.hf_text_config, "recurrent_depth")


@pytest.mark.parametrize(
    "change",
    [
        {"strategy": "anderson"},
        {"n_lora": 8},
        {"blocks": [[0, 3], [3, 4]]},
        {"blocks": [[4, NUM_LAYERS]]},
        {"passes": -1},
    ],
)
def test_unsupported_midloop_is_rejected(tmp_path, change):
    with pytest.raises(ValueError, match="midloop"):
        model_config(tmp_path, {**MIDLOOP, **change})


@pytest.mark.parametrize("glide", [False, True])
@pytest.mark.parametrize("passes", [1, 3])
def test_checkpoint_logits_match_real_layers(tmp_path, vllm_runner, glide, passes):
    """LLM must repeat real layers only for GLiDE, leaving D3's readout unchanged."""
    torch.manual_seed(0)
    config = checkpoint_config(tmp_path, {**MIDLOOP, "passes": passes}, glide=glide)
    reference = Qwen3_5ForConditionalGeneration(config).eval()
    reference.set_attn_implementation("eager")
    weights = {
        name if glide else name.removeprefix("model."): tensor.contiguous()
        for name, tensor in reference.state_dict().items()
        if not name.startswith("lm_head.")
    }
    save_file(weights, tmp_path / "model.safetensors")
    readout = torch.randn(2, config.text_config.hidden_size) / 16
    save_file({"weight": readout}, tmp_path / "readout.safetensors")

    token_ids = [3, 7, 5, 8, 9]
    with torch.inference_mode():
        text_model = reference.model.language_model.to(
            device=current_platform.device_type, dtype=torch.bfloat16
        )
        ids = torch.tensor([token_ids], device=current_platform.device_type)
        hidden_states = text_model.embed_tokens(ids)
        positions = torch.arange(len(token_ids), device=ids.device).unsqueeze(0)
        position_embeddings = text_model.rotary_emb(hidden_states, positions)
        for first, last in [*BLOCKS, [5, 5]]:
            block_passes = passes if glide and last != 5 else 1
            for _ in range(block_passes):
                block_output = hidden_states
                for layer in text_model.layers[first : last + 1]:
                    block_output = layer(block_output, position_embeddings)
                hidden_states = torch.lerp(
                    hidden_states.float(), block_output.float(), 1.0 / block_passes
                ).to(hidden_states.dtype)
        expected = torch.nn.functional.linear(
            text_model.norm(hidden_states)[0, -1].float(), readout.to(ids.device)
        ).cpu()
    del reference, text_model

    with vllm_runner(
        str(tmp_path),
        runner="pooling",
        dtype="bfloat16",
        head_dtype="float32",
        skip_tokenizer_init=True,
        max_model_len=32,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as model:
        output = model.llm.encode(
            TokensPrompt(prompt_token_ids=token_ids),
            pooling_task="classify",
            pooling_params=PoolingParams(use_activation=False),
        )[0]
    torch.testing.assert_close(output.outputs.data, expected, rtol=1e-2, atol=2e-3)
