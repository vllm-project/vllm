# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A W4A16 compressed-tensors DFlash drafter loads and drafts (#53116).

The drafter's fused qkv_proj and gate_up_proj used to be built unquantized, so
loading its packed weights failed. Both checkpoints are tiny, random and written
to tmp_path, so the test needs no download."""

import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from vllm import SamplingParams
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="W4A16 Marlin kernels require CUDA"
)

HIDDEN = 256
INTERMEDIATE = 512
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
GROUP = 64
VOCAB = 1024
QWEN3 = {
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "vocab_size": VOCAB,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
}
PROJECTIONS = {
    "self_attn.q_proj": (HEADS * HEAD_DIM, HIDDEN),
    "self_attn.k_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.v_proj": (KV_HEADS * HEAD_DIM, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
    "mlp.gate_proj": (INTERMEDIATE, HIDDEN),
    "mlp.up_proj": (INTERMEDIATE, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTERMEDIATE),
}
W4A16 = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "targets": [
                r"re:.*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
            ],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": GROUP,
                "dynamic": False,
            },
            "input_activations": None,
            "output_activations": None,
        }
    },
    "ignore": ["fc"],
}


def _write_target(path: Path) -> Path:
    model = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **QWEN3))
    model.to(torch.bfloat16).save_pretrained(path)
    return path


def _write_w4a16_dflash_draft(path: Path) -> Path:
    """One DFlash layer with random 4-bit weights for every projection."""
    path.mkdir()
    config = {
        **QWEN3,
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "torch_dtype": "bfloat16",
        "block_size": 4,
        "dflash_config": {"mask_token_id": 0, "target_layer_ids": [0, 1]},
        "quantization_config": W4A16,
    }
    (path / "config.json").write_text(json.dumps(config))

    def bf16(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape) * 0.02).to(torch.bfloat16)

    def ones(size: int) -> torch.Tensor:
        return torch.ones(size, dtype=torch.bfloat16)

    weights = {
        "fc.weight": bf16(HIDDEN, 2 * HIDDEN),
        "hidden_norm.weight": ones(HIDDEN),
        "norm.weight": ones(HIDDEN),
        "lm_head.weight": bf16(VOCAB, HIDDEN),
        "layers.0.input_layernorm.weight": ones(HIDDEN),
        "layers.0.post_attention_layernorm.weight": ones(HIDDEN),
        "layers.0.self_attn.q_norm.weight": ones(HEAD_DIM),
        "layers.0.self_attn.k_norm.weight": ones(HEAD_DIM),
    }
    for name, (rows, cols) in PROJECTIONS.items():
        prefix = f"layers.0.{name}"
        # Signed 4-bit range [-8, 7].
        values = torch.randint(-8, 8, (rows, cols), dtype=torch.int8)
        weights[f"{prefix}.weight_packed"] = pack_to_int32(values, 4)
        weights[f"{prefix}.weight_scale"] = torch.full(
            (rows, cols // GROUP), 0.01, dtype=torch.bfloat16
        )
        weights[f"{prefix}.weight_shape"] = torch.tensor([rows, cols])
    save_file(weights, path / "model.safetensors")
    return path


def test_w4a16_dflash_draft_loads_and_drafts(tmp_path: Path, vllm_runner):
    torch.manual_seed(0)
    target = _write_target(tmp_path / "target")
    draft = _write_w4a16_dflash_draft(tmp_path / "draft")

    with vllm_runner(
        str(target),
        skip_tokenizer_init=True,
        max_model_len=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.3,
        speculative_config={
            "method": "dflash",
            "model": str(draft),
            "num_speculative_tokens": 3,
        },
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3, 4]}],
            SamplingParams(
                max_tokens=8, temperature=0.0, detokenize=False, ignore_eos=True
            ),
        )

    assert len(outputs[0].outputs[0].token_ids) == 8
