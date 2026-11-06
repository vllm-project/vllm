# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`enable_weights_track` on NVFP4 MoE checkpoints.

Weight-only NVFP4 exports write no input scales, and no weight-only backend
reads them, so the check must accept these checkpoints. It must still report
the tensors they do need, and the input scales of W4A4 checkpoints.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.quantization.utils import is_quant_method_supported
from vllm import SamplingParams

pytestmark = pytest.mark.skipif(
    not is_quant_method_supported("modelopt_fp4"),
    reason="NVFP4 is not supported on this GPU type.",
)

HIDDEN, MOE_INTERMEDIATE, EXPERTS, HEADS, KV_HEADS, HEAD_DIM, VOCAB = (
    512,
    256,
    8,
    4,
    2,
    128,
    256,
)
_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "dynamic": False,
}
_ACTIVATIONS = dict(_WEIGHTS, dynamic="local")
_MODELOPT = {
    "quant_method": "modelopt",
    "group_size": 16,
    "kv_cache_quant_algo": None,
    "exclude_modules": ["lm_head", "model.layers.*.mlp.gate"],
}
QUANT_CONFIGS = {
    "modelopt_w4a16": dict(_MODELOPT, quant_algo="W4A16_NVFP4"),
    # quant_algo NVFP4 with weight-only config groups, as in vLLM#54189.
    "modelopt_nvfp4_weight_only": dict(
        _MODELOPT,
        quant_algo="NVFP4",
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": _WEIGHTS,
                "input_activations": None,
            }
        },
    ),
    "modelopt_nvfp4": dict(_MODELOPT, quant_algo="NVFP4"),
    "ct_nvfp4a16": {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": _WEIGHTS,
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*mlp.gate$"],
    },
}


def _nvfp4(prefix: str, out: int, inp: int, fmt: str) -> dict[str, torch.Tensor]:
    weight = torch.randint(0, 256, (out, inp // 2), dtype=torch.uint8)
    block_scale = torch.full((out, inp // 16), 0.5).to(torch.float8_e4m3fn)
    if fmt.startswith("ct"):
        return {
            f"{prefix}.weight_packed": weight,
            f"{prefix}.weight_scale": block_scale,
            f"{prefix}.weight_global_scale": torch.tensor([100.0]),
        }
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.weight_scale": block_scale,
        f"{prefix}.weight_scale_2": torch.tensor(0.01),
    }
    if fmt == "modelopt_nvfp4":
        tensors[f"{prefix}.input_scale"] = torch.tensor(0.02)
    return tensors


def _write_checkpoint(path, fmt: str, drop: str | None = None) -> None:
    """One-layer Qwen3-MoE with NVFP4 attention and experts."""
    torch.manual_seed(0)
    bf16 = torch.bfloat16
    p = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, dtype=bf16) * 0.02,
        "model.norm.weight": torch.ones(HIDDEN, dtype=bf16),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, dtype=bf16) * 0.02,
        f"{p}.input_layernorm.weight": torch.ones(HIDDEN, dtype=bf16),
        f"{p}.post_attention_layernorm.weight": torch.ones(HIDDEN, dtype=bf16),
        f"{p}.self_attn.q_norm.weight": torch.ones(HEAD_DIM, dtype=bf16),
        f"{p}.self_attn.k_norm.weight": torch.ones(HEAD_DIM, dtype=bf16),
        f"{p}.mlp.gate.weight": torch.randn(EXPERTS, HIDDEN, dtype=bf16) * 0.02,
    }
    for name, out, inp in (
        ("q_proj", HEADS * HEAD_DIM, HIDDEN),
        ("k_proj", KV_HEADS * HEAD_DIM, HIDDEN),
        ("v_proj", KV_HEADS * HEAD_DIM, HIDDEN),
        ("o_proj", HIDDEN, HEADS * HEAD_DIM),
    ):
        tensors |= _nvfp4(f"{p}.self_attn.{name}", out, inp, fmt)
    for e in range(EXPERTS):
        for name, out, inp in (
            ("gate_proj", MOE_INTERMEDIATE, HIDDEN),
            ("up_proj", MOE_INTERMEDIATE, HIDDEN),
            ("down_proj", HIDDEN, MOE_INTERMEDIATE),
        ):
            tensors |= _nvfp4(f"{p}.mlp.experts.{e}.{name}", out, inp, fmt)
    if drop is not None:
        tensors = {k: v for k, v in tensors.items() if drop not in k}
    save_file(tensors, path / "model.safetensors")
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": 2 * HIDDEN,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "quantization_config": QUANT_CONFIGS[fmt],
    }
    (path / "config.json").write_text(json.dumps(config))


def _generate(path, vllm_runner) -> list[int]:
    with vllm_runner(
        str(path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_flashinfer_autotune=False,
        load_format="safetensors",
        model_loader_extra_config={"enable_weights_track": True},
        max_model_len=32,
        max_num_seqs=1,
        kv_cache_memory_bytes=16 << 20,
    ) as llm:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        return llm.generate([[1, 2, 3]], params)[0][0][0]


@pytest.mark.parametrize(
    "fmt", ["modelopt_w4a16", "modelopt_nvfp4_weight_only", "ct_nvfp4a16"]
)
def test_weights_track_accepts_weight_only_nvfp4(tmp_path, vllm_runner, fmt):
    _write_checkpoint(tmp_path, fmt)
    assert len(_generate(tmp_path, vllm_runner)) == 7


@pytest.mark.parametrize(
    "fmt, drop, missing",
    [
        ("modelopt_w4a16", "o_proj.weight_scale", "self_attn.o_proj.weight_scale"),
        ("ct_nvfp4a16", "o_proj.weight_scale", "self_attn.o_proj.weight_scale"),
        # W4A4 backends read the input scales.
        ("modelopt_nvfp4", "input_scale", "mlp.experts.routed_experts.w13_input_scale"),
    ],
)
def test_weights_track_reports_missing_nvfp4_tensors(
    tmp_path, monkeypatch, vllm_runner, fmt, drop, missing
):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _write_checkpoint(tmp_path, fmt, drop=drop)
    with pytest.raises(ValueError, match="not initialized from checkpoint") as exc:
        _generate(tmp_path, vllm_runner)
    assert f"'model.layers.0.{missing}'" in str(exc.value)
