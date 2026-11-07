# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load plan per model family, on real model classes with tiny random configs.

For each family: a random checkpoint is written as safetensors, the model is
cold started from it, ``plan_loads`` is run after ``start_reload`` (as the
NCCL engine does at the first update), and that update and a second one
reusing the plan, each with a new random checkpoint (boxes into ``get_dest``,
fallbacks through ``load_weights``), must leave every parameter and buffer
byte for byte as a cold start on that checkpoint, at the same addresses.

The checks are those of ``test_load_plan.py``; this file only supplies real
models and checkpoints. Which names are direct and which fall back is printed
(``-s``) but not asserted, so a family's fallbacks can shrink without a test
change; equality is what must always hold.
"""

import json
from collections import Counter
from math import prod
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.reload import start_reload
from vllm.model_executor.model_loader.reload.plan import LoadPlan
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from .test_load_plan import (
    BF16,
    _bytes,
    _cold_start,
    _plan,
    _planned_reload,
    _state,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="real model layers are built on CUDA"
)

Spec = tuple[str, torch.dtype, tuple[int, ...]]  # checkpoint name, dtype, shape

FP8 = torch.float8_e4m3fn
F32 = torch.float32

_VOCAB, _HIDDEN, _INTER = 512, 256, 256  # multiples of 128 for block FP8
_HEADS, _KV_HEADS, _HEAD_DIM = 4, 2, 64


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")


# ------------------------------------------------------------ checkpoint names


def _qwen3_attention(p: str, gate: bool = False) -> list[Spec]:
    q_rows = _HEADS * _HEAD_DIM * (2 if gate else 1)
    kv_rows = _KV_HEADS * _HEAD_DIM
    return [
        (f"{p}.q_proj.weight", BF16, (q_rows, _HIDDEN)),
        (f"{p}.k_proj.weight", BF16, (kv_rows, _HIDDEN)),
        (f"{p}.v_proj.weight", BF16, (kv_rows, _HIDDEN)),
        (f"{p}.o_proj.weight", BF16, (_HIDDEN, _HEADS * _HEAD_DIM)),
        (f"{p}.q_norm.weight", BF16, (_HEAD_DIM,)),
        (f"{p}.k_norm.weight", BF16, (_HEAD_DIM,)),
    ]


def _gated_mlp(p: str) -> list[Spec]:
    return [
        (f"{p}.gate_proj.weight", BF16, (_INTER, _HIDDEN)),
        (f"{p}.up_proj.weight", BF16, (_INTER, _HIDDEN)),
        (f"{p}.down_proj.weight", BF16, (_HIDDEN, _INTER)),
    ]


def _experts(p: str, num_experts: int, inter: int) -> list[Spec]:
    """The Hub layout: one tensor per expert and projection."""
    specs: list[Spec] = [(f"{p}.gate.weight", BF16, (num_experts, _HIDDEN))]
    for e in range(num_experts):
        specs += [
            (f"{p}.experts.{e}.gate_proj.weight", BF16, (inter, _HIDDEN)),
            (f"{p}.experts.{e}.up_proj.weight", BF16, (inter, _HIDDEN)),
            (f"{p}.experts.{e}.down_proj.weight", BF16, (_HIDDEN, inter)),
        ]
    return specs


def _embeddings_and_head(prefix: str = "model") -> tuple[list[Spec], list[Spec]]:
    head = [
        (f"{prefix}.norm.weight", BF16, (_HIDDEN,)),
        ("lm_head.weight", BF16, (_VOCAB, _HIDDEN)),
    ]
    return [(f"{prefix}.embed_tokens.weight", BF16, (_VOCAB, _HIDDEN))], head


def _qwen3(layers: int, num_experts: int = 0, moe_inter: int = 128) -> list[Spec]:
    first, last = _embeddings_and_head()
    specs = list(first)
    for i in range(layers):
        p = f"model.layers.{i}"
        specs += [
            (f"{p}.input_layernorm.weight", BF16, (_HIDDEN,)),
            (f"{p}.post_attention_layernorm.weight", BF16, (_HIDDEN,)),
            *_qwen3_attention(f"{p}.self_attn"),
        ]
        if num_experts:
            specs += _experts(f"{p}.mlp", num_experts, moe_inter)
        else:
            specs += _gated_mlp(f"{p}.mlp")
    return specs + last


# Qwen3.5 gated delta net (GDN) dimensions
_K_HEADS, _V_HEADS, _K_DIM, _V_DIM, _CONV = 2, 4, 32, 32, 4


def _qwen3_5(layer_types: list[str]) -> list[Spec]:
    key_dim, value_dim = _K_HEADS * _K_DIM, _V_HEADS * _V_DIM
    first, last = _embeddings_and_head()
    specs = list(first)
    for i, kind in enumerate(layer_types):
        p = f"model.layers.{i}"
        specs += [
            (f"{p}.input_layernorm.weight", BF16, (_HIDDEN,)),
            (f"{p}.post_attention_layernorm.weight", BF16, (_HIDDEN,)),
        ]
        if kind == "full_attention":
            specs += _qwen3_attention(f"{p}.self_attn", gate=True)
        else:
            a = f"{p}.linear_attn"
            specs += [
                (f"{a}.in_proj_qkv.weight", BF16, (2 * key_dim + value_dim, _HIDDEN)),
                (f"{a}.in_proj_z.weight", BF16, (value_dim, _HIDDEN)),
                (f"{a}.in_proj_b.weight", BF16, (_V_HEADS, _HIDDEN)),
                (f"{a}.in_proj_a.weight", BF16, (_V_HEADS, _HIDDEN)),
                (f"{a}.conv1d.weight", BF16, (2 * key_dim + value_dim, 1, _CONV)),
                (f"{a}.dt_bias", BF16, (_V_HEADS,)),
                (f"{a}.A_log", F32, (_V_HEADS,)),
                (f"{a}.norm.weight", F32, (_V_DIM,)),
                (f"{a}.out_proj.weight", BF16, (_HIDDEN, value_dim)),
            ]
        specs += _gated_mlp(f"{p}.mlp")
    return specs + last


# Nemotron-H Mamba2 dimensions
_M_HEADS, _M_HEAD_DIM, _M_GROUPS, _M_STATE, _M_CONV = 8, 32, 2, 16, 4


def _nemotron_h(pattern: str) -> list[Spec]:
    d_inner = _M_HEADS * _M_HEAD_DIM
    conv_dim = d_inner + 2 * _M_GROUPS * _M_STATE
    in_proj = 2 * d_inner + 2 * _M_GROUPS * _M_STATE + _M_HEADS
    kv_rows = _KV_HEADS * _HEAD_DIM
    specs: list[Spec] = [("backbone.embeddings.weight", BF16, (_VOCAB, _HIDDEN))]
    for i, kind in enumerate(pattern):
        p = f"backbone.layers.{i}"
        specs.append((f"{p}.norm.weight", BF16, (_HIDDEN,)))
        m = f"{p}.mixer"
        if kind == "M":
            specs += [
                (f"{m}.in_proj.weight", BF16, (in_proj, _HIDDEN)),
                (f"{m}.conv1d.weight", BF16, (conv_dim, 1, _M_CONV)),
                (f"{m}.conv1d.bias", BF16, (conv_dim,)),
                (f"{m}.dt_bias", F32, (_M_HEADS,)),
                (f"{m}.A_log", F32, (_M_HEADS,)),
                (f"{m}.D", F32, (_M_HEADS,)),
                (f"{m}.norm.weight", BF16, (d_inner,)),
                (f"{m}.out_proj.weight", BF16, (_HIDDEN, d_inner)),
            ]
        elif kind == "*":
            specs += [
                (f"{m}.q_proj.weight", BF16, (_HEADS * _HEAD_DIM, _HIDDEN)),
                (f"{m}.k_proj.weight", BF16, (kv_rows, _HIDDEN)),
                (f"{m}.v_proj.weight", BF16, (kv_rows, _HIDDEN)),
                (f"{m}.o_proj.weight", BF16, (_HIDDEN, _HEADS * _HEAD_DIM)),
            ]
        else:  # "-": a plain MLP
            specs += [
                (f"{m}.up_proj.weight", BF16, (_INTER, _HIDDEN)),
                (f"{m}.down_proj.weight", BF16, (_HIDDEN, _INTER)),
            ]
    return specs + [
        ("backbone.norm_f.weight", BF16, (_HIDDEN,)),
        ("lm_head.weight", BF16, (_VOCAB, _HIDDEN)),
    ]


def _is_linear_weight(name: str, shape: tuple[int, ...]) -> bool:
    return (
        name.endswith(".weight")
        and len(shape) == 2
        and not any(s in name for s in ("embed", "lm_head", "norm", ".mlp.gate."))
    )


def _fp8_block(specs: list[Spec], block: int = 128) -> list[Spec]:
    """Linear weights as FP8 with a ``weight_scale_inv`` per block (fp32, as
    the parameter is: a BF16 scale would be a dtype-cast fallback)."""
    out: list[Spec] = []
    for name, dtype, shape in specs:
        if _is_linear_weight(name, shape):
            scale = (-(-shape[0] // block), -(-shape[1] // block))
            out += [
                (name, FP8, shape),
                (name.replace(".weight", ".weight_scale_inv"), F32, scale),
            ]
        else:
            out.append((name, dtype, shape))
    return out


def _modelopt_fp8(specs: list[Spec]) -> list[Spec]:
    """ModelOpt static per-tensor FP8: a ``weight_scale`` and ``input_scale``
    (scalars) per linear and per expert projection."""
    out: list[Spec] = []
    for name, dtype, shape in specs:
        if _is_linear_weight(name, shape):
            out += [
                (name, FP8, shape),
                (name.replace(".weight", ".weight_scale"), F32, ()),
                (name.replace(".weight", ".input_scale"), F32, ()),
            ]
        else:
            out.append((name, dtype, shape))
    return out


# ------------------------------------------------------------ configs

_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTER,
    "num_hidden_layers": 2,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "vocab_size": _VOCAB,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "torch_dtype": "bfloat16",
}

_QWEN3_MOE_CONFIG = {
    **_QWEN3_CONFIG,
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 128,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}

_QWEN3_5_LAYERS = ["linear_attention", "full_attention"]
_QWEN3_5_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTER,
    "num_hidden_layers": len(_QWEN3_5_LAYERS),
    "layer_types": _QWEN3_5_LAYERS,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "attn_output_gate": True,
    "linear_num_key_heads": _K_HEADS,
    "linear_num_value_heads": _V_HEADS,
    "linear_key_head_dim": _K_DIM,
    "linear_value_head_dim": _V_DIM,
    "linear_conv_kernel_dim": _CONV,
    "vocab_size": _VOCAB,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

_NEMOTRON_PATTERN = "M-*"
_NEMOTRON_H_CONFIG = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTER,
    "num_hidden_layers": len(_NEMOTRON_PATTERN),
    "hybrid_override_pattern": _NEMOTRON_PATTERN,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "mamba_num_heads": _M_HEADS,
    "mamba_head_dim": _M_HEAD_DIM,
    "n_groups": _M_GROUPS,
    "ssm_state_size": _M_STATE,
    "conv_kernel": _M_CONV,
    "chunk_size": 64,
    "mamba_hidden_act": "silu",
    "mlp_hidden_act": "relu2",
    "mlp_bias": False,
    "use_bias": False,
    "use_conv_bias": True,
    "layer_norm_epsilon": 1e-5,
    "residual_in_fp32": True,
    "vocab_size": _VOCAB,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

_FP8_BLOCK_QUANT = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
_MODELOPT_FP8_QUANT = {
    "quant_method": "modelopt",
    "quant_algo": "FP8",
    "exclude_modules": ["lm_head", "*mlp.gate"],
}

FAMILIES = {
    "qwen3-bf16": (_QWEN3_CONFIG, _qwen3(2)),
    "qwen3-fp8-block": (
        {**_QWEN3_CONFIG, "quantization_config": _FP8_BLOCK_QUANT},
        _fp8_block(_qwen3(2)),
    ),
    "qwen3-moe-modelopt-fp8": (
        {**_QWEN3_MOE_CONFIG, "quantization_config": _MODELOPT_FP8_QUANT},
        _modelopt_fp8(_qwen3(2, num_experts=4)),
    ),
    "qwen3_5-gdn": (_QWEN3_5_CONFIG, _qwen3_5(_QWEN3_5_LAYERS)),
    "nemotron-h-mamba2": (_NEMOTRON_H_CONFIG, _nemotron_h(_NEMOTRON_PATTERN)),
}


# ------------------------------------------------------------ checkpoints


def _random_tensor(name: str, dtype: torch.dtype, shape, gen) -> torch.Tensor:
    if dtype == FP8:  # finite values only: exponent below the NaN encoding
        raw = torch.randint(0, 0x78, shape, generator=gen, dtype=torch.uint8)
        raw |= torch.randint(0, 2, shape, generator=gen, dtype=torch.uint8) << 7
        return raw.view(FP8)
    if "scale" in name:  # positive, as a quantizer writes them
        return torch.rand(shape, generator=gen) * 0.02 + 0.005
    return (torch.randn(shape, generator=gen) * 0.05).to(dtype)


def _write_checkpoint(path: Path, specs: list[Spec], seed: int, file: str):
    gen = torch.Generator().manual_seed(seed)
    tensors = {n: _random_tensor(n, dt, sh, gen) for n, dt, sh in specs}
    save_file(tensors, str(path / file))


def _vllm_config(path: Path) -> VllmConfig:
    return EngineArgs(
        model=str(path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=256,
        enforce_eager=True,
    ).create_engine_config()


def _summary(family: str, plan: LoadPlan, specs: list[Spec], model) -> dict:
    kinds = Counter()
    nbytes = Counter()
    for name, dtype, shape in specs:
        entry = plan.entries[name]
        kind = (
            "fallback"
            if entry.writes is None
            else "direct"
            if entry.boxes
            else "unneeded"
        )
        kinds[kind] += 1
        nbytes[kind] += prod(shape) * dtype.itemsize
    state = _state(model)
    return {
        "family": family,
        "checkpoint_names": len(specs),
        "checkpoint_bytes": sum(nbytes.values()),
        "model_tensors": len(state),
        "model_bytes": sum(t[3].numel() * t[3].element_size() for t in state.values()),
        "names": dict(kinds),
        "bytes": dict(nbytes),
        "fallback_reasons": dict(
            Counter(e.reason for e in plan.entries.values() if e.reason)
        ),
    }  # fmt: skip


@pytest.mark.parametrize("family", list(FAMILIES))
def test_plan_equals_cold_start(dist_init, tmp_path, family):
    """A planned update equals a cold start on the same weights, byte for byte,
    for every parameter and buffer of the real model class, and moves none."""
    config, specs = FAMILIES[family]
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_checkpoint(tmp_path, specs, seed=0, file="model.safetensors")
    updates = []
    for seed in (1, 2):
        (tmp_path / f"update{seed}").mkdir()
        _write_checkpoint(
            tmp_path / f"update{seed}", specs, seed=seed, file="model.safetensors"
        )
        updates.append(load_file(str(tmp_path / f"update{seed}" / "model.safetensors")))
    start = load_file(str(tmp_path / "model.safetensors"))
    shapes = {n: (sh, dt) for n, dt, sh in specs}
    vllm_config = _vllm_config(tmp_path)

    def make():
        # one config per model: each registers its layers by name
        config = _vllm_config(tmp_path)
        with (
            set_current_vllm_config(config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device(current_platform.device_type),
        ):
            return initialize_model(config)

    with set_current_vllm_config(vllm_config):
        model = _cold_start(make, start)
        before = _state(model)
        start_reload(model)
        plan = _plan(model, shapes)
        assert plan.entries.keys() == shapes.keys()
        for i, update in enumerate(updates):
            if i:
                start_reload(model)
            _planned_reload(model, plan, update)
            actual, expected = _state(model), _state(_cold_start(make, update))
            assert actual.keys() == expected.keys()
            for name, (dtype, shape, stride, data, ptr) in actual.items():
                assert (dtype, shape, stride) == expected[name][:3], name
                assert torch.equal(_bytes(data), _bytes(expected[name][3])), name
                assert ptr == before[name][4], f"{name} moved"
    # the updates really replaced the first checkpoint
    assert any(
        not torch.equal(_bytes(before[n][3]), _bytes(d))
        for n, (_, _, _, d, _) in actual.items()
    )
    summary = _summary(family, plan, specs, model)
    assert summary["names"].get("direct", 0) > 0
    print("P2_SUMMARY " + json.dumps(summary))
