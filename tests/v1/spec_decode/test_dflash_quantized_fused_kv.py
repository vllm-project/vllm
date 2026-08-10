# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DFlash fused-KV buffers with a quantized drafter.

The DFlash fused-KV path (``precompute_and_store_context_kv``) slices
``qkv_proj.weight`` and feeds it to a single ``F.linear``.  For a quantized
drafter this bypasses the quant method: an FP8 weight is stored as
``float8_e4m3fn``, so the GEMM raises a dtype mismatch, and other schemes
silently drop their scales.  The fix dequantizes quantized qkv weights with
the canonical ``get_and_maybe_dequant_weights`` at buffer-build time (the
buffers are built lazily on first use, after ``process_weights_after_loading``).

See https://github.com/vllm-project/vllm/issues/51581.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

HIDDEN = 256
HEADS = 8
KV_HEADS = 4
HEAD_DIM = 32
Q_SIZE = HEADS * HEAD_DIM  # 256
KV_SIZE = KV_HEADS * HEAD_DIM  # 128
NUM_LAYERS = 2


@pytest.fixture
def fp8_vllm_config(tmp_path):
    """VllmConfig with a usable model_config dtype (the default fixture leaves
    model_config unset, which breaks Fp8LinearMethod construction)."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 100,
        "hidden_size": HIDDEN,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 64,
    }))
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="bfloat16",
        seed=0,
        trust_remote_code=False,
    )
    config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(config):
        yield config


def _make_fp8_qkv():
    """Fused FP8-serialized QKV projection: fp8 [out, in] weight with a
    per-tensor scale (equal across the q/k/v shards, like a fused checkpoint)."""
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
    )
    qkv = QKVParallelLinear(
        hidden_size=HIDDEN,
        head_size=HEAD_DIM,
        total_num_heads=HEADS,
        total_num_kv_heads=KV_HEADS,
        bias=False,
        quant_config=quant_config,
        prefix="test.qkv_proj",
    ).cuda()

    out = Q_SIZE + 2 * KV_SIZE
    w_bf16 = torch.randn(out, HIDDEN, dtype=torch.bfloat16, device="cuda")
    scale = w_bf16.abs().max() / 448.0
    qkv.weight.data.copy_((w_bf16 / scale).to(torch.float8_e4m3fn))
    qkv.weight_scale.data.copy_(scale.expand(3))
    return qkv


def _kv_rows(qkv):
    """Dequantized KV rows of a processed qkv projection ([out, in])."""
    return get_and_maybe_dequant_weights(qkv, out_dtype=torch.bfloat16)[Q_SIZE:]


def test_fused_projection_matches_per_layer(dist_init, fp8_vllm_config):
    """Output-level check: the fused KV GEMM (one F.linear over the dequantized
    fused weight, as precompute_and_store_context_kv runs) must produce the
    same K/V as projecting each layer's quantized KV rows separately."""
    qkvs = [_make_fp8_qkv(), _make_fp8_qkv()]
    for qkv in qkvs:
        qkv.quant_method.process_weights_after_loading(qkv)
    attns = [SimpleNamespace(qkv_proj=q, q_size=Q_SIZE, kv_size=KV_SIZE)
             for q in qkvs]

    num_ctx = 6
    normed = torch.randn(num_ctx, HIDDEN, dtype=torch.bfloat16, device="cuda")

    # Fused path: concatenate the dequantized KV rows into one weight.
    fused = torch.cat([_kv_rows(a.qkv_proj) for a in attns], dim=0)
    fused_out = F.linear(normed, fused)  # [num_ctx, n_layers * 2 * KV_SIZE]

    # Per-layer path: project each layer separately.
    per_layer_out = torch.cat(
        [F.linear(normed, _kv_rows(a.qkv_proj)) for a in attns], dim=-1)

    torch.testing.assert_close(fused_out, per_layer_out, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Model-level regression: a real DFlashQwen3ForCausalLM with an FP8-serialized
# drafter checkpoint must load, build a BF16 fused-KV buffer (lazily, after
# process_weights_after_loading), and run the fused context-KV path without
# the dtype crash from #51581.
# ---------------------------------------------------------------------------


def _write_dflash_configs(tmp_path):
    """Write a plain Qwen3 target config and a DFlash (FP8) draft config."""
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    base_cfg = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": HIDDEN,
        "intermediate_size": 640,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    }
    (target / "config.json").write_text(json.dumps({
        **base_cfg, "architectures": ["Qwen3ForCausalLM"],
    }))
    (draft / "config.json").write_text(json.dumps({
        **base_cfg,
        "architectures": ["DFlashDraftModel"],
        "draft_vocab_size": 151936,
        "target_hidden_size": HIDDEN,
        "aux_hidden_state_layer_ids": [0],
        "mask_token_id": 151665,
        "dflash_config": {
            "use_aux_hidden_state": True,
            "mask_token_id": 151665,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        },
    }))
    return str(target), str(draft)


def _make_fp8_draft_ckpt(model):
    """Synthetic FP8-serialized DFlash drafter checkpoint: every quantized
    linear layer is stored as FP8 with a per-tensor scalar scale, using the
    on-disk split names (q_proj/k_proj/v_proj, gate_proj/up_proj) that
    DFlash's weight loading maps onto the fused modules."""
    ckpt: dict[str, torch.Tensor] = {}
    params = dict(model.named_parameters())
    # Use the module-tree names (not ``module.prefix``, which carries the
    # target-layer offset) because weight loading keys off the former.
    quant_modules = [
        (name, m) for name, m in model.named_modules()
        if type(getattr(m, "quant_method", None)).__name__ == "Fp8LinearMethod"
    ]

    def add_fp8(name, w_bf16):
        scale = w_bf16.abs().max() / 448.0
        ckpt[f"{name}.weight"] = (w_bf16 / scale).to(torch.float8_e4m3fn)
        ckpt[f"{name}.weight_scale"] = scale

    for tree_name, module in quant_modules:
        rel = tree_name[len("model."):] if tree_name.startswith("model.") \
            else tree_name
        hidden = module.input_size_per_partition
        if rel.endswith("qkv_proj"):
            q = module.num_heads * module.head_size
            kv = module.num_kv_heads * module.head_size
            base = rel[:-len("qkv_proj")]
            for shard, rows in (("q", q), ("k", kv), ("v", kv)):
                add_fp8(f"{base}{shard}_proj",
                        torch.randn(rows, hidden, dtype=torch.bfloat16,
                                    device="cuda"))
        elif rel.endswith("gate_up_proj"):
            sizes = module.output_partition_sizes
            base = rel[:-len("gate_up_proj")]
            for shard, rows in (("gate", sizes[0]), ("up", sizes[1])):
                add_fp8(f"{base}{shard}_proj",
                        torch.randn(rows, hidden, dtype=torch.bfloat16,
                                    device="cuda"))
        else:
            add_fp8(rel, torch.randn(module.weight.shape,
                                     dtype=torch.bfloat16, device="cuda"))

    quant_names = set()
    for tree_name, module in quant_modules:
        for pname in module.state_dict():
            quant_names.add(f"{tree_name}.{pname}")
    for name, param in params.items():
        if name in quant_names or name.endswith(".weight_scale"):
            continue
        short = name[len("model."):] if name.startswith("model.") else name
        if param.dtype == torch.long:
            ckpt[short] = torch.zeros_like(param)
        else:
            ckpt[short] = torch.randn(param.shape, dtype=torch.bfloat16)
    return ckpt


def _checkpoint_truth_fused_kv(model):
    """Ground-truth fused KV weight: dequant of the loaded pre-process fp8
    weights with the checkpoint's own per-shard scales (all equal for a
    per-tensor checkpoint, so this matches the post-processing dequant)."""
    rows = []
    for layer in model.model.layers:
        attn = layer.self_attn
        proj = attn.qkv_proj
        w = proj.weight[attn.q_size:].to(torch.bfloat16)
        s = proj.weight_scale.to(torch.bfloat16)
        per_row = torch.cat(
            [s[1].expand(attn.kv_size), s[2].expand(attn.kv_size)])
        rows.append(w * per_row.unsqueeze(1))
    return torch.cat(rows, dim=0)


def test_dflash_fused_kv_quantized_model(dist_init, tmp_path):
    """The full model path used to crash at engine init with an FP8 drafter
    (`_fused_kv_weight` was fp8 and `F.linear` raised a dtype mismatch).  The
    fix must build a BF16 fused buffer (lazily, after
    process_weights_after_loading) that exactly matches the checkpoint-truth
    per-layer computation, and the fused context-KV path must run."""
    from vllm.config import CompilationMode, ParallelConfig, SpeculativeConfig
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    target_dir, draft_dir = _write_dflash_configs(tmp_path)
    target_mc = ModelConfig(
        model=target_dir, tokenizer=target_dir, dtype="bfloat16", seed=0,
        trust_remote_code=False, max_model_len=64,
    )
    spec = SpeculativeConfig(
        target_model_config=target_mc,
        target_parallel_config=ParallelConfig(),
        model=draft_dir,
        method="dflash",
        num_speculative_tokens=2,
    )
    vc = VllmConfig(model_config=target_mc, speculative_config=spec,
                    parallel_config=ParallelConfig())
    vc.compilation_config.mode = CompilationMode.NONE

    with set_current_vllm_config(vc):
        model = DFlashQwen3ForCausalLM(vllm_config=vc).cuda()
        ckpt = _make_fp8_draft_ckpt(model)
        model.load_weights(list(ckpt.items()))
        truth = _checkpoint_truth_fused_kv(model)

        # The loader processes weights after load_weights; the fused buffers
        # are built lazily on first use (the fixed path).
        process_weights_after_loading(model, target_mc, torch.device("cuda"))

        ctx = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
        positions = torch.arange(4, device="cuda")
        model.precompute_and_store_context_kv(ctx, positions, None)

        fused = model.model._fused_kv_weight
        assert fused.dtype == torch.bfloat16, fused.dtype
        L = model.model._num_attn_layers
        assert L == NUM_LAYERS
        assert fused.shape == (NUM_LAYERS * 2 * KV_SIZE, HIDDEN), fused.shape
        # The fused buffer must exactly reconstruct the checkpoint-truth
        # quantized KV weights (per-shard fp8 x own scales).
        torch.testing.assert_close(fused, truth, rtol=0, atol=0)

        # The fused context-KV projection must equal the checkpoint-truth
        # per-layer computation (this is the path that raised the #51581
        # dtype error).
        nkv = model.model._num_kv_heads
        hd = model.model._head_dim
        all_k, all_v = model.model._project_context_kv(ctx, 4, L, nkv, hd)

        normed = torch.empty_like(ctx)
        ops.rms_norm(normed, ctx, model.model._hidden_norm_weight,
                     model.model._rms_norm_eps)
        ref_flat = F.linear(normed, truth)
        ref = (ref_flat.view(4, L, 2, nkv, hd).permute(2, 1, 0, 3, 4)
               .contiguous())
        torch.testing.assert_close(all_k, ref[0], rtol=0, atol=0)
        torch.testing.assert_close(all_v, ref[1], rtol=0, atol=0)
