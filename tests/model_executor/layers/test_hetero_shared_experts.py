# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous fused shared experts for DeepSeek-V4.1 (ROCm gfx950).

Contracts: the folded FHMoE output matches the separate-path fallback
(no requant, correct layout); the router appends the shared slot only
inside the ``MIN..MAX_TOKENS`` window and never on decode steps; the
in-kernel append is bit-identical to the cat append; the window knobs
parse with safe fallbacks.

Numerics methodology (do not "simplify"): isolate the shared path with
negligible-but-nonzero routed weights (1e-9; all-zero weights corrupt the
shared contribution at some M); keep routed scales modest (E8M0 byte 120
= 2^-7) so routed outputs stay O(1).
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import _aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.models.deepseek_v4.amd import model as dsv4_model
from vllm.platforms import current_platform

H, E_R, K = 512, 8, 2
E = E_R + 1
MS = (7, 256, 2048)

_rocm = current_platform.is_rocm()
try:
    from vllm.models.deepseek_v4.amd.model import on_gfx950

    _gfx950 = _rocm and on_gfx950()
except Exception:
    _gfx950 = False
try:
    import aiter  # noqa: F401
    import aiter.fhmoe  # noqa: F401

    _aiter = True
except Exception:
    _aiter = False

_NUMERICS = pytest.mark.skipif(
    not (_rocm and _gfx950 and _aiter),
    reason="heterogeneous FHMoE numerics require ROCm gfx950 with AITER",
)

# Verbatim text_config of the DSV4.1-Flash checkpoint, sizes shrunk (every
# override key must exist here, so config-schema drift fails loudly).
_REAL_TEXT_CONFIG_JSON = """
    {
      "model_type": "deepseek_v41_text",
      "vocab_size": 129280,
      "hidden_size": 5120,
      "moe_intermediate_size": 2304,
      "num_hidden_layers": 40,
      "num_attention_heads": 64,
      "num_key_value_heads": 1,
      "head_dim": 512,
      "qk_rope_head_dim": 64,
      "q_lora_rank": 1280,
      "o_lora_rank": 1024,
      "o_groups": 8,
      "hidden_act": "silu",
      "swiglu_limit": 10.0,
      "rms_norm_eps": 1e-20,
      "attention_bias": false,
      "attention_dropout": 0.0,
      "initializer_range": 0.02,
      "use_cache": true,
      "tie_word_embeddings": false,
      "max_position_embeddings": 1048576,
      "rope_theta": 10000,
      "n_routed_experts": 384,
      "n_shared_experts": 1,
      "num_experts_per_tok": 6,
      "scoring_func": "sqrtsoftplus",
      "norm_topk_prob": true,
      "routed_scaling_factor": 1.5,
      "compress_ratios": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "first_k_dense_replace": 0,
      "kv_source_layer_ids": [2, 8, 14, 20],
      "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
      "index_n_heads": 32,
      "index_head_dim": 128,
      "index_topk": 512,
      "candidate_source_layer_id": 20,
      "candidate_topk_blocks": 2048,
      "candidate_block_size": 8,
      "hc_mult": 4,
      "hc_sinkhorn_iters": 20,
      "hc_eps": 1e-06,
      "engram_layer_ids": [],
      "engram_num_embeddings": [],
      "engram_max_ngram_size": 4,
      "engram_vocab_size": 16000000,
      "engram_n_heads": 8,
      "engram_head_dim": 256,
      "engram_pad_token_id": 2,
      "engram_compressed_vocab_size": 99092,
      "num_nextn_predict_layers": 3,
      "dspark_block_size": 5,
      "dspark_noise_token_id": 128799,
      "dspark_target_layer_ids": [37, 38, 39],
      "dspark_markov_rank": 256,
      "dspark_n_routed_experts": 128,
      "dspark_num_experts_per_tok": 3
    }
"""

_REAL_QUANT_CFG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

_TINY_OVERRIDES: dict = {
    "num_hidden_layers": 2,
    "hidden_size": H,
    "moe_intermediate_size": 96,  # rounds to 128: exercises the real
    # per-slab gate/up padding of the Flash TP4 topology (576 -> 640)
    "n_routed_experts": E_R,
    "num_experts_per_tok": K,
    "n_shared_experts": 1,
    "num_attention_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 2048,
    "compress_ratios": [0, 0],
    "kv_source_layer_ids": [2],
    "index_source_layer_ids": [],
}


def _ensure_dist_initialized() -> None:
    if torch.distributed.is_initialized():
        return
    from tests.utils import ensure_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    handle, init_file = tempfile.mkstemp(prefix="dsv41_hetero_dist_")
    os.close(handle)
    with suppress(OSError):
        os.unlink(init_file)
    with ensure_current_vllm_config():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{init_file}",
            local_rank=0,
            backend="gloo" if current_platform.is_cpu() else "nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )


def _reset_aiter_topk_singleton() -> None:
    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    rocm_aiter_moe.aiter_topK_meta_data = None


def _build():
    """Tiny heterogeneous layer, registry-faithful (real config, tiny sizes)."""
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    _ensure_dist_initialized()
    _reset_aiter_topk_singleton()
    real = json.loads(_REAL_TEXT_CONFIG_JSON)
    assert set(_TINY_OVERRIDES) <= set(real), "tiny override key drift"
    config = DeepseekV41Config(
        text_config={**real, **_TINY_OVERRIDES},
        quantization_config=_REAL_QUANT_CFG,
    )
    qc = get_quantization_config("deepseek_v4_fp8").from_config(
        dict(config.quantization_config)
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        dtype=torch.bfloat16,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=None,
        quantization_config=None,
        runner_type="generate",
        is_moe=True,
        logits_processors=None,
        rswa_window=None,
    )
    vllm_config.parallel_config.enable_expert_parallel = False
    vllm_config.quant_config = qc
    intermediate = _TINY_OVERRIDES["moe_intermediate_size"]
    with (
        set_current_vllm_config(vllm_config),
        torch.device(current_platform.device_type),
    ):
        shared = dsv4_model.DeepseekV4MLP(
            hidden_size=H,
            intermediate_size=intermediate,
            hidden_act="silu",
            swiglu_limit=None,
            quant_config=qc,
            reduce_results=False,
            prefix="model.layers.0.ffn.shared_experts",
        )
        experts = FusedMoEFactory(
            num_experts=E_R,
            top_k=K,
            hidden_size=H,
            intermediate_size=intermediate,
            n_shared_experts=1,
            fuse_shared_experts=True,
            shared_expert_prefix="model.layers.0.ffn.shared_experts",
            prefix="model.layers.0.ffn.experts",
            quant_config=qc,
            tp_size=1,
            routed_experts_cls=dsv4_model.DeepseekV4HeterogeneousSharedRoutedExperts,
            routed_experts_args={"shared_expert": shared, "shared_expert_id": E_R},
        ).routed_experts
    return shared, experts, intermediate


def _fill_weights(shared, experts, intermediate: int, i_sp: int, dev):
    """Deterministic O(1)-magnitude weights, then the production
    weight-processing order (shared post-load, experts MXFP4 shuffle +
    heterogeneous prepare, shared preshuffle)."""
    torch.manual_seed(0)
    experts.w13_weight.data.copy_(
        torch.randint(0, 256, (E, 2 * i_sp, H // 2), dtype=torch.uint8, device=dev)
    )
    experts.w13_weight_scale.data.fill_(120)
    experts.w2_weight.data.copy_(
        torch.randint(0, 256, (E, H, i_sp // 2), dtype=torch.uint8, device=dev)
    )
    experts.w2_weight_scale.data.fill_(120)
    gu_w = (torch.randn(2 * intermediate, H, device=dev) * 0.05).to(torch.float8_e4m3fn)
    gu_s = (
        (torch.randint(-6, 3, (2 * intermediate // 32, H // 32), device=dev) + 127)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    dn_w = (torch.randn(H, intermediate, device=dev) * 0.05).to(torch.float8_e4m3fn)
    dn_s = (
        (torch.randint(-6, 3, (H // 32, intermediate // 32), device=dev) + 127)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    shared.gate_up_proj.weight.data.copy_(gu_w)
    shared.gate_up_proj.weight_scale.data.copy_(
        gu_s.repeat_interleave(32, dim=0).view(torch.uint8)
    )
    shared.down_proj.weight.data.copy_(dn_w)
    shared.down_proj.weight_scale.data.copy_(
        dn_s.repeat_interleave(32, dim=0).view(torch.uint8)
    )
    shared.gate_up_proj.quant_method.process_weights_after_loading(shared.gate_up_proj)
    shared.down_proj.quant_method.process_weights_after_loading(shared.down_proj)
    experts.quant_method.process_weights_after_loading(experts)
    shared.prepare_gateup_preshuffle()
    return gu_w, gu_s, dn_w, dn_s


def _dequant(w_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    s = scale_e8m0.to(torch.float32)
    s = s.view(n // 32, 1, k // 32, 1).expand(n // 32, 32, k // 32, 32).reshape(n, k)
    return w_fp8.to(torch.float32) * s


@pytest.fixture()
def aiter_moe_env(monkeypatch: pytest.MonkeyPatch):
    """Enable the AITER MoE path for this test only (env restored after)."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("AITER_TRITON_LOG_LEVEL", "ERROR")
    monkeypatch.delenv("VLLM_DISABLE_FUSED_SHARED_EXPERTS", raising=False)
    _aiter_ops.rocm_aiter_ops.refresh_env_variables()
    yield
    _aiter_ops.rocm_aiter_ops.refresh_env_variables()


@_NUMERICS
def test_fold_matches_separate_paths(aiter_moe_env: None) -> None:
    """Folded output == separate-path fallback; the fused shared expert
    tracks the fp32 oracle as well as the separate FP8 MLP does."""
    shared, experts, intermediate = _build()
    moe = experts.moe_config
    if not moe.rocm_aiter_fmoe_enabled:
        pytest.skip("VLLM_ROCM_USE_AITER_MOE=0 disables the FHMoE path")
    i_sp = moe.intermediate_size_per_partition
    assert i_sp == 128  # MXFP4 rounds to a 128 multiple
    dev = experts.w13_weight.device
    gu_w, gu_s, dn_w, dn_s = _fill_weights(shared, experts, intermediate, i_sp, dev)
    assert experts._fhmoe_csv_dispatch is False
    assert experts._fhmoe_shape_key == (H, 128, E, K + 1, 128 - intermediate)

    gu = _dequant(gu_w, gu_s)
    dn = _dequant(dn_w, dn_s)

    def oracle_shared(x_bf16: torch.Tensor) -> torch.Tensor:
        xf = x_bf16.to(torch.float32)
        gate = xf @ gu[:intermediate].T
        up = xf @ gu[intermediate:].T
        act = torch.nn.functional.silu(gate) * up
        return act @ dn.T

    probe_attr = "fused_moe_supports_heterogeneous_shared_expert_shape"
    orig_probe = getattr(_aiter_ops.rocm_aiter_ops, probe_attr)

    def run_fallback(x, tw, ti):
        try:
            setattr(_aiter_ops.rocm_aiter_ops, probe_attr, lambda *a, **k: False)
            return experts.forward_modular(x, tw, ti)
        finally:
            setattr(_aiter_ops.rocm_aiter_ops, probe_attr, orig_probe)

    def mean_abs(a, b) -> float:
        return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()

    try:
        for m in MS:
            torch.manual_seed(1000 + m)
            x = torch.randn(m, H, dtype=torch.bfloat16, device=dev)
            rids = torch.randint(0, E_R, (m, K), device=dev, dtype=torch.int32)
            rw = torch.rand(m, K, device=dev)
            rw = rw / rw.sum(-1, keepdim=True)
            ids = torch.cat(
                [rids, torch.full((m, 1), E_R, device=dev, dtype=torch.int32)], dim=1
            )
            w1 = torch.cat([rw, torch.ones(m, 1, device=dev)], dim=1)
            w_tiny = torch.cat(
                [torch.full((m, K), 1e-9, device=dev), torch.ones(m, 1, device=dev)],
                dim=1,
            )

            assert orig_probe(m, *experts._fhmoe_shape_key) is True
            out_shared = experts.forward_modular(x, w_tiny, ids)
            out_full = experts.forward_modular(x, w1, ids)
            out_fb = run_fallback(x, w1, ids)
            mlp_out = shared(x)
            oracle = oracle_shared(x)

            mlp_err = mean_abs(mlp_out, oracle)
            shared_err = mean_abs(out_shared, oracle)
            assert mlp_err < 0.15, (m, mlp_err)
            assert shared_err <= mlp_err + 0.05, (m, mlp_err, shared_err)
            assert mean_abs(out_full, out_fb) < 0.05, m
    finally:
        _reset_aiter_topk_singleton()


# ---------------------------------------------------------------------------
# Router gating (pure logic, any platform)
# ---------------------------------------------------------------------------

from vllm.forward_context import override_forward_context  # noqa: E402


class _Ctx:
    """Minimal forward-context stand-in: a one-entry attn_metadata dict
    whose common metadata carries max_query_len."""

    def __init__(self, max_query_len: int):
        self.attn_metadata = {"_": SimpleNamespace(max_query_len=max_query_len)}


def _router(min_tokens=256, max_tokens=2048, prefill_only=True):
    r = FusedTopKBiasRouter(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=torch.randn(384),
        renormalize=True,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=1,
    )
    r.shared_expert_append_min_tokens = min_tokens
    r.shared_expert_append_max_tokens = max_tokens
    r.shared_expert_append_prefill_only = prefill_only
    return r


@pytest.mark.parametrize(
    ("min_tokens", "max_tokens", "num_tokens", "expected"),
    [
        (256, 2048, 256, False),  # at the dual-stream threshold: stock
        (256, 2048, 257, True),  # just above the threshold: fusion
        (256, 2048, 2048, True),  # ceiling inclusive
        (256, 2048, 4096, False),  # above the window
        (256, 0, 1 << 20, True),  # 0 = unbounded
        (0, 2048, 1, True),  # 0 = no minimum
    ],
)
def test_append_gate_truth_table(min_tokens, max_tokens, num_tokens, expected) -> None:
    with override_forward_context(_Ctx(1 << 20)):
        assert (
            _router(min_tokens, max_tokens)._append_shared_experts_for(num_tokens)
            is expected
        )


@pytest.mark.parametrize(("max_tokens", "num_tokens"), [(2048, 8), (0, 1 << 20)])
def test_append_gate_holds_during_decode(max_tokens, num_tokens) -> None:
    """Decode forwards never append, regardless of the size gates."""
    with override_forward_context(_Ctx(1)):
        assert _router(256, max_tokens)._append_shared_experts_for(num_tokens) is False


def test_env_parsing(monkeypatch: pytest.MonkeyPatch):
    from vllm.models.deepseek_v4.amd.model import (
        _heterogeneous_shared_expert_max_tokens,
        _heterogeneous_shared_expert_min_tokens,
    )

    monkeypatch.delenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", raising=False)
    assert _heterogeneous_shared_expert_min_tokens() == 256
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "64")
    assert _heterogeneous_shared_expert_min_tokens() == 64

    monkeypatch.delenv("VLLM_HETERO_SHARED_EXPERT_MAX_TOKENS", raising=False)
    assert _heterogeneous_shared_expert_max_tokens() == 2048
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MAX_TOKENS", "0")
    assert _heterogeneous_shared_expert_max_tokens() == 0
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MAX_TOKENS", "abc")
    assert _heterogeneous_shared_expert_max_tokens() == 2048


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the top-k kernel only runs on GPU"
)
@pytest.mark.parametrize(("num_experts", "topk", "m"), [(64, 6, 128), (384, 6, 2048)])
def test_router_append_bit_exact(num_experts: int, topk: int, m: int) -> None:
    """The in-kernel shared-slot append is bit-identical to the reference
    top-k + cat append."""
    from vllm.model_executor.layers.fused_moe.router import (
        fused_topk_bias_router as router_mod,
    )

    vllm_topk_softplus_sqrt = router_mod.vllm_topk_softplus_sqrt
    _get_fused_shared_topk_buffers = router_mod._get_fused_shared_topk_buffers

    torch.manual_seed(0)
    device = torch.device("cuda")
    gating = torch.randn(m, num_experts, device=device, dtype=torch.float32)
    bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    base, weight, rsf = num_experts, 1.0 / 1.5, 2.5

    # Reference: top-k into fresh tensors, then cat-append the shared slot.
    ref_w = torch.empty(m, topk, dtype=torch.float32, device=device)
    ref_ids = torch.empty(m, topk, dtype=torch.int32, device=device)
    tei = torch.empty(m, topk, dtype=torch.int32, device=device)
    with patch.object(router_mod.envs, "VLLM_MOE_SKIP_PADDING", False):
        vllm_topk_softplus_sqrt(
            ref_w,
            ref_ids,
            tei,
            gating,
            True,
            e_score_correction_bias=bias,
            routed_scaling_factor=rsf,
        )
    ref_w = torch.cat([ref_w, torch.full((m, 1), weight, device=device)], dim=1)
    ref_ids = torch.cat(
        [ref_ids, torch.full((m, 1), base, device=device, dtype=torch.int32)], dim=1
    )

    # Fast path: the append lands in the persistent buffer.
    bufs = _get_fused_shared_topk_buffers(m, topk, 1, base, device, torch.int32)
    assert bufs is not None
    ids_buf, w_buf = bufs
    fast_ids, fast_w = ids_buf[:m], w_buf[:m]
    with patch.object(router_mod.envs, "VLLM_MOE_SKIP_PADDING", False):
        vllm_topk_softplus_sqrt(
            fast_w,
            fast_ids,
            tei,
            gating,
            True,
            e_score_correction_bias=bias,
            routed_scaling_factor=rsf,
            num_fused_shared_experts=1,
            shared_expert_base=base,
            shared_expert_weight=weight,
        )
    torch.testing.assert_close(fast_w, ref_w)
    torch.testing.assert_close(fast_ids, ref_ids)
    assert torch.all(fast_ids[:, topk:] == base)
    assert torch.all(fast_w[:, topk:] == weight)
