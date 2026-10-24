# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU numerics for the heterogeneous FHMoE shared expert (ROCm gfx950).

End-to-end check of the DeepSeek-V4.1-Flash heterogeneous fusion on a tiny
layer built registry-faithfully (AITER MXFP4 routed experts + a native-FP8
block-32 shared expert appended as the last FHMoE slot). The tiny config
embeds the checkpoint's verbatim text_config/quantization_config with the
sizes shrunk, so the suite needs no local checkpoint. Two topologies:

* ``intermediate=128`` - no MXFP4 padding.
* ``intermediate=96``  - rounds to 128, exercising the per-slab gate/up
  padding and pad-aware kernels of the real Flash TP4 topology
  (per-rank 576 -> 640).

For every token count M the test checks that
1. AITER's default-config FHMoE shape probe covers the shape,
2. the shared expert computed inside FHMoE (routing weights ~0) matches a
   fp32 dequant oracle as well as the separate FP8 MLP does - i.e. the
   fusion adds no numerics change of its own (no requant, correct layout),
3. the full FHMoE output (routed + shared) matches the separate-path
   fallback (FHMoE disabled) with normal routing weights.

Methodology notes (pitfalls that produce false failures - do not "simplify"
them away):
* Isolate the shared expert with negligible-but-nonzero routed weights
  (1e-9). Random garbage routed weights explode to rms ~3e5 through the
  two MoE stages, and the fp32 atomic accumulation reordering noise
  (~1e-6 relative) on those magnitudes poisons any shared-path
  differential.
* For check 3 keep the routed scales modest (E8M0 byte 120 = 2^-7) so the
  routed contribution stays O(1).
* Never use all-zero routed weights: an AITER edge case corrupts the
  shared contribution at some M. Unreachable in production because
  renormalized routing weights are strictly positive.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from types import SimpleNamespace

import pytest
import torch

from vllm import _aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization import (
    get_quantization_config,
)
from vllm.models.deepseek_v4.amd import model as dsv4_model
from vllm.platforms import current_platform

H, E_R, K = 512, 8, 2
E = E_R + 1
MS = (1, 7, 64, 256, 1024)

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

pytestmark = pytest.mark.skipif(
    not (_rocm and _gfx950 and _aiter),
    reason="heterogeneous FHMoE numerics require ROCm gfx950 with AITER",
)

# ---------------------------------------------------------------------------
# Tiny registry-faithful config (embedded, no checkpoint dependency)
# ---------------------------------------------------------------------------

#: Verbatim ``text_config`` of the real DeepSeek-V4.1-Flash checkpoint
#: ``config.json``. The tiny layer config below shrinks the sizes on
#: this dict, so every field the construction path reads keeps its
#: production value (the recipe of the tiny-checkpoint builder, without
#: the local-checkpoint dependency).
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
      "rope_scaling": {
        "rope_type": "yarn",
        "factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536
      },
      "n_routed_experts": 384,
      "n_shared_experts": 1,
      "num_experts_per_tok": 6,
      "scoring_func": "sqrtsoftplus",
      "topk_method": "noaux_tc",
      "norm_topk_prob": true,
      "routed_scaling_factor": 1.5,
      "sliding_window": 128,
      "compress_ratios": [
        0,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0
      ],
      "compress_rope_theta": 160000,
      "kv_source_layer_ids": [
        2,
        8,
        14,
        20
      ],
      "index_source_layer_ids": [
        2,
        8,
        14,
        20,
        24,
        28,
        32,
        36
      ],
      "index_n_heads": 32,
      "index_head_dim": 128,
      "index_topk": 512,
      "candidate_source_layer_id": 20,
      "candidate_topk_blocks": 2048,
      "candidate_block_size": 8,
      "hc_mult": 4,
      "hc_sinkhorn_iters": 20,
      "hc_eps": 1e-06,
      "engram_layer_ids": [
        1,
        14
      ],
      "engram_num_embeddings": [
        384006168,
        384016682
      ],
      "engram_max_ngram_size": 4,
      "engram_vocab_size": 16000000,
      "engram_n_heads": 8,
      "engram_head_dim": 256,
      "engram_pad_token_id": 2,
      "engram_compressed_vocab_size": 99092,
      "num_nextn_predict_layers": 3,
      "dspark_block_size": 5,
      "dspark_noise_token_id": 128799,
      "dspark_target_layer_ids": [
        37,
        38,
        39
      ],
      "dspark_markov_rank": 256,
      "dspark_n_routed_experts": 128,
      "dspark_num_experts_per_tok": 3
    }
"""

#: Verbatim top-level ``quantization_config`` of the same checkpoint:
#: FP8-E4M3 [32, 32] ue8m0 block scales for the linear/shared layers +
#: MXFP4 routed experts.
_REAL_QUANT_CFG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

#: Sizes shrunk on the real text_config (every key must exist there, so
#: config-schema drift fails loudly instead of masking a synthetic field).
_TINY_OVERRIDES: dict = {
    "num_hidden_layers": 2,
    "hidden_size": 512,
    "moe_intermediate_size": 128,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "engram_layer_ids": [],
    "num_attention_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 2048,
    # Per-layer id lists shortened to the tiny depth (layer indices >= 2 do
    # not exist; compress_ratios needs one entry per layer).
    "compress_ratios": [0, 0],
    # The real checkpoint's first KV source is layer 2 -- outside the tiny
    # depth, so no tiny layer is a source. The entry must stay non-empty:
    # the model __init__ takes an unconditional max() over the list.
    "kv_source_layer_ids": [2],
    "index_source_layer_ids": [],
}

_REAL_TEXT_CONFIG = json.loads(_REAL_TEXT_CONFIG_JSON)


def _tiny_hf_config():
    """(tiny_hf_config, quant_cfg_dict): the real V4.1-Flash config with
    tiny sizes, built from the embedded dicts (no checkpoint on disk)."""
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    assert set(_TINY_OVERRIDES) <= set(_REAL_TEXT_CONFIG), (
        "a tiny override key is not a real config field (schema drift)"
    )
    config = DeepseekV41Config(
        text_config={**_REAL_TEXT_CONFIG, **_TINY_OVERRIDES},
        quantization_config=_REAL_QUANT_CFG,
    )
    return config, dict(config.quantization_config)


def _tiny_vllm_config(config, *, quantization_config):
    """VllmConfig for the tiny layer construction: ``model_config`` is a
    SimpleNamespace carrying exactly the fields the layer code reads (the
    upstream tiny-builder recipe)."""
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
    vllm_config.quant_config = quantization_config
    return vllm_config


def _ensure_dist_initialized() -> None:
    """Bootstrap the single-process distributed state (idempotent):
    world size 1, rank 0, TP=PP=1. Under pytest the dist_init fixture has
    already run and this is a no-op."""
    if torch.distributed.is_initialized():
        return
    from tests.utils import ensure_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    handle, init_file = tempfile.mkstemp(prefix="dsv41_tiny_dist_")
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
    """Clear the AITER topK singleton (module state + lru_cache): it
    initializes on every fused-shared-expert construction and its
    single-init assert trips on a second construction with different
    arguments (e.g. the next test's expert counts)."""
    from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    rocm_aiter_moe.aiter_topK_meta_data = None


def _dequant(w_fp8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    s = scale_e8m0.to(torch.float32)
    s = s.view(n // 32, 1, k // 32, 1).expand(n // 32, 32, k // 32, 32).reshape(n, k)
    return w_fp8.to(torch.float32) * s


def _build(intermediate: int, tag: int):
    _ensure_dist_initialized()
    _reset_aiter_topk_singleton()
    config, quant_cfg = _tiny_hf_config()
    qc = get_quantization_config("deepseek_v4_fp8").from_config(quant_cfg)
    vllm_config = _tiny_vllm_config(config, quantization_config=qc)
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
            prefix=f"model.layers.{tag}.ffn.shared_experts",
        )
        experts = FusedMoEFactory(
            num_experts=E_R,
            top_k=K,
            hidden_size=H,
            intermediate_size=intermediate,
            n_shared_experts=1,
            fuse_shared_experts=True,
            shared_expert_prefix=f"model.layers.{tag}.ffn.shared_experts",
            prefix=f"model.layers.{tag}.ffn.experts",
            quant_config=qc,
            tp_size=1,
            routed_experts_cls=dsv4_model.DeepseekV4HeterogeneousSharedRoutedExperts,
            routed_experts_args={"shared_expert": shared, "shared_expert_id": E_R},
        ).routed_experts
    return shared, experts


@pytest.fixture()
def aiter_moe_env(monkeypatch: pytest.MonkeyPatch):
    """Enable the AITER MoE path for this test only.

    The env vars and the latched rocm_aiter_ops class flags are restored
    afterwards. Mutating os.environ at module scope instead would leak
    into every other test collected in the same session: distributed
    init and platform probes call refresh_env_variables(), re-latching
    _AITER_ENABLED from the polluted environment for unrelated tests
    (e.g. the FP8 requant suite, whose FusedMoEConfig then selects the
    AITER FP8 backend and rejects its quant scheme).
    """
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("AITER_TRITON_LOG_LEVEL", "ERROR")
    monkeypatch.delenv("VLLM_DISABLE_FUSED_SHARED_EXPERTS", raising=False)
    _aiter_ops.rocm_aiter_ops.refresh_env_variables()
    yield
    _aiter_ops.rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize("intermediate", [128, 96])
def test_heterogeneous_fhmoe_matches_separate_paths(
    intermediate: int, aiter_moe_env: None
) -> None:
    shared, experts = _build(intermediate, tag=intermediate)
    moe = experts.moe_config
    if not moe.rocm_aiter_fmoe_enabled:
        pytest.skip("VLLM_ROCM_USE_AITER_MOE=0 disables the FHMoE path")
    i_sp = moe.intermediate_size_per_partition
    assert i_sp == 128  # MXFP4 rounds both topologies to a 128 multiple
    dev = experts.w13_weight.device

    torch.manual_seed(0)
    experts.w13_weight.data.copy_(
        torch.randint(0, 256, (E, 2 * i_sp, H // 2), dtype=torch.uint8, device=dev)
    )
    # Routed scales small (byte 120 = 2^-7) so routed outputs stay O(1).
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
    # Production weight-processing order: shared linears' ModelOpt post-load
    # (folds the per-row scales to block-32), then the experts' MXFP4
    # shuffle + heterogeneous prepare, then the shared preshuffle.
    shared.gate_up_proj.quant_method.process_weights_after_loading(shared.gate_up_proj)
    shared.down_proj.quant_method.process_weights_after_loading(shared.down_proj)
    experts.quant_method.process_weights_after_loading(experts)
    shared.prepare_gateup_preshuffle()

    # The tiny shape is not DSV4-big: dispatch goes through AITER's
    # default-config path, keyed on the MXFP4-rounded width and the
    # gate/up padding the shared expert grew into.
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

    orig_probe = (
        _aiter_ops.rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert_shape
    )

    probe_attr = "fused_moe_supports_heterogeneous_shared_expert_shape"

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
            # Negligible-but-nonzero routed weights isolate the shared path.
            w_tiny = torch.cat(
                [torch.full((m, K), 1e-9, device=dev), torch.ones(m, 1, device=dev)],
                dim=1,
            )

            # 1. AITER's default configs cover this shape at this M.
            assert orig_probe(m, *experts._fhmoe_shape_key) is True

            out_shared = experts.forward_modular(x, w_tiny, ids)
            out_full = experts.forward_modular(x, w1, ids)
            out_fb = run_fallback(x, w1, ids)
            mlp_out = shared(x)
            oracle = oracle_shared(x)

            # 2. The fused shared expert tracks the fp32 oracle exactly as
            # well as the separate FP8 MLP does: the fusion itself adds no
            # numerics change (no requant, correct GUGU layout).
            mlp_err = mean_abs(mlp_out, oracle)
            shared_err = mean_abs(out_shared, oracle)
            assert mlp_err < 0.15, (m, mlp_err)
            assert shared_err <= mlp_err + 0.05, (m, mlp_err, shared_err)

            # 3. Full output agrees with the separate-path fallback.
            full_err = mean_abs(out_full, out_fb)
            assert full_err < 0.05, (m, full_err)
    finally:
        _reset_aiter_topk_singleton()


def _fill_weights(shared, experts, intermediate: int, i_sp: int, dev):
    """Deterministic O(1)-magnitude weights for the numerics tests."""
    torch.manual_seed(0)
    experts.w13_weight.data.copy_(
        torch.randint(0, 256, (E, 2 * i_sp, H // 2), dtype=torch.uint8, device=dev)
    )
    # Routed scales small (byte 120 = 2^-7) so routed outputs stay O(1).
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
    # Production weight-processing order: shared linears ModelOpt post-load
    # (folds the per-row scales to block-32), then the experts MXFP4
    # shuffle + heterogeneous prepare, then the shared preshuffle.
    shared.gate_up_proj.quant_method.process_weights_after_loading(shared.gate_up_proj)
    shared.down_proj.quant_method.process_weights_after_loading(shared.down_proj)
    experts.quant_method.process_weights_after_loading(experts)
    shared.prepare_gateup_preshuffle()


@pytest.mark.parametrize("intermediate", [128, 96])
@pytest.mark.parametrize("overlap", [True, False])
def test_routed_width_routes_run_separate_shared_path(
    intermediate: int, overlap: bool, aiter_moe_env: None
) -> None:
    """Routed-width (decode-sized) routes dispatch to the separate path.

    Below the model shared_expert_append_min_tokens the router emits
    routed-width (width K) top-k; forward_modular must then run the routed
    experts through the plain AITER kernel and add the separate shared
    expert - on the aux stream when overlap is available (the stock decode
    layout), serially when it is not. Both arms must compute the same
    function as the width-(K+1) fused call with shared weight 1.0:
    routed(rw) + shared(x).
    """
    shared, experts = _build(intermediate, tag=2000 + intermediate)
    moe = experts.moe_config
    if not moe.rocm_aiter_fmoe_enabled:
        pytest.skip("VLLM_ROCM_USE_AITER_MOE=0 disables the FHMoE path")
    i_sp = moe.intermediate_size_per_partition
    assert i_sp == 128  # MXFP4 rounds both topologies to a 128 multiple
    dev = experts.w13_weight.device

    if not overlap:
        # Serial arm of the decode branch (overlap stream disabled).
        experts._overlap_stream = None
    else:
        assert experts._overlap_stream is not None, (
            "overlap arm must exercise the aux-stream code path"
        )

    _fill_weights(shared, experts, intermediate, i_sp, dev)

    def mean_abs(a, b) -> float:
        return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()

    try:
        for m in MS:
            torch.manual_seed(1000 + m)
            x = torch.randn(m, H, dtype=torch.bfloat16, device=dev)
            rids = torch.randint(0, E_R, (m, K), device=dev, dtype=torch.int32)
            rw = torch.rand(m, K, device=dev)
            rw = rw / rw.sum(-1, keepdim=True)

            # Decode branch: routed-width routes, no shared slot.
            out_dec = experts.forward_modular(x, rw, rids)
            assert out_dec.shape == (m, H)

            # Same routing through the width-(K+1) fused call (shared
            # weight 1.0) must produce the same function value.
            ids7 = torch.cat(
                [rids, torch.full((m, 1), E_R, device=dev, dtype=torch.int32)],
                dim=1,
            )
            w7 = torch.cat([rw, torch.ones(m, 1, device=dev)], dim=1)
            out_fused = experts.forward_modular(x, w7, ids7)

            err = mean_abs(out_dec, out_fused)
            assert err < 0.05, (m, overlap, err)
    finally:
        _reset_aiter_topk_singleton()


@pytest.mark.parametrize("intermediate", [128, 96])
def test_decode_branch_cuda_graph_capture(
    intermediate: int, aiter_moe_env: None
) -> None:
    """The decode branch must be CUDA-graph-capturable.

    Breakable CUDA graphs capture decode sizes {1, 2, 4, 8}; every one of
    those captures runs the width-K decode branch with its cross-stream
    pattern (shared expert on the aux stream behind input/output events,
    routed chain on the capture stream, join before the sum). This test
    captures exactly that pattern and proves replay correctness vs an
    eager recomputation with mutated inputs -- the same fenced
    cross-stream capture contract the stock SharedExperts runner
    satisfies in production decode graphs.
    """
    shared, experts = _build(intermediate, tag=3000 + intermediate)
    moe = experts.moe_config
    if not moe.rocm_aiter_fmoe_enabled:
        pytest.skip("VLLM_ROCM_USE_AITER_MOE=0 disables the FHMoE path")
    i_sp = moe.intermediate_size_per_partition
    assert i_sp == 128
    dev = experts.w13_weight.device
    assert experts._overlap_stream is not None, (
        "capture test requires the overlapped decode branch"
    )

    _fill_weights(shared, experts, intermediate, i_sp, dev)

    def mean_abs(a, b) -> float:
        return (a.to(torch.float32) - b.to(torch.float32)).abs().mean().item()

    try:
        m = 8  # decode-graph-like padded size (b=1, DSpark k=5 -> 6 -> 8)
        torch.manual_seed(42)
        x = torch.randn(m, H, dtype=torch.bfloat16, device=dev)
        rids = torch.randint(0, E_R, (m, K), device=dev, dtype=torch.int32)
        rw = torch.rand(m, K, device=dev)
        rw = rw / rw.sum(-1, keepdim=True)

        # Warmup on a side stream (torch graph-capture protocol); this
        # also triggers any lazy AITER JIT for these shapes so capture
        # only records kernel launches.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                experts.forward_modular(x, rw, rids)
        torch.cuda.current_stream().wait_stream(side)
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = experts.forward_modular(x, rw, rids)

        # Mutate the captured input buffers and replay: the graph must
        # track the mutations (decode graphs are replayed with fresh
        # batch contents in the same buffers).
        torch.manual_seed(43)
        x.copy_(torch.randn(m, H, dtype=torch.bfloat16, device=dev))
        rids.copy_(torch.randint(0, E_R, (m, K), device=dev, dtype=torch.int32))
        rw_new = torch.rand(m, K, device=dev)
        rw.copy_(rw_new / rw_new.sum(-1, keepdim=True))

        graph.replay()
        eager = experts.forward_modular(x, rw, rids)
        err = mean_abs(static_out, eager)
        # Same kernels, same order as the eager recompute: only fp32
        # atomic reordering noise is allowed.
        assert err < 0.01, (intermediate, err)
    finally:
        _reset_aiter_topk_singleton()
