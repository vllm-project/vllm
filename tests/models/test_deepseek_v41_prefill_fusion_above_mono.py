# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-fusion-above-mono unit tests (zero-GPU).

Object under test: the re-scope of the heterogeneous fused-shared-experts
yield to ``VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO`` — under
``--monokernel`` the hetero construction is now ALLOWED (mono takes
decode-only steps of <= 48 rows; the fused hetero FHMoE engages only for
prefill-shaped batches), and the mono decode kernels read the routed
buffers through the stock 384-expert views
(``DeepseekV4HeterogeneousSharedRoutedExperts.mono_stock_moe_views``)
that slice the appended (never-read) shared slot off.

Covered:
* the kill switch restores the stock process-global yield under mono;
* with the gate armed the eligibility chain PROCEEDS past the mono
  branch (the yield no longer fires);
* the kill var is inert without ``--monokernel``;
* ``VLLM_DISABLE_FUSED_SHARED_EXPERTS`` still kills first (inheritance);
* ``mono_stock_moe_views`` recovers byte-identical stock-row views with
  the stock shapes, at a small synthetic shape AND at the serving shape
  (via meta tensors, checked against the mono runner's own ``want``
  contract at TP4), and rejects a slot-less (stock) buffer loudly;
* the rocm.py platform hook defaults the AITER coordination flag ON for
  the DeepSeek native FP8-block-32 combo — under armed mono and on the
  classic (non-mono) path — never over an explicit opt-in/out or the
  kill switch;
* ``VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS`` parsing and its wiring into
  the router.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.amd import model as v4_amd_model
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4HeterogeneousSharedRoutedExperts,
    _heterogeneous_shared_expert_enabled,
)

KILL_VAR = "VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO"
FSE_VAR = "VLLM_DISABLE_FUSED_SHARED_EXPERTS"

#: Verbatim ``quantization_config`` of the DeepSeek-V4.1-Flash checkpoint
#: (the native FP8-block-32 + MXFP4-routed combo the platform hook scopes
#: its default to).
_REAL_QUANT_CFG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _real_dsv41_flash_vllm_config(*, monokernel: bool):
    """Attribute-duck vllm_config carrying the REAL DeepSeek-V4.1-Flash TP4
    serving shape (5120/2304/384/6/1, silu, noaux_tc) and the resolved
    deepseek_v4_fp8 quant state, so the full eligibility chain can run
    zero-GPU with only the device/AITER probes mocked."""
    hf_config = SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        hidden_act="silu",
        topk_method="noaux_tc",
        hidden_size=5120,
        moe_intermediate_size=2304,
    )
    quant_config = SimpleNamespace(
        get_name=lambda: "deepseek_v4_fp8",
        moe_quant_algo="fp4",
        weight_block_size=[32, 32],
        is_checkpoint_fp8_serialized=True,
        is_scale_e8m0=True,
        ignored_layers=None,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=False,
            enable_eplb=False,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        kernel_config=SimpleNamespace(monokernel=monokernel, moe_backend="aiter"),
        offload_config=None,
    )


def _mono_cfg():
    return _real_dsv41_flash_vllm_config(monokernel=True)


def _plain_cfg():
    return _real_dsv41_flash_vllm_config(monokernel=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KILL_VAR, raising=False)
    monkeypatch.delenv(FSE_VAR, raising=False)


class _Sentinel(Exception):
    pass


def _arm_downstream_sentinels(monkeypatch):
    """Make everything AFTER the mono branch raise: reaching it proves the
    yield did not fire; the yield returning False proves it was not reached
    (a return before the sentinels can only come from the kill-var checks,
    which run in a fixed order: FSE_VAR first, then the mono branch)."""

    def boom(*args, **kwargs):
        raise _Sentinel("the chain proceeded past the mono branch")

    monkeypatch.setattr(v4_amd_model, "on_gfx950", boom, raising=False)
    monkeypatch.setattr(
        v4_amd_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )


def test_yield_restored_by_kill_var(monkeypatch):
    monkeypatch.setenv(KILL_VAR, "1")
    _arm_downstream_sentinels(monkeypatch)
    assert _heterogeneous_shared_expert_enabled(_mono_cfg()) is False


def test_mono_branch_proceeds_when_armed(monkeypatch):
    _arm_downstream_sentinels(monkeypatch)
    with pytest.raises(_Sentinel, match="proceeded past the mono branch"):
        _heterogeneous_shared_expert_enabled(_mono_cfg())


def test_kill_var_inert_without_mono(monkeypatch):
    monkeypatch.setenv(KILL_VAR, "1")
    _arm_downstream_sentinels(monkeypatch)
    with pytest.raises(_Sentinel, match="proceeded past the mono branch"):
        _heterogeneous_shared_expert_enabled(_plain_cfg())


def test_fused_shared_experts_kill_inherits(monkeypatch):
    """The fused-shared-experts kill switch still wins under mono with the
    gate armed: the shared var must disable the fusion regardless of the
    mono branch."""
    monkeypatch.setenv(FSE_VAR, "1")
    _arm_downstream_sentinels(monkeypatch)
    assert _heterogeneous_shared_expert_enabled(_mono_cfg()) is False


def _synthetic_buffers(n: int, inter: int, hidden: int, down_cols: int):
    """(n + 1)-wide routed buffers in the loaded A8W4 layout, tagged so
    every row is distinguishable."""
    rows = n + 1
    w13 = torch.arange(rows * 2 * inter * (hidden // 2), dtype=torch.uint8)
    w13 = w13.reshape(rows, 2 * inter, hidden // 2)
    w2 = torch.arange(rows * hidden * (inter // 2), dtype=torch.uint8)
    w2 = w2.reshape(rows, hidden, inter // 2)
    w13_s = torch.arange(rows * 2 * inter * 2, dtype=torch.uint8).reshape(
        rows * 2 * inter, 2
    )
    w2_s = torch.arange(rows * hidden * down_cols, dtype=torch.uint8).reshape(
        rows * hidden, down_cols
    )
    return SimpleNamespace(
        shared_expert_id=n,
        w13_weight=w13,
        w13_weight_scale=w13_s,
        w2_weight=w2,
        w2_weight_scale=w2_s,
    )


def test_mono_stock_moe_views_bytes_and_shapes():
    n, inter, hidden, down_cols = 8, 32, 64, 3
    buf = _synthetic_buffers(n, inter, hidden, down_cols)
    w13, w13_s, w2, w2_s = (
        DeepseekV4HeterogeneousSharedRoutedExperts.mono_stock_moe_views(buf)
    )
    # byte-identical to the routed rows; the appended slot is gone
    assert torch.equal(w13, buf.w13_weight[:n])
    assert torch.equal(w2, buf.w2_weight[:n])
    assert torch.equal(w13_s, buf.w13_weight_scale[: n * 2 * inter])
    assert torch.equal(w2_s, buf.w2_weight_scale[: n * hidden])
    # stock shapes and contiguity (what MonoLayerWeights.check demands)
    assert w13.shape == (n, 2 * inter, hidden // 2)
    assert w2.shape == (n, hidden, inter // 2)
    assert w13_s.shape == (n * 2 * inter, w13_s.shape[-1])
    assert w2_s.shape == (n * hidden, down_cols)
    assert w13.is_contiguous() and w2.is_contiguous()
    assert w13_s.is_contiguous() and w2_s.is_contiguous()
    assert getattr(w13, "is_shuffled", None) is True
    assert getattr(w2, "is_shuffled", None) is True


def test_mono_stock_moe_views_serving_shape_contract():
    """At the SERVING shape (TP4: every rank holds all 384 routed experts,
    intermediate 2304/4 = 576 padded to 640), the views must satisfy the
    mono runner's own weight contract exactly — meta tensors carry the
    shapes without the ~2 GB of real buffers."""
    # CPU-importable mono geometry modules (moe_shape/runner pull in
    # flydsl, which needs a GPU driver): EXPERTS/TOPK are the frozen
    # 384/6 routing the mono build keys on (mono_decode.create refuses
    # anything else).
    from vllm.models.deepseek_v41.amd.mono.attention.plan import HIDDEN
    from vllm.models.deepseek_v41.amd.mono.stages.dims import Dims as MoeDims

    EXPERTS = 384
    tp = 4
    d = MoeDims(tp)
    n = EXPERTS
    buf = SimpleNamespace(
        shared_expert_id=n,
        w13_weight=torch.empty(
            (n + 1, 2 * d.inter, HIDDEN // 2),
            dtype=torch.float4_e2m1fn_x2,
            device="meta",
        ),
        w13_weight_scale=torch.empty(
            ((n + 1) * 2 * d.inter, HIDDEN // 32), dtype=torch.uint8, device="meta"
        ),
        w2_weight=torch.empty(
            (n + 1, HIDDEN, d.inter // 2),
            dtype=torch.float4_e2m1fn_x2,
            device="meta",
        ),
        w2_weight_scale=torch.empty(
            ((n + 1) * HIDDEN, d.down_scale_cols),
            dtype=torch.uint8,
            device="meta",
        ),
    )
    w13, w13_s, w2, w2_s = (
        DeepseekV4HeterogeneousSharedRoutedExperts.mono_stock_moe_views(buf)
    )
    fp4 = torch.float4_e2m1fn_x2
    assert w13.shape == (EXPERTS, 2 * d.inter, HIDDEN // 2) and w13.dtype == fp4
    assert w13_s.shape == (EXPERTS * 2 * d.inter, HIDDEN // 32)
    assert w13_s.dtype == torch.uint8
    assert w2.shape == (EXPERTS, HIDDEN, d.inter // 2) and w2.dtype == fp4
    assert w2_s.shape == (EXPERTS * HIDDEN, d.down_scale_cols)
    assert w2_s.dtype == torch.uint8


def test_mono_stock_moe_views_rejects_wrong_layout():
    """A buffer without the appended slot (a stock build reaching the
    method by mistake) must fail loudly, never slice silently."""
    buf = _synthetic_buffers(8, 32, 64, 3)
    buf.w13_weight = buf.w13_weight[:8]  # stock 8 rows, no append
    with pytest.raises(RuntimeError, match="routed \\+ 1 appended"):
        DeepseekV4HeterogeneousSharedRoutedExperts.mono_stock_moe_views(buf)


# ---------------------------------------------------------------------------
# AITER-flag coordination: the platform defaults
# VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS ON under mono (rocm.py
# _maybe_default_fused_shared_experts), the FP8->MXFP4 requant fusion is
# refused under mono at the model level (_fuse_shared_experts_enabled),
# and an armed-but-blocked eligibility logs "ARMED but unavailable"
# rather than silently no-op'ing.
# ---------------------------------------------------------------------------


def _mock_eligibility_probes(monkeypatch, *, aiter_flag: bool):
    monkeypatch.setattr(
        v4_amd_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: aiter_flag,
    )
    monkeypatch.setattr(v4_amd_model, "on_gfx950", lambda: True, raising=False)
    monkeypatch.setattr(
        v4_amd_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert_shape",
        lambda *a, **k: True,
    )


def test_full_chain_true_when_armed_under_mono(monkeypatch, caplog):
    """Armed mono + the AITER flag coordinated ON + every other check green
    -> eligibility True, BOTH the engaged WARNING and the 'is ENABLED' line
    fire."""
    _mock_eligibility_probes(monkeypatch, aiter_flag=True)
    with caplog.at_level(logging.WARNING, logger=v4_amd_model.__name__):
        got = _heterogeneous_shared_expert_enabled(
            _real_dsv41_flash_vllm_config(monokernel=True)
        )
    assert got is True
    assert any("prefill fusion above mono engaged" in m for m in caplog.messages)
    assert any("is ENABLED for" in m for m in caplog.messages)


def test_armed_but_aiter_flag_off_warns(monkeypatch, caplog):
    """Armed mono with the AITER coordination flag OFF must NOT silently
    no-op — it returns False AND logs 'ARMED but unavailable'."""
    _mock_eligibility_probes(monkeypatch, aiter_flag=False)
    with caplog.at_level(logging.WARNING, logger=v4_amd_model.__name__):
        got = _heterogeneous_shared_expert_enabled(
            _real_dsv41_flash_vllm_config(monokernel=True)
        )
    assert got is False
    assert any("ARMED but unavailable" in m for m in caplog.messages)
    assert not any("prefill fusion above mono engaged" in m for m in caplog.messages)


def test_armed_but_blocked_reason_warns(monkeypatch, caplog):
    """Armed mono + flag ON + a downstream reason (quant config mismatch)
    -> False with the reason surfaced in the ARMED-but-unavailable line."""
    _mock_eligibility_probes(monkeypatch, aiter_flag=True)
    cfg = _real_dsv41_flash_vllm_config(monokernel=True)
    cfg.quant_config = SimpleNamespace(
        get_name=lambda: "something_else",
        moe_quant_algo="fp4",
        weight_block_size=[32, 32],
        is_checkpoint_fp8_serialized=True,
        is_scale_e8m0=True,
        ignored_layers=None,
    )
    with caplog.at_level(logging.WARNING, logger=v4_amd_model.__name__):
        got = _heterogeneous_shared_expert_enabled(cfg)
    assert got is False
    blocked = [m for m in caplog.messages if "ARMED but unavailable" in m]
    assert blocked and "quantization config is not active" in blocked[-1]


def test_fuse_shared_experts_requant_refused_under_mono(monkeypatch):
    """The FP8->MXFP4 requant fusion is refused under --monokernel even with
    the AITER flag ON and a shared-expert checkpoint: it consumes the
    shared-expert FP8 weights into the routed MXFP4 table, which the mono
    decode kernels read in their checkpoint layout."""
    from vllm.models.deepseek_v4.amd.model import _fuse_shared_experts_enabled

    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    config = SimpleNamespace(n_shared_experts=1)
    real_cfg = _real_dsv41_flash_vllm_config(monokernel=True)
    assert _fuse_shared_experts_enabled(config, real_cfg) is False
    # Non-mono keeps the env-driven behavior (the historical default path).
    assert (
        _fuse_shared_experts_enabled(
            config, _real_dsv41_flash_vllm_config(monokernel=False)
        )
        is True
    )


def test_maybe_default_env_under_mono_p5_armed(monkeypatch):
    """The platform defaults VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
    under mono with the gate armed (the hetero construction needs the
    flag), and declines when the kill var is set under mono (the stock
    behavior)."""
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
    from vllm.platforms.rocm import RocmPlatform

    monkeypatch.delenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", raising=False)
    monkeypatch.setattr(
        DeepseekV4FP8Config,
        "_is_native_fp8_block32_mxfp4_shared",
        classmethod(lambda cls, qc: True),
    )
    cfg = _real_dsv41_flash_vllm_config(monokernel=True)
    cfg.model_config.hf_config.model_type = "deepseek_v41"
    cfg.model_config.hf_config.quantization_config = {"mocked": True}
    try:
        RocmPlatform._maybe_default_fused_shared_experts(cfg, True)
        assert os.environ.get("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS") == "1", (
            "armed mono must default the AITER coordination flag ON"
        )
        # Kill var set under mono: the defaulting declines.
        del os.environ["VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"]
        monkeypatch.setenv("VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO", "1")
        RocmPlatform._maybe_default_fused_shared_experts(cfg, True)
        assert "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS" not in os.environ, (
            "killed mono must keep the flag unset (stock behavior)"
        )
    finally:
        os.environ.pop("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", None)


# ---------------------------------------------------------------------------
# Min-tokens knob and the classic (non-mono) arm of the platform default
# (folded in from the deleted test_deepseek_v41_fse_plumbing.py).
# ---------------------------------------------------------------------------


def test_hetero_min_tokens_env_parsing(monkeypatch: pytest.MonkeyPatch):
    """VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS parsing for the prefill-only
    gating of the heterogeneous fused path.

    The knob sets the router's shared_expert_append_min_tokens on
    heterogeneous layers: batches below the threshold keep the stock
    overlapped separate shared-expert decode layout. Default 16 matches
    the DSV4.1-Flash serving workload ISL; 0 restores always-fused.
    Invalid values fall back to the default rather than crashing startup.
    """
    from vllm.models.deepseek_v4.amd.model import (
        _heterogeneous_shared_expert_min_tokens,
    )

    monkeypatch.delenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", raising=False)
    assert _heterogeneous_shared_expert_min_tokens() == 16
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "0")
    assert _heterogeneous_shared_expert_min_tokens() == 0
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "64")
    assert _heterogeneous_shared_expert_min_tokens() == 64
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "abc")
    assert _heterogeneous_shared_expert_min_tokens() == 16
    monkeypatch.setenv("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "-5")
    assert _heterogeneous_shared_expert_min_tokens() == 0


def test_hetero_min_tokens_wired_to_router() -> None:
    """The DeepSeek-V4 MoE ctor must hand the threshold to the router on
    heterogeneous layers (the only producer of routed-width top-k below
    the threshold). Repo-grep style."""
    model_path = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "models"
        / "deepseek_v4"
        / "amd"
        / "model.py"
    )
    src = model_path.read_text()
    assert "shared_expert_append_min_tokens = (" in src, (
        "DeepseekV4MoE ctor no longer sets router.shared_expert_append_min_tokens"
    )
    assert "VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS" in src


def test_native_combo_defaults_fusion_env_via_platform_hook() -> None:
    """The platform hook restores the toggle contract (non-mono arm).

    ``RocmPlatform`` defaults ``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``
    to "1" — flipping the env var itself and refreshing the cached AITER
    flags — when (and only when) the model is a DeepSeek v4/v41 native
    FP8-block-32 checkpoint with shared experts, AITER fused MoE is
    active, the parallel layout is TP-only, and neither the kill switch
    nor an explicit opt-in/out is set. This is what lets the canonical
    server command (which sets only VLLM_ROCM_USE_AITER and
    VLLM_ROCM_USE_AITER_MOE) exercise the feature, per the documented
    contract "default: unset -> feature ON".
    """
    import vllm.platforms.rocm as rocm_mod
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import RocmPlatform

    fusion_var = "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"
    kill_var = "VLLM_DISABLE_FUSED_SHARED_EXPERTS"
    touched = (fusion_var, kill_var, "VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE")
    saved = {name: os.environ.get(name) for name in touched}

    def eligible_cfg(quant_cfg: dict):
        cfg = _real_dsv41_flash_vllm_config(monokernel=False)
        cfg.model_config.hf_config.model_type = "deepseek_v41"
        cfg.model_config.hf_config.quantization_config = quant_cfg
        return cfg

    try:
        # Eligible configuration: AITER fused MoE on, no explicit FSE env,
        # no kill switch, native-combo quant config with shared experts.
        os.environ.pop(fusion_var, None)
        os.environ.pop(kill_var, None)
        os.environ["VLLM_ROCM_USE_AITER"] = "1"
        os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1"
        rocm_aiter_ops.refresh_env_variables()
        assert not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

        RocmPlatform._maybe_default_fused_shared_experts(
            eligible_cfg(_REAL_QUANT_CFG), True
        )

        # The env flipped and every cached reader observes it.
        assert os.environ[fusion_var] == "1"
        assert rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

        # Kill switch set: no default-on.
        os.environ.pop(fusion_var, None)
        os.environ[kill_var] = "1"
        rocm_aiter_ops.refresh_env_variables()
        RocmPlatform._maybe_default_fused_shared_experts(
            eligible_cfg(_REAL_QUANT_CFG), True
        )
        assert fusion_var not in os.environ
        assert not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

        # Explicit opt-out is preserved verbatim.
        os.environ.pop(kill_var, None)
        os.environ[fusion_var] = "0"
        rocm_aiter_ops.refresh_env_variables()
        RocmPlatform._maybe_default_fused_shared_experts(
            eligible_cfg(_REAL_QUANT_CFG), True
        )
        assert os.environ[fusion_var] == "0"
        assert not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

        # A falsified native axis does not default the feature on.
        os.environ.pop(fusion_var, None)
        falsified = {**_REAL_QUANT_CFG, "weight_block_size": [128, 128]}
        RocmPlatform._maybe_default_fused_shared_experts(eligible_cfg(falsified), True)
        assert fusion_var not in os.environ
        assert not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

        # The hook stays wired into the platform default chain.
        rocm_src = Path(rocm_mod.__file__).read_text()
        assert (
            "cls._maybe_default_fused_shared_experts(vllm_config, use_aiter_fused_moe)"
            in rocm_src
        ), "apply_config_platform_defaults no longer calls the hook"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        rocm_aiter_ops.refresh_env_variables()
