# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN R3: monolithic trtllm-gen NVFP4 MoE backend under allgather-EP.

Selection-level tests for the ``VLLM_GDN_MOE_MONO_AGRS`` gate on
``TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config``:

- TP4+EP (allgather_reducescatter all2all, sequence-parallel MoE):
  flag=1 (opt-in) -> Monolithic accepted, wins oracle ordering, and
  pairs with ``MoEPrepareAndFinalizeNaiveDPEPMonolithic``;
  flag unset/=0 (default) -> exact previous behavior (Monolithic rejected ->
  Modular + ``MoEPrepareAndFinalizeNaiveDPEPModular``).
- TP1 (no EP, no all2all): Monolithic accepted under BOTH flag values —
  TP1 selection is provably untouched by the gate.
- Every other all2all backend and the EPLB case remain rejected under both
  flag values.

The predicate/factory tests are pure-CPU. The oracle-path test exercises
``is_supported_config`` exactly as ``select_nvfp4_moe_backend`` does and
therefore requires a Blackwell GPU + flashinfer (skipped elsewhere).
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
    MoEPrepareAndFinalizeNaiveDPEPMonolithic,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)

FLAG = "VLLM_GDN_MOE_MONO_AGRS"


def _parallel_config(
    *,
    ep_size: int = 1,
    sp_size: int = 1,
    dp_size: int = 1,
    all2all_backend: str = "allgather_reducescatter",
    enable_eplb: bool = False,
) -> FusedMoEParallelConfig:
    """Build the config exactly as FusedMoEParallelConfig.make does.

    TP4 + --enable-expert-parallel + allgather_reducescatter (the GDN serve
    shape) flattens to: tp_size=1, ep_size=4, dp_size=1, sp_size=4
    (use_sequence_parallel_moe), so use_all2all_kernels is True via the
    sequence-parallel branch.
    """
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=ep_size > 1,
        all2all_backend=all2all_backend,
        enable_eplb=enable_eplb,
    )


def _tp4_ep_agrs(**kwargs) -> FusedMoEParallelConfig:
    return _parallel_config(ep_size=4, sp_size=4, **kwargs)


def _tp1() -> FusedMoEParallelConfig:
    return _parallel_config()


def _set_flag(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)


def _select_trtllm_kernel(moe_parallel_config) -> type[mk.FusedMoEExperts]:
    """First kernel class whose parallel predicate accepts the config, in
    oracle preference order (mirrors select_nvfp4_moe_backend's loop for the
    parallel-config axis)."""
    for k_cls in backend_to_kernel_cls(NvFp4MoeBackend.FLASHINFER_TRTLLM):
        if k_cls._supports_parallel_config(moe_parallel_config):
            return k_cls
    raise AssertionError("no trtllm kernel accepted the parallel config")


class TestParallelConfigSanity:
    def test_tp4_ep_agrs_shape(self):
        cfg = _tp4_ep_agrs()
        assert cfg.use_all2all_kernels
        assert cfg.use_ag_rs_all2all_kernels

    def test_tp1_shape(self):
        cfg = _tp1()
        assert not cfg.use_all2all_kernels
        assert not cfg.use_ag_rs_all2all_kernels

    def test_oracle_prefers_monolithic(self):
        # Selection order is Monolithic first — required for the flag to
        # change the outcome at all.
        assert backend_to_kernel_cls(NvFp4MoeBackend.FLASHINFER_TRTLLM) == [
            TrtLlmNvFp4ExpertsMonolithic,
            TrtLlmNvFp4ExpertsModular,
        ]


class TestMonoAgrsPredicate:
    @pytest.mark.parametrize("flag", ["1"])
    def test_tp4_ep_agrs_flag_on_selects_monolithic(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        cfg = _tp4_ep_agrs()
        assert TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg)
        assert _select_trtllm_kernel(cfg) is TrtLlmNvFp4ExpertsMonolithic

    @pytest.mark.parametrize("flag", [None, "0"])
    def test_tp4_ep_agrs_flag_off_selects_modular(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        cfg = _tp4_ep_agrs()
        assert not TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg)
        # Exact previous behavior: fall through to the Modular kernel.
        assert TrtLlmNvFp4ExpertsModular._supports_parallel_config(cfg)
        assert _select_trtllm_kernel(cfg) is TrtLlmNvFp4ExpertsModular

    @pytest.mark.parametrize("flag", [None, "0", "1"])
    def test_tp1_selects_monolithic_under_both_flags(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        cfg = _tp1()
        assert TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg)
        assert _select_trtllm_kernel(cfg) is TrtLlmNvFp4ExpertsMonolithic

    @pytest.mark.parametrize("flag", [None, "0", "1"])
    @pytest.mark.parametrize(
        "backend",
        [
            "deepep_high_throughput",
            "deepep_low_latency",
            "flashinfer_all2allv",
            "naive",
        ],
    )
    def test_other_all2all_backends_still_rejected(self, monkeypatch, flag, backend):
        _set_flag(monkeypatch, flag)
        cfg = _tp4_ep_agrs(all2all_backend=backend)
        assert not TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg)

    @pytest.mark.parametrize("flag", [None, "0", "1"])
    def test_eplb_still_rejected(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        cfg = _tp4_ep_agrs(enable_eplb=True)
        assert not TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg)
        # EPLB without any all2all is also rejected (pre-existing behavior).
        cfg_no_a2a = _parallel_config(enable_eplb=True)
        assert not TrtLlmNvFp4ExpertsMonolithic._supports_parallel_config(cfg_no_a2a)


class TestPrepareFinalizePairing:
    """The oracle passes use_monolithic=issubclass(experts_cls,
    FusedMoEExpertsMonolithic) into maybe_make_prepare_finalize, which for the
    ag_rs backend calls make_moe_prepare_and_finalize_naive_dp_ep. Verify the
    subclass relationships and the factory mapping."""

    def test_kernel_interface_subclassing(self):
        assert issubclass(TrtLlmNvFp4ExpertsMonolithic, mk.FusedMoEExpertsMonolithic)
        assert not issubclass(TrtLlmNvFp4ExpertsModular, mk.FusedMoEExpertsMonolithic)

    def test_factory_monolithic(self):
        pf = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=True, is_sequence_parallel=True, num_dispatchers=4
        )
        assert type(pf) is MoEPrepareAndFinalizeNaiveDPEPMonolithic
        assert pf.num_dispatchers() == 4

    def test_factory_modular(self):
        pf = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=False, is_sequence_parallel=True, num_dispatchers=4
        )
        assert type(pf) is MoEPrepareAndFinalizeNaiveDPEPModular


def _fake_moe_config(moe_parallel_config) -> SimpleNamespace:
    """Just the attributes is_supported_config reads."""
    return SimpleNamespace(
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        moe_parallel_config=moe_parallel_config,
        routing_method=RoutingMethodType.Renormalize,
        router_logits_dtype=torch.bfloat16,
        hidden_dim=2048,
        is_lora_enabled=False,
    )


def _oracle_select(moe_parallel_config) -> type[mk.FusedMoEExperts]:
    """Mirror select_nvfp4_moe_backend's _return_or_raise over the trtllm
    kernel list, through the real is_supported_config entry point."""
    moe_config = _fake_moe_config(moe_parallel_config)
    for k_cls in backend_to_kernel_cls(NvFp4MoeBackend.FLASHINFER_TRTLLM):
        supported, _ = k_cls.is_supported_config(
            k_cls,
            moe_config,
            kNvfp4Static,
            kNvfp4Dynamic,
            mk.FusedMoEActivationFormat.Standard,
        )
        if supported:
            return k_cls
    raise AssertionError("no trtllm kernel supported the config")


requires_trtllm_device = pytest.mark.skipif(
    not TrtLlmNvFp4ExpertsMonolithic._supports_current_device(),
    reason="requires Blackwell GPU with flashinfer trtllm fused MoE",
)


@requires_trtllm_device
class TestOracleSelection:
    """Full is_supported_config path (device + quant + routing + parallel)."""

    @pytest.mark.parametrize("flag", [None, "1"])
    def test_tp4_ep_agrs_flag_on(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        assert _oracle_select(_tp4_ep_agrs()) is TrtLlmNvFp4ExpertsMonolithic

    def test_tp4_ep_agrs_flag_off(self, monkeypatch):
        _set_flag(monkeypatch, "0")
        assert _oracle_select(_tp4_ep_agrs()) is TrtLlmNvFp4ExpertsModular

    @pytest.mark.parametrize("flag", [None, "0", "1"])
    def test_tp1_both_flags(self, monkeypatch, flag):
        _set_flag(monkeypatch, flag)
        assert _oracle_select(_tp1()) is TrtLlmNvFp4ExpertsMonolithic
