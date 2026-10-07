# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing/selection tests for the AITER ASM cprr DCP-verify route.

Companion to the upstream segmented-verify tests. Everything here is pure CPU:
it pins the *decisions* (which route a KV group takes, which head count the
kernel runs at, which configurations are refused) rather than kernel numerics.
The numerics parity test against the segmented route needs a GPU and both
routes built; it is marked and skipped when unavailable.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

rocm_aiter_mla = pytest.importorskip(
    "vllm.v1.attention.backends.mla.rocm_aiter_mla",
    reason="ROCm AITER MLA backend not available",
)

_heads = rocm_aiter_mla._asm_dcp_verify_heads
_configured = rocm_aiter_mla._asm_dcp_verify_configured
_select_route = rocm_aiter_mla._select_dcp_decode_route
Route = rocm_aiter_mla._DCPDecodeRoute
NATIVE = rocm_aiter_mla._NATIVE_CPRR_HEADS
MIN_QLEN = rocm_aiter_mla._MIN_CPRR_QLEN


def test_dcp_verify_env_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", raising=False)
    assert rocm_aiter_mla.envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY == "auto"


@pytest.mark.parametrize("route", ["auto", "asm", "segmented"])
def test_dcp_verify_env_accepts_routes(monkeypatch, route):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", route)
    assert route == rocm_aiter_mla.envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY


def test_dcp_verify_env_rejects_unknown_route(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", "unknown")
    with pytest.raises(ValueError, match="Invalid value"):
        _ = rocm_aiter_mla.envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY


# --------------------------------------------------------------------------
# _asm_dcp_verify_heads -- which head count the cprr kernel runs at
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", NATIVE)
def test_native_head_counts_are_used_as_is(n):
    assert _heads(n) == n


def test_non_native_head_count_pads_up_to_the_next_native():
    """K3 at TP8/DCP8 gathers 96 heads for the target; the kernel has no 96
    variant, so it must run at 128, not silently fall out of the asm path."""
    assert _heads(96) == 128
    assert _heads(1) == 16
    assert _heads(17) == 32
    assert _heads(65) == 128


def test_head_count_above_the_largest_native_is_unservable():
    """0 means 'cannot serve'; anything else would route a shape the kernel
    has no build for."""
    assert _heads(max(NATIVE) + 1) == 0
    assert _heads(256) == 0


# --------------------------------------------------------------------------
# _asm_dcp_verify_configured -- reachability
# --------------------------------------------------------------------------


@pytest.fixture
def on_gfx950(monkeypatch):
    """The cprr kernels are built only for gfx950, so the gate checks the arch.
    Patched rather than detected, so both branches run on any runner."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)


def test_route_needs_dcp(on_gfx950):
    assert _configured(dcp_world_size=1, cp_interleave=1, multi_token_decode=True) is (
        False
    )
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=True) is (
        True
    )


def test_round_robin_interleave_other_than_one_is_excluded(on_gfx950):
    """The kernel's global-position causal window assumes interleave 1, the
    same restriction the segmented gate carries."""
    assert _configured(dcp_world_size=8, cp_interleave=2, multi_token_decode=True) is (
        False
    )
    assert _configured(dcp_world_size=8, cp_interleave=16, multi_token_decode=True) is (
        False
    )


def test_route_needs_multi_token_decode(on_gfx950):
    """Single-token decode is already served. Enabling the route for it would
    pad the head count on every step and buy nothing."""
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=False) is (
        False
    )


def test_route_needs_gfx950(monkeypatch):
    """hsa/gfx942/mla and hsa/gfx1250/mla carry no cprr rows, so off gfx950 the
    kernel lookup fails at the first verify step rather than falling back."""
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: False)
    assert _configured(dcp_world_size=8, cp_interleave=1, multi_token_decode=True) is (
        False
    )


# --------------------------------------------------------------------------
# AiterMLAMetadataBuilder -- the route the env value resolves to
# --------------------------------------------------------------------------

DCP = 8


def _build(monkeypatch, route, gathered_heads, segmented_supported=True):
    """Run the real builder __init__ on CPU at TP8/DCP8 with speculative decoding.

    Stubs only what needs a GPU, AITER, or a DCP group, so the route decision
    under test is the one the builder actually makes.
    """

    def init_common_builder(self, *args, **kwargs):
        self.num_heads = gathered_heads // DCP
        self.dcp_world_size = DCP
        self.reorder_batch_threshold = 5

    monkeypatch.setattr(
        rocm_aiter_mla.MLACommonMetadataBuilder, "__init__", init_common_builder
    )
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_segmented_dcp_verify_supported",
        lambda *args: segmented_supported,
    )
    monkeypatch.setattr(rocm_aiter_mla, "_fp8_mla_prefill_supported", lambda: False)
    monkeypatch.setattr(
        rocm_aiter_mla, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=256),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(
            dtypes=SimpleNamespace(fp8="fp8", fp16="fp16", bf16="bf16"),
            get_mla_metadata_info_v1=lambda *args, **kwargs: tuple(
                (1, torch.int32) for _ in range(6)
            ),
        ),
    )
    if route is None:
        monkeypatch.delenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_AITER_MLA_DCP_VERIFY", route)

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=DCP, cp_kv_cache_interleave_size=1
        ),
        model_config=SimpleNamespace(max_model_len=16, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3", num_gpu_blocks=None),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
    )
    return rocm_aiter_mla.AiterMLAMetadataBuilder(
        kv_cache_spec=SimpleNamespace(block_size=1, dtype=torch.bfloat16),
        layer_names=["layer.0"],
        vllm_config=config,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("route", [None, "auto", "AUTO"])
def test_auto_selects_cprr_for_k3_gathered_heads(on_gfx950, monkeypatch, route):
    """K3 at TP8/DCP8 gathers 96 heads; auto must take cprr padded to 128."""
    builder = _build(monkeypatch, route, gathered_heads=96)
    assert builder._asm_dcp_verify
    assert builder._asm_dcp_verify_heads == 128
    assert builder._num_attention_heads == 128


def test_auto_falls_back_to_segmented_above_largest_native(on_gfx950, monkeypatch):
    builder = _build(monkeypatch, "auto", gathered_heads=256)
    assert not builder._asm_dcp_verify
    assert builder._supports_segmented_dcp_verify


def test_auto_without_any_route_names_the_missing_segmented_build(
    on_gfx950, monkeypatch
):
    with pytest.raises(ValueError) as exc:
        _build(monkeypatch, "auto", gathered_heads=256, segmented_supported=False)
    assert "VLLM_ROCM_AITER_MLA_DCP_VERIFY=auto" in str(exc.value)
    assert "lacks segmented MLA decode" in str(exc.value)


@pytest.mark.parametrize("route", ["asm", "ASM"])
def test_explicit_asm_is_case_insensitive(on_gfx950, monkeypatch, route):
    """env_with_choices returns the raw string; ASM used to mean segmented."""
    assert _build(monkeypatch, route, gathered_heads=96)._asm_dcp_verify


def test_explicit_asm_refuses_unservable_heads(on_gfx950, monkeypatch):
    with pytest.raises(ValueError) as exc:
        _build(monkeypatch, "ASM", gathered_heads=256)
    assert "VLLM_ROCM_AITER_MLA_DCP_VERIFY=asm" in str(exc.value)
    assert "Set VLLM_ROCM_AITER_MLA_DCP_VERIFY=segmented" in str(exc.value)


@pytest.mark.parametrize("route", ["segmented", "SEGMENTED"])
def test_explicit_segmented_never_selects_cprr(on_gfx950, monkeypatch, route):
    assert not _build(monkeypatch, route, gathered_heads=96)._asm_dcp_verify


def test_auto_off_gfx950_stays_segmented(monkeypatch):
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: False)
    assert not _build(monkeypatch, "auto", gathered_heads=96)._asm_dcp_verify


# --------------------------------------------------------------------------
# qlen floor
# --------------------------------------------------------------------------


def test_min_cprr_qlen_is_above_two():
    """Qlen 2 is below CPRR's floor and must use segmented verification."""
    assert MIN_QLEN > 2


@pytest.mark.parametrize(
    "supports,causal,qlen,asm,expected",
    [
        (True, True, 1, True, Route.PLAIN),
        (True, True, 2, True, Route.SEGMENTED),
        (True, True, 3, True, Route.CPRR),
        (True, True, 4, True, Route.CPRR),
        (True, True, 4, False, Route.SEGMENTED),
        (True, False, 2, True, Route.PLAIN),
        (True, False, 4, False, Route.PLAIN),
        (False, True, 3, True, Route.CPRR),
    ],
)
def test_dcp_decode_route(supports, causal, qlen, asm, expected):
    assert _select_route(supports, causal, qlen, asm) is expected


def test_causal_multi_token_batch_without_a_valid_route_fails():
    with pytest.raises(RuntimeError, match="requires either segmented MLA"):
        _select_route(False, True, 2, True)


# Kernel numerics live in test_rocm_aiter_mla_dcp_cprr_numerics.py, which runs
# the real builder and the real asm kernel on a gfx950 GPU.
