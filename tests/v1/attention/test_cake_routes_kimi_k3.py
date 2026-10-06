# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the opt-in Cake Kimi-K3 routes (``VLLM_CAKE_ROUTES``)."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.envs as envs
from vllm.utils import cake_routes
from vllm.utils.cake_routes import kimi_k3_mla_decode_admits
from vllm.utils.flashinfer import _fused_kda_decode_has_cake_signature

pytestmark = pytest.mark.cpu_test

FP8 = torch.float8_e4m3fn


def _routes(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("VLLM_CAKE_ROUTES", raising=False)
    else:
        monkeypatch.setenv("VLLM_CAKE_ROUTES", value)
    # envs.__getattr__ may have been wrapped in functools.cache by an engine.
    getattr(envs.__getattr__, "cache_clear", lambda: None)()


def test_routes_unset_selects_nothing(monkeypatch):
    _routes(monkeypatch, None)
    assert cake_routes.cake_routes() == frozenset()
    assert not cake_routes.cake_route_enabled("kda_decode")
    assert not cake_routes.cake_route_enabled("kimi_k3_mla")


def test_routes_parse_names_and_ignore_unknown(monkeypatch):
    _routes(monkeypatch, " kda_decode, kimi_k3_mla ,bogus,")
    assert cake_routes.cake_routes() == {"kda_decode", "kimi_k3_mla"}
    assert cake_routes.cake_route_enabled("kda_decode")
    with pytest.raises(ValueError):
        cake_routes.cake_route_enabled("bogus")


def _mla_args(**over):
    args = dict(
        q=torch.zeros(4, 1, 12, 576, dtype=FP8),
        kv_cache=torch.zeros(3, 64, 576, dtype=FP8),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        compute_capability=(10, 3),
    )
    args.update(over)
    return args


def test_mla_admission_accepts_the_kimi_k3_tp8_fp8_decode_shape():
    assert kimi_k3_mla_decode_admits(**_mla_args())
    assert kimi_k3_mla_decode_admits(**_mla_args(compute_capability=(10, 0)))


@pytest.mark.parametrize(
    "over",
    [
        dict(q=torch.zeros(4, 1, 12, 576, dtype=torch.bfloat16)),
        dict(kv_cache=torch.zeros(3, 64, 576, dtype=torch.bfloat16)),
        dict(kv_cache=torch.zeros(3, 32, 576, dtype=FP8)),
        dict(q=torch.zeros(4, 1, 24, 576, dtype=FP8)),
        dict(q=torch.zeros(4, 1, 96, 576, dtype=FP8)),
        dict(q=torch.zeros(4, 2, 12, 576, dtype=FP8)),
        dict(q=torch.zeros(4, 12, 576, dtype=FP8)),
        dict(kv_lora_rank=256),
        dict(qk_rope_head_dim=128),
        dict(compute_capability=(9, 0)),
        dict(compute_capability=None),
        dict(compute_capability=(12, 0)),
    ],
    ids=[
        "bf16-query",
        "bf16-cache",
        "page-32",
        "heads-24",
        "heads-96-tp1",
        "q-len-2-mtp",
        "compact-3d-query",
        "lora-256",
        "rope-128",
        "sm90",
        "no-capability",
        "sm120",
    ],
)
def test_mla_admission_rejects_other_contracts(over):
    assert not kimi_k3_mla_decode_admits(**_mla_args(**over))


def _mla_impl(flashinfer_mla, *, route_on: bool):
    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = False
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.num_heads = 12
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "fp8"
    impl._cake_kimi_k3_mla = route_on
    impl._cake_kimi_k3_mla_decision = None
    impl._compute_capability = (10, 3)
    impl._mla_counter_bytes = 1
    impl._mla_counter_max_batch = 2
    return impl


def _mla_metadata(max_query_len: int):
    attn_metadata = MagicMock()
    attn_metadata.causal = True
    attn_metadata.num_decodes = 2
    attn_metadata.num_decode_tokens = 2 * max_query_len
    attn_metadata.max_seq_len = 1
    attn_metadata.decode.block_table = torch.zeros(2, 1, dtype=torch.int32)
    attn_metadata.decode.seq_lens = torch.ones(2, dtype=torch.int32)
    attn_metadata.decode.max_query_len = max_query_len
    attn_metadata.decode.query_start_loc = torch.arange(
        0, 2 * max_query_len + 1, max_query_len, dtype=torch.int32
    )
    return attn_metadata


@pytest.mark.parametrize(
    ("route_on", "max_query_len", "expect_cake"),
    [(True, 1, True), (False, 1, False), (True, 2, False)],
    ids=["t1-admitted", "route-off", "mtp-ragged-not-admitted"],
)
def test_mla_forward_passes_backend_cake_only_for_admitted_decodes(
    monkeypatch, route_on, max_query_len, expect_cake
):
    flashinfer_mla = pytest.importorskip(
        "vllm.v1.attention.backends.mla.flashinfer_mla"
    )
    impl = _mla_impl(flashinfer_mla, route_on=route_on)
    attn_metadata = _mla_metadata(max_query_len)
    query = torch.ones(2 * max_query_len, 12, 576, dtype=FP8)
    kv_cache = torch.ones(1, 64, 576, dtype=FP8)
    kernel = MagicMock(
        return_value=torch.ones(2, max_query_len, 12, 512, dtype=torch.bfloat16)
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(
        flashinfer_mla, "_get_multi_ctas_kv_counter_buffer", MagicMock()
    )
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)

    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, MagicMock()
    )

    assert lse is None and output.shape == (2 * max_query_len, 12, 512)
    assert (kernel.call_args.kwargs.get("backend") == "cake") is expect_cake
    if route_on:
        # The admission decision is cached per call-shape signature.
        assert impl._cake_kimi_k3_mla_decision[1] is expect_cake


def test_kda_cake_probe_requires_the_backend_and_state_indices_kwargs():
    def with_cake(x, output, *, backend="auto", state_indices_mode="unique"):
        pass

    def without_cake(x, output):
        pass

    assert _fused_kda_decode_has_cake_signature(with_cake)
    assert not _fused_kda_decode_has_cake_signature(without_cake)
    assert not _fused_kda_decode_has_cake_signature(object())


@pytest.mark.parametrize("route_on", [True, False], ids=["cake", "default"])
def test_kda_fused_decode_passes_cake_kwargs_only_when_the_route_is_on(
    monkeypatch, route_on
):
    kda_mod = pytest.importorskip("vllm.models.kimi_k3.nvidia.kda")
    kernel = MagicMock()
    monkeypatch.setattr(kda_mod, "flashinfer_fused_kda_decode", kernel)
    layer = SimpleNamespace(
        cake_kda_decode=route_on,
        decode_conv1d_weight=torch.zeros(3, 4, 8),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        decode_norm_weight=torch.ones(128),
        gate_lower_bound=None,
        o_norm=SimpleNamespace(eps=1e-6),
    )
    tensors = [torch.zeros(1) for _ in range(8)]

    kda_mod.KimiK3DeltaAttention._flashinfer_fused_kda_decode(layer, *tensors)

    kwargs = kernel.call_args.kwargs
    assert kwargs["output"] is tensors[-1]
    if route_on:
        assert kwargs["backend"] == "cake"
        assert kwargs["state_indices_mode"] == "unique_or_null"
    else:
        assert "backend" not in kwargs and "state_indices_mode" not in kwargs


def test_kda_forward_keeps_the_eager_break_decorator():
    # Regression guard: the decorator is the identity unless breakable
    # CUDA-graph capture is enabled, so a lost decorator is invisible at runtime.
    kda_mod = pytest.importorskip("vllm.models.kimi_k3.nvidia.kda")
    tree = ast.parse(inspect.getsource(kda_mod))
    layer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "KimiK3DeltaAttention"
    )
    forward = next(
        node
        for node in layer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward"
    )
    assert any(
        getattr(decorator, "id", None) == "eager_break_during_capture"
        for decorator in forward.decorator_list
    )
