# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side logic of the MiniMax-M3 mono decode path that needs no GPU.

The kernel package imports flydsl at module import, so these tests run where it
is installed (the ROCm images); elsewhere they are skipped.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")

from vllm.models.minimax_m3.amd.mono import config as mono_config  # noqa: E402
from vllm.models.minimax_m3.amd.mono import dispatch as mono_dispatch  # noqa: E402
from vllm.models.minimax_m3.amd.mono import envs as mono_envs  # noqa: E402
from vllm.models.minimax_m3.amd.mono.config import (  # noqa: E402
    MAX_CONTEXT,
    MAX_TOKENS,
    SPARSE_BLOCK,
    TP,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.dispatch import (  # noqa: E402
    _config_refusal,
    _step_refusal,
)
from vllm.models.minimax_m3.amd.mono.runner import (  # noqa: E402
    dense_prefix,
    token_rows,
)


def _layer(moe: bool, sparse: bool):
    attn = (
        SimpleNamespace(
            layer_name="attn",
            indexer=SimpleNamespace(index_cache=SimpleNamespace(prefix="idx")),
        )
        if sparse
        else SimpleNamespace()
    )
    return SimpleNamespace(is_moe_layer=moe, self_attn=attn)


def test_token_rows_single_query_token_passes_tables_through():
    bt = torch.arange(12, dtype=torch.int32).view(3, 4)
    seq = torch.tensor([10, 20, 30], dtype=torch.int32)
    rows_bt, rows_seq = token_rows(bt, seq, q=1, n=3)
    assert rows_bt is bt and rows_seq is seq


def test_token_rows_expands_speculative_verify_rows():
    bt = torch.arange(8, dtype=torch.int32).view(2, 4)
    seq = torch.tensor([10, 20], dtype=torch.int32)
    rows_bt, rows_seq = token_rows(bt, seq, q=3, n=6)
    # token j of a request sees seq_len - (q - 1 - j) keys
    assert rows_seq.tolist() == [8, 9, 10, 18, 19, 20]
    assert rows_bt.shape == (6, 4) and torch.equal(rows_bt[:3], bt[0].expand(3, 4))
    assert rows_bt.stride(0) == rows_bt.shape[1]


def test_dense_prefix_counts_leading_dense_ffn_layers():
    layers = [_layer(False, True)] * 3 + [_layer(True, True)] * 5
    assert dense_prefix(layers) == 3
    assert dense_prefix([_layer(True, True)] * 2) == 0


@pytest.mark.parametrize(
    "layers",
    [
        [_layer(False, True)] * 3,  # no MoE layer at all
        [
            _layer(True, True),
            _layer(False, True),
            _layer(True, True),
        ],  # dense after MoE
        [
            _layer(False, True),
            _layer(True, False),
        ],  # MoE layer without sparse attention
    ],
)
def test_dense_prefix_refuses_other_layer_mixes(layers):
    with pytest.raises(MonoUnsupported):
        dense_prefix(layers)


def test_configure_accepts_only_the_two_kv_page_strides():
    saved = (mono_config.KV_PAGE_STRIDE, mono_config.STATIC_KV_SCALE)
    try:
        mono_config.configure(kv_page_stride=16, static_kv_scale=True)
        assert (mono_config.KV_PAGE_STRIDE, mono_config.STATIC_KV_SCALE) == (16, True)
        with pytest.raises(AssertionError):
            mono_config.configure(kv_page_stride=4, static_kv_scale=True)
    finally:
        mono_config.configure(kv_page_stride=saved[0], static_kv_scale=saved[1])


def _vllm_config(**overrides):
    cg = SimpleNamespace(requires_piecewise_compilation=lambda: False)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=TP,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=SPARSE_BLOCK),
        compilation_config=SimpleNamespace(cudagraph_mode=cg),
        model_config=SimpleNamespace(
            max_model_len=32768, hf_text_config=SimpleNamespace()
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: "fp8"
        ),
    )
    for path, value in overrides.items():
        obj, _, attr = path.rpartition(".")
        setattr(eval(f"cfg.{obj}") if obj else cfg, attr, value)
    return cfg


def _model(fse: bool = True, shared_mlp: bool = True, aux=()):
    moe = _layer(True, True)
    moe.block_sparse_moe = SimpleNamespace(
        is_fused_shared_expert_enabled=fse,
        shared_experts=object() if shared_mlp else None,
    )
    return SimpleNamespace(
        is_fused_shared_expert_enabled=fse,
        aux_hidden_state_layers=aux,
        layers=[_layer(False, True)] * 3 + [moe] * 2,
    )


@pytest.mark.parametrize("fse", [True, False])
@pytest.mark.parametrize("aux", [(), (2, 30, 57)])
def test_config_refusal_accepts_the_supported_deployment(fse, aux):
    assert _config_refusal(_vllm_config(), _model(fse=fse, aux=aux)) is None


@pytest.mark.parametrize(
    "cfg, model, why",
    [
        (_vllm_config(**{"parallel_config.tensor_parallel_size": 8}), _model(), "TP"),
        (_vllm_config(**{"cache_config.cache_dtype": "auto"}), _model(), "kv cache"),
        (_vllm_config(**{"cache_config.block_size": 64}), _model(), "block size"),
        (
            _vllm_config(**{"model_config.max_model_len": MAX_CONTEXT + 1}),
            _model(),
            "max_model_len",
        ),
        (_vllm_config(), _model(fse=False, shared_mlp=False), "neither fused nor"),
    ],
)
def test_config_refusal_names_the_unmet_condition(cfg, model, why):
    assert why in _config_refusal(cfg, model)


def _fwd(n: int, q: int = 1, reqs: int | None = None, prefills: int = 0):
    reqs = n // q if reqs is None else reqs
    decode = SimpleNamespace(
        decode_query_len=q,
        block_table=torch.zeros(reqs, 8, dtype=torch.int32),
        seq_lens=torch.ones(reqs, dtype=torch.int32),
    )
    md = SimpleNamespace(decode=decode, num_prefills=prefills)
    return SimpleNamespace(
        ubatch_slices=None,
        attn_metadata={"attn": md},
        slot_mapping={"attn": torch.zeros(n, dtype=torch.int64)},
    )


def test_step_refusal_accepts_a_pure_decode():
    pos = torch.zeros(4, dtype=torch.int64)
    assert _step_refusal(_fwd(4), "attn", 4, pos) is None
    assert _step_refusal(_fwd(6, q=3), "attn", 6, pos) is None


def test_step_refusal_rejects_what_the_kernels_cannot_serve():
    pos = torch.zeros(4, dtype=torch.int64)
    assert "tokens" in _step_refusal(_fwd(MAX_TOKENS + 1), "attn", MAX_TOKENS + 1, pos)
    assert "prefill" in _step_refusal(_fwd(4, prefills=1), "attn", 4, pos)
    assert "per request" in _step_refusal(_fwd(4, q=3, reqs=2), "attn", 4, pos)
    assert "shorter" in _step_refusal(_fwd(4, reqs=2), "attn", 4, pos)
    assert "positions" in _step_refusal(_fwd(4), "attn", 4, pos.to(torch.int32))
    fwd = _fwd(4)
    fwd.attn_metadata["attn"].decode.block_table = fwd.attn_metadata[
        "attn"
    ].decode.block_table.t()
    assert "contiguous" in _step_refusal(fwd, "attn", 4, pos)


# ---- MonoDecode host contract: enable, runner life cycle, slot check, routing ----
# Everything the kernels are not: these pin the dispatcher's behaviour so a refactor
# of the host side (or of the switch) must keep passing them.


class _FakeRunner:
    built = 0

    def __init__(self, model, config, *, stale=False, fp8=False):
        _FakeRunner.built += 1
        self._stale = stale
        self.shared_fp8 = fp8
        self.sparse = []
        self.rank = 0
        self.closed = False
        self.calls = []

    def stale(self):
        return self._stale

    def close(self):
        self.closed = True

    def forward(self, input_ids, positions, inputs_embeds):
        self.calls.append(positions.numel())
        return "mono-out"


@pytest.fixture
def mono(monkeypatch):
    """An enabled dispatcher on a supported deployment, outside graph capture,
    with the runner build stubbed."""
    monkeypatch.setattr(mono_dispatch.envs, "MONO_ENABLE", True)
    monkeypatch.setattr(mono_dispatch.envs, "MONO_CHECK", False)
    monkeypatch.setattr(mono_dispatch, "MonoDecodeRunner", _FakeRunner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _FakeRunner.built = 0
    return mono_dispatch.MonoDecode(_model(), _vllm_config())


def _fwd_shared(n: int, shared: bool = True):
    fwd = _fwd(n)
    slots = fwd.slot_mapping["attn"]
    fwd.slot_mapping["idx"] = slots.clone() if shared else slots + 1
    return fwd


def test_env_flag_parsing(monkeypatch):
    for raw, want in (("1", True), ("true", True), ("True", True), ("0", False)):
        monkeypatch.setenv("VLLM_ROCM_MINIMAX_M3_MONO_TESTFLAG", raw)
        assert mono_envs._flag("VLLM_ROCM_MINIMAX_M3_MONO_TESTFLAG") is want
    monkeypatch.delenv("VLLM_ROCM_MINIMAX_M3_MONO_TESTFLAG")
    assert mono_envs._flag("VLLM_ROCM_MINIMAX_M3_MONO_TESTFLAG") is False


def test_disabled_without_the_switch(monkeypatch):
    monkeypatch.setattr(mono_dispatch.envs, "MONO_ENABLE", False)
    mono = mono_dispatch.MonoDecode(_model(), _vllm_config())
    assert not mono._enabled
    assert not mono.supports(None, torch.zeros(1, dtype=torch.int64), None, None)


def test_unsupported_deployment_falls_back(monkeypatch):
    monkeypatch.setattr(mono_dispatch.envs, "MONO_ENABLE", True)
    bad = _vllm_config(**{"parallel_config.tensor_parallel_size": 8})
    assert not mono_dispatch.MonoDecode(_model(), bad)._enabled


def test_enabled_dispatcher_reads_the_first_sparse_layer(mono):
    assert mono._enabled
    assert mono._layer_name == "attn" and mono._index_prefix == "idx"
    assert mono._runner is None  # built lazily, on the first eligible step


def test_runner_is_built_once_outside_capture(mono):
    assert mono._ensure_runner() and _FakeRunner.built == 1
    assert mono._ensure_runner() and _FakeRunner.built == 1


def test_runner_is_not_built_inside_capture(mono, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not mono._ensure_runner() and _FakeRunner.built == 0
    assert mono._enabled  # not a refusal: the next uncaptured step builds it


def test_stale_runner_is_rebuilt_after_a_kv_rebind(mono, monkeypatch):
    first = _FakeRunner(None, None, stale=True)
    mono._runner, mono._slots_verified = first, True
    assert mono._ensure_runner()
    assert first.closed and mono._runner is not first
    assert not mono._slots_verified  # the new cache's slots are checked again


def test_stale_runner_is_kept_while_capturing(mono, monkeypatch):
    first = _FakeRunner(None, None, stale=True)
    mono._runner = first
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not mono._ensure_runner() and not first.closed


@pytest.mark.parametrize("exc", [MonoUnsupported("no CUs"), RuntimeError("boom")])
def test_runner_build_failure_disables_mono(mono, monkeypatch, exc):
    def boom(model, config):
        raise exc

    monkeypatch.setattr(mono_dispatch, "MonoDecodeRunner", boom)
    assert not mono._ensure_runner()
    assert not mono._enabled and mono._runner is None


def test_slot_check_runs_once_and_caches(mono):
    assert mono._slots_shared(_fwd_shared(4), 4) and mono._slots_verified
    assert mono._slots_shared(_fwd_shared(4, shared=False), 4)  # cached verdict


@pytest.mark.parametrize("missing", [True, False])
def test_unshared_index_slots_disable_mono(mono, missing):
    fwd = _fwd_shared(4, shared=False)  # mismatched ids ...
    if missing:
        del fwd.slot_mapping["idx"]  # ... or no index slots at all
    assert not mono._slots_shared(fwd, 4) and not mono._enabled


def test_supports_routes_only_pure_decodes(mono, monkeypatch):
    pos = torch.zeros(4, dtype=torch.int64)
    monkeypatch.setattr(mono_dispatch, "get_forward_context", lambda: _fwd_shared(4))
    assert mono.supports(None, pos, None, None) and _FakeRunner.built == 1
    assert not mono.supports(None, pos, object(), None)  # pipeline-parallel input
    monkeypatch.setattr(
        mono_dispatch, "get_forward_context", lambda: _fwd_shared(4, shared=True)
    )
    fwd = _fwd(4, prefills=1)
    monkeypatch.setattr(mono_dispatch, "get_forward_context", lambda: fwd)
    assert not mono.supports(None, pos, None, None) and mono._enabled  # just declined


def test_refused_step_does_not_build_the_runner(mono, monkeypatch):
    fwd = _fwd(MAX_TOKENS + 1)
    monkeypatch.setattr(mono_dispatch, "get_forward_context", lambda: fwd)
    pos = torch.zeros(MAX_TOKENS + 1, dtype=torch.int64)
    assert not mono.supports(None, pos, None, None) and _FakeRunner.built == 0


def test_forward_runs_the_runner_or_the_self_check(mono, monkeypatch):
    mono._ensure_runner()
    pos = torch.zeros(2, dtype=torch.int64)
    assert mono.forward(None, pos, None) == "mono-out" and mono._runner.calls == [2]
    monkeypatch.setattr(mono_dispatch.envs, "MONO_CHECK", True)
    monkeypatch.setattr(
        mono_dispatch, "forward_checked", lambda runner, *a: ("checked", runner)
    )
    assert mono.forward(None, pos, None) == ("checked", mono._runner)
