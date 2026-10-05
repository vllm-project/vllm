# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the persistent FlashInfer multi-CTA-KV counter buffer (P-1).

These tests cover the port of SGLang's persistent multi-CTA-KV counter buffer
ownership pattern into the FlashInfer sparse MLA decode path. The buffer eliminates the
per-step ``FillFunctor<uint8>`` launch that FlashInfer's public
``trtllm_batch_decode_with_kv_cache_mla`` entry point otherwise performs when it
allocates + zeroes the counter internally on every call.

The tests are CPU-runnable and self-contained: they inject a local stub of the
FlashInfer counter API (``get_trtllm_gen_multi_ctas_kv_counter_bytes`` /
``get_device_sm_count``) and force ``_FI_HAS_MULTI_CTAS_COUNTER_API=True`` so
they run even in environments where FlashInfer is not installed. The stub formula is
validated against the real FlashInfer implementation in
``test_counter_bytes_formula_matches_flashinfer`` when FlashInfer is available.
"""

import os

# Force CPU platform resolution BEFORE any vllm import so VllmConfig() can be
# built in _construct_real_impl without a GPU. current_platform is resolved at
# import time from this env var, so it must be set first.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _get_multi_ctas_kv_counter_buffer,
    _trtllm_gen_mla_decode_supports_num_heads,
)


@pytest.fixture(autouse=True)
def _reset_counter_global():
    """Reset the module-global counter buffer before each test.

    The buffer only ever grows (never shrinks), so without a reset a test that
    allocates a large buffer would leak its size into later tests that assert exact
    sizes.
    """
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    fi_sparse._fi_sparse_multi_ctas_kv_counter = None
    yield
    fi_sparse._fi_sparse_multi_ctas_kv_counter = None


def _fi_counter_bytes_stub(batch_size, num_qo_heads, sm_count):
    """Local replica of flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes."""
    return round_up(max(batch_size * num_qo_heads, sm_count), 8) * 4


def _make_counter_impl(**overrides):
    """Build a SimpleNamespace impl exposing every attribute forward_mqa reads."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    attrs = dict(
        # _normalize_lse is a staticmethod on the real impl; expose the plain
        # function so SimpleNamespace-bound forward_mqa can call self._normalize_lse.
        _normalize_lse=FlashInferMLASparseImpl._normalize_lse,
        # The upstream base commit extracted per-layer scale preparation into
        # _prepare_mqa_kernel (a no-op here: this fixture always provides
        # bmm1/bmm2 scales and a workspace buffer).
        _prepare_mqa_kernel=lambda layer, device: None,
        # The upstream base's _run_mqa_kernel also routes on index_group /
        # is_nope_mla (HiSparse safe-lengths block, upstream of the counter
        # wiring this patch tests). Neutral values: no HiSparse, rope heads.
        index_group=None,
        is_nope_mla=False,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dtype="auto",
        topk_indices_buffer=torch.zeros(4, 8, dtype=torch.int32),
        dcp_world_size=1,
        dcp_rank=0,
        _workspace_buffer=torch.zeros(1024, dtype=torch.int8),
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        need_to_return_lse_for_decode=False,
        scale=0.08838834764831845,
        _mla_counter_max_batch=2048,
        _mla_counter_max_heads=16,
        _mla_counter_bytes=None,
        _persistent_mla_counter_enabled=True,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _make_counter_metadata(**overrides):
    """Build a SimpleNamespace attn_metadata exposing every attr forward_mqa reads."""
    attrs = dict(
        topk_tokens=8,
        req_id_per_token=torch.zeros(4, dtype=torch.int32),
        block_table=torch.zeros(4, 1, dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
        decode=SimpleNamespace(seq_lens=torch.ones(4, dtype=torch.int32)),
        num_decode_tokens=4,
        num_decodes=4,
        causal=True,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _install_counter_mocks(monkeypatch, fake_kernel, api_available=True):
    """Install the kernel + helper mocks forward_mqa depends on.

    ``forward_mqa`` imports ``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla``
    lazily inside the function body, so a fake ``flashinfer`` package is injected into
    ``sys.modules`` when FlashInfer is not installed. Returns the
    ``flashinfer_mla_sparse`` module for further patching.
    """
    import sys
    from types import ModuleType

    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    if "flashinfer" not in sys.modules:
        fake_fi = ModuleType("flashinfer")
        fake_decode = ModuleType("flashinfer.decode")
        fake_decode.trtllm_batch_decode_with_kv_cache_mla = fake_kernel
        fake_fi.decode = fake_decode
        sys.modules["flashinfer"] = fake_fi
        sys.modules["flashinfer.decode"] = fake_decode
        monkeypatch.setattr(sys, "modules", sys.modules)
    else:
        monkeypatch.setattr(
            "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla", fake_kernel
        )
    monkeypatch.setattr(
        fi_sparse,
        "triton_convert_req_index_to_global_index",
        lambda *a, **k: (torch.zeros(4, 8, dtype=torch.int32),
                          torch.ones(4, dtype=torch.int32)),
    )
    monkeypatch.setattr(fi_sparse, "_get_workspace_buffer",
                      lambda device: torch.zeros(1024, dtype=torch.int8))
    monkeypatch.setattr(fi_sparse, "_FI_HAS_MULTI_CTAS_COUNTER_API",
                      api_available)
    # The real names are only bound when the guarded import succeeds (FlashInfer
    # installed). Inject them with raising=False so forward_mqa resolves them
    # regardless of the installed FlashInfer version, and monkeypatch restores the
    # module afterwards (plain setattr would mutate the module permanently).
    monkeypatch.setattr(fi_sparse, "get_trtllm_gen_multi_ctas_kv_counter_bytes",
                        _fi_counter_bytes_stub, raising=False)
    monkeypatch.setattr(fi_sparse, "get_device_sm_count", lambda device: 148,
                      raising=False)
    return fi_sparse


def _run_counter_forward(monkeypatch,
                       impl,
                       metadata,
                       q_heads=16,
                       page_size=64,
                       api_available=True):
    """Drive FlashInferMLASparseImpl.forward_mqa with a mocked kernel.

    Returns (kernel_kwargs, output, lse). The kernel is mocked to return a
    tensor shaped like the real kernel output.
    """
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    q = torch.zeros(4, q_heads, 160, dtype=torch.float16)
    kv_cache = torch.zeros(4, page_size, 144, dtype=torch.float16)

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        o = torch.zeros(4, 1, q_heads, 112, dtype=torch.float16)
        if kwargs.get("return_lse"):
            return o, torch.zeros(4, q_heads, dtype=torch.float16)
        return o

    _install_counter_mocks(monkeypatch, fake_kernel, api_available)

    impl = _make_counter_impl(**impl)
    _make_counter_metadata(**metadata)  # kept for overrides validation
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    topk_indices = impl.topk_indices_buffer
    seq_lens = torch.ones(q.shape[0], dtype=torch.int32)
    out, lse = bound(q, kv_cache, topk_indices, seq_lens)
    return captured, out, lse


def test_trtllm_gen_mla_decode_supports_num_heads():
    # Tileable: num_heads divides min(num_heads, tile) (tile=8/16/64).
    # 1..16 always divide themselves; 32 divides 16; 33..64 divide themselves;
    # 128 divides 64.
    # 1..16: min(heads, tile) == heads -> always tileable.
    for heads in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16):
        assert _trtllm_gen_mla_decode_supports_num_heads(heads), heads
    # 17..32: tile == 16 -> only multiples of 16 (i.e. 32) are tileable.
    for heads in (32,):
        assert _trtllm_gen_mla_decode_supports_num_heads(heads), heads
    # 33..64: min(heads, 64) == heads -> always tileable.
    for heads in (33, 38, 47, 63, 64):
        assert _trtllm_gen_mla_decode_supports_num_heads(heads), heads
    # > 64: tile == 64 -> only multiples of 64 are tileable.
    for heads in (128, 192, 320):
        assert _trtllm_gen_mla_decode_supports_num_heads(heads), heads
    # Untileable: 17..31, or > 64 and not a multiple of 64.
    for heads in (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 65, 67, 69, 70, 71, 76, 89, 94, 103,
                 115, 127, 175, 253):
        assert not _trtllm_gen_mla_decode_supports_num_heads(heads), heads


def test_counter_bytes_formula_matches_flashinfer():
    """Validate the local stub against the REAL FlashInfer formula when available."""
    try:
        from flashinfer.utils import (  # noqa: F401
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )
    except ImportError:
        pytest.skip("FlashInfer not installed; stub formula is used elsewhere")
    for batch, heads, sm in ((1, 16, 148), (2048, 16, 148), (1, 64, 148),
                           (8192, 128, 148)):
        assert get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, heads, sm) == _fi_counter_bytes_stub(batch, heads, sm)


def test_get_multi_ctas_kv_counter_buffer_allocates_zeroed_uint8():
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    fi_sparse._fi_sparse_multi_ctas_kv_counter = None
    buf = _get_multi_ctas_kv_counter_buffer(1024, torch.device("cpu"))
    assert buf.dtype == torch.uint8
    assert buf.numel() == 1024
    assert (buf == 0).all()
    assert fi_sparse._fi_sparse_multi_ctas_kv_counter is buf


def test_get_multi_ctas_kv_counter_buffer_reuses_when_sufficient():
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    fi_sparse._fi_sparse_multi_ctas_kv_counter = None
    b1 = _get_multi_ctas_kv_counter_buffer(1024, torch.device("cpu"))
    b2 = _get_multi_ctas_kv_counter_buffer(1024, torch.device("cpu"))
    assert b1.data_ptr() == b2.data_ptr()


def test_get_multi_ctas_kv_counter_buffer_grows_when_undersized():
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    fi_sparse._fi_sparse_multi_ctas_kv_counter = None
    b1 = _get_multi_ctas_kv_counter_buffer(8, torch.device("cpu"))
    b2 = _get_multi_ctas_kv_counter_buffer(4096, torch.device("cpu"))
    assert b1.data_ptr() != b2.data_ptr()
    assert b2.numel() == 4096
    assert (b2 == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                   reason="CUDA required for device-change realloc")
def test_get_multi_ctas_kv_counter_buffer_device_change_realloc():
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    fi_sparse._fi_sparse_multi_ctas_kv_counter = None
    b1 = _get_multi_ctas_kv_counter_buffer(1024, torch.device("cpu"))
    b2 = _get_multi_ctas_kv_counter_buffer(1024, torch.device("cuda"))
    assert b1.data_ptr() != b2.data_ptr()
    assert b2.device.type == "cuda"


def test_counter_nope_mla_lens_coexists_with_buffer(monkeypatch):
    """nope-MLA (qk_rope_head_dim=0) must forward sparse_mla_top_k_lens AND
    the persistent multi-CTA-KV counter buffer in the SAME kernel call.

    Regression guard: the counter wiring originally re-initialised ``extra_kwargs``
    after the nope-MLA lens was staged, silently dropping
    ``sparse_mla_top_k_lens`` from the kernel invocation. Both kwargs must reach
    the kernel together.
    """
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(4, 1296, 914, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl = _make_counter_impl(is_nope_mla=True)
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 960, 930, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 938, dtype=torch.float16)
    seq_lens = torch.ones(q.shape[0], dtype=torch.int32)
    out, lse = bound(q, kv_cache, impl.topk_indices_buffer, seq_lens)
    assert "multi_ctas_kv_counter_buffer" in captured
    assert "sparse_mla_top_k_lens" in captured
    assert captured["sparse_mla_top_k_lens"].shape == (4,)
    assert captured["sparse_mla_top_k_lens"].equal(seq_lens)


def test_counter_eligible_path_passes_buffer(monkeypatch):
    captured, _, _ = _run_counter_forward(monkeypatch, {}, {})
    buf = captured["multi_ctas_kv_counter_buffer"]
    assert buf.dtype == torch.uint8
    assert buf.device.type == "cpu"
    assert "backend" not in captured
    assert buf.numel() >= 2048 * 16


def test_counter_persists_across_calls(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs["multi_ctas_kv_counter_buffer"])
        return torch.zeros(4, 1, 16, 100, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl = _make_counter_impl()
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 16, 140, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 132, dtype=torch.float16)
    bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert len(calls) == 2
    assert calls[0].data_ptr() == calls[1].data_ptr()


def test_counter_grows_on_larger_batch(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs["multi_ctas_kv_counter_buffer"])
        return torch.zeros(4, 1, 16, 108, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    # Start with a small worst-case batch so the first call computes a small
    # counter-buffer size (608 bytes for batch=1, heads=16, sm=148).
    impl = _make_counter_impl(_mla_counter_max_batch=1)
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 16, 136, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 130, dtype=torch.float16)
    bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    small_expected = _fi_counter_bytes_stub(1, 16, 148)
    assert calls[0].numel() == small_expected

    # Enlarge the worst-case batch and reset the cached byte count so the next call
    # recomputes a larger buffer (Rev B: exact equality, not just "bigger").
    impl._mla_counter_max_batch = 2048
    impl._mla_counter_bytes = None
    bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert len(calls) == 2
    assert calls[0].data_ptr() != calls[1].data_ptr()
    expected = _fi_counter_bytes_stub(2048, 16, 148)
    assert calls[1].numel() == expected


def test_counter_untileable_heads_no_buffer_no_backend(monkeypatch):
    captured, _, _ = _run_counter_forward(monkeypatch, {}, {}, q_heads=24)
    assert "multi_ctas_kv_counter_buffer" not in captured
    assert "backend" not in captured


def test_counter_ineligible_page_size_no_buffer(monkeypatch):
    captured, _, _ = _run_counter_forward(monkeypatch, {}, {}, page_size=128)
    assert "multi_ctas_kv_counter_buffer" not in captured
    assert "backend" not in captured


def test_counter_api_unavailable_fallback(monkeypatch):
    captured, _, _ = _run_counter_forward(monkeypatch, {}, {},
                                        api_available=False)
    assert "multi_ctas_kv_counter_buffer" not in captured
    assert "backend" not in captured


def test_counter_disabled_env_var_fallback(monkeypatch):
    captured, _, _ = _run_counter_forward(
        monkeypatch, {"_persistent_mla_counter_enabled": False}, {})
    assert "multi_ctas_kv_counter_buffer" not in captured
    assert "backend" not in captured


def test_counter_no_hardcoded_backend(monkeypatch):
    captured, _, _ = _run_counter_forward(monkeypatch, {}, {})
    assert "backend" not in captured


def test_counter_dcp_sizing_and_branch(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        # need_to_return_lse_for_decode=True -> the kernel returns (out, lse).
        o = torch.zeros(4, 1, 32, 296, dtype=torch.float16)
        return o, torch.zeros(4, 32, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)
    # dcp_world_size=2 routes forward_mqa through the DCP index filter.
    monkeypatch.setattr(
        fi_sparse,
        "triton_filter_and_convert_dcp_index",
        lambda *a, **k: (torch.zeros(4, 8, dtype=torch.int32),
                          torch.ones(4, dtype=torch.int32)),
    )

    impl = _make_counter_impl(
        dcp_world_size=2,
        num_heads=16,
        _mla_counter_max_heads=32,
        need_to_return_lse_for_decode=True,
    )
    metadata = _make_counter_metadata(
        req_id_per_token=torch.zeros(4, dtype=torch.int32),
        block_table=torch.zeros(4, 1, dtype=torch.int32),
    )
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 32, 280, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 272, dtype=torch.float16)
    out, lse = bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert "multi_ctas_kv_counter_buffer" in captured
    assert captured["return_lse"] is True
    assert lse is not None

def test_counter_one_time_logs(monkeypatch, caplog_vllm):
    import logging

    import vllm.logger as vllm_logger
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    # info_once is lru_cache'd process-wide; clear it so this test observes the
    # one-time log deterministically regardless of prior tests.
    vllm_logger._print_info_once.cache_clear()

    with caplog_vllm.at_level(logging.INFO):
        _run_counter_forward(monkeypatch, {}, {})
        _run_counter_forward(monkeypatch, {}, {})
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert any("persistent multi-CTA-KV counter buffer" in m for m in messages)


def test_counter_tuple_q_input(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(4, 1, 16, 104, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl = _make_counter_impl()
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = (torch.zeros(4, 16, 92, dtype=torch.float16),
         torch.zeros(4, 16, 44, dtype=torch.float16))
    kv_cache = torch.zeros(4, 64, 168, dtype=torch.float16)
    q_cat = torch.cat(q, dim=-1)
    bound(q_cat, kv_cache, impl.topk_indices_buffer,
          torch.ones(q_cat.shape[0], dtype=torch.int32))
    assert "multi_ctas_kv_counter_buffer" in captured


def test_counter_empty_rows_lse_masking_unchanged(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        o = torch.zeros(4, 1, 16, 106, dtype=torch.float16)
        return o, torch.zeros(4, 16, dtype=torch.float16)

    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    _install_counter_mocks(monkeypatch, fake_kernel)
    # All top-k index rows are invalid (-1) -> every row is an "empty row" whose
    # LSE must be masked to -inf and output zeroed.
    monkeypatch.setattr(
        fi_sparse,
        "triton_convert_req_index_to_global_index",
        lambda *a, **k: (torch.full((4, 8), -1, dtype=torch.int32),
                          torch.ones(4, dtype=torch.int32)),
    )

    impl = _make_counter_impl(
        need_to_return_lse_for_decode=True,
        topk_indices_buffer=torch.full((4, 8), -1, dtype=torch.int32),
    )
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 16, 186, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 218, dtype=torch.float16)
    out, lse = bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert (out == 0).all()
    assert (lse == float("-inf")).all()
    assert "multi_ctas_kv_counter_buffer" in captured


def test_counter_module_global_shared_across_instances(monkeypatch):
    """Two different impl instances must pass the SAME module-global buffer.

    This is the "one buffer per backend" property (SGLang parity): the buffer is
    shared by ALL layers of the model, so a regression to per-instance buffers
    (memory blowup across 78 layers) would be caught here.
    """
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs["multi_ctas_kv_counter_buffer"])
        return torch.zeros(4, 1, 16, 105, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl1 = _make_counter_impl()
    impl2 = _make_counter_impl()
    metadata = _make_counter_metadata()
    bound1 = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl1)
    bound2 = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl2)
    q = torch.zeros(4, 16, 133, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 153, dtype=torch.float16)
    bound1(q, kv_cache, impl1.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    bound2(q, kv_cache, impl2.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert len(calls) == 2
    assert calls[0].data_ptr() == calls[1].data_ptr()


def test_counter_max_batch_fallback(monkeypatch):
    """_mla_counter_max_batch=0 (both scheduler fields 0/None) must not crash."""
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(4, 1, 16, 107, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl = _make_counter_impl(_mla_counter_max_batch=0)
    metadata = _make_counter_metadata()
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 16, 137, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 149, dtype=torch.float16)
    out, lse = bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    buf = captured["multi_ctas_kv_counter_buffer"]
    assert buf.numel() >= _fi_counter_bytes_stub(0, 16, 148)


def test_counter_fp8_quantized_path(monkeypatch):
    """kv_cache_dtype='fp8' -> bmm1_scale incorporates layer scales and buffer."""
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as fi_sparse

    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return torch.zeros(4, 1, 16, 109, dtype=torch.float16)

    _install_counter_mocks(monkeypatch, fake_kernel)

    impl = _make_counter_impl(kv_cache_dtype="fp8", bmm1_scale=None,
                             bmm2_scale=None)
    layer = SimpleNamespace(_q_scale_float=2.0, _k_scale_float=3.0)
    metadata = _make_counter_metadata()
    # The upstream base moved per-layer scale preparation into
    # _prepare_mqa_kernel; run the real one so the fp8 scaling assertions
    # still exercise the production code path.
    MethodType(fi_sparse.FlashInferMLASparseImpl._prepare_mqa_kernel, impl)(
        layer, q_dev := torch.device("cpu"))
    bound = MethodType(fi_sparse.FlashInferMLASparseImpl._run_mqa_kernel, impl)
    q = torch.zeros(4, 16, 141, dtype=torch.float16)
    kv_cache = torch.zeros(4, 64, 147, dtype=torch.float16)
    out, lse = bound(q, kv_cache, impl.topk_indices_buffer, torch.ones(q.shape[0], dtype=torch.int32))
    assert "multi_ctas_kv_counter_buffer" in captured
    assert impl.bmm1_scale == 0.08838834764831845 * 2.0 * 3.0
    assert impl.bmm2_scale == 1.0 * 3.0


def _construct_real_impl(monkeypatch, num_heads=16):
    """Construct a real FlashInferMLASparseImpl with heavy deps mocked.

    The real __init__ reads get_current_vllm_config() (requires a
    set_current_vllm_config context) and SparseMLACommonImpl.__init__ probes
    masked-MHA availability (requires TP group + flash-attn version). Those are
    mocked here so the env-var -> flag wiring and the C2 max-heads sizing in
    __init__ can be tested on CPU.
    """
    import vllm.model_executor.layers.attention.sparse_mla_attention as sma
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    # Force CPU device inference so VllmConfig() can be built without a GPU.
    # DeviceConfig(device="cpu") bypasses the platform-probing __post_init__.
    monkeypatch.setattr(sma, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(sma, "get_flash_attn_version", lambda **kw: None)
    monkeypatch.setattr(sma.current_platform, "is_device_capability_family",
                      lambda *a: False)

    class _DummyKVProj:
        weight = torch.zeros(1, 1)

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        return FlashInferMLASparseImpl(
            num_heads=num_heads,
            head_size=576,
            scale=0.08838834764831845,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=128,
            kv_b_proj=_DummyKVProj(),
            topk_indices_buffer=torch.zeros(4, 8, dtype=torch.int32),
        )


def test_counter_init_captures_worst_case_batch_and_heads(monkeypatch):
    """Real __init__ must size the counter for worst-case batch AND max heads."""
    impl = _construct_real_impl(monkeypatch, num_heads=16)
    # VllmConfig() defaults: max_num_batched_tokens=2048, max_num_seqs=256.
    assert impl._mla_counter_max_batch == 2048
    assert impl._mla_counter_max_heads == 16
    assert impl._mla_counter_bytes is None
    assert impl._persistent_mla_counter_enabled is True


def test_counter_init_env_var_disables_feature(monkeypatch):
    """VLLM_DISABLE_PERSISTENT_MLA_COUNTER=1 must flip the flag in __init__."""
    monkeypatch.setenv("VLLM_DISABLE_PERSISTENT_MLA_COUNTER", "1")
    impl = _construct_real_impl(monkeypatch, num_heads=16)
    assert impl._persistent_mla_counter_enabled is False


def test_counter_init_env_var_unset_enables_feature(monkeypatch):
    """Unset env var -> feature enabled by default."""
    monkeypatch.delenv("VLLM_DISABLE_PERSISTENT_MLA_COUNTER", raising=False)
    impl = _construct_real_impl(monkeypatch, num_heads=16)
    assert impl._persistent_mla_counter_enabled is True


def test_counter_init_startup_logs(monkeypatch, caplog_vllm):
    """Startup log must announce ENABLED/DISABLED per the feature toggle."""
    import logging

    import vllm.logger as vllm_logger

    vllm_logger._print_info_once.cache_clear()
    with caplog_vllm.at_level(logging.INFO):
        _construct_real_impl(monkeypatch, num_heads=16)
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert any(
        "Persistent FlashInfer multi-CTA-KV counter buffer is ENABLED" in m
        for m in messages
    )

    vllm_logger._print_info_once.cache_clear()
    monkeypatch.setenv("VLLM_DISABLE_PERSISTENT_MLA_COUNTER", "1")
    with caplog_vllm.at_level(logging.INFO):
        _construct_real_impl(monkeypatch, num_heads=16)
    messages = [r.getMessage() for r in caplog_vllm.records]
    assert any(
        "Persistent FlashInfer multi-CTA-KV counter buffer is DISABLED "
        "(env override)" in m
        for m in messages
    )
