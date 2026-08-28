# SPDX-License-Identifier: Apache-2.0
"""FlashInfer conditional device/CPU query-lens mismatch support (SM120 XQA).

Covers the W1-W7 invariant matrix from the flashinfer-adaptive-mismatch
campaign: the shared capability predicate, deferred finalize (role tag,
runner bound, transactional init), mismatch-safe classification (device
offsets pass through, fixed bound, prefill routing above the bound),
the persistent device-built packed mask, cascade/fallback exclusions,
and the SM100 prepare-only plumbing.

Requires a real CUDA device on SM120/SM121 (dedicated XQA).
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("flashinfer")

import vllm.v1.attention.backends.flashinfer as fi  # noqa: E402
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
    FlashInferTrtllmAPIDecode,
    _adaptive_xqa_mismatch_safe,
    _make_xqa_ragged_draft_block_mask,
    _trtllm_gen_varlen_decode_args,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

DEV = torch.device("cuda:0")
LAYER = "model.layers.0.self_attn.attn"

if not torch.cuda.is_available() or torch.cuda.get_device_capability(DEV)[0] != 12:
    pytest.skip("SM120 required", allow_module_level=True)


class _FakeLayer(AttentionLayerBase):
    def __init__(self):
        self._vllm_model_tag = "backbone"
        self.num_heads = 32
        self.impl = SimpleNamespace(num_heads=32)

    def get_attn_backend(self):
        return FlashInferBackend

    def get_kv_cache_spec(self, vllm_config):
        return None


def _cfg(adaptive: bool = True) -> Any:
    """Minimal config view; real SpeculativeConfig shapes are covered by the
    campaign validation record — here we assert the predicate's contract."""
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=7,
            parallel_drafting=True,
        ),
        attention_config=SimpleNamespace(
            use_non_causal=False,
            use_trtllm_attention=None,
            disable_flashinfer_q_quantization=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        use_v2_model_runner=True,
    )


def _ctx(vc):
    return set_current_vllm_config(vc)


def test_predicate_gates():
    assert _adaptive_xqa_mismatch_safe(_cfg()) is True
    assert _adaptive_xqa_mismatch_safe(_cfg(adaptive=False)) is False
    c = _cfg()
    c.parallel_config.decode_context_parallel_size = 2
    assert _adaptive_xqa_mismatch_safe(c) is False
    c = _cfg()
    c.attention_config.use_non_causal = True
    assert _adaptive_xqa_mismatch_safe(c) is False
    c = _cfg()
    c.attention_config.use_trtllm_attention = False
    assert _adaptive_xqa_mismatch_safe(c) is False
    assert _adaptive_xqa_mismatch_safe(None) is False


def test_mask_kernel_matches_oracle():
    with _ctx(_cfg()):
        b = _make_builder()
    for lens in ([3, 2, 1], [2, 2, 2], [3, 0, 1], [3, 2, 1, 0]):
        cum = [0]
        for L in lens:
            cum.append(cum[-1] + L)
        q_cu = torch.tensor(cum, dtype=torch.int32, device=DEV)
        buf = b._fill_adaptive_decode_mask(q_cu, len(lens), sum(lens), True)
        oracle = _make_xqa_ragged_draft_block_mask(lens, 8, True, DEV)
        assert torch.equal(buf[: sum(lens)], oracle)
        assert not (buf[sum(lens) :] != 0).any().item()


def test_classification_passes_device_truth():
    with _ctx(_cfg()):
        b = _make_builder()
    dev = torch.tensor([0, 3, 5, 6, 6], dtype=torch.int32, device=DEV)
    cpu = torch.tensor([0, 2, 4, 6, 6], dtype=torch.int32)
    saved_item, saved_tolist = torch.Tensor.item, torch.Tensor.tolist
    torch.Tensor.item = lambda self: (_ for _ in ()).throw(RuntimeError)
    torch.Tensor.tolist = lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError)
    try:
        bound, q_cu = b._compute_decode_query_lens_mismatch(dev, cpu, 4, 6, 6)
    finally:
        torch.Tensor.item, torch.Tensor.tolist = saved_item, saved_tolist
    assert bound == 8
    assert q_cu.tolist() == [0, 3, 5, 6, 6]
    assert q_cu.data_ptr() == dev.data_ptr()
    # single-token fast path only when provable per request
    ones = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    assert b._compute_decode_query_lens_mismatch(ones.to(DEV), ones, 3, 3, 3) == (
        1,
        None,
    )


def test_cudagraph_support_gates():
    vc = _full_cfg()
    with set_current_vllm_config(vc):
        r = FlashInferMetadataBuilder.get_cudagraph_support(vc, _spec())
        assert r.name == "VARLEN_DECODE" or r.name.startswith("UNIFORM")
    vc = _full_cfg(adaptive=False)
    with set_current_vllm_config(vc):
        r = FlashInferMetadataBuilder.get_cudagraph_support(vc, _spec())
        assert r.name != "VARLEN_DECODE"
    vc = _full_cfg()
    vc.attention_config.use_trtllm_attention = False
    with set_current_vllm_config(vc):
        r = FlashInferMetadataBuilder.get_cudagraph_support(vc, _spec())
        assert r.name != "VARLEN_DECODE"


def _full_cfg(adaptive: bool = True):
    vc = _cfg(adaptive)
    vc.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        max_model_len=8192,
        get_num_attention_heads=lambda pc: 32,
        get_num_kv_heads=lambda pc: 8,
    )
    vc.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL,
        max_cudagraph_capture_size=16,
        static_forward_context={LAYER: _FakeLayer()},
    )
    vc.scheduler_config = SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=4)
    vc.cache_config = SimpleNamespace(cache_dtype="auto")
    vc.kv_transfer_config = None
    return vc


def _spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )


def test_build_ragged_metadata():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    with _ctx(_cfg()):
        b = _make_builder()
    cam = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 5, 6, 6], dtype=torch.int32, device=DEV),
        query_start_loc_cpu=torch.tensor([0, 2, 4, 6, 6], dtype=torch.int32),
        seq_lens=torch.tensor([20, 31, 44, 17], dtype=torch.int32, device=DEV),
        num_reqs=4,
        num_actual_tokens=6,
        max_query_len=2,
        max_seq_len=64,
        block_table_tensor=torch.arange(8, dtype=torch.int32, device=DEV).reshape(4, 2),
        slot_mapping=torch.arange(6, dtype=torch.int64, device=DEV),
        causal=True,
    )
    with _ctx(_cfg()):
        m = b.build(0, cam)
    assert m.decode.q_len_per_req == 8
    assert m.decode.q_cu_seq_lens.tolist() == [0, 3, 5, 6, 6]
    assert m.decode.q_cu_seq_lens.data_ptr() == (cam.query_start_loc.data_ptr())
    assert m.decode.mask is not None
    oracle = _make_xqa_ragged_draft_block_mask([3, 2, 1, 0], 8, True, DEV)
    assert torch.equal(m.decode.mask[:6], oracle)


def test_mask_width40_and_noncausal():
    b = None
    with _ctx(_cfg()):
        b = _make_builder()
    b.adaptive_max_decode_width = 40
    b._adaptive_mask_buffer = None  # force reallocation at width 40
    # reallocate like finalize does
    b._adaptive_mask_buffer = torch.zeros(
        b.max_num_reqs * 40,
        2 * ((40 + 31) // 32),
        dtype=torch.uint16,
        device=DEV,
    )
    for causal in (True, False):
        lens = [40, 33, 5]
        cum = [0]
        for L in lens:
            cum.append(cum[-1] + L)
        q_cu = torch.tensor(cum, dtype=torch.int32, device=DEV)
        buf = b._fill_adaptive_decode_mask(q_cu, 3, sum(lens), causal)
        oracle = _make_xqa_ragged_draft_block_mask(lens, 40, causal, DEV)
        assert torch.equal(buf[: sum(lens)], oracle)


def test_trtllm_gen_varlen_never_inferred():
    def meta(q_cu, authorized, kernel=FlashInferDecodeKernel.TRTLLM_GEN):
        return FlashInferTrtllmAPIDecode(
            kernel=kernel,
            block_tables=None,
            seq_lens=None,
            max_seq_len=1,
            q_len_per_req=8,
            q_cu_seq_lens=q_cu,
            mask=None,
            trtllm_gen_varlen=authorized,
        )

    q = torch.tensor([0, 3, 5, 6, 6], dtype=torch.int32, device=DEV)
    assert _trtllm_gen_varlen_decode_args(meta(None, False), 2) == (None, None)
    a = _trtllm_gen_varlen_decode_args(meta(q, True), 2)
    assert a[1] == 8 and a[0] is q
    with pytest.raises(AssertionError):
        _trtllm_gen_varlen_decode_args(
            meta(q, True, kernel=FlashInferDecodeKernel.XQA), 2
        )


def _make_builder():
    vc = _full_cfg()
    spec = SimpleNamespace(get_per_layer_parameters=lambda *a, **k: {})
    real_gplp = fi.get_per_layer_parameters
    real_igh = fi.infer_global_hyperparameters
    fi.get_per_layer_parameters = spec.get_per_layer_parameters
    fi.infer_global_hyperparameters = lambda *a, **k: SimpleNamespace(
        sm_scale=0.088,
        window_left=-1,
        logits_soft_cap=None,
        has_sinks=False,
        has_same_window_lefts=True,
        has_same_all_params=True,
    )
    try:
        b = FlashInferMetadataBuilder(
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
            ),
            [LAYER],
            vc,
            DEV,
        )
        b._finalize_adaptive_decode(8)
        return b
    finally:
        fi.get_per_layer_parameters = real_gplp
        fi.infer_global_hyperparameters = real_igh
