# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]

    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _dsv4_inputs(s_q, h_q, num_valid, seed=1234):
    """DSv4 prefill/decode shape: head 512, no rope split (block_dpe=0)."""
    device = torch.device("xpu")
    s_kv, d_qk, topk = 1024, 512, 512
    torch.random.manual_seed(seed)
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((s_kv, 1, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.full((s_q, 1, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        indices[t, 0, :num_valid] = torch.randperm(s_kv, device=device)[:num_valid].to(
            torch.int32
        )
    return q, kv, indices, d_qk**-0.5


@pytest.mark.parametrize("h_q", [8, 64])
@pytest.mark.parametrize("num_valid", [1, 4, 16, 128, 512])
@pytest.mark.parametrize("sink_mode", ["checkpoint_range", "comparable_mass"])
@requires_xpu
def test_bf16_triton_sparse_mla_attn_sink(h_q, num_valid, sink_mode):
    """attn_sink is an extra per-head logit in the softmax denominator.

    ``checkpoint_range`` draws sinks from the released DSv4-Flash range
    (~[-2.2, 2.5]); ``comparable_mass`` sets them near log(num_valid) so the
    sink holds about as much softmax mass as the keys and a wrong scale or
    offset shows up for every row length.
    """
    q, kv, indices, sm_scale = _dsv4_inputs(8, h_q, num_valid)
    rand = torch.rand(h_q, device=q.device)
    if sink_mode == "checkpoint_range":
        attn_sink = rand * 4.7 - 2.2
    else:
        attn_sink = math.log(num_valid) + rand * 2 - 1

    out, _, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, 512, block_dpe=0, attn_sink=attn_sink
    )
    ref_out, _, _, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, 512, attn_sink=attn_sink
    )
    assert out.isfinite().all()
    assert _rel_l2(out, ref_out) < 1e-2
    # lse is reported over the selected keys only (sink excluded).
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)

    out_nosink, _, _ = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, 512, block_dpe=0
    )
    if sink_mode == "comparable_mass" or num_valid <= 16:
        assert _rel_l2(out_nosink, ref_out) > 1e-2


@pytest.mark.parametrize("h_q", [8, 64])
@requires_xpu
def test_bf16_triton_sparse_mla_attn_sink_empty_row(h_q):
    """A row without valid keys and a finite sink has an all-zero output."""
    q, kv, indices, sm_scale = _dsv4_inputs(4, h_q, 16)
    indices[0] = -1
    attn_sink = torch.rand(h_q, device=q.device) * 4.7 - 2.2
    out, _, _ = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, 512, block_dpe=0, attn_sink=attn_sink
    )
    ref_out, _, _, _ = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, 512, attn_sink=attn_sink
    )
    assert out.isfinite().all()
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert _rel_l2(out[1:], ref_out[1:]) < 1e-2


@requires_xpu
def test_bf16_triton_sparse_mla_attn_sink_neg_inf_is_noop():
    """The default sink value (-inf) leaves the output bitwise unchanged."""
    q, kv, indices, sm_scale = _dsv4_inputs(4, 16, 128, seed=7)
    sink = torch.full((16,), float("-inf"), device=q.device)
    a, _, _ = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, 512, block_dpe=0
    )
    b, _, _ = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, 512, block_dpe=0, attn_sink=sink
    )
    assert torch.equal(a, b)


# The V4 call sites must hand the sink to the kernel interface; the kernel
# tests above call the interface directly and cannot see a dropped argument.
_H = 8


def _sink():
    # Longer than the head count, as the module-level parameter is.
    return torch.arange(2 * _H, dtype=torch.float32, device="xpu") * 0.1


def _attn_layer(**attrs):
    from vllm.models.deepseek_v4.xpu.xpu_sparse import DeepseekV4XPUAttention

    layer = object.__new__(DeepseekV4XPUAttention)
    torch.nn.Module.__init__(layer)
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


@requires_xpu
def test_v4_prefill_call_site_passes_attn_sink(monkeypatch):
    from vllm.models.deepseek_v4.xpu import xpu_sparse

    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["q"]), None, None

    class _Workspace:
        def get_simultaneous(self, *specs):
            return [torch.zeros(4, 80, 512, dtype=torch.bfloat16, device="xpu")]

    sink = _sink()
    layer = _attn_layer(
        PREFILL_CHUNK_SIZE=4,
        window_size=16,
        max_num_batched_tokens=64,
        max_model_len=128,
        compress_ratio=1,
        scale=0.1,
        attn_sink=sink,
        topk_indices_buffer=torch.zeros(4, 4, dtype=torch.int32, device="xpu"),
    )
    loc = torch.tensor([0, 4], dtype=torch.int32)
    swa_metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=4,
        num_decodes=0,
        num_decode_tokens=0,
        prefill_seq_lens=torch.tensor([4], device="xpu"),
        prefill_gather_lens=torch.tensor([4], device="xpu"),
        query_start_loc_cpu=loc,
        query_start_loc=loc.to("xpu"),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device="xpu"),
        block_size=16,
    )
    monkeypatch.setattr(xpu_sparse, "current_workspace_manager", _Workspace)
    monkeypatch.setattr(
        xpu_sparse, "dequantize_and_gather_k_cache", lambda *a, **k: None
    )
    monkeypatch.setattr(
        xpu_sparse,
        "combine_topk_swa_indices",
        lambda *a, **k: (torch.zeros(4, 16, dtype=torch.int32, device="xpu"), None),
    )
    monkeypatch.setattr(xpu_sparse, "triton_bf16_mla_sparse_interface", spy)

    q = torch.zeros(4, _H, 512, dtype=torch.bfloat16, device="xpu")
    layer._forward_prefill(
        q,
        None,
        None,
        torch.zeros(1, device="xpu"),
        torch.empty_like(q),
        None,
        swa_metadata,
    )
    assert seen["attn_sink"] is not None
    assert torch.equal(seen["attn_sink"], sink[:_H])


@requires_xpu
def test_v4_decode_call_sites_pass_attn_sink(monkeypatch):
    from vllm.models.deepseek_v4.xpu import xpu_sparse
    from vllm.models.deepseek_v4.xpu import xpu_sparse_decode_fp8 as decode_mod

    sink = _sink()
    q = torch.zeros(2, _H, 512, dtype=torch.bfloat16, device="xpu")

    # xpu_sparse_decode_fp8 -> kernel interface
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["q"]), None, None

    monkeypatch.setattr(decode_mod, "dequant_gather_slots", lambda *a, **k: None)
    monkeypatch.setattr(decode_mod, "triton_bf16_mla_sparse_interface", spy)
    decode_mod.xpu_sparse_decode_fp8(
        q=q,
        kv_cache=None,
        swa_kv_cache=torch.zeros(1, 16, 584, dtype=torch.uint8, device="xpu"),
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=torch.zeros(2, 1, 16, dtype=torch.int32, device="xpu"),
        swa_lens=torch.full((2,), 16, dtype=torch.int32, device="xpu"),
        attn_sink=sink,
        softmax_scale=0.1,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        out=torch.empty_like(q),
    )
    assert seen["attn_sink"] is not None
    assert torch.equal(seen["attn_sink"], sink[:_H])

    # DeepseekV4XPUAttention._forward_decode -> xpu_sparse_decode_fp8
    seen_layer = {}
    monkeypatch.setattr(
        xpu_sparse, "xpu_sparse_decode_fp8", lambda **kw: seen_layer.update(kw)
    )
    layer = _attn_layer(
        compress_ratio=1,
        scale=0.1,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=sink,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros(1, device="xpu")),
    )
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.zeros(2, 1, 16, dtype=torch.int32, device="xpu"),
        decode_swa_lens=torch.full((2,), 16, dtype=torch.int32, device="xpu"),
    )
    layer._forward_decode(q, None, swa_metadata, None, True, torch.empty_like(q))
    assert seen_layer["attn_sink"] is sink
