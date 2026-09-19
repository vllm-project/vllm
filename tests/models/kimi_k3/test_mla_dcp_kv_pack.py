# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP chunked-context K/V pack hook and Kimi-K3's fused packer.

``MLACommonImpl._context_parallel_compute_prefill_context`` grew an optional
``kv_pack``: with ``None`` its per-chunk tail must be the stock
cast/split/concat, and with a packer the impl must hand the prefill backend
exactly the packer's ``(k, v)``. The Kimi-K3 layer's packer must be byte-exact
with the stock tail and honour its kill switches.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.attention.mla_attention as mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
from vllm.models.kimi_k3.nvidia import mla as kimi_mla
from vllm.platforms import current_platform

_KV_LORA_RANK = 512
_QK_NOPE = 128
_QK_ROPE = 64
_V_HEAD_DIM = 128
_NUM_HEADS = 2
_DCP = 2
_TOKENS = 6


def _stock_kv_pack(
    kv_nope: torch.Tensor, k_pe: torch.Tensor, use_fp8_prefill: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The generic impl's DCP chunk tail, statement for statement.

    Cast ``kv_nope``/``k_pe`` to fp8 when the prefill query is fp8, split off
    ``v`` and write ``[k_nope | k_pe]`` with two slice copies.
    """
    if use_fp8_prefill:
        fp8 = current_platform.fp8_dtype()
        kv_nope = kv_nope.to(fp8)
        k_pe = k_pe.to(fp8)
    k_nope, v = kv_nope.split([_QK_NOPE, _V_HEAD_DIM], dim=-1)
    k = torch.empty(
        (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )
    k[..., : k_nope.shape[-1]] = k_nope
    k[..., k_nope.shape[-1] :] = k_pe
    return k, v


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-exact equality (NaN-safe: compares the raw storage, not values)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    raw = torch.uint8 if a.element_size() == 1 else torch.int16
    return torch.equal(a.contiguous().view(raw), b.contiguous().view(raw))


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def run_prefill_context_chunk(self, *, chunk, q, k, v):
        self.calls.append((k, v))
        out = torch.zeros((q.shape[0], _NUM_HEADS, _V_HEAD_DIM), dtype=torch.bfloat16)
        lse = torch.zeros((_NUM_HEADS, q.shape[0]), dtype=torch.float32)
        return out, lse


def _run_dcp_loop(monkeypatch, *, use_fp8_prefill: bool, kv_pack):
    """Drive the generic DCP loop on CPU with one chunk and a fake gather."""
    torch.manual_seed(0)
    q_dtype = current_platform.fp8_dtype() if use_fp8_prefill else torch.bfloat16
    weight = torch.randn(
        _KV_LORA_RANK, _NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM), dtype=torch.bfloat16
    )
    entry = _KV_LORA_RANK + _QK_ROPE
    workspace = torch.randn((_TOKENS * (_DCP + 1), entry), dtype=torch.bfloat16)

    fake_self = SimpleNamespace(
        kv_cache_dtype="auto",
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE,
        qk_nope_head_dim=_QK_NOPE,
        v_head_dim=_V_HEAD_DIM,
        num_heads=_NUM_HEADS,
        kv_b_proj=lambda x: (x @ weight, None),
        _use_flashinfer_concat_mla_k=False,
    )
    fake_self._concat_k_nope_k_pe = MLACommonImpl._concat_k_nope_k_pe.__get__(fake_self)

    monkeypatch.setattr(mla_attention.ops, "cp_gather_cache", lambda **_: None)
    monkeypatch.setattr(mla_attention, "_get_kv_b_proj_input_dtype", lambda *_: None)
    monkeypatch.setattr(
        mla_attention,
        "reorg_kvcache",
        lambda kv_c, k_pe, **_: (kv_c[:_TOKENS].squeeze(1), k_pe[:_TOKENS]),
    )
    chunk = SimpleNamespace(
        padded_local_seq_lens=[_TOKENS],
        local_context_lens_allranks=[[_TOKENS // _DCP] * _DCP],
        padded_local_cu_seq_lens=torch.tensor([0, _TOKENS]),
        padded_local_token_to_seq=torch.zeros(_TOKENS, dtype=torch.int32),
        local_starts=[0],
        num_local_context_tokens=_TOKENS,
        request_slice=slice(0, 1),
        num_requests=1,
        starts=torch.tensor([0]),
        num_context_tokens=_TOKENS,
        max_seq_len=_TOKENS,
        token_slice=slice(0, 4),
    )
    backend = _RecordingBackend()
    chunked_context = SimpleNamespace(
        workspace=workspace,
        chunks=[chunk],
        empty_token_slices=False,
        dcp_manager=SimpleNamespace(kv_gather=lambda dst, src: None),
    )
    attn_metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            prefill_backend=backend,
            chunked_context=chunked_context,
            q_data_type=q_dtype,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        )
    )
    q = torch.randn((4, _NUM_HEADS, _QK_NOPE + _QK_ROPE), dtype=torch.bfloat16)
    MLACommonImpl._context_parallel_compute_prefill_context(
        fake_self,
        q,
        torch.empty(0),
        attn_metadata,
        k_scale=torch.ones(1),
        dcp_world_size=_DCP,
        kv_pack=kv_pack,
    )
    # What the loop fed into the tail: the reorg'ed latent and k_pe.
    gathered = workspace[_TOKENS : _TOKENS * (1 + _DCP)].unsqueeze(1)
    kv_c_normed = gathered[..., :_KV_LORA_RANK][:_TOKENS].squeeze(1)
    k_pe = gathered[..., _KV_LORA_RANK:][:_TOKENS]
    kv_nope = (kv_c_normed @ weight).view(-1, _NUM_HEADS, _QK_NOPE + _V_HEAD_DIM)
    return backend, kv_nope, k_pe


@pytest.mark.parametrize("use_fp8_prefill", [False, True], ids=["bf16", "fp8"])
def test_dcp_tail_without_packer_is_the_stock_cast_split_concat(
    monkeypatch, use_fp8_prefill: bool
) -> None:
    backend, kv_nope, k_pe = _run_dcp_loop(
        monkeypatch, use_fp8_prefill=use_fp8_prefill, kv_pack=None
    )
    assert len(backend.calls) == 1
    k, v = backend.calls[0]
    ref_k, ref_v = _stock_kv_pack(kv_nope, k_pe, use_fp8_prefill)
    assert k.dtype == ref_k.dtype and v.dtype == ref_v.dtype
    assert _same_bytes(k, ref_k)
    assert _same_bytes(v, ref_v)


def test_dcp_tail_uses_the_packer_outputs_when_given(monkeypatch) -> None:
    seen: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
    sentinel_k = torch.zeros((_TOKENS, _NUM_HEADS, _QK_NOPE + _QK_ROPE))
    sentinel_v = torch.zeros((_TOKENS, _NUM_HEADS, _V_HEAD_DIM))

    def packer(kv_nope, k_pe, use_fp8_prefill):
        seen.append((kv_nope, k_pe, use_fp8_prefill))
        return sentinel_k, sentinel_v

    backend, kv_nope, k_pe = _run_dcp_loop(
        monkeypatch, use_fp8_prefill=True, kv_pack=packer
    )
    assert len(seen) == 1
    got_kv_nope, got_k_pe, got_fp8 = seen[0]
    # The packer sees the un-cast kv_b_proj output and the gathered k_pe.
    assert got_kv_nope.dtype == torch.bfloat16
    assert torch.equal(got_kv_nope, kv_nope)
    assert torch.equal(got_k_pe, k_pe)
    assert got_fp8 is True
    k, v = backend.calls[0]
    assert k is sentinel_k and v is sentinel_v


def test_kill_switch_reads_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK", raising=False)
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", raising=False)
    assert kimi_mla._dcp_fused_kv_pack_enabled()
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK", "0")
    assert not kimi_mla._dcp_fused_kv_pack_enabled()
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK", "1")
    assert kimi_mla._dcp_fused_kv_pack_enabled()
    # Disable polarity (the FLASHINFER_SPECIALIZED_KERNEL_DISABLE idiom): "1"
    # keeps the stock tail even with the enable switch explicitly on.
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", "1")
    assert not kimi_mla._dcp_fused_kv_pack_enabled()
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", "0")
    assert kimi_mla._dcp_fused_kv_pack_enabled()
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK", raising=False)
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", "1")
    assert not kimi_mla._dcp_fused_kv_pack_enabled()


def test_kill_switch_markers_name_the_switch_that_fired(monkeypatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(
        kimi_mla.logger, "info_once", lambda msg, *args: logged.append(msg % args)
    )
    layer = SimpleNamespace(dcp_world_size=8, impl=None)
    resolve = kimi_mla.MultiHeadLatentAttention._resolve_dcp_fused_kv_pack
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK", raising=False)
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", raising=False)

    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK", "0")
    assert resolve(layer) is False
    assert logged == [
        "Kimi-K3 DCP chunked-context KV pack: stock "
        "(kill switch VLLM_K3_DCP_FUSED_KV_PACK=0)"
    ]

    logged.clear()
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK", raising=False)
    monkeypatch.setenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", "1")
    assert resolve(layer) is False
    assert logged == [
        "Kimi-K3 DCP chunked-context KV pack: stock "
        "(kill switch VLLM_K3_DCP_FUSED_KV_PACK_DISABLE=1)"
    ]

    # No kill switch: an impl without the hook keeps the stock tail, an impl
    # with it gets the fused packer; each route logs its own marker.
    logged.clear()
    monkeypatch.delenv("VLLM_K3_DCP_FUSED_KV_PACK_DISABLE", raising=False)
    layer.impl = SimpleNamespace(
        _context_parallel_compute_prefill_context=lambda q, kv_c_and_k_pe: None
    )
    assert resolve(layer) is False
    assert logged == [
        "Kimi-K3 DCP chunked-context KV pack: stock (impl accepts no kv_pack)"
    ]

    logged.clear()
    loop_with_hook = MLACommonImpl._context_parallel_compute_prefill_context
    layer.impl = SimpleNamespace(
        _context_parallel_compute_prefill_context=loop_with_hook
    )
    assert resolve(layer) is True
    assert logged == [
        "Kimi-K3 DCP chunked-context KV pack: fused (VLLM_K3_DCP_FUSED_KV_PACK=1)"
    ]


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not hasattr(torch.ops._C, "fused_kimi_k3_mla_kv_concat_quant_fp8"),
    reason="Kimi-K3 fused K/V concat kernels require CUDA",
)
@pytest.mark.parametrize("use_fp8_prefill", [False, True], ids=["bf16", "fp8"])
def test_fused_packer_is_byte_exact_with_the_stock_tail(use_fp8_prefill: bool) -> None:
    torch.manual_seed(0)
    # Arbitrary shape; large enough for several blocks of the fused kernels.
    tokens, heads = 49, 12
    kv_nope = torch.randn(
        (tokens, heads, _QK_NOPE + _V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    k_pe = torch.randn((tokens, 1, _QK_ROPE), dtype=torch.bfloat16, device="cuda")
    layer = SimpleNamespace(
        qk_nope_head_dim=_QK_NOPE, v_head_dim=_V_HEAD_DIM, _dcp_fused_kv_pack=True
    )
    k, v = kimi_mla.MultiHeadLatentAttention._dcp_kv_pack(
        layer, kv_nope, k_pe, use_fp8_prefill
    )
    ref_k, ref_v = _stock_kv_pack(kv_nope, k_pe, use_fp8_prefill)
    assert _same_bytes(k, ref_k)
    assert _same_bytes(v, ref_v)
