# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.gemma4_qkv_glue import qkv_glue

EPS = 1e-6
NQ, NKV = 8, 2


def _rms(x, w):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        out = out * w.float()
    return out.to(torch.bfloat16)


def _rope(x, cos_sin, pos, rot_pairs):
    cos, sin = cos_sin.index_select(0, pos).float().chunk(2, dim=-1)
    half = x.shape[-1] // 2
    c = torch.ones(cos.shape[0], half, device=x.device)
    s = torch.zeros_like(c)
    c[:, :rot_pairs], s[:, :rot_pairs] = cos[:, :rot_pairs], sin[:, :rot_pairs]
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(torch.bfloat16)


def _fp8(x, scale):
    fp8 = current_platform.fp8_dtype()
    fmax = torch.finfo(fp8).max
    return (x.float() / scale).clamp(-fmax, fmax).to(fp8)


@pytest.mark.parametrize("head_dim,rot_pairs", [(256, 128), (512, 64)])
@pytest.mark.parametrize("write_kv", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 37, 300])
def test_qkv_glue(head_dim: int, rot_pairs: int, write_kv: bool, num_tokens: int):
    torch.manual_seed(0)
    dev, bs, fp8 = "cuda", 64, current_platform.fp8_dtype()
    cols = (NQ + 2 * NKV) * head_dim if write_kv else NQ * head_dim
    qkv = (torch.randn(num_tokens, cols, device=dev) * 3).to(torch.bfloat16)
    pos = torch.randint(0, 4096, (num_tokens,), device=dev)
    cos_sin = torch.randn(4096, head_dim, device=dev).to(torch.bfloat16)
    q_w = (torch.randn(head_dim, device=dev) * 0.5 + 1).to(torch.bfloat16)
    k_w = (torch.randn(head_dim, device=dev) * 0.5 + 1).to(torch.bfloat16)
    scales = [torch.tensor(s, device=dev) for s in (0.5, 1.0, 2.0)]
    nblocks = num_tokens // bs + 3
    kv = torch.zeros(nblocks, NKV, bs, 2 * head_dim, dtype=torch.uint8, device=dev)
    kc, vc = (t.view(fp8) for t in kv.transpose(1, 2).split(head_dim, dim=-1))
    slots = torch.randperm(nblocks * bs, device=dev)[:num_tokens]
    slots[::5] = -1

    q = qkv_glue(
        qkv, pos, cos_sin, q_w, k_w, *scales,
        kc if write_kv else None, vc if write_kv else None,
        slots if write_kv else None, NQ, NKV, head_dim, rot_pairs, EPS,
    )  # fmt: skip

    def check(out, ref):
        torch.testing.assert_close(out.float(), ref.float(), rtol=0.14, atol=1e-2)

    q_ref = _rope(_rms(qkv[:, : NQ * head_dim].view(-1, NQ, head_dim), q_w), cos_sin, pos, rot_pairs)
    check(q, _fp8(q_ref, scales[0]).view(num_tokens, -1))
    if not write_kv:
        return
    k = qkv[:, NQ * head_dim : (NQ + NKV) * head_dim].view(-1, NKV, head_dim)
    v = qkv[:, (NQ + NKV) * head_dim :].view(-1, NKV, head_dim)
    k_ref = _fp8(_rope(_rms(k, k_w), cos_sin, pos, rot_pairs), scales[1])
    v_ref = _fp8(_rms(v, None), scales[2])
    ok = slots >= 0
    blk, off = slots[ok] // bs, slots[ok] % bs
    check(kc[blk, off], k_ref[ok])
    check(vc[blk, off], v_ref[ok])
    untouched = torch.ones(nblocks * bs, dtype=torch.bool, device=dev)
    untouched[slots[ok]] = False
    idx = torch.arange(nblocks * bs, device=dev)[untouched]
    assert not kc[idx // bs, idx % bs].view(torch.uint8).any()
