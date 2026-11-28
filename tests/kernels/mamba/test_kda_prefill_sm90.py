# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the SM90 KDA chunked prefill op `_C.kda_prefill_sm90`.

The op is checked against a pure-PyTorch fp32 chunked reference on varlen
batches that reach each of its kernel paths, with q/k/v as strided views of
one packed [T, 3, H, 128] buffer, strided beta and out rows, and a workspace
full of NaN bytes, as `Glm5NextLinearAttention._flashkda_prefill` calls it.
When FlashKDA (`_flashkda_C`) is built, its `fwd` runs on the same inputs.
"""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

D = 128
SCALE = D**-0.5
LOWER_BOUND = -5.0
REF_CHUNK = 16  # keeps exp(+-cumulative gate) inside fp32: 16 * 5 = 80

# Bounds: max |err| <= atol + rtol * |ref| elementwise and rms(err) <= rms.
# Tolerances against an fp64 reference on GLM-5.3-Flash-like inputs (the same
# generator as below).
TOLERANCES = {
    "out": dict(atol=5.6e-3, rtol=1.6e-2, rms=1.5e-4),
    "final_state": dict(atol=2.6e-1, rtol=1e-5, rms=1.5e-3),
    "checkpoint_state": dict(atol=8.8e-2, rtol=1e-5, rms=1.2e-3),
}


def _op_available() -> bool:
    return hasattr(torch.ops._C, "kda_prefill_sm90")


def _flashkda_fwd():
    try:
        import vllm._flashkda_C  # noqa: F401
    except ImportError:
        return None
    return getattr(torch.ops._flashkda_C, "fwd", None)


pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(90) or not _op_available(),
    reason="kda_prefill_sm90 needs SM90 and a build with the op.",
)

# (num_heads, sequence lengths). With 132 SMs these reach every launch path:
# 64-token chunks (H * N <= 16, with and without the overlapped prep), the
# overlapped 32-token path, one CTA per SM, two CTAs per SM and the H > 32
# variant of it.
CASES = {
    "h8_short_64tok": (8, [300, 1]),
    "h16_long_64tok_overlap": (16, [4500]),
    "h16_long_overlap": (16, [4100, 37]),
    "h8_one_cta_per_sm": (8, [1, 700, 50, 129]),
    "h16_mixed_two_cta_per_sm": (
        16,
        [1, 17, 32, 33, 100, 128, 255, 1, 64, 513, 1000, 2047],
    ),
    "h16_many_short": (16, [1 + (7 * i) % 40 for i in range(64)]),
    "h64_mixed": (64, [1, 40, 200]),
}


def _make_inputs(num_heads: int, lens: list[int], seed: int, device: str):
    """GLM-5.3-Flash-like KDA inputs: SiLU q/k/v with rare outliers, raw gate
    and beta logits, and in every fourth head of each sequence a long-memory
    head (no decay, small beta) whose state carries the whole history."""
    g_ = torch.Generator(device=device).manual_seed(seed)
    h, t, n = num_heads, sum(lens), len(lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g_, device=device)

    def rand(*shape):
        return torch.rand(*shape, generator=g_, device=device)

    a_log = 0.35 + 2.0 * rand(h)
    dt_bias = (-4.0 * rand(h, 1) ** 2) + (0.1 + 0.9 * rand(h, 1)) * randn(h, D)
    x = randn(t, 3, h, D)
    x = torch.where(rand(*x.shape) < 1e-3, 6.0 * x, x)
    qkv = torch.nn.functional.silu(x).to(torch.bfloat16)

    lens_t = torch.tensor(lens, device=device)
    seq_of_token = torch.repeat_interleave(torch.arange(n, device=device), lens_t)
    g = 0.5 * randn(h, D) + 0.5 * randn(t, h, D)
    beta = 1.5 * randn(t, h)
    long_off = (rand(n) * 4).long()[seq_of_token]
    long_mem = (torch.arange(h, device=device)[None] % 4) == long_off[:, None]
    g_long = -dt_bias - 20.0 / a_log.exp()[:, None]
    g = torch.where(long_mem[..., None], g_long + 0.1 * randn(t, h, D), g)
    beta = torch.where(long_mem, -5.0 + 0.5 * randn(t, h), beta)

    cu = torch.zeros(n + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lens_t, 0)
    return dict(
        qkv=qkv,
        g=g.to(torch.bfloat16),
        beta=beta.to(torch.bfloat16),
        A_log=a_log,
        dt_bias=dt_bias.contiguous(),
        cu_seqlens=cu,
        initial_state=0.3 * randn(n, h, D, D),
        checkpoint_in=randn(n, h, D, D),
    )


def _checkpoint_offsets(lens: list[int]) -> list[int]:
    """Multiples of 16 in (0, len]: none, the last boundary (the sequence end
    when len % 16 == 0) or the first one, by turns."""
    offsets = []
    for i, length in enumerate(lens):
        if i % 3 == 1:
            offsets.append(length // 16 * 16)
        elif i % 3 == 2:
            offsets.append(16 if length >= 16 else 0)
        else:
            offsets.append(0)
    return offsets


def _reference(
    qkv, g, beta, A_log, dt_bias, initial_state, cu_seqlens, offsets, ckpt_in
):
    """fp32 chunked KDA. Per sequence and head, token by token from
    S = initial_state ([V, K]):

        qh, kh = q, k / sqrt(sum(x * x) + 1e-6)
        gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); b = sigmoid(beta)
        S = exp(gate)[None, :] * S
        S += b * (v - S @ kh)[:, None] * kh[None, :]
        out = bf16(scale * S @ qh)

    computed in chunks of REF_CHUNK tokens (WY form, triangular solve), all
    sequences advancing together. Returns out, final state and the state after
    `offsets[n]` tokens (input checkpoint where the offset is 0)."""
    f32, c = torch.float32, REF_CHUNK
    dev = qkv.device
    qkv = qkv.to(f32)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(
        A_log.exp()[None, :, None] * (g.to(f32) + dt_bias[None])
    )
    b = torch.sigmoid(beta.to(f32))

    cu = cu_seqlens.tolist()
    n_seq = len(cu) - 1
    nch = [-(-(cu[i + 1] - cu[i]) // c) for i in range(n_seq)]
    max_ch = max(nch)
    # chunk j of sequence i at row i * max_ch + j; empty chunks are padding
    starts = torch.tensor(
        [cu[i] + j * c for i in range(n_seq) for j in range(max_ch)], device=dev
    )
    ends = torch.tensor(
        [cu[i + 1] for i in range(n_seq) for _ in range(max_ch)], device=dev
    )
    pos = starts[:, None] + torch.arange(c, device=dev)[None]
    valid = pos < ends[:, None]
    pos = torch.where(valid, pos, 0)

    def gather(x):  # [T, H, ...] -> [N, max_ch, H, C, ...], zero on padding
        y = x[pos]
        y = torch.where(valid.view(*valid.shape, *([1] * (y.dim() - 2))), y, 0)
        return y.transpose(1, 2).unflatten(0, (n_seq, max_ch))

    Q, K, V, B = gather(q), gather(k), gather(v), gather(b)
    G = torch.cumsum(gather(gate), dim=3)  # log decay from the chunk start
    eG, emG = G.exp(), (-G).exp()
    Kp, Km, Qp = K * eG, K * emG, Q * eG
    kk = (Kp @ Km.transpose(-1, -2)).tril(-1)  # <k_i, k_j> exp(G_i - G_j), j < i
    qk = (Qp @ Km.transpose(-1, -2)).tril()  # <q_i, k_j> exp(G_i - G_j), j <= i
    lower = torch.eye(c, dtype=f32, device=dev) + B[..., None] * kk
    W1 = torch.linalg.solve_triangular(
        lower, B[..., None] * V, upper=False, unitriangular=True
    )
    W2 = torch.linalg.solve_triangular(
        lower, B[..., None] * Kp, upper=False, unitriangular=True
    )
    eGl = eG[..., -1, :]  # decay over the whole chunk [N, max_ch, H, K]
    Kdecay = Km * eGl[..., None, :]

    S = initial_state.to(f32).clone()  # [N, H, V, K]
    Sstart = torch.empty(n_seq, max_ch, *S.shape[1:], dtype=f32, device=dev)
    U = torch.empty_like(V)
    for j in range(max_ch):  # padding chunks leave S unchanged (eGl = 1, U = 0)
        Sstart[:, j] = S
        U[:, j] = W1[:, j] - W2[:, j] @ S.transpose(-1, -2)
        S = S * eGl[:, j, :, None, :] + U[:, j].transpose(-1, -2) @ Kdecay[:, j]
    chunk_out = Qp @ Sstart.transpose(-1, -2) + qk @ U  # [N, max_ch, H, C, V]

    out = torch.empty_like(v)
    out[pos[valid]] = chunk_out.flatten(0, 1).transpose(1, 2)[valid]
    ckpt = ckpt_in.clone()
    for i, off in enumerate(offsets):
        if off > 0:  # a multiple of 16: the state at a chunk boundary
            ckpt[i] = S[i] if off // c == nch[i] else Sstart[i, off // c]
    return (SCALE * out).to(torch.bfloat16), S, ckpt


def _error_ratios(actual, expected, tol):
    """(max |err| / (atol + rtol |expected|), rms(err) / rms tolerance)."""
    a, e = actual.float(), expected.float()
    if not torch.isfinite(a).all():
        return math.inf, math.inf
    err = (a - e).abs()
    ratio = (err / (tol["atol"] + tol["rtol"] * e.abs())).max().item()
    rms = err.double().pow(2).mean().sqrt().item()
    return ratio, rms / tol["rms"]


def _assert_close(name, actual, expected, scale=1.0):
    tol = {key: scale * value for key, value in TOLERANCES[name].items()}
    ratio, rms_ratio = _error_ratios(actual, expected, tol)
    print(f"{name}: max error ratio {ratio:.3f}, rms ratio {rms_ratio:.3f}")
    assert ratio <= 1.0 and rms_ratio <= 1.0, (
        f"{name}: max error ratio {ratio:.3f}, rms ratio {rms_ratio:.3f}"
    )


@pytest.mark.parametrize("use_checkpoints", [False, True])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_kda_prefill_sm90(case, use_initial_state, use_checkpoints):
    device = "cuda"
    num_heads, lens = CASES[case]
    h, t, n = num_heads, sum(lens), len(lens)
    inp = _make_inputs(h, lens, seed=len(lens) * 1000 + t, device=device)
    if not use_initial_state:
        inp["initial_state"].zero_()  # vLLM's state for a new prompt
    offsets = _checkpoint_offsets(lens) if use_checkpoints else [0] * n
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device=device)

    # vLLM's layout: q, k, v views of the packed conv output (token stride
    # 3 H D plus padding here), beta a column slice of the merged projection,
    # out rows inside a wider buffer, and a workspace that holds other data.
    row = 3 * h * D + 64
    conv = torch.full((t, row), float("nan"), dtype=torch.bfloat16, device=device)
    conv[:, : 3 * h * D] = inp["qkv"].reshape(t, 3 * h * D)
    q, k, v = (conv[:, i * h * D : (i + 1) * h * D].view(1, t, h, D) for i in range(3))
    proj = torch.full(
        (t, 3 * h * D + h + 2 * D), float("nan"), dtype=torch.bfloat16, device=device
    )
    proj[:, 3 * h * D : 3 * h * D + h] = inp["beta"]
    beta = proj[:, 3 * h * D : 3 * h * D + h].unsqueeze(0)
    out_buf = torch.full(
        (t, h * D + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = out_buf[:, : h * D].view(1, t, h, D)
    workspace = torch.full(
        (ops.kda_prefill_sm90_workspace_size(t, h, n),),
        0xFF,  # NaN in every float format
        dtype=torch.uint8,
        device=device,
    )
    final_state = torch.full_like(inp["initial_state"], float("nan"))
    checkpoint_state = inp["checkpoint_in"].clone() if use_checkpoints else None

    ops.kda_prefill_sm90(
        q,
        k,
        v,
        inp["g"].unsqueeze(0),
        beta,
        SCALE,
        out,
        workspace,
        inp["A_log"],
        inp["dt_bias"],
        LOWER_BOUND,
        inp["initial_state"],
        final_state,
        inp["cu_seqlens"],
        checkpoint_state,
        offsets_t if use_checkpoints else None,
    )

    ref_out, ref_state, ref_ckpt = _reference(
        inp["qkv"],
        inp["g"],
        inp["beta"],
        inp["A_log"],
        inp["dt_bias"],
        inp["initial_state"],
        inp["cu_seqlens"],
        offsets,
        inp["checkpoint_in"],
    )
    _assert_close("out", out[0], ref_out)
    _assert_close("final_state", final_state, ref_state)
    assert torch.isnan(out_buf[:, h * D :].float()).all(), "wrote past out rows"
    if use_checkpoints:
        _assert_close("checkpoint_state", checkpoint_state, ref_ckpt)
        keep = offsets_t == 0
        assert torch.equal(checkpoint_state[keep], inp["checkpoint_in"][keep])

    fwd = _flashkda_fwd()
    if fwd is None:
        return
    fk_out = torch.empty(1, t, h, D, dtype=torch.bfloat16, device=device)
    fk_state = torch.empty_like(final_state)
    fk_ckpt = inp["checkpoint_in"].clone() if use_checkpoints else None
    fk_workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(t, h, n),
        dtype=torch.uint8,
        device=device,
    )
    fwd(
        q,
        k,
        v,
        inp["g"].unsqueeze(0),
        beta.contiguous(),
        SCALE,
        fk_out,
        fk_workspace,
        inp["A_log"],
        inp["dt_bias"],
        LOWER_BOUND,
        inp["initial_state"],
        fk_state,
        inp["cu_seqlens"],
        fk_ckpt,
        offsets_t if use_checkpoints else None,
    )
    # Both within the bounds of the reference, so within twice the bounds of
    # each other.
    _assert_close("out", out[0], fk_out[0], scale=2.0)
    _assert_close("final_state", final_state, fk_state, scale=2.0)
    if use_checkpoints:
        _assert_close("checkpoint_state", checkpoint_state, fk_ckpt, scale=2.0)
