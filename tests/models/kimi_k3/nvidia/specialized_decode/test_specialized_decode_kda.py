# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's KdaSplit against the stock KDA decode it replaces, at
the production dtypes (FP32 conv weight, BF16 conv state, FP32 recurrent state)
and TP8 heads.

Stock decode per step kind (``vllm/models/kimi_k3/nvidia/kda.py``):
- verify (q > 1), DS conv layout (production): Triton ``causal_conv1d_update``
  (Q/K/V stored BF16) -> ``fused_recurrent_kda`` (output stored BF16) -> gated
  RMSNorm (``rms_norm_gated``);
- verify, SD conv layout: FlashInfer ``packed_fused_kda_decode`` (BF16 Q/K/V and
  output in shared memory);
- q = 1 next to a speculative config (production, ``kda_decode_backend ==
  "triton"``): Triton ``causal_conv1d_update`` ->
  ``fused_recurrent_kda_packed_decode`` -> gated RMSNorm.

KdaSplit runs each stage at stock precision; only the reduction order differs. So
the states it writes match stock to FP32 reduction noise and almost every output
element is bit-identical. The RecoverSSM key record (the BF16 pre-normalization
key) is bit-identical to the stock convolution's key."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMRecords,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (  # noqa: E402
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E402
    PreparedKda,
    decode_conv_weight,
)
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    rms_norm_gated,
)

D, HEADS = 128, 12  # TP8 (and DCP8) Kimi-K3 KDA heads
DIM = HEADS * D
# Q, K, V, output gate, F_A, beta and the stock padding to 16 columns.
PITCH = 4 * DIM + D + HEADS + 4
LOWER_BOUND, EPS = -5.0, 1e-5
DEVICE = torch.device("cuda")


def _attention(seed, spec="native"):
    """A KDA layer's decode parameters at the production dtypes."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    return SimpleNamespace(
        conv1d=SimpleNamespace(weight=randn(3 * DIM, 1, 4) * 0.5, bias=None),
        dt_bias=randn(DIM) * 0.5,
        A_log=randn(HEADS) * 0.5,
        # F_B weights and F_A rows on a 2^-6 / integer grid: the gate projection is
        # exact in any order, so the raw gates (BF16) agree bit for bit.
        f_b_proj=SimpleNamespace(
            weight=(randn(DIM, D) * 3).round().clamp(-8, 8).bfloat16() / 64
        ),
        o_norm=SimpleNamespace(
            weight=(torch.rand(D, device=DEVICE, generator=g) + 0.5).bfloat16(),
            eps=EPS,
        ),
        gate_lower_bound=LOWER_BOUND,
        recoverssm_records=None,
        kda_spec_decode_backend=spec,
    )


def _inputs(seed, rows, slots, window):
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    projected = randn(rows, PITCH).bfloat16()
    projected[:, 4 * DIM : 4 * DIM + D] = (randn(rows, D) * 1.5).round().clamp(-3, 3)
    conv = randn(slots, 3 * DIM, window).bfloat16()  # DS: [slot, dim, window]
    state = randn(slots, HEADS, D, D) * 0.1
    return projected, conv, state


def _split(attention, projected):
    n = projected.shape[0]
    mixed = projected[:, : 3 * DIM]
    g2 = projected[:, 3 * DIM : 4 * DIM].view(n, HEADS, D)
    f_a = projected[:, 4 * DIM : 4 * DIM + D]
    beta = projected[:, 4 * DIM + D : 4 * DIM + D + HEADS].unsqueeze(0)
    g1 = (f_a @ attention.f_b_proj.weight.t()).view(1, n, HEADS, D)
    return mixed, g1, g2, beta


def _qkv(mixed):
    return (x.reshape(1, mixed.shape[0], HEADS, D) for x in mixed.split(DIM, dim=-1))


def _gated_norm(attention, core, g2):
    out = rms_norm_gated(
        core, g2, attention.o_norm.weight, None, "sigmoid", eps=attention.o_norm.eps
    )
    return out.reshape(core.shape[1], DIM)


def _stock_verify_triton(attention, projected, conv, state, table, accepted, q):
    """The stock verify on the DS conv layout (kda.py ``_forward``, native)."""
    n = projected.shape[0]
    mixed, g1, g2, beta = _split(attention, projected)
    starts = torch.arange(0, n + 1, q, dtype=torch.int32, device=DEVICE)
    weight = attention.conv1d.weight.view(3 * DIM, 4)
    mixed = causal_conv1d_update(
        mixed,
        conv,
        weight,
        None,
        activation="silu",
        conv_state_indices=table[:, 0],
        num_accepted_tokens=accepted,
        query_start_loc=starts,
        max_query_len=q,
        validate_data=False,
        out=torch.empty_like(mixed),
    )
    core, _ = fused_recurrent_kda(
        *_qkv(mixed),
        raw_g=g1,
        raw_beta=beta,
        A_log=attention.A_log,
        dt_bias=attention.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=state,
        cu_seqlens=starts,
        ssm_state_indices=table,
        num_accepted_tokens=accepted,
    )
    return _gated_norm(attention, core, g2), mixed


def _stock_verify_flashinfer(attention, projected, conv, state, table, accepted, q):
    """The stock verify on the SD conv layout (FlashInfer packed fused decode);
    ``conv`` is the [slot, dim, window] view of the SD cache."""
    from vllm.utils.flashinfer import flashinfer_packed_fused_kda_decode

    n = projected.shape[0]
    mixed, g1, g2, beta = _split(attention, projected)
    out = torch.empty(1, n, HEADS, D, dtype=torch.bfloat16, device=DEVICE)
    flashinfer_packed_fused_kda_decode(
        x=mixed,
        weight=decode_conv_weight(attention.conv1d.weight),
        conv_state=conv,
        raw_gate=g1,
        raw_beta=beta,
        A_log=attention.A_log,
        dt_bias=attention.dt_bias,
        state_indices=table,
        query_start_loc=torch.arange(0, n + 1, q, dtype=torch.int32, device=DEVICE),
        num_accepted_tokens=accepted,
        state=state,
        output_gate=g2,
        norm_weight=attention.o_norm.weight.float(),
        lower_bound=LOWER_BOUND,
        norm_eps=EPS,
        output=out,
    )
    return out.view(n, DIM)


def _stock_decode_triton(attention, projected, conv, state, slots):
    """The stock q = 1 decode next to a speculative config (backend "triton")."""
    mixed, g1, g2, beta = _split(attention, projected)
    weight = attention.conv1d.weight.view(3 * DIM, 4)
    mixed = causal_conv1d_update(
        mixed,
        conv,
        weight,
        None,
        activation="silu",
        conv_state_indices=slots,
        validate_data=True,
        out=torch.empty_like(mixed),
    )
    core, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed,
        raw_g=g1,
        raw_beta=beta,
        A_log=attention.A_log,
        dt_bias=attention.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=state,
        state_indices=slots,
    )
    return _gated_norm(attention, core, g2)


def _compare(name, ours, stock, states, stock_states):
    """The fraction of bit-identical output elements, the worst output difference
    relative to its row's RMS, and the mean written-state error relative to the
    mean state magnitude."""
    same = (ours.view(torch.int16) == stock.view(torch.int16)).float().mean().item()
    ours, stock = ours.float(), stock.float()
    rms = stock.pow(2).mean(-1, keepdim=True).sqrt()
    out_err = ((ours - stock).abs() / rms).max().item()
    state_err = (
        (states - stock_states).abs().mean() / stock_states.abs().mean()
    ).item()
    print(f"{name}: same {same:.5f} out {out_err:.3g} state {state_err:.3g}")
    return same, out_err, state_err


def _assert_stock_precision(same, out_err, state_err):
    # BF16 Q/K/V and output roundings matched: only FP32 reduction order (and, on
    # the SD layout, FlashInfer's rearranged recurrence) is left. Without them
    # about half of the outputs and every state differ (state error ~1e-3).
    assert same > 0.99
    assert out_err < 2e-2  # a few BF16 ulps at the row's scale
    assert state_err < 2e-5


def _view(cache, dim_first):
    """The [slot, dim, window] view the kernels take of a conv cache."""
    return cache if dim_first else cache.transpose(-1, -2)


def _run_verify(attention, projected, conv, state, table, accepted, q, dim_first):
    """``conv``: the conv cache in the deployment's layout."""
    requests = table.shape[0]
    caches = (conv, state)
    kda = PreparedKda(attention, {(requests, q): (PITCH, 1)}, dim_first)
    out = torch.empty(requests * q, DIM, device=DEVICE, dtype=torch.bfloat16)
    kda.prepare_caches(caches).run(
        projected, out, caches=caches, slots=table, accepted=accepted, q=q
    )
    return out


@pytest.mark.parametrize("q", [2, 5, 8])
@pytest.mark.parametrize("requests", [1, 4, 12])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
@torch.inference_mode()
def test_verify_matches_stock(requests, q, dim_first):
    """Verify steps against the stock verify of the deployment's conv layout."""
    attention = _attention(requests * 10 + q, spec="native" if dim_first else "fi")
    slots = 1 + requests * q
    projected, conv, state = _inputs(requests * 10 + q, requests * q, slots, q + 2)
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE).view(-1, q)
    accepted = torch.randint(
        1,
        q + 1,
        (requests,),
        dtype=torch.int32,
        device=DEVICE,
        generator=torch.Generator(device=DEVICE).manual_seed(q),
    )
    if not dim_first:  # the SD cache: [slot, window, dim]
        conv = conv.transpose(-1, -2).contiguous()

    ours_conv, ours_state = conv.clone(), state.clone()
    ours = _run_verify(
        attention, projected, ours_conv, ours_state, table, accepted, q, dim_first
    )
    stock_conv, stock_state = conv.clone(), state.clone()
    if dim_first:
        stock, _ = _stock_verify_triton(
            attention, projected, stock_conv, stock_state, table, accepted, q
        )
    else:
        stock = _stock_verify_flashinfer(
            attention,
            projected,
            _view(stock_conv, False),
            stock_state,
            table,
            accepted,
            q,
        )
    torch.testing.assert_close(ours_conv, stock_conv, atol=0, rtol=0)
    written = table.flatten().long()
    errors = _compare(
        f"verify R{requests} q{q} {'DS' if dim_first else 'SD'}",
        ours,
        stock,
        ours_state[written],
        stock_state[written],
    )
    _assert_stock_precision(*errors)


@pytest.mark.parametrize("q", [2, 5, 8])
@torch.inference_mode()
def test_verify_key_record_matches_stock_convolution(q):
    """RecoverSSM's key record is the convolution's BF16 key: bit for bit the
    stock ``causal_conv1d_update`` output."""
    requests = 4
    attention = _attention(q)
    records = KDARecoverSSMRecords.allocate(
        requests, HEADS, q, D, torch.bfloat16, DEVICE
    )
    attention.recoverssm_records = records
    slots = 1 + requests * q
    projected, conv, state = _inputs(q, requests * q, slots, q + 2)
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE).view(-1, q)
    checkpoints = table[:, :1].contiguous()
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)  # as RecoverSSM
    ours_conv = conv.clone()
    _run_verify(
        attention, projected, ours_conv, state.clone(), checkpoints, accepted, q, True
    )
    _, mixed = _stock_verify_triton(
        attention, projected, conv.clone(), state.clone(), table, accepted, q
    )
    key = mixed[:, DIM : 2 * DIM].reshape(requests, q, HEADS, D).transpose(1, 2)
    torch.testing.assert_close(records.key_gate[..., :D], key, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 64])
@torch.inference_mode()
def test_decode_matches_stock(batch):
    """Single-token steps against the stock single-token decode production runs (the
    Triton one, next to its speculative config)."""
    attention = _attention(batch)
    slots = 1 + batch
    projected, conv, state = _inputs(batch, batch, slots, 3)
    rows = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    mode = 1 if batch == 1 else 2 if batch <= 8 else 0
    kda = PreparedKda(attention, {batch: (PITCH, mode)}, True)
    ours_conv, ours_state = conv.clone(), state.clone()
    out = torch.empty(batch, DIM, device=DEVICE, dtype=torch.bfloat16)
    caches = (ours_conv, ours_state)
    kda.prepare_caches(caches).run(projected, out, caches=caches, slots=rows)
    stock_conv, stock_state = conv.clone(), state.clone()
    stock = _stock_decode_triton(attention, projected, stock_conv, stock_state, rows)
    torch.testing.assert_close(ours_conv, stock_conv, atol=0, rtol=0)
    errors = _compare(
        f"decode B{batch}",
        out,
        stock,
        ours_state[rows.long()],
        stock_state[rows.long()],
    )
    _assert_stock_precision(*errors)


@torch.inference_mode()
def test_div_full_is_reciprocal_product():
    """KdaSplit's composed precision divides by a positive b as a * rcp.approx(b),
    both scaled by 1/4 past b = 2^126 (and takes reciprocals of shared divisors
    once), relying on Triton's FP32 ``/`` (div.full.f32) being exactly that: here
    over random operands and bit patterns."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def divide(a_ptr, b_ptr, full_ptr, product_ptr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a, b = tl.load(a_ptr + i), tl.load(b_ptr + i)
        tl.store(full_ptr + i, a / b)
        product = tl.inline_asm_elementwise(
            "{ .reg .f32 a,b,r; .reg .pred big; mov.f32 a,$1; mov.f32 b,$2; "
            "setp.gt.f32 big,b,0f7E800000; @big mul.f32 a,a,0f3E800000; "
            "@big mul.f32 b,b,0f3E800000; rcp.approx.ftz.f32 r,b; mul.f32 $0,a,r; }",
            "=r,r,r",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(product_ptr + i, product)

    n = 1 << 24
    g = torch.Generator(device=DEVICE).manual_seed(0)
    randn = lambda scale: torch.randn(n, device=DEVICE, generator=g) * scale
    bits = (
        lambda low, high: torch.randint(
            low, high, (n,), device=DEVICE, generator=g, dtype=torch.int64
        )
        .int()
        .view(torch.float32)
    )
    cases = {
        "silu": (randn(8), 1 + torch.exp(randn(8))),
        "sigmoid": (torch.ones(n, device=DEVICE), 1 + torch.exp(randn(30))),
        "norm": (randn(1), torch.rand(n, device=DEVICE, generator=g) * 100 + 1e-3),
        "bits": (bits(-(2**31), 2**31 - 1), bits(0x00800000, 0x7F800000)),
        "large": (bits(-(2**31), 2**31 - 1), bits(0x7E000000, 0x7F800000)),
    }
    for name, (a, b) in cases.items():
        full, product = torch.empty_like(a), torch.empty_like(a)
        divide[(n // 1024,)](a, b, full, product, BLOCK=1024)
        finite = torch.isfinite(full)
        same = full.view(torch.int32) == product.view(torch.int32)
        assert bool((same | ~finite).all()), name
