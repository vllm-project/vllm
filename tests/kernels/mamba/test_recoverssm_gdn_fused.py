# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM verify in the fused CUDA MTP kernel (replay mode).

The replay mode must produce the native fused kernel's outputs (same math, norm and
gate), the replay records the Triton RecoverSSM verify writes (which the commit folds),
leave the checkpoints untouched, and skip CUDA graph padding rows."""

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip(
        reason="The fused GDN MTP kernel requires CUDA compute capability 8.0+.",
        allow_module_level=True,
    )

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (  # noqa: E402
    GDNRecoverSSMCommitContext,
    gdn_recoverssm_verify,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID  # noqa: E402

if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp_replay"):
    pytest.skip(
        reason="fused_gdn_decode_post_conv_mtp_replay is not built",
        allow_module_level=True,
    )

H, K, V, T = 4, 128, 128, 4
EPS = 1e-6


def _make(qlens, hv_per_h=3, state_dtype=torch.float32, nb=48, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    hv = H * hv_per_h
    tot = sum(qlens)
    qsl = [0]
    for q in qlens:
        qsl.append(qsl[-1] + q)
    return dict(
        hv=hv,
        mixed_qkv=rnd(tot, 2 * H * K + hv * V).bfloat16(),
        a=rnd(tot, hv).bfloat16(),
        b=rnd(tot, hv).bfloat16(),
        A_log=(torch.rand(hv, device="cuda", generator=g) * 2 - 1).float(),
        dt_bias=(rnd(hv) * 0.5).bfloat16(),
        state=(rnd(nb, hv, V, K) * 0.05).to(state_dtype),
        replay=torch.zeros(nb, hv, T, V + K + 1, device="cuda"),
        gate=rnd(tot, hv, V).bfloat16(),
        norm_weight=(1 + 0.1 * rnd(V)).float(),
        qsl=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
        slots=torch.arange(2, 2 + len(qlens), dtype=torch.int32, device="cuda"),
    )


def _replay_op(d, slots=None, qsl=None, replay=None, strided=False):
    slots = d["slots"] if slots is None else slots
    qsl = d["qsl"] if qsl is None else qsl
    if strided:
        # the layer passes the checkpoint column of [N, 1 + num_spec] indices
        wide = torch.full((slots.numel(), T), -7, dtype=torch.int32, device="cuda")
        wide[:, 0] = slots
        state_indices = wide[:, :1]
    else:
        state_indices = slots[:, None].contiguous()
    return ops.fused_gdn_decode_post_conv_mtp_replay(
        mixed_qkv=d["mixed_qkv"],
        a=d["a"],
        b=d["b"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        state_indices=state_indices,
        cu_seqlens=qsl,
        state=d["state"],
        replay=d["replay"] if replay is None else replay,
        output_gate=d["gate"],
        norm_weight=d["norm_weight"],
        scale=K**-0.5,
        norm_eps=EPS,
        output_gate_activation="sigmoid",
    )


def _native_op(d):
    """The snapshot path: every window position gets its own destination slot."""
    n = d["slots"].numel()
    state = d["state"].clone()
    extra = torch.arange(
        state.shape[0] - n * T, state.shape[0], dtype=torch.int32, device="cuda"
    )
    idx = torch.cat([d["slots"][:, None], extra.view(n, T)[:, 1:]], dim=1)
    out = ops.fused_gdn_decode_post_conv_mtp(
        mixed_qkv=d["mixed_qkv"],
        a=d["a"],
        b=d["b"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        state_indices=idx.contiguous(),
        cu_seqlens=d["qsl"],
        num_accepted_tokens=torch.ones(n, dtype=torch.int32, device="cuda"),
        state=state,
        output_gate=d["gate"],
        norm_weight=d["norm_weight"],
        scale=K**-0.5,
        norm_eps=EPS,
        output_gate_activation="sigmoid",
    )
    return out, state, idx


def _triton_verify(d, replay):
    hv, tot = d["hv"], d["mixed_qkv"].shape[0]
    qkv = d["mixed_qkv"]
    q = qkv[:, : H * K].reshape(1, tot, H, K)
    k = qkv[:, H * K : 2 * H * K].reshape(1, tot, H, K)
    v = qkv[:, 2 * H * K :].reshape(1, tot, hv, V)
    gdn_recoverssm_verify(
        d["A_log"],
        d["a"],
        d["b"],
        d["dt_bias"],
        q,
        k,
        v,
        checkpoint_state=d["state"],
        replay_cache=replay,
        query_start_loc=d["qsl"],
        state_indices=d["slots"],
        spec_query_len=T,
    )


def _valid_positions(d):
    lens = d["qsl"].diff().tolist()
    return [(int(s), t) for s, n in zip(d["slots"].tolist(), lens) for t in range(n)]


@pytest.mark.parametrize("qlens", [(4, 4, 4), (4, 2, 3, 1)])
@pytest.mark.parametrize("hv_per_h", [1, 3])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_replay_matches_native_outputs_and_triton_records(
    qlens, hv_per_h, state_dtype, strided
):
    d = _make(qlens, hv_per_h, state_dtype)
    before = d["state"].clone()
    out = _replay_op(d, strided=strided)
    ref_out, _, _ = _native_op(d)
    # Same kernel math, norm and gate: identical outputs.
    assert torch.equal(out, ref_out)
    assert torch.equal(d["state"], before)

    ref_replay = torch.zeros_like(d["replay"])
    _triton_verify(d, ref_replay)
    for slot, t in _valid_positions(d):
        torch.testing.assert_close(
            d["replay"][slot, :, t], ref_replay[slot, :, t], rtol=2e-4, atol=2e-5
        )
    # Only the verified slots and positions are written.
    written = torch.zeros_like(d["replay"], dtype=torch.bool)
    for slot, t in _valid_positions(d):
        written[slot, :, t] = True
    assert not d["replay"][~written].any()


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_commit_after_replay_matches_native_states(accepted):
    qlens = (4, 4, 3)
    d = _make(qlens)
    _replay_op(d)
    _, native_state, idx = _native_op(d)
    ckpt = d["state"].clone()
    conv = [torch.zeros(ckpt.shape[0], 8, 3 + T - 1, device="cuda").bfloat16()]
    ctx = GDNRecoverSSMCommitContext.from_tensors(
        conv, [ckpt], [d["replay"]], spec_query_len=T, max_num_reqs=8
    )
    acc = torch.tensor(
        [min(accepted, q) for q in qlens], dtype=torch.int32, device="cuda"
    )
    ctx.commit(acc, d["slots"], d["qsl"])
    for i, slot in enumerate(d["slots"].tolist()):
        ref = native_state[idx[i, int(acc[i]) - 1]]
        torch.testing.assert_close(ckpt[slot], ref, rtol=1e-4, atol=1e-5)


def test_graph_padding_rows_are_skipped():
    """FULL graph replay pads with null-slot, zero-length rows; a null slot with
    tokens gets zero output and writes nothing."""
    d = _make((4, 2, 3))
    ref_out = _replay_op(d)
    ref_replay = d["replay"].clone()

    d["replay"].zero_()
    padded_slots = torch.cat(
        [
            d["slots"],
            torch.full((2,), NULL_BLOCK_ID, dtype=torch.int32, device="cuda"),
        ]
    )
    padded_qsl = torch.cat([d["qsl"], d["qsl"][-1:].repeat(2)])
    out = _replay_op(d, slots=padded_slots, qsl=padded_qsl)
    assert torch.equal(out, ref_out)
    assert torch.equal(d["replay"], ref_replay)

    d["replay"].zero_()
    null_first = d["slots"].clone()
    null_first[0] = NULL_BLOCK_ID
    out = _replay_op(d, slots=null_first)
    assert not out[:4].any()
    assert torch.equal(out[4:], ref_out[4:])
    assert not d["replay"][NULL_BLOCK_ID].any()


def test_replay_rejects_bad_shapes():
    d = _make((4, 4))
    with pytest.raises(RuntimeError, match="replay must have shape"):
        _replay_op(d, replay=torch.zeros(48, d["hv"], T, V + K, device="cuda"))
    with pytest.raises(RuntimeError, match="one checkpoint column"):
        ops.fused_gdn_decode_post_conv_mtp_replay(
            mixed_qkv=d["mixed_qkv"],
            a=d["a"],
            b=d["b"],
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            state_indices=torch.stack([d["slots"], d["slots"]], 1).contiguous(),
            cu_seqlens=d["qsl"],
            state=d["state"],
            replay=d["replay"],
            output_gate=d["gate"],
            norm_weight=d["norm_weight"],
            scale=K**-0.5,
            norm_eps=EPS,
            output_gate_activation="sigmoid",
        )
