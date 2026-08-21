# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused ROCm KDA chunk kernel must match the Triton chunk path.

The fused kernel reassociates nothing, but it keeps the chunk states in
registers and rounds them to bfloat16 only where the Triton path does, so the
outputs agree to bfloat16 rounding and the final states nearly exactly.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(),
    reason="The fused KDA chunk kernel is only built for gfx950",
)

HEAD_DIM = 128
NUM_HEADS = 12
LOWER_BOUND = -5.0


def _requires_kernel() -> None:
    if not hasattr(torch.ops._C, "fused_kda_chunk"):
        pytest.skip("vLLM was built without the fused KDA chunk kernel")


def _inputs(seqlens: list[int], seed: int) -> dict:
    torch.manual_seed(seed)
    dev, dtype = "cuda", torch.bfloat16
    total = sum(seqlens)
    cu = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    shape = (1, total, NUM_HEADS, HEAD_DIM)
    return dict(
        q=torch.randn(shape, device=dev, dtype=dtype) * 0.5,
        k=torch.randn(shape, device=dev, dtype=dtype) * 0.5,
        v=torch.randn(shape, device=dev, dtype=dtype) * 0.5,
        raw_g=torch.randn(shape, device=dev, dtype=dtype),
        raw_beta=torch.randn(1, total, NUM_HEADS, device=dev, dtype=dtype),
        A_log=torch.randn(NUM_HEADS, device=dev, dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(NUM_HEADS * HEAD_DIM, device=dev, dtype=torch.float32)
        * 0.5,
        h0=torch.randn(
            len(seqlens), NUM_HEADS, HEAD_DIM, HEAD_DIM, device=dev, dtype=torch.float32
        )
        * 0.1,
        cu=cu,
    )


def _run(inp: dict, use_fused: bool, out: torch.Tensor | None = None, **kw):
    from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill

    return chunk_kda_prefill(
        q=inp["q"].clone(),
        k=inp["k"].clone(),
        v=inp["v"].clone(),
        raw_g=inp["raw_g"].clone(),
        raw_beta=inp["raw_beta"],
        A_log=inp["A_log"],
        g_bias=inp["dt_bias"],
        lower_bound=LOWER_BOUND,
        initial_state=inp["h0"].clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu"],
        use_fused_chunk=use_fused,
        out=out,
        **kw,
    )


def _force_groups(monkeypatch, groups: int) -> None:
    """Pin the group count, which is otherwise a function of the device."""
    import vllm.models.kimi_k3.amd.ops.kda_chunk as kda_chunk

    monkeypatch.setattr(kda_chunk, "_chunk_groups", lambda *_: groups)


@pytest.mark.parametrize(
    "seqlens",
    [
        [1],  # shorter than a chunk
        [64],  # exactly one chunk
        [1024],
        [4096],  # many chunks, one sequence
        [513, 64, 1, 1200],  # ragged, including a partial tail chunk
        [512] * 6,
    ],
)
def test_fused_chunk_matches_triton(seqlens: list[int]) -> None:
    _requires_kernel()

    inp = _inputs(seqlens, seed=len(seqlens))
    o_ref, state_ref = _run(inp, use_fused=False)
    o_fused, state_fused = _run(inp, use_fused=True)

    torch.testing.assert_close(o_fused.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("use_fused", [False, True])
def test_out_argument_receives_the_result(use_fused: bool) -> None:
    """``out`` is a promise about where the result lands.

    The layer passes a slice of its own output buffer so neither path has to
    copy afterwards, and it then skips the copy based on the returned tensor
    aliasing that buffer. Both must hold for the fused path and for the Triton
    fallback, which allocates its own buffer internally.
    """
    if use_fused:
        _requires_kernel()

    seqlens = [321, 64, 700]
    inp = _inputs(seqlens, seed=13)
    o_ref, _ = _run(inp, use_fused=use_fused)

    # A non-zero offset into a larger buffer, as the layer produces when a
    # mixed batch puts decode tokens first.
    total = sum(seqlens)
    buf = torch.empty(
        1, total + 8, NUM_HEADS, HEAD_DIM, device="cuda", dtype=o_ref.dtype
    )
    view = buf[:, 8:]
    o_got, _ = _run(inp, use_fused=use_fused, out=view)

    assert o_got.data_ptr() == view.data_ptr()
    torch.testing.assert_close(view.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


def test_fused_chunk_accepts_projection_views() -> None:
    """`beta` and `g` arrive as last-dim slices of the fused QKVGFAB projection.

    Those views carry the projection's row stride, so a kernel that indexes them
    densely reads the wrong elements or rejects the tensor outright. Build the
    inputs the way the layer does rather than as standalone tensors.
    """
    _requires_kernel()

    seqlens = [777, 64, 391]
    inp = _inputs(seqlens, seed=7)
    total = sum(seqlens)

    # One wide projection, split on the last dim exactly like KimiK3DeltaAttention.
    qkv_width = 3 * NUM_HEADS * HEAD_DIM
    projection = torch.randn(
        total, qkv_width + NUM_HEADS, device="cuda", dtype=torch.bfloat16
    )
    beta_view = projection.split([qkv_width, NUM_HEADS], dim=-1)[1].unsqueeze(0)
    assert not beta_view.is_contiguous()

    inp["raw_beta"] = beta_view.contiguous()
    o_ref, state_ref = _run(inp, use_fused=False)

    inp["raw_beta"] = beta_view
    o_fused, state_fused = _run(inp, use_fused=True)

    torch.testing.assert_close(o_fused.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=1e-3, atol=1e-3)


def test_fused_chunk_survives_correlated_keys_and_real_gate_bias() -> None:
    """The input regime the served model actually produces.

    This is what forces the intra-chunk inverse to keep fp32 MFMA operands: at
    bf16 the Neumann doubling's intermediate powers cancel catastrophically
    here, diverging by orders of magnitude and driving the final state to inf.

    Three properties of real traffic combine here and in none of the tests
    above, all measured from tensors captured off a live TP8 server:

    * repetitive text gives near-duplicate L2-normalized keys, so ``k_i . k_j``
      approaches 1 and the Neumann terms stop shrinking;
    * the checkpoint's ``dt_bias`` lies in ``[-7.8, -1.4]``, which makes the
      within-chunk gate cumsum span ~300 powers of two;
    * ``beta`` saturates near 0.95.

    ``_inputs`` draws all three from ``randn``, landing in the benign regime.
    """
    _requires_kernel()

    torch.manual_seed(11)
    dev, dtype = "cuda", torch.bfloat16
    total = 1088
    shape = (1, total, NUM_HEADS, HEAD_DIM)
    base = torch.randn(1, 1, NUM_HEADS, HEAD_DIM, device=dev, dtype=dtype)

    def correlated():
        return (base + torch.randn(shape, device=dev, dtype=dtype) * 0.02).contiguous()

    inp = dict(
        q=correlated(),
        k=correlated(),
        v=torch.randn(shape, device=dev, dtype=dtype) * 0.5,
        raw_g=torch.randn(shape, device=dev, dtype=dtype),
        raw_beta=torch.full((1, total, NUM_HEADS), 3.0, device=dev, dtype=dtype),
        A_log=torch.rand(NUM_HEADS, device=dev, dtype=torch.float32) * 0.6 - 0.42,
        dt_bias=torch.rand(NUM_HEADS * HEAD_DIM, device=dev, dtype=torch.float32) * 6.4
        - 7.8,
        h0=torch.zeros(
            1, NUM_HEADS, HEAD_DIM, HEAD_DIM, device=dev, dtype=torch.float32
        ),
        cu=torch.tensor([0, total], device=dev, dtype=torch.int32),
    )

    o_ref, state_ref = _run(inp, use_fused=False)
    o_fused, state_fused = _run(inp, use_fused=True)

    # Looser than the tests above: near-identical keys plus a saturated beta
    # are past anything real text produces, and the inverse is rounded to bf16
    # before the w/u GEMMs. The tolerance still catches divergence, which in
    # this regime is orders of magnitude rather than a few times the bound.
    torch.testing.assert_close(o_fused.float(), o_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize(
    "seqlens",
    [
        [1],  # every conv tap comes from the cache
        [2],  # cache and batch both contribute
        [64],
        [1024],
        [513, 64, 1, 1200],  # ragged, tail chunk shorter than the conv width
    ],
)
@pytest.mark.parametrize("has_state", [False, True])
def test_fused_conv_matches_reference_conv(seqlens: list[int], has_state: bool) -> None:
    """The prologue's optional fused convolution must match a separate conv.

    The layer does not use this path -- recomputing the conv once per load
    phase costs more than the ``causal_conv1d_fn`` launches it saves -- so it
    is driven through the ops directly rather than through
    ``chunk_kda_prefill``, which does not expose it.

    The reference is computed in torch rather than with ``causal_conv1d_fn``,
    which is not self-consistent for these shapes on ROCm (it leaves part of
    its output unwritten, so repeated identical calls disagree).

    The cases that matter are the ones touching the sequence boundary, where
    the fused path takes taps from the conv cache instead of the batch:
    sequences shorter than the kernel width, and a cache that is or is not
    populated. The rolled-forward cache is checked too, since a wrong shift
    only shows up on the next step.
    """
    import torch.nn.functional as F

    from vllm.models.kimi_k3.amd.ops.kda_chunk import (
        fused_kda_chunk,
        fused_kda_prologue,
    )

    _requires_kernel()

    dev, dtype = "cuda", torch.bfloat16
    lp = NUM_HEADS * HEAD_DIM
    width, n = 4, len(seqlens)
    inp = _inputs(seqlens, seed=7 + len(seqlens) + int(has_state))
    total = int(inp["cu"][-1])

    torch.manual_seed(11 + int(has_state))
    # The layer hands the prologue a strided view of the QKVGFAB projection.
    projection = torch.randn(total, 3 * lp + 64, device=dev, dtype=dtype) * 0.5
    mixed = projection[:, : 3 * lp]
    # conv1d is built with bias=False and fp32 weights; the fused kernel takes
    # the width-major [qkv, width, channel] mirror the decode path already
    # keeps.
    weight = torch.randn(3, width, lp, device=dev, dtype=torch.float32) * 0.3
    store = torch.zeros(n, width - 1, 3 * lp, device=dev, dtype=dtype)
    if has_state:
        store.normal_(0.0, 0.5)
    state = store.transpose(-1, -2)
    idx = torch.arange(n, device=dev, dtype=torch.int32)
    has_init = torch.full((n,), has_state, device=dev, dtype=torch.bool)

    # Reference conv, per sequence, in fp32 then rounded like a bf16 output.
    conv_ref = torch.empty(total, 3 * lp, device=dev, dtype=dtype)
    state_ref = torch.empty_like(store)
    flat_w = weight.permute(0, 2, 1).reshape(3 * lp, width)
    for i, (b, e) in enumerate(zip(inp["cu"][:-1].tolist(), inp["cu"][1:].tolist())):
        pre = (
            store[i].float()
            if has_state
            else torch.zeros(width - 1, 3 * lp, device=dev, dtype=torch.float32)
        )
        xs = torch.cat([pre, mixed[b:e].float()], 0)
        acc = torch.zeros(e - b, 3 * lp, device=dev, dtype=torch.float32)
        for w in range(width):
            acc += xs[w : w + (e - b)] * flat_w[:, w].unsqueeze(0)
        conv_ref[b:e] = F.silu(acc).to(dtype)
        state_ref[i] = xs[-(width - 1) :].to(dtype)

    def _bands(x, contiguous):
        out = [
            b.unsqueeze(0).unflatten(-1, (NUM_HEADS, HEAD_DIM))
            for b in x.split(lp, dim=-1)
        ]
        return [b.contiguous() for b in out] if contiguous else out

    # Compare through both kernels rather than the prologue workspaces: `aqk`
    # is deliberately left unwritten past each chunk's valid rows, so those
    # elements are whatever the allocator last held.
    scale = HEAD_DIM**-0.5

    def _through_kernels(q, k, v, conv: dict | None):
        ws = fused_kda_prologue(
            q=q,
            k=k,
            v=v,
            raw_g=inp["raw_g"],
            raw_beta=inp["raw_beta"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            scale=scale,
            lower_bound=LOWER_BOUND,
            cu_seqlens=inp["cu"],
            **(conv or {}),
        )
        return fused_kda_chunk(
            qg=ws["qg"],
            w=ws["w"],
            u=ws["u"],
            kg_t=ws["kg_t"],
            aqk=ws["aqk"],
            decay=ws["decay"],
            out=torch.empty_like(ws["u"]),
            scale=scale,
            cu_seqlens=inp["cu"],
            initial_state=inp["h0"].clone(),
            output_final_state=True,
        )

    qr, kr, vr = _bands(conv_ref, True)
    o_ref, s_ref = _through_kernels(qr, kr, vr, None)

    got_state = state.clone()
    qf, kf, vf = _bands(mixed, False)
    o_got, s_got = _through_kernels(
        qf,
        kf,
        vf,
        dict(
            conv_weight=weight,
            conv_state=got_state,
            conv_state_indices=idx,
            conv_has_initial_state=has_init,
        ),
    )

    torch.testing.assert_close(o_got.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s_got, s_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        got_state.transpose(-1, -2).float(), state_ref.float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("groups", [2, 4, 8])
@pytest.mark.parametrize("seqlens", [[64], [64, 64], [100, 64], [513, 64, 1]])
def test_chunk_group_scan_matches_serial_walk(
    groups: int, seqlens: list[int], monkeypatch
) -> None:
    """Splitting the chunk walk into parallel groups must not change the result.

    Two bugs lived here, both invisible while the path was gated to >= 128
    chunks per sequence:

    * a group with no chunks returned before writing its ``B_g``, so the scan
      folded whatever the workspace last held into the final state — these
      shapes have fewer chunks than groups, so most groups are empty;
    * the second pass started every group after the first from a zero state
      instead of the one the scan composed, which only looked right because a
      strongly decaying gate makes the dropped term small.
    """
    _requires_kernel()

    inp = _inputs(seqlens, seed=5 + groups)
    _force_groups(monkeypatch, 1)
    o_ref, state_ref = _run(inp, use_fused=True)
    _force_groups(monkeypatch, groups)
    o_got, state_got = _run(inp, use_fused=True)

    torch.testing.assert_close(o_got.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_got, state_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("seqlens", [[320], [513, 64, 1, 1200], [64, 64, 64]])
def test_accepts_chunk_metadata_from_the_attention_builder(
    seqlens: list[int],
) -> None:
    """The layer passes precomputed chunk metadata, not None.

    ``KimiK3ROCmKDAMetadataBuilder`` builds these on device to avoid a blocking
    device->host copy. Passing ``None`` instead takes a different path that
    builds them locally, so drive the op with the builder's own tensors.
    """
    _requires_kernel()

    from vllm.models.kimi_k3.amd.kda_metadata import prepare_chunk_metadata_device
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    inp = _inputs(seqlens, seed=17)
    cu = inp["cu"]
    chunk_indices, chunk_offsets = prepare_chunk_metadata_device(
        cu, cu.cpu(), FLA_CHUNK_SIZE
    )

    o_ref, state_ref = _run(inp, use_fused=True)
    o_got, state_got = _run(
        inp, use_fused=True, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
    )

    torch.testing.assert_close(o_got.float(), o_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_got, state_ref, rtol=1e-3, atol=1e-3)


def _checkpoint_buffer(rows: int) -> torch.Tensor:
    """A destination filled with NaN, so an unwritten row is unmistakable."""
    return torch.full(
        (rows, NUM_HEADS, HEAD_DIM, HEAD_DIM),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )


def _prefix_inputs(inp: dict, seq: int, num_tokens: int) -> dict:
    """Sequence ``seq`` truncated to its first ``num_tokens`` tokens."""
    bos = int(inp["cu"][seq])
    tokens = slice(bos, bos + num_tokens)
    prefix = dict(inp)
    for key in ("q", "k", "v", "raw_g", "raw_beta"):
        prefix[key] = inp[key][:, tokens].contiguous()
    prefix["h0"] = inp["h0"][seq : seq + 1].contiguous()
    prefix["cu"] = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    return prefix


def _run_with_checkpoints(
    inp: dict, offsets: list[int], rows: int | None = None, **kw
) -> tuple[torch.Tensor, tuple]:
    ckpt = _checkpoint_buffer(len(offsets) if rows is None else rows)
    result = _run(
        inp,
        use_fused=True,
        checkpoint_state=ckpt,
        checkpoint_offsets=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        **kw,
    )
    return ckpt, result


@pytest.mark.parametrize(
    "seqlens,offsets",
    [
        ([256], [64]),  # the first interior boundary
        ([256], [192]),  # the last one
        ([1024], [512]),
        ([320, 320], [64, 256]),  # every sequence checkpoints
        ([513, 64, 1, 1200], [448, 0, 0, 640]),  # ragged, most opted out
    ],
)
def test_checkpoint_matches_a_prefill_that_stops_at_the_offset(
    seqlens: list[int], offsets: list[int], monkeypatch
) -> None:
    """A checkpoint is only useful if resuming from it is indistinguishable.

    The scheduler serves the exported state on a later prefix-cache hit as if
    the prompt had been prefilled up to that block and no further, so the
    kernel has to hand back exactly what a prefill truncated there produces --
    not merely something close. The walk passes through that state anyway, so
    equality is bit-exact and the test asserts it as such.

    Sequences with offset 0 opt out, and their rows must stay untouched.
    """
    _requires_kernel()
    _force_groups(monkeypatch, 1)

    inp = _inputs(seqlens, seed=23 + len(seqlens))
    ckpt, _ = _run_with_checkpoints(inp, offsets)

    for i, offset in enumerate(offsets):
        if offset == 0:
            assert ckpt[i].isnan().all()
            continue
        _, truncated = _run(_prefix_inputs(inp, i, offset), use_fused=True)
        torch.testing.assert_close(ckpt[i], truncated[0], rtol=0, atol=0)


def test_checkpoint_leaves_the_output_and_final_state_alone() -> None:
    """Exporting a checkpoint must not perturb the walk it is taken from."""
    _requires_kernel()

    seqlens = [768, 320]
    inp = _inputs(seqlens, seed=29)
    o_ref, state_ref = _run(inp, use_fused=True)
    _, (o_got, state_got) = _run_with_checkpoints(inp, [512, 128])

    torch.testing.assert_close(o_got.float(), o_ref.float(), rtol=0, atol=0)
    torch.testing.assert_close(state_got, state_ref, rtol=0, atol=0)


@pytest.mark.parametrize("offset", [100, 63, 65, 1024])
def test_unusable_checkpoint_offsets_write_nothing(offset: int) -> None:
    """Offsets the walk never reaches must leave the destination untouched.

    The recurrent state only exists at chunk boundaries, so an offset that is
    not a multiple of ``KDA_CHECKPOINT_ALIGNMENT`` -- or that lies past the end
    of the sequence -- cannot be honoured. Rounding it to a nearby boundary
    would hand the scheduler a state for the wrong token count, so the kernel
    stores nothing and leaves the caller to filter these out before the block
    is hashed.
    """
    _requires_kernel()

    from vllm.models.kimi_k3.amd.ops.kda_chunk import KDA_CHECKPOINT_ALIGNMENT

    assert offset % KDA_CHECKPOINT_ALIGNMENT != 0 or offset > 512
    inp = _inputs([512], seed=31)
    ckpt, _ = _run_with_checkpoints(inp, [offset])

    assert ckpt.isnan().all()


def test_checkpoint_state_indices_pick_the_destination_row() -> None:
    """With indices the export lands straight in the paged state cache.

    FlashKDA writes one row per sequence into a staging buffer that the caller
    then scatters. The ROCm walk holds the state in registers already, so it
    can be told which cache block each sequence owns and skip the staging
    buffer and its scatter launch entirely. A negative row opts a sequence out,
    the same way a zero offset does.
    """
    _requires_kernel()

    seqlens = [256, 256, 256]
    rows = 8
    inp = _inputs(seqlens, seed=37)
    indices = [5, -1, 2]
    ckpt, _ = _run_with_checkpoints(
        inp,
        [64, 64, 128],
        rows=rows,
        checkpoint_state_indices=torch.tensor(
            indices, device="cuda", dtype=torch.int32
        ),
    )

    written = {5: (0, 64), 2: (2, 128)}
    for row in range(rows):
        if row not in written:
            assert ckpt[row].isnan().all(), f"row {row} was written"
            continue
        seq, offset = written[row]
        _, truncated = _run(_prefix_inputs(inp, seq, offset), use_fused=True)
        torch.testing.assert_close(ckpt[row], truncated[0], rtol=0, atol=0)


@pytest.mark.parametrize("groups", [2, 4])
def test_checkpoint_survives_the_group_split(groups: int, monkeypatch) -> None:
    """The split walks each group from a composed state, not from zero.

    Only the second pass holds the true state at an interior boundary, so a
    checkpoint taken from the first pass would be the group-local state and
    silently wrong.
    """
    _requires_kernel()

    inp = _inputs([1024], seed=41)
    _force_groups(monkeypatch, 1)
    _, serial = _run(_prefix_inputs(inp, 0, 512), use_fused=True)
    _force_groups(monkeypatch, groups)
    ckpt, _ = _run_with_checkpoints(inp, [512])

    torch.testing.assert_close(ckpt[0], serial[0], rtol=1e-3, atol=1e-3)


def test_triton_fallback_refuses_a_checkpoint_export() -> None:
    """Dropping the export silently would serve an unwritten state.

    The scheduler allocates and hashes the checkpoint block on the strength of
    the layer opting in, so a backend that cannot write it has to say so rather
    than return a plausible output.
    """
    inp = _inputs([256], seed=43)
    with pytest.raises(NotImplementedError):
        _run(
            inp,
            use_fused=False,
            checkpoint_state=_checkpoint_buffer(1),
            checkpoint_offsets=torch.tensor([64], device="cuda", dtype=torch.int32),
        )
