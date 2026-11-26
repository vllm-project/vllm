# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash planar KDA prefill conv (`kda_prefill_conv`: `auto`, the default,
takes it on SM10x): the kernel, its startup JIT warmup and its wiring in the KDA
layer.

Kernel. `causal_conv1d_planar` replaces `causal_conv1d_fn` on the chunked
prefill path, so its conv-state updates must be bit-identical to the stock
kernel's and its outputs within one bf16 ulp of them for every batch shape the
scheduler produces: decode rows mixed with prefills, sequences shorter than
the conv window, sequences starting inside a program's tiles, fresh and
continued sequences, null cache slots and both conv-state layouts. The outputs
are bit-identical too where Triton contracts the stock kernel's fp32
multiply-adds into fused multiply-adds in tap order (the planar kernel writes
them out explicitly), as it does on the GPUs tested; where it does not, they
differ by one ulp on a few elements per million, which the comparison allows so
that a codegen change in the stock kernel cannot fail these tests. Inputs are
built the model's way: q|k|v is the leading column slice of the merged
projection, whose row holds `3 * H * D + H + 2 * D` elements, so the row is
16-byte aligned (TMA) iff `H % 8 == 0`. The head counts below cover both
kernels at that stride (2 and 12 fall back to registers, 16 and 64 take TMA on
SM90+), next to a synthetic misaligned row.

Warmup. `planar_conv_warmup_plan` lists one layer configuration's kernels (the
layer registers each entry); compiling their keys must leave nothing for the
first prefill step to compile: a real call with that configuration's shapes
(the model's row stride and the dense row of a spec-decode step's gathered
rows, the conv-state layout, cache indices from block tables of any width) must
hit Triton's kernel cache for all three launches: the per-sequence kernels do
not specialize on the strides that vary between steps, so the plan holds three
or four compile keys.

Layer. `_forward` runs one mixed chunked-prefill step (decode rows next to a
fresh and a continued prompt) with the stock conv and with the planar conv on
cloned caches, through the Triton chunk path and, where it is available,
FlashKDA, without and with a speculative sequence in the step (the layer then
gathers the non-spec rows into a dense tensor and the conv state is wider), and
once with fp32 conv states (`--mamba-cache-dtype float32`); a real layer built
with test_kda_flashinfer.py's helpers runs a mixed step whose decode rows
FlashInfer's fused decode has already advanced (the conv sees them as null
slots), through the Triton and the FlashInfer prefill consumers; conv states
must agree bit for bit, the layer output and the recurrent state within 1e-2
(the conv outputs may differ by one bf16 ulp, see above). The dispatch guard
must send a short step to the stock kernel, and the option resolver must
resolve `auto` by device family and reject what it should.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.models.glm5next.nvidia.ops import kda_prefill_conv
from vllm.models.glm5next.nvidia.ops.kda_prefill_conv import (
    PLANAR_MIN_PREFILL_TOKENS,
    causal_conv1d_planar,
    planar_conv_impl,
    planar_conv_pays_off,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128
WIDTH = 4


# --- The kernel: causal_conv1d_planar against causal_conv1d_fn ---
# (sequence lengths, has_initial_state, cache slots); slot NULL_BLOCK_ID marks
# a padded sequence, which causal_conv1d_fn skips; PAD_SLOT_ID only appears on
# empty sequences.
CASES = {
    "prefill_with_decode_rows": (
        [1] * 5 + [700, 1000, 33],
        [True] * 5 + [True, False, True],
        list(range(1, 9)),
    ),
    "short_and_null": (
        [2, 1, 3, 5, 0, 31, 64, 129],
        [True, True, True, False, True, True, False, True],
        [1, 2, NULL_BLOCK_ID, 3, PAD_SLOT_ID, 5, 6, 7],
    ),
    "starts_inside_tiles": (
        [1] * 9 + [301, 129, 2, 700, 1, 5, 300, 15, 17, 4],
        [True] * 9 + [True, False, True, True, False, True, False, True, True, True],
        list(range(1, 20)),
    ),
    # Fewer tokens than one tile, every sequence continued: the fixup alone
    # decides the output.
    "all_short": ([1, 2, 5], [True, True, True], [1, 2, 3]),
    # Continued short sequences at the end of the token range, in the last
    # partial tile after a longer prompt.
    "trailing_short": ([129, 2, 1], [False, True, True], [1, 2, 3]),
}
# A full chunked-prefill step: 194 decode rows, then eight 8K prompts, half of
# them continued (has_initial_state) and half fresh; about 65K tokens.
LARGE_CASE = (
    [1] * 194 + [8192] * 8,
    [True] * 194 + [True, False] * 4,
    list(range(1, 203)),
)


def model_row_pad(num_heads: int) -> int:
    """Columns after q|k|v in the merged projection (beta and the two gates)."""
    return num_heads + 2 * D


def _assert_within_one_ulp(out: torch.Tensor, ref: torch.Tensor) -> None:
    """`out` equals `ref` or lies one ulp of `ref`'s dtype away from it."""
    assert out.shape == ref.shape and out.dtype == ref.dtype
    differ = out != ref
    if not bool(differ.any()):
        return
    o, r = out[differ], ref[differ]
    up = torch.nextafter(r, torch.full_like(r, torch.inf)).float() - r.float()
    down = r.float() - torch.nextafter(r, torch.full_like(r, -torch.inf)).float()
    # One ulp in the direction of the difference (the two differ at powers of
    # two).
    distance = (o.float() - r.float()).abs() / torch.where(o < r, down, up)
    assert bool((distance <= 1).all()), (
        f"{int(differ.sum())} of {ref.numel()} outputs differ from "
        f"causal_conv1d_fn, up to {float(distance.max()):.1f} ulp"
    )


def _run(
    case,
    num_heads: int,
    dim_first: bool,
    row_pad: int,
    impl: str,
    *,
    bias: bool = False,
    weight_dtype: torch.dtype = torch.float32,
    state_dtype: torch.dtype = torch.bfloat16,
    num_planes: int = 3,
    activation: str | None = "silu",
    **launch,
) -> torch.Tensor:
    """Compare the planar kernel with causal_conv1d_fn on one case; returns x.

    `launch` overrides the kernel's block_n, tiles, num_warps or num_stages.
    """
    lens, has_init, slots = CASES[case] if isinstance(case, str) else case
    set_random_seed(0)
    device = "cuda"
    dim = 3 * num_heads * D
    num_tokens = sum(lens)
    # q|k|v is a row-strided column slice of the merged projection.
    x = torch.randn(num_tokens, dim + row_pad, device=device, dtype=torch.bfloat16)
    x = x[:, :dim]
    weight = (torch.randn(dim, WIDTH, device=device) * 0.5).to(weight_dtype)
    bias_t = (torch.randn(dim, device=device) * 0.1).to(weight_dtype) if bias else None
    num_slots = max(slots) + 2
    if dim_first:
        pool = torch.randn(num_slots, dim, WIDTH - 1, device=device, dtype=state_dtype)
    else:
        pool = torch.randn(num_slots, WIDTH - 1, dim, device=device, dtype=state_dtype)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    # Strided like column 0 of a block table.
    cache_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    cache_indices = torch.stack([cache_indices, cache_indices + 1000], 1)[:, 0]
    has_initial_state = torch.tensor(has_init, dtype=torch.bool, device=device)

    def states(p):
        return p if dim_first else p.transpose(-1, -2)

    def stock(p, cache_indices, has_initial_state):
        return causal_conv1d_fn(
            x.transpose(0, 1),
            weight,
            bias_t,
            conv_states=states(p),
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
        ).transpose(0, 1)

    ref_pool, pool = pool.clone(), pool.clone()
    ref = stock(ref_pool, cache_indices, has_initial_state)
    out = causal_conv1d_planar(
        x,
        weight,
        bias_t,
        states(pool),
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        num_planes=num_planes,
        activation=activation,
        impl=impl,
        **launch,
    )
    assert out.shape == (num_planes, num_tokens, dim // num_planes)
    assert all(plane.is_contiguous() for plane in out.unbind(0))
    out = torch.cat(out.unbind(0), dim=1)

    # causal_conv1d_fn leaves rows of null-slot sequences unwritten.
    valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    starts = query_start_loc.tolist()
    for s, slot in enumerate(slots):
        if slot == NULL_BLOCK_ID:
            valid[starts[s] : starts[s + 1]] = False
    _assert_within_one_ulp(out[valid], ref[valid])
    torch.testing.assert_close(pool, ref_pool, atol=0, rtol=0)

    if not bool(valid.all()):
        # The planar kernel writes those rows as if the sequence had no
        # initial state: the stock kernel's output for the same sequences on
        # scratch slots without state is the reference for them.
        scratch_slots = [
            num_slots + s if slot == NULL_BLOCK_ID else slot
            for s, slot in enumerate(slots)
        ]
        scratch_pool = torch.cat(
            [ref_pool, torch.zeros_like(ref_pool[:1]).expand(len(slots), -1, -1)]
        )
        ref_null = stock(
            scratch_pool,
            torch.tensor(scratch_slots, dtype=torch.int32, device=device),
            torch.tensor(
                [h and slot != NULL_BLOCK_ID for h, slot in zip(has_init, slots)],
                device=device,
            ),
        )
        _assert_within_one_ulp(out[~valid], ref_null[~valid])
    return x


def _expected_impl(impl: str, num_heads: int) -> str:
    tma = current_platform.has_device_capability(90) and num_heads % 8 == 0
    return "tma" if impl == "auto" and tma else "regs"


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize(
    ("num_heads", "impl"),
    # 2 and 12 heads: the projection's row is not 16-byte aligned, so "auto"
    # resolves to the register kernel (asserted below); 16 and 64 heads run
    # both kernels.
    [(2, "auto"), (12, "auto"), (16, "regs"), (16, "auto"), (64, "regs"), (64, "auto")],
    ids=["h2", "h12", "h16-regs", "h16-auto", "h64-regs", "h64-auto"],
)
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
@torch.inference_mode()
def test_planar_conv_matches_causal_conv1d(
    case: str, num_heads: int, dim_first: bool, impl: str
):
    x = _run(case, num_heads, dim_first, model_row_pad(num_heads), impl)
    assert planar_conv_impl(impl, x) == _expected_impl(impl, num_heads)


@pytest.mark.parametrize(
    ("bias", "weight_dtype"),
    [(True, torch.float32), (False, torch.bfloat16), (True, torch.bfloat16)],
    ids=["bias", "bf16", "bias-bf16"],
)
@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_bias_and_bf16_weights(bias, weight_dtype, impl):
    # Dead for GLM-5.3-Flash (fp32 weights, no bias), alive for a generic op.
    x = _run(
        "starts_inside_tiles",
        16,
        True,
        model_row_pad(16),
        impl,
        bias=bias,
        weight_dtype=weight_dtype,
    )
    assert planar_conv_impl(impl, x) == _expected_impl(impl, 16)


@pytest.mark.parametrize(
    ("block_n", "tiles", "num_stages"),
    [(256, 4, 2), (128, 1, 4)],
    ids=["b256-t4-s2", "b128-t1-s4"],
)
@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_launch_knobs(block_n, tiles, num_stages, impl):
    # Off-default geometry: narrower channel blocks, shorter tile runs (down
    # to one tile per program, so every tile's history comes from another
    # program) and a pipeline deeper than the tile run.
    x = _run(
        "starts_inside_tiles",
        16,
        True,
        model_row_pad(16),
        impl,
        block_n=block_n,
        tiles=tiles,
        num_stages=num_stages,
    )
    assert planar_conv_impl(impl, x) == _expected_impl(impl, 16)


@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_single_plane_without_activation(impl):
    x = _run(
        "short_and_null",
        16,
        False,
        model_row_pad(16),
        impl,
        num_planes=1,
        activation=None,
    )
    assert planar_conv_impl(impl, x, num_planes=1) == _expected_impl(impl, 16)


@torch.inference_mode()
def test_planar_conv_empty_batch():
    # A step without tokens (the early return) leaves the states alone.
    dim = 3 * 16 * D
    x = torch.empty(0, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, WIDTH, device="cuda")
    pool = torch.randn(3, dim, WIDTH - 1, device="cuda", dtype=torch.bfloat16)
    before = pool.clone()
    out = causal_conv1d_planar(
        x,
        weight,
        None,
        pool,
        query_start_loc=torch.zeros(2, dtype=torch.int32, device="cuda"),
        cache_indices=torch.tensor([PAD_SLOT_ID], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([False], device="cuda"),
    )
    assert out.shape == (3, 0, dim // 3)
    assert torch.equal(pool, before)


@pytest.mark.parametrize("impl", ["regs", "auto"])
@torch.inference_mode()
def test_planar_conv_full_chunked_prefill_step(impl):
    # About 65K tokens at the TP8 shard: the full grid and the TMA coordinates
    # at the serving size.
    x = _run(LARGE_CASE, 8, True, model_row_pad(8), impl)
    assert planar_conv_impl(impl, x) == _expected_impl(impl, 8)


def test_planar_conv_pays_off_rule():
    # The dispatch guard: prefill tokens > intercept(heads) + 6 x decode rows.
    for heads, intercept in PLANAR_MIN_PREFILL_TOKENS.items():
        assert planar_conv_pays_off(intercept + 1, 0, heads)
        assert not planar_conv_pays_off(intercept, 0, heads)
        assert planar_conv_pays_off(intercept + 6 * 500 + 1, 500, heads)
        assert not planar_conv_pays_off(intercept + 6 * 500, 500, heads)
    # Measured brackets at 64 heads: two 4K prompts next to 500 decode rows win
    # (1.4x), one 1K prompt next to 1000 rows loses (0.6x).
    assert planar_conv_pays_off(8192, 500, 64)
    assert not planar_conv_pays_off(1024, 1000, 64)
    # Head counts between table entries take the next smaller entry; below
    # the table (unmeasured) the bar doubles.
    assert planar_conv_pays_off(2049, 0, 24) and not planar_conv_pays_off(2048, 0, 24)
    assert planar_conv_pays_off(8193, 0, 4) and not planar_conv_pays_off(8192, 0, 4)


@torch.inference_mode()
def test_planar_conv_unaligned_rows_fall_back():
    # Rows that are not 16-byte aligned cannot use TMA descriptors: impl="auto"
    # must take the register kernel with the same results, and impl="tma" must
    # say so instead of silently doing the same.
    x = _run("starts_inside_tiles", 2, True, 3, "auto")
    assert planar_conv_impl("auto", x) == "regs"
    with pytest.raises(ValueError, match="aligned"):
        planar_conv_impl("tma", x)


@torch.inference_mode()
def test_planar_conv_narrow_blocks_fall_back():
    # TMA boxes need channel blocks of at least 16 bytes: four bf16 channels
    # per program take the register kernel under impl="auto" (same results)
    # and are refused under impl="tma".
    x = _run("starts_inside_tiles", 16, True, model_row_pad(16), "auto", block_n=4)
    assert planar_conv_impl("auto", x, block_n=4) == "regs"
    with pytest.raises(ValueError, match="16 bytes"):
        planar_conv_impl("tma", x, block_n=4)


@pytest.mark.parametrize("dim_first", [True, False])
@torch.inference_mode()
def test_planar_conv_fp32_states(dim_first):
    # fp32 weights with fp32 conv states (the pairing of a model that keeps its
    # conv states in fp32): causal_conv1d_fn widens x to fp32 exactly and forms
    # the same fp32 fused multiply-adds, so the states still match bit for bit
    # and the outputs within one ulp (identical on the GPU tested).
    _run(
        "short_and_null",
        16,
        dim_first,
        model_row_pad(16),
        "auto",
        state_dtype=torch.float32,
    )


@torch.inference_mode()
def test_planar_conv_rejects_bad_inputs():
    # Contract violations raise with a message instead of failing inside a
    # kernel (or, with a short out=, writing past its end).
    device = "cuda"
    dim, num_tokens = 3 * 2 * D, 8
    x = torch.randn(num_tokens, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, WIDTH, device=device)
    pool = torch.zeros(3, dim, WIDTH - 1, device=device, dtype=torch.bfloat16)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([1], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([False], device=device)

    def call(**overrides):
        kwargs = dict(
            x=x,
            weight=weight,
            bias=None,
            conv_states=pool,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
        )
        kwargs.update(overrides)
        return causal_conv1d_planar(**kwargs)

    call()  # the valid baseline
    with pytest.raises(ValueError, match="width 4"):
        call(weight=weight[:, :3])
    with pytest.raises(ValueError, match="channels"):
        call(weight=weight[: dim - 1])
    with pytest.raises(ValueError, match="bias"):
        call(bias=torch.randn(dim - 1, device=device))
    with pytest.raises(ValueError, match="dense channels"):
        call(x=x.transpose(0, 1))
    with pytest.raises(ValueError, match="num_planes"):
        call(num_planes=5)
    with pytest.raises(ValueError, match="conv_states"):
        call(conv_states=pool[:, :, :1])
    with pytest.raises(ValueError, match="activation"):
        call(activation="gelu")
    with pytest.raises(ValueError, match="impl"):
        call(impl="TMA")
    with pytest.raises(ValueError, match="power of two"):
        call(block_n=48)
    with pytest.raises(ValueError, match="tiles"):
        call(tiles=0)
    with pytest.raises(ValueError, match="num_warps"):
        call(num_warps=3)
    with pytest.raises(ValueError, match="entry per sequence"):
        call(cache_indices=cache_indices[:0])
    with pytest.raises(ValueError, match="entry per sequence"):
        call(has_initial_state=has_initial_state[:0])
    with pytest.raises(ValueError, match="contiguous 1D integer"):
        call(query_start_loc=query_start_loc[None, :])
    with pytest.raises(ValueError, match="contiguous 1D integer"):
        call(query_start_loc=torch.stack([query_start_loc] * 2, 1)[:, 0])
    with pytest.raises(ValueError, match="contiguous 1D integer"):
        call(query_start_loc=query_start_loc.float())
    with pytest.raises(ValueError, match="must be 1D"):
        call(cache_indices=cache_indices[None, :])
    with pytest.raises(ValueError, match="integer tensor"):
        call(cache_indices=cache_indices.float())
    with pytest.raises(TypeError, match="cache_indices must be on"):
        call(cache_indices=cache_indices.cpu())
    with pytest.raises(TypeError, match="conv_states must be on"):
        call(conv_states=pool.cpu())
    short = torch.empty(3, num_tokens - 1, dim // 3, device=device, dtype=x.dtype)
    with pytest.raises(ValueError, match="out must be"):
        call(out=short)
    wrong_dtype = torch.empty(3, num_tokens, dim // 3, device=device)
    with pytest.raises(TypeError, match="dtype"):
        call(out=wrong_dtype)
    strided = torch.empty(3, num_tokens, 2 * (dim // 3), device=device, dtype=x.dtype)
    with pytest.raises(ValueError, match="dense last dim"):
        call(out=strided[:, :, ::2])
    with pytest.raises(TypeError, match="device"):
        call(out=torch.empty(3, num_tokens, dim // 3, dtype=x.dtype))
    # Conv states in another dtype than x match causal_conv1d_fn only when both
    # the weights and the states are fp32 (x widened exactly, fp32 FMAs on both
    # sides).
    call(conv_states=pool.float())
    with pytest.raises(TypeError, match="conv_states dtype"):
        call(conv_states=pool.half())
    with pytest.raises(TypeError, match="conv_states dtype"):
        call(weight=weight.to(torch.bfloat16), conv_states=pool.float())


# --- Startup JIT warmup: the plan covers every runtime variant ---


def _kernel_caches(dispatchers) -> list[dict]:
    """Triton's compiled-variant caches of the kernels for the current device."""
    device = triton.runtime.driver.active.get_current_device()
    return [d.kernel.device_caches[device][0] for d in dispatchers]


def _compiled(dispatchers) -> list[int]:
    return [len(c) for c in _kernel_caches(dispatchers)]


def _probe_step(
    dim: int, row: int, dim_first: bool, cache_indices: torch.Tensor, state_len
):
    lens = [1, 1, 300, 40]
    x = torch.randn(sum(lens), row, device="cuda", dtype=torch.bfloat16)[:, :dim]
    weight = torch.randn(dim, 4, device="cuda")
    num_slots = int(cache_indices.max()) + 1  # every index must own a slot
    shape = (num_slots, dim, state_len) if dim_first else (num_slots, state_len, dim)
    pool = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    return causal_conv1d_planar(
        x,
        weight,
        None,
        pool if dim_first else pool.transpose(-1, -2),
        query_start_loc=torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        ),
        cache_indices=cache_indices,
        has_initial_state=torch.tensor([True, True, False, True], device="cuda"),
    )


# 4 heads: the fused projection's row is not 16-byte aligned, so the register
# kernel; 24 heads: TMA. The probe below starts from cleared kernel caches, so
# the test's position in the run does not matter.
@pytest.mark.parametrize("num_heads", [4, 24])
@pytest.mark.parametrize("dim_first", [True, False], ids=["DS", "SD"])
# 3 state positions (no speculative tokens) and 16 (13 speculative tokens):
# the two Triton stride classes of the conv-state pool in the DS layout.
@pytest.mark.parametrize("state_len", [3, 16])
@torch.inference_mode()
def test_planar_conv_warmup_covers_the_runtime_variants(
    num_heads, dim_first, state_len
):
    dim = 3 * num_heads * D
    row = dim + num_heads + 2 * D  # the fused projection's row
    plan = kda_prefill_conv.planar_conv_warmup_plan(
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        bias_dtype=None,
        dim=dim,
        num_planes=3,
        # The fused projection's row and the dense row of the gathered
        # non-spec rows of a spec-decode step.
        row_strides=(row, dim),
        conv_state_dim_first=dim_first,
        state_len=state_len,
    )
    # The dense row is 16-byte aligned at every head count, the projection's
    # row only at multiples of 8 heads.
    dense = "tma" if current_platform.has_device_capability(90) else "regs"
    strided = dense if num_heads % 8 == 0 else "regs"

    def stencil(kind):
        return (
            kda_prefill_conv._planar_stencil_tma
            if kind == "tma"
            else kda_prefill_conv._planar_stencil_regs
        )

    # One stencil case per row stride, then the fixup and the state update
    # once: three or four compile keys per layer.
    assert [d for d, _ in plan] == [
        stencil(strided),
        stencil(dense),
        kda_prefill_conv._planar_fixup,
        kda_prefill_conv._planar_state_update,
    ]
    dispatchers = list(
        dict.fromkeys(
            [
                stencil(strided),
                stencil(dense),
                kda_prefill_conv._planar_fixup,
                kda_prefill_conv._planar_state_update,
            ]
        )
    )
    # Start from empty kernel caches so the probe below sees this case's
    # compiles only (another case or test may have compiled the same variants;
    # the disk cache keeps the recompile cheap).
    for cache in _kernel_caches(dispatchers):
        cache.clear()
    for dispatcher, kwargs in plan:
        keys = dispatcher.get_warmup_keys(**kwargs)
        assert keys, (dispatcher.__name__, kwargs)
        dispatcher.compile_many(keys)
    before = _compiled(dispatchers)
    assert all(before)
    # Cache indices as column 0 of block tables of three widths (strides 16, 7
    # and 1): the per-sequence kernels do not specialize on that stride, so
    # none of them may compile anything.
    for width in (16, 7, 1):
        table = torch.arange(1, 1 + 4 * width, dtype=torch.int32, device="cuda")
        out = _probe_step(dim, row, dim_first, table.view(4, width)[:, 0], state_len)
        torch.accelerator.synchronize()
        assert out.shape == (3, 342, dim // 3)
        assert _compiled(dispatchers) == before, (
            f"the call with cache-index stride {width} compiled a variant the "
            "warmup did not cover"
        )
    # The dense rows of a spec-decode step.
    table = torch.arange(1, 5, dtype=torch.int32, device="cuda")
    out = _probe_step(dim, dim, dim_first, table, state_len)
    torch.accelerator.synchronize()
    assert out.shape == (3, 342, dim // 3)
    assert _compiled(dispatchers) == before, (
        "the dense-row call compiled a variant the warmup did not cover"
    )
    # Positive control for the probe: int64 cache indices are a new variant
    # of the two per-sequence kernels (the stencil never sees them).
    table = torch.arange(1, 5, dtype=torch.int64, device="cuda")
    _probe_step(dim, row, dim_first, table, state_len)
    torch.accelerator.synchronize()
    after = _compiled(dispatchers)
    assert after[:-2] == before[:-2]
    assert after[-2:] == [n + 1 for n in before[-2:]]


# --- The layer: _forward with both convs, the dispatch guard, the resolver ---

HEADS = 2
LOWER_BOUND = -5.0
# Decode rows first, then one fresh and one continued prompt. 8.3K prefill
# tokens clear the dispatch guard's bar at 2 heads (8192 + 6 per decode row).
LONG_STEP = ([1, 1, 1, 5000, 3300], [True, True, True, False, True], [1, 2, 3, 4, 5])
SHORT_STEP = ([1, 1, 300, 200], [True, True, False, True], [1, 2, 3, 4])
NUM_SLOTS = 6  # the non-spec slots; a speculative sequence's slots follow


def _hand_built_layer(kda, backend: str, dim_first: bool, device, monkeypatch):
    """A layer assembled without __init__, as test_kda_recurrent.py does."""
    from vllm.model_executor.layers.mamba.kda_checkpoint import (
        FlashKDAPrefillCheckpointExporter,
    )
    from vllm.v1.worker.workspace import WorkspaceManager

    width = 3 * HEADS * D
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = HEADS
    layer.head_dim = D
    layer.conv_size = 4
    layer.local_projection_size = HEADS * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = backend
    # The decode rows of these steps take the Triton kernels, so _forward
    # reaches the conv with every row.
    layer.kda_decode_backend = "triton"
    layer._conv_state_dim_first = dim_first
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(width, 1, 4, device=device) * 0.1, bias=None
    )
    layer.A_log = torch.randn(1, 1, HEADS, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(HEADS * D, device=device) * 0.1
    layer._flashkda_buffer_specs = None
    # As __init__ sets them: the checkpoint exporter for both consumers and the
    # checkpoint buffer spec for the Triton one. The steps below carry no
    # checkpoint, so _forward reads neither.
    layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
    layer._triton_checkpoint_spec = (
        ((NUM_SLOTS, HEADS, D, D), torch.float32) if backend == "triton" else None
    )
    if backend == "flashkda":
        monkeypatch.setattr(
            kda, "current_workspace_manager", lambda: WorkspaceManager(device)
        )
    return layer


def _flashkda_buffer_specs(step):
    """The layer's FlashKDA workspace buffers, sized for one step as __init__
    would size them for the whole batch."""
    num_tokens, num_seqs = sum(step[0]), len(step[0])
    state_shape = (num_seqs, HEADS, D, D)
    return (
        (state_shape, torch.float32),
        (state_shape, torch.float32),
        (
            (torch.ops._flashkda_C.get_workspace_size(num_tokens, HEADS, num_seqs),),
            torch.uint8,
        ),
        ((1, num_tokens, HEADS, D), torch.bfloat16),
    )


def _metadata(step, device):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    lens, has_init, slots = step
    cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).int()
    nums, batch_ptr, chunk_offsets = compute_causal_conv1d_metadata(
        cu_cpu, device=device
    )
    num_decodes = sum(n == 1 for n in lens)
    return GDNAttentionMetadata(
        num_prefills=len(lens) - num_decodes,
        num_prefill_tokens=sum(lens) - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=sum(lens),
        has_initial_state=torch.tensor(has_init, device=device),
        non_spec_query_start_loc=cu_cpu.to(device),
        non_spec_state_indices_tensor=torch.tensor(
            slots, dtype=torch.int32, device=device
        ),
        nums_dict=nums,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=chunk_offsets,
    )


def _add_spec_sequence(metadata, num_spec: int, num_seqs: int, device):
    """One speculative sequence of `num_spec + 1` rows ahead of the step's
    non-spec rows (as test_kda_recurrent.py sets it up), on the slots after
    the non-spec ones; returns its row count. The step's plain decode rows
    become prefills, as the metadata builder counts them once speculative
    rows are present."""
    spec_tokens = num_spec + 1
    metadata.num_spec_decodes = 1
    metadata.num_spec_decode_tokens = spec_tokens
    metadata.num_actual_tokens += spec_tokens
    metadata.num_prefills += metadata.num_decodes
    metadata.num_prefill_tokens += metadata.num_decode_tokens
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.spec_sequence_masks = torch.tensor(
        [True] + [False] * num_seqs, device=device
    )
    metadata.spec_query_start_loc = torch.tensor(
        [0, spec_tokens], dtype=torch.int32, device=device
    )
    metadata.spec_state_indices_tensor = torch.arange(
        NUM_SLOTS, NUM_SLOTS + spec_tokens, dtype=torch.int32, device=device
    ).unsqueeze(0)
    metadata.spec_token_indx = torch.arange(spec_tokens, device=device)
    metadata.non_spec_token_indx = torch.arange(
        spec_tokens, metadata.num_actual_tokens, device=device
    )
    metadata.num_accepted_tokens = torch.ones(1, dtype=torch.int32, device=device)
    return spec_tokens


def _run_step(
    kda, layer, step, conv, recurrent, conv_option, monkeypatch, device, num_spec=0
):
    """One _forward on clones of the caches; returns (out, conv, recurrent)."""
    lens = step[0]
    width = 3 * HEADS * D
    metadata = _metadata(step, device)
    spec_tokens = (
        _add_spec_sequence(metadata, num_spec, len(lens), device) if num_spec else 0
    )
    num_tokens = sum(lens) + spec_tokens
    torch.manual_seed(1)
    # q|k|v is the leading, row-strided slice of the fused projection.
    projected = torch.randn(
        num_tokens, width + HEADS + 2 * D, device=device, dtype=torch.bfloat16
    )
    qkv, beta = projected[:, :width], projected[:, width : width + HEADS].unsqueeze(0)
    gate = torch.randn(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    # The output gate; _forward reads it only on the FlashInfer decode paths.
    out_gate = torch.randn(num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    conv, recurrent = conv.clone(), recurrent.clone()
    layer.kv_cache = (conv, recurrent)
    layer.kda_prefill_conv = conv_option
    if layer.kda_prefill_backend == "flashkda":
        layer._flashkda_buffer_specs = _flashkda_buffer_specs(step)
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )
    out = torch.empty(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    layer._forward(qkv, gate, beta, out, out_gate)
    return out, conv, recurrent


def _check_planar_matches_stock_in_layer(
    monkeypatch, backend, dim_first, num_spec, conv_dtype
):
    from vllm.models.glm5next.common import kda

    if backend == "flashkda":
        if (
            kda._resolve_kda_prefill_backend("auto", D, torch.bfloat16, LOWER_BOUND)
            != "flashkda"
        ):
            pytest.skip("requires FlashKDA")
        import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(42)
    device = torch.device("cuda")
    layer = _hand_built_layer(kda, backend, dim_first, device, monkeypatch)
    width, state_len = 3 * HEADS * D, 3 + num_spec
    num_slots = NUM_SLOTS + (num_spec + 1 if num_spec else 0)
    conv_shape = (width, state_len) if dim_first else (state_len, width)
    conv = torch.randn(num_slots, *conv_shape, device=device, dtype=conv_dtype)
    recurrent = torch.randn(num_slots, HEADS, D, D, device=device) * 0.1

    planar_calls = []
    planar = kda.causal_conv1d_planar

    def counting_planar(*args, **kwargs):
        planar_calls.append(1)
        return planar(*args, **kwargs)

    monkeypatch.setattr(kda, "causal_conv1d_planar", counting_planar)

    args = (kda, layer, LONG_STEP, conv, recurrent)
    out_stock, conv_stock, rec_stock = _run_step(
        *args, "causal_conv1d", monkeypatch, device, num_spec
    )
    assert not planar_calls
    out_planar, conv_planar, rec_planar = _run_step(
        *args, "planar", monkeypatch, device, num_spec
    )
    assert len(planar_calls) == 1
    # Conv states match bit for bit; the conv outputs may differ from the
    # stock kernel's by one bf16 ulp on a few elements (see the kernel tests),
    # which the layer output and the recurrent state carry as differences far
    # below these tolerances, where a wiring error shows as large ones.
    assert torch.equal(conv_planar, conv_stock)
    torch.testing.assert_close(out_planar, out_stock, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rec_planar, rec_stock, atol=1e-2, rtol=1e-2)

    # Short step: the dispatch guard keeps the stock kernel.
    args = (kda, layer, SHORT_STEP, conv, recurrent)
    out_stock, conv_stock, _ = _run_step(
        *args, "causal_conv1d", monkeypatch, device, num_spec
    )
    out_guarded, conv_guarded, _ = _run_step(
        *args, "planar", monkeypatch, device, num_spec
    )
    assert len(planar_calls) == 1
    assert torch.equal(conv_guarded, conv_stock)
    assert torch.equal(out_guarded, out_stock)


@pytest.mark.parametrize("backend", ["triton", "flashkda"])
@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
# 0: the projection's row-strided q|k|v reaches the conv; 3: a speculative
# sequence shares the step, the layer gathers the non-spec rows into a dense
# tensor and the conv state has num_spec more positions.
@pytest.mark.parametrize("num_spec", [0, 3])
@torch.inference_mode()
def test_planar_prefill_conv_matches_stock_in_layer(
    monkeypatch, backend, dim_first, num_spec
):
    _check_planar_matches_stock_in_layer(
        monkeypatch, backend, dim_first, num_spec, torch.bfloat16
    )


@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
@torch.inference_mode()
def test_planar_prefill_conv_fp32_states_in_layer(monkeypatch, dim_first):
    # --mamba-cache-dtype float32: fp32 conv states under the layer's fp32 conv
    # weights, the other pairing the kernel matches (states bit for bit, outputs
    # within one ulp; the stock
    # kernel then runs on a copy of x widened to fp32), in a speculative step
    # with its wider state.
    _check_planar_matches_stock_in_layer(
        monkeypatch, "triton", dim_first, 3, torch.float32
    )


def test_resolve_kda_prefill_conv(monkeypatch):
    from vllm.models.glm5next.common import kda

    bf16 = torch.bfloat16
    assert kda._resolve_kda_prefill_conv("causal_conv1d", 4, bf16, bf16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("planar", 4, bf16, bf16) == "planar"
    with pytest.raises(ValueError, match="causal_conv1d"):
        kda._resolve_kda_prefill_conv("fused", 4, bf16, bf16)
    with pytest.raises(ValueError, match="width 4"):
        kda._resolve_kda_prefill_conv("planar", 3, bf16, bf16)
    # The stock kernel casts x to the conv-state dtype before multiplying; with
    # the layer's fp32 conv weights, bf16 activations match it (states exactly,
    # outputs within one ulp) with bf16 and with fp32 conv states, and the layer
    # accepts those two pairings.
    assert kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float32) == "planar"
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float16)
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, torch.float16, torch.float16)
    assert (
        kda._resolve_kda_prefill_conv("causal_conv1d", 4, torch.bfloat16, torch.float32)
        == "causal_conv1d"
    )
    # The fp32-state pairing relies on fp32 conv weights (the layer's); with
    # bf16 weights it is refused like any other mixed pairing.
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float32, bf16)
    # "auto" (the default) takes the planar kernel on SM10x devices only, for
    # the two pairings above, and never raises; "planar" stays available on
    # other CUDA devices. The device capability is patched at class level so
    # the real family check runs.
    platform_cls = type(kda.current_platform)

    def set_capability(major, minor):
        monkeypatch.setattr(
            platform_cls,
            "get_device_capability",
            classmethod(lambda cls, device_id=0: DeviceCapability(major, minor)),
        )

    set_capability(10, 3)
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, bf16) == "planar"
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float32) == "planar"
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float32, bf16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("auto", 3, bf16, bf16) == "causal_conv1d"
    for major, minor in ((9, 0), (12, 0)):
        set_capability(major, minor)
        assert kda._resolve_kda_prefill_conv("auto", 4, bf16, bf16) == ("causal_conv1d")
        assert kda._resolve_kda_prefill_conv("planar", 4, bf16, bf16) == "planar"


@pytest.fixture(scope="module")
def glm5next_model_dir(tmp_path_factory) -> str:
    """A one-layer GLM-5.3-Flash config, as test_kda_flashinfer.py builds it."""
    from transformers import Glm5NextTextConfig

    from tests.models.glm5next.test_kda_flashinfer import HIDDEN

    path = tmp_path_factory.mktemp("glm5next")
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN, num_hidden_layers=1, layer_types=["linear_attention"]
    )
    config.architectures = ["Glm5NextForCausalLM"]
    config.save_pretrained(path)
    return str(path)


# The prefill consumers of the planar output: the Triton chunk path and
# FlashInfer's CuTe DSL prefill, which skips where unsupported.
@pytest.mark.parametrize("prefill", ["triton", "flashinfer"])
@pytest.mark.usefixtures("dist_init")
@torch.inference_mode()
def test_planar_prefill_conv_with_flashinfer_decode(
    glm5next_model_dir, workspace_init, monkeypatch, prefill
):
    """A mixed step whose decode rows go through FlashInfer's fused decode
    (conv, recurrence and norm in one launch): `_forward` then hands the conv
    the decode rows as null slots, which neither kernel may touch, and the
    planar kernel must match the stock kernel on the prefill rows, whichever
    prefill backend consumes them. Built with test_kda_flashinfer.py's
    real-layer helpers."""
    from tests.models.glm5next.test_kda_flashinfer import (
        _forward,
        _hidden,
        _make_layer,
        _non_spec_metadata,
        _require_flashinfer_decode,
        _require_flashinfer_prefill,
    )
    from vllm.models.glm5next.common import kda

    heads = 16
    if prefill == "flashinfer":
        _require_flashinfer_prefill(torch.float32)
    # "auto" resolves to the fused decode on SM10x with FlashInfer's kernel
    # and the test skips elsewhere; an explicit "flashinfer" would raise there.
    stock = _make_layer(glm5next_model_dir, heads, decode="auto", prefill=prefill)
    _require_flashinfer_decode(stock[0])
    planar = _make_layer(glm5next_model_dir, heads, decode="auto", prefill=prefill)
    assert planar[0].kda_decode_backend == "flashinfer"
    stock[0].kda_prefill_conv = "causal_conv1d"
    planar[0].kda_prefill_conv = "planar"
    planar_calls = []
    real_planar = kda.causal_conv1d_planar

    def counting_planar(*args, **kwargs):
        planar_calls.append(1)
        return real_planar(*args, **kwargs)

    monkeypatch.setattr(kda, "causal_conv1d_planar", counting_planar)
    # Decode rows first, then a fresh prompt, a one-token prompt and a
    # continued prompt; 2.2K prefill tokens clear the guard at 16 heads
    # (2048 + 6 per decode row).
    decode_slots = [5, 2, 20, 9]
    n = len(decode_slots)
    query_lens = [1] * n + [1500, 1, 700]
    slots = decode_slots + [12, 14, 17]
    meta = _non_spec_metadata(query_lens, n, slots, [True] * n + [False, True, True])
    hidden = _hidden(sum(query_lens))
    out_stock = _forward(*stock, meta, hidden)
    assert not planar_calls
    out_planar = _forward(*planar, meta, hidden)
    assert len(planar_calls) == 1
    (conv_stock, rec_stock), (conv_planar, rec_planar) = (
        stock[0].kv_cache,
        planar[0].kv_cache,
    )
    # Decode slots were advanced by the fused decode in both layers, prefill
    # slots by the two convs: all of them must agree exactly.
    torch.testing.assert_close(conv_planar, conv_stock, atol=0, rtol=0)
    torch.testing.assert_close(rec_planar, rec_stock, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out_planar, out_stock, atol=1e-2, rtol=1e-2)
