# SPDX-License-Identifier: Apache-2.0
"""CB re-RoPE on MLA latents: declared rotation windows.

An MLA cache row is one latent plane ``[c_kv | k_pe]`` — content dims first,
rope dims trailing — not a ``[n_heads, head_size]`` K plane. Width-based
geometry inference cannot tell the two apart (GLM's 576-wide latent divides
into nine phantom 64-dim "heads" and passes every legacy check), so MLA
groups *declare* their rotation window ``(offset, width)`` at registration
and only that window rotates.

Three layers under test:

1. ``_cb_group_rope_geometry`` — the declared-window branch and its guards
   (CPU, no CUDA).
2. ``rotary_embedding_k_fused_strided`` on a trailing-window slice — the
   Python batched-rope path (`_apply_cb_rope_batched`): the slice advances
   ``data_ptr`` to the window, ``head_stride`` hops full latent rows, and the
   content dims must come back bit-identical.
3. ``execute_cb_retrieve_plan_flat`` with ``CBGroupSpec.rope_base_offset`` —
   the native-plan path: same window expressed as a byte offset off each tmp
   slot's base pointer (staging + rope only; scatter is a separate concern).
"""

# Standard
from types import SimpleNamespace

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.modules.blend_v3 import _cb_group_rope_geometry

_CONTENT, _ROPE = 24, 8  # latent = [content | rope], hidden = 32
_HIDDEN = _CONTENT + _ROPE
_DTYPE = torch.bfloat16


def _group(tokens_per_block=4):
    """Minimal kernel-group stand-in for the geometry helper."""
    return SimpleNamespace(
        tokens_per_block=tokens_per_block,
        slots_per_block=tokens_per_block,
        engine_kv_format=None,
    )


# --------------------------------------------------------------------------
# 1. Geometry rules (CPU)
# --------------------------------------------------------------------------


def test_declared_mla_window_yields_single_head():
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(), 1, 576, 64, 0, rot=(512, 64)
    )
    assert (fused, per_head, n_heads, rot_offset) == (False, 576, 1, 512)


def test_declared_window_must_end_the_row():
    with pytest.raises(RuntimeError, match="does not end the row"):
        _cb_group_rope_geometry(_group(), 1, 576, 64, 0, rot=(500, 64))


def test_declared_window_requires_single_plane():
    with pytest.raises(RuntimeError, match="kv_size=2"):
        _cb_group_rope_geometry(_group(), 2, 576, 64, 0, rot=(512, 64))


def test_legacy_inference_unchanged_and_offset_zero():
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(), 2, 8 * 64, 64, 0
    )
    assert (fused, per_head, n_heads, rot_offset) == (False, 64, 8, 0)


def test_undeclared_mla_width_still_infers_phantom_heads():
    """The hazard the declaration exists for: without ``rot``, a 576-wide
    single plane is indistinguishable from a key-only cache and infers nine
    64-dim heads. If this ever starts raising instead, the declaration
    plumbing may be removable — until then it must stay declared."""
    fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
        _group(), 1, 576, 64, 0
    )
    assert (fused, per_head, n_heads, rot_offset) == (False, 64, 9, 0)


# --------------------------------------------------------------------------
# 2/3. Kernel + native plan (GPU)
# --------------------------------------------------------------------------

_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _lmc_ops():
    lmc_ops = pytest.importorskip("lmcache.c_ops")
    if not hasattr(lmc_ops, "rotary_embedding_k_fused_strided"):
        pytest.skip("c_ops build lacks rotary_embedding_k_fused_strided")
    return lmc_ops


def _cos_sin_cache(max_pos, width, device):
    """[max_pos, width] fp32->bf16 cache, cos in the first half, sin second
    (the vLLM layout both the kernel and the blend server assume)."""
    half = width // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float64) * 2 / width))
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return (
        torch.cat([ang.cos(), ang.sin()], dim=1).to(_DTYPE).to(device),
        torch.cat([ang.cos(), ang.sin()], dim=1).to(torch.float32).to(device),
    )


def _reference_rerope(rows_f32, old_pos, new_pos, cache_f32):
    """Interleaved (GPT-J) re-rotation of the trailing window, fp32.

    Inverse-rotate each (2i, 2i+1) pair by the stored position, forward-rotate
    by the new one — the exact op the fused kernel performs in one pass.
    """
    out = rows_f32.clone()
    half = _ROPE // 2
    win = out[:, _CONTENT:]
    x, y = win[:, 0::2].clone(), win[:, 1::2].clone()
    co, so = cache_f32[old_pos, :half], cache_f32[old_pos, half:]
    cn, sn = cache_f32[new_pos, :half], cache_f32[new_pos, half:]
    xr = x * co + y * so
    yr = y * co - x * so
    win[:, 0::2] = xr * cn - yr * sn
    win[:, 1::2] = yr * cn + xr * sn
    return out


@_gpu
def test_strided_kernel_rotates_only_the_trailing_window():
    """The batched-rope path's MLA launch: slice the trailing window off a
    (tokens, 1, hidden) latent view and rotate in place. Content dims must be
    bit-identical afterwards; the window must match the fp32 reference."""
    lmc_ops = _lmc_ops()
    torch.manual_seed(0)
    device = "cuda"
    n_tok, max_pos = 64, 4096

    cache_bf16, cache_f32 = _cos_sin_cache(max_pos, _ROPE, device)
    latents = torch.randn(n_tok, 1, _HIDDEN, dtype=_DTYPE, device=device)
    before = latents.clone()
    old_pos = torch.randint(0, max_pos, (n_tok,), device=device)
    new_pos = torch.randint(0, max_pos, (n_tok,), device=device)

    ref = _reference_rerope(before[:, 0].to(torch.float32), old_pos, new_pos, cache_f32)

    lmc_ops.rotary_embedding_k_fused_strided(
        old_pos,
        new_pos,
        latents[..., _CONTENT:],  # data_ptr advances to the window start
        _ROPE,  # window width
        _HIDDEN,  # head_stride: the full latent row
        cache_bf16,
        False,  # GLM/DeepSeek MLA is interleaved (GPT-J), not NeoX
    )
    torch.cuda.synchronize()

    assert torch.equal(latents[:, 0, :_CONTENT], before[:, 0, :_CONTENT]), (
        "content dims were rotated — the exact corruption the declared "
        "window exists to prevent"
    )
    torch.testing.assert_close(
        latents[:, 0, _CONTENT:].to(torch.float32),
        ref[:, _CONTENT:],
        atol=2e-2,
        rtol=2e-2,
    )


@_gpu
def test_native_plan_rope_base_offset():
    """The native-plan path: same window as a byte offset in CBGroupSpec.
    Stage two host chunks into tmp slots and re-RoPE them (no scatters);
    the slots must match the strided-kernel result."""
    lmc_ops = _lmc_ops()
    if not hasattr(lmc_ops, "execute_cb_retrieve_plan_flat"):
        pytest.skip("c_ops build lacks execute_cb_retrieve_plan_flat")
    torch.manual_seed(1)
    device = torch.device("cuda")
    nl, spc, max_pos = 2, 8, 4096  # layers, slot tokens
    n_chunks = 2

    cache_bf16, cache_f32 = _cos_sin_cache(max_pos, _ROPE, device)
    host = [
        torch.randn(1, nl, spc, _HIDDEN, dtype=_DTYPE, device="cuda").cpu().pin_memory()
        for _ in range(n_chunks)
    ]
    slots = [
        torch.zeros(1, nl, spc, _HIDDEN, dtype=_DTYPE, device=device)
        for _ in range(n_chunks)
    ]
    old_sts, cur_sts = [128, 512], [640, 96]

    try:
        spec = lmc_ops.CBGroupSpec(
            paged_kv_ptrs=0,
            temp_buffer_ptrs=[s.data_ptr() for s in slots],
            num_layers=nl,
            slot_tokens=spc,
            hidden_elems=_HIDDEN,
            element_size=_DTYPE.itemsize,
            engine_kv_format=lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
            page_buffer_size=1,
            block_size=1,
            head_size=_ROPE,
            slot_mapping_base=0,
            slot_mapping_capacity=0,
            cos_sin_cache=cache_bf16.data_ptr(),
            rot_dim=_ROPE,
            rope_num_kv_heads=1,
            rope_head_stride=_HIDDEN,
            key_scalar_type=15,  # at::ScalarType::BFloat16
            is_neox=False,
            rope_base_offset=_CONTENT * _DTYPE.itemsize,
        )
    except TypeError:
        pytest.skip("c_ops build predates CBGroupSpec.rope_base_offset")

    chunk_bytes = nl * spc * _HIDDEN * _DTYPE.itemsize
    staging = [
        (slots[i].data_ptr(), host[i].data_ptr(), chunk_bytes, 0)
        for i in range(n_chunks)
    ]
    ropes = [(0, i, old_sts[i], cur_sts[i]) for i in range(n_chunks)]
    step_offsets = [(len(staging), len(ropes), 0)]
    lmc_ops.execute_cb_retrieve_plan_flat(
        device,
        1 << 26,
        [spec],
        np.asarray(staging, dtype=np.int64),
        np.asarray(ropes, dtype=np.int64),
        np.zeros((0, 4), dtype=np.int64),
        np.asarray(step_offsets, dtype=np.int64),
    )
    torch.cuda.synchronize()

    for i in range(n_chunks):
        # Positions ramp per slot token and repeat across layers (the ramp
        # kernel derives them in-kernel); layers share the same ramp.
        rows = host[i][0].reshape(nl * spc, _HIDDEN).to(torch.float32).to(device)
        ramp = torch.arange(spc, device=device).repeat(nl)
        ref = _reference_rerope(rows, old_sts[i] + ramp, cur_sts[i] + ramp, cache_f32)
        got = slots[i][0].reshape(nl * spc, _HIDDEN)
        assert torch.equal(
            got[:, :_CONTENT],
            host[i][0].reshape(nl * spc, _HIDDEN)[:, :_CONTENT].to(device),
        ), f"chunk {i}: content dims were rotated"
        torch.testing.assert_close(
            got[:, _CONTENT:].to(torch.float32),
            ref[:, _CONTENT:],
            atol=2e-2,
            rtol=2e-2,
        )


def test_rot_for_group_dtype_skip_under_declared_map():
    """One engine group, two kernel groups (GLM): under a declared map the
    quantized kernel group is skipped by dtype; the float one gets the
    window. Legacy maps keep today's behavior (no dtype skip)."""
    # First Party
    from lmcache.v1.multiprocess.modules.blend_v3 import _CBRopeState

    declared = _CBRopeState(
        head_size=64,
        is_neox_style=False,
        cos_sin_caches=[],
        group_to_cache=[],
        group_rot=[(512, 64)],
    )
    assert declared.rot_for_group(0, torch.bfloat16) == (512, 64)
    assert declared.rot_for_group(0, torch.uint8) is None

    legacy = _CBRopeState(
        head_size=64, is_neox_style=False, cos_sin_caches=[], group_to_cache=[]
    )
    assert legacy.rot_for_group(0, torch.uint8) == (0, 64)
