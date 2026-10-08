# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionGemmaSelfConditioning and batched _apply_self_conditioning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import vllm.model_executor.models.diffusion_gemma as dg_mod
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionGemmaSelfConditioning,
    _gelu_tanh_and_mul_kernel,
    _sc_add_rmsnorm_kernel,
    _sc_rmsnorm_kernel,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.triton_utils import triton

requires_cuda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and dg_mod.HAS_TRITON
        and dg_mod.current_platform.is_cuda_alike()
    ),
    reason="CUDA + Triton required for Triton SC tests",
)
pytestmark = pytest.mark.usefixtures("default_vllm_config")

_ATOL_RTOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


def _reference_sc(
    module: DiffusionGemmaSelfConditioning,
    inputs_embeds: torch.Tensor,
    soft_embeds: torch.Tensor,
) -> torch.Tensor:
    if soft_embeds.dtype != inputs_embeds.dtype:
        soft_embeds = soft_embeds.to(inputs_embeds.dtype)
    x = module.pre_norm(soft_embeds)
    sc_signal = module.down_proj(
        F.gelu(module.gate_proj(x), approximate="tanh") * module.up_proj(x)
    )
    return module.post_norm(inputs_embeds + sc_signal)


def _make_sc_module(
    hidden_size: int,
    inter_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int = 0,
    *,
    requires_grad: bool = False,
) -> DiffusionGemmaSelfConditioning:
    torch.manual_seed(seed)
    mod = DiffusionGemmaSelfConditioning(
        hidden_size=hidden_size,
        self_conditioning_size=inter_size,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        mod.pre_norm.weight.copy_(
            torch.randn(hidden_size, device=device, dtype=dtype) * 0.5 + 1.0
        )
        scale = 0.05
        mod.gate_proj.weight.copy_(
            torch.randn(inter_size, hidden_size, device=device, dtype=dtype) * scale
        )
        mod.up_proj.weight.copy_(
            torch.randn(inter_size, hidden_size, device=device, dtype=dtype) * scale
        )
        mod.down_proj.weight.copy_(
            torch.randn(hidden_size, inter_size, device=device, dtype=dtype) * scale
        )
    mod.eval().requires_grad_(requires_grad)
    return mod


def _make_model_state(
    sc_module: nn.Module,
) -> DiffusionGemmaModelState:
    state = DiffusionGemmaModelState.__new__(DiffusionGemmaModelState)
    state.model = SimpleNamespace(self_conditioning=sc_module)
    return state


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m", [0, 1, 64, 256, 512])
def test_self_conditioning_parity_shapes(device: str, dtype: torch.dtype, m: int):
    h, inter = 96, 1536
    mod = _make_sc_module(h, inter, device=device, dtype=dtype)
    torch.manual_seed(10 + m)
    inp = torch.randn(m, h, device=device, dtype=dtype)
    soft = torch.randn(m, h, device=device, dtype=dtype)

    out = mod(inp, soft)
    ref = _reference_sc(mod, inp, soft)
    atol, rtol = _ATOL_RTOL[dtype]
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("soft_dtype", [None, torch.float32])
def test_self_conditioning_cuda_dtypes_and_mixed_soft(
    dtype: torch.dtype, soft_dtype: torch.dtype | None
):
    h, inter, m = 128, 1280, 64
    mod = _make_sc_module(h, inter, device="cuda", dtype=dtype, seed=3)
    torch.manual_seed(4)
    inp = torch.randn(m, h, device="cuda", dtype=dtype)
    soft = torch.randn(m, h, device="cuda", dtype=soft_dtype or dtype)

    out = mod(inp, soft)
    ref = _reference_sc(mod, inp, soft)
    atol, rtol = _ATOL_RTOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", _devices())
def test_self_conditioning_non_contiguous_inputs(device: str):
    h, inter = 64, 256
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    mod = _make_sc_module(h, inter, device=device, dtype=dtype, seed=5)
    torch.manual_seed(6)
    inp_t = torch.randn(h, 32, device=device, dtype=dtype).transpose(0, 1)
    soft_t = torch.randn(h, 32, device=device, dtype=dtype).transpose(0, 1)
    assert inp_t.stride(-1) != 1
    assert soft_t.stride(-1) != 1

    out = mod(inp_t, soft_t)
    ref = _reference_sc(mod, inp_t, soft_t)
    atol, rtol = _ATOL_RTOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
def test_triton_kernels_dispatch_on_cuda():
    h, inter, m = 64, 256, 32
    inp = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    soft = torch.randn(m, h, device="cuda", dtype=torch.float32)

    calls = {"rmsnorm": 0, "gelu": 0, "add_rmsnorm": 0}
    orig_rms = dg_mod._sc_rmsnorm_kernel
    orig_gelu = dg_mod._gelu_tanh_and_mul_kernel
    orig_add = dg_mod._sc_add_rmsnorm_kernel

    class _Spy:
        def __init__(self, target, key: str):
            self._target = target
            self._key = key

        def __getitem__(self, grid):
            calls[self._key] += 1
            return self._target[grid]

    # Instantiate inside custom_ops=["none"] (as under VLLM_COMPILE), and
    # execute outside set_current_vllm_config: Triton kernels must still dispatch.
    with set_current_vllm_config(
        VllmConfig(compilation_config=CompilationConfig(custom_ops=["none"]))
    ):
        mod = _make_sc_module(h, inter, device="cuda", dtype=torch.bfloat16, seed=7)

    with (
        set_current_vllm_config(None),  # type: ignore[arg-type]
        patch.object(dg_mod, "_sc_rmsnorm_kernel", _Spy(orig_rms, "rmsnorm")),
        patch.object(dg_mod, "_gelu_tanh_and_mul_kernel", _Spy(orig_gelu, "gelu")),
        patch.object(dg_mod, "_sc_add_rmsnorm_kernel", _Spy(orig_add, "add_rmsnorm")),
    ):
        with pytest.raises(AssertionError):
            get_current_vllm_config()
        out = mod(inp, soft)

    assert calls == {"rmsnorm": 1, "gelu": 1, "add_rmsnorm": 1}
    ref = _reference_sc(mod, inp, soft)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_sc_jit_warmup_covers_serving_variants():
    h, inter = 96, 1536
    registry = JitWarmupRegistry(
        SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    hidden_size=h,
                    intermediate_size=inter,
                ),
                hf_config=SimpleNamespace(self_conditioning_size=inter),
                dtype=torch.bfloat16,
            ),
        )
    )
    kernels = (
        _sc_rmsnorm_kernel,
        _gelu_tanh_and_mul_kernel,
        _sc_add_rmsnorm_kernel,
    )
    for k in kernels:
        k.kernel.device_caches.clear()
    with registry.activate():
        for k in kernels:
            k.register_warmup()
    registry.warmup()

    def num_variants() -> list[int]:
        return [
            sum(len(c[0]) for c in k.kernel.device_caches.values()) for k in kernels
        ]

    warmed = num_variants()
    assert all(warmed)

    mod = _make_sc_module(h, inter, device="cuda", dtype=torch.bfloat16, seed=9)
    inp = torch.randn(16, h, device="cuda", dtype=torch.bfloat16)
    for soft_dtype in (torch.bfloat16, torch.float32):
        mod(inp, torch.randn(16, h, device="cuda", dtype=soft_dtype))
    assert num_variants() == warmed


@requires_cuda
def test_apply_self_conditioning_no_gpu_sync():
    h, inter, cl, max_reqs = 64, 128, 32, 6
    sc_mod = _make_sc_module(h, inter, device="cuda", dtype=torch.bfloat16, seed=31)
    state = _make_model_state(sc_mod)

    inp = torch.randn(200, h, device="cuda", dtype=torch.bfloat16)
    sc_embeds = torch.randn(max_reqs, cl, h, device="cuda", dtype=torch.float32)

    # Warm up Triton kernels before enabling strict sync debug mode.
    qsl_c = np.array([0, 10, 10 + cl, 10 + 2 * cl, 10 + 3 * cl], dtype=np.int32)
    state._apply_self_conditioning(
        np.array([4, 0, 2], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        qsl_c,
        inp,
        sc_embeds,
    )
    qsl_var = np.array([0, 20, 20 + cl, 20 + cl + 12, 20 + 2 * cl + 5], dtype=np.int32)
    state._apply_self_conditioning(
        np.array([5, 1], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        qsl_var,
        inp,
        sc_embeds,
    )

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        state._apply_self_conditioning(
            np.array([4, 0, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
            qsl_c,
            inp,
            sc_embeds,
        )
        state._apply_self_conditioning(
            np.array([5, 1], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            qsl_var,
            inp,
            sc_embeds,
        )
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_individual_triton_kernels(dtype: torch.dtype):
    torch.manual_seed(11)
    m, h, inter = 17, 80, 1100
    x = torch.randn(m, h, device="cuda", dtype=torch.float32)
    w = torch.randn(h, device="cuda", dtype=dtype)
    eps = 1e-6

    normed = torch.empty((m, h), device="cuda", dtype=dtype)
    block_h = triton.next_power_of_2(h)
    _sc_rmsnorm_kernel[(m,)](
        x,
        w,
        normed,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        normed.stride(0),
        normed.stride(1),
        h,
        eps,
        BLOCK_H=block_h,
        CAST_INPUT_FIRST=True,
    )
    x_cast = x.to(dtype).float()
    ref_normed = (x_cast * torch.rsqrt(x_cast.pow(2).mean(-1, keepdim=True) + eps)).to(
        dtype
    ) * w
    atol, rtol = _ATOL_RTOL[dtype]
    torch.testing.assert_close(normed, ref_normed, atol=atol, rtol=rtol)

    gate_up = torch.randn(m, 2 * inter, device="cuda", dtype=dtype)
    act = torch.empty((m, inter), device="cuda", dtype=dtype)
    _gelu_tanh_and_mul_kernel[(m, triton.cdiv(inter, 1024))](
        gate_up,
        act,
        gate_up.stride(0),
        act.stride(0),
        inter,
        BLOCK_I=1024,
    )
    g, u = gate_up.chunk(2, dim=-1)
    ref_act = F.gelu(g, approximate="tanh") * u
    torch.testing.assert_close(act, ref_act, atol=atol, rtol=rtol)

    inp = torch.randn(m, h, device="cuda", dtype=dtype)
    sig = torch.randn(m, h, device="cuda", dtype=dtype)
    out = torch.empty_like(inp)
    _sc_add_rmsnorm_kernel[(m,)](
        inp,
        sig,
        out,
        inp.stride(0),
        inp.stride(1),
        sig.stride(0),
        sig.stride(1),
        out.stride(0),
        out.stride(1),
        h,
        eps,
        BLOCK_H=block_h,
    )
    s = (inp + sig).float()
    ref_out = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", _devices())
def test_merged_gate_up_invalidation_on_mutation_and_reload(device: str):
    h, inter = 32, 64
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    mod = _make_sc_module(h, inter, device=device, dtype=dtype, seed=13)
    mod.process_weights_after_loading()
    assert mod._merged_gate_up_w is not None
    ptr_before = mod._merged_gate_up_w.data_ptr()

    inp = torch.randn(8, h, device=device, dtype=dtype)
    soft = torch.randn(8, h, device=device, dtype=dtype)
    atol, rtol = _ATOL_RTOL[dtype]
    torch.testing.assert_close(
        mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
    )

    # 1. In-place weight mutation via copy_() bumps _version and refills in-place.
    with torch.no_grad():
        mod.gate_proj.weight.copy_(torch.randn_like(mod.gate_proj.weight) * 0.1)
    torch.testing.assert_close(
        mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
    )
    assert mod._merged_gate_up_w.data_ptr() == ptr_before

    # 2. In-place param.data.copy_() (as in default_weight_loader / RL refit /
    #    sleep-wake reload) leaves _version and data_ptr unchanged;
    #    process_weights_after_loading() refills _merged_gate_up_w in-place.
    with torch.no_grad():
        mod.gate_proj.weight.data.copy_(torch.randn_like(mod.gate_proj.weight) * 0.1)
        mod.up_proj.weight.data.copy_(torch.randn_like(mod.up_proj.weight) * 0.1)
    mod.process_weights_after_loading()
    assert mod._merged_gate_up_w is not None
    assert mod._merged_gate_up_w.data_ptr() == ptr_before
    torch.testing.assert_close(
        mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
    )

    # 3. load_state_dict() post-hook clears _merged_gate_up_key while keeping
    #    _merged_gate_up_w so the next refill stays in-place.
    sd = {k: v.clone() for k, v in mod.state_dict().items()}
    sd["up_proj.weight"] = torch.randn_like(sd["up_proj.weight"]) * 0.1
    mod.load_state_dict(sd)
    assert mod._merged_gate_up_key is None
    assert mod._merged_gate_up_w is not None
    torch.testing.assert_close(
        mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
    )
    assert mod._merged_gate_up_w.data_ptr() == ptr_before

    # 4. Lazy allocation inside inference_mode() followed by weight mutation
    #    and process_weights_after_loading() outside inference_mode() must
    #    not raise an inference-tensor inplace error.
    mod._merged_gate_up_w = None
    mod._merged_gate_up_key = None
    with torch.inference_mode():
        torch.testing.assert_close(
            mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
        )
    assert mod._merged_gate_up_w is not None
    assert not mod._merged_gate_up_w.is_inference()
    with torch.no_grad():
        mod.gate_proj.weight.data.copy_(torch.randn_like(mod.gate_proj.weight) * 0.1)
    mod.process_weights_after_loading()
    torch.testing.assert_close(
        mod(inp, soft), _reference_sc(mod, inp, soft), atol=atol, rtol=rtol
    )

    # 5. Module .to() clears cached merged weight via _apply.
    mod.to(dtype=torch.float32)
    assert mod._merged_gate_up_w is None
    inp32 = inp.float()
    soft32 = soft.float()
    torch.testing.assert_close(
        mod(inp32, soft32), _reference_sc(mod, inp32, soft32), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", _devices())
def test_self_conditioning_autograd_bypasses_merge(device: str):
    h, inter = 16, 32
    mod = _make_sc_module(
        h, inter, device=device, dtype=torch.float32, seed=15, requires_grad=True
    )
    inp = torch.randn(4, h, device=device, dtype=torch.float32, requires_grad=True)
    soft = torch.randn(4, h, device=device, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        assert mod._get_merged_gate_up_weight() is None
        out = mod(inp, soft)
        out.sum().backward()
    assert mod._merged_gate_up_w is None
    assert inp.grad is not None and soft.grad is not None
    assert mod.gate_proj.weight.grad is not None
    assert mod.up_proj.weight.grad is not None


@pytest.mark.parametrize("device", _devices())
def test_merged_gate_up_falls_back_for_bias_hooks_and_batch_invariant(
    device: str, monkeypatch: pytest.MonkeyPatch
):
    h, inter = 32, 64
    dtype = torch.float32
    mod = _make_sc_module(h, inter, device=device, dtype=dtype, seed=17)
    inp = torch.randn(6, h, device=device, dtype=dtype)
    soft = torch.randn(6, h, device=device, dtype=dtype)

    # Forward hook (e.g. LoRA adapter) bypasses merged cache cleanly.
    hook_calls = 0

    def _hook(_mod, _args, output):
        nonlocal hook_calls
        hook_calls += 1
        return output * 1.25

    handle = mod.gate_proj.register_forward_hook(_hook)
    assert mod._get_merged_gate_up_weight() is None
    out = mod(inp, soft)
    assert hook_calls == 1
    assert mod._merged_gate_up_w is None
    handle.remove()

    # Custom linear bias falls back cleanly.
    mod.up_proj = nn.Linear(h, inter, bias=True).to(device=device, dtype=dtype)
    mod.up_proj.requires_grad_(False)
    with torch.no_grad():
        mod.up_proj.bias.fill_(0.25)
    assert mod._get_merged_gate_up_weight() is None
    torch.testing.assert_close(
        out := mod(inp, soft), _reference_sc(mod, inp, soft), atol=1e-4, rtol=1e-4
    )
    assert out.shape == inp.shape

    # VLLM_BATCH_INVARIANT=1 disables Triton path cleanly without poisoning envs.
    mod2 = _make_sc_module(h, inter, device=device, dtype=dtype, seed=19)
    if device == "cuda":
        assert mod2._can_use_triton(inp, soft, mod2._get_merged_gate_up_weight())
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not mod2._can_use_triton(inp, soft, mod2._get_merged_gate_up_weight())
    torch.testing.assert_close(
        mod2(inp, soft), _reference_sc(mod2, inp, soft), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("cl", [32, 64])
def test_apply_self_conditioning_batch_paths(device: str, cl: int):
    h, inter = 64, 128
    dtype = torch.bfloat16
    sc_mod = _make_sc_module(h, inter, device=device, dtype=dtype, seed=21)
    state = _make_model_state(sc_mod)

    max_reqs, total_tokens = 6, 300
    torch.manual_seed(22)
    base_inp = torch.randn(total_tokens, h, device=device, dtype=dtype)
    sc_embeds = torch.randn(max_reqs, cl, h, device=device, dtype=torch.float32)

    def _run_ref(
        decode_slots: list[int],
        decode_idx: list[int],
        query_start_loc: list[int],
    ) -> torch.Tensor:
        out = base_inp.clone()
        for slot, idx in zip(decode_slots, decode_idx):
            s = query_start_loc[idx]
            e = query_start_loc[idx + 1]
            soft = sc_embeds[slot, : e - s].to(out.dtype)
            out[s:e] = sc_mod(out[s:e], soft)
        return out

    # n_dec = 0: no-op
    inp_0 = base_inp.clone()
    state._apply_self_conditioning(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([0, 10], dtype=np.int32),
        inp_0,
        sc_embeds,
    )
    assert torch.equal(inp_0, base_inp)

    # n_dec = 1 (truncated span < cl)
    qsl_1 = np.array([0, 15, 15 + cl - 5], dtype=np.int32)
    inp_1 = base_inp.clone()
    state._apply_self_conditioning(
        np.array([3], dtype=np.int64),
        np.array([1], dtype=np.int64),
        qsl_1,
        inp_1,
        sc_embeds,
    )
    torch.testing.assert_close(
        inp_1, _run_ref([3], [1], qsl_1.tolist()), atol=2e-2, rtol=2e-2
    )

    # n_dec > 1 contiguous spans + contiguous slots [1, 2, 3]
    qsl_c = np.array([0, 10, 10 + cl, 10 + 2 * cl, 10 + 3 * cl], dtype=np.int32)
    inp_c = base_inp.clone()
    state._apply_self_conditioning(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        qsl_c,
        inp_c,
        sc_embeds,
    )
    torch.testing.assert_close(
        inp_c, _run_ref([1, 2, 3], [1, 2, 3], qsl_c.tolist()), atol=2e-2, rtol=2e-2
    )

    # n_dec > 1 contiguous spans + non-contiguous slots [4, 0, 2]
    inp_nc_slots = base_inp.clone()
    state._apply_self_conditioning(
        np.array([4, 0, 2], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        qsl_c,
        inp_nc_slots,
        sc_embeds,
    )
    torch.testing.assert_close(
        inp_nc_slots,
        _run_ref([4, 0, 2], [1, 2, 3], qsl_c.tolist()),
        atol=2e-2,
        rtol=2e-2,
    )

    # n_dec > 1 non-adjacent spans + variable truncated widths
    qsl_var = np.array(
        [0, 20, 20 + cl, 20 + cl + 12, 20 + 2 * cl + 12 - 7], dtype=np.int32
    )
    inp_var = base_inp.clone()
    state._apply_self_conditioning(
        np.array([5, 1], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        qsl_var,
        inp_var,
        sc_embeds,
    )
    torch.testing.assert_close(
        inp_var, _run_ref([5, 1], [1, 3], qsl_var.tolist()), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("device", _devices())
def test_apply_self_conditioning_matching_dtype_avoids_copy(device: str):
    h, inter, cl = 16, 32, 8
    seen: list[tuple[int, torch.dtype]] = []
    sc_mod = _make_sc_module(h, inter, device=device, dtype=torch.bfloat16, seed=25)
    state = _make_model_state(sc_mod)
    orig_forward = DiffusionGemmaSelfConditioning.forward

    def _spy_forward(
        self: DiffusionGemmaSelfConditioning,
        inputs_embeds: torch.Tensor,
        soft_embeds: torch.Tensor,
    ) -> torch.Tensor:
        seen.append((soft_embeds.data_ptr(), soft_embeds.dtype))
        return orig_forward(self, inputs_embeds, soft_embeds)

    base_inp = torch.randn(cl, h, device=device, dtype=torch.bfloat16)
    sc_bf16 = torch.randn(2, cl, h, device=device, dtype=torch.bfloat16)
    sc_fp32 = torch.randn(2, cl, h, device=device, dtype=torch.float32)
    qsl = np.array([0, cl], dtype=np.int32)

    with patch.object(DiffusionGemmaSelfConditioning, "forward", _spy_forward):
        inp_bf16 = base_inp.clone()
        state._apply_self_conditioning(
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            qsl,
            inp_bf16,
            sc_bf16,
        )
        assert seen == [(sc_bf16[1, :cl].data_ptr(), torch.bfloat16)]

        # Mixed dtype: fp32 sc_embeds flows zero-copy straight into
        # DiffusionGemmaSelfConditioning (using CAST_INPUT_FIRST=True in Triton).
        seen.clear()
        inp_fp32 = base_inp.clone()
        state._apply_self_conditioning(
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            qsl,
            inp_fp32,
            sc_fp32,
        )
        assert seen == [(sc_fp32[1, :cl].data_ptr(), torch.float32)]

    ref = _reference_sc(sc_mod, base_inp, sc_fp32[1, :cl])
    torch.testing.assert_close(inp_fp32, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", _devices())
def test_apply_self_conditioning_oversized_span_guard(device: str):
    h, cl = 16, 8
    sc_mod = _make_sc_module(h, 32, device=device, dtype=torch.float32, seed=29)
    state = _make_model_state(sc_mod)

    inp = torch.randn(40, h, device=device, dtype=torch.float32)
    before = inp.clone()
    sc_embeds = torch.randn(4, cl, h, device=device, dtype=torch.float32)

    # n_dec == 1 with ln > cl raises cleanly without modifying inputs_embeds.
    with pytest.raises(RuntimeError, match="canvas length"):
        state._apply_self_conditioning(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0, cl + 3], dtype=np.int32),
            inp,
            sc_embeds,
        )
    assert torch.equal(inp, before)

    # n_dec > 1 with one oversized span (ln > cl) raises cleanly without
    # shifting or corrupting any request's rows in inputs_embeds.
    with pytest.raises(RuntimeError, match="canvas length"):
        state._apply_self_conditioning(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0, cl + 4, 2 * cl], dtype=np.int32),
            inp,
            sc_embeds,
        )
    assert torch.equal(inp, before)
