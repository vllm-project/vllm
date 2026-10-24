# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel layers of a loaded model, the refusals that need the loaded model
(``refuse_model``) or the bound KV caches (``refuse_caches``), and the fail-stop
watch on the kernels' error words. Each refusal reports every reason, not only
the first, and reads every attribute through ``getattr``: a missing one is a
reason, never an AttributeError."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm.models.muse_spark.amd.specialized_decode.mono import spec

SLIDING, FULL = "sliding", "full"
_MISSING = object()
# vLLM allocates an fp8 KV cache as uint8 and views it as the platform's e4m3
FP8_KV_DTYPES = (torch.uint8, torch.float8_e4m3fn)


def _get(obj: Any, path: str) -> Any:
    """``obj.a.b.c`` for ``path`` "a.b.c", None when any link is missing."""
    for name in path.split("."):
        obj = getattr(obj, name, _MISSING)
        if obj is _MISSING or obj is None:
            return None
    return obj


def kernel_layers(backbone) -> dict[int, str]:
    """``{layer_idx: "sliding" | "full"}`` for every layer of this stage whose FFN
    is a ``MuseSpark1xMoE`` with routed experts on AITER's MXFP4 x MXFP4 MoE
    (``MuseSparkQuarkOCPMXFusedMoE``, ``AITER_MXFP4_MXFP4``). Detected on the
    loaded model, never hard-coded: layers 1..60 of the checkpoint (Quark keeps
    the first and the last layer's experts in BF16)."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.muse_spark.amd.model import MuseSpark1xMoE
    from vllm.models.muse_spark.amd.muse_moe_runner import (
        MuseSparkQuarkOCPMXFusedMoE,
    )

    kinds = backbone.config.layer_types
    out = {}
    for i in range(backbone.start_layer, backbone.end_layer):
        ffn = getattr(backbone.layers[i], "feed_forward", None)
        if not isinstance(ffn, MuseSpark1xMoE):
            continue
        method = _get(ffn, "experts.routed_experts.quant_method")
        if not isinstance(method, MuseSparkQuarkOCPMXFusedMoE):
            continue
        if getattr(method, "mxfp4_backend", None) is not (
            Mxfp4MoeBackend.AITER_MXFP4_MXFP4
        ):
            continue
        out[i] = FULL if kinds[i] == "full_attention" else SLIDING
    return out


def _linear_ep_map(m) -> bool:
    """``m`` (vLLM's expert map, or AITER's 0/1 expert mask with its trailing
    sentinel) gives this TP rank exactly the experts [32 r, 32 r + 32), in
    order: TEP8's linear placement, the one K2(ep) indexes by."""
    if not isinstance(m, torch.Tensor) or m.dim() != 1:
        return False
    from vllm.distributed import get_tp_group

    r = get_tp_group().rank_in_group
    lo, n = spec.EXPERTS_LOCAL * r, spec.EXPERTS_LOCAL
    m = m.detach().to("cpu", torch.int64)
    if m.numel() == spec.EXPERTS + 1:  # AITER expert_mask: 1 = local, sentinel 0
        want = torch.zeros(spec.EXPERTS + 1, dtype=torch.int64)
        want[lo : lo + n] = 1
        return bool(torch.equal(m, want))
    if m.numel() == spec.EXPERTS:  # global -> local id, -1 elsewhere
        want = torch.full((spec.EXPERTS,), -1, dtype=torch.int64)
        want[lo : lo + n] = torch.arange(n)
        return bool(torch.equal(m, want))
    return False


def refuse_model(
    backbone,
    layers: Mapping[int, str],
    front: str = "stock",
    core: str = "tensor",
    max_model_len: int | None = None,
    moe: str = "tp",
) -> list[str]:
    """Stage 2: every kernel layer's modules and weights as K2 reads them (in
    place, as contiguous views of the loaded parameters), and the previous
    layer's boundary modules the kernel layer's input carries. With Phase B / C
    kernels (``front`` "k1", K2 ``core`` "gated" / "split" at q = 1; Phase A's
    by default) also what they read: K1 the previous layer's FFN boundary, the
    pre-qkv norm, qkv_proj, QK-norm and RoPE; the gated core the output norm
    and o_gate; the split core the attention window, the scale and the impl
    options (``max_model_len`` bounds the positions a RoPE table must hold)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl

    reasons: list[str] = []
    bf, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    fp4 = tuple(
        d for d in (u8, getattr(torch, "float4_e2m1fn_x2", None)) if d is not None
    )
    e8m0 = tuple(
        d for d in (u8, getattr(torch, "float8_e8m0fnu", None)) if d is not None
    )

    def need(name: str, t, shape: tuple, dtypes) -> bool:
        dtypes = dtypes if isinstance(dtypes, tuple) else (dtypes,)
        if not isinstance(t, torch.Tensor):
            reasons.append(f"{name}: missing")
            return False
        if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
            want = dtypes[0] if len(dtypes) == 1 else dtypes
            reasons.append(
                f"{name}: {tuple(t.shape)} {t.dtype}"
                f"{'' if t.is_contiguous() else ' strided'},"
                f" kernel reads contiguous {shape} {want}"
            )
            return False
        return True

    def norm_ok(norm, eps: float, before: bool) -> bool:
        # the kernels normalize over the whole row (no variance_size_override)
        return (
            getattr(norm, "fused_gain_center", None) == spec.GAIN
            and getattr(norm, "variance_epsilon", None) == eps
            and getattr(norm, "applies_weight_before_norm", None) is before
            and getattr(norm, "variance_size_override", 0) is None
        )

    def coefficients(conn, name: str) -> None:
        alpha, beta = getattr(conn, "alpha", None), getattr(conn, "beta", None)
        if not (
            getattr(conn, "_coefficients_ready", False)
            and isinstance(alpha, torch.Tensor)
            and isinstance(beta, torch.Tensor)
        ):
            reasons.append(f"{name} coefficients not materialized")
            return
        need(f"{name}.alpha", alpha, (spec.HIDDEN,), f32)
        need(f"{name}.beta", beta, (spec.HIDDEN,), f32)

    if not layers:
        return ["no layer has AITER_MXFP4_MXFP4 routed experts"]
    if 0 in layers:
        # its input is the embedding stream, not a previous layer's boundary
        reasons.append("layer 0 as a kernel layer (its input is the embedding stream)")
    H, MH, E = spec.HIDDEN, spec.MOE_HIDDEN, spec.EXPERTS
    for i in sorted(layers):
        p = f"layer {i}"
        L = backbone.layers[i]
        A, F = getattr(L, "self_attn", None), getattr(L, "feed_forward", None)
        X = _get(F, "experts.routed_experts")

        # attention: o_proj's input is the gated core [S, HEADS * HEAD_DIM]
        need(f"{p} o_proj", _get(A, "o_proj.weight"), (H, spec.CORE), bf)
        if (
            getattr(A, "num_heads", None) != spec.HEADS
            or getattr(A, "head_dim", None) != spec.HEAD_DIM
            or getattr(A, "num_kv_heads", None) != spec.KV_HEADS
        ):
            reasons.append(f"{p}: attention heads")
        if not getattr(A, "fused_qk_norm_rope_kv_insert", False):
            reasons.append(f"{p}: the fused QK-norm/RoPE/KV insert is off")
        impl = _get(A, "attn.impl")
        if not isinstance(impl, AiterFlashAttentionImpl):
            reasons.append(
                f"{p}: attention impl {type(impl).__name__} (ROCM_AITER_FA only)"
            )

        # the attention boundary
        coefficients(getattr(L, "add_attn", None), f"{p}: add_attn")
        post_attn = getattr(L, "post_attn_norm", None)
        if (
            getattr(post_attn, "has_weight", True)
            or getattr(post_attn, "variance_epsilon", None) != spec.POST_ATTN_EPS
            or getattr(post_attn, "variance_size_override", 0) is not None
        ):
            reasons.append(f"{p}: post_attn_norm (weightless, eps 1e-8)")
        pre_ffn = getattr(L, "pre_ffn_norm", None)
        need(f"{p} pre_ffn_norm", _get(pre_ffn, "weight"), (H,), bf)
        if not norm_ok(pre_ffn, spec.PRE_FFN_EPS, False):
            reasons.append(f"{p}: pre_ffn_norm (gain 1, eps 1e-5, weight after)")

        # router and routing
        need(f"{p} router", _get(F, "router.weight"), (E, H), f32)
        # the kernels read the router weight with the multiplier folded in:
        # fold_router_output_multiplier keeps the multiplier set and records the
        # fold in _router_multiplier_folded (None: a router without a multiplier)
        if getattr(F, "router_output_multiplier", 0.0) is not None and not getattr(
            F, "_router_multiplier_folded", False
        ):
            reasons.append(f"{p}: router output multiplier not folded")
        need(f"{p} expert_bias", getattr(F, "expert_bias", None), (E,), f32)
        if getattr(F, "route_scale", None) != spec.ROUTE_SCALE:
            reasons.append(f"{p}: route_scale {getattr(F, 'route_scale', None)}")
        if getattr(F, "is_sequence_parallel", True):
            reasons.append(f"{p}: sequence-parallel MoE")

        # pooler, experts, unpooler
        need(f"{p} pooler", _get(F, "pooler.weight"), (MH, H), bf)
        pre_exp = getattr(F, "pre_expert_norm", None)
        need(f"{p} pre_expert_norm", _get(pre_exp, "weight"), (MH,), bf)
        if not norm_ok(pre_exp, spec.PRE_EXP_EPS, False):
            reasons.append(f"{p}: pre_expert_norm (gain 1, eps 1e-5, weight after)")
        geom = spec.moe_geom(moe)
        XE, XI = geom.experts, geom.inter  # TP8: 256 x 512, TEP8: 32 x 4096
        w13 = getattr(X, "w13_weight", None)
        need(f"{p} w13", w13, (XE, 2 * XI, MH // 2), fp4)
        if not getattr(w13, "is_shuffled", False):
            reasons.append(f"{p}: w13 not AITER-shuffled")
        need(
            f"{p} w13 scale",
            getattr(X, "w13_weight_scale", None),
            (XE, 2 * XI, MH // 32),
            e8m0,
        )
        w2 = getattr(X, "w2_weight", None)
        need(f"{p} w2", w2, (XE, MH, XI // 2), fp4)
        if not getattr(w2, "is_shuffled", False):
            reasons.append(f"{p}: w2 not AITER-shuffled")
        need(
            f"{p} w2 scale",
            getattr(X, "w2_weight_scale", None),
            (XE, MH, XI // 32),
            e8m0,
        )
        if moe == "ep":
            if not _linear_ep_map(getattr(X, "expert_map", None)):
                reasons.append(f"{p}: the expert map is not this rank's linear 32")
        elif getattr(X, "expert_map", None) is not None:
            reasons.append(f"{p}: an expert map (EP)")
        if getattr(X, "activation", None) != MoEActivation.SILU:
            reasons.append(f"{p}: expert activation {getattr(X, 'activation', None)}")
        post_exp = getattr(F, "post_expert_norm", None)
        need(f"{p} post_expert_norm", _get(post_exp, "weight"), (MH,), bf)
        if not norm_ok(post_exp, spec.POST_EXP_EPS, True):
            reasons.append(f"{p}: post_expert_norm (gain 1, eps 1e-8, weight first)")
        need(f"{p} unpooler", _get(F, "unpooler.weight"), (H, MH), bf)

        # Phases B and C: what K1 and K2's gated / split cores read
        if front == "k1" or core != "tensor":
            reasons += _attention_reasons(p, A, layers[i], front, core, max_model_len)
        if front == "k1":
            qkv = _get(A, "qkv_proj.weight")
            need(f"{p} qkv_proj", qkv, (spec.QKV_ROWS, H), bf)
            if _get(A, "qkv_proj.weight_scale") is not None:
                reasons.append(f"{p}: qkv_proj quantized")
            pre_qkv = getattr(A, "pre_qkv_proj_norm", None)
            need(f"{p} pre_qkv_proj_norm", _get(pre_qkv, "weight"), (H,), bf)
            if not norm_ok(pre_qkv, spec.RMS_EPS, False):
                reasons.append(
                    f"{p}: pre_qkv_proj_norm (gain 1, eps 1e-5, weight after)"
                )

        # the previous layer's FFN boundary, which this layer's input carries
        if i == 0:
            continue
        P = backbone.layers[i - 1]
        need(f"layer {i - 1} post_ffn_norm", _get(P, "post_ffn_norm.weight"), (H,), bf)
        if not getattr(getattr(P, "add_ffn", None), "_coefficients_ready", False):
            reasons.append(f"layer {i - 1}: add_ffn coefficients not materialized")
        if front == "k1":
            # K1 runs this boundary: its post norm and both coefficients
            if not norm_ok(getattr(P, "post_ffn_norm", None), spec.POST_FFN_EPS, True):
                reasons.append(
                    f"{p}: previous post_ffn_norm (gain 1, eps 1e-8, weight first)"
                )
            conn = getattr(P, "add_ffn", None)
            if not all(
                isinstance(t, torch.Tensor)
                and tuple(t.shape) == (H,)
                and t.dtype == f32
                and t.is_contiguous()
                for t in (getattr(conn, "alpha", None), getattr(conn, "beta", None))
            ):
                reasons.append(f"{p}: previous add_ffn coefficients not materialized")
    return reasons


def _attention_reasons(
    p: str, A, kind: str, front: str, core: str, max_model_len: int | None
) -> list[str]:
    """A kernel layer's attention as Phase B / C reads it (DESIGN_BC 8.2):
    weightless QK and output norms with the gate fused (K1 normalizes q / k,
    the gated and split cores apply the output norm and o_gate); for K1 and the
    split core also the layer's kind (the window, RoPE only on sliding layers,
    GPT-J over the whole head) and the impl's options."""
    reasons = []
    if not (
        _get(A, "qk_norm") is not None
        and getattr(A.qk_norm, "has_weight", True) is False
        and _get(A, "o_norm") is not None
        and getattr(A.o_norm, "has_weight", True) is False
        and getattr(A, "fuse_o_norm_gate", False) is True
        and getattr(A, "rms_norm_eps", None) == spec.RMS_EPS
        and getattr(A, "use_attn_temperature_tuning", True) is False
    ):
        reasons.append(f"{p}: QK / output norm")
    if front != "k1" and core != "split":
        return reasons
    impl = _get(A, "attn.impl")
    rope = _get(A, "rotary_emb")
    window = getattr(impl, "sliding_window", None)
    if kind == SLIDING:
        table = _get(rope, "cos_sin_cache")
        ok = (
            window == (spec.WINDOW - 1, 0)
            and rope is not None
            and getattr(rope, "is_neox_style", True) is False
            and getattr(rope, "rotary_dim", None) == spec.HEAD_DIM
            and getattr(rope, "head_size", None) == spec.HEAD_DIM
            and isinstance(table, torch.Tensor)
            and table.dtype == torch.float32
            and table.is_contiguous()
            and table.ndim == 2
            and table.shape[1] == spec.HEAD_DIM
            and (max_model_len is None or table.shape[0] >= max_model_len)
        )
    else:
        ok = window == (-1, -1) and rope is None
    if not ok:
        reasons.append(f"{p}: {kind} attention (window / RoPE)")
    if not (
        impl is not None
        and getattr(impl, "sinks", 0) is None
        and getattr(impl, "alibi_slopes", 0) is None
        and getattr(impl, "logits_soft_cap", None) == 0
        and getattr(impl, "kv_sharing_target_layer_name", 0) is None
        and getattr(impl, "num_queries_per_kv", None) == spec.Q_PER_KV
        and isinstance(getattr(impl, "scale", None), (int, float))
        and float(impl.scale) == spec.QK_SCALE
    ):
        reasons.append(f"{p}: attention impl options")
    return reasons


def refuse_caches(
    backbone, layers: Mapping[int, str], vllm_config, reads_kv: bool = False
) -> list[str]:
    """Stage 3, at every KV bind: every kernel layer's KV cache as the Phase B/C
    kernels address it ([blocks, 2, 16, 128] bf16, or fp8 e4m3 under
    ``kv_cache_dtype`` fp8, dense in its last dim), and
    the resolved decode CUDA-graph mode. Phase A's K2 reads no KV cache; with a
    kernel that does (``reads_kv``: K1, the split core) also the page addressing
    (DESIGN_BC 8.3): the KV manager's blocks are the kernel's, rows 16-byte
    aligned, a page within the 32-bit buffer window.

    Not refused: a view spanning more than 4 GiB (every real layer's does; the
    kernels use 64-bit page bases) and layers sharing one tensor (KV groups alias
    the same bytes)."""
    from vllm.config import CUDAGraphMode
    from vllm.models.muse_spark.amd.specialized_decode.runtime import kv_fp8

    reasons = []
    block = getattr(getattr(vllm_config, "cache_config", None), "block_size", None)
    fp8 = kv_fp8(vllm_config)
    dtypes, want = (FP8_KV_DTYPES, "fp8 e4m3") if fp8 else ((torch.bfloat16,), "bf16")
    for i in sorted(layers):
        kv = _get(backbone.layers[i], "self_attn.attn.kv_cache")
        if not isinstance(kv, torch.Tensor):
            reasons.append(f"layer {i}: no KV cache tensor")
            continue
        if kv.dtype not in dtypes:
            reasons.append(f"layer {i}: KV cache {kv.dtype} ({want} only)")
        if kv.ndim != 4 or kv.shape[1] != 2 or kv.shape[3] != 2 * spec.HEAD_DIM:
            reasons.append(
                f"layer {i}: KV cache shape {tuple(kv.shape)} "
                "(needs [blocks, 2, 16, 128])"
            )
        elif kv.shape[2] != 16:
            reasons.append(
                f"layer {i}: KV kernel block {kv.shape[2]} (--block-size 16 only)"
            )
        if kv.ndim and kv.stride(-1) != 1:
            reasons.append(f"layer {i}: KV cache not dense in its last dim")
        if not reads_kv or kv.ndim != 4:
            continue
        if block != kv.shape[2]:
            reasons.append(
                f"layer {i}: KV manager block {block} != kernel block {kv.shape[2]}"
            )
        n = kv.element_size()
        if (
            kv.data_ptr() % 16
            or (kv.stride(2) * n) % 16
            or min(kv.stride(0), kv.stride(1), kv.stride(2)) <= 0
            or kv.stride(3) != 1
        ):
            reasons.append(
                f"layer {i}: KV rows not 16-byte aligned / not contiguous in the "
                "head dim"
            )
        if spec.KV_BLOCK * kv.stride(2) * n >= 2**31:
            reasons.append(f"layer {i}: KV page exceeds the 32-bit window")
    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is not None and mode.decode_mode() == CUDAGraphMode.PIECEWISE:
        reasons.append(
            "PIECEWISE decode CUDA graphs (use FULL_AND_PIECEWISE or FULL_DECODE_ONLY)"
        )
    return reasons


class FailStop:
    """Stops every TP rank at the same step once any of them gave up an in-kernel
    wait (its health word FAILED, ``mono.abi``). Each step queues a D2H copy of
    this rank's word and keeps the newest copy that has landed, without blocking;
    every CHECK_EVERY steps the ranks agree on those over the TP CPU group and all
    raise together. A rank that raised alone would leave the others in the next
    collective for ever. A failure is seen within about CHECK_EVERY steps."""

    CHECK_EVERY = 16

    def __init__(self, failed: torch.Tensor, rank: int, world: int, group):
        self.failed, self.rank, self.world, self.group = failed, rank, world, group
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied: torch.cuda.Event | None = None
        self.seen = False
        self.steps = 0

    def after_step(self) -> None:
        """Once per step on every TP rank (compute_logits, itself collective)."""
        if torch.cuda.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.cuda.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        torch.distributed.all_reduce(votes, group=self.group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"Muse Spark specialized decode: TP ranks {ranks} gave up an "
                "in-kernel wait; stopping the engine (decode_backend=auto without "
                "--monokernel runs vLLM's decode)"
            )
