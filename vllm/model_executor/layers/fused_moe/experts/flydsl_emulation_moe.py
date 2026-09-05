# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL BF16 2-stage fused-MoE for the MXFP8-emulation path (gfx942).

Selected via ``--moe-backend aiter``; falls back to Triton on any unsupported
regime or runtime failure.
"""

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
    Mxfp8EmulationTritonExperts,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

_FLY: dict | None = None
_FLY_LOAD_FAILED = False

_EXE: dict = {}
_WSHUF: dict = {}
_POOL = {"rows": 0, "gemm1_raw": None, "a2": None, "scale_dummy": None}


def _load_fly():
    global _FLY, _FLY_LOAD_FAILED
    if _FLY is not None or _FLY_LOAD_FAILED:
        return _FLY
    try:
        import flydsl.compiler as flyc  # noqa: F401
        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
            compile_moe_gemm1,
            compile_moe_gemm2,
        )
        from aiter.ops.shuffle import shuffle_weight

        _FLY = dict(
            flyc=flyc,
            compile_moe_gemm1=compile_moe_gemm1,
            compile_moe_gemm2=compile_moe_gemm2,
            shuffle_weight=shuffle_weight,
            moe_sorting=moe_sorting,
        )
    except Exception as e:
        _FLY_LOAD_FAILED = True
        logger.warning_once("FlyDSL MoE unavailable, staying on Triton: %s", e)
        _FLY = None
    return _FLY


def is_flydsl_emulation_available() -> bool:
    """ROCm gfx942 + aiter flydsl runtime importable."""
    from vllm.platforms.rocm import on_gfx942

    if not (current_platform.is_rocm() and on_gfx942()):
        return False
    return _load_fly() is not None


_FLY_SHUFFLE_LAYOUT = (16, 16)


def shuffle_weight_to_fly_layout(w: torch.Tensor) -> torch.Tensor:
    """Shuffle ``w`` into FlyDSL layout."""
    ws = _load_fly()["shuffle_weight"](w, layout=_FLY_SHUFFLE_LAYOUT).contiguous()
    ws._fly_shuffled = True
    return ws


def shuffle_weight_inplace_ready(w: torch.Tensor):
    """Return a flat FlyDSL-layout view, using cached shuffle if available."""
    key = w.data_ptr()
    hit = _WSHUF.get(key)
    if hit is None:
        if getattr(w, "_fly_shuffled", False):
            hit = w.view(-1)
        else:
            hit = shuffle_weight_to_fly_layout(w).view(-1)
        _WSHUF[key] = hit
    return hit


def _tile_cfg(M: int):
    # k_batch must be >= 2: with k_batch=1 stage1 fuses SiLU internally instead
    # of emitting raw gate/up halves for our external SwiGLU activation.
    if M <= 256:
        kb = int(os.environ.get("FLY_DECODE_KBATCH", "6"))
        return (16, 128, 256, 128, max(2, kb))
    default_tm = 32 if M <= 768 else 64
    ptm = int(os.environ.get("FLY_PREFILL_TILE_M", str(default_tm)))
    ptk = int(os.environ.get("FLY_PREFILL_TILE_K", "64"))
    kb = int(os.environ.get("FLY_PREFILL_KBATCH", "2"))
    return (ptm, 128, 256, ptk, max(2, kb))


def _pool(rows_needed, two_inter, inter, device):
    if _POOL["rows"] < rows_needed:
        _POOL["gemm1_raw"] = torch.zeros(
            (rows_needed, two_inter), dtype=torch.bfloat16, device=device
        )
        _POOL["a2"] = torch.empty(
            (rows_needed, inter), dtype=torch.bfloat16, device=device
        )
        _POOL["rows"] = rows_needed
        if _POOL["scale_dummy"] is None:
            _POOL["scale_dummy"] = torch.empty((0,), dtype=torch.float32, device=device)
    return (
        _POOL["gemm1_raw"][:rows_needed],
        _POOL["a2"][:rows_needed],
        _POOL["scale_dummy"],
    )


def _build_exes(M, E, H, inter, topk):
    fly = _load_fly()
    tile_m, tn1, tn2, tk, kb = _tile_cfg(M)
    nt = int(os.environ.get("FLY_NT", "0"))
    nt_kw = {"weight_cache_modifier": nt} if nt else {}
    exe1 = fly["compile_moe_gemm1"](
        model_dim=H,
        inter_dim=inter,
        experts=E,
        topk=topk,
        in_dtype="bf16",
        group_size=-1,
        tile_m=tile_m,
        tile_n=tn1,
        tile_k=tk,
        doweight_stage1=False,
        # cshuffle epilog only supports f16 out in stage1; keep OFF for bf16.
        use_cshuffle_epilog=False,
        out_dtype="bf16",
        scale_is_bf16=False,
        k_batch=int(kb),
        **nt_kw,
    )
    exe2 = fly["compile_moe_gemm2"](
        model_dim=H,
        inter_dim=inter,
        experts=E,
        topk=topk,
        in_dtype="bf16",
        group_size=-1,
        tile_m=tile_m,
        tile_n=tn2,
        tile_k=tk,
        doweight_stage2=True,
        use_cshuffle_epilog=True,
        accumulate=True,
        out_dtype="bf16",
        scale_is_bf16=False,
        **nt_kw,
    )
    return dict(exe1=exe1, exe2=exe2, cexe1=None, cexe2=None, pool_gen=None)


def _run(
    experts,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
    global_num_experts: int,
):
    """Run the BF16 2-stage FlyDSL MoE in-place into *output*.
    Raises on failure so the caller can fall back to Triton."""
    fly = _load_fly()
    flyc = fly["flyc"]
    M, H = hidden_states.shape
    E = w1.shape[0]
    inter = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    if global_num_experts not in (-1, E):
        raise RuntimeError(f"EP global_num_experts={global_num_experts} != E={E}")
    dev = hidden_states.device

    key = (M, E, H, inter, topk)
    st = _EXE.get(key)
    if st is None:
        st = _build_exes(M, E, H, inter, topk)
        _EXE[key] = st

    gemm1_raw, a2, scale_dummy = _pool(M * topk, 2 * inter, inter, dev)
    pool_gen = id(_POOL["gemm1_raw"])

    hs = (
        hidden_states.to(torch.bfloat16).contiguous()
        if hidden_states.dtype != torch.bfloat16
        else hidden_states.contiguous()
    )
    w1s = shuffle_weight_inplace_ready(w1)
    w2s = shuffle_weight_inplace_ready(w2)
    stream = torch.cuda.current_stream()

    sids, sw, seids, nvi, _buf = fly["moe_sorting"](
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        E,
        H,
        torch.float16,
        _tile_cfg(M)[0],
    )
    if nvi.numel() > 1:
        nvi = nvi[:1].contiguous()
    sids = sids.contiguous()
    seids = seids.contiguous()
    sw1d = sw.contiguous().view(-1)
    blocks = int(seids.numel())

    if st["cexe1"] is None or st.get("pool_gen") != pool_gen:
        st["cexe1"] = flyc.compile(
            st["exe1"],
            gemm1_raw.view(-1),
            hs.view(-1),
            w1s,
            scale_dummy,
            scale_dummy,
            sids,
            seids,
            sw1d,
            nvi,
            M,
            inter,
            H,
            blocks,
            stream,
        )
        st["cexe2"] = flyc.compile(
            st["exe2"],
            output.view(-1),
            a2.view(-1),
            w2s,
            scale_dummy,
            scale_dummy,
            sids,
            seids,
            sw1d,
            nvi,
            M,
            H,
            inter,
            blocks,
            stream,
        )
        st["pool_gen"] = pool_gen

    gemm1_raw.zero_()
    st["cexe1"](
        gemm1_raw.view(-1),
        hs.view(-1),
        w1s,
        scale_dummy,
        scale_dummy,
        sids,
        seids,
        sw1d,
        nvi,
        M,
        inter,
        H,
        blocks,
        stream,
    )
    experts.activation(activation, a2, gemm1_raw.view(-1, 2 * inter))
    output.zero_()
    st["cexe2"](
        output.view(-1),
        a2.view(-1),
        w2s,
        scale_dummy,
        scale_dummy,
        sids,
        seids,
        sw1d,
        nvi,
        M,
        H,
        inter,
        blocks,
        stream,
    )


class FlydslEmulationExperts(Mxfp8EmulationTritonExperts):
    """FlyDSL BF16 MoE experts for gfx942, selected via ``--moe-backend aiter``."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        logger.info_once(
            "Using FlyDSL BF16 emulation MoE backend (gfx942) via --moe-backend aiter."
        )

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms.rocm import on_gfx942

        return current_platform.is_rocm() and on_gfx942()

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        # EP not supported (expert_map not wired up).
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if is_supported and not is_flydsl_emulation_available():
            return False, (
                "kernel requires the aiter flydsl runtime, which is not installed"
            )
        return is_supported, reason

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if (
            expert_map is None
            and not apply_router_weight_on_input
            and w1.element_size() >= 2
            and topk_ids.shape[1] == 4
        ):
            try:
                _run(
                    self,
                    output=output,
                    hidden_states=hidden_states,
                    w1=w1,
                    w2=w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=activation,
                    global_num_experts=global_num_experts,
                )
                return
            except Exception as e:
                logger.warning_once(
                    "FlyDSL MoE failed (M=%d), falling back to Triton: %s",
                    hidden_states.shape[0],
                    e,
                )

        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
