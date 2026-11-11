# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stock AITER routed-MoE call of one MiniMax-M3 layer, split at the sort.

Stock: vLLM `rocm_aiter_fused_experts` -> `aiter.fused_moe.fused_moe` ->
`_fused_moe_impl`: a tuned 2-stage row by padded M (`get_2stage_cfgs`), the
top-k sorted for that row (`moe_sorting`: its block_m, accumulate and aux
layout), the activation quantized, then `fused_moe_2stages`. `AiterMoe`
derives the same arguments and the same tuned row, so a caller that brings its
own sort (the decode glue kernel) or its own prequantized FP4 rows (SP prefill)
runs exactly the stage-1 / stage-2 kernels the stock path would run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

H_DTYPE = torch.bfloat16


def _func(stage) -> Any:
    return getattr(stage, "func", stage)


def _kw(stage) -> dict:
    return getattr(stage, "keywords", None) or {}


@dataclass
class AiterMoe:
    """`w1` / `w2` / scales: the tensors the stock path hands AITER (their
    `is_shuffled` / `gate_up_interleave` attributes steer the tuned row)."""

    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    topk: int
    activation: Any = None  # aiter ActivationType; None: Swiglu
    gate_mode: Any = None  # aiter GateMode; None: SEPARATED
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float | None = 7.0
    dispatch_policy: int = 0
    _md: dict = field(default_factory=dict, repr=False)
    _tokens: dict = field(
        default_factory=dict, repr=False
    )  # q_dtype_a -> token counts with a tuned row

    def __post_init__(self) -> None:
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import get_inter_dim, quant_remap
        from aiter.ops.flydsl.moe_common import GateMode

        a = self.activation
        self.activation = (
            ActivationType.Swiglu
            if a is None
            else ActivationType(getattr(a, "value", a))
        )
        self.gate_mode = GateMode(
            getattr(self.gate_mode, "value", self.gate_mode) or GateMode.SEPARATED.value
        )
        self.quant_type = quant_remap.get(QuantType.per_1x32, QuantType.per_1x32)
        self.E, self.model_dim, self.inter_dim = get_inter_dim(
            self.w1.shape, self.w2.shape
        )
        self.is_g1u1 = self.inter_dim != self.w1.shape[1]
        s1, s2 = (
            getattr(self.w1, "is_shuffled", False),
            getattr(self.w2, "is_shuffled", False),
        )
        self.is_shuffled = s1 or s2
        self.opus_shuffled = s1 and s2
        self.q_dtype_w = self.w1.dtype

    # ------------------------------------------------------------ from vLLM
    @classmethod
    def from_moe_module(cls, moe) -> AiterMoe:
        """`moe`: a `MiniMaxM3MoE`. Mirrors `rocm_aiter_fused_experts` for
        its `AiterMxfp4Experts` (after process_weights_after_loading)."""
        from aiter.ops.flydsl.moe_common import GateMode

        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            AiterExperts,
            AiterQuantType,
            _resolve_aiter_activation,
            aiter_moe_pads,
        )

        routed = moe.experts.routed_experts
        kernel = getattr(routed.quant_method, "moe_kernel", None)
        fe = getattr(kernel, "fused_experts", None)
        if (
            not isinstance(fe, AiterExperts)
            or fe._AITER_QUANT_TYPE != AiterQuantType.BLOCK_1X32
        ):
            raise ValueError(f"routed experts run {type(fe).__name__}, not AITER MXFP4")
        if fe._extra_fused_moe_kwargs():
            raise ValueError("AITER MXFP4 w4a16 / DSv4 overrides are not supported")
        if routed.expert_map is not None or routed.apply_router_weight_on_input:
            raise ValueError(
                "expert parallelism / router weight on input are not supported"
            )
        mc, qc = fe.moe_config, fe.quant_config
        if qc.w1_bias is not None or qc.w2_bias is not None:
            raise ValueError("MoE biases are not supported")
        if routed.activation == MoEActivation.SITU:
            raise ValueError("SiTU activation is not supported")
        act, interleave = _resolve_aiter_activation(routed.activation)
        gate_mode = (
            None
            if interleave is None
            else (GateMode.INTERLEAVE if interleave else GateMode.SEPARATED)
        )
        w1, w2 = routed.w13_weight, routed.w2_weight
        hidden_pad = w2.shape[1] - mc.hidden_dim_unpadded
        inter_pad = (
            mc.intermediate_size_per_partition
            - (mc.intermediate_size_per_partition_unpadded or 0)
            if mc.intermediate_pad is None
            else mc.intermediate_pad
        )
        hidden_pad, inter_pad = aiter_moe_pads(
            hidden_pad,
            inter_pad,
            routed.activation,
            qc.use_mxfp4_w4a4,
            qc.use_mxfp4_w4a16,
            mc.tp_size,
        )
        if qc.use_mxfp4_w4a16:
            # As rocm_aiter_fused_experts: AITER's bias-less W4A16 FlyDSL stage 1
            # leaves the padded intermediate columns unwritten (ROCm/aiter#6598);
            # the padded weights are zero, so no pad is declared.
            inter_pad = 0
        return cls(
            w1=w1,
            w2=w2,
            w1_scale=qc.w1_scale,
            w2_scale=qc.w2_scale,
            topk=mc.experts_per_token,
            activation=act,
            gate_mode=gate_mode,
            hidden_pad=hidden_pad,
            intermediate_pad=inter_pad,
            swiglu_limit=0.0 if mc.swiglu_limit is None else float(mc.swiglu_limit),
            dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        )

    # ------------------------------------------------------------ tuned row
    def tuned_tokens(self, q_dtype_a) -> frozenset:
        """Token counts with a tuned row of this layer's shape in the active fmoe table
        (AITER_CONFIG_FMOE)."""
        key = str(q_dtype_a)
        if key not in self._tokens:
            import csv

            from aiter.fused_moe import get_cu_num
            from aiter.jit.core import AITER_CONFIGS

            want = {
                "model_dim": str(self.model_dim),
                "inter_dim": str(self.inter_dim),
                "expert": str(self.E),
                "topk": str(self.topk),
                "q_dtype_a": key,
                "q_dtype_w": str(self.q_dtype_w),
                "q_type": str(self.quant_type),
                "cu_num": str(get_cu_num()),
            }
            with open(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE) as f:
                toks = {
                    int(r["token"])
                    for r in csv.DictReader(f)
                    if all(r.get(k) == v for k, v in want.items())
                }
            self._tokens[key] = frozenset(toks)
        return self._tokens[key]

    def metadata(self, m: int, exact: bool = False):
        """(tuned row, activation dtype) of the stock call for `m` bf16 rows,
        as `_fused_moe_impl` resolves them (cached; host only). `exact`: the
        row tuned for exactly `m` rows when the table has one (the decode glue's
        widths), else the stock padded-M row."""
        from aiter.fused_moe import (
            _normalize_mxfp4_activation_params,
            get_2stage_cfgs,
            get_padded_M,
            resolve_activation_dtype,
        )

        interleave = getattr(self.w1, "gate_up_interleave", None)
        q_dtype_a = resolve_activation_dtype(
            self.quant_type,
            self.q_dtype_w,
            activation=self.activation,
            gate_mode=self.gate_mode,
            M=m,
            hidden_dtype=H_DTYPE,
            has_a1_scale=False,
            gate_up_interleave=interleave,
        )
        tok = m if exact and m in self.tuned_tokens(q_dtype_a) else get_padded_M(m)
        key = (tok, q_dtype_a)
        if key not in self._md:
            beta, linear_beta, _ = _normalize_mxfp4_activation_params(
                self.activation, None, None, self.swiglu_limit
            )
            md = get_2stage_cfgs(
                key[0],
                self.model_dim,
                self.inter_dim,
                self.E,
                self.topk,
                H_DTYPE,
                q_dtype_a,
                self.q_dtype_w,
                self.quant_type,
                self.is_g1u1,
                self.activation,
                False,
                self.hidden_pad,
                self.intermediate_pad,
                self.is_shuffled,
                self.gate_mode,
                is_ep=False,
                has_stage1_bias=False,
                has_stage2_bias=False,
                situ_beta=beta,
                situ_linear_beta=linear_beta,
                swiglu_limit=self.swiglu_limit,
                opus_weights_shuffled=self.opus_shuffled,
                gate_up_interleave=interleave,
                config_file=None,
                _disable_inline_sort=False,
                q_dtype_a2=None,
                input_dtype=H_DTYPE,
                has_stage2_scatter=False,
                has_activation_scales=False,
                has_num_local_tokens=False,
            )
            self._md[key] = (md, q_dtype_a)
        return self._md[key]

    def describe(self, m: int, exact: bool = False) -> str:
        md, q = self.metadata(m, exact)
        k1, k2 = _kw(md.stage1), _kw(md.stage2)
        tok = m if exact and m in self.tuned_tokens(q) else "padded"
        return (
            f"M={m} (row {tok}) q_dtype_a={q} block_m={md.block_m} "
            f"aux={md.output_aux!r} "
            f"g1={k1.get('kernelName1') or k1.get('kernelName')} "
            f"g2={k2.get('kernelName2') or k2.get('kernelName')}"
        )

    def _reduces(self, md, m: int) -> bool:
        from aiter.fused_moe import _stage2_runtime_uses_route_reduce

        return _stage2_runtime_uses_route_reduce(md.stage2, m, self.model_dim)

    def scatter_deferrable(self, m: int) -> bool:
        """The stock row for `m` rows ends in the bf16 scatter epilogue (FlyDSL v2 GEMM2
        + AITER `mxfp4_moe_scatter_reduce`) and this AITER build can hand back the
        uncombined intermediate (`scatter_intermediate`, patched `fused_moe.py`). Host
        only, cached by padded M."""
        from aiter.fused_moe import get_padded_M

        key = ("scatter", get_padded_M(m))
        if key not in self._md:
            self._md[key] = self._scatter_deferrable(m)
        return self._md[key]

    def _scatter_deferrable(self, m: int) -> bool:
        import inspect

        from aiter import fused_moe as fm
        from aiter.ops.flydsl.mxfp4_kname import parse_g2_kname_any

        wrap = getattr(fm, "_flydsl_v2_stage2_wrapper", None)
        fw = getattr(fm, "_mxfp4_a4w4_stage2_fw", None)
        if wrap is None or fw is None:
            return False
        if any(
            "scatter_intermediate" not in inspect.signature(f).parameters
            for f in (wrap, fw)
        ):
            return False  # stock AITER: no deferral kwarg
        if getattr(fm, "_mxfp4_intermediate_enabled", lambda: False)():
            # fp4 scatter intermediate (AITER_MXFP4_INTERMEDIATE=1): reduced inside
            return False
        md, _ = self.metadata(m)
        func, k2 = _func(md.stage2), _kw(md.stage2)
        if func not in (wrap, fw) or md.run_1stage or md.full_impl is not None:
            return False
        name = k2.get("kernelName2") or k2.get("kernelName", "")
        try:
            cfg = parse_g2_kname_any(name)
        except Exception:  # noqa: BLE001
            return False
        if func is fw and not cfg.get("v2"):
            return False  # native MXMOE GEMM2: not the FlyDSL v2 wrapper
        cfg2 = fm.parse_flydsl_v2_gemm2_kernel(name)
        return bool(cfg2) and cfg2.get("epilog") == "scatter"

    def decode_refusal(self, m: int, exact: bool = False) -> str | None:
        """Why the glue's sort (block 16, accumulate, plain layout, zeroed
        moe_buf) cannot feed the stock row at `m` rows, or None."""
        from aiter.fused_moe import _flydsl_stage1_wrapper, _flydsl_stage2_wrapper

        md, q = self.metadata(m)
        if q != torch.bfloat16:
            return f"activation dtype {q} (A16W4 expected)"
        if (
            md.run_1stage
            or md.full_impl is not None
            or int(md.flat) != 0
            or md.output_aux
        ):
            return "not a plain two-stage row"
        if int(md.block_m) != 16:
            return f"block_m {md.block_m}"
        if (
            _func(md.stage1) is not _flydsl_stage1_wrapper
            or _func(md.stage2) is not _flydsl_stage2_wrapper
        ):
            return "not the FlyDSL A16W4 stages"
        if self._reduces(md, m):
            return "reduce epilogue"
        return None

    def sp_refusal(self, m: int) -> str | None:
        """Why prequantized FP4 rows cannot feed the stock row at `m` rows."""
        from aiter import dtypes
        from aiter.fused_moe import (
            _flydsl_stage1_wrapper,
            _get_flydsl_moe_kernels,
            _is_mxfp4_inline_sort,
            _mxfp4_a4w4_stage1_fw,
        )
        from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname

        md, q = self.metadata(m)
        if q != dtypes.fp4x2:
            return f"activation dtype {q} (A4W4 expected)"
        if md.run_1stage or md.full_impl is not None or int(md.flat) != 0:
            return "not a plain two-stage row"
        if _is_mxfp4_inline_sort(md):
            return "inline-sort row"
        s1, k1 = _func(md.stage1), _kw(md.stage1)
        if s1 is _mxfp4_a4w4_stage1_fw:
            p1 = _parse_mxfp4_g1_kname(k1.get("kernelName1", ""))
            if p1["a_dtype"] != "fp4" or p1["inline_quant"]:
                return "GEMM1 quantizes bf16 rows inline"
            return None
        if s1 is _flydsl_stage1_wrapper and not md.output_aux and md.prequant:
            params = _get_flydsl_moe_kernels().get_flydsl_kernel_params(
                k1.get("kernelName", "")
            )
            if params and params.get("a_dtype") == "fp4":
                return None
        return f"stage1 {getattr(s1, '__name__', s1)}"

    # ------------------------------------------------------------ the call after the
    # sort
    def sort(self, md, ids: torch.Tensor, w: torch.Tensor):
        """The stock sort for row `md`: (sorted ids, weights, expert ids,
        num_valid, moe_buf, m_indices, reverse_sorted)."""
        from aiter.fused_moe import AsmMoeMode, moe_sorting
        from aiter.ops.flydsl.mxfp4_kname import parse_g2_kname_any

        m = ids.shape[0]
        reduce = self._reduces(md, m)
        bm = int(md.block_m)
        if md.output_aux:
            k2 = _kw(md.stage2)
            atomic = parse_g2_kname_any(
                k2.get("kernelName2") or k2.get("kernelName", "")
            )["atomic"]
            ret = moe_sorting(
                ids,
                w,
                self.E,
                self.model_dim,
                H_DTYPE,
                bm,
                None,
                None,
                return_local_topk_ids=False,
                accumulate=atomic and not reduce,
                output_aux=md.output_aux,
                output=None,
            )
            return tuple(ret[:7])
        ret = moe_sorting(
            ids,
            w,
            self.E,
            self.model_dim,
            H_DTYPE,
            bm,
            None,
            None,
            self.dispatch_policy,
            return_local_topk_ids=False,
            accumulate=not reduce,
            flat=AsmMoeMode(md.flat),
            output=None,
        )
        return (*ret[:5], None, None)

    def run_sorted(
        self,
        md,
        q_dtype_a,
        hidden,
        sorted7,
        ids,
        w,
        a1=None,
        a1_scale=None,
        shared=None,
        scatter=None,
    ):
        """`fused_moe_2stages` on a ready sort with row `md` (the routed
        output, [M, model_dim] bf16; `a1` / `a1_scale`: prequantized rows and
        their sorted scales). `shared`: (gate_up, down, scratch) of a dense bf16
        shared expert the a16w4 kernels add in (gate_up [2 I, H] / down [H, I]
        preshuffled, scratch [M, I] bf16); the output is then routed + shared.
        `scatter`: a dict, only for rows where `scatter_deferrable`: stage 2
        skips the top-k combine and fills it with the intermediate (AITER's
        `scatter_intermediate`); the returned output is then NOT written."""
        from aiter.fused_moe import fused_moe_2stages

        sids, sw, seids, nv, buf, m_idx, rev = sorted7
        extra1 = extra2 = None
        if shared is not None:
            gu, dn, scratch = shared
            extra1 = {"shared_w1_bf16": gu, "shared_inter_bf16": scratch}
            extra2 = {"shared_w2_bf16": dn, "shared_inter_bf16": scratch}
        if scatter is not None:
            extra2 = {**(extra2 or {}), "scatter_intermediate": scatter}
        return fused_moe_2stages(
            hidden,
            self.w1,
            self.w2,
            self.topk,
            sids,
            sw,
            seids,
            nv,
            buf,
            self.is_g1u1,
            int(md.block_m),
            activation=self.activation,
            quant_type=self.quant_type,
            doweight_stage1=False,
            q_dtype_a=q_dtype_a,
            q_dtype_w=self.q_dtype_w,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            a1_scale=None,
            a2_scale=None,
            num_local_tokens=None,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            bias1=None,
            bias2=None,
            topk_ids=ids,
            topk_weights=w,
            swiglu_limit=self.swiglu_limit,
            beta=None,
            linear_beta=None,
            gate_mode=self.gate_mode,
            expert_mask=None,
            m_indices=m_idx,
            reverse_sorted=rev,
            _metadata_transform=lambda _: md,
            routing_num_experts=self.E,
            prequantized_a1=a1,
            prequantized_a1_scale=a1_scale,
            _stage1_extra_args=extra1,
            _stage2_extra_args=extra2,
        )

    def run_prequant(
        self, fp4: torch.Tensor, scale: torch.Tensor, ids, w, scatter=None
    ) -> torch.Tensor:
        """SP: the stock row for M = len(ids) on prequantized rows. `fp4`: [M, model_dim
        / 2] packed codes; `scale`: their per-row E8M0 [M, model_dim / 32]. `scatter`:
        see `run_sorted` (pass it only when `scatter_deferrable(M)`)."""
        from aiter import dtypes
        from aiter.ops.quant import mxfp4_moe_sort_fwd

        m = ids.shape[0]
        md, q = self.metadata(m)
        sorted7 = self.sort(md, ids, w)
        a1_scale = mxfp4_moe_sort_fwd(
            scale.view(dtypes.fp8_e8m0),
            sorted_ids=sorted7[0],
            num_valid_ids=sorted7[3],
            token_num=m,
            cols=self.model_dim,
        )
        a1 = fp4.view(dtypes.fp4x2)
        return self.run_sorted(
            md, q, a1, sorted7, ids, w, a1=a1, a1_scale=a1_scale, scatter=scatter
        )

    def run_fused_prequant(self, fp4, scale, ids, w) -> torch.Tensor:
        """SP, cross-check variant: aiter.fused_moe on the prequantized rows and
        their unsorted scales (its MoRI-dispatch input path)."""
        from aiter import dtypes
        from aiter.fused_moe import fused_moe

        return fused_moe(
            fp4.view(dtypes.fp4x2),
            self.w1,
            self.w2,
            w,
            ids,
            None,
            self.activation,
            self.quant_type,
            False,
            self.w1_scale,
            self.w2_scale,
            scale.view(dtypes.fp8_e8m0),
            None,
            moe_sorting_dispatch_policy=self.dispatch_policy,
            dtype=H_DTYPE,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            swiglu_limit=self.swiglu_limit,
            gate_mode=self.gate_mode.value,
        )
