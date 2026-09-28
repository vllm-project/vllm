# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the RMSNorm feeding an XPU fp8 linear into the linear.

Two rewrites, both into vllm-xpu-kernels ops that run one GEMV launch for
decode rows (and the unfused norm + fp8_gemm_w8a16 otherwise):

* Gated RMSNorm + linear (GDN output projection)::

      y = RMSNormGated(x.view(-1, D), z.view(-1, D))   # norm_before_gate
      out = fp8_gemm_w8a16(y.view(T, H, D).flatten(-2), W, s)
  ->  out = _xpu_C.gated_rmsnorm_fp8_gemm(x, z, w_norm, eps, W, s)

* Gemma residual-add RMSNorm + linear(s) (attention qkv, GDN in_proj)::

      h, res' = vllm_ir.fused_add_rms_norm(x, res, w.float() + 1, eps)
      out = fp8_gemm_w8a16(h, W, s)            # or fp8_gemm_w8a16_pair
  ->  out, res' = _xpu_C.resadd_rmsnorm_fp8_gemm(x, res, w, eps, W, s)

  when ``h`` has no other user.

Only applied for compile ranges ending at <= 8 tokens.
"""

import inspect
import operator

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNormGated

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

MAX_TOKEN_NUM = 8


def _ops_available() -> bool:
    return hasattr(torch.ops, "_xpu_C") and all(
        hasattr(torch.ops._xpu_C, name)
        for name in (
            "gated_rmsnorm_fp8_gemm",
            "resadd_rmsnorm_fp8_gemm",
            "resadd_rmsnorm_fp8_gemm_pair",
        )
    )


class GatedRMSNormFp8GemmPattern:
    """RMSNormGated (norm_before_gate, no group) + fp8_gemm_w8a16."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float,
        dtype: torch.dtype,
        config: VllmConfig,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.dtype = dtype
        self.config = config

    def get_inputs(self) -> list[torch.Tensor]:
        T, H, D = 5, self.num_heads, self.head_dim

        def empty(*shape, dtype=self.dtype):
            return torch.empty(*shape, dtype=dtype, device="xpu")

        x = empty(T, H, D)
        z = empty(T, H, D)
        norm_weight = empty(D)
        weight = empty(2048, H * D, dtype=torch.float8_e4m3fn).t()
        scale = empty(1, dtype=torch.float32)
        return [x, z, norm_weight, weight, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        D, eps, dtype = self.head_dim, self.eps, self.dtype

        def pattern(x, z, norm_weight, weight, scale):
            z_shape = z.shape
            y = RMSNormGated.forward_static(
                x.reshape(-1, D),
                z.reshape(-1, D),
                norm_weight,
                eps,
                dtype,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
            )
            y = y.reshape(z_shape).flatten(-2)
            return torch.ops._xpu_C.fp8_gemm_w8a16.default(y, weight, scale, None)

        def replacement(x, z, norm_weight, weight, scale):
            return torch.ops._xpu_C.gated_rmsnorm_fp8_gemm.default(
                x, z, norm_weight, eps, weight, scale
            )

        noop = NoOpEliminationPass(self.config)

        def eliminate_noops(gm: fx.GraphModule) -> None:
            noop(gm.graph)
            gm.recompile()

        def trace_fn(*args, **kwargs):
            gm = pm.fwd_only(*args, **kwargs)
            from torch._inductor.fx_passes.post_grad import view_to_reshape

            view_to_reshape(gm)
            eliminate_noops(gm)
            return gm

        inputs = self.get_inputs()
        search_fn_pattern = pm.fx_to_pattern(
            trace_fn(pattern, inputs),
            ignore_types=(int, torch.SymInt),
            argnames=[*inspect.signature(pattern).parameters.keys()],
        )
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            trace_fn,
            pm_pass,
            extra_check=self._check,
            search_fn_pattern=search_fn_pattern,
        )

    def _check(self, match: pm.Match) -> bool:
        def val(name):
            node = match.kwargs.get(name)
            return node.meta.get("val") if isinstance(node, fx.Node) else None

        x, z, nw, w = val("x"), val("z"), val("norm_weight"), val("weight")
        H, D = self.num_heads, self.head_dim
        return (
            all(isinstance(t, torch.Tensor) for t in (x, z, nw, w))
            and x.dim() == 3
            and tuple(x.shape[1:]) == (H, D)
            and tuple(z.shape) == tuple(x.shape)
            and x.dtype == z.dtype == nw.dtype == self.dtype
            and tuple(nw.shape) == (D,)
            and w.dim() == 2
            and w.shape[0] == H * D
            and w.dtype == torch.float8_e4m3fn
        )


def _gemma_weight(weight: object) -> fx.Node | None:
    """Return w if `weight` is `w.float() + 1.0` with w a 1-D fp16/bf16 tensor."""
    if not (
        isinstance(weight, fx.Node)
        and is_func(weight, torch.ops.aten.add.Tensor)
        and len(weight.args) == 2
        and weight.args[1] == 1.0
    ):
        return None
    conv = weight.args[0]
    if not (
        isinstance(conv, fx.Node)
        and is_func(conv, torch.ops.prims.convert_element_type.default)
        and conv.args[1] == torch.float32
    ):
        return None
    w = conv.args[0]
    val = w.meta.get("val") if isinstance(w, fx.Node) else None
    if not (
        isinstance(val, torch.Tensor)
        and val.dim() == 1
        and val.dtype in (torch.float16, torch.bfloat16)
    ):
        return None
    return w


class XpuNormFp8GemmFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _ops_available()
        self.patterns = PatternMatcherPass(pass_name="xpu_gated_norm_fp8_gemm")
        self.gated_count = 0
        self.resadd_count = 0
        self._keys: tuple = ()
        if not self.enabled:
            logger.warning_once(
                "XpuNormFp8GemmFusionPass disabled: vllm-xpu-kernels has no "
                "norm + fp8 GEMV ops."
            )
            return
        mc = config.model_config
        hf = mc.hf_text_config if mc is not None else None
        heads = getattr(hf, "linear_num_value_heads", None)
        head_dim = getattr(hf, "linear_value_head_dim", None)
        eps = getattr(hf, "rms_norm_eps", None)
        tp = config.parallel_config.tensor_parallel_size
        if heads and head_dim and eps and heads % tp == 0:
            key = (heads // tp, head_dim, eps, mc.dtype)
            GatedRMSNormFp8GemmPattern(
                heads // tp, head_dim, eps, mc.dtype, config
            ).register(self.patterns)
            self._keys = (key,)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= MAX_TOKEN_NUM

    def _fuse_resadd(self, graph: fx.Graph) -> None:
        gemm = torch.ops._xpu_C.fp8_gemm_w8a16.default
        pair = torch.ops._xpu_C.fp8_gemm_w8a16_pair.default
        for norm in list(graph.nodes):
            if not is_func(norm, torch.ops.vllm_ir.fused_add_rms_norm.default):
                continue
            if len(norm.args) > 4 and norm.args[4] is not None:
                continue
            x, residual, weight, eps = norm.args[:4]
            w = _gemma_weight(weight)
            getitems = {
                u.args[1]: u for u in norm.users if u.target is operator.getitem
            }
            if w is None or len(getitems) != len(norm.users) or 0 not in getitems:
                continue
            h = getitems[0]
            if len(h.users) != 1:
                continue
            user = next(iter(h.users))
            if is_func(user, gemm) and user.args[0] is h:
                args = list(user.args) + [None] * (4 - len(user.args))
                if args[3] is not None:
                    continue
                target = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm.default
                extra = (args[1], args[2])
                n_out = 1
            elif is_func(user, pair) and user.args[0] is h:
                target = torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm_pair.default
                extra = tuple(user.args[1:5])
                n_out = 2
            else:
                continue
            with graph.inserting_before(user):
                fused = graph.call_function(target, (x, residual, w, eps, *extra))
                outs = [
                    graph.call_function(operator.getitem, (fused, i))
                    for i in range(n_out + 1)
                ]
            res_val = getitems[1].meta.get("val") if 1 in getitems else None
            if n_out == 1:
                outs[0].meta["val"] = user.meta["val"]
                user.replace_all_uses_with(outs[0])
            else:
                for i, g in ((0, 0), (1, 1)):
                    for u in [u for u in user.users if u.args[1] == g]:
                        outs[i].meta["val"] = u.meta.get("val")
                        u.replace_all_uses_with(outs[i])
                        graph.erase_node(u)
            outs[n_out].meta["val"] = res_val
            fused.meta["val"] = tuple(o.meta.get("val") for o in outs)
            if 1 in getitems:
                getitems[1].replace_all_uses_with(outs[n_out])
            graph.erase_node(user)
            for g in getitems.values():
                graph.erase_node(g)
            graph.erase_node(norm)
            self.resadd_count += 1

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.gated_count = self.patterns.apply(graph)
        self.resadd_count = 0
        self._fuse_resadd(graph)
        logger.debug(
            "XpuNormFp8GemmFusionPass fused %d gated and %d residual-add norms",
            self.gated_count,
            self.resadd_count,
        )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self, GatedRMSNormFp8GemmPattern, _gemma_weight, repr(self._keys)
        )
