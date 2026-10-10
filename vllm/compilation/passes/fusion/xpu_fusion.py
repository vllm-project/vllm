# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU fusion passes for decode-sized compile ranges.

* `XpuMoESharedFusionPass`: routed + shared-expert MoE (and optionally its
  input RMSNorm) into one fused XPU op.
* `XpuQkvNormRopeFusionPass`: gated QKV split + q/k RMSNorm + (M)RoPE into
  one kernel.
* `XpuFp8GemmPairFusionPass`: two fp8 linears sharing an input into one op.
* `XpuNormFp8GemmFusionPass`: the RMSNorm feeding an fp8 linear into the
  linear.

All of them are only applied for compile ranges ending at <= 8 tokens (the
MoE pass uses `pass_config.xpu_moe_shared_fusion_max_token_num`).
"""

import inspect
import operator
from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._functorch.compile_utils import fx_graph_cse
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
)
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.mrope import (
    apply_interleaved_rope,
)

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

MAX_TOKEN_NUM = 8

P = ParamSpec("P")


def _is_fp8_per_tensor(config: VllmConfig) -> bool:
    quant = config.quant_config
    if quant is None:
        return False
    if quant.get_name() == "online":
        # Online quantization of an unquantized checkpoint (what
        # --quantization fp8 resolves to for one): fp8 per-tensor static
        # weights, unquantized activations, for both linears and experts.
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8StaticTensorSym,
        )

        args = getattr(quant, "args", None)
        return (
            args is not None
            and args.targets is None
            and all(
                spec is not None
                and spec.weight == kFp8StaticTensorSym
                and spec.activation is None
                for spec in (args.linear, args.moe)
            )
        )
    if quant.get_name() != "fp8":
        return False
    return getattr(quant, "weight_block_size", None) is None


def _xpu_moe_shared_fused_norm_available() -> bool:
    try:
        from vllm._xpu_ops import xpu_moe_shared_fused_norm_available
    except ImportError:
        return False
    return xpu_moe_shared_fused_norm_available(2048)


def _xpu_moe_shared_fused_available() -> bool:
    # vllm._xpu_ops needs vllm_xpu_kernels; treat an import failure as "not
    # available" so compilation never fails because of this pass.
    try:
        from vllm._xpu_ops import xpu_moe_shared_fused_available
    except ImportError:
        return False
    return xpu_moe_shared_fused_available()


class XpuMoESharedFusionPass(VllmInductorPass):
    """Replace routed + shared MoE with one fused XPU op for small token counts.

    Matches, per MoE layer::

        out = vllm.moe_forward_shared(hidden, hidden, hidden, None, layer, 0)
        shared, routed = out[0], out[1]
        y = shared + routed

    and rewrites it to ``y = vllm.xpu_moe_shared_fused(hidden, hidden, layer)``.
    The TP all-reduce consuming ``y`` is kept. Only applied for compile ranges
    whose end is <= ``pass_config.xpu_moe_shared_fusion_max_token_num``, and only
    when every MoE layer of the model supports the fused op; anything that does
    not match the pattern is left unchanged.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        pass_config = config.compilation_config.pass_config
        self.max_token_num = pass_config.xpu_moe_shared_fusion_max_token_num
        mc = config.model_config
        hf = mc.hf_text_config if mc is not None else None
        pc = config.parallel_config
        self.enabled = (
            hf is not None
            and mc.dtype in (torch.float16, torch.bfloat16)
            and getattr(hf, "hidden_size", None) == 2048
            and getattr(hf, "num_experts", None) == 256
            and getattr(hf, "num_experts_per_tok", None) == 8
            and getattr(hf, "shared_expert_intermediate_size", 0) > 0
            and _is_fp8_per_tensor(config)
            and config.lora_config is None
            and not pc.enable_expert_parallel
            and not pc.enable_eplb
            and _xpu_moe_shared_fused_available()
        )
        if not self.enabled:
            logger.warning_once(
                "XpuMoESharedFusionPass disabled: model, quantization, parallel "
                "config or kernel not supported."
            )
        else:
            self.enabled = self._all_moe_layers_supported(config)
        self.matched_count = 0
        self.norm_fused_count = 0
        self.fuse_input_norm = self.enabled and _xpu_moe_shared_fused_norm_available()

    @staticmethod
    def _all_moe_layers_supported(config: VllmConfig) -> bool:
        from vllm._xpu_ops import xpu_moe_shared_fused_unsupported_reason
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        runners = get_layers_from_vllm_config(config, MoERunner)
        if not runners:
            logger.warning_once("XpuMoESharedFusionPass disabled: no MoE layers.")
            return False
        for name, runner in runners.items():
            reason = xpu_moe_shared_fused_unsupported_reason(runner)
            if reason is not None:
                logger.warning_once(
                    "XpuMoESharedFusionPass disabled: MoE layer %s is not "
                    "supported (%s).",
                    name,
                    reason,
                )
                return False
        return True

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= self.max_token_num

    @staticmethod
    def _match(node: fx.Node) -> tuple[fx.Node | None, str]:
        """Return the `add` node combining shared and routed outputs."""
        hidden, router_in, shared_in, input_ids, _layer, unpadded = node.args[:6]
        if router_in is not hidden:
            return None, "router input is not the hidden states"
        if shared_in is not hidden or input_ids is not None or unpadded != 0:
            return None, "unsupported moe_forward_shared arguments"
        getitems = {}
        for user in node.users:
            if user.target is not operator.getitem or len(user.users) != 1:
                return None, "unexpected users of moe_forward_shared"
            getitems[user.args[1]] = user
        if set(getitems) != {0, 1}:
            return None, "unexpected users of moe_forward_shared"
        add = next(iter(getitems[0].users))
        if (
            add is not next(iter(getitems[1].users))
            or add.target is not torch.ops.aten.add.Tensor
            or set(add.args) != {getitems[0], getitems[1]}
        ):
            return None, "shared and routed outputs are not simply added"
        val = hidden.meta.get("val")
        if (
            val is None
            or val.dtype not in (torch.float16, torch.bfloat16)
            or val.shape[-1] != 2048
        ):
            return None, "unsupported hidden states dtype/shape"
        return add, ""

    @staticmethod
    def _match_input_norm(hidden: fx.Node, moe: fx.Node):
        """If `hidden` is the normed output of a Gemma fused_add_rms_norm
        used only by `moe`, return (norm node, x, residual, weight, eps)."""
        if hidden.target is not operator.getitem or hidden.args[1] != 0:
            return None
        norm = hidden.args[0]
        if not (
            isinstance(norm, fx.Node)
            and norm.target is torch.ops.vllm_ir.fused_add_rms_norm.default
            and set(hidden.users) == {moe}
        ):
            return None
        if len(norm.args) > 4 and norm.args[4] is not None:
            return None
        if any(
            u.target is not operator.getitem or u.args[1] not in (0, 1)
            for u in norm.users
        ):
            return None
        x, residual, weight, eps = norm.args[:4]
        # GemmaRMSNorm passes weight.float() + 1.0.
        if not (
            isinstance(weight, fx.Node)
            and weight.target is torch.ops.aten.add.Tensor
            and len(weight.args) == 2
            and weight.args[1] == 1.0
        ):
            return None
        conv = weight.args[0]
        if not (
            isinstance(conv, fx.Node)
            and conv.target is torch.ops.prims.convert_element_type.default
            and conv.args[1] == torch.float32
        ):
            return None
        w = conv.args[0]
        w_val = w.meta.get("val") if isinstance(w, fx.Node) else None
        hidden_val = hidden.meta.get("val")
        if not (
            isinstance(w_val, torch.Tensor)
            and isinstance(hidden_val, torch.Tensor)
            and w_val.dtype == hidden_val.dtype
            and tuple(w_val.shape) == (2048,)
        ):
            return None
        for input_node in (x, residual):
            value = (
                input_node.meta.get("val") if isinstance(input_node, fx.Node) else None
            )
            if not (
                isinstance(value, torch.Tensor)
                and value.dtype == hidden_val.dtype
                and value.shape == hidden_val.shape
            ):
                return None
        return norm, x, residual, w, eps

    def _rewrite_with_norm(self, graph, node, add, layer_name, m) -> None:
        norm, x, residual, w, eps = m
        with graph.inserting_before(add):
            fused = graph.call_function(
                torch.ops.vllm.xpu_moe_shared_fused_resadd_norm.default,
                (x, residual, w, eps, layer_name),
            )
            out = graph.call_function(operator.getitem, (fused, 0))
            new_residual = graph.call_function(operator.getitem, (fused, 1))
        out.meta["val"] = add.meta["val"]
        new_residual.meta["val"] = add.meta["val"]
        fused.meta["val"] = (add.meta["val"], add.meta["val"])
        add.replace_all_uses_with(out)
        norm_users = list(norm.users)
        for u in norm_users:
            if u.args[1] == 1:
                u.replace_all_uses_with(new_residual)
        getitems = list(node.users)
        graph.erase_node(add)
        for g in getitems:
            graph.erase_node(g)
        graph.erase_node(node)
        for u in norm_users:
            graph.erase_node(u)
        graph.erase_node(norm)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        self.norm_fused_count = 0
        skipped = 0
        target = torch.ops.vllm.moe_forward_shared.default
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not target:
                continue
            add, reason = self._match(node)
            if add is None:
                logger.debug("XpuMoESharedFusionPass skipped %s: %s", node, reason)
                skipped += 1
                continue
            hidden, router_in, _, _, layer_name = node.args[:5]
            norm_match = (
                self._match_input_norm(hidden, node) if self.fuse_input_norm else None
            )
            if norm_match is not None:
                self._rewrite_with_norm(graph, node, add, layer_name, norm_match)
                self.matched_count += 1
                self.norm_fused_count += 1
                continue
            with graph.inserting_before(add):
                fused = graph.call_function(
                    torch.ops.vllm.xpu_moe_shared_fused.default,
                    (hidden, router_in, layer_name),
                )
            fused.meta["val"] = add.meta["val"]
            add.replace_all_uses_with(fused)
            getitems = list(node.users)
            graph.erase_node(add)
            for g in getitems:
                graph.erase_node(g)
            graph.erase_node(node)
            self.matched_count += 1
        logger.info(
            "XpuMoESharedFusionPass replaced %d MoE layers (%d with the input "
            "RMSNorm, %d not matched)",
            self.matched_count,
            self.norm_fused_count,
            skipped,
        )

    def uuid(self) -> str:
        return (
            self.hash_source(self)
            + f"|{self.enabled}|{self.fuse_input_norm}|{self.max_token_num}"
        )


SUPPORTED_HEAD_DIMS: tuple[int, ...] = (128, 256)


def _xpu_qkv_split_norm_rope_available() -> bool:
    return hasattr(torch.ops, "_xpu_C") and hasattr(
        torch.ops._xpu_C, "qkv_split_norm_rope"
    )


class _RopeSpec:
    """NeoX (M)RoPE parameters of the model's full-attention layers."""

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        mrope_section: list[int] | None,
        mrope_interleaved: bool,
    ) -> None:
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

    def key(self) -> tuple:
        return (
            self.head_dim,
            self.rotary_dim,
            tuple(self.mrope_section or ()),
            self.mrope_interleaved,
        )

    def apply(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Same ops as MRotaryEmbedding.forward_native (and, for 1-D
        # positions, RotaryEmbedding with a partial rotary_dim), but with
        # cos_sin_cache as an input so it matches the lifted graph buffer.
        num_tokens = positions.shape[-1]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section is not None
            if self.mrope_interleaved:
                cos = apply_interleaved_rope(cos, self.mrope_section)
                sin = apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(self.mrope_section, -1))],
                    dim=-1,
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(self.mrope_section, -1))],
                    dim=-1,
                )
        out = []
        for x in (query, key):
            shape = x.shape
            x = x.view(num_tokens, -1, self.head_dim)
            x_rot = ApplyRotaryEmb.forward_static(
                x[..., : self.rotary_dim], cos, sin, True, False
            )
            x = torch.cat((x_rot, x[..., self.rotary_dim :]), dim=-1)
            out.append(x.reshape(shape))
        return out[0], out[1]


class XpuGatedQkvNormRopePattern:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        rope: _RopeSpec,
        mrope_positions: bool,
        dtype: torch.dtype,
        config: VllmConfig,
    ) -> None:
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = rope.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.eps = eps
        self.rope = rope
        self.mrope_positions = mrope_positions
        self.dtype = dtype

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5

        def empty(*shape, dtype=self.dtype):
            return torch.empty(*shape, dtype=dtype, device="xpu")

        qkv = empty(T, 2 * self.q_size + 2 * self.kv_size)
        positions = empty(
            *((3, T) if self.mrope_positions else (T,)), dtype=torch.int64
        )
        q_weight = empty(self.head_dim)
        k_weight = empty(self.head_dim)
        cos_sin_cache = empty(4096, self.rope.rotary_dim)
        return [qkv, positions, q_weight, k_weight, cos_sin_cache]

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)
            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], -1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, nh, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
            # GemmaRMSNorm.forward_native: x * (1 + w) with an fp32 weight.
            q = vllm.ir.ops.rms_norm(
                q.view(-1, nh, hd), q_weight.float() + 1.0, self.eps
            ).view(-1, self.q_size)
            k = vllm.ir.ops.rms_norm(
                k.view(-1, nkv, hd), k_weight.float() + 1.0, self.eps
            ).view(-1, self.kv_size)
            q, k = self.rope.apply(positions, q, k, cos_sin_cache)
            # Attention.forward views q/k/v per head; after no-op elimination
            # the graph hands the RoPE outputs straight to attention.
            return q.view(-1, nh, hd), k.view(-1, nkv, hd), v.view(-1, nkv, hd), gate

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = qkv.shape[0]
            q_out = torch.empty(
                num_tokens, self.q_size, device=qkv.device, dtype=qkv.dtype
            )
            k_out = torch.empty(
                num_tokens, self.kv_size, device=qkv.device, dtype=qkv.dtype
            )
            gate_out = torch.empty(
                num_tokens, self.q_size, device=qkv.device, dtype=qkv.dtype
            )
            result = auto_functionalized(
                torch.ops._xpu_C.qkv_split_norm_rope.default,
                qkv=qkv,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                q_out=q_out,
                k_out=k_out,
                gate_out=gate_out,
                num_q_heads=nh,
                num_kv_heads=nkv,
                head_dim=hd,
                rotary_dim=self.rope.rotary_dim,
                eps=self.eps,
                weight_offset=1.0,
                mrope_section=list(self.rope.mrope_section or [0, 0, 0]),
                mrope_interleaved=self.rope.mrope_interleaved,
            )
            v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], -1)[2]
            return (
                result[1].view(-1, nh, hd),
                result[2].view(-1, nkv, hd),
                v.view(-1, nkv, hd),
                result[3],
            )

        # The graph reaching this pass has had no-op reshapes removed; do the
        # same to the traced pattern, and ignore ints so the dynamic token
        # count matches. Register a CSE'd variant as well, for graphs whose
        # duplicate rotary index computations were merged.
        noop = NoOpEliminationPass(self.config)

        def eliminate_noops(gm: fx.GraphModule) -> None:
            noop(gm.graph)
            gm.recompile()

        def cse(gm: fx.GraphModule) -> None:
            gm.graph = fx_graph_cse(gm.graph)
            gm.recompile()

        inputs = self.get_inputs()
        argnames = [*inspect.signature(pattern).parameters.keys()]
        for normalize in ((eliminate_noops,), (eliminate_noops, cse)):
            trace_fn = XpuGatedQkvNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                XpuGatedQkvNormRopePattern.fx_view_to_reshape,
                *normalize,
            )
            search_fn_pattern = pm.fx_to_pattern(
                trace_fn(pattern, inputs),
                ignore_types=(int, torch.SymInt),
                argnames=argnames,
            )
            # Different head geometries share the wildcard search topology.
            # Keep every shape guard without treating it as a duplicate rule.
            registration_pass = PatternMatcherPass()
            pm.register_replacement(
                pattern,
                replacement,
                inputs,
                trace_fn,
                registration_pass,
                extra_check=self._check,
                search_fn_pattern=search_fn_pattern,
            )
            for target, entries in registration_pass.patterns.items():
                pm_pass.patterns[target].extend(entries)

    def _check(self, match: pm.Match) -> bool:
        # Ints are wildcards in the search pattern; pin the geometry here.
        def meta(name: str) -> torch.Tensor | None:
            node = match.kwargs.get(name)
            return node.meta.get("val") if isinstance(node, fx.Node) else None

        qkv, positions = meta("qkv"), meta("positions")
        q_weight, k_weight, cache = (
            meta("q_weight"),
            meta("k_weight"),
            meta("cos_sin_cache"),
        )
        if any(t is None for t in (qkv, positions, q_weight, k_weight, cache)):
            return False
        return (
            qkv.dim() == 2
            and qkv.shape[-1] == 2 * self.q_size + 2 * self.kv_size
            and qkv.dtype == self.dtype
            and positions.dim() == (2 if self.mrope_positions else 1)
            and positions.dtype == torch.int64
            and q_weight.shape == (self.head_dim,)
            and k_weight.shape == (self.head_dim,)
            and cache.dim() == 2
            and cache.shape[-1] == self.rope.rotary_dim
            and cache.dtype == self.dtype
        )


def _rope_spec_from_config(config: VllmConfig, head_dim: int) -> _RopeSpec | None:
    hf = config.model_config.hf_text_config
    rope_parameters = getattr(hf, "rope_parameters", None) or {}
    if rope_parameters.get("rope_type", "default") not in ("default", "mrope"):
        return None
    factor = rope_parameters.get(
        "partial_rotary_factor", getattr(hf, "partial_rotary_factor", 1.0)
    )
    rotary_dim = int(head_dim * factor)
    section = rope_parameters.get("mrope_section")
    return _RopeSpec(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=list(section) if section else None,
        mrope_interleaved=bool(rope_parameters.get("mrope_interleaved", False)),
    )


class XpuQkvNormRopeFusionPass(VllmPatternMatcherPass):
    """Fuse the gated QKV split + Gemma-style q/k RMSNorm + (M)RoPE on XPU.

    Matches the unfused post-projection sequence of gated full attention
    (Qwen3-Next / Qwen3.5 / Qwen3.6)::

        q_gate, k, v = qkv.split([2 * q_size, kv_size, kv_size], -1)
        q, gate = chunk(q_gate.view(T, num_heads, 2 * head_dim), 2, -1)
        q = rms_norm(q.view(T, num_heads, head_dim), q_w.float() + 1, eps)
        k = rms_norm(k.view(T, num_kv_heads, head_dim), k_w.float() + 1, eps)
        q, k = neox_rope(positions, q, k, cos_sin_cache)  # plain or MRoPE

    and replaces it with one ``_xpu_C.qkv_split_norm_rope`` kernel that writes
    q, k and the (pre-sigmoid) gate; v stays a view of ``qkv``. Every candidate
    pattern is traced from the same reference code as the model, so a graph that
    differs in any op (other norm, rotary layout, eps, ...) is left unchanged.

    Attention layers that run this step as the fused Triton kernel
    (``fused_qk_rmsnorm_rope_gate``) get the same op in its place, when the
    kernel's geometry, eps and (M)RoPE sections match a registered pattern.

    Only applied for compile ranges ending at <= 8 tokens (decode). For large
    (prefill / memory-profiling) ranges the separate q / k / gate outputs raise
    the device memory held after profiling, which comes out of the KV cache,
    and the unfused inductor code is as fast there.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="xpu_qkv_norm_rope_fusion_pass"
        )
        self._keys: tuple = ()
        self.matched_count = 0

        dtype = config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            logger.warning_once(
                "XPU QKV norm+RoPE fusion disabled: unsupported dtype %s", dtype
            )
            return
        if not _xpu_qkv_split_norm_rope_available():
            logger.warning_once(
                "XPU QKV norm+RoPE fusion disabled: vllm-xpu-kernels has no "
                "qkv_split_norm_rope op."
            )
            return
        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )
        geometries = {
            (layer.head_size, layer.num_heads, layer.num_kv_heads)
            for layer in attn_layers.values()
            if layer.head_size in SUPPORTED_HEAD_DIMS
        }
        eps = getattr(config.model_config.hf_text_config, "rms_norm_eps", None)
        keys = []
        for head_dim, num_heads, num_kv_heads in sorted(geometries):
            rope = _rope_spec_from_config(config, head_dim)
            if rope is None or eps is None:
                continue
            half = rope.rotary_dim // 2
            if rope.rotary_dim <= 0 or half % (head_dim // 16) != 0:
                continue
            mrope_options = [False]
            if rope.mrope_section is not None and len(rope.mrope_section) == 3:
                mrope_options.append(True)
            for mrope_positions in mrope_options:
                XpuGatedQkvNormRopePattern(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    eps=eps,
                    rope=rope,
                    mrope_positions=mrope_positions,
                    dtype=dtype,
                    config=config,
                ).register(self.patterns)
                keys.append((num_heads, num_kv_heads, eps, rope.key(), mrope_positions))
        self._keys = tuple(keys)
        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= MAX_TOKEN_NUM

    def _triton_call_args(self, node: fx.Node) -> dict | None:
        """Arguments for `_xpu_C.qkv_split_norm_rope` if `node` is a
        `fused_qk_rmsnorm_rope_gate` Triton launch on the q_gate / k slices
        of one qkv split, with a geometry this pass registered."""
        if node.target is not triton_kernel_wrapper_functional:
            return None
        kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
        if getattr(getattr(kernel, "fn", None), "__name__", None) != (
            "_fused_qk_rmsnorm_rope_gate_kernel"
        ):
            return None
        kw = node.kwargs["kwargs"]
        consts = kernel_side_table.get_constant_args(node.kwargs["constant_args_idx"])
        try:
            nh, nkv = consts["num_q_heads"], consts["num_kv_heads"]
            hd, rot = consts["head_dim"], consts["rotary_dim"]
            eps, beta = consts["eps"], consts["norm_beta"]
            q_gate, k, pos = kw["q_gate_ptr"], kw["k_ptr"], kw["positions_ptr"]
        except KeyError:
            return None
        if not all(isinstance(n, fx.Node) for n in (q_gate, k, pos)):
            return None
        # q_gate and k must be slices 0 and 1 of split(qkv, [2q, kv, kv], -1).
        split = q_gate.args[0] if q_gate.target is operator.getitem else None
        if not (
            isinstance(split, fx.Node)
            and is_func(split, torch.ops.aten.split_with_sizes.default)
            and q_gate.args[1] == 0
            and k.target is operator.getitem
            and k.args == (split, 1)
            and list(split.args[1]) == [2 * nh * hd, nkv * hd, nkv * hd]
            and (len(split.args) < 3 or split.args[2] in (-1, 1))
        ):
            return None
        qkv = split.args[0]
        pos_val = pos.meta.get("val")
        if not isinstance(pos_val, torch.Tensor) or pos_val.dim() not in (1, 2):
            return None
        mrope = pos_val.dim() == 2
        for num_heads, num_kv_heads, key_eps, rope_key, mrope_positions in self._keys:
            head_dim, rotary_dim, section, interleaved = rope_key
            if (num_heads, num_kv_heads, head_dim, rotary_dim, key_eps) != (
                nh,
                nkv,
                hd,
                rot,
                eps,
            ) or mrope_positions != mrope:
                continue
            if bool(consts.get("HAS_MROPE")) != mrope or (
                mrope
                and not (
                    interleaved
                    and consts.get("MROPE_SECTION_H") == section[1]
                    and consts.get("MROPE_SECTION_W") == section[2]
                )
            ):
                continue
            return dict(
                qkv=qkv,
                positions=pos,
                q_weight=kw["q_weight_ptr"],
                k_weight=kw["k_weight_ptr"],
                cos_sin_cache=kw["cos_sin_cache_ptr"],
                q_out=kw["q_out_ptr"],
                k_out=kw["k_out_ptr"],
                gate_out=kw["gate_out_ptr"],
                num_q_heads=nh,
                num_kv_heads=nkv,
                head_dim=hd,
                rotary_dim=rot,
                eps=eps,
                weight_offset=float(beta),
                mrope_section=list(section) if mrope else [0, 0, 0],
                mrope_interleaved=bool(mrope and interleaved),
            )
        return None

    def _replace_triton_qk_norm_rope(self, graph: fx.Graph) -> int:
        """Run the model's fused Triton q/k norm + RoPE + gate kernel (used
        when the attention layer fuses that step itself) as the SYCL op,
        which has a much lower per-call host cost at decode sizes."""
        outs = {"q_out_ptr": 1, "k_out_ptr": 2, "gate_out_ptr": 3}
        count = 0
        for node in list(graph.nodes):
            args = self._triton_call_args(node)
            if args is None or any(
                u.target is not operator.getitem or u.args[1] not in outs
                for u in node.users
            ):
                continue
            with graph.inserting_before(node):
                af = graph.call_function(
                    auto_functionalized,
                    (torch.ops._xpu_C.qkv_split_norm_rope.default,),
                    args,
                )
            af.meta["val"] = (None,) + tuple(
                args[name].meta.get("val") for name in ("q_out", "k_out", "gate_out")
            )
            for user in list(node.users):
                with graph.inserting_before(user):
                    new = graph.call_function(
                        operator.getitem, (af, outs[user.args[1]])
                    )
                new.meta["val"] = user.meta.get("val")
                user.replace_all_uses_with(new)
                graph.erase_node(user)
            graph.erase_node(node)
            count += 1
        return count

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        if self._keys:
            self.matched_count += self._replace_triton_qk_norm_rope(graph)
        logger.debug("XPU QKV norm+RoPE fusion replaced %d sites", self.matched_count)

    def uuid(self) -> str:
        return (
            VllmInductorPass.hash_source(
                self,
                XpuGatedQkvNormRopePattern,
                _RopeSpec,
                repr(self._keys),
            )
            + f"|{MAX_TOKEN_NUM}"
        )


def _pair_op_available() -> bool:
    return hasattr(torch.ops, "_xpu_C") and hasattr(
        torch.ops._xpu_C, "fp8_gemm_w8a16_pair"
    )


def _is_per_tensor_fp8_gemm(node: fx.Node) -> bool:
    if not is_func(node, torch.ops._xpu_C.fp8_gemm_w8a16.default):
        return False
    args = list(node.args) + [None] * (4 - len(node.args))
    _, weight, scale, bias = args[:4]
    if bias is not None or not isinstance(scale, fx.Node):
        return False
    w, s = weight.meta.get("val"), scale.meta.get("val")
    return (
        isinstance(w, torch.Tensor)
        and w.dtype == torch.float8_e4m3fn
        and isinstance(s, torch.Tensor)
        and s.numel() == 1
    )


class XpuFp8GemmPairFusionPass(VllmInductorPass):
    """Run two XPU fp8 linears that share their input as one op.

    Rewrites exactly two ``_xpu_C.fp8_gemm_w8a16(a, B_i, scale_i, None)``
    calls on the same activation ``a`` (e.g. the GDN in_proj_qkvz and
    in_proj_ba projections) into one ``_xpu_C.fp8_gemm_w8a16_pair`` call, which
    runs a single GEMV launch for decode-sized inputs and two oneDNN GEMMs
    otherwise. Only applied for compile ranges whose end is <= 8 tokens.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _pair_op_available()
        self.matched_count = 0

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= MAX_TOKEN_NUM

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        by_input: dict[fx.Node, list[fx.Node]] = {}
        for node in graph.nodes:
            if _is_per_tensor_fp8_gemm(node) and isinstance(node.args[0], fx.Node):
                by_input.setdefault(node.args[0], []).append(node)
        for a, gemms in by_input.items():
            if len(gemms) != 2:
                continue
            g1, g2 = gemms
            # Insert after both weights/scales are available, before the
            # first use of either result (graph order is topological).
            with graph.inserting_before(g1):
                pair = graph.call_function(
                    torch.ops._xpu_C.fp8_gemm_w8a16_pair.default,
                    (a, g1.args[1], g1.args[2], g2.args[1], g2.args[2]),
                )
                out1 = graph.call_function(operator.getitem, (pair, 0))
                out2 = graph.call_function(operator.getitem, (pair, 1))
            if any(
                isinstance(arg, fx.Node) and not _before(arg, g1)
                for arg in (g2.args[1], g2.args[2])
            ):
                # g2's weight is defined after g1; keep the original pair.
                graph.erase_node(out2)
                graph.erase_node(out1)
                graph.erase_node(pair)
                continue
            pair.meta["val"] = (g1.meta["val"], g2.meta["val"])
            out1.meta["val"], out2.meta["val"] = g1.meta["val"], g2.meta["val"]
            g1.replace_all_uses_with(out1)
            g2.replace_all_uses_with(out2)
            graph.erase_node(g1)
            graph.erase_node(g2)
            self.matched_count += 1
        logger.debug("XpuFp8GemmPairFusionPass fused %d pairs", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, _is_per_tensor_fp8_gemm)


def _before(a: fx.Node, b: fx.Node) -> bool:
    """Whether node a precedes node b in the graph."""
    node = a
    while node is not None and node.op != "root":
        if node is b:
            return True
        node = node.next
    return False


def _norm_fp8_gemm_ops_available() -> bool:
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

        def norm(x, z, norm_weight):
            return RMSNormGated.forward_static(
                x,
                z,
                norm_weight,
                eps,
                dtype,
                group_size=None,
                norm_before_gate=True,
                activation="silu",
            )

        def pattern(x, z, norm_weight, weight, scale):
            # Norm on (T * H, D) rows, as the GDN output path reshapes them.
            z_shape = z.shape
            y = norm(x.reshape(-1, D), z.reshape(-1, D), norm_weight)
            y = y.reshape(z_shape).flatten(-2)
            return torch.ops._xpu_C.fp8_gemm_w8a16.default(y, weight, scale, None)

        def pattern_3d(x, z, norm_weight, weight, scale):
            # Norm directly on the (T, H, D) tensors.
            y = norm(x, z, norm_weight).flatten(-2)
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
        for search_fn in (pattern, pattern_3d):
            search_fn_pattern = pm.fx_to_pattern(
                trace_fn(search_fn, inputs),
                ignore_types=(int, torch.SymInt),
                argnames=[*inspect.signature(search_fn).parameters.keys()],
            )
            pm.register_replacement(
                search_fn,
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
    """Fuse the RMSNorm feeding an XPU fp8 linear into the linear.

    Two rewrites, both into vllm-xpu-kernels ops that run one GEMV launch for
    decode rows (and the unfused norm + fp8_gemm_w8a16 otherwise):

    * Gated RMSNorm + linear (GDN output projection)::

          y = RMSNormGated(x, z)   # norm_before_gate; (T, H, D) or (T * H, D)
          out = fp8_gemm_w8a16(y.view(T, H, D).flatten(-2), W, s)
      ->  out = _xpu_C.gated_rmsnorm_fp8_gemm(x, z, w_norm, eps, W, s)

    * Gemma residual-add RMSNorm + linear(s) (attention qkv, GDN in_proj)::

          h, res' = vllm_ir.fused_add_rms_norm(x, res, w.float() + 1, eps)
          out = fp8_gemm_w8a16(h, W, s)            # or fp8_gemm_w8a16_pair
      ->  out, res' = _xpu_C.resadd_rmsnorm_fp8_gemm(x, res, w, eps, W, s)

      when ``h`` has no other user.

    Only applied for compile ranges ending at <= 8 tokens.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _norm_fp8_gemm_ops_available()
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
