# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""

import inspect
from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._functorch.compile_utils import fx_graph_cse
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.mrope import (
    apply_interleaved_rope,
)

from ..inductor_pass import enable_fake_mode
from ..utility.noop_elimination import NoOpEliminationPass
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

P = ParamSpec("P")

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
    """Fuse gated QKV split + q/k Gemma RMSNorm + NeoX (M)RoPE on XPU."""

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

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("XPU QKV norm+RoPE fusion replaced %d sites", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            XpuGatedQkvNormRopePattern,
            _RopeSpec,
            repr(self._keys),
        )
