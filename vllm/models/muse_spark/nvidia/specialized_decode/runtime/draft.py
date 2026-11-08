# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode of the Muse Spark DSpark draft.

With ``decode_backend=latency_specialized`` the draft model's forward runs
``DraftDecode.run`` over its stock layers: the first layer's input norm is a
CuTe kernel (``cutedsl/draft/input_norm.py``) and each residual boundary
(post-norm, gated fp32 residual, next pre-norm) is one CuTe kernel
(``cutedsl/common/draft_boundary.py``), at TP4 and TP8 one with the branch's
all-reduce in it (``cutedsl/oproj/draft_all_reduce_boundary.py``). These are
bitwise equal to the stock eager layers; a boundary (or the first norm) the
kernels do not cover keeps its stock modules (``plan_fused_boundaries``). On 1
to 16 tokens each layer's QKV projection, QK-norm, RoPE and KV insert are the
QKV front CuTe kernel (``cutedsl/attention/qkv_front.py``), equal to the stock
layers up to the projection GEMM's summation order. At TP8 the o_proj and
down_proj partials run on the package GEMM (``cutedsl/gemm/unpooler.py``), which
publishes them into the boundary's mailbox: not bitwise to cuBLAS (summation
order).

The draft model builds it once its weights are loaded (``build_draft_decode``).
"""

import copy
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import torch
import torch.nn as nn

import vllm.models.muse_spark.nvidia.model as muse_model
from vllm.config import CompilationMode
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.llama import LlamaMLP
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
)
from vllm.platforms import current_platform

from ..cutedsl.common import policy
from . import cute_ops, flashinfer_ops

if TYPE_CHECKING:
    from vllm.models.muse_spark.common.dspark import (
        MuseSparkDSparkDecoderLayer,
        MuseSparkDSparkModel,
    )

    from ..cutedsl.common.draft_boundary import DraftBoundary, DraftBoundaryConstants
    from ..cutedsl.common.policy import UnpoolerShape
    from ..cutedsl.common.row_norm import RowNormConstants
    from ..cutedsl.draft.context import (
        ContextKvStore,
        ContextNormConstants,
        ContextNorms,
    )
    from ..cutedsl.draft.input_norm import InputNorm
    from ..cutedsl.gemm.unpooler import Unpooler
    from ..cutedsl.oproj.draft_all_reduce_boundary import BoundDraftBoundary

logger = init_logger(__name__)

# (post_norm, connection, next_norm) of one residual boundary.
_Boundary = tuple[nn.Module, nn.Module, nn.Module]


class _Partial(NamedTuple):
    """A residual branch before its all-reduce: this rank's partial and
    whether its GEMM has already published it into the boundary's mailbox."""

    tensor: torch.Tensor
    published: bool


# A branch's GEMM before its all-reduce: (RowParallelLinear, x) -> partial.
_Projection = Callable[[nn.Module, torch.Tensor], _Partial]
_T = TypeVar("_T")


def _norm_unsupported(norm: nn.Module, hidden_size: int) -> str | None:
    """Why the fused kernels do not take this norm as the stock one, or None:
    the stock MuseSpark1xRMSNorm forward over all ``hidden_size`` columns,
    its weight (if any) a contiguous ``[hidden_size]``."""
    if (
        not isinstance(norm, MuseSpark1xRMSNorm)
        or type(norm).forward is not MuseSpark1xRMSNorm.forward
    ):
        return f"a norm other than the stock MuseSpark1xRMSNorm ({type(norm).__name__})"
    if norm.variance_size_override is not None or norm.hidden_size != hidden_size:
        return f"a norm over other than all {hidden_size} columns"
    weight = norm.weight if norm.has_weight else None
    if weight is not None and (
        tuple(weight.shape) != (hidden_size,) or not weight.is_contiguous()
    ):
        return f"a norm weight other than a contiguous [{hidden_size}] tensor"
    return None


def _connection_unsupported(conn: nn.Module, hidden_size: int) -> str | None:
    """Why the fused boundary does not take this residual connection, or
    None: the stock norm-preserving connection with materialized contiguous
    FP32 ``[hidden_size]`` coefficients."""
    if (
        not isinstance(conn, MuseSpark1xNormPreservingResidualConnection)
        or type(conn).forward is not MuseSpark1xNormPreservingResidualConnection.forward
    ):
        return (
            "a residual connection other than the stock norm-preserving one "
            f"({type(conn).__name__})"
        )
    if not getattr(conn, "_coefficients_ready", False) or not all(
        isinstance(t, torch.Tensor)
        and t.dtype == torch.float32
        and tuple(t.shape) == (hidden_size,)
        and t.is_contiguous()
        for t in (getattr(conn, "alpha", None), getattr(conn, "beta", None))
    ):
        return (
            "residual coefficients other than materialized contiguous FP32 "
            f"[{hidden_size}] tensors"
        )
    return None


def _constants(
    build: Callable[..., _T], *args, **kwargs
) -> tuple[_T | None, str | None]:
    """``(build(*args, **kwargs), None)``, or ``(None, why)`` where it raises
    ValueError: a kernel's constants function refusing what the kernel does
    not reproduce."""
    try:
        return build(*args, **kwargs), None
    except ValueError as error:
        return None, str(error)


def _boundary_constants(
    boundary: _Boundary, hidden_size: int
) -> tuple["DraftBoundaryConstants | None", str | None]:
    """The constants of the CuTe boundary kernel of this boundary, or
    ``(None, why)`` where it does not reproduce it (the boundary keeps its
    stock modules)."""
    from ..cutedsl.common.draft_boundary import draft_boundary_constants

    post, conn, pre = boundary
    reason = (
        _norm_unsupported(post, hidden_size)
        or _norm_unsupported(pre, hidden_size)
        or _connection_unsupported(conn, hidden_size)
    )
    if reason is not None:
        return None, reason
    return _constants(draft_boundary_constants, *boundary)


def _layer_boundaries(
    model: "MuseSparkDSparkModel",
) -> list[tuple[nn.Module, tuple[_Boundary, _Boundary]]]:
    """Each layer with its two residual boundaries: after attention, and
    after the feed-forward into the next layer's (or the final) norm."""
    layers = list(model.layers)
    return [
        (
            layer,
            (
                (layer.post_attn_norm, layer.add_attn, layer.pre_ffn_norm),
                (
                    layer.post_ffn_norm,
                    layer.add_ffn,
                    layers[i + 1].self_attn.pre_qkv_proj_norm
                    if i + 1 < len(layers)
                    else model.norm,
                ),
            ),
        )
        for i, layer in enumerate(layers)
    ]


class DraftPlan(NamedTuple):
    """The draft's fused forward as planned at build (``plan_fused_boundaries``):
    every residual boundary with the constants of its CuTe kernel, and the
    first layer's input norm's. A None keeps that part's stock modules;
    ``refused`` says why, per part."""

    boundaries: list[_Boundary]
    constants: list["DraftBoundaryConstants | None"]
    input_norm: "RowNormConstants | None"
    refused: list[str]


def _draft_unsupported(model: "MuseSparkDSparkModel") -> str | None:
    """Why no part of the draft's forward runs fused (it keeps its stock
    layers), or None: it needs CUDA, the boundary kernels' hidden size (8192:
    the single-tile norm of fused_rmsnorm_fwd), and a bf16 compute and fp32
    residual dtype in every layer."""
    hidden_size = model.config.hidden_size
    if not current_platform.is_cuda():
        return "not CUDA"
    from ..cutedsl.common import draft_boundary

    if hidden_size != draft_boundary.H:
        return f"hidden size {hidden_size}, not {draft_boundary.H}"
    if model.default_dtype != torch.bfloat16:
        return f"compute dtype {model.default_dtype}, not bfloat16"
    if model.config.residual_dtype != torch.float32:
        return f"residual dtype {model.config.residual_dtype}, not float32"
    for i, layer in enumerate(model.layers):
        if (
            layer.residual_dtype != torch.float32
            or layer.default_dtype != model.default_dtype
        ):
            return f"layer {i}: dtypes other than the model's"
    return None


def plan_fused_boundaries(
    model: "MuseSparkDSparkModel",
) -> tuple[DraftPlan | None, str | None]:
    """``(plan, None)`` with every residual boundary that the CuTe boundary
    kernel reproduces as one kernel and the first norm on the CuTe input
    norm where it reproduces it; ``(None, why)`` where no part runs fused.

    A boundary runs fused with the stock norms and norm-preserving connection
    (materialized coefficients) and norms ``draft_boundary_constants``
    takes; the first norm where ``input_norm_constants`` takes it (its
    weight aligned). Any other part keeps its stock modules.
    """
    reason = _draft_unsupported(model)
    if reason is not None:
        return None, reason
    from ..cutedsl.common.host import pdl_supported
    from ..cutedsl.draft.input_norm import input_norm_constants

    hidden_size = model.config.hidden_size
    boundaries: list[_Boundary] = []
    constants: list[DraftBoundaryConstants | None] = []
    refused: list[str] = []
    for i, (_, pair) in enumerate(_layer_boundaries(model)):
        for kind, boundary in zip(("attention", "feed-forward"), pair):
            fused, why = _boundary_constants(boundary, hidden_size)
            if why is not None:
                refused.append(f"layer {i}'s {kind} boundary: {why}")
            boundaries.append(boundary)
            constants.append(fused)
    input_norm = None
    if model.layers:
        input_norm, why = _constants(
            input_norm_constants,
            model.layers[0].self_attn.pre_qkv_proj_norm,
            enable_pdl=pdl_supported(),
        )
        if why is not None:
            refused.append(f"the draft's first norm: {why}")
    if input_norm is None and not any(c is not None for c in constants):
        return None, "; ".join(refused) or "no layers"
    return DraftPlan(boundaries, constants, input_norm, refused), None


def _norm_params(norm: nn.Module) -> tuple:
    """What the fused norms read of a stock norm, besides its weight."""
    return (
        norm.has_weight,
        norm.weight.dtype if norm.has_weight else None,
        norm.variance_epsilon,
        norm.fused_gain_center,
        norm.applies_weight_before_norm and norm.has_weight,
    )


def _stock_rope(rotary_emb: nn.Module | None, head_dim: int) -> bool:
    """Whether the context keys' RoPE is the stock CUDA op on GPT-J pairs over
    the whole head, which the store kernel reproduces."""
    if rotary_emb is None:
        return True
    forward = getattr(rotary_emb, "_forward_method", None)
    return (
        type(rotary_emb).forward_cuda is RotaryEmbedding.forward_cuda
        and getattr(forward, "__func__", None) is RotaryEmbedding.forward_cuda
        and not flashinfer_ops.library_rope(rotary_emb)
        and not rotary_emb.is_neox_style
        and rotary_emb.rotary_dim == head_dim
        and rotary_emb.head_size == head_dim
    )


def _slot_groups(
    slot_mapping: torch.Tensor | list[torch.Tensor | None] | tuple,
    num_layers: int,
) -> list[tuple[int, list[torch.Tensor]]]:
    """The context store's calls: ``(first layer, slots of each layer)`` for
    every run of consecutive layers whose slot mappings have one shape. A
    layer without slots (None in a per-layer list) stores nothing."""
    if isinstance(slot_mapping, torch.Tensor):
        return [(0, [slot_mapping] * num_layers)]
    groups: list[tuple[int, list[torch.Tensor]]] = []
    for i, slots in enumerate(slot_mapping):
        if slots is None:
            continue
        if groups:
            first, run = groups[-1]
            if first + len(run) == i and run[0].shape == slots.shape:
                run.append(slots)
                continue
        groups.append((i, [slots]))
    return groups


class ContextKV:
    """``MuseSparkDSparkModel.precompute_and_store_context_kv`` as two CuTe
    kernels around its K/V projection (``cutedsl/draft/context.py``), bitwise
    equal to it: the hidden norm, the layers' input norms and the stack before
    the stock ``torch.bmm``; the K-norm, RoPE and every layer's cache insert
    after it. ``prepare`` binds and compiles them (at build, before CUDA graph
    capture)."""

    def __init__(
        self, model: "MuseSparkDSparkModel", constants: "ContextNormConstants"
    ) -> None:
        pre = model._context_pre_norms[0]
        self.model = model
        self.constants = constants
        self.hidden_norm = model.hidden_norm
        self.pre_norm = pre
        # The layers' input-norm weights in one [L, D] table, read by layer.
        self.pre_weights = (
            torch.stack([norm.weight for norm in model._context_pre_norms])
            if pre.has_weight
            else None
        )
        self.hidden_size = model.config.hidden_size
        self.num_layers = model._num_attn_layers
        self.num_kv_heads = model._num_kv_heads
        self.head_dim = model._head_dim
        self.eps = model._rms_norm_eps
        self.rotary_emb = model._rotary_emb
        self.attn_layers = model._attn_layers
        # The kernels bound to the model's norms and KV geometry (prepare).
        self.norms: ContextNorms | None = None
        self.kv_store: ContextKvStore | None = None

    def prepare(self) -> None:
        """Bind and compile the kernels every covered call launches; raise
        ValueError for a model whose rows or KV heads are not the kernels'
        (``context_kv_unsupported`` refuses them)."""
        from ..cutedsl.draft import context as cute_context

        if (self.hidden_size, self.head_dim) != (
            cute_context.CONTEXT_HIDDEN_SIZE,
            cute_context.HEAD_DIM,
        ):
            raise ValueError(
                "The fused context K/V kernels take "
                f"{cute_context.CONTEXT_HIDDEN_SIZE}-wide rows and "
                f"{cute_context.HEAD_DIM}-wide KV heads, not {self.hidden_size} "
                f"and {self.head_dim}"
            )
        hidden, pre = self.hidden_norm, self.pre_norm
        self.norms = cute_context.ContextNorms(
            hidden.weight if hidden.has_weight else None,
            self.pre_weights,
            self.num_layers,
            hidden.variance_epsilon,
            pre.variance_epsilon,
            device=self.model._stacked_kv_weight.device,
            hidden_gain_center=hidden.fused_gain_center,
            pre_gain_center=pre.fused_gain_center,
            hidden_input_scale=hidden.applies_weight_before_norm,
            pre_input_scale=pre.applies_weight_before_norm,
            constants=self.constants,
        )
        self.kv_store = cute_context.ContextKvStore(
            self.num_kv_heads, self.rotary_emb is not None, self.eps
        )

    def _project(self, context_states: torch.Tensor) -> torch.Tensor:
        """The stock K/V projection of the fused norms: ``[L, T, 2 * kv]``."""
        assert self.norms is not None, "ContextKV.prepare has not run"
        per_layer = self.norms(context_states)
        return torch.bmm(per_layer, self.model._stacked_kv_weight.transpose(1, 2))

    def store(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> bool:
        """The stock method's work, or False (nothing done) where the kernels
        do not cover the call. A strided ``context_states`` is not covered:
        the stock hidden norm takes its torch path for it, not the
        single-tile kernel the fused norms reproduce."""
        if not (
            context_states.dim() == 2
            and context_states.dtype == torch.bfloat16
            and context_states.shape[0] > 0
            and context_states.shape[1] == self.hidden_size
            and context_states.is_contiguous()
            and context_positions.dim() == 1
            and context_positions.dtype == torch.int64
            and context_positions.is_contiguous()
            and context_positions.shape[0] == context_states.shape[0]
        ):
            return False
        cos_sin_cache = None
        if self.rotary_emb is not None:
            # The stock RoPE call's table, with its side effect: the module's
            # table is cast to the keys' dtype and kept so (the context's own
            # copy under the QKV front, _own_context_rope).
            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(context_states)
            if not (
                cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
            ):
                return False
        if context_slot_mapping is None:
            # The stock method stores nothing without slots (the memory
            # profiling run), but the profile must see the chain's buffers.
            self._project(context_states)
            return True
        groups = _slot_groups(context_slot_mapping, self.num_layers)
        caches = []
        for attn in self.attn_layers:
            kv_cache = attn.kv_cache
            if not (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() == 4
                and kv_cache.dtype == torch.bfloat16
            ):
                return False
            # The stock KV insert's views (flashinfer_ops.bf16_kv_insert).
            k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
            if (
                v_cache.data_ptr() - k_cache.data_ptr()
                != self.head_dim * kv_cache.element_size()
                or k_cache.stride(3) != 1
                # Lanes move (2 j, 2 j + 1) pairs as 4-byte words.
                or k_cache.data_ptr() % 4
            ):
                return False
            caches.append(k_cache)
        if (
            any(
                c.shape != caches[0].shape or c.stride() != caches[0].stride()
                for c in caches
            )
            or any(stride % 2 for stride in caches[0].stride()[:3])
            or any(
                s.dim() != 1 or s.dtype != torch.int64 or not s.is_contiguous()
                for _, slots in groups
                for s in slots
            )
            # The store kernel takes a slot in 32 bits.
            or caches[0].shape[0] * caches[0].shape[1] >= 2**31
        ):
            return False

        all_kv = self._project(context_states)
        kv = all_kv.view(self.num_layers, -1, 2, self.num_kv_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        assert self.kv_store is not None
        for first, slots in groups:
            last = first + len(slots)
            self.kv_store(
                k[first:last],
                v[first:last],
                context_positions,
                slots,
                cos_sin_cache,
                caches[first:last],
            )
        return True


def _context_norm_constants(
    model: "MuseSparkDSparkModel",
) -> tuple["ContextNormConstants | None", str | None]:
    """The constants of the fused context chain's norms kernel where the
    chain is bitwise the stock one for this draft, else ``(None, why)`` (the
    draft keeps the stock method)."""
    if not hasattr(model, "_num_attn_layers"):
        return None, "no fused K/V buffers"
    from ..cutedsl.draft import context as cute_context

    hidden = model.config.hidden_size
    if hidden != cute_context.CONTEXT_HIDDEN_SIZE:
        return None, f"hidden size {hidden}, not {cute_context.CONTEXT_HIDDEN_SIZE}"
    if model.default_dtype != torch.bfloat16:
        return None, f"compute dtype {model.default_dtype}, not bfloat16"
    norms = model._context_pre_norms
    for norm in (model.hidden_norm, *norms):
        reason = _norm_unsupported(norm, hidden)
        if reason is not None:
            return None, reason
    if len({_norm_params(norm) for norm in norms}) != 1:
        return None, "layer input norms that differ in more than their weights"
    hidden_norm, pre = model.hidden_norm, norms[0]
    constants, reason = _constants(
        cute_context.context_norm_constants,
        hidden_norm.weight if hidden_norm.has_weight else None,
        pre.weight if pre.has_weight else None,
        hidden_input_scale=hidden_norm.applies_weight_before_norm,
        pre_input_scale=pre.applies_weight_before_norm,
    )
    if reason is not None:
        return None, reason
    if model._head_dim != cute_context.HEAD_DIM:
        return None, f"{model._head_dim}-wide KV heads, not {cute_context.HEAD_DIM}"
    if model._num_kv_heads & (model._num_kv_heads - 1):
        return None, f"{model._num_kv_heads} KV heads, not a power of two"
    if model._stacked_kv_weight.dtype != torch.bfloat16:
        return None, "a K/V projection weight other than bfloat16"
    if not _stock_rope(model._rotary_emb, model._head_dim):
        return None, "a RoPE other than the CUDA GPT-J op over the whole head"
    if not all(
        flashinfer_ops.bf16_kv_insert(attn, model._num_kv_heads, model._head_dim)
        for attn in model._attn_layers
    ):
        return None, "a KV insert other than the attention backend's bf16 cache"
    return constants, None


def context_kv_unsupported(model: "MuseSparkDSparkModel") -> str | None:
    """Why the fused context K/V chain is not bitwise the stock one for this
    draft (the draft keeps the stock method), or None."""
    return _context_norm_constants(model)[1]


def plan_context_kv(
    model: "MuseSparkDSparkModel",
) -> tuple[ContextKV | None, str | None]:
    """``(chain, None)``: the fused context K/V chain, bound to its norms
    kernel's constants, where it is bitwise the stock one; else ``(None,
    why)``."""
    constants, reason = _context_norm_constants(model)
    if constants is None:
        return None, reason
    return ContextKV(model, constants), None


def may_fuse_qkv_front(attn: nn.Module) -> bool:
    """Whether this draft layer's QKV projection, QK-norm, RoPE and KV insert
    may run as one kernel (the QKV front) on 1 to 16 tokens.

    The deployment checks (SM100, no LoRA, no batch-invariant mode) are the
    specialized decode's own (``unsupported``); its draft adds the kernel's.
    """
    return (
        attn.fused_qk_norm_rope_kv_insert
        and attn.use_o_gate
        and attn.head_dim == cute_ops.HEAD_DIM
        and attn.default_dtype == torch.bfloat16
        and attn.attn.kv_cache_dtype in ("auto", "bfloat16")
        and cute_ops.unsupported_projection(attn) is None
    )


def _fused_qkv_projection(
    layer: "MuseSparkDSparkDecoderLayer",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    """The projection with QK-norm/RoPE applied and K/V inserted, or None
    when the call is outside what the fused kernel covers."""
    attn = layer.self_attn
    _, _, kv_cache, slot_mapping = get_attention_context(attn.attn.layer_name)
    if slot_mapping is None or kv_cache.numel() == 0:
        return None
    cos_sin_cache = None
    if attn.use_rope:
        assert attn.rotary_emb is not None
        cos_sin_cache = attn.rotary_emb.cos_sin_cache
    weight = attn.qkv_proj.weight
    if not cute_ops.supports_call(
        hidden_states,
        weight,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        attn.num_kv_heads,
    ):
        return None
    # Allocated only for a call the front takes; the checked launch also
    # checks this output.
    out = hidden_states.new_empty((hidden_states.shape[0], weight.shape[0]))
    cute_ops.qkv_front(
        hidden_states,
        weight,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_cache,
        out,
        attn.rms_norm_eps,
    )
    return out


def _partial(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A plain RowParallelLinear's output before its all-reduce: the GEMM its
    forward runs (``_plain_row_parallel``)."""
    return linear.quant_method.apply(linear, x, None)


def _plain_row_parallel(linear: nn.Module | None, tp: int, width: int) -> bool:
    """Whether ``linear`` is the stock RowParallelLinear forward over a bf16
    GEMM without bias into ``width`` features, whose output is all-reduced
    over the TP group."""
    return (
        isinstance(linear, RowParallelLinear)
        and type(linear).forward is RowParallelLinear.forward
        and isinstance(linear.quant_method, UnquantizedLinearMethod)
        and linear.input_is_parallel
        and linear.reduce_results
        and linear.bias is None
        and linear.tp_size == tp
        and linear.weight.dtype == torch.bfloat16
        and linear.output_size == width
    )


def _ffn_partial(mlp: nn.Module, x: torch.Tensor, project: _Projection) -> _Partial:
    """``LlamaMLP.forward`` without the down projection's all-reduce, the down
    projection run by ``project``."""
    x, _ = mlp.gate_up_proj(x)
    x = maybe_fused_act_quant(mlp.act_fn, x, mlp.down_proj)
    return project(mlp.down_proj, x)


def _package_gemm_weight(
    linear: nn.Module, shape: "UnpoolerShape"
) -> torch.Tensor | None:
    """The weight of a plain row-parallel draft projection where the package
    GEMM runs it at launch shape ``shape`` (on the GPU, every feature of the
    rows the boundary's mailbox holds, and nothing ``Unpooler.unsupported``
    refuses), else None: cuBLAS."""
    weight = linear.weight
    if not weight.is_cuda:
        return None
    from ..cutedsl.gemm.unpooler import Unpooler
    from ..cutedsl.oproj import draft_all_reduce_boundary

    if (
        Unpooler.unsupported(weight, shape) is None
        and weight.shape[0] == draft_all_reduce_boundary.H
    ):
        return weight
    return None


def all_reduce_unsupported(model: "MuseSparkDSparkModel") -> str | None:
    """Why the draft's residual boundaries keep the stock all-reduce of their
    branch, or None when every one takes it fused (``DraftAllReduce``).

    Reads the configuration only, the same on every rank: the fused form
    builds a mailbox collectively.
    """
    if not model_parallel_is_initialized():
        return "no TP group"
    tp = get_tensor_model_parallel_world_size()
    if tp not in policy.FUSED_TPS:
        return f"TP{tp} (the target's attention boundary all-reduces unfused)"
    from ..cutedsl.oproj import draft_all_reduce_boundary

    # The fused boundary's rows: every feature of the branch.
    width = draft_all_reduce_boundary.H
    for layer in model.layers:
        attn, mlp = layer.self_attn, layer.feed_forward
        if not (
            isinstance(attn, muse_model.MuseSpark1xAttention)
            and type(attn)._project_output
            is muse_model.MuseSpark1xAttention._project_output
            and not attn.is_sequence_parallel_output
            and _plain_row_parallel(attn.o_proj, tp, width)
        ):
            return "an attention output projection other than the stock one"
        if not (
            isinstance(mlp, LlamaMLP)
            and type(mlp).forward is LlamaMLP.forward
            and _plain_row_parallel(mlp.down_proj, tp, width)
        ):
            return "a feed-forward other than the stock MLP"
    return None


def all_reduce_boundaries_unsupported(plan: DraftPlan) -> str | None:
    """Why the fused all-reduce boundary does not take every boundary of the
    plan (the draft all-reduces its branches unfused), or None."""
    from ..cutedsl.oproj.draft_all_reduce_boundary import boundary_unsupported

    for index, constants in enumerate(plan.constants):
        reason = (
            "it keeps its stock modules"
            if constants is None
            else boundary_unsupported(constants)
        )
        if reason is not None:
            return f"boundary {index}: {reason}"
    return None


class DraftAllReduce:
    """The draft's residual boundaries with their branch's all-reduce fused in
    (TP4 and TP8, as the target's attention boundary).

    Each boundary is one CuTe kernel (``DraftAllReduceBoundary``): every rank
    multicasts its branch partial into a Lamport mailbox over the TP group, sums
    the ranks' rows in the order of the stock all-reduce (the backend it
    resolved to, ``flashinfer_ops.stock_sum_order``, as the target's attention
    boundary) and runs the boundary on the sum, bit for bit the stock
    all-reduce followed by the boundary. Construction is collective over the TP
    group; ``bind_gemms`` and then ``prepare`` set up every boundary before CUDA
    graph capture.
    """

    def __init__(self, device: torch.device) -> None:
        from ..cutedsl.oproj.all_reduce_dual_norm import fused_mailbox
        from ..cutedsl.oproj.draft_all_reduce_boundary import DraftAllReduceBoundary

        tp = get_tp_group()
        self.tp = tp.world_size
        if self.tp not in policy.FUSED_TPS:
            raise ValueError(f"TP{self.tp} does not fuse the draft's all-reduce")
        sum_order = flashinfer_ops.stock_sum_order(tp.device_communicator)
        self.fp32_adds = sum_order == policy.FP32_ADDS
        self.mailbox = fused_mailbox(
            tp.device_group, tp.cpu_group, sum_order=sum_order, device=device
        )
        self.op = DraftAllReduceBoundary(
            self.mailbox.tensor,
            self.mailbox.multicast,
            tp.rank_in_group,
            sum_order=sum_order,
        )
        # Per boundary (bind_gemms): the package GEMM that publishes its branch
        # partial into the mailbox, or None (cuBLAS; the boundary publishes).
        self.gemms: list[Unpooler | None] = []
        # Per boundary (prepare): the kernel bound to its constants and its
        # gates' and norm weights' views, its launched variants compiled.
        self.bound: list[BoundDraftBoundary] = []

    def bind_gemms(self, layers: list[nn.Module]) -> int:
        """At ``policy.PACKAGE_GEMM_TPS``, run each boundary's branch GEMM (a
        layer's o_proj, then its down_proj) on the package GEMM at its role's
        launch shape (``DRAFT_GEMMS``), whose epilogue multicasts the partial
        into the mailbox; its boundary then only consumes it. Returns how many
        boundaries do. Call before ``prepare``."""
        from ..cutedsl.gemm.unpooler import Unpooler

        package = self.tp in policy.PACKAGE_GEMM_TPS
        self.gemms = []
        for layer in layers:
            for role, linear in (
                ("o_proj", layer.self_attn.o_proj),
                ("down_proj", layer.feed_forward.down_proj),
            ):
                shape = policy.DRAFT_GEMMS[role]
                weight = _package_gemm_weight(linear, shape) if package else None
                self.gemms.append(
                    None
                    if weight is None
                    else Unpooler(
                        weight, self.mailbox, shape=shape, ticket=self.op.ticket
                    )
                )
        return sum(gemm is not None for gemm in self.gemms)

    def prepare(
        self,
        boundaries: list[_Boundary],
        constants: list["DraftBoundaryConstants"],
    ) -> None:
        """Bind the draft's boundaries (one per ``bind_gemms`` GEMM) to their
        constants (``all_reduce_boundaries_unsupported`` takes them) and
        compile the kernel variants ``DraftDecode.run`` launches (the first
        boundary also on a BF16 stream, for a BF16 embedding)."""
        assert len(boundaries) == len(constants) == len(self.gemms)
        self.bound = []
        for index, ((post, conn, pre), kind, gemm) in enumerate(
            zip(boundaries, constants, self.gemms)
        ):
            bound = self.op.bind(
                kind,
                conn.alpha,
                conn.beta,
                post.weight if post.has_weight else None,
                pre.weight if pre.has_weight else None,
            )
            store = index + 1 < len(boundaries)
            published = gemm is not None
            bound.prepare(False, store, published)
            if index == 0:
                bound.prepare(True, store, published)
            self.bound.append(bound)

    def takes(self, partial: torch.Tensor) -> bool:
        """Whether this call fuses the all-reduce: on the row count alone, the
        same on every rank (the others would wait for this rank's rows)."""
        return partial.dim() == 2 and 1 <= partial.shape[0] <= self.mailbox.capacity

    def project(self, index: int, linear: nn.Module, x: torch.Tensor) -> _Partial:
        """Boundary ``index``'s branch partial ``x @ linear.weight^T``: on its
        package GEMM, which also publishes it, where it has one and the call
        fuses the all-reduce, else the stock GEMM."""
        gemm = self.gemms[index]
        if gemm is None or not self.takes(x):
            return _Partial(_partial(linear, x), False)
        x = x.contiguous()
        return _Partial(gemm.run(x, x.new_empty((x.shape[0], gemm.n))), True)

    def run(
        self,
        index: int,
        partial: _Partial,
        stream: torch.Tensor,
        store_stream: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``DraftBoundary.run`` of boundary ``index`` (``prepare``) on
        the all-reduced ``partial``; where its GEMM published it, the kernel
        only consumes."""
        return self.bound[index].run(
            partial.tensor, stream.contiguous(), store_stream, partial.published
        )


def _attention(
    layer: "MuseSparkDSparkDecoderLayer",
    fused_qkv: bool,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    project: _Projection | None = None,
) -> torch.Tensor | _Partial:
    """Attention over the normalized input: the fused QKV front where it
    covers the call, else the stock attention module. Returns the all-reduced
    output; with ``project`` the fused front's output is this rank's o_proj
    partial instead, from ``project(o_proj, x)``."""
    qkv = None
    if fused_qkv:
        qkv = _fused_qkv_projection(layer, positions, hidden_states)
    if qkv is None:
        return layer.self_attn(positions=positions, hidden_states=hidden_states)
    attn = layer.self_attn
    q, _, _, og = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    # The front inserted K/V; the attention layer must not rewrite them.
    attn_output = attn.attn(q, None, None, output_dtype=qkv.dtype)
    # The o-norm and sigmoid gate of MuseSpark1xAttention.forward.
    attn_output = attn_output.reshape(-1, attn.num_heads, attn.head_dim)
    og = og.reshape(-1, attn.num_heads, attn.head_dim)
    # (The front needs the o-gate, so the o-norm is fused with it if set.)
    if attn.cute_o_norm_gate:
        assert muse_model.o_norm_sigmoid_gate is not None
        muse_model.o_norm_sigmoid_gate(
            attn_output, og, attn.rms_norm_eps, enable_pdl=True
        )
    elif attn.fuse_o_norm_gate:
        attn_output = muse_model.rms_norm_gated(
            attn_output, og, None, None, "sigmoid", eps=attn.rms_norm_eps
        )
    else:
        attn_output = og.sigmoid() * attn_output
    attn_output = attn_output.view(-1, attn.num_heads * attn.head_dim)
    if project is not None:
        return project(attn.o_proj, attn_output)
    return attn._project_output(attn_output)


class DraftDecode:
    """The forward of the DSpark draft's layers with the fused boundaries and,
    per layer, the fused QKV front, as planned (``DraftPlan``: a part the
    kernels do not reproduce runs its stock modules); with ``all_reduce``,
    each boundary also takes its branch's all-reduce. ``prepare`` binds the
    first layer's input norm and the boundaries' kernels before CUDA graph
    capture."""

    def __init__(
        self,
        model: "MuseSparkDSparkModel",
        plan: DraftPlan,
        fused_qkv: list[bool],
        context: ContextKV | None = None,
        all_reduce: DraftAllReduce | None = None,
    ) -> None:
        self.layers = list(model.layers)
        self.default_dtype = model.default_dtype
        self.plan = plan
        self.boundaries = plan.boundaries
        self.fused_qkv = fused_qkv
        self.context = context
        self.all_reduce = all_reduce
        # Per boundary (prepare): the CuTe boundary of the all-reduced branch,
        # or None (its stock modules).
        self.fused: list[DraftBoundary | None] = []
        # The first layer's input norm (prepare), or None (the stock norm).
        self.input_norm: InputNorm | None = None

    def prepare(self) -> None:
        """Bind the first layer's input norm and every planned boundary's
        kernel to their constants and compile the variants ``run`` launches:
        the QKV front of every layer on it, the first boundary also on a BF16
        stream (a BF16 embedding), every boundary for both row regimes of
        ``DraftBoundary.late_rows``."""
        from ..cutedsl.common.draft_boundary import DraftBoundary
        from ..cutedsl.draft.input_norm import InputNorm

        for fused_qkv, layer in zip(self.fused_qkv, self.layers):
            if fused_qkv:
                cute_ops.prepare_qkv_front(layer.self_attn.qkv_proj.weight.shape[0])
        plan = self.plan
        if plan.input_norm is not None:
            # InputNorm compiles its kernel at construction.
            self.input_norm = InputNorm(
                self.layers[0].self_attn.pre_qkv_proj_norm, plan.input_norm
            )

        self.fused = []
        for index, ((post, conn, pre), constants) in enumerate(
            zip(self.boundaries, plan.constants)
        ):
            if constants is None:
                self.fused.append(None)
                continue
            fused = DraftBoundary(
                constants,
                conn.alpha,
                conn.beta,
                post.weight if post.has_weight else None,
                pre.weight if pre.has_weight else None,
            )
            store = index + 1 < len(self.boundaries)
            for late in (False, True):
                fused.prepare(False, store, late)
                if index == 0:
                    fused.prepare(True, store, late)
            self.fused.append(fused)

    def store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> bool:
        """``MuseSparkDSparkModel.precompute_and_store_context_kv`` with the
        fused context K/V chain; False where it does not cover the call (the
        stock method runs)."""
        return self.context is not None and self.context.store(
            context_states, context_positions, context_slot_mapping
        )

    def _residual_boundary(
        self,
        branch: torch.Tensor | _Partial,
        stream: torch.Tensor,
        index: int,
        store_stream: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Boundary ``index`` of ``branch``, or of its all-reduce where it is
        this rank's partial."""
        post, conn, pre = self.boundaries[index]
        if isinstance(branch, _Partial):
            assert self.all_reduce is not None
            if self.all_reduce.takes(branch.tensor):
                return self.all_reduce.run(index, branch, stream, store_stream)
            branch = tensor_model_parallel_all_reduce(branch.tensor)
        fused = self.fused[index]
        if fused is None or not fused.takes(branch, stream):
            new_stream = conn(stream=stream.to(torch.float32), branch=post(branch))
            return new_stream, pre(
                new_stream.to(self.default_dtype), out_dtype=self.default_dtype
            )
        return fused.launch(branch, stream, store_stream)

    def _projection(self, index: int) -> _Projection | None:
        """Boundary ``index``'s branch GEMM before the all-reduce, or None
        (the branch is all-reduced unfused)."""
        if self.all_reduce is None:
            return None
        return functools.partial(self.all_reduce.project, index)

    def run(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """``MuseSparkDSparkModel.forward`` after the embedding: the layers
        and the final norm."""
        # A bf16 embedding round-trips through the fp32 stream unchanged, so the
        # first norm and the first boundary read it directly.
        if hidden_states.dtype == self.default_dtype:
            stream = x = hidden_states
        else:
            stream = hidden_states.to(torch.float32)
            x = stream.to(self.default_dtype)
        num_layers = len(self.layers)
        assert len(self.fused) == len(self.boundaries), (
            "DraftDecode.prepare has not run"
        )
        # The first layer's pre_qkv_proj_norm.
        if self.input_norm is not None:
            hidden_states = self.input_norm.run(x.contiguous())
        else:
            hidden_states = self.layers[0].self_attn.pre_qkv_proj_norm(
                x.contiguous(), out_dtype=self.default_dtype
            )
        for i, layer in enumerate(self.layers):
            attn_out = _attention(
                layer,
                self.fused_qkv[i],
                positions,
                hidden_states,
                self._projection(2 * i),
            )
            stream, ffn_in = self._residual_boundary(attn_out, stream, 2 * i, True)
            project = self._projection(2 * i + 1)
            ffn_out = (
                layer.feed_forward(ffn_in)
                if project is None
                else _ffn_partial(layer.feed_forward, ffn_in, project)
            )
            stream, hidden_states = self._residual_boundary(
                ffn_out, stream, 2 * i + 1, i + 1 < num_layers
            )
        return hidden_states


def _own_context_rope(model: "MuseSparkDSparkModel", fused_qkv: list[bool]) -> None:
    """Give the context keys their own copy of the first layer's RoPE, its
    table already in the keys' dtype, where a layer on the QKV front shares it.

    The context keys are rotated with the first layer's RoPE, and that call
    casts the module's table to bf16 for good, which would make the first layer
    fail the front's fp32-table check. The copy holds the same cast, so the
    context cache is unchanged and the first layer keeps its fp32 table.
    """
    rope = model._rotary_emb
    if rope is None or not any(
        fused and layer.self_attn.rotary_emb is rope
        for fused, layer in zip(fused_qkv, model.layers)
    ):
        return
    table = rope.cos_sin_cache
    # The copy shares nothing with the layer's module and holds only the cast.
    model._rotary_emb = copy.deepcopy(rope, {id(table): table.to(model.default_dtype)})


def build_draft_decode(
    model: "MuseSparkDSparkModel", compilation_mode: CompilationMode
) -> DraftDecode | None:
    """The draft's specialized decode once its weights are loaded, or None
    where the draft keeps its stock layers (and says why)."""
    if compilation_mode != CompilationMode.NONE:
        # The fused kernels are exact to the eager layers, not to an
        # Inductor-compiled draft, and the QKV front's call checks would be
        # traced: the specialized decode refuses compiled deployments.
        raise ValueError(
            "The Muse Spark specialized decode runs the DSpark draft "
            f"uncompiled, not under compilation mode {compilation_mode}"
        )
    plan, reason = plan_fused_boundaries(model)
    if plan is None:
        logger.warning_once(
            "The Muse Spark DSpark draft keeps its stock layers: %s.", reason
        )
        return None
    boundaries = plan.boundaries
    for refused in plan.refused:
        logger.warning_once(
            "The Muse Spark DSpark draft keeps the stock modules of %s.", refused
        )
    logger.info_once(
        "Using the fused DSpark draft residual boundary CuTe kernel "
        "(%d of %d boundaries).",
        sum(c is not None for c in plan.constants),
        len(boundaries),
        scope="process",
    )
    has_front = cute_ops.qkv_front_available()
    fused_qkv = [
        has_front and may_fuse_qkv_front(layer.self_attn) for layer in model.layers
    ]
    # Before the context chain is planned: it reads the context keys' RoPE.
    _own_context_rope(model, fused_qkv)
    context, reason = plan_context_kv(model)
    if context is None:
        logger.warning_once(
            "The Muse Spark DSpark draft keeps the unfused context K/V chain: %s.",
            reason,
        )
    else:
        context.prepare()
        logger.info_once(
            "Using the fused DSpark draft context K/V kernels (CuTe).",
            scope="process",
        )
    if not has_front:
        logger.warning_once(
            "The Muse Spark DSpark draft keeps the unfused QKV projection: this "
            "GPU cannot run %s",
            cute_ops.QKV_FRONT_KERNEL,
        )
    else:
        logger.info_once(
            "The Muse Spark DSpark draft runs the fused QKV projection + "
            "QK-norm/RoPE/KV-cache insert kernel in %d of its %d layers.",
            sum(fused_qkv),
            len(fused_qkv),
            scope="process",
        )
    reason = all_reduce_unsupported(model) or all_reduce_boundaries_unsupported(plan)
    all_reduce = None
    if reason is not None:
        logger.info_once(
            "The Muse Spark DSpark draft all-reduces its residual branches "
            "unfused: %s.",
            reason,
        )
    else:
        # Collective: every rank plans the same layers.
        all_reduce = DraftAllReduce(model.layers[0].self_attn.o_proj.weight.device)
        gemms = all_reduce.bind_gemms(list(model.layers))
        all_reduce.prepare(boundaries, [c for c in plan.constants if c is not None])
        logger.info_once(
            "The Muse Spark DSpark draft fuses the all-reduce of its residual "
            "branches into its CuTe boundary kernel (%s sum).",
            "fp32" if all_reduce.fp32_adds else "bf16",
            scope="process",
        )
        if gemms:
            logger.info_once(
                "The Muse Spark DSpark draft at TP%d runs %d of its %d branch "
                "GEMMs on the package GEMM, partials published zero-copy into "
                "its boundary mailbox.",
                all_reduce.tp,
                gemms,
                len(boundaries),
                scope="process",
            )
    draft = DraftDecode(model, plan, fused_qkv, context, all_reduce)
    draft.prepare()
    return draft
