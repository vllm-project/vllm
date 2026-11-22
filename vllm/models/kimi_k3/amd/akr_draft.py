# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's amd_kernels routes (draft route) for a Kimi-K3 target. Each
piece reads its modules and shapes from the draft, so both Kimi-K3.1 drafts take
them: the preview's MLA draft (`K3DSparkForCausalLM`) and the release's GQA
sliding-window draft (`Qwen3DSparkForCausalLM`, a `DFlashQwen3Model` backbone).

* `CTXPROJ`  `gemm/bf16_common.bf16_linear_gather`: the context projection (35840 ->
  7168, 514 MB replicated GEMM) + its RMSNorm as a sharded GEMM (this rank's columns,
  64 MB at TP8) + gather over the TP ranks + RMSNorm in two launches. The target aux
  hidden states are packed into the kernel's tile layout (`linear_gather_pack_aux`)
  instead of `torch.cat`. Rows <= 256 (the whole decode graph's context); the
  original weight stays for larger prefill chunks. Extra memory: the packed shard
  64 MB + slabs 29 MB per rank. The preview's `context_proj` + `context_norm` (its
  `combine_hidden_states` norms); the release's `fc` + `hidden_norm`, whose stock
  norm runs in `precompute_and_store_context_kv` (`_normalize_context_states`): the
  stored rows are normed here instead (by the kernel, or by that stock norm for rows
  it does not take) and the precompute's norm becomes the identity. Off when
  `VLLM_K31_BEAT_DRAFTFC` already split `fc` over TP.
* `DMLP`     `decoder/bf16_dense_mlp_common.bf16_dense_mlp`: every layer's dense MLP
  (gate_up + SwiGLU, down) through the op `vllm.kimi_akr_draft_dense_mlp` (opaque to
  torch.compile); the weights are replaced in place by their packed layout (no extra
  memory). An MLP whose down projection all-reduces (the release's `Qwen2MLP`) keeps
  that all-reduce after the op. Needs `max_num_seqs` x the query rows a request (k)
  <= 256 (checked at set-up; the stock weights are gone).
* `DAR`      the all-reduce + residual add + RMSNorm pairs. The preview: AITER's fused
  op (`fused_allreduce_rms_norm`). The release (uncompiled under breakable CUDA graphs,
  so vLLM's fusion pass never sees it): `norm/attnres_rmsnorm_common`
  `allreduce_attnres_rmsnorm` without bank (`AkrArNorm`) on the o_proj -> post-attention
  norm and MLP -> next input norm / final norm pairs (2 a layer), on the target's
  communicator (else one of 14.7 MiB); calls past `AR_MAX_ROWS` all-reduce first.
  A compiled backbone keeps the stock pairs.
* `DMLA`     the dense MLA decode (`akr_draft_mla`): MLA drafts only.
* `DQKV`, `DCTXKV` the GQA draft's q / k norm + RoPE + KV insert of the query block
  and of the context rows (`akr_draft_gqa`): the release only.
* the lm_head all-gather exchange on `compute_draft_logits` (`akr_lmhead`, switch
  `LMHEAD`) and the fused Markov chain (`akr_sampling`, switch `MARKOV`).

Each piece is checked at set-up with its entry's `*_unsupported_reason` (communicator,
weights, the largest call it takes) and stays on the stock path otherwise.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import (
    akr_config,
    akr_lmhead,
    akr_nonlayer,
    akr_sampling,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

CTX_MAX_ROWS = 256  # rows of the staging buffers (the decode graph's context)
MLP_MAX_ROWS = 256
AR_MAX_ROWS = 152  # all-reduce + norm: a draft step loses 0.5 us at 156 rows, 2 at 160


def aiter_ar() -> bool:
    """Whether the drafter's all-reduce + RMSNorm takes AITER's fused op."""
    return akr_nonlayer.switch("DAR")


def compute_logits(model, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """`K3DSparkForCausalLM.compute_logits` on the lm_head exchange, or None (stock)."""
    if not getattr(model, "_akr_lmhead", False):
        return None
    return akr_lmhead.logits("draft", model.lm_head, hidden_states)


def _setup_lmhead(model, vllm_config) -> bool:
    """Collective: the draft's exchange. The preview's `compute_logits` reads its
    `_akr_lmhead` flag; another draft gets its `compute_draft_logits` wrapped."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(model, "compute_draft_logits"):
        return False
    ok = akr_lmhead.setup("draft", lm_head, model.logits_processor, vllm_config)
    if hasattr(model, "_akr_lmhead"):
        model._akr_lmhead = ok
    elif ok:
        stock = model.compute_draft_logits

        def compute_draft_logits(hidden_states: torch.Tensor) -> torch.Tensor:
            out = akr_lmhead.logits("draft", model.lm_head, hidden_states)
            return stock(hidden_states) if out is None else out

        model.compute_draft_logits = compute_draft_logits
    return ok


def _gemm():
    from amd_kernels.gemm import bf16_common

    return bf16_common


# ------------------------------------------------------------------ context_proj
def _ctx_layout(model) -> tuple[Any, Any, bool] | str:
    """(projection, norm, the norm runs in the precompute) of the draft's context path,
    or why the route does not know it."""
    m = model.model
    if hasattr(m, "context_proj") and hasattr(m, "context_norm"):
        return m.context_proj, m.context_norm, False
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    if isinstance(m, DFlashQwen3Model) and getattr(m, "use_aux_hidden_state", False):
        return m.fc, m.hidden_norm, True
    return f"no context projection layout known for {type(m).__name__}"


def _ctx_modules(model) -> tuple[Any, Any, bool]:
    """`_ctx_layout` of a draft whose layout is known."""
    layout = _ctx_layout(model)
    assert not isinstance(layout, str), layout
    return layout


def _ctx_dims(model) -> tuple[int, int, int]:
    """(rank, N shard, K) of this rank's rows of the context projection."""
    g = akr_nonlayer.tp_group()
    proj, _, _ = _ctx_modules(model)
    n, k = proj.weight.data.shape
    return g.rank_in_group, n // g.world_size, k


class ContextProj:
    def __init__(self, model, device: torch.device):
        """Collective on the TP group (the communicator)."""
        lib = _gemm()
        self.lib = lib
        r, n_shard, k = _ctx_dims(model)
        proj, norm, self.norm_in_precompute = _ctx_modules(model)
        w = proj.weight.data
        self.w_packed = lib.linear_gather_pack_weight(
            w[n_shard * r : n_shard * (r + 1)].contiguous()
        )
        g = akr_nonlayer.tp_group()
        self.comm = lib.linear_gather_comm(r, g.cpu_group, device, n_shard=n_shard, k=k)
        self.hidden, self.k = w.shape
        self.num_aux = self.k // self.hidden
        self.norm_weight = norm.weight
        self.eps = float(norm.variance_epsilon)
        # rows the kernel does not take: normed as the kernel's (the precompute's
        # stock norm for the release; the preview's combine already norms)
        self.post_norm: Callable[[torch.Tensor], torch.Tensor] = (
            model.model._normalize_context_states
            if self.norm_in_precompute
            else _identity
        )
        self._model = model.model
        n = self._numel(CTX_MAX_ROWS)
        # zero once: rows past a call's rows keep their (finite) previous content,
        # as the stock staging does
        self.graph_stage = torch.zeros(n, dtype=torch.bfloat16, device=device)
        self.tmp_stage = torch.zeros(n, dtype=torch.bfloat16, device=device)

    def _numel(self, rows: int) -> int:
        tiles = self.lib.linear_gather_packed_tiles(rows, w_packed=self.w_packed)
        return tiles * (self.k // 32) * 512

    def _view(self, stage: torch.Tensor, rows: int) -> torch.Tensor:
        return stage[: self._numel(rows)]

    def _x(self, stage: torch.Tensor, rows: int) -> torch.Tensor:
        return self._view(stage, rows).view(-1, self.k // 32, 64, 8)

    def _pack(self, aux: list[torch.Tensor], stage: torch.Tensor, rows: int) -> None:
        self.lib.linear_gather_pack_aux(
            aux, self._view(stage, rows), rows, w_packed=self.w_packed
        )

    def _gather(self, stage: torch.Tensor, rows: int, out: torch.Tensor) -> None:
        self.lib.bf16_linear_gather(
            self.comm,
            self._x(stage, rows),
            self.w_packed,
            self.norm_weight,
            rows,
            self.eps,
            out=out,
        )

    def pack_graph(self, aux: list[torch.Tensor], rows: int) -> None:
        """Eager, before the graph replay: the graph's first `rows` context rows."""
        self._pack([h[:rows] for h in aux], self.graph_stage, rows)

    def combine_graph(self, rows: int, out: torch.Tensor) -> None:
        """Inside the FULL graph: out = context_norm(context_proj(staged rows))."""
        self._gather(self.graph_stage, rows, out)

    def combine_eager(
        self, aux: list[torch.Tensor], start: int, end: int, out: torch.Tensor
    ) -> None:
        rows = end - start
        self._pack([h[start:end] for h in aux], self.tmp_stage, rows)
        self._gather(self.tmp_stage, rows, out)

    def reason(self, device: torch.device) -> str | None:
        """The entries' reasons for the largest call (CTX_MAX_ROWS rows)."""
        rows = CTX_MAX_ROWS
        aux = [
            torch.empty(rows, self.hidden, dtype=torch.bfloat16, device=device)
            for _ in range(self.num_aux)
        ]
        out = torch.empty(rows, self.hidden, dtype=torch.bfloat16, device=device)
        return self.lib.linear_gather_pack_aux_unsupported_reason(
            aux, self._view(self.tmp_stage, rows), rows, w_packed=self.w_packed
        ) or self.lib.bf16_linear_gather_unsupported_reason(
            self.comm,
            self._x(self.tmp_stage, rows),
            self.w_packed,
            self.norm_weight,
            rows,
            self.eps,
            out=out,
        )

    def install(self) -> None:
        """The release: the precompute's context norm becomes the identity (every stored
        row is normed by `store_hidden_states` / `prepare_graph_forward`)."""
        if self.norm_in_precompute:
            self._model._normalize_context_states = _identity

    def warm_up(self, device: torch.device) -> None:
        """Collective: compile every tier of pack_aux and the projection."""
        aux = [
            torch.zeros(CTX_MAX_ROWS, self.hidden, dtype=torch.bfloat16, device=device)
            for _ in range(self.num_aux)
        ]
        out = torch.empty(
            CTX_MAX_ROWS, self.hidden, dtype=torch.bfloat16, device=device
        )
        for rows in (16, CTX_MAX_ROWS):
            self.combine_eager(aux, 0, rows, out[:rows])
        torch.accelerator.synchronize(device)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _ctx_reason(spec, model) -> str | None:
    from vllm.model_executor.layers.linear import ReplicatedLinear

    layout = _ctx_layout(model)
    if isinstance(layout, str):
        return layout
    proj, norm, norm_in_precompute = layout
    if not isinstance(proj, ReplicatedLinear):
        # e.g. VLLM_K31_BEAT_DRAFTFC's TP-split fc (latency_specialized), which wins
        return f"the context projection is a {type(proj).__name__}"
    if proj.weight.dtype != torch.bfloat16 or proj.bias is not None:
        return "the context projection is not a bf16 weight without bias"
    n, k = proj.weight.shape
    if k % n:
        return f"context projection {n} x {k} is not a concat of {n}-wide aux states"
    if spec._aux_staging is None:
        return "no FULL-graph aux staging (decode cudagraph mode is not FULL)"
    if norm.weight.dtype != torch.bfloat16 or abs(norm.variance_epsilon - 1e-5) > 1e-12:
        return "the context norm is not bf16 / eps 1e-5"
    if norm_in_precompute:
        m = model.model
        if not hasattr(m, "_hidden_norm_weight"):
            return "the context K/V buffers are not built"
        if (
            m._rms_norm_eps != norm.variance_epsilon
            or m._hidden_norm_weight.data_ptr() != norm.weight.data_ptr()
        ):
            return "the precompute's context norm is not hidden_norm"
    rows = spec._num_graph_context_tokens(spec.max_num_reqs)
    if rows > CTX_MAX_ROWS:
        return f"{rows} graph context rows (max {CTX_MAX_ROWS})"
    return None


def _context_proj(spec, model) -> ContextProj | None:
    """Collective: the sharded context projection, or None (warned once)."""
    if akr_nonlayer.blocked("CTXPROJ", _ctx_reason(spec, model)):
        return None
    r, n_shard, k = _ctx_dims(model)
    g = akr_nonlayer.tp_group()
    if akr_nonlayer.blocked(
        "CTXPROJ",
        _gemm().linear_gather_comm_unsupported_reason(
            r, g.cpu_group, spec.device, n_shard=n_shard, k=k
        ),
    ):
        return None
    ctx = ContextProj(model, spec.device)
    if akr_nonlayer.blocked("CTXPROJ", ctx.reason(spec.device)):
        ctx.comm.close()
        return None
    return ctx


# ------------------------------------------------------------------ dense MLP
def _dense():
    from amd_kernels.decoder import bf16_dense_mlp_common

    return bf16_dense_mlp_common


class _Mlp:
    def __init__(self, mlp, ws, device):
        lib = _dense()
        self.ws, self.lib = ws, lib
        self.gate_up = lib.dense_mlp_pack_gate_up(mlp.gate_up_proj.weight.data)
        self.down = lib.dense_mlp_pack_down(mlp.down_proj.weight.data)
        # in place: the stock weights are not kept (no extra memory)
        mlp.gate_up_proj.weight.data = self.gate_up
        mlp.down_proj.weight.data = self.down
        self.x_stage = torch.zeros(
            _x_numel(self.down, MLP_MAX_ROWS), dtype=torch.bfloat16, device=device
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.models.kimi_k3.amd.akr_draft_ops import pack_tiles

        rows = x.shape[0]
        assert (
            x.ndim == 2
            and 1 <= rows <= MLP_MAX_ROWS
            and x.is_contiguous()
            and x.dtype == torch.bfloat16
        )
        pack_tiles(
            x, self.x_stage, self.lib.dense_mlp_packed_tiles(rows, down=self.down)
        )
        return self.lib.bf16_dense_mlp(
            self.ws,
            _x_view(self.x_stage, self.down, rows),
            self.gate_up,
            self.down,
            rows,
        )


def _x_numel(down: torch.Tensor, rows: int) -> int:
    return (
        _dense().dense_mlp_packed_tiles(rows, down=down) * (down.shape[0] // 32) * 512
    )


def _x_view(stage: torch.Tensor, down: torch.Tensor, rows: int) -> torch.Tensor:
    return stage[: _x_numel(down, rows)].view(-1, down.shape[0] // 32, 64, 8)


# One entry per installed layer MLP, called by the op below (inside the draft's
# compiled / captured forward, so the kernel launch is opaque to torch.compile).
_MLPS: dict[int, _Mlp] = {}


def _dense_mlp_op(x: torch.Tensor, key: int) -> torch.Tensor:
    return _MLPS[key](x if x.is_contiguous() else x.contiguous())


def _dense_mlp_fake(x: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty(x.shape)  # contiguous, as the kernel's output


direct_register_custom_op(
    op_name="kimi_akr_draft_dense_mlp",
    op_func=_dense_mlp_op,
    fake_impl=_dense_mlp_fake,
)


class AkrDenseMlp(nn.Module):
    """A draft layer's dense MLP on `bf16_dense_mlp` (this rank's partial sum), then the
    stock MLP's own all-reduce when it had one (`down_proj.reduce_results`), which keeps
    it visible to vLLM's all-reduce + RMSNorm fusion pass in a compiled backbone."""

    def __init__(self, stock: nn.Module, key: int) -> None:
        super().__init__()
        self.stock = stock  # keeps the (packed) weights registered
        self.key = key
        down = stock.down_proj
        self.all_reduce = bool(down.reduce_results) and down.tp_size > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.vllm.kimi_akr_draft_dense_mlp(x, self.key)
        if self.all_reduce:
            from vllm.distributed import tensor_model_parallel_all_reduce

            out = tensor_model_parallel_all_reduce(out)
        return out


def _mlp_reason(spec, model) -> str | None:
    # query rows a step (k, or 1 + k for a fill-in block without sample_from_anchor)
    q = getattr(spec, "num_query_per_req", spec.num_speculative_steps)
    rows = spec.max_num_reqs * q
    if rows > MLP_MAX_ROWS:
        return f"up to {rows} draft rows (max {MLP_MAX_ROWS})"
    lib = _dense()
    if not lib.available():
        return "needs gfx950 with FlyDSL"
    for layer in model.model.layers:
        mlp = layer.mlp
        if not all(hasattr(mlp, a) for a in ("gate_up_proj", "down_proj", "act_fn")):
            return f"{type(mlp).__name__} is not a gate_up / act / down MLP"
        if type(mlp.act_fn).__name__ != "SiluAndMul":
            return f"activation {type(mlp.act_fn).__name__}"
        gu, dn = mlp.gate_up_proj.weight, mlp.down_proj.weight
        if tuple(dn.shape) not in lib.bf16_dense_mlp_shapes() or tuple(gu.shape) != (
            2 * dn.shape[1],
            dn.shape[0],
        ):
            return (
                f"no bf16_dense_mlp for the shard {tuple(gu.shape)} / {tuple(dn.shape)}"
            )
        if gu.dtype != torch.bfloat16 or dn.dtype != torch.bfloat16:
            return "quantized MLP weights"
        if (
            getattr(mlp.gate_up_proj, "bias", None) is not None
            or getattr(mlp.down_proj, "bias", None) is not None
        ):
            return "MLP projections with a bias"
        if getattr(mlp, "shard_sequence_parallel", False):
            return "MLP shards the sequence"
    return None


def _install_mlp(spec, model) -> int:
    lib = _dense()
    device = spec.device
    hidden, inter = model.model.layers[0].mlp.down_proj.weight.shape
    kw = dict(hidden=hidden, inter=inter)
    if akr_nonlayer.blocked(
        "DMLP", lib.dense_mlp_workspace_unsupported_reason(device, **kw)
    ):
        return 0
    ws = lib.dense_mlp_workspace(device, **kw)
    # the largest call of every layer, on the raw weights (the packed ones keep their
    # shapes), before any weight is replaced
    for layer in model.model.layers:
        gu, dn = layer.mlp.gate_up_proj.weight.data, layer.mlp.down_proj.weight.data
        stage = torch.empty(
            _x_numel(dn, MLP_MAX_ROWS), dtype=torch.bfloat16, device=device
        )
        reason = lib.bf16_dense_mlp_unsupported_reason(
            ws, _x_view(stage, dn, MLP_MAX_ROWS), gu, dn, MLP_MAX_ROWS
        )
        if akr_nonlayer.blocked("DMLP", reason):
            return 0
    n = 0
    for layer in model.model.layers:
        key = len(_MLPS)
        _MLPS[key] = m = _Mlp(layer.mlp, ws, device)
        layer.mlp = AkrDenseMlp(layer.mlp, key)
        n += 1
    # compile every tier (first call of the mode warms them all)
    probe = torch.zeros(16, hidden, dtype=torch.bfloat16, device=device)
    m(probe)
    torch.accelerator.synchronize(device)
    return n


# ------------------------------------------------------------------ all-reduce + norm
def _norm_entry():
    from amd_kernels.norm import attnres_rmsnorm_common

    return attnres_rmsnorm_common


class AkrArNorm(nn.Module):
    """`norm(all_reduce(x), residual)` for the TP partial `x` of a projection that no
    longer all-reduces: `allreduce_attnres_rmsnorm` without bank (all-reduce + residual
    add + RMSNorm in one launch; `residual` is updated in place) for the calls its entry
    takes up to `AR_MAX_ROWS` rows, else the all-reduce and the stock norm. Host-only
    per-call check: capture-safe, and the kernel registers nothing per graph."""

    def __init__(self, stock: nn.Module, comm: Any) -> None:
        super().__init__()
        self.stock = stock  # keeps the weight registered
        self.comm = comm
        self.eps = float(stock.variance_epsilon)
        self._reason = _norm_entry().allreduce_attnres_rmsnorm_unsupported_reason

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args: tuple[Any, ...] = (
            x,
            residual,
            None,
            None,
            None,
            0,
            -1,
            self.stock.weight,
        )
        args += (self.eps, self.eps)
        if x.shape[0] <= AR_MAX_ROWS and not akr_config.call_unsupported(
            "draft", self._reason(self.comm, *args)
        ):
            return self.comm.allreduce_attnres_rmsnorm(*args)
        from vllm.distributed import tensor_model_parallel_all_reduce

        return self.stock(tensor_model_parallel_all_reduce(x), residual)


def _mlp_reduces(mlp) -> bool:
    if isinstance(mlp, AkrDenseMlp):
        return mlp.all_reduce
    down = getattr(mlp, "down_proj", None)
    return (
        down is not None
        and bool(getattr(down, "reduce_results", False))
        and down.tp_size > 1
    )


def _defer_mlp(mlp) -> None:
    """The MLP returns its TP partial (its all-reduce moves into the next norm)."""
    if isinstance(mlp, AkrDenseMlp):
        mlp.all_reduce = False
    else:
        mlp.down_proj.reduce_results = False


def _ar_reason(spec, model) -> str | None:
    """Why the release draft's all-reduce + RMSNorm pairs stay stock, or None."""
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3DecoderLayer,
        DFlashQwen3Model,
    )

    m = model.model
    if not isinstance(m, DFlashQwen3Model) or type(m).forward is not (
        DFlashQwen3Model.forward
    ):
        return f"{type(m).__name__} is not the DFlash Qwen3 backbone"
    cc = getattr(m, "compilation_config", spec.vllm_config.compilation_config)
    if (
        not getattr(m, "do_not_compile", True)
        or cc.mode == CompilationMode.STOCK_TORCH_COMPILE
    ):
        return "the draft backbone is compiled (vLLM's all-reduce + RMSNorm fusion)"
    if akr_nonlayer.tp_group().world_size == 1:
        return "TP 1 (no all-reduce)"
    norms = [m.norm]
    for layer in m.layers:
        if type(layer).forward is not DFlashQwen3DecoderLayer.forward:
            return f"{type(layer).__name__} is not a DFlash Qwen3 decoder layer"
        o_proj = getattr(layer.self_attn, "o_proj", None)
        if not isinstance(o_proj, RowParallelLinear) or not o_proj.reduce_results:
            return "o_proj is not an all-reducing RowParallelLinear"
        if not _mlp_reduces(layer.mlp):
            return f"{type(layer.mlp).__name__} does not end in a TP all-reduce"
        norms += [layer.input_layernorm, layer.post_attention_layernorm]
    for n in norms:
        if (
            type(n) is not RMSNorm
            or not n.has_weight
            or n.variance_size_override is not None
            or n.weight.dtype != torch.bfloat16
        ):
            return f"{n} is not a plain bf16 RMSNorm"
    return None


_DRAFT_NORM_COMM: Any = None  # the draft's own, when the target has none


def _draft_norm_comm(device: torch.device, hidden: int):
    """Collective: the target's communicator (the kdamoe `norm` route's, one stream: the
    draft runs after the target on it), else one for the draft; None (warned once)."""
    global _DRAFT_NORM_COMM
    from vllm.models.kimi_k3.amd import akr_runtime

    if akr_runtime.attnres_comm is not None:
        return akr_runtime.attnres_comm
    if _DRAFT_NORM_COMM is not None:
        return _DRAFT_NORM_COMM
    lib = _norm_entry()
    if akr_nonlayer.blocked("DAR", None if lib.available() else "needs gfx950"):
        return None
    g = akr_nonlayer.tp_group()
    kw = dict(hidden=hidden)
    reason = lib.attnres_comm_unsupported_reason(
        g.rank_in_group, g.cpu_group, device, **kw
    )
    if akr_nonlayer.blocked("DAR", reason):
        return None
    _DRAFT_NORM_COMM = lib.attnres_comm(g.rank_in_group, g.cpu_group, device, **kw)
    return _DRAFT_NORM_COMM


def _install_ar(spec, model) -> int:
    """Collective: every all-reduce + RMSNorm pair of the release draft (o_proj ->
    post-attention norm, MLP -> next input norm / final norm) on `AkrArNorm`; the
    projections return their partials. Returns the number of pairs (0: stock)."""
    m = model.model
    norm = m.norm
    hidden = norm.weight.shape[0]
    comm = _draft_norm_comm(spec.device, hidden)
    if comm is None:
        return 0
    z = torch.zeros(16, hidden, dtype=torch.bfloat16, device=spec.device)
    args: tuple[Any, ...] = (z, z.clone(), None, None, None, 0, -1, norm.weight)
    args += (norm.variance_epsilon, norm.variance_epsilon)
    if akr_nonlayer.blocked(
        "DAR", _norm_entry().allreduce_attnres_rmsnorm_unsupported_reason(comm, *args)
    ):
        return 0
    comm.allreduce_attnres_rmsnorm(*args)  # compiles the variant before any capture
    torch.accelerator.synchronize(spec.device)
    n = 0
    for i, layer in enumerate(m.layers):
        layer.self_attn.o_proj.reduce_results = False
        layer.post_attention_layernorm = AkrArNorm(layer.post_attention_layernorm, comm)
        _defer_mlp(layer.mlp)
        if i:
            layer.input_layernorm = AkrArNorm(layer.input_layernorm, comm)
        n += 2
    m.norm = AkrArNorm(norm, comm)
    return n


def _dar(spec, model) -> None:
    """The release draft's all-reduce + RMSNorm (the preview's layers carry their own
    `aiter_ar` flag)."""
    layers = getattr(model.model, "layers", ())
    if not layers or hasattr(layers[0], "aiter_ar") or not akr_nonlayer.switch("DAR"):
        return
    if akr_nonlayer.blocked("DAR", _ar_reason(spec, model)):
        return
    n = _install_ar(spec, model)
    if n:
        logger.info(
            "amd_kernels drafter: all-reduce + RMSNorm fused on %d pairs "
            "(allreduce_attnres_rmsnorm)",
            n,
        )


# ------------------------------------------------------------------ set-up
def _target_reason(target_model) -> str | None:
    if target_model is None:
        return None
    if not type(target_model).__module__.startswith("vllm.models.kimi_k3."):
        return f"target {type(target_model).__name__} is not a Kimi-K3 model"
    return None


def setup(spec, target_model) -> None:
    """From `DSparkSpeculator.load_model` (collective on the TP group: same order on
    every rank, after the weights load, before any capture)."""
    model = spec.model
    spec._akr_ctx = None
    spec._akr_markov = None
    if akr_nonlayer.blocked("draft", _target_reason(target_model)):
        return
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(spec.vllm_config):
        _setup_lmhead(model, spec.vllm_config)
        if akr_nonlayer.switch("CTXPROJ"):
            ctx = _context_proj(spec, model)
            if ctx is not None:
                ctx.warm_up(spec.device)
                ctx.install()
                spec._akr_ctx = ctx
                logger.info(
                    "amd_kernels drafter: sharded context projection (+%.0f MB a rank)",
                    ctx.w_packed.numel() * 2 / 2**20,
                )
        if akr_nonlayer.switch("DMLP") and not akr_nonlayer.blocked(
            "DMLP", _mlp_reason(spec, model)
        ):
            n = _install_mlp(spec, model)
            if n:
                logger.info("amd_kernels drafter: dense MLP kernels on %d layers", n)
        _dar(spec, model)
        akr_sampling.setup_speculator(spec)
        if getattr(spec.draft_model_config, "use_mla", False):
            from vllm.models.kimi_k3.amd import akr_draft_mla

            akr_draft_mla.setup(spec, model)
        else:
            from vllm.models.kimi_k3.amd import akr_draft_gqa

            akr_draft_gqa.setup(spec, model)


def store_hidden_states(
    spec, num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens
) -> bool:
    """`DFlashSpeculator._store_hidden_states` with the packed context projection; every
    stored row is the normed projection. False (nothing done) when it is not on."""
    ctx: ContextProj | None = getattr(spec, "_akr_ctx", None)
    if ctx is None:
        return False
    aux = aux_hidden_states
    packed = bool(aux) and len(aux) == ctx.num_aux
    start = 0
    if num_graph_tokens and spec._aux_staging is not None:
        assert packed, "the FULL graph's context rows need the aux hidden states"
        start = min(num_tokens, num_graph_tokens)
        ctx.pack_graph(aux, start)
    if start == num_tokens:
        return True
    out = spec.hidden_states[start:num_tokens]
    if packed and num_tokens - start <= CTX_MAX_ROWS:
        ctx.combine_eager(aux, start, num_tokens, out)
    elif aux:
        hidden = spec.model.combine_hidden_states(
            torch.cat([h[start:num_tokens] for h in aux], dim=-1)
        )
        out.copy_(ctx.post_norm(hidden))
    else:
        out.copy_(ctx.post_norm(last_hidden_states[start:num_tokens]))
    return True


def prepare_graph_forward(spec, num_reqs: int, attn_metadata: Any) -> bool:
    """`DFlashSpeculator._prepare_graph_forward` with the packed context projection.
    False (nothing done) when the context projection is not on."""
    ctx: ContextProj | None = getattr(spec, "_akr_ctx", None)
    if ctx is None or spec._aux_staging is None:
        return False
    num_context = spec._num_graph_context_tokens(num_reqs)
    ctx.combine_graph(num_context, spec.hidden_states[:num_context])
    spec.prepare_context_anchor(
        num_reqs, spec.target_input_buffers.query_start_loc, spec._num_rejected
    )
    spec._precompute_context_kv(0, num_context)
    if attn_metadata is not None and spec.supports_decode_metadata_update:
        spec._update_draft_decode_metadata(attn_metadata, num_reqs)
    return True
