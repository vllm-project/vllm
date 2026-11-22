# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's amd_kernels routes (draft route):

* `CTXPROJ`  `gemm/bf16_common.bf16_linear_gather`: `context_proj` (35840 -> 7168,
  514 MB replicated GEMM) + `context_norm` as a sharded GEMM (this rank's columns,
  64 MB at TP8) + gather over the TP ranks + RMSNorm in two launches. The target aux
  hidden states are packed into the kernel's tile layout (`linear_gather_pack_aux`)
  instead of `torch.cat`. Rows <= 192 (the whole decode graph's context); the original
  weight stays for larger prefill chunks. Extra memory: the packed shard 64 MB + slabs
  22 MB per rank.
* `DMLP`     `decoder/bf16_dense_mlp_common.bf16_dense_mlp`: the dense MLP's gate_up +
  SwiGLU and down GEMMs; the weights are replaced in place by their packed layout (no
  extra memory). Needs `max_num_seqs * k <= 144` (checked at set-up; the stock weights
  are gone).
* `DAR`      AITER's fused all-reduce + RMSNorm in `fused_allreduce_rms_norm`.
* the lm_head all-gather exchange (`akr_lmhead`, switch `LMHEAD`) and the fused Markov
  chain (`akr_sampling`, switch `MARKOV`).

Each piece is checked at set-up with its entry's `*_unsupported_reason` (communicator,
weights, the largest call it takes) and stays on the stock path otherwise.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_lmhead, akr_nonlayer, akr_sampling

logger = init_logger(__name__)

CTX_MAX_ROWS = 192  # rows of the staging buffers (the decode graph's context)
MLP_MAX_ROWS = 144  # the kernel loses to the stock GEMMs at 192 rows (0.95x)


def aiter_ar() -> bool:
    """Whether the drafter's all-reduce + RMSNorm takes AITER's fused op."""
    return akr_nonlayer.switch("DAR")


def compute_logits(model, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """`K3DSparkForCausalLM.compute_logits` on the lm_head exchange, or None (stock)."""
    if not getattr(model, "_akr_lmhead", False):
        return None
    return akr_lmhead.logits("draft", model.lm_head, hidden_states)


def _gemm():
    from amd_kernels.gemm import bf16_common

    return bf16_common


# ------------------------------------------------------------------ context_proj
def _ctx_dims(model) -> tuple[int, int, int]:
    """(rank, N shard, K) of this rank's rows of the context projection."""
    g = akr_nonlayer.tp_group()
    n, k = model.model.context_proj.weight.data.shape
    return g.rank_in_group, n // g.world_size, k


class ContextProj:
    def __init__(self, model, device: torch.device):
        """Collective on the TP group (the communicator)."""
        lib = _gemm()
        self.lib = lib
        r, n_shard, k = _ctx_dims(model)
        w = model.model.context_proj.weight.data
        self.w_packed = lib.linear_gather_pack_weight(
            w[n_shard * r : n_shard * (r + 1)].contiguous()
        )
        g = akr_nonlayer.tp_group()
        self.comm = lib.linear_gather_comm(r, g.cpu_group, device, n_shard=n_shard, k=k)
        self.hidden, self.k = w.shape
        self.num_aux = self.k // self.hidden
        self.norm_weight = model.model.context_norm.weight
        self.eps = float(model.model.context_norm.variance_epsilon)
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


def _ctx_reason(spec, model) -> str | None:
    m = model.model
    proj = getattr(m, "context_proj", None)
    if proj is None or proj.weight.dtype != torch.bfloat16:
        return "context_proj is not a bf16 weight"
    if spec._aux_staging is None:
        return "no FULL-graph aux staging (decode cudagraph mode is not FULL)"
    if (
        m.context_norm.weight.dtype != torch.bfloat16
        or abs(m.context_norm.variance_epsilon - 1e-5) > 1e-12
    ):
        return "context_norm is not bf16 / eps 1e-5"
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


def _mlp_reason(spec, model) -> str | None:
    if spec.max_num_reqs * spec.num_speculative_steps > MLP_MAX_ROWS:
        rows = spec.max_num_reqs * spec.num_speculative_steps
        return f"up to {rows} draft rows (max {MLP_MAX_ROWS})"
    lib = _dense()
    if not lib.available():
        return "needs gfx950 with FlyDSL"
    for layer in model.model.layers:
        mlp = layer.mlp
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
            getattr(mlp, "shard_sequence_parallel", False)
            or mlp.down_proj.reduce_results
        ):
            return "MLP all-reduces itself or shards the sequence"
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
        mlp = layer.mlp
        m = _Mlp(mlp, ws, device)
        mlp.forward = m
        n += 1
    # compile every tier (first call of the mode warms them all)
    probe = torch.zeros(16, hidden, dtype=torch.bfloat16, device=device)
    m(probe)
    torch.accelerator.synchronize(device)
    return n


# ------------------------------------------------------------------ set-up
def setup(spec, target_model) -> None:
    """From `DSparkSpeculator.load_model` (collective on the TP group: same order on
    every rank, after the weights load, before any capture)."""
    model = spec.model
    spec._akr_ctx = None
    if type(model).__name__ != "K3DSparkForCausalLM":
        return
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(spec.vllm_config):
        model._akr_lmhead = akr_lmhead.setup(
            "draft", model.lm_head, model.logits_processor, spec.vllm_config
        )
        if akr_nonlayer.switch("CTXPROJ"):
            ctx = _context_proj(spec, model)
            if ctx is not None:
                ctx.warm_up(spec.device)
                spec._akr_ctx = ctx
                logger.info(
                    "amd_kernels drafter: sharded context_proj (+%.0f MB per rank)",
                    ctx.w_packed.numel() * 2 / 2**20,
                )
        if akr_nonlayer.switch("DMLP") and not akr_nonlayer.blocked(
            "DMLP", _mlp_reason(spec, model)
        ):
            n = _install_mlp(spec, model)
            if n:
                logger.info("amd_kernels drafter: dense MLP kernels on %d layers", n)
        akr_sampling.setup_speculator(spec)
        from vllm.models.kimi_k3.amd import akr_draft_mla

        akr_draft_mla.setup(spec, model)


def store_hidden_states(
    spec, num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens
) -> bool:
    """`DFlashSpeculator._store_hidden_states` with the packed context projection.
    False (nothing done) when the context projection is not on."""
    ctx: ContextProj | None = getattr(spec, "_akr_ctx", None)
    if ctx is None or not aux_hidden_states or len(aux_hidden_states) != ctx.num_aux:
        return False
    start = 0
    if num_graph_tokens and spec._aux_staging is not None:
        start = min(num_tokens, num_graph_tokens)
        ctx.pack_graph(aux_hidden_states, start)
    if start == num_tokens:
        return True
    rows = num_tokens - start
    if rows <= CTX_MAX_ROWS:
        ctx.combine_eager(
            aux_hidden_states, start, num_tokens, spec.hidden_states[start:num_tokens]
        )
    else:
        hidden = spec.model.combine_hidden_states(
            torch.cat([h[start:num_tokens] for h in aux_hidden_states], dim=-1)
        )
        spec.hidden_states[start:num_tokens].copy_(hidden)
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
