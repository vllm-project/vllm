# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Granite Switch model.

Granite Switch is a Granite MoE-hybrid derivative that carries several LoRA (or
aLoRA) adapters INSIDE a single checkpoint and selects one PER TOKEN at run
time. A short, trainable-free "switch" reads the prompt, spots an adapter
control token, and emits a per-token adapter index; every LoRA-targeted
projection then applies that token's adapter in the same GEMM it uses for the
base weight.

Pipeline::

    input ids
      -> MultiSwitch            (control-token detection -> adapter index per
                                 token; control ids rewritten to substitutes)
      -> embedding              (frozen)
      -> decoder layers         (frozen base + frozen adapters, per-token
                                 selection)
      -> norm, lm_head

Everything is frozen. There is no training path and no adapter is ever loaded,
swapped or evicted at run time.

Why this does not use vLLM's LoRA subsystem
-------------------------------------------
``vllm/lora/`` serves *runtime-swappable* adapters: ``LoRARequest`` names an
adapter per request, ``LoRAModelManager`` pages adapter weights into a fixed
number of slots, and ``PunicaWrapper`` dispatches a request-uniform adapter id.
Granite Switch is a different problem on every one of those axes:

* The adapters are resident. They ship inside the checkpoint and are fused into
  the base weight at load time - ``w_ext`` concatenates the base rows with every
  applicable adapter's shrink rows, so one GEMM yields the base projection and
  all shrink vectors at once and the two are thereafter inseparable. There is
  nothing to swap, and no swap cost to amortize.
* Selection is per TOKEN, not per request. One sequence can route its system
  prompt to the base model, one turn to adapter 2 and the next to adapter 5.
  ``LoRARequest`` cannot express that.
* The ranks are heterogeneous and fused together. A single module may hold
  rank-16 and rank-256 adapters side by side, grouped into compile-time tiers in
  one kernel launch.
* The kernel additionally fuses SwiGLU (for the gate/up projection) and the
  Shadow Residual cross-stream shrink, neither of which has an analogue in
  ``vllm/lora/``.

So the per-token routing kernel and ``SwitchedLoRALinear`` live here, beside the
model, rather than in ``vllm/lora/``, and this model does NOT declare
``SupportsLoRA``: a request-level ``LoRARequest`` against it would be silently
meaningless.

Checkpoint format
-----------------
The loader expects the following on disk. Anything absent that is not a LoRA
delta is rejected at load time rather than served as uninitialized memory (see
``_audit_loaded``).

Config (``GraniteSwitchConfig``, ``model_type: granite_switch``):

* ``num_adapters``, ``adapter_names``, ``adapter_ranks`` (per-adapter rank),
  ``lora_target_modules``, ``adapter_token_ids`` (the control token id that
  fires each adapter) and ``adapter_substitute_token_ids`` (the in-distribution
  id each control token is rewritten to).
* ``projection_head_dim`` - the attention head size, required whenever
  ``num_adapters > 0``.
* ``num_hidden_layers`` is INFLATED by ``SWITCH_CACHE_LAYERS`` (2) when
  ``num_adapters > 0``: the switch's counting and memory heads are real
  paged-KV attention layers and each consumes one KV-cache group. The model
  subtracts the two to recover the decoder-layer count.
* ``fused_add_norm`` records which RMSNorm residual-add reduction order the
  weights were fitted against; see ``rms_norm_select``.
* ``cross_stream_rank`` is set (an int) on a Shadow Residual checkpoint and
  ``None`` on a plain LoRA/aLoRA one. That single field selects the decoder
  tier.

Weights:

* Base weights in vLLM's own fused layout: ``self_attn.qkv_proj.base_layer`` is
  a single pre-fused Q|K|V tensor and ``shared_mlp.input_linear.base_layer`` a
  single pre-fused gate|up tensor. The loader does not fuse separate
  ``q_proj``/``k_proj``/``v_proj`` tensors.
* Per-module LoRA deltas as ``lora_A_slices.{s}`` ``[NA, r, K]`` and
  ``lora_B_slices.{s}`` ``[NA, N_s, r]``, one slice ``s`` per fused output
  slice. Every adapter is padded to the module's maximum rank; an adapter that
  does not target a module is all-zero there and is compacted out at load time.
  Ranks are snapped up to ``SUPPORTED_RANKS`` and zero-padded, which is
  numerically identical and lets each tier be a kernel constexpr.
* The expert bank stacked per layer as
  ``block_sparse_moe.experts.gate_up_proj`` ``[E, 2I, H]``,
  ``block_sparse_moe.experts.down_proj`` ``[E, H, I]`` and
  ``block_sparse_moe.router.weight`` ``[E, H]``; it is fanned out onto
  ``FusedMoE`` at load time. The experts are frozen and never LoRA targets.
* Shadow Residual additionally carries ``cross_stream.lora_A``
  ``[NA, 1, cross_rank, H]`` and ``cross_stream.lora_B``
  ``[NA, 1, H, cross_rank]`` per layer. ``cross_stream`` has NO base weight.

Known limitation
----------------
The switch's counting head recovers a control token's write address from a
``1/(1 + n)`` attention signal, and that signal takes the KV-cache dtype because
the head is a real paged-KV attention layer. bfloat16 inverts the signal exactly
only up to ``n = 188`` control tokens in one sequence (189 aliases onto 188);
float32 is exact well past 4095. This is a separate bound from the codebook's
capacity, which limits the memory head rather than the counting head.
``GraniteSwitchConfigVerifier`` reports it at startup.

NOTE: this module must NOT use ``from __future__ import annotations``.
``@support_torch_compile`` infers dynamic dims from the real annotation objects
and breaks on stringized annotations.
"""

import abc
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.granite_switch import (
    SWITCH_CACHE_LAYERS,
    GraniteSwitchConfig,
)

from .granite_switch_kernels import (
    BLOCK_M,
    BLOCK_N,
    SUPPORTED_RANKS,
    FusedLoRAKernelMeta,
    LoRAContext,
    SRFusedLoRAKernelMeta,
    SRLoRAContext,
    build_w_ext,
    granite_switch_lora_expand,
    granite_switch_lora_expand_swiglu,
    granite_switch_lora_shrink_expand,
    promote_rank,
)
from .granite_switch_utils import KerdockDGCodeGenerator, recover_count_from_signal
from .granitemoe import GraniteMoeMoE
from .granitemoeshared import GraniteMoeSharedMLP
from .interfaces import SupportsPP
from .utils import is_pp_missing_parameter, make_layers, maybe_prefix

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Token exchange
#
# The switch reads the ORIGINAL input_ids to select an adapter, then rewrites
# each control-token id to its substitute id so the decoder embeds a clean
# sequence and never knows a control token was there. The substitute embedding
# is a real in-distribution token, which is what keeps the control token from
# perturbing the hidden states it was only meant to route.
# ---------------------------------------------------------------------------


def build_control_to_substitute_lut(config) -> torch.Tensor | None:
    """Derive the control -> substitute lookup table from *config*.

    Shape ``[max(vocab_size, max_ctrl_id + 1)]``: ``-1`` at every non-control
    id, and the substitute id at each control slot. In a finished checkpoint
    ``vocab_size`` already covers the control ids, so the ``max`` is a no-op;
    it is kept because indexing the table by raw ``input_ids`` must not be able
    to run off the end.

    Only the real adapter control tokens are mapped. A base-reset token, if the
    checkpoint has one, is deliberately not rewritten: it is only ever placed
    at an in-distribution turn boundary, so it needs no substitute.

    Returns ``None`` when the checkpoint carries no token-exchange mapping, in
    which case ``input_ids`` is left untouched. An empty id list counts as no
    mapping, and is tested for falsiness rather than ``is None`` because
    ``max(())`` raises.
    """
    if config is None:
        return None
    ctrl_ids = getattr(config, "adapter_token_ids", None)
    sub_ids = getattr(config, "adapter_substitute_token_ids", None)
    if not ctrl_ids or not sub_ids:
        return None

    lut_size = max(getattr(config, "vocab_size", 0), max(ctrl_ids) + 1)
    lut = torch.full((lut_size,), -1, dtype=torch.long)
    for ctrl_id, sub_id in zip(ctrl_ids, sub_ids):
        lut[ctrl_id] = sub_id
    return lut


def apply_token_exchange(
    lut: torch.Tensor | None, input_ids: torch.Tensor
) -> torch.Tensor:
    """Rewrite each control token's id to its substitute id via ``lut``.

    Deliberately branch-free: ``torch.where`` runs every step. The decoder is
    wrapped in ``@support_torch_compile``, which rules out a ``tensor.any()``
    short-circuit. When ``lut`` is ``None`` the input is returned unchanged.
    """
    if lut is None:
        return input_ids
    sub_id_per_pos = lut[input_ids]
    is_control = sub_id_per_pos >= 0
    return torch.where(is_control, sub_id_per_pos, input_ids)


# ---------------------------------------------------------------------------
# Fused switched-LoRA linear layer
#
# One GEMM produces the base projection and every active adapter's shrink
# vectors at once; a Triton expand kernel then accumulates the LoRA delta in
# place. See ``granite_switch_kernels`` for the kernel backend.
# ---------------------------------------------------------------------------


class SwitchedLoRALinear(nn.Module):
    """Fused LoRA linear layer using the switch-LoRA kernel.

    Forward path:

    1. ``x_ext = x @ w_ext.T`` - a single GEMM yielding the base output plus
       every applicable adapter's shrink vectors.
    2. ``granite_switch_lora_expand(...)`` - Triton kernel accumulating the
       LoRA delta in place into the base columns of ``x_ext``.

    Weights are stored in checkpoint layout while loading, then converted to
    the fused layout by :meth:`finalize_weights`.

    Memory layout (post ``finalize_weights``)
    -----------------------------------------
    For a layer with S slices (S=1 for o_proj/down_proj, S=2 for gate_up,
    S=3 for QKV), ``N_total = sum(N_s)`` output features, K input features,
    and adapters grouped into rank tiers:

    INVARIANTS

    1. Each adapter has a rank per module. That rank applies to lora_A and
       lora_B for every slice within that module. All adapters of the same
       rank (within a module) belong to the same rank tier. Different modules
       may assign different ranks to the same adapter.

    2. An adapter may or may not be applicable to a given module (i.e. have
       non-zero trained weights for it). Non-applicable adapters contribute no
       rows to ``w_ext`` and are remapped to 0 (base) in this module's
       ``remap_table``.

    3. The rank-tier ordering is global (same across all modules). The
       ``remap_table`` is per-module (non-applicable adapters are compacted
       out, applicable adapters are sorted by rank). The bitmask is per-module,
       computed from kernel-local (post-remap) adapter indices so that bitmask
       bit ``a`` corresponds exactly to kernel-local adapter ``a + 1``. Column
       offsets (``col_r``) into ``x_ext`` are also per-module.

    PER-MODULE data (built at ``finalize_weights``, differs across instances)

    ``remap_table``
        global adapter_id -> kernel-local position for this module (0 for
        non-applicable adapters).
    ``bitmask``
        computed per-forward by ``FusedLoRAKernelMeta`` from kernel-local
        indices; exact for this module.
    ``w_ext``
        only applicable adapters, sorted by rank tier.
    ``lora_B_merged``
        same - only applicable adapters, merged along ``N_total``.
    ``NA_r``
        count of applicable adapters per rank tier (a constexpr in the kernel;
        0 means the entire tier compiles away).
    ``col_r``
        base column offset in ``x_ext`` for each rank tier.

    ``w_ext`` is ``[N_total + sum_{a applicable}(S * r_a), K]``::

        rows 0 .. N_total-1     : W_base (all slices fused, as stored by vLLM)
        tier r0, adapter a0     : lora_A_a0_s0, lora_A_a0_s1, ... (S*r0 rows)
        tier r0, adapter a1     : lora_A_a1_s0, lora_A_a1_s1, ... (S*r0 rows)
        ...
        tier r1, adapter b0     : S*r1 rows
        ...
        (non-applicable adapters contribute no rows)

    ``x_ext = x @ w_ext.T`` is
    ``[M, N_total + sum_{a applicable}(S * r_a)]``::

        cols 0 .. N_total-1     : base outputs (all slices concatenated)
        tier r0, adapter a0     : shrink_s0, shrink_s1, ... each of width r0
        tier r0, adapter a1     : shrink_s0, shrink_s1, ...
        ...

    For token m, adapter a (rank r, per-module tier position ``pos_a``),
    slice s::

        shrink = x_ext[
            m, col_r + pos_a * S * r + s * r : col_r + pos_a * S * r + s * r + r
        ]

    ``lora_B_merged`` per rank tier is ``[n_r_local, N_total, r]``::

        lora_B_merged[pos_a, 0:N_s0, :]         = lora_B for adapter a, slice 0
        lora_B_merged[pos_a, N_s0:N_s0+N_s1, :] = lora_B for adapter a, slice 1
        ...

    ``N_s`` denotes the output feature count of slice s - a property of the
    base layer geometry alone, independent of adapter count or rank. For a
    fused layer, ``N_total = sum(N_s)`` and ``W_base`` is the ``[N_total, K]``
    weight matrix with the slice sub-matrices stacked vertically (e.g. for
    ``qkv_proj``: N_0=q_size, N_1=k_size, N_2=v_size).

    Tile ``(pid_m, pid_n)`` in the expand kernel covers output columns
    ``[pid_n*BLOCK_N, (pid_n+1)*BLOCK_N)``. For the per-tile slice lookup
    (``TileSlice[pid_n]``) to be correct, every tile must fall entirely within
    one slice - no tile may straddle a slice boundary. This requires BLOCK_N to
    divide every ``N_s`` exactly (``N_s % BLOCK_N == 0`` for all s), which
    ensures that every slice boundary is also a tile boundary.
    ``BLOCK_N <= min(N_s)`` alone is not sufficient.
    :meth:`finalize_weights` enforces this on the local (post-TP-shard) slice
    sizes at load time.
    """

    # Class-level defaults so both attributes exist statically (torch.compile
    # sees a stable attribute, not a per-instance add). ``_lora_ctx`` is wired
    # post-init by GraniteSwitchModel via ``object.__setattr__`` to a single
    # shared LoRAContext; ``_module_idx`` is assigned by the decoder tier's
    # ``register_remap_tables``, which is also what gives this module its row
    # in the context's per-module tables.
    _lora_ctx: LoRAContext | None = None
    _module_idx: int = -1

    def __init__(
        self,
        base_layer: nn.Module,
        num_adapters: int,
        max_lora_rank: int,
        num_slices: int = 1,
        output_slices: tuple[int, ...] | None = None,
        fuse_swiglu: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank
        self.num_slices = num_slices
        # When True (shared-MLP gate/up projection only), forward() fuses the
        # LoRA expand with the SwiGLU activation and returns the activated
        # [M, H] directly - no strided base_out, no separate SiluAndMul.
        # Requires the merged 2-slice (gate, up) layout with equal slice widths.
        self.fuse_swiglu = fuse_swiglu

        if hasattr(base_layer, "weight"):
            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
        elif hasattr(base_layer, "qweight"):
            in_features = base_layer.input_size
            out_features = base_layer.output_size
            device = base_layer.qweight.device
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported base layer type: {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self._device = device
        self._dtype = dtype

        # TP config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self._is_column_parallel = isinstance(
            base_layer,
            (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear),
        )
        self._is_row_parallel = isinstance(base_layer, RowParallelLinear)
        self._row_parallel_reduce = (
            self._is_row_parallel
            and self.tp_size > 1
            and getattr(base_layer, "reduce_results", False)
        )

        # Output slices for packed modules
        if num_slices > 1:
            if output_slices is None:
                raise ValueError("output_slices required for packed modules")
            if self._is_column_parallel and self.tp_size > 1:
                # Assumes each s is divisible by tp_size - enforced by vLLM's
                # column-parallel layer constructors, not re-checked here.
                self.output_slices = tuple(s // self.tp_size for s in output_slices)
            else:
                self.output_slices = output_slices
        else:
            self.output_slices = (out_features,)

        # Checkpoint-layout parameters: populated by weight_loader, consumed by
        # finalize_weights.
        if num_slices == 1:
            self.lora_A = nn.Parameter(
                torch.zeros(
                    num_adapters,
                    1,
                    max_lora_rank,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    num_adapters,
                    1,
                    out_features,
                    max_lora_rank,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_A.weight_loader = self._make_weight_loader("a")
            self.lora_B.weight_loader = self._make_weight_loader("b")
        else:
            self.lora_A_slices = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            num_adapters,
                            1,
                            max_lora_rank,
                            in_features,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    for _ in range(num_slices)
                ]
            )
            self.lora_B_slices = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            num_adapters,
                            1,
                            output_size,
                            max_lora_rank,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    for output_size in self.output_slices
                ]
            )
            for i, p in enumerate(self.lora_A_slices):
                p.weight_loader = self._make_weight_loader("a", i)
            for i, p in enumerate(self.lora_B_slices):
                p.weight_loader = self._make_weight_loader("b", i)

        # Fused kernel state (populated by finalize_weights)
        self._finalized = False

    @property
    def weight(self):
        return self.base_layer.weight

    def slice_lora_a_weight(
        self, full_weight: torch.Tensor, slice_idx: int = 0
    ) -> torch.Tensor:
        if self.tp_size <= 1 or not self._is_row_parallel:
            return full_weight
        full_in = full_weight.shape[-1]
        shard_size = full_in // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size]

    def slice_lora_b_weight(
        self, full_weight: torch.Tensor, slice_idx: int = 0
    ) -> torch.Tensor:
        if self.tp_size <= 1 or not self._is_column_parallel:
            return full_weight
        full_out = full_weight.shape[-2]
        shard_size = full_out // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size, :]

    def _make_weight_loader(self, ab: str, slice_idx: int = 0):
        slicer = self.slice_lora_a_weight if ab == "a" else self.slice_lora_b_weight

        def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            sliced = slicer(loaded_weight, slice_idx)
            param.data.copy_(sliced)

        return weight_loader

    def finalize_weights(self, adapter_ranks: list[int], block_n: int | None = None):
        """Convert checkpoint-layout LoRA weights to the fused kernel layout.

        Called once after ``load_weights()``. Builds ``w_ext``,
        ``lora_B_merged`` and the adapter index remap table. Handles both
        single-slice (S=1) and multi-slice (S>1) layers through one code path.

        TP assumption: lora_A/lora_B are sharded by even integer division of
        the in/out feature dim across ``tp_size`` (see
        :meth:`slice_lora_a_weight` / :meth:`slice_lora_b_weight`), which
        mirrors how vLLM's parallel linear layers shard the base weight. The
        per-rank shard sizes are therefore exact (no ragged final shard).

        Args:
            adapter_ranks: Rank per adapter for this module (length
                ``num_adapters``). Each rank applies to all slices within this
                module. Currently the same list is passed to every module
                (from ``config.adapter_ranks``); per-module differentiation
                comes from zero-detection only. The interface accepts a
                per-module list to support future per-module rank assignment
                (e.g. adapter 0 rank 16 here but rank 32 elsewhere). Rank 0,
                or all-zero lora_A rows, marks an adapter as non-applicable to
                this module.
            block_n: Output tile width the precomputed tile/slice tables are
                built for. Defaults to the kernel's ``BLOCK_N``.

        Raises:
            ValueError: If ``block_n`` does not divide every local output
                slice, which would let a tile straddle a slice boundary.

        """
        if self._finalized:
            return

        # block_n determines the precomputed tile/slice tables, so it is bound
        # here at finalize time (not per launch). Defaults to the kernel's
        # BLOCK_N; must divide every output slice so no tile straddles a slice
        # boundary (checked below).
        if block_n is None:
            block_n = BLOCK_N

        device = self._device
        dtype = self._dtype
        NA = self.num_adapters
        S = self.num_slices

        # Collect lora_A and lora_B checkpoint data for all slices
        if S == 1:
            lora_A_all = [self.lora_A.data]  # [NA, 1, max_rank, K]
            lora_B_all = [self.lora_B.data]  # [NA, 1, N, max_rank]
        else:
            lora_A_all = [p.data for p in self.lora_A_slices]
            lora_B_all = [p.data for p in self.lora_B_slices]

        # Adapters with all-zero lora_A (over their first r rows, any slice) do
        # not cover this module. One batched reduction with a single host sync
        # rather than NA*S .item() calls, whose GPU->CPU stalls dominated load
        # time across every SwitchedLoRALinear in the model.
        max_rank = lora_A_all[0].shape[2]
        ranks_t = torch.tensor(adapter_ranks, device=device)  # [NA]
        # rank_mask[i, j] = j < r_i - restricts the check to each adapter's rows.
        rank_mask = torch.arange(max_rank, device=device)[None, :] < ranks_t[:, None]
        applicable_t = torch.zeros(NA, dtype=torch.bool, device=device)
        for s in range(S):
            # [NA, max_rank, K] -> nonzero per (adapter, rank-row) -> [NA, max_rank]
            nz = lora_A_all[s][:, 0, :, :].ne(0).any(dim=-1)
            applicable_t |= (nz & rank_mask).any(dim=1)
        applicable = applicable_t.tolist()  # single sync

        # remap_table: global adapter_id (1-based) -> kernel-local position;
        # non-applicable adapters map to 0 (base, no LoRA contribution).
        applicable_adapters = [i for i in range(NA) if applicable[i]]

        # Off-tier ranks (rank 8 is a common LoRA choice) are promoted to the
        # next tier and zero-padded, which is numerically identical since the
        # padded rows contribute nothing. Promoting rather than extending
        # SUPPORTED_RANKS: slice_col_r is [S, 6] and _na a 6-tuple, so the tier
        # count is fixed at six.
        eff_ranks = [promote_rank(r) for r in adapter_ranks]
        pad_to = max((eff_ranks[i] for i in applicable_adapters), default=0)
        if pad_to > max_rank:
            pad = pad_to - max_rank
            # lora_A is [NA, 1, max_rank, K] - pad the rank dim (second to last).
            lora_A_all = [
                torch.nn.functional.pad(a, (0, 0, 0, pad)) for a in lora_A_all
            ]
            # lora_B is [NA, 1, N, max_rank] - pad the rank dim (last).
            lora_B_all = [torch.nn.functional.pad(b, (0, pad)) for b in lora_B_all]
        adapter_ranks = eff_ranks

        rank_order = sorted(applicable_adapters, key=lambda i: adapter_ranks[i])
        remap = torch.zeros(NA + 1, dtype=torch.long, device=device)
        for kernel_idx, orig_idx in enumerate(rank_order):
            remap[orig_idx + 1] = kernel_idx + 1
        self.register_buffer("remap_table", remap, persistent=False)

        # Build rank tiers (applicable adapters only, ascending rank order).
        # Plain dict: insertion order is the ascending-rank order of
        # rank_order, which is what the tier layout depends on.
        tiers: dict[int, list[int]] = {}
        for orig_idx in rank_order:
            r = adapter_ranks[orig_idx]
            if r not in tiers:
                tiers[r] = []
            tiers[r].append(orig_idx)

        # Build lora_A_by_rank: {rank: [n_r, S, rank, K]}. Layout is adapter
        # outer, slice middle, rank-row inner -> tier/adapter/slice order in
        # w_ext.
        lora_A_by_rank: dict[int, torch.Tensor] = {}
        for rank, orig_indices in tiers.items():
            A_list = []
            for oi in orig_indices:
                # [S, rank, K] - all slices for this adapter
                slices = torch.stack(
                    [lora_A_all[s][oi, 0, :rank, :] for s in range(S)], dim=0
                )
                A_list.append(slices)
            lora_A_by_rank[rank] = torch.stack(A_list, dim=0)  # [n_r, S, rank, K]

        # w_ext = [W_base | tier_r0_a0_s0, a0_s1..., a1_s0, a1_s1... | tier_r1 ...]
        W_base = self.base_layer.weight.data  # [N_total, K]
        w_ext = build_w_ext(W_base, lora_A_by_rank)
        self.register_buffer("w_ext", w_ext, persistent=False)

        N_total = W_base.shape[0]
        self._N_total = N_total

        # Build lora_B_merged per tier: {rank: [n_r, N_total, rank]}. Slices
        # are concatenated along N_total so tiles can access any output column
        # uniformly.
        tier_info: dict[int, int] = {}
        lora_B_merged: dict[int, torch.Tensor] = {}
        for rank, orig_indices in tiers.items():
            B_list = []
            for oi in orig_indices:
                # Cat lora_B across slices -> [N_total, rank]
                B_adapter = torch.cat(
                    [lora_B_all[s][oi, 0, :, :rank] for s in range(S)], dim=0
                )
                B_list.append(B_adapter)
            lora_B_merged[rank] = torch.stack(B_list, dim=0)  # [n_r, N_total, rank]
            tier_info[rank] = len(orig_indices)

        self._num_applicable = sum(tier_info.values())

        self._block_n = block_n
        if any(N_s % block_n != 0 for N_s in self.output_slices):
            raise ValueError(
                f"block_n={block_n} must divide every local output slice, got "
                f"{self.output_slices}. Otherwise an output tile straddles a "
                "slice boundary and the per-tile slice lookup silently reads "
                "the wrong slice's shrink columns."
            )

        # Build tile_to_slice[num_tiles_N]: slice index for each output tile
        num_tiles_N = (N_total + block_n - 1) // block_n
        N_cumsum = [0]
        for N_s in self.output_slices:
            N_cumsum.append(N_cumsum[-1] + N_s)

        tile_slice_data = torch.zeros(num_tiles_N, dtype=torch.int32, device=device)
        for t in range(num_tiles_N):
            col_start = t * block_n
            for s in range(S):
                if N_cumsum[s] <= col_start < N_cumsum[s + 1]:
                    tile_slice_data[t] = s
                    break
        self.register_buffer("tile_to_slice", tile_slice_data, persistent=False)

        # Build slice_col_r[S, 6]: for each (slice, tier), the effective base
        # column in x_ext for shrink reads.
        # slice_col_r[s, t] = N_total + sum_{t'<t}(n_t' * S * r_t') + s * r_t
        tier_col_bases = []
        offset = N_total
        for r in SUPPORTED_RANKS:
            tier_col_bases.append(offset)
            n_r = tier_info.get(r, 0)
            offset += n_r * S * r

        slice_col_r_data = torch.tensor(
            [
                [tier_col_bases[t] + s * SUPPORTED_RANKS[t] for t in range(6)]
                for s in range(S)
            ],
            dtype=torch.int32,
            device=device,
        )  # [S, 6]
        self.register_buffer("slice_col_r", slice_col_r_data, persistent=False)

        # Pre-cache expand kernel arguments
        self._precompute_expand_args(lora_B_merged, tier_info, device, dtype, S)

        # Fused gate/up + SwiGLU setup (shared-MLP first projection only).
        if self.fuse_swiglu:
            if S != 2 or self.output_slices[0] != self.output_slices[1]:
                raise ValueError(
                    "fuse_swiglu requires the merged 2-slice (gate, up) layout "
                    f"with equal widths; got S={S}, "
                    f"output_slices={self.output_slices}"
                )
            # SwiGLU is applied to the post-projection gate/up; a base bias
            # would have to be folded in before silu*mul. Granite gate/up is
            # bias-free.
            if getattr(self.base_layer, "bias", None) is not None:
                raise ValueError(
                    "fuse_swiglu does not support a biased gate/up projection"
                )
            self._H = N_total // 2

        # Register the buffer slot as None first, then assign through it:
        # register_buffer rejects a name that already exists as a plain
        # attribute.
        self.register_buffer("_fused_bias", None, persistent=False)
        self._output_bias = None
        if getattr(self.base_layer, "bias", None) is not None:
            if not getattr(self.base_layer, "skip_bias_add", False):
                self._fused_bias = self.base_layer.bias.data
            else:
                self._output_bias = self.base_layer.bias

        # Freeze checkpoint-layout parameters (data retained for state_dict)
        if S == 1:
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)
        else:
            for p in self.lora_A_slices:
                p.requires_grad_(False)
            for p in self.lora_B_slices:
                p.requires_grad_(False)

        self._finalized = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass using the fused switch-LoRA kernel."""
        assert self._finalized, "finalize_weights() must be called before forward()"

        x_ext = torch.mm(x, self.w_ext.T)

        # Shared-MLP gate/up: fuse expand + SwiGLU and return the activated
        # [M, H] directly. No strided base_out ever escapes (the kernel reads
        # x_ext by explicit stride), so no .contiguous() and no SiluAndMul.
        if self.fuse_swiglu:
            return self._forward_swiglu(x, x_ext)

        base_out = x_ext[:, : self._N_total]

        ctx = self._lora_ctx
        remapped = ctx.remapped_indices if ctx is not None else None
        if remapped is not None and self._num_applicable > 0:
            M = x.shape[0]
            # Kernel-local indices were gathered once for all modules in
            # prepare_and_store(); this is a stride-1 contiguous row-view into
            # ctx.remapped_indices [num_modules, num_tokens] - no per-module
            # gather op, no launch.
            adapter_indices = remapped[self._module_idx, :M]
            self._run_expand(x_ext, adapter_indices, ctx)

        if self._row_parallel_reduce:
            # base_out is a column-slice of x_ext (row stride N+shrink_cols),
            # so it is non-contiguous; all-reduce's internal .view() requires a
            # packed layout. Copy to contiguous before the reduce. NOTE: this
            # copy is on the TP>1 row-parallel critical path - if it proves
            # costly for large models, revisit fusing the base output into a
            # standalone buffer rather than sharing x_ext.
            base_out = tensor_model_parallel_all_reduce(base_out.contiguous())

        # Bias goes in after the all-reduce: a row-parallel rank holds a partial
        # sum, so pre-reduce would sum the bias tp_size times. The LoRA delta IS
        # a partial and must go pre-reduce (above). Under skip_bias_add the
        # caller applies _output_bias instead.
        if self._fused_bias is not None:
            base_out = base_out + self._fused_bias

        # base_out is a strided view of x_ext (shrink columns make the row
        # stride > N_total). That is fine for every consumer in the Granite
        # stack: qkv -> attention split+RoPE and o/down -> residual add are all
        # stride-safe. The one consumer that assumes packed rows (SiluAndMul on
        # the gate/up output) is handled by the fuse_swiglu path above, which
        # never exposes a strided base. So no .contiguous() is needed here.
        return base_out, self._output_bias

    def _forward_swiglu(self, x: torch.Tensor, x_ext: torch.Tensor):
        """Fused gate/up expand + SwiGLU -> contiguous [M, H] activation."""
        M = x.shape[0]
        H = self._H
        out = torch.empty(M, H, device=x_ext.device, dtype=x_ext.dtype)

        ctx = self._lora_ctx
        remapped = ctx.remapped_indices if ctx is not None else None
        bitmasks = ctx.per_module_bitmasks if ctx is not None else None
        if remapped is not None and bitmasks is not None and self._num_applicable > 0:
            adapter_indices = remapped[self._module_idx, :M]
            bitmask = bitmasks[self._module_idx]
        else:
            # No kernel metadata / no applicable adapters: a zero bitmask makes
            # the kernel skip all LoRA work and emit silu(gate)*up of the base
            # only.
            num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
            adapter_indices = torch.zeros(M, dtype=torch.long, device=x_ext.device)
            bitmask = torch.zeros(num_tiles_m, dtype=torch.int64, device=x_ext.device)

        granite_switch_lora_expand_swiglu(
            out,
            x_ext,
            adapter_indices,
            bitmask,
            self._lb_packed,
            self.slice_col_r,
            self._na[0],
            self._na[1],
            self._na[2],
            self._na[3],
            self._na[4],
            self._na[5],
            self._S,
            self._block_n,
            H,
            self._N_total,
        )
        return out, self._output_bias

    def _precompute_expand_args(self, lora_B_merged, tier_info, device, dtype, S):
        """Cache expand kernel arguments at finalize time."""
        self._na = tuple(tier_info.get(r, 0) for r in SUPPORTED_RANKS)
        self._S = S

        # Single packed lora_B buffer - a contiguous concat over PRESENT tiers
        # of [NA_r, N_total, r] (row-major). Empty tiers (NA_r == 0) contribute
        # zero elements; the kernel computes each tier's base offset as
        # cumsum(NA_r*N*r) from the NA_* constexprs + N, so this must match
        # exactly that ordering.
        packed_parts = [
            lora_B_merged[r].reshape(-1) for r in SUPPORTED_RANKS if r in lora_B_merged
        ]
        lb_packed = (
            torch.cat(packed_parts)
            if packed_parts
            else torch.zeros(1, device=device, dtype=dtype)
        )
        self.register_buffer("_lb_packed", lb_packed.contiguous(), persistent=False)

    def _run_expand(self, x_ext, adapter_indices, ctx):
        """Accumulate the LoRA delta in place into ``x_ext[:, :N]``.

        All slices go in one launch. Whole-buffer in-place accumulate: x_ext is
        folded as both shrink-read source and base-write target (disjoint
        columns), with a single packed lora_B. Mutating x_ext itself (not its
        ``[:, :N]`` view) keeps the inductor graph glue-free (no clone +
        slice_scatter). ``base_out`` in forward() already aliases
        ``x_ext[:, :N_total]``, so no rebind is needed after this call.
        """
        bitmask = ctx.per_module_bitmasks[self._module_idx]
        granite_switch_lora_expand(
            x_ext,
            adapter_indices,
            bitmask,
            self._lb_packed,
            self.tile_to_slice,
            self.slice_col_r,
            self._na[0],
            self._na[1],
            self._na[2],
            self._na[3],
            self._na[4],
            self._na[5],
            self._S,
            self._block_n,
            self._N_total,
        )


# ---------------------------------------------------------------------------
# Coded-memory MultiSwitch
# ---------------------------------------------------------------------------

# Large negative finite value for masking. NOT literal -inf because IEEE 754
# defines 0 * (+/-)inf = NaN, and the one-hot Q vectors below have zeros at the
# dimensions where K holds the mask value. -1e9 gives 0 * (-1e9) == 0 (clean)
# and exp(-1e9) ~= 0 in softmax (the same masking effect).
_NEG_INF = -1e9


class MultiSwitch(nn.Module):
    """Coded-memory multi-transition switch, on paged-KV attention.

    Performs coarse-grained, multi-transition adapter routing (base to exp1 to
    exp2 back to base, arbitrarily many times per request) with two tiny
    attention heads:

    1. **Counting head (single query head).** A position-0 anchor holds
       ``V=1``; every control token holds ``V=0`` and an un-masked key. A
       one-hot query reads back ``1 / (1 + n)`` where ``n`` is the number of
       control tokens seen so far (causally). ``recover_count_from_signal``
       inverts this to the integer *write address* ``n``. Non-participating
       tokens are masked with a large finite negative key value (``-1e9``, not
       literal ``-inf``: ``0 * -inf`` is NaN under IEEE-754, and the one-hot
       queries have zeros where the mask sits). The count is
       sequence-length-independent: it depends only on how many control tokens
       precede a position, not on absolute position.

    2. **Memory head (single head, Kerdock/DG coded).** At each control token
       the key is ``code(n) * memory_gain`` and the value is the ``expert_id``.
       Every token queries with ``code(n)``. Because Kerdock/DG codewords have
       provably low mutual coherence, the softmax over coded keys concentrates
       on the matching address, so the attention output is the ``expert_id``
       most recently written at the current address. ``round`` plus
       ``clamp[0, num_adapters]`` yields the integer adapter index.

    The codebook is precomputed once into a registered buffer, so the forward
    path does a plain index lookup (``self.codebook[write_addresses]``) - no
    numpy, no lazy init - which keeps the whole switch
    ``@support_torch_compile`` safe. All masking is arithmetic
    (``torch.where`` / multiply), there is no boolean-index assignment and no
    data-dependent branching.

    Token exchange: adapter *selection* reads the ORIGINAL ``input_ids``; at
    the end of ``forward`` the control-token ids are rewritten to their
    substitute ids via a precomputed LUT (``apply_token_exchange``) so the
    decoder embeds a clean sequence and never knows a control token existed.

    Cache layers: this switch constructs two ``Attention`` layers (the counting
    slot and the memory slot), each consuming one KV-cache group. That is why
    ``SWITCH_CACHE_LAYERS`` is 2 and why the on-disk ``num_hidden_layers`` is
    inflated by 2.

    Batching: the counting/memory attention runs over the flat token stream
    vLLM hands the model, and the counting anchor is derived from the
    ``positions`` tensor (``positions == 0``). Because vLLM's ``positions``
    restart at 0 for each request, a flattened multi-request batch carries one
    anchor PER REQUEST, which is exactly what the ``1 / (1 + n)`` counting
    needs. The two heads are real paged-KV ``Attention`` modules, so vLLM
    already confines each request's attention to its own tokens. This only
    holds when the caller forwards the real ``positions``: ``forward``
    therefore REQUIRES it and raises rather than synthesizing an ``arange``,
    which would anchor only the first request in the batch and silently
    misroute the others.

    Decode reads the control token back out of the cache, and that is the
    mechanism rather than a gap. A control token seen during prefill is absent
    from a decode step's ``input_ids``, but because both heads are real
    paged-KV ``Attention`` modules a decode query still attends over the cached
    anchor and control-token keys: the counting head recovers the same ``n``
    from ``1 / (1 + n)``, and the memory head reads back the codeword written
    at address ``n``, and with it the active expert id.

    ``memory_gain``: the default 28.0 (``ms_memory_gain``) is validated for
    exact retrieval at the codebook's full capacity with margin. A gain of 16.0
    is the smallest that still retrieves exactly at capacity; 8.0 does not. If
    you change the code or the gain, re-validate exact retrieval - the right
    value is the smallest gain that retrieves exactly with margin.

    Args:
        config: The checkpoint's ``GraniteSwitchConfig``. Supplies the backbone
            head geometry, the token-exchange substitute ids and the ``ms_*``
            coded-engine parameters.
        vllm_config: The engine config, for dtype and the cache/quant configs
            the two ``Attention`` layers need.

    """

    def __init__(self, config: GraniteSwitchConfig, vllm_config: VllmConfig):
        super().__init__()
        # Index 0 means base/no-adapter, so valid indices are 0..num_adapters.
        self.num_adapters = config.num_adapters
        self.dtype = vllm_config.model_config.dtype

        # Two adapter_token_ids layouts: num_adapters entries, where id[i] fires
        # adapter i+1; or num_adapters + 1, where id[0] is a base-reset token
        # letting a sequence return to base mid-stream (needed for agentic
        # per-step switching or a preserved multi-turn history). Resolved to a
        # shape-derived constant here so forward stays branch-free.
        ctrl_ids = config.adapter_token_ids
        if ctrl_ids is not None and len(ctrl_ids) == self.num_adapters + 1:
            self._expert_id_offset = 0  # base-reset layout; argmax is the id
        else:
            self._expert_id_offset = 1  # no base slot; adapter i+1 for slot i

        self.memory_gain = config.ms_memory_gain

        # The codebook buffer below must stay persistent. Absent from the
        # state_dict it would be left all zeros by a meta-device loader, which
        # zeros every memory key and query, flattens the retrieval softmax to
        # uniform, and averages visible expert ids instead of selecting the most
        # recent - silent misrouting rather than a missing-key error. ~512 KB.
        self.code_gen = KerdockDGCodeGenerator(
            m=config.ms_code_m, code_type=config.ms_code_type
        )
        self.capacity = self.code_gen.capacity
        self.memory_dim = self.code_gen.N
        codebook = self.code_gen.precompute_codebook(dtype=torch.float32)
        self.register_buffer("codebook", codebook, persistent=True)

        # Head geometry. Counting is a SINGLE query head over a SINGLE KV head;
        # both counting and memory use num_heads == num_kv_heads == 1.
        # FlashAttention requires head_size >= 32. For the counting head only
        # dim 0 (the one-hot signal channel) is used; the rest is zero padding.
        self.counting_head_dim = max(int(config.ms_counting_head_dim), 32)

        # The memory head_size must hold the full code vector (>= memory_dim)
        # AND satisfy the >= 32 kernel constraint. Kerdock m=6 gives
        # memory_dim=64, so the code already exceeds 32; extra dims, if any, are
        # zero-padded. Prefer aligning to the backbone projection_head_dim when
        # that is >= memory_dim, so every Attention layer shares one head_size
        # (page-size compatibility); otherwise use memory_dim directly.
        backbone_head_dim = config.projection_head_dim
        if backbone_head_dim is not None and backbone_head_dim >= self.memory_dim:
            self.memory_head_dim = backbone_head_dim
        else:
            self.memory_head_dim = max(self.memory_dim, 32)

        # Single counting plus single memory KV head. Under TP every rank builds
        # identical one-hot / coded Q/K/V locally, and a single head has nothing
        # to shard, so no TP head division is applied.
        self.num_heads = 1
        self.num_kv_heads = 1

        # Counting head: single query head, single KV head.
        self.counting_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.counting_head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="switch.multi.0",
        )

        # Memory head: single head, Kerdock/DG-coded keys for exact retrieval.
        self.memory_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.memory_head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="switch.multi.1",
        )

        # Token-exchange LUT (control id -> substitute id), None if the
        # checkpoint carries no mapping. persistent=True for the same reason as
        # ``codebook`` above: a non-persistent buffer is zeroed by checkpoint
        # loading, and this LUT uses -1 as its "not a control token" sentinel,
        # so an all-zero LUT would rewrite EVERY token id to 0.
        lut = build_control_to_substitute_lut(config)
        if lut is not None:
            self.register_buffer("control_to_substitute_lut", lut, persistent=True)
        else:
            self.control_to_substitute_lut = None

    @property
    def num_cache_layers(self) -> int:
        """KV-cache slots this switch uses: the counting slot plus the memory slot.

        One per ``Attention`` layer built above. ``SWITCH_CACHE_LAYERS`` is the
        same number seen from the config side, where it is what
        ``num_hidden_layers`` on disk is inflated by.
        """
        return SWITCH_CACHE_LAYERS

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token adapter indices and rewrite control tokens.

        Adapter selection reads the ORIGINAL ``input_ids``. Token exchange is
        applied at the very end.

        Args:
            input_ids: ``[total_tokens]`` flattened token ids from the vLLM
                scheduler.
            adapter_token_ids: ``[num_adapters (+1)]`` activating control token
                ids. ``adapter_token_ids[i]`` fires adapter ``i`` (index 0 is
                the base/no-adapter slot when present).
            positions: ``[total_tokens]`` REAL per-request token positions, as
                supplied by the model forward. Each request's first token must
                carry position 0 - that is the ``1 / (1 + n)`` counting anchor.
                Required: passing ``None`` raises, because synthesizing
                ``arange(total_tokens)`` over a flattened batch would anchor
                only the first request and silently misroute the rest.

        Returns:
            ``(adapter_indices, modified_input_ids)``, both ``[total_tokens]``.
            ``adapter_indices`` is 0 for base and 1+ for adapters;
            ``modified_input_ids`` has control ids rewritten to substitute ids.

        Raises:
            ValueError: If ``positions`` is None.

        """
        total_tokens = input_ids.shape[0]
        device = input_ids.device
        # The KV-cache dtype, not a choice: both heads are paged-KV Attention
        # modules, so their Q/K/V must match the cache. A bf16 cache quantizes
        # the 1/(1+n) signal and inverts exactly only up to n = 188 (189
        # aliases); float32 inverts exactly past n = 4095. This is a different
        # bound from the codebook's capacity, which limits the memory head
        # rather than the counting head.
        dtype = self.dtype

        # Position-0 anchor for the 1/(1+n) counting. Derived from positions;
        # NOT KV-hidden.
        #
        # ``positions`` must be the REAL per-request positions. There is
        # deliberately no ``arange(total_tokens)`` fallback: vLLM flattens a
        # batch into one flat tensor, so a fabricated arange places an anchor
        # only in the first request, and every later request counts against a
        # missing baseline, recovers a wrong write address and retrieves an
        # arbitrary adapter. That failure is silent - routing looks plausible
        # and only diverges once a request carries >= 3 control tokens - so a
        # loud error here is much cheaper than the misrouting it replaces.
        if positions is None:
            raise ValueError(
                "MultiSwitch.forward() requires per-request `positions`: the "
                "1/(1+n) counting head places its anchor at `positions == 0`. "
                "vLLM flattens batches into one [total_tokens] tensor, so a "
                "synthesized arange() would give only the first request an "
                "anchor and silently misroute every other request. Pass the "
                "`positions` that the model forward already receives."
            )
        is_counting_anchor = positions == 0  # [total_tokens]

        # Vectorized control-token matching against the ORIGINAL input_ids.
        matches = input_ids.unsqueeze(1) == adapter_token_ids.unsqueeze(0)  # [T, A]
        is_control_token = matches.any(dim=1)  # [total_tokens]
        # expert_id = argmax + offset (the offset selects the layout; see
        # __init__). 0 for non-control tokens.
        expert_ids = torch.where(
            is_control_token,
            matches.long().argmax(dim=1) + self._expert_id_offset,
            torch.zeros_like(input_ids, dtype=torch.long),
        )  # [total_tokens]

        # Step 1: counting via single-head attention -> write address n.
        _zero = torch.tensor(0.0, dtype=dtype, device=device)
        _one = torch.tensor(1.0, dtype=dtype, device=device)

        # Keys: masked (-1e9) by default; un-masked (0) at the anchor and at
        # control tokens, so the one-hot query attends only to those. Arithmetic
        # (torch.where), no boolean-index assignment.
        k_count = torch.full(
            (total_tokens, self.num_kv_heads, self.counting_head_dim),
            _NEG_INF,
            device=device,
            dtype=dtype,
        )
        anchor_or_control = is_counting_anchor | is_control_token
        k_count[:, 0, 0] = torch.where(anchor_or_control, _zero, k_count[:, 0, 0])

        # Values: v=1 at the position-0 anchor only (the 1/(1+n) numerator).
        v_count = torch.zeros(
            (total_tokens, self.num_kv_heads, self.counting_head_dim),
            device=device,
            dtype=dtype,
        )
        v_count[:, 0, 0] = torch.where(is_counting_anchor, _one, _zero)

        # Query: one-hot on dim 0, so Q dot K == K[0] (0 when attended, -1e9
        # when masked).
        q_count = torch.zeros(
            (total_tokens, self.num_heads, self.counting_head_dim),
            device=device,
            dtype=dtype,
        )
        q_count[:, 0, 0] = _one

        count_output = self.counting_attn(q_count, k_count, v_count)
        count_output = count_output.reshape(
            total_tokens, self.num_heads, self.counting_head_dim
        )
        counting_signal = count_output[:, 0, 0]  # [total_tokens] = 1/(1+n)

        # n = round(1/signal - 1), clamped to the codebook capacity.
        write_addresses = recover_count_from_signal(
            counting_signal, capacity=self.capacity
        )

        # Step 2: coded memory -> expert id.
        # Look up the codeword for each token's address (a compile-safe gather).
        all_code_vectors = self.codebook[write_addresses].to(dtype)  # [T, memory_dim]
        write_mask = is_control_token.unsqueeze(-1).to(dtype)  # [T, 1]

        # Keys: code(n) * memory_gain at control tokens, zero elsewhere
        # (arithmetic masking). Values: the expert_id at control tokens.
        k_memory = torch.zeros(
            (total_tokens, self.num_kv_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        k_memory[:, 0, : self.memory_dim] = (
            all_code_vectors * self.memory_gain * write_mask
        )

        v_memory = torch.zeros(
            (total_tokens, self.num_kv_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        v_memory[:, 0, 0] = expert_ids.to(dtype) * is_control_token.to(dtype)

        # Query: code(n) for every token, reading back the value at address n.
        q_memory = torch.zeros(
            (total_tokens, self.num_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        q_memory[:, 0, : self.memory_dim] = all_code_vectors

        memory_output = self.memory_attn(q_memory, k_memory, v_memory)
        memory_output = memory_output.reshape(
            total_tokens, self.num_heads, self.memory_head_dim
        )

        # Step 3: extract, round and clamp the adapter indices.
        adapter_indices = memory_output[:, 0, 0]  # [total_tokens]
        adapter_indices = torch.round(adapter_indices).long()
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        # Token-exchange rewrite (branch-free; runs every step under compile).
        modified_input_ids = apply_token_exchange(
            self.control_to_substitute_lut, input_ids
        )

        return adapter_indices, modified_input_ids


# ---------------------------------------------------------------------------
# LoRA / aLoRA decoder tier
# ---------------------------------------------------------------------------


class GraniteLoRAEmbeddedAttention(nn.Module):
    """Granite attention with conditional LoRA on the QKV and output projections.

    Applies a different adapter to Q, K, V and O per token, driven by the
    adapter indices the switch produced.
    """

    _lora_ctx: LoRAContext | None = None  # Wired post-init by GraniteSwitchModel

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        num_adapters = config.num_adapters
        max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.projection_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.attention_multiplier

        base_qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        if "qkv_proj" in config.lora_target_modules:
            self.qkv_proj = SwitchedLoRALinear(
                base_qkv_proj,
                num_adapters,
                max_lora_rank,
                num_slices=3,
                output_slices=tuple(base_qkv_proj.output_sizes),
            )
        else:
            self.qkv_proj = base_qkv_proj

        base_o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if "o_proj" in config.lora_target_modules:
            self.o_proj = SwitchedLoRALinear(base_o_proj, num_adapters, max_lora_rank)
        else:
            self.o_proj = base_o_proj

        # Rotary embeddings. The switch model is attention-only with RoPE.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

        # head_size is the native projection_head_dim - token exchange does not
        # widen the KV cache.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # SwitchedLoRALinear reads its LoRA metadata from the shared LoRAContext.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


def rms_norm_select(
    norm: RMSNorm,
    block_output: torch.Tensor,
    residual: torch.Tensor | None,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select between the one-arg and two-arg RMSNorm calling conventions.

    vLLM model classes differ in how they combine the residual add with the
    norm. Granite and GraniteMoeHybrid add the residual explicitly and then
    call one-arg ``norm(x)``; Llama/Mistral/Qwen2 call two-arg
    ``norm(x, residual)``, which fuses the addition into a single CUDA kernel.

    Both are mathematically identical, but in bfloat16 the fused kernel rounds
    differently. A checkpoint's weights were fitted against exactly one of the
    two reduction orders, so ``config.fused_add_norm`` records which one and
    this function honors it.

    Args:
        norm: The RMSNorm layer.
        block_output: Output of the attention or MLP block.
        residual: The running residual (None on the very first call).
        fused: True selects the two-arg fused kernel, False a separate add
            followed by the norm.

    Returns:
        ``(hidden_states, residual)``.

    """
    if residual is None:
        # First layer: nothing to add yet.
        residual = block_output
        hidden_states = norm(block_output)
    elif fused:
        hidden_states, residual = norm(block_output, residual)
    else:
        residual = residual + block_output
        hidden_states = norm(residual)
    return hidden_states, residual


def replace_shared_mlp_projections_with_lora(mlp, config):
    """Swap the shared MLP's projections for SwitchedLoRALinear, in place.

    Operates on a ``GraniteMoeSharedMLP``. Returns
    ``(has_input_lora, has_output_lora)``.
    """
    num_adapters = config.num_adapters
    max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
    has_input_lora = False
    has_output_lora = False

    if "shared_input_linear" in config.lora_target_modules:
        base = mlp.input_linear
        mlp.input_linear = SwitchedLoRALinear(
            base,
            num_adapters,
            max_lora_rank,
            num_slices=2,
            output_slices=tuple(base.output_sizes),
            fuse_swiglu=True,
        )
        # The gate/up projection now applies SwiGLU internally and returns the
        # activated [M, H], so the MLP's own activation becomes a pass-through.
        # That removes the separate SiluAndMul launch and its read of the
        # strided gate|up output (the contiguity hazard) entirely.
        mlp.act_fn = nn.Identity()
        has_input_lora = True

    if "shared_output_linear" in config.lora_target_modules:
        base = mlp.output_linear
        mlp.output_linear = SwitchedLoRALinear(
            base,
            num_adapters,
            max_lora_rank,
        )
        has_output_lora = True

    return has_input_lora, has_output_lora


class GraniteSwitchDecoderLayer(nn.Module):
    """Attention decoder layer with switch-determined adapter selection.

    Covers all three MLP shapes Granite ships: a dense shared MLP alone
    (granite 4.0/4.1), a frozen expert bank alongside it (the 4.x MoE hybrid),
    and the expert bank alone (granitemoe, ``shared_intermediate_size == 0``).
    The experts are never LoRA targets in any of them.
    """

    _lora_ctx: LoRAContext | None = None  # Wired post-init by GraniteSwitchModel

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.residual_multiplier = config.residual_multiplier
        self.fused_add_norm = config.fused_add_norm
        self.layer_type = "attention"

        self.self_attn = GraniteLoRAEmbeddedAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        self.has_experts = config.num_local_experts > 0
        if self.has_experts:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        # A pure sparse-MoE base (granitemoe) has no dense shared MLP, encoded
        # as shared_intermediate_size == 0. It must be skipped, not built
        # zero-width: GraniteMoeSharedMLP sizes itself from that value, so it
        # would register [0, H] / [H, 0] weights no checkpoint ships and add
        # their output into the MoE result. Same gate as granitemoehybrid.py.
        self.has_shared_mlp = config.shared_intermediate_size > 0
        if self.has_shared_mlp:
            shared_mlp = GraniteMoeSharedMLP(
                config=config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.shared_mlp",
            )
            self._has_shared_input_lora, self._has_shared_output_lora = (
                replace_shared_mlp_projections_with_lora(shared_mlp, config)
            )
            self.shared_mlp: GraniteMoeSharedMLP | None = shared_mlp
        else:
            if not self.has_experts:
                raise ValueError(
                    "A decoder layer needs at least one MLP path: got "
                    "num_local_experts=0 and shared_intermediate_size=0."
                )
            self.shared_mlp = None
            self._has_shared_input_lora = False
            self._has_shared_output_lora = False

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Match the RMSNorm calling convention the checkpoint was built against
        # (see rms_norm_select).
        hidden_states, residual = rms_norm_select(
            self.input_layernorm,
            hidden_states,
            residual,
            self.fused_add_norm,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = hidden_states * self.residual_multiplier

        hidden_states, residual = rms_norm_select(
            self.post_attention_layernorm,
            hidden_states,
            residual,
            self.fused_add_norm,
        )

        if self.shared_mlp is None:
            # Experts only (granitemoe). Still clone: FusedMoE modifies its
            # input in place, and hidden_states aliases the tensor the caller's
            # residual pair was normed from.
            hidden_states = self.block_sparse_moe(hidden_states.clone())
        elif not self.has_experts:
            hidden_states = self.shared_mlp(hidden_states)
        else:
            moe_output = self.block_sparse_moe(hidden_states.clone())
            hidden_states = moe_output + self.shared_mlp(hidden_states)

        hidden_states = hidden_states * self.residual_multiplier
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Shadow Residual dual-stream tier
# ---------------------------------------------------------------------------


def interleave_q_heads(
    q_base: torch.Tensor,
    q_adapt: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Interleave two per-stream query tensors into one doubled-Q tensor.

    The "even=base / odd=adapter" layout is REQUIRED, not cosmetic. GQA maps
    query head ``m`` to KV head ``m // (num_q_heads // num_kv_heads)``. With the
    query-head count doubled, that group size doubles too, so heads ``2i`` and
    ``2i+1`` both map to the same KV head that vanilla head ``i`` used - i.e.
    base and adapter attend the same base-only K/V. A ``[base..., adapt...]``
    concatenation would instead map the adapter heads onto the wrong KV heads.

    Args:
        q_base: ``[N, num_heads * head_dim]`` base-stream queries.
        q_adapt: ``[N, num_heads * head_dim]`` adapter-stream queries.
        num_heads: Query heads PER STREAM.
        head_dim: Per-head dimension.

    Returns:
        ``[N, 2 * num_heads * head_dim]`` with base head ``i`` at slot ``2i``
        and adapter head ``i`` at slot ``2i+1``.

    """
    n = q_base.shape[0]
    qb = q_base.reshape(n, num_heads, head_dim)
    qa = q_adapt.reshape(n, num_heads, head_dim)
    # New axis between head and dim -> [N, Hq, 2, d]; the row-major flatten
    # merges (Hq, 2) so stream s of head i lands at head slot 2*i + s.
    stacked = torch.stack((qb, qa), dim=2)
    return stacked.reshape(n, 2 * num_heads * head_dim)


def deinterleave_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``interleave_q_heads``.

    Args:
        x: ``[N, 2 * num_heads * head_dim]`` doubled-Q attention output.
        num_heads: Query heads PER STREAM.
        head_dim: Per-head dimension.

    Returns:
        ``(base, adapt)``, each ``[N, num_heads * head_dim]`` - even heads to
        base, odd heads to adapter.

    """
    n = x.shape[0]
    grouped = x.reshape(n, num_heads, 2, head_dim)
    base = grouped[:, :, 0, :].reshape(n, num_heads * head_dim)
    adapt = grouped[:, :, 1, :].reshape(n, num_heads * head_dim)
    return base, adapt


class WCrossShunt(nn.Module):
    """Shrink-only cross-stream shunt (base -> adapter) for one SR layer.

    ``W_cross`` is a "W-less" LoRA module: it has NO base weight. Its output is
    purely ``(h_base @ lora_A_cross.T) @ lora_B_cross.T`` for adapter tokens and
    exactly zero for base tokens. It uses the shrink-only SWITCH kernel
    (``granite_switch_lora_shrink_expand``), so - unlike a SwitchedLoRALinear
    over a zeroed base - it does NOT pay a full ``[H, H]`` base GEMM; its only
    matmul is the small shrink projection ``h_base @ w_ext_cross.T`` (width =
    sum of applicable cross ranks).

    It reads the SR context's M-length REAL-id shunt metadata
    (``remapped_indices_shunt`` / ``per_module_bitmasks_shunt``), NOT the 2M
    base-half zeros; see ``SRFusedLoRAKernelMeta``.

    On-disk layout mirrors a single-slice SwitchedLoRALinear so the
    ``cross_stream.lora_A`` / ``cross_stream.lora_B`` tensors load name-based:
    ``lora_A [NA, 1, cross_rank, H]``, ``lora_B [NA, 1, H, cross_rank]``.
    """

    # Both wired post-init by GraniteSwitchModel. Declared at class level so the
    # attributes exist statically rather than being added per instance.
    _lora_ctx: SRLoRAContext | None = None
    _module_idx: int = -1

    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        cross_rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_adapters = num_adapters
        self.cross_rank = cross_rank
        self._device = device
        self._dtype = dtype

        # Checkpoint-format params (single-slice layout; consumed by finalize).
        self.lora_A = nn.Parameter(
            torch.zeros(
                num_adapters, 1, cross_rank, hidden_size, dtype=dtype, device=device
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                num_adapters, 1, hidden_size, cross_rank, dtype=dtype, device=device
            )
        )
        self._finalized = False

    def finalize_weights(
        self, adapter_ranks: list[int], block_n: int | None = None
    ) -> None:
        """Build the shrink-only fused kernel state from the loaded LoRA weights.

        Mirrors ``SwitchedLoRALinear.finalize_weights`` for a single slice
        (``S == 1``, ``N_total == hidden_size``) EXCEPT that the base region is
        empty: ``w_ext_cross`` is built from an empty ``[0, H]`` base, so the
        shrink columns - and therefore ``slice_col_r`` - start at 0.

        Args:
            adapter_ranks: Per-adapter cross rank, typically
                ``[cross_stream_rank] * num_adapters``.
            block_n: Output-tile width; defaults to the kernel's ``BLOCK_N``.

        """
        if self._finalized:
            return
        if block_n is None:
            block_n = BLOCK_N

        device, dtype = self._device, self._dtype
        NA = self.num_adapters
        H = self.hidden_size
        S = 1

        lora_A = self.lora_A.data  # [NA, 1, max_rank, H]
        lora_B = self.lora_B.data  # [NA, 1, H, max_rank]
        max_rank = lora_A.shape[2]

        # Coverage: adapters whose lora_A is all-zero over their first r rows do
        # not apply here. One batched reduction plus a single host sync, as in
        # SwitchedLoRALinear.
        ranks_t = torch.tensor(adapter_ranks, device=device)  # [NA]
        rank_mask = torch.arange(max_rank, device=device)[None, :] < ranks_t[:, None]
        nz = lora_A[:, 0, :, :].ne(0).any(dim=-1)  # [NA, max_rank]
        applicable = (nz & rank_mask).any(dim=1).tolist()

        applicable_adapters = [i for i in range(NA) if applicable[i]]

        # Off-tier cross ranks are promoted to the next supported tier and the
        # lora_A rows / lora_B columns zero-padded to match - see promote_rank
        # and the matching block in SwitchedLoRALinear.finalize_weights.
        eff_ranks = [promote_rank(r) for r in adapter_ranks]
        pad_to = max((eff_ranks[i] for i in applicable_adapters), default=0)
        if pad_to > max_rank:
            pad = pad_to - max_rank
            lora_A = torch.nn.functional.pad(lora_A, (0, 0, 0, pad))  # rank dim -2
            lora_B = torch.nn.functional.pad(lora_B, (0, pad))  # rank dim -1
        adapter_ranks = eff_ranks

        rank_order = sorted(applicable_adapters, key=lambda i: adapter_ranks[i])

        remap = torch.zeros(NA + 1, dtype=torch.long, device=device)
        for kernel_idx, orig_idx in enumerate(rank_order):
            remap[orig_idx + 1] = kernel_idx + 1
        self.register_buffer("remap_table", remap, persistent=False)

        tiers: dict[int, list[int]] = {}
        for orig_idx in rank_order:
            tiers.setdefault(adapter_ranks[orig_idx], []).append(orig_idx)

        # lora_A_by_rank: {rank: [n_r, S=1, rank, H]}
        lora_A_by_rank: dict[int, torch.Tensor] = {}
        for rank, orig_indices in tiers.items():
            A_list = [
                torch.stack([lora_A[oi, 0, :rank, :]], dim=0)  # [1, rank, H]
                for oi in orig_indices
            ]
            lora_A_by_rank[rank] = torch.stack(A_list, dim=0)  # [n_r, 1, rank, H]

        # w_ext_cross: EMPTY base [0, H] followed by shrink rows only.
        empty_base = torch.empty(0, H, device=device, dtype=dtype)
        w_ext_cross = build_w_ext(empty_base, lora_A_by_rank)  # [sum_r n_r*rank, H]
        self.register_buffer("w_ext_cross", w_ext_cross, persistent=False)

        # lora_B_merged per tier: {rank: [n_r, H, rank]} (single slice, N_total = H).
        tier_info: dict[int, int] = {}
        lora_B_merged: dict[int, torch.Tensor] = {}
        for rank, orig_indices in tiers.items():
            B_list = [lora_B[oi, 0, :, :rank] for oi in orig_indices]  # each [H, rank]
            lora_B_merged[rank] = torch.stack(B_list, dim=0)  # [n_r, H, rank]
            tier_info[rank] = len(orig_indices)

        self._num_applicable = sum(tier_info.values())
        self._N = H

        if H % block_n != 0:
            raise ValueError(
                f"block_n={block_n} must divide hidden_size={H}. Otherwise an "
                "output tile straddles the end of the single output slice."
            )
        self._block_n = block_n

        # Single slice -> tile_to_slice is all zeros; N_total = H.
        num_tiles_N = (H + block_n - 1) // block_n
        self.register_buffer(
            "tile_to_slice",
            torch.zeros(num_tiles_N, dtype=torch.int32, device=device),
            persistent=False,
        )

        # slice_col_r [S=1, 6]: per-tier shrink-column base, starting at 0
        # because there is no base region.
        tier_col_bases = []
        offset = 0
        for r in SUPPORTED_RANKS:
            tier_col_bases.append(offset)
            offset += tier_info.get(r, 0) * S * r
        slice_col_r_data = torch.tensor(
            [[tier_col_bases[t] for t in range(len(SUPPORTED_RANKS))]],  # s == 0
            dtype=torch.int32,
            device=device,
        )  # [1, 6]
        self.register_buffer("slice_col_r", slice_col_r_data, persistent=False)

        # Packed lora_B: contiguous concat over present tiers of [n_r, H, r]
        # row-major.
        self._na = tuple(tier_info.get(r, 0) for r in SUPPORTED_RANKS)
        self._S = S
        packed_parts = [
            lora_B_merged[r].reshape(-1) for r in SUPPORTED_RANKS if r in lora_B_merged
        ]
        lb_packed = (
            torch.cat(packed_parts)
            if packed_parts
            else torch.zeros(1, device=device, dtype=dtype)
        )
        self.register_buffer("_lb_packed", lb_packed.contiguous(), persistent=False)

        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        self._finalized = True

    def forward(self, h_base: torch.Tensor) -> torch.Tensor:
        """Shunt delta for the M base-half rows: ``[M, H]``, zero for base tokens."""
        if not self._finalized:
            raise RuntimeError("finalize_weights() must be called before forward()")
        M = h_base.shape[0]
        out = torch.empty(M, self._N, device=h_base.device, dtype=h_base.dtype)

        ctx = self._lora_ctx
        remapped = ctx.remapped_indices_shunt if ctx is not None else None
        bitmasks = ctx.per_module_bitmasks_shunt if ctx is not None else None
        if remapped is not None and bitmasks is not None and self._num_applicable > 0:
            # Keyed by the M-length REAL adapter ids: base tokens (id 0) map to
            # kernel-local 0, for which the shrink kernel emits exactly zero.
            adapter_indices = remapped[self._module_idx, :M]
            bitmask = bitmasks[self._module_idx]
            x_ext = torch.mm(h_base, self.w_ext_cross.T)  # [M, sum_r n_r*rank]
            # The op takes NA_16..NA_512 as separate ints, so unpack the tuple
            # here - exactly as SwitchedLoRALinear._run_expand does.
            granite_switch_lora_shrink_expand(
                out,
                x_ext,
                adapter_indices,
                bitmask,
                self._lb_packed,
                self.tile_to_slice,
                self.slice_col_r,
                self._na[0],
                self._na[1],
                self._na[2],
                self._na[3],
                self._na[4],
                self._na[5],
                self._S,
                self._block_n,
                self._N,
            )
        else:
            # No applicable adapters / no kernel meta: the shunt contributes
            # nothing.
            out.zero_()
        return out


@contextlib.contextmanager
def _doubled_padding_mask(m: int):
    """Match the forward context's ``is_padding`` to SR's doubled router batch.

    The fused-MoE top-k router reads ``is_padding`` off the forward context and
    slices it to the number of gating rows, and the kernel then requires the
    lengths to agree. The runner sizes the mask to the padded token count ``m``,
    but SR routes the whole ``[2M, H]`` stack in one expert call, so the kernel
    would see ``2 * m`` gating rows against an ``m``-entry mask.

    Doubling the mask is the correct answer rather than a workaround: adapter row
    ``i`` is the same token as base row ``i``, so it is padding exactly when the
    base row is.

    Inert where there is nothing to match: no forward context, no mask, or a mask
    whose length is not ``m``. In that last case the premise of the doubling does
    not hold, and failing loudly in the kernel beats silently masking the wrong
    rows.
    """
    ctx = get_forward_context() if is_forward_context_available() else None
    if ctx is None:
        yield
        return
    mask = ctx.is_padding
    if mask is None or mask.shape[0] != m:
        yield
        return
    ctx.is_padding = torch.cat([mask, mask], dim=0)
    try:
        yield
    finally:
        ctx.is_padding = mask


class ShadowResidualAttention(nn.Module):
    """Doubled-Q / base-only-KV attention for one Shadow Residual layer.

    Every projection is a ``SwitchedLoRALinear`` run ONCE over the ``[2M, H]``
    stack. The base half carries kernel-local id 0 (via the 2M metadata on the
    shared ``SRLoRAContext``), so it gets no delta - a pristine base projection
    and a base-only K/V. The adapter half carries the real id and gets the
    adapter delta.
    """

    _lora_ctx: SRLoRAContext | None = None  # Wired post-init by GraniteSwitchModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        tp_size = get_tensor_model_parallel_world_size()

        num_adapters = config.num_adapters
        max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0

        self.hidden_size = config.hidden_size
        # As in GraniteLoRAEmbeddedAttention: parallel linears take TOTAL head
        # counts (they shard internally); the doubled-Q glue and qkv split use
        # LOCAL counts. SR's [2M, H] stack is on the token axis, orthogonal to
        # head sharding, so both halves shard identically. Divisible-KV and
        # replicated-KV (num_key_value_heads < tp_size) are both supported.
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.projection_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_q_size = self.total_num_heads * self.head_dim
        self.scaling = config.attention_multiplier

        base_qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.qkv_proj = SwitchedLoRALinear(
            base_qkv,
            num_adapters,
            max_lora_rank,
            num_slices=3,
            output_slices=tuple(base_qkv.output_sizes),
        )

        base_o = RowParallelLinear(
            self.total_q_size,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.o_proj = SwitchedLoRALinear(base_o, num_adapters, max_lora_rank)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

        # Doubled query heads (base + adapter interleaved) against base-only K/V.
        self.attn = Attention(
            2 * self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """Run the dual-stream attention.

        Args:
            positions: ``[M]`` per-request token positions.
            normed: Pre-normed stacked stream ``[2M, H]``.

        Returns:
            ``[2M, H]``.

        """
        m = normed.shape[0] // 2

        # One fused qkv GEMM+expand over the 2M stack. The base half (id 0) gets
        # the base projection; the adapter half gets adapted Q. The K/V LoRA
        # slices are zero, so the adapter half's K/V equal the base K/V - and we
        # take the base half anyway.
        qkv, _ = self.qkv_proj(normed)
        q_2m, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_base, v_base = k[:m].contiguous(), v[:m].contiguous()  # base-only K/V

        q_dbl = interleave_q_heads(
            q_2m[:m], q_2m[m:], self.num_heads, self.head_dim
        )  # [M, 2 * q_size]

        q_dbl, k_base = self.rotary_emb(positions, q_dbl, k_base)

        attn = self.attn(q_dbl, k_base, v_base)  # ONE call, base-only KV
        attn_base, attn_adapt = deinterleave_heads(attn, self.num_heads, self.head_dim)
        attn_stacked = torch.cat([attn_base, attn_adapt], dim=0)  # [2M, q_size]

        o, _ = self.o_proj(attn_stacked)  # base + adapter-only O delta
        return o


class ShadowResidualDecoderLayer(nn.Module):
    """One Shadow Residual decoder layer (dual-stream, stacked on the token dim).

    Granite's non-fused residual convention, materialized: each block runs on
    ``norm(hs)`` and adds ``block * residual_multiplier`` back to ``hs``. The
    base half never receives a delta or a shunt, so it stays base-equivalent.

    Covers all three MLP shapes Granite ships: a dense shared MLP alone (granite
    4.0/4.1), a frozen expert bank alongside it (the 4.x MoE hybrid), and the
    expert bank alone (granitemoe, ``shared_intermediate_size == 0``). With
    experts, the adapter stream always inherits the base stream's expert
    assignment. Only *routing* is ever shared - the shared MLP, where one exists,
    always runs per-stream with its own adapter context.
    """

    _lora_ctx: SRLoRAContext | None = None  # Wired post-init by GraniteSwitchModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        num_adapters = config.num_adapters
        cross_rank = int(config.cross_stream_rank or 0)
        hidden = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        self.layer_type = "attention"

        self.self_attn = ShadowResidualAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        # Routed expert bank (the 4.x MoE hybrid, and the ONLY MLP path on a pure
        # sparse base like granitemoe). Frozen: the experts are never LoRA
        # targets, so there is no SwitchedLoRALinear wrapping here.
        self.has_experts = config.num_local_experts > 0
        if self.has_experts:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=hidden,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        # The dense shared MLP is ABSENT on a pure sparse MoE base (granitemoe),
        # which is encoded as shared_intermediate_size == 0. Building it anyway
        # would register zero-width [0, H] / [H, 0] weights that no checkpoint
        # ships. Same gate as GraniteSwitchDecoderLayer.
        self.has_shared_mlp = config.shared_intermediate_size > 0
        if self.has_shared_mlp:
            # Fused shared MLP (gate|up with in-kernel SwiGLU, plus down), each
            # wrapped in SwitchedLoRALinear over the [2M, H] stack; the base
            # half gets no delta. Wrapped unconditionally, unlike
            # replace_shared_mlp_projections_with_lora: a checkpoint without MLP
            # LoRA just leaves the tiers zero, which is the same result.
            shared_mlp = GraniteMoeSharedMLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
            )
            max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
            base_in = shared_mlp.input_linear
            shared_mlp.input_linear = SwitchedLoRALinear(
                base_in,
                num_adapters,
                max_lora_rank,
                num_slices=2,
                output_slices=tuple(base_in.output_sizes),
                fuse_swiglu=True,
            )
            # gate/up now applies SwiGLU in-kernel and returns the activated
            # [2M, H], so the MLP's own activation becomes a pass-through.
            shared_mlp.act_fn = nn.Identity()
            base_out = shared_mlp.output_linear
            shared_mlp.output_linear = SwitchedLoRALinear(
                base_out, num_adapters, max_lora_rank
            )
            self.shared_mlp: GraniteMoeSharedMLP | None = shared_mlp
        elif not self.has_experts:
            raise ValueError(
                "A decoder layer needs at least one MLP path: got "
                "num_local_experts=0 and shared_intermediate_size=0."
            )
        else:
            self.shared_mlp = None

        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

        # Cross-stream shunt (base -> adapter), shrink-only kernel, keyed by the
        # real adapter ids.
        self.cross_stream = WCrossShunt(
            hidden_size=hidden,
            num_adapters=num_adapters,
            cross_rank=cross_rank,
            device=vllm_config.device_config.device,
            dtype=vllm_config.model_config.dtype,
        )

    def forward(self, positions: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        """Run one dual-stream layer.

        Args:
            positions: ``[M]`` per-request token positions.
            hs: Stacked ``[2M, H]``, base rows then adapter rows.

        Returns:
            ``[2M, H]``.

        """
        m = hs.shape[0] // 2

        normed = self.input_layernorm(hs)
        o = self.self_attn(positions, normed)
        hs = hs + o * self.residual_multiplier

        normed = self.post_attention_layernorm(hs)
        mlp_out = None

        if self.has_experts:
            # FusedMoE takes router_logits as an argument and derives top-k and
            # the renormalized gates itself, so duplicating the base half's raw
            # logits is equivalent to sharing post-softmax routing and routes
            # the whole [2M, H] stack in one expert call. GraniteMoeMoE.forward
            # is bypassed to inject them.
            logits, _ = self.block_sparse_moe.gate(normed[:m])  # [M, E]
            logits = torch.cat([logits, logits], dim=0)  # [2M, E]
            # FusedMoE modifies its input in place, hence the clone. The doubled
            # rows also need a doubled padding mask.
            with _doubled_padding_mask(m):
                mlp_out = self.block_sparse_moe.experts(normed.clone(), logits)

        if self.shared_mlp is not None:
            # Only ROUTING is shared. The dense shared MLP still runs per-stream
            # with this stream's own adapter context (SwitchedLoRALinear inside;
            # the base half gets no delta).
            shared = self.shared_mlp(normed)
            mlp_out = shared if mlp_out is None else mlp_out + shared

        hs = hs + mlp_out * self.residual_multiplier

        # base -> adapter injection over the M base-half rows (adapter-active
        # tokens only).
        cs = self.cross_stream(hs[:m])  # [M, H]
        hs = torch.cat([hs[:m], hs[m:] + cs], dim=0)
        return hs


# ---------------------------------------------------------------------------
# Decoder interfaces
#
# LoRA/aLoRA and Shadow Residual are two adaptations of the same host model,
# each reached through a DecoderInterface held as self.decoder_interface, so the
# shared __init__ and forward never branch on which is in use. The split is
# confined to the decoder tier: layer type, kernel-metadata layout, ctx-wire
# types, weight-fuse rules, and for SR the [M, H] -> [2M, H] stream doubling and
# terminal merge.
# ---------------------------------------------------------------------------

# Skipped when loading a Shadow Residual checkpoint: cross_stream.base_layer
# (WCrossShunt is W-less, but a checkpoint may carry a zeros base for it) and the
# two LUTs, which are regenerated from config.
_SR_SKIP_SUBSTRINGS = (
    "adapter_token_ids",
    "control_to_substitute_lut",
    ".cross_stream.base_layer.",
)

# LoRA deltas are constructed zeroed, so a checkpoint omitting one is correct -
# that adapter does not target the slice. Nothing else may go unloaded: an
# unloaded base weight, expert bank or norm is uninitialized memory, which
# serves plausible garbage instead of raising.
_ZERO_INIT_PARAM_MARKERS = (".lora_A", ".lora_B")


def _audit_loaded(params_dict, loaded_params, model, *, label: str) -> None:
    """Raise if a non-zero-init parameter went unloaded; warn about the rest."""
    unloaded = [
        n
        for n in params_dict
        if n not in loaded_params and not is_pp_missing_parameter(n, model)
    ]
    if not unloaded:
        return

    uninitialized = [
        n for n in unloaded if not any(m in n for m in _ZERO_INIT_PARAM_MARKERS)
    ]
    if uninitialized:
        shown = "\n".join(f"  - {n}" for n in uninitialized[:20])
        more = (
            f"\n  ... and {len(uninitialized) - 20} more"
            if len(uninitialized) > 20
            else ""
        )
        raise ValueError(
            f"{label}: {len(uninitialized)} parameter(s) absent from the "
            f"checkpoint would be served UNINITIALIZED:\n{shown}{more}"
        )

    logger.warning(
        "%s: %d LoRA delta parameter(s) absent from the checkpoint; they stay "
        "zero (no delta), as expected when an adapter does not target that "
        "slice:\n%s",
        label,
        len(unloaded),
        "\n".join(f"  - {n}" for n in unloaded[:10]),
    )


# Stacked MoE -> FusedMoE. The checkpoint stacks the expert bank; FusedMoE
# wants per-expert shards addressed by (shard_id, expert_id):
#
#   block_sparse_moe.experts.gate_up_proj [E, 2I, H]
#       -> experts.routed_experts.w13_weight (w1|w3)
#   block_sparse_moe.experts.down_proj    [E, H, I]
#       -> experts.routed_experts.w2_weight  (w2)
#   block_sparse_moe.router.weight        [E, H]
#       -> block_sparse_moe.gate.weight
#
# The ``routed_experts.`` segment comes from FusedMoEFactory. Identical for SR.
_MOE_INPUT_SUFFIX = ".block_sparse_moe.experts.gate_up_proj"
_MOE_OUTPUT_SUFFIX = ".block_sparse_moe.experts.down_proj"
_MOE_ROUTER_SUFFIX = ".block_sparse_moe.router.weight"


def _try_load_stacked_moe(name, loaded_weight, params_dict, loaded_params, model):
    """Remap one stacked-MoE checkpoint tensor onto vLLM's FusedMoE.

    Returns:
        True if ``name`` was an MoE tensor and the caller should move on; False
        if the caller should fall through to its own direct-by-name load.

    """

    def _load_expert(param_name, shard, weight_name, shard_id, expert_id):
        if is_pp_missing_parameter(param_name, model):
            return
        if param_name not in params_dict:
            return
        param = params_dict[param_name]
        param.weight_loader(
            param,
            shard,
            weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        loaded_params.add(param_name)

    if name.endswith(_MOE_INPUT_SUFFIX):
        # gate|up are concatenated on dim 0 of each expert: w1 = gate, w3 = up.
        w13_param = name.replace(
            ".experts.gate_up_proj", ".experts.routed_experts.w13_weight"
        )
        for e in range(loaded_weight.size(0)):
            w1, w3 = loaded_weight[e].chunk(2, dim=0)
            for shard, shard_id in ((w1, "w1"), (w3, "w3")):
                _load_expert(
                    w13_param,
                    shard,
                    name.replace(
                        _MOE_INPUT_SUFFIX,
                        f".block_sparse_moe.experts.{e}.{shard_id}.weight",
                    ),
                    shard_id=shard_id,
                    expert_id=e,
                )
        return True

    if name.endswith(_MOE_OUTPUT_SUFFIX):
        w2_param = name.replace(
            ".experts.down_proj", ".experts.routed_experts.w2_weight"
        )
        for e in range(loaded_weight.size(0)):
            _load_expert(
                w2_param,
                loaded_weight[e],
                name.replace(
                    _MOE_OUTPUT_SUFFIX,
                    f".block_sparse_moe.experts.{e}.w2.weight",
                ),
                shard_id="w2",
                expert_id=e,
            )
        return True

    if name.endswith(_MOE_ROUTER_SUFFIX):
        gate_name = name.replace(_MOE_ROUTER_SUFFIX, ".block_sparse_moe.gate.weight")
        if not is_pp_missing_parameter(gate_name, model) and gate_name in params_dict:
            param = params_dict[gate_name]
            getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
            loaded_params.add(gate_name)
        return True

    return False


@dataclass
class LoRAStackState:
    """Threaded through the LoRA decoder loop: the fused/separate residual pair."""

    hidden_states: torch.Tensor
    residual: torch.Tensor | None


@dataclass
class SRStackState:
    """Threaded through the SR decoder loop: the stacked ``[2M, H]`` stream."""

    hs: torch.Tensor


class DecoderInterface(abc.ABC):
    """Hooks that let the shared model build and run either adaptation.

    Build-time hooks are called from ``GraniteSwitchModel.__init__`` and
    ``GraniteSwitchForCausalLM``; forward-time hooks from
    ``GraniteSwitchModel.forward``. All are pure factories or pure tensor
    transforms - none mutate the model.
    """

    # ---- build-time ----
    @abc.abstractmethod
    def make_kernel_meta(self, device) -> tuple[FusedLoRAKernelMeta, LoRAContext]:
        """Build the (kernel-meta, ctx) pair for this adaptation."""

    @abc.abstractmethod
    def make_decoder_layer(self, vllm_config: VllmConfig, prefix: str) -> nn.Module:
        """Build one decoder layer for this adaptation."""

    @abc.abstractmethod
    def ctx_wire_types(self) -> tuple[type, ...]:
        """Module types onto which the shared LoRA ctx is wired."""

    @abc.abstractmethod
    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        """Populate the ctx kernel metadata from per-token ``adapter_indices``."""

    @abc.abstractmethod
    def load_weights(self, model, weights: Iterable[tuple[str, torch.Tensor]]) -> set:
        """Apply checkpoint weights into ``model``.

        Owns the whole per-adaptation weight-application decision, not just a
        one-to-one name rename: LoRA fans stacked-MoE tensors into per-expert
        FusedMoE loads; SR additionally marks the intentionally-absent shared-KV
        LoRA slices loaded. ``finalize_modules`` runs afterwards, driven by
        ``GraniteSwitchForCausalLM.process_weights_after_loading``.

        Args:
            model: The ``GraniteSwitchForCausalLM`` (so ``named_parameters`` and
                ``is_pp_missing_parameter`` are reachable).
            weights: ``(name, tensor)`` pairs from the checkpoint.

        Returns:
            The set of loaded parameter names.

        """

    @abc.abstractmethod
    def finalize_modules(self, model, config) -> None:
        """Post-load: finalize fused kernel state and register remap tables.

        Args:
            model: The ``GraniteSwitchForCausalLM``, so that both ``.modules()``
                and ``.model.lora_meta`` are reachable.
            config: The ``GraniteSwitchConfig``.

        """

    # ---- forward-time ----
    @abc.abstractmethod
    def enter_decoder_stack(self, hidden_states, residual):
        """Turn the embedded/handed-off tensors into this adaptation's state."""

    @abc.abstractmethod
    def run_layer(self, layer, positions, state):
        """Run one decoder layer, threading and returning the state."""

    @abc.abstractmethod
    def to_intermediate(self, state) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Collapse the state into two token-leading tensors for the PP wire.

        Returns:
            ``(hidden_states, residual)`` - both token-leading ``[M, H]`` (or
            ``residual=None``), so vLLM's per-token ``IntermediateTensors`` slice
            is safe.

        """

    @abc.abstractmethod
    def exit_decoder_stack(self, state, adapter_indices, norm, config) -> torch.Tensor:
        """Last-rank finalize: produce the final ``[M, H]`` hidden states."""


class LoRADecoderInterface(DecoderInterface):
    """Single-stream LoRA/aLoRA - the host default."""

    def make_kernel_meta(self, device):
        return FusedLoRAKernelMeta(device=device), LoRAContext()

    def make_decoder_layer(self, vllm_config, prefix):
        return GraniteSwitchDecoderLayer(vllm_config=vllm_config, prefix=prefix)

    def ctx_wire_types(self):
        return (
            SwitchedLoRALinear,
            GraniteLoRAEmbeddedAttention,
            GraniteSwitchDecoderLayer,
        )

    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        lora_meta.prepare_and_store(adapter_indices, lora_ctx)

    def load_weights(self, model, weights):
        """Load the checkpoint weights for a LoRA/aLoRA Granite Switch model.

        Every parameter name matches this model exactly and loads directly,
        except the expert bank, which is stored stacked and must be fanned out
        into per-expert FusedMoE shards (see ``_try_load_stacked_moe``).
        """
        params_dict = dict(model.named_parameters())
        loaded_params: set = set()

        def _load_direct(name, loaded_weight):
            if name.endswith(".bias") and name not in params_dict:
                return
            if is_pp_missing_parameter(name, model):
                return
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        for name, loaded_weight in weights:
            # Stacked MoE -> FusedMoE (shared with the SR loader).
            if _try_load_stacked_moe(
                name, loaded_weight, params_dict, loaded_params, model
            ):
                continue
            # Direct load: every other weight.
            _load_direct(name, loaded_weight)

        _audit_loaded(params_dict, loaded_params, model, label="LoRA weight load")
        return loaded_params

    def finalize_modules(self, model, config) -> None:
        if config.adapter_ranks is None:
            return
        adapter_ranks = config.adapter_ranks
        for module in model.modules():
            if isinstance(module, SwitchedLoRALinear):
                module.finalize_weights(adapter_ranks)

        # Assign sequential indices to every SwitchedLoRALinear and register
        # their remap tables, so lora_meta can compute exact per-module bitmasks
        # at forward time.
        if model.model.lora_meta is not None:
            lora_modules = [
                m for m in model.modules() if isinstance(m, SwitchedLoRALinear)
            ]
            for idx, m in enumerate(lora_modules):
                m._module_idx = idx
            all_remap_tables = torch.stack([m.remap_table for m in lora_modules], dim=0)
            model.model.lora_meta.register_remap_tables(all_remap_tables)

    def enter_decoder_stack(self, hidden_states, residual):
        return LoRAStackState(hidden_states=hidden_states, residual=residual)

    def run_layer(self, layer, positions, state):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=state.hidden_states,
            residual=state.residual,
        )
        return LoRAStackState(hidden_states=hidden_states, residual=residual)

    def to_intermediate(self, state):
        return state.hidden_states, state.residual

    def exit_decoder_stack(self, state, adapter_indices, norm, config):
        # Fold in the last residual through rms_norm_select, so the same
        # fused/separate convention is used throughout.
        hidden_states, _ = rms_norm_select(
            norm,
            state.hidden_states,
            state.residual,
            config.fused_add_norm,
        )
        return hidden_states


class SRDecoderInterface(DecoderInterface):
    """Dual-stream Shadow Residual: base ++ adapter ``[2M, H]``, merged per token.

    The doubling is per *stream*, not per adapter: the adapter half carries each
    token's own real adapter id, and K/V always comes from the base half, so
    tokens on different adapters never reach each other through attention.

    The ``[2M, H]`` stack lives strictly intra-rank. At a PP boundary only two
    token-leading ``[M, H]`` halves cross (via ``to_intermediate``); each rank
    re-stacks them in ``enter_decoder_stack``. Nothing ``2M`` is ever sent, so
    vLLM's per-token ``IntermediateTensors`` slice stays correct.
    """

    def make_kernel_meta(self, device):
        return SRFusedLoRAKernelMeta(device=device), SRLoRAContext()

    def make_decoder_layer(self, vllm_config, prefix):
        return ShadowResidualDecoderLayer(vllm_config=vllm_config, prefix=prefix)

    def ctx_wire_types(self):
        # SwitchedLoRALinear projections read the 2M metadata; WCrossShunt reads
        # the M-length real-id metadata. Both index the same tables by
        # _module_idx.
        return (SwitchedLoRALinear, WCrossShunt)

    def prepare_kernel_meta(self, lora_meta, adapter_indices, lora_ctx) -> None:
        # Prepares BOTH layouts (2M for the projections, M-real for the shunt)
        # from the token-leading [M] real ids, so every PP rank reconstructs the
        # kernel metadata locally.
        lora_meta.prepare_and_store_sr(adapter_indices, lora_ctx)

    def load_weights(self, model, weights):
        """Direct-by-name load of a pre-fused Shadow Residual checkpoint.

        An on-disk SR checkpoint carries q/k/v pre-fused into ``qkv_proj`` and
        gate/up pre-fused into ``shared_mlp.input_linear`` (the config rejects
        ``unfused_qkv=True``), so its parameter names match this model 1:1. Every
        tensor loads directly by name through its own vLLM ``weight_loader`` -
        the fused base tensor's shape already matches ``*.base_layer.weight``.
        There is no fuse-at-load shard logic.

        Three SR-specific behaviours layer on top of the plain direct load:

        * skip ``_SR_SKIP_SUBSTRINGS`` (``WCrossShunt`` has no ``base_layer``;
          the config-regenerated buffers);
        * mark the fused-qkv K/V LoRA slices loaded - they carry no delta
          (``lora_B`` is zero), so vLLM's strict init check must accept them
          whether or not the checkpoint ships them;
        * remap the stacked expert bank onto FusedMoE through
          ``_try_load_stacked_moe``. The experts are frozen and never LoRA
          targets, so an SR checkpoint carries them in exactly the stacked layout
          the LoRA loader sees - the remap is shared, not duplicated.
        """
        params = dict(model.named_parameters())
        loaded: set = set()

        def _load(pname: str, w) -> bool:
            if pname not in params:
                return False
            if is_pp_missing_parameter(pname, model):
                loaded.add(pname)
                return True
            param = params[pname]
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, w)
            loaded.add(pname)
            return True

        for name, w in weights:
            if any(s in name for s in _SR_SKIP_SUBSTRINGS):
                continue
            if name.endswith(".bias") and name not in params:
                continue
            if _try_load_stacked_moe(name, w, params, loaded, model):
                continue
            _load(name, w)

        # Shared K/V: the K/V slices of the fused qkv LoRA carry no delta
        # (lora_B is zero), so mark them loaded - vLLM's strict init check must
        # accept them whether or not a checkpoint ships them.
        for pname in params:
            if (
                ".self_attn.qkv_proj.lora_A_slices.1" in pname
                or ".self_attn.qkv_proj.lora_A_slices.2" in pname
                or ".self_attn.qkv_proj.lora_B_slices.1" in pname
                or ".self_attn.qkv_proj.lora_B_slices.2" in pname
            ):
                loaded.add(pname)

        _audit_loaded(params, loaded, model, label="SR weight load")
        return loaded

    def finalize_modules(self, model, config) -> None:
        adapter_ranks = config.adapter_ranks
        if adapter_ranks is None:
            return
        cross_rank = int(config.cross_stream_rank or 0)
        cross_ranks = [cross_rank] * config.num_adapters

        sll = [m for m in model.modules() if isinstance(m, SwitchedLoRALinear)]
        for m in sll:
            m.finalize_weights(adapter_ranks)
        shunts = [m for m in model.modules() if isinstance(m, WCrossShunt)]
        for m in shunts:
            m.finalize_weights(cross_ranks)

        # _module_idx is shared across BOTH the projections and the shunts, so
        # the SR kernel meta can index them by _module_idx in the 2M and the
        # M-real layout alike.
        if model.model.lora_meta is not None:
            all_modules = sll + shunts
            for idx, m in enumerate(all_modules):
                m._module_idx = idx
            all_remap_tables = torch.stack([m.remap_table for m in all_modules], dim=0)
            model.model.lora_meta.register_remap_tables(all_remap_tables)

    def enter_decoder_stack(self, hidden_states, residual):
        # First rank: a single embedding stream, doubled (base ++ adapter, equal
        # pre-invocation). Later ranks: hidden_states is the base half and
        # residual the adapter half (see to_intermediate), so re-stack them.
        if residual is None:
            hs = torch.cat([hidden_states, hidden_states], dim=0)  # [2M, H]
        else:
            hs = torch.cat([hidden_states, residual], dim=0)  # [2M, H]
        return SRStackState(hs=hs)

    def run_layer(self, layer, positions, state):
        return SRStackState(hs=layer(positions, state.hs))

    def to_intermediate(self, state):
        # Split the [2M, H] stack into two token-leading [M, H] halves. The base
        # and adapter halves diverge after layer 0 (the adapter half accumulates
        # deltas and the shunt), so BOTH must cross; residual carries the
        # adapter half.
        m = state.hs.shape[0] // 2
        return state.hs[:m].contiguous(), state.hs[m:].contiguous()

    def exit_decoder_stack(self, state, adapter_indices, norm, config):
        hs = state.hs
        m = adapter_indices.shape[0]
        # Per-token merge: adapter-active tokens read the adapter stream, the
        # rest read the (base-equivalent) base stream. SR uses a plain norm, not
        # rms_norm_select, because its residual is materialized inside each
        # layer.
        select = (adapter_indices > 0).unsqueeze(1)  # [M, 1]
        merged = torch.where(select, hs[m:], hs[:m])
        return norm(merged)


def is_shadow_residual(config) -> bool:
    """Shadow Residual checkpoints set ``cross_stream_rank``; LoRA leaves it None."""
    return config.cross_stream_rank is not None


def select_decoder_interface(config) -> DecoderInterface:
    """Pick the decoder interface for ``config``.

    Keyed on ``config.cross_stream_rank`` rather than ``config.architectures``:
    the SR interface needs the ``cross_stream_rank`` value to build its
    ``WCrossShunt`` anyway, so the discriminant and the data are the same field.
    """
    return (
        SRDecoderInterface() if is_shadow_residual(config) else LoRADecoderInterface()
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _get_intermediate_tensor(
    tensors: IntermediateTensors,
    name: str,
) -> torch.Tensor | None:
    try:
        return tensors[name]
    except KeyError:
        return None


@support_torch_compile
class GraniteSwitchModel(nn.Module):
    """Granite transformer with per-token adapter switching.

    The switch detects control tokens, selects the appropriate adapter, and
    rewrites each control token's id to its substitute id (token exchange). The
    decoder embeds the rewritten ids and is otherwise oblivious to the
    substitution. Adapter indices reach the LoRA layers through a single shared
    context object.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        if not isinstance(config, GraniteSwitchConfig):
            raise TypeError(
                f"Expected GraniteSwitchConfig, got {type(config).__name__}"
            )

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Adaptation strategy: confines the LoRA-vs-Shadow-Residual split to the
        # decoder tier. Keyed on config.cross_stream_rank (None -> LoRA, an int
        # -> SR). The shared __init__ / forward call its hooks, so one code path
        # drives both.
        self.decoder_interface = select_decoder_interface(config)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        num_adapters = config.num_adapters
        if num_adapters > 0:
            self.switch: MultiSwitch | None = MultiSwitch(config, vllm_config)

            # Control token ids, one per adapter: the switch matches these
            # against the input sequence to decide which adapter to activate,
            # and the position in the tensor is the adapter index. All values
            # come from config. Stored as a buffer rather than a Parameter so
            # they do not pollute the state_dict and stay torch.compile-friendly
            # (no .item() needed), while still following .to(device).
            token_ids = config.adapter_token_ids
            if token_ids is not None:
                self.register_buffer(
                    "adapter_token_ids",
                    torch.tensor(token_ids, dtype=torch.long),
                )
            else:
                self.register_buffer(
                    "adapter_token_ids",
                    torch.zeros(num_adapters, dtype=torch.long),
                )

            # Fused kernel metadata (bitmask-based). The adaptation builds its
            # own (kernel-meta, ctx) pair: LoRA -> (FusedLoRAKernelMeta,
            # LoRAContext); SR -> (SRFusedLoRAKernelMeta, SRLoRAContext).
            # SRFusedLoRAKernelMeta / SRLoRAContext subclass the LoRA pair, so
            # one annotation covers both tiers.
            lora_meta, lora_ctx = self.decoder_interface.make_kernel_meta(
                vllm_config.device_config.device,
            )
            self.lora_meta: FusedLoRAKernelMeta | None = lora_meta
            self.lora_ctx: LoRAContext | None = lora_ctx
        else:
            self.switch = None
            self.adapter_token_ids = None
            self.lora_meta = None
            self.lora_ctx = None

        # With adapters, config.num_hidden_layers counts the switch's KV-cache
        # slots (2 for MultiSwitch) so a Transformers DynamicCache sizes
        # correctly. vLLM auto-discovers Attention layers, so subtract them to
        # get the decoder-layer count. The verifier enforces >= 1 remains.
        if self.switch is not None:
            num_decoder_layers = config.num_hidden_layers - self.switch.num_cache_layers
        else:
            num_decoder_layers = config.num_hidden_layers

        def _make_decoder_layer(prefix: str):
            return self.decoder_interface.make_decoder_layer(vllm_config, prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_decoder_layers,
            _make_decoder_layer,
            prefix=f"{prefix}.layers",
        )

        # Wire the shared LoRA context onto every module that reads per-forward
        # metadata: a single object populated once per forward and read by every
        # layer that needs it. object.__setattr__ bypasses
        # nn.Module.__setattr__ so the context is NOT registered as a
        # submodule/buffer - it carries live per-forward tensors, not parameters
        # - and the attribute stays stable for torch.compile.
        if num_adapters > 0:
            ctx_types = self.decoder_interface.ctx_wire_types()
            for module in self.modules():
                if isinstance(module, ctx_types):
                    object.__setattr__(module, "_lora_ctx", self.lora_ctx)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Allocate pipeline-parallel profiling buffers for token-leading tensors.

        vLLM slices every ``IntermediateTensors`` entry by token count, so only
        token-leading metadata belongs here. The fixed-size LoRA kernel metadata
        is recomputed on each PP rank from ``adapter_indices``.
        """
        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "residual": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "adapter_indices": torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=device,
            ),
        }
        return IntermediateTensors(tensors)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the switch and the decoder stack.

        The class is decorated with ``@support_torch_compile``, so the switch
        runs inside the compiled region.

        Args:
            input_ids: ``[num_tokens]`` token ids.
            positions: ``[num_tokens]`` per-request token positions for RoPE and
                for the switch's counting anchor.
            intermediate_tensors: Pipeline-parallel handoff from the previous
                rank.
            inputs_embeds: Pre-computed embeddings, first rank only.

        Returns:
            The final ``[num_tokens, hidden_size]`` hidden states on the last
            rank, otherwise the ``IntermediateTensors`` to hand on.

        """
        # Step 1: the switch. Determine the adapter for each token and rewrite
        # the control tokens via token exchange. First rank only.
        if get_pp_group().is_first_rank:
            # input_ids can be None when the request supplies prompt embeddings
            # directly, and the switch reads ids - hence the guard.
            if self.switch is not None and input_ids is not None:
                # positions must be forwarded, not fabricated: the coded engine
                # anchors on positions == 0, and vLLM flattens the batch, so a
                # local arange would anchor only the first request and misroute
                # every other one to an arbitrary adapter.
                adapter_indices, modified_input_ids = self.switch(
                    input_ids=input_ids,
                    adapter_token_ids=self.adapter_token_ids,
                    positions=positions,
                )
            else:
                # No switch, or no ids to read: run on the base model, adapter
                # id 0 everywhere. Sizing off inputs_embeds in the latter case is
                # what makes this safe; reading input_ids.device unconditionally
                # would raise instead.
                if input_ids is not None:
                    num_tokens = input_ids.shape[0]
                    device = input_ids.device
                elif inputs_embeds is not None:
                    num_tokens = inputs_embeds.shape[0]
                    device = inputs_embeds.device
                else:
                    raise ValueError(
                        "forward() needs input_ids or inputs_embeds on the "
                        "first pipeline rank; both were None."
                    )
                adapter_indices = torch.zeros(
                    num_tokens,
                    dtype=torch.long,
                    device=device,
                )
                modified_input_ids = input_ids

            # Prepare the kernel metadata ONCE for every decoder layer.
            if self.lora_meta is not None and self.lora_ctx is not None:
                self.decoder_interface.prepare_kernel_meta(
                    self.lora_meta, adapter_indices, self.lora_ctx
                )

            # Carry the indices across the pipeline-parallel boundary.
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            intermediate_tensors["adapter_indices"] = adapter_indices
        else:
            # Later ranks: recompute the fixed-size LoRA metadata from the
            # token-leading adapter_indices received over the PP wire.
            if intermediate_tensors is not None:
                adapter_indices = intermediate_tensors["adapter_indices"]
                if self.lora_ctx is not None:
                    self.decoder_interface.prepare_kernel_meta(
                        self.lora_meta, adapter_indices, self.lora_ctx
                    )
            else:
                # No metadata available; should not happen in normal operation.
                num_tokens = input_ids.shape[0] if input_ids is not None else 0
                if input_ids is not None:
                    fallback_device = input_ids.device
                elif self.lora_meta is not None:
                    fallback_device = self.lora_meta.device
                else:
                    fallback_device = self.embed_tokens.weight.device
                adapter_indices = torch.zeros(
                    num_tokens,
                    dtype=torch.long,
                    device=fallback_device,
                )

        # Step 2: embeddings, or the hidden states from the previous PP stage.
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # Embed the (possibly rewritten) ids the switch returned. The
                # token-exchange rewrite already happened, so this single lookup
                # produces the correct embeddings for both control positions
                # (the substitute id) and content positions.
                hidden_states = self.embed_input_ids(modified_input_ids)

            hidden_states *= self.config.embedding_multiplier
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = _get_intermediate_tensor(intermediate_tensors, "residual")

        # Step 3: the decoder stack, through the adaptation's hooks. The
        # adaptation owns the stack shape: LoRA runs single-stream [M, H] and
        # threads (hidden_states, residual); SR re-stacks to [2M, H]
        # rank-locally in enter_decoder_stack and collapses it in
        # exit_decoder_stack. All per-forward metadata sits on the shared ctx.
        state = self.decoder_interface.enter_decoder_stack(hidden_states, residual)
        for i in range(self.start_layer, self.end_layer):
            state = self.decoder_interface.run_layer(self.layers[i], positions, state)

        if get_pp_group().is_last_rank:
            # Adaptation-specific finalize: LoRA folds the last residual through
            # rms_norm_select (honoring the fused/separate convention); SR does
            # the per-token where-merge of its two streams, then a plain norm.
            return self.decoder_interface.exit_decoder_stack(
                state, adapter_indices, self.norm, self.config
            )

        # Non-last rank: ship two token-leading [M, H] tensors so vLLM's
        # per-token IntermediateTensors slice stays correct. LoRA sends
        # (hidden_states, residual); SR overloads residual as its adapter half.
        if intermediate_tensors is None:
            intermediate_tensors = IntermediateTensors({})
        h_out, r_out = self.decoder_interface.to_intermediate(state)
        intermediate_tensors["hidden_states"] = h_out
        intermediate_tensors["residual"] = r_out
        return intermediate_tensors


class GraniteSwitchForCausalLM(nn.Module, SupportsPP):
    """Granite Switch for causal language modeling.

    ``IsHybrid`` and ``HasInnerState`` are deliberately NOT declared, even
    though the config's parent is ``GraniteMoeHybridConfig``. The architecture
    is attention-only with no Mamba state, so neither interface would have
    anything to contribute - and declaring ``IsHybrid`` would make vLLM size the
    KV cache for ZERO attention layers, because
    ``GraniteMoeHybridConfig.attribute_map`` aliases ``layers_block_type`` onto
    ``layer_types``, and ``get_num_layers_by_block_type`` counts entries equal to
    the literal ``"attention"`` - which a Transformers-normalized config spells
    ``"full_attention"``.

    That is a silent-wrong-output failure gated on a detail of the installed
    Transformers version, so it is not left to coincidence:
    ``GraniteSwitchConfigVerifier`` (registered in ``MODELS_CONFIG_MAP``)
    recounts the attention layers the way vLLM will and raises at startup if the
    count does not match ``num_hidden_layers``.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = GraniteSwitchModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size,
            1.0 / config.logits_scaling,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids, applying the switch's token-exchange rewrite first.

        Control ids are rewritten to their substitute ids before the lookup, so
        a control token gets its in-distribution embedding exactly as on the
        decoder's own path. Adapter *detection* still runs in
        ``GraniteSwitchModel.forward`` against the raw ids.

        Returns UN-scaled embeddings; the Granite ``embedding_multiplier`` is
        applied once in the model forward, over everything.
        """
        switch = self.model.switch
        if switch is None:
            return self.model.embed_tokens(input_ids)
        ids = apply_token_exchange(switch.control_to_substitute_lut, input_ids)
        return self.model.embed_tokens(ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states.

        No control-token logit suppression is applied. Control tokens are
        intended to be freely generatable. Even if an interim suppression were
        wanted it could not live here: ``compute_logits`` receives
        sample-extracted hidden states, which are no longer aligned with the
        per-token ``adapter_indices`` computed in ``forward``. Any suppression
        would have to act where those two still share a token dimension.
        """
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, delegating to the adaptation's loader.

        See ``LoRADecoderInterface.load_weights`` and
        ``SRDecoderInterface.load_weights``.
        """
        return self.model.decoder_interface.load_weights(self, weights)

    def process_weights_after_loading(self) -> None:
        """Build w_ext and register the per-module remap tables.

        vLLM's model-level post-load hook rather than the tail of
        ``load_weights``, so the fusion also runs under ``load_format="dummy"``,
        whose loader substitutes its own ``load_weights``. Otherwise a
        dummy-weight run would leave every projection unfinalized and trip the
        ``_finalized`` assert on the first forward.
        """
        self.model.decoder_interface.finalize_modules(self, self.config)
