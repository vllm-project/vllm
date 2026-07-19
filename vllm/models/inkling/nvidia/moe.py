# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling mixture-of-experts on vLLM's FusedMoE abstraction.

Overfit to the served checkpoint: sigmoid gate (+ selection bias) top-k over
the routed experts, log-sigmoid renormalization over the k routed + S shared
"sink" logits, scaled by route_scale * global_scale. The routed top-k goes
through vLLM's FusedMoE (which handles TP/EP); the sink experts run in
:class:`InklingSinkExperts` -- replicated across EP ranks (every token
activates every sink) and always bf16 (the checkpoint excludes every
``shared_experts`` from quantization).

NVFP4 routed experts reuse vLLM's ModelOpt NVFP4 fused-MoE method; excluded
(bf16) layers fall back to the unquantized method. The checkpoint's fused
stacked tensors (interleaved gate/up rows, ``.scale`` / ``.scale2`` /
``.input_amax`` aux tensors) are translated to the standard per-expert loads
in :meth:`InklingMoE.load_expert_weight`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

from ..configs import InklingModelConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm.model_executor.layers.quantization import QuantizationConfig

# ---------------------------------------------------------------------------
# Gate / expert selection
# ---------------------------------------------------------------------------

_INKLING_LL_BF16_MAX_TOKENS = 64
_NVFP4_INPUT_SCALE_DENOMINATOR = torch.finfo(torch.float8_e4m3fn).max * 6.0


def _linear_with_fp32_out(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    leading = list(x.shape[:-1])
    flat = x.flatten(0, -2)
    if (
        flat.shape[0] <= _INKLING_LL_BF16_MAX_TOKENS
        and flat.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and flat.is_cuda
        and flat.is_contiguous()
        and weight.is_contiguous()
        and flat.shape[1] % 8 == 0
        and current_platform.has_device_capability(90)
        and ll_bf16.is_available()
    ):
        out = ll_bf16.ll_bf16_gemm(flat, weight)
    else:
        out = torch.mm(flat, weight.T, out_dtype=torch.float32)
    return out.view(*leading, weight.shape[0])


@triton.jit(do_not_specialize=["T", "route_scale"])
def _inkling_gate_select_kernel(
    logits_ptr,  # [T, G] fp32 gate logits (stride_logits_0 may include pad)
    bias_ptr,  # [R] fp32 selection bias (or 0 ptr if HAS_BIAS=False)
    global_scale_ptr,  # [1] fp32 (or unused if HAS_GSCALE=False)
    ids_ptr,  # [T, K + S] int32 out: selected expert ids
    weights_ptr,  # [T, K + S] fp32 out: renormalized weights
    route_scale,
    T,
    G: tl.constexpr,  # total gate experts (routed + shared)
    stride_logits_0,
    R: tl.constexpr,  # routed experts
    K: tl.constexpr,  # top-k routed
    S: tl.constexpr,  # shared (sink) experts
    HAS_BIAS: tl.constexpr,
    HAS_GSCALE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return
    offs = tl.arange(0, BLOCK_G)
    mask_r = offs < R
    logits = tl.load(
        logits_ptr + pid * stride_logits_0 + offs,
        mask=offs < G,
        other=float("-inf"),
    ).to(tl.float32)

    # Selection scores: sigmoid(routed logits) (+ bias), non-routed lanes -inf.
    sel = tl.where(mask_r, tl.sigmoid(logits), float("-inf"))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=mask_r, other=0.0).to(tl.float32)
        sel = tl.where(mask_r, sel + bias, float("-inf"))

    scale = route_scale
    if HAS_GSCALE:
        scale = scale * tl.load(global_scale_ptr).to(tl.float32)

    # Iterative top-K (K is small); argmax tie-breaks to the lowest index
    # (stable ordering).
    A: tl.constexpr = K + S
    offs_a = tl.arange(0, A)
    top_ids = tl.zeros([A], dtype=tl.int32)
    active = tl.zeros([A], dtype=tl.float32)
    for kk in tl.static_range(K):
        idx = tl.argmax(sel, axis=0).to(tl.int32)
        raw = tl.max(tl.where(offs == idx, logits, float("-inf")), axis=0)
        top_ids = tl.where(offs_a == kk, idx, top_ids)
        active = tl.where(offs_a == kk, raw, active)
        sel = tl.where(offs == idx, float("-inf"), sel)
    if S > 0:
        # Shared sink logits sit at the tail of the gate output; their expert
        # ids continue after the routed range (R + j).
        for jj in tl.static_range(S):
            raw = tl.max(tl.where(offs == R + jj, logits, float("-inf")), axis=0)
            top_ids = tl.where(offs_a == K + jj, tl.full([], R + jj, tl.int32), top_ids)
            active = tl.where(offs_a == K + jj, raw, active)

    # Log-sigmoid renormalization over the K + S active logits.
    abs_l = tl.abs(active)
    min_l = tl.minimum(active, 0.0)
    log_probs = min_l - tldevice.log1p(tldevice.exp(-abs_l))
    max_lp = tl.max(log_probs, axis=0)
    exp_shifted = tldevice.exp(log_probs - max_lp)
    sum_exp = tl.sum(exp_shifted, axis=0)
    weights = exp_shifted / sum_exp * scale

    tl.store(ids_ptr + pid * A + offs_a, top_ids)
    tl.store(weights_ptr + pid * A + offs_a, weights)


def inkling_gate_select(
    logits: torch.Tensor,  # [T, >=G] fp32 (rows may carry GEMM padding)
    n_gate_experts: int,
    n_routed_experts: int,
    topk: int,
    n_shared_experts: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid + bias + top-k + log-sigmoid renorm; returns (weights, ids)."""
    assert logits.dtype == torch.float32
    tokens = logits.shape[0]
    active = topk + n_shared_experts
    topk_ids = torch.empty((tokens, active), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty(
        (tokens, active), dtype=torch.float32, device=logits.device
    )
    if tokens == 0:
        return topk_weights, topk_ids
    _inkling_gate_select_kernel[(tokens,)](
        logits,
        bias if bias is not None else logits,
        global_scale if global_scale is not None else logits,
        topk_ids,
        topk_weights,
        route_scale,
        tokens,
        n_gate_experts,
        logits.stride(0),
        n_routed_experts,
        topk,
        n_shared_experts,
        HAS_BIAS=bias is not None,
        HAS_GSCALE=global_scale is not None,
        BLOCK_G=triton.next_power_of_2(n_gate_experts),
    )
    return topk_weights, topk_ids


class InklingGate(nn.Module):
    """Sigmoid gate with selection bias, log-sigmoid renorm after top-k, and
    global scale (the served checkpoint's only configuration)."""

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_shared_experts: int,
        experts_per_token: int,
        route_scale: float,
        *,
        use_global_scale: bool = False,
        use_gate_bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_total_experts = n_routed_experts + n_shared_experts
        self.topk = experts_per_token
        self.route_scale = route_scale

        padded_experts = self.n_total_experts + (-self.n_total_experts) % 8
        self.weight = Parameter(
            torch.empty(padded_experts, d_model), requires_grad=False
        )
        set_weight_attrs(self.weight, {"weight_loader": self._load_weight})
        if use_global_scale:
            self.global_scale = Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
        else:
            self.global_scale = None
        if use_gate_bias:
            self.bias = Parameter(
                torch.empty(n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.bias = None

    @staticmethod
    def _load_weight(param: Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.zero_()
        param.data[: loaded_weight.shape[0]].copy_(loaded_weight)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 gate logits [T, n_total_experts + pad] (pad columns are junk)."""
        return _linear_with_fp32_out(x, self.weight)

    def select_experts(
        self, gating_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full selection: (weights, ids) of [T, K + S]. The first K entries
        are the routed top-k; the S trailing entries are the sink gammas."""
        return inkling_gate_select(
            gating_output,
            self.n_total_experts,
            self.n_routed_experts,
            self.topk,
            self.n_shared_experts,
            self.bias,
            self.route_scale,
            self.global_scale,
        )


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------


def _inkling_moe_ep_size() -> int:
    """EP size the FusedMoE layer will run with (mirrors
    FusedMoEParallelConfig.make: experts shard over tp * dp * pcp when
    expert parallelism is enabled)."""
    parallel_config = get_current_vllm_config().parallel_config
    if not parallel_config.enable_expert_parallel:
        return 1
    world = (
        get_tensor_model_parallel_world_size()
        * get_dp_group().world_size
        * get_pcp_group().world_size
    )
    return world if world > 1 else 1


class InklingSinkExperts(nn.Module):
    """Shared "sink" experts with per-token gammas, in bf16.

    Replicated across EP ranks (every token activates every sink, so
    EP-sharding them would hotspot the owning rank) and TP-sharded on the
    intermediate dim so the output remains a TP-partial sum like the routed
    output. The sinks are always bf16 (the checkpoint excludes every
    ``shared_experts`` from quantization): the experts concatenate into two
    plain dense GEMMs with the fused sink epilogue between them.
    """

    def __init__(
        self, n_experts: int, d_model: int, d_mlp: int, *, prefix: str = ""
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        intermediate_pp = d_mlp // tp_size
        self.w13_weight = Parameter(
            torch.empty(n_experts, 2 * intermediate_pp, d_model),
            requires_grad=False,
        )
        self.w2_weight = Parameter(
            torch.empty(d_model, n_experts * intermediate_pp),
            requires_grad=False,
        )
        self._unit: torch.Tensor | None = None

    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:
        """Load one checkpoint sink tensor (stacked over the S experts)."""
        if key == "w13_weight":
            if weight.shape != self.w13_weight.shape:
                shard = self.w13_weight.shape[1]
                weight = weight.narrow(1, self.tp_rank * shard, shard)
            self.w13_weight.data.copy_(weight)
            return [key]

        assert key == "w2_weight"
        shard = self.w2_weight.shape[1] // self.n_experts
        shard_start = 0 if weight.shape[2] == shard else self.tp_rank * shard
        for expert_idx, expert_weight in enumerate(weight):
            local_weight = expert_weight.narrow(1, shard_start, shard)
            start = expert_idx * shard
            self.w2_weight.data[:, start : start + shard].copy_(local_weight)
        return [key]

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """``sum_e gammas[:, e] * MLP_e(x)`` (TP-partial along d_mlp)."""
        from .ops import sink_silu_mul_epilogue

        # One GEMM over the experts' stacked w13 (a view), fused epilogue,
        # then one GEMM whose K-reduction over the K-concatenated w2 performs
        # the expert sum.
        if self._unit is None or self._unit.device != x.device:
            self._unit = torch.ones(
                self.n_experts, dtype=torch.float32, device=x.device
            )
        raw = x @ self.w13_weight.view(-1, x.shape[-1]).T  # (T, S*2F)
        h = sink_silu_mul_epilogue(
            raw, self._unit, gammas, self._unit, self.n_experts, x.dtype
        )
        return h @ self.w2_weight.T  # (T, D)


class InklingSinkExpertsLinear(nn.Module):
    """LoRA-capable implementation of the Inkling sink experts."""

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_mlp: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        self.n_experts = n_experts
        self.d_mlp = d_mlp
        total = n_experts * d_mlp
        self.w13 = MergedColumnParallelLinear(
            input_size=d_model,
            output_sizes=[total, total],
            bias=False,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=total,
            output_size=d_model,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.w2",
        )
        self._w2_input_pp = self.w2.input_size_per_partition
        self._col_expert: torch.Tensor | None = None

    def _gamma_expand(self, gammas: torch.Tensor) -> torch.Tensor:
        if self._col_expert is None or self._col_expert.device != gammas.device:
            local = self._w2_input_pp
            start = get_tensor_model_parallel_rank() * local
            cols = torch.arange(start, start + local, device=gammas.device)
            self._col_expert = (cols // self.d_mlp).long()
        return gammas[:, self._col_expert]

    def load_weight(self, key: str, weight: torch.Tensor) -> list[str]:
        if key == "w13_weight":
            d_model = weight.shape[-1]
            gate = weight[:, 0::2, :].reshape(-1, d_model).contiguous()
            up = weight[:, 1::2, :].reshape(-1, d_model).contiguous()
            self.w13.weight_loader(self.w13.weight, gate, 0)
            self.w13.weight_loader(self.w13.weight, up, 1)
            return ["w13.weight"]
        w = weight.permute(1, 0, 2).reshape(weight.shape[1], -1).contiguous()
        self.w2.weight_loader(self.w2.weight, w)
        return ["w2.weight"]

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        hidden_states = (hidden_states * self._gamma_expand(gammas)).to(x.dtype)
        output, _ = self.w2(hidden_states)
        return output


class InklingMoE(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        *,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Overfit to the served checkpoint: sigmoid gate renormalized after
        # top-k, shared sink experts, interleaved gate/up checkpoint rows.
        assert config.gate_activation == "sigmoid" and config.norm_after_topk
        assert config.n_shared_experts > 0 and config.shared_expert_sink
        assert config.inference_moe_w13_interleaved
        n_routed = config.n_routed_experts
        n_shared = config.n_shared_experts
        self.n_routed_experts = n_routed
        self.gate = InklingGate(
            d_model=config.hidden_size,
            n_routed_experts=n_routed,
            n_shared_experts=n_shared,
            experts_per_token=config.num_experts_per_tok,
            route_scale=config.route_scale,
            use_global_scale=config.use_global_scale,
            use_gate_bias=config.use_gate_bias,
        )

        # TRTLLM MoE kernels assume equal, contiguous per-rank expert slabs
        # (local_expert_offset = ep_rank * local_num_experts), so pad the
        # expert count to a multiple of the EP size. A no-op for the usual
        # power-of-two EP sizes (n_routed is a power of two).
        num_experts = n_routed + (-n_routed) % _inkling_moe_ep_size()

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self._select_routed,
            router_logits_dtype=torch.float32,
            activation="silu",
        )
        # The decoder layer reduce-scatters the MoE delta into the sconv
        # stream itself (RS -> shard sconv -> AG); the runner must return the
        # per-rank partial sum instead of all-reducing.
        self.experts.moe_config.skip_final_all_reduce = True

        self._routed_sel: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        sink_experts_cls = (
            InklingSinkExpertsLinear
            if get_current_vllm_config().lora_config is not None
            else InklingSinkExperts
        )
        self.sink_experts = sink_experts_cls(
            n_experts=n_shared,
            d_model=config.hidden_size,
            d_mlp=config.intermediate_size,
            prefix=f"{prefix}.shared_experts",
        )

        # Sink chain overlaps the routed MoE call on the aux stream for
        # decode-sized batches (same pattern as the runner's SharedExperts
        # multi-stream overlap). The routed GEMM runs on the default stream and
        # the sink chain on the aux stream, joined via these two events by
        # ``maybe_execute_in_parallel``.
        self._sink_stream: torch.cuda.Stream | None = aux_stream()
        self._sink_events = (torch.cuda.Event(), torch.cuda.Event())

    def _select_routed(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FusedMoE ``custom_routing_function``: the routed top-k slice of the
        full (routed + sink) selection.

        forward() stashes its selection (keyed by logits identity) so the
        gate select runs once per layer; the fallback covers paths where the
        runner re-derives the logits (e.g. naive DP dispatch).
        """
        del hidden_states, renormalize
        assert topk == self.gate.topk
        cached = self._routed_sel
        self._routed_sel = None
        if cached is not None and cached[0] is gating_output:
            return cached[1], cached[2]
        weights, ids = self.gate.select_experts(gating_output)
        return weights[:, :topk].contiguous(), ids[:, :topk].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        router_logits = self.gate.compute_logits(x)
        num_tokens = x.shape[0]
        # One gate select per layer: the routed slice is stashed for the
        # routing function inside the FusedMoE op; the sink gammas are the
        # trailing columns.
        k = self.gate.topk
        weights, ids = self.gate.select_experts(router_logits)
        self._routed_sel = (
            router_logits,
            weights[:, :k].contiguous(),
            ids[:, :k].contiguous(),
        )
        gammas = weights[:, k:]

        out, sink_out = maybe_execute_in_parallel(
            lambda: self.experts(hidden_states=x, router_logits=router_logits),
            lambda: self.sink_experts(x, gammas),
            self._sink_events[0],
            self._sink_events[1],
            self._sink_stream
            if num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            else None,
        )
        self._routed_sel = None

        return out + sink_out

    # -- weight loading ----------------------------------------------------

    def _local_expert_slots(self) -> dict[int, int]:
        """Global expert id -> local slot for this rank's expert partition."""
        manager = self.experts.routed_experts.expert_map_manager
        if manager.expert_map is None:
            return {g: g for g in range(manager.global_num_experts)}
        emap = manager.expert_map.tolist()
        return {g: slot for g, slot in enumerate(emap) if slot >= 0}

    def load_expert_weight(self, name: str, weight: torch.Tensor) -> list[str]:
        """Load one checkpoint expert tensor.

        ``name`` is relative to the mlp module: ``experts.<t>`` (routed
        stack) or ``shared_experts.shared_<t>`` (sink experts). Returns the
        loaded param names (relative to this module).
        """
        if name.startswith("shared_experts."):
            key = name.split(".", 1)[1].replace("shared_", "", 1)
            return [
                f"sink_experts.{p}" for p in self.sink_experts.load_weight(key, weight)
            ]

        experts: RoutedExperts = self.experts.routed_experts
        key = name.split(".", 1)[1]

        # original_shape is unused by the vLLM serving layout.
        if key.endswith(".original_shape"):
            return []
        if key.endswith(".input_amax"):
            projection = "w13" if key.startswith("w13") else "w2"
            amax = float(weight.max())
            assert math.isfinite(amax) and amax > 0, (
                f"bad {projection} input_amax: {amax}"
            )
            input_scale = getattr(experts, f"{projection}_input_scale")
            input_scale.data.fill_(amax / _NVFP4_INPUT_SCALE_DENOMINATOR)
            return [f"experts.routed_experts.{projection}_input_scale"]

        param = getattr(experts, key)
        slots = self._local_expert_slots()
        gids = sorted(slots)
        lids = [slots[g] for g in gids]
        tp_rank = experts.moe_config.moe_parallel_config.tp_rank

        if key.endswith("_scale_2"):
            # Per-expert scalars, vectorized over the local experts. The
            # fused w13 param carries one slot per gate/up half.
            vals = weight[gids].float().to(param.device)
            param.data[lids] = vals[:, None] if param.data.ndim == 2 else vals
        elif key.startswith("w13"):
            # Checkpoint w13 rows are interleaved [g0, u0, g1, u1, ...]; the
            # fused param layout is [w1(gate); w3(up)]. The TP-local rows form
            # one contiguous slab of the interleaved tensor, so upload just
            # that slab (a single bounded synchronous H2D; pre-uploading whole
            # untrimmed tensors pins the mmap pages of the entire checkpoint
            # and OOMs the host) and de-interleave on device.
            half = param.shape[1] // 2
            for gid, lid in slots.items():
                slab = weight[gid].narrow(0, tp_rank * 2 * half, 2 * half)
                slab = slab.to(param.device)
                param.data[lid, :half].copy_(slab[0::2])
                param.data[lid, half:].copy_(slab[1::2])
        else:
            # w2: shard the packed intermediate (last) dim.
            shard = param.shape[2]
            for gid, lid in slots.items():
                param.data[lid].copy_(weight[gid].narrow(1, tp_rank * shard, shard))
        return [f"experts.routed_experts.{key}"]

    def finalize_load(self) -> list[str]:
        """Post-load fixups for zeroed padding experts."""
        experts = self.experts.routed_experts
        out: list[str] = []
        # Zero the EP-alignment padding experts (if any) so their
        # (never-routed) slots hold defined values.
        slots = self._local_expert_slots()
        for gid in range(self.n_routed_experts, experts.global_num_experts):
            lid = slots.get(gid)
            if lid is None:
                continue
            for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ):
                p = getattr(experts, pname, None)
                if p is not None:
                    p.data[lid].zero_()
        return out
