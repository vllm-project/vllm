# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed experts of the specialized decode's TP 1 steps of up to 32 rows.

Problem. At TP 1 a Qwen3.8-Flash-Next MoE layer of a small decode step runs,
between its router GEMM and its expert GEMMs, vLLM's NVFP4 quantization of
the layer input and FlashInfer's TRT-LLM routing (one or two launches: the
softmax top-10 and the routing metadata the GEMMs read). `RoutedPath` runs
both as one CuTe DSL kernel (`RoutedFront`, `..moe.front`) whose outputs
equal theirs bit for bit, then the same TRT-LLM-gen expert GEMMs on that
record through FlashInfer's prepared-routing body entry point (the
`da_routing_metadata` arguments of its FP4 MoE operation). Three or four
launches with their launch gaps become one; at 32 rows (eight requests of
four tokens) the routed chain is the longer branch of the MoE block, so the
step is shorter by what the routed chain saves.

What stays stock. The router GEMM (the runner's gate), the shared expert on
the runner's side stream, the expert GEMMs, their tactic and the finalize
(FlashInfer's kernel, or the boundary kernel's folded finalize in steps of
`FOLD_MIN_ROWS` rows, `boundary.py`). The tactic is the one FlashInfer's
autotuner picks for the stock call of the same shape (`stock_tactic`: the
stock operation's own runner and tuning configuration, looked up in the same
tuning cache; the folded call hands FlashInfer an `output` of the finalized
shape for the same reason), so the GEMMs are the same kernels with the same
configuration, and the routed output, finalized or folded, is bitwise the
stock output.

Which calls. `RoutedPath.build` serves a MoE layer whose runner holds
FlashInfer's monolithic TRT-LLM NVFP4 experts with the routing and shapes the
front implements (`unsupported`); `RoutedPath.forward` then takes calls of
up to `MAX_ROWS` rows whose stock GEMMs use the front's 8-row token tile
(a tuned tactic of that tile, or FlashInfer's fallback tactic, whose tile is
8 for these row counts), and `RoutedPath.tactic` returns None for every
other call, which runs the stock runner. The executor builds it at TP 1
only (TP 4 keeps the stock MoE front).

Buffers. Each served layer owns a routing record (`RoutingRecord`: the
`RoutingMetadata` and the quantized input, about 60 KB). The expert GEMMs'
own workspaces come from FlashInfer's body preparation at each call, as the
stock operation allocates them inside its call (from the CUDA graph's pool
during capture).

Ordering (the kernels of one served layer, in stream order; "PDL" marks a
launch with programmatic stream serialization, which lets the kernel start
once its predecessor triggers or exits and orders nothing before the kernel's
own grid-dependency wait):

1. The router GEMM (the runner's gate) writes the logits; the shared expert
   runs on the runner's side stream, which joins before the boundary kernel
   that reads its output, as in the stock runner.
2. The front (PDL) triggers its dependents only at exit. Every CTA executes
   the wait before its first global load or store, so it reads complete
   logits and input (the router GEMM completed) and writes the record only
   after every earlier kernel that the router GEMM followed.
3. GEMM1, GEMM2 and the finalize (PDL, FlashInfer's body) read the record.
   FlashInfer's contract for its own chain is the one the front relies on:
   the batched GEMMs wait before every routing-dependent load (its routing
   kernels trigger right after their own wait, before writing the routing),
   and the activations are gathered through the routing, so GEMM1 reads
   nothing the front writes before its wait; the front completes only after
   its own wait, so GEMM1's wait covers the router GEMM too.
4. With the folded finalize, the boundary kernel reads the record's weights
   and expert slots after its own wait, which returns after GEMM2, hence
   after the front (`boundary.py`, point 1 of the folded finalize).
5. The next writer of a record is the same layer's front in a later replay
   (or eager call): CUDA graph launches on one stream run after the previous
   launch has completed, and no other kernel writes a record.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    from ...model import Qwen4ExpSparseMoeBlock
    from ..moe.front import RoutingMetadata

logger = init_logger(__name__)

# FlashInfer's RoutingMethodType.RenormalizeNaive: softmax, top-k, renormalize.
RENORMALIZE_NAIVE = 4
# FlashInfer's RoutingInputMode.PackedPrecomputed, the body's routing input.
PACKED_PRECOMPUTED = 1
# Rows of the largest call the routed path serves (``..moe.front.MAX_ROWS``).
MAX_ROWS = 32
# FlashInfer's fallback tactic, which the launcher resolves to its default
# token tile (``fallback_tile``) and that tile's default configuration.
FALLBACK = [-1, -1]
# FlashInfer's token tiles for NVFP4 activations and weights, ascending.
FP4_TILES = (8, 16, 32, 64, 128, 192)


class RoutingRecord(NamedTuple):
    """One layer's front outputs, read by its expert body."""

    metadata: "RoutingMetadata"
    # E2M1 pairs, uint8 [MAX_ROWS, hidden / 2].
    x_fp4: torch.Tensor
    # E4M3 scales, uint8 [MAX_ROWS, hidden / 16].
    x_scale: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device) -> "RoutingRecord":
        """A record for calls of up to `MAX_ROWS` rows."""
        from ..moe.front import BLOCKS, HIDDEN, RoutingMetadata

        return cls(
            RoutingMetadata(MAX_ROWS, device),
            torch.zeros(MAX_ROWS, HIDDEN // 2, dtype=torch.uint8, device=device),
            torch.zeros(MAX_ROWS, BLOCKS, dtype=torch.uint8, device=device),
        )

    def front(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        input_scale: torch.Tensor,
        early_trigger: bool = False,
        stall_us: int = 0,
    ) -> None:
        """Fill the record from router `logits` and the input `x`
        (`routed_front`)."""
        from ..moe.front import routed_front

        routed_front(
            logits,
            x,
            input_scale,
            self.metadata,
            self.x_fp4,
            self.x_scale,
            early_trigger=early_trigger,
            stall_us=stall_us,
        )

    def body(
        self, params: "ExpertParams", rows: int, tactic: list[int], finalize: bool
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """The expert GEMMs of the record's `rows` rows, then the finalize
        or, without `finalize`, the unfinalized output (`run_body`)."""
        return run_body(
            params,
            self.metadata,
            rows,
            self.x_fp4[:rows],
            self.x_scale[:rows].view(torch.float8_e4m3fn),
            tactic,
            finalize,
        )


class ExpertParams(NamedTuple):
    """One layer's routed experts as FlashInfer's FP4 MoE operation takes them
    (the arguments vLLM's `TrtLlmNvFp4ExpertsMonolithic.apply` passes)."""

    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm1_alpha: torch.Tensor | None
    gemm1_beta: torch.Tensor | None
    gemm1_clamp_limit: torch.Tensor | None
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    output1_scale_scalar: torch.Tensor
    output1_scale_gate_scalar: torch.Tensor
    output2_scale_scalar: torch.Tensor
    # The layer input's NVFP4 global scale: FP32, once per expert, of which
    # the quantization reads the first (as vLLM's does).
    input_scale: torch.Tensor
    num_experts: int
    top_k: int
    intermediate_size: int
    activation_type: int
    routing_method_type: int
    tune_max_num_tokens: int


def expert_params(mlp: "Qwen4ExpSparseMoeBlock") -> ExpertParams:
    """The parameters of `mlp`'s routed experts as their `apply` passes
    them to FlashInfer."""
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    routed = mlp.experts.routed_experts
    experts = _monolithic_experts(mlp)
    assert experts is not None
    config = experts.quant_config
    assert config.w1_scale is not None and config.w2_scale is not None
    assert config.a1_gscale is not None
    assert config.g1_alphas is not None and config.g2_alphas is not None
    return ExpertParams(
        gemm1_weights=routed.w13_weight,
        gemm1_weights_scale=config.w1_scale.view(torch.float8_e4m3fn),
        gemm1_alpha=experts.gemm1_alpha,
        gemm1_beta=experts.gemm1_beta,
        gemm1_clamp_limit=experts.gemm1_clamp_limit,
        gemm2_weights=routed.w2_weight,
        gemm2_weights_scale=config.w2_scale.view(torch.float8_e4m3fn),
        output1_scale_scalar=experts.g1_scale_c,
        output1_scale_gate_scalar=config.g1_alphas,
        output2_scale_scalar=config.g2_alphas,
        input_scale=config.a1_gscale,
        num_experts=experts.local_num_experts,
        top_k=experts.topk,
        intermediate_size=experts.intermediate_size_per_partition,
        activation_type=activation_to_flashinfer_int(routed.activation),
        routing_method_type=int(experts.routing_method_type),
        tune_max_num_tokens=fi_moe_largest_bucket(experts.moe_config),
    )


def stock_tactic(
    params: ExpertParams,
    logits: torch.Tensor,
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
) -> list[int]:
    """The `[tile_n, config]` tactic FlashInfer's autotuner gives the stock
    FP4 MoE call on these shapes (`logits` `[rows, experts]`, the
    quantized input, the top-k outputs and the finalized `output`): the
    lookup of `trtllm_fp4_block_scale_moe_op` with its runner and tuning
    configuration. `FALLBACK` when the tuning cache has no tactic or holds
    the fallback, as the stock call then passes."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.backends.trtllm import MoERunner
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.tllm_enums import (
        Fp8QuantizationType,
        RoutingInputMode,
        SfLayout,
        WeightLayout,
        deduce_trtllm_gen_tensor_dtype,
    )

    hidden = x_fp4.shape[-1] * 2
    runner = MoERunner(
        get_trtllm_moe_sm100_module().moe_op,
        top_k=params.top_k,
        num_local_experts=params.num_experts,
        dtype_act=deduce_trtllm_gen_tensor_dtype(x_fp4, x_scale),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(
            params.gemm1_weights, params.gemm1_weights_scale
        ),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden,
        hidden_size_output=hidden,
        intermediate_size=params.intermediate_size,
        activation_type=params.activation_type,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=params.num_experts,
        num_fused_shared_experts=0,
    )
    inputs = MoeRunnerInputs(
        output=output,
        routing_logits=logits,
        topk_ids=ids,
        expert_weights=weights,
        hidden_states=x_fp4,
        hidden_states_scale=x_scale,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    config = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=params.tune_max_num_tokens,
        routing_input_mode=RoutingInputMode.FromLogits,
        hidden_states_scale_layout=SfLayout.layout_linear,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::trtllm_fp4_block_scale_moe", [runner], config, inputs.to_list()
    )
    if tactic == -1:
        return FALLBACK
    return [int(value) for value in tactic]


def fallback_tile(rows: int, top_k: int, experts: int) -> int:
    """The token tile FlashInfer's FP4 MoE launcher resolves `FALLBACK` to
    in a call of `rows` rows (`selectDefaultTileN`): of its FP4 tiles,
    the one below the smallest tile that covers the average rows per expert,
    or that tile when it is the smallest."""
    average = rows * top_k / experts
    cover = next(
        (i for i, tile in enumerate(FP4_TILES) if tile >= average),
        len(FP4_TILES) - 1,
    )
    return FP4_TILES[max(cover - 1, 0)]


def run_body(
    params: ExpertParams,
    metadata: "RoutingMetadata",
    rows: int,
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    tactic: list[int],
    finalize: bool,
) -> torch.Tensor | UnfinalizedMoEOutput:
    """The expert GEMMs of `rows` rows on `metadata` and the quantized
    input (`x_fp4` and `x_scale`, `rows` rows each) with `tactic`:
    FlashInfer's FP4 MoE operation, once to prepare the body's buffers and
    once to run it. With `finalize`, the body ends with FlashInfer's
    finalize and the BF16 `[rows, hidden]` output is returned; without, it
    ends with GEMM2 and its rows are returned with the record's routing
    weights and expert slots (the stock deferred call's output)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.quantization.nvfp4_quantization_utils import (
        NVFP4_4OVER6_CODE_FROM_ENV,
    )

    from ..moe.front import TOP_K

    op = get_trtllm_moe_sm100_module().moe_op.trtllm_fp4_block_scale_moe
    routing = metadata.tensors(rows)
    # The finalized output; never written without ``finalize``, where its
    # shape sizes GEMM2's N as in the stock deferred call.
    output = torch.empty(
        rows, x_fp4.shape[-1] * 2, dtype=torch.bfloat16, device=x_fp4.device
    )

    def call(workspace: list[torch.Tensor], prepare: bool) -> list[torch.Tensor]:
        """One call of the operation on the prepared routing."""
        return op(
            PACKED_PRECOMPUTED,
            None,  # routing logits
            metadata.ids[:rows],  # top-k placeholders the body does not read
            metadata.weights[:rows],
            None,  # routing bias
            x_fp4,
            x_scale,
            params.gemm1_weights,
            params.gemm1_weights_scale,
            None,  # gemm1 bias
            None,  # gemm1 LoRA delta
            params.gemm1_alpha,
            params.gemm1_beta,
            params.gemm1_clamp_limit,
            params.gemm2_weights,
            params.gemm2_weights_scale,
            None,  # gemm2 bias
            params.output1_scale_scalar,
            params.output1_scale_gate_scalar,
            params.output2_scale_scalar,
            None,  # per-token scale
            params.num_experts,
            params.top_k,
            0,  # fused shared experts
            0,  # expert groups
            0,  # top-k groups
            params.intermediate_size,
            0,  # local expert offset
            params.num_experts,
            None,  # routed scaling factor
            params.routing_method_type,
            finalize,
            True,  # programmatic dependent launch
            params.activation_type,
            output,
            tactic,
            True,  # renormalize the top-k
            None,  # routing replay
            routing,
            workspace,
            prepare,
            None,
            None,
            None,  # workspace buffer: none with a prepared routing
            False,  # not a workspace size query
            NVFP4_4OVER6_CODE_FROM_ENV,  # the stock call's default
        )

    workspace = list(call([], True))
    result = call(workspace, False)
    if finalize:
        return output
    gemm2_rows, weights, slots = result
    return UnfinalizedMoEOutput(
        gemm2_permuted=gemm2_rows,
        expert_weights=weights,
        expanded_idx_to_permuted_idx=slots[: rows * TOP_K],
    )


def unsupported(mlp: "Qwen4ExpSparseMoeBlock") -> str:
    """What running `mlp`'s routed experts on the front needs that it
    misses, or `""`."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )

    from ..moe import front

    runner = mlp.experts
    routed = runner.routed_experts
    experts = _monolithic_experts(mlp)
    if not isinstance(experts, TrtLlmNvFp4ExpertsMonolithic):
        return "FlashInfer's monolithic TRT-LLM NVFP4 experts"
    config = experts.quant_config
    # Not None: ``_monolithic_experts`` found the kernel's experts.
    kernel = routed.quant_method.moe_kernel
    assert kernel is not None
    impl = kernel.impl
    requirements = {
        "softmax top-10 of 512 experts, renormalized, over hidden size 2560": (
            int(experts.routing_method_type) == RENORMALIZE_NAIVE
            and experts.topk == front.TOP_K
            and experts.local_num_experts == front.EXPERTS
            and experts.moe_config.num_experts == front.EXPERTS
            and runner.moe_config.hidden_dim == front.HIDDEN
            and routed.e_score_correction_bias is None
            and not routed.num_expert_group
            and not routed.topk_group
            and routed.routed_scaling_factor in (None, 1.0)
            and not routed.apply_router_weight_on_input
        ),
        "a BF16 router GEMM on the layer input, its logits passed through, and "
        "a shared expert the runner runs outside its kernel": (
            runner.gate is not None
            and not runner._fse_fuse_gate
            and runner.gate.weight.dtype == torch.bfloat16
            and type(runner.router).prepare_monolithic_logits
            is FusedMoERouter.prepare_monolithic_logits
            and runner.shared_experts is not None
            and not runner.shared_experts._mk_can_overlap_shared_experts()
        ),
        "linear-layout NVFP4 input scales": (
            isinstance(impl.prepare_finalize, MoEPrepareAndFinalizeNoDPEPMonolithic)
            and config.use_nvfp4_w4a4
            and not config.is_scale_swizzled
            and config.a1_gscale is not None
            and not experts.expects_unquantized_inputs
        ),
        "no expert parallelism, routing replay or post-GEMM hook": (
            experts.ep_rank == 0
            and not experts._uses_post_gemm2_hook(1)
            and routed.routing_sink is None
        ),
    }
    return "; ".join(need for need, ok in requirements.items() if not ok)


def _monolithic_experts(
    mlp: "Qwen4ExpSparseMoeBlock",
) -> "TrtLlmNvFp4ExpertsMonolithic | None":
    """The experts of `mlp`'s monolithic modular kernel, or None."""
    kernel = getattr(mlp.experts.routed_experts.quant_method, "moe_kernel", None)
    impl = getattr(kernel, "impl", None)
    return getattr(impl, "fused_experts", None)


class RoutedPath:
    """The front and the expert bodies of the executor's TP 1 MoE layers.

    Args:
        layers: Per position among the executor's layers, its MoE layer when
            the routed path serves it, else None.
        device: The layers' device.

    """

    def __init__(
        self, layers: list["Qwen4ExpSparseMoeBlock | None"], device: torch.device
    ) -> None:
        from ..moe import front

        assert front.MAX_ROWS == MAX_ROWS
        self.params: list[ExpertParams | None] = [
            None if mlp is None else expert_params(mlp) for mlp in layers
        ]
        self.records: list[RoutingRecord | None] = [
            None if mlp is None else RoutingRecord.allocate(device) for mlp in layers
        ]
        # Shapes of the stock call for the tactic lookup (never read).
        self._shapes = RoutingRecord.allocate(device)
        self._logits = torch.zeros(
            MAX_ROWS, front.EXPERTS, dtype=torch.bfloat16, device=device
        )
        self._output = torch.zeros(
            MAX_ROWS, front.HIDDEN, dtype=torch.bfloat16, device=device
        )
        # The tuned stock tactic per row count (``tactic``).
        self._tactics: dict[int, list[int]] = {}

    @classmethod
    def build(
        cls, layers: list["Qwen4ExpSparseMoeBlock | None"], owner: str
    ) -> "RoutedPath | None":
        """The routed path of the MoE layers in `layers` (None at positions
        without one) that it supports, or None when it supports none; `owner`
        names the layers' model for the log."""
        served: list[Qwen4ExpSparseMoeBlock | None] = []
        missing: dict[str, None] = {}
        for mlp in layers:
            reason = "" if mlp is None else unsupported(mlp)
            if reason:
                missing[reason] = None
            served.append(None if reason else mlp)
        if missing:
            logger.info_once(
                "Qwen4Exp specialized decode: some MoE layers of %s keep the stock "
                "routing (the routed front needs %s).",
                owner,
                "; ".join(missing),
            )
        first = next((mlp for mlp in served if mlp is not None), None)
        if first is None:
            return None
        logger.info_once(
            "Qwen4Exp specialized decode: the routed front serves %d MoE layers "
            "of %s in steps of up to %d rows.",
            sum(mlp is not None for mlp in served),
            owner,
            MAX_ROWS,
        )
        return cls(served, first.experts.routed_experts.w13_weight.device)

    def warm_up(self) -> None:
        """Compile and run the front once (one row) on zero logits and
        inputs."""
        params = next(p for p in self.params if p is not None)
        x = torch.zeros_like(self._output[:1])
        self._shapes.front(self._logits[:1], x, params.input_scale)
        torch.accelerator.synchronize(self._output.device)

    def tactic(self, position: int, rows: int) -> list[int] | None:
        """The tactic of the routed path for the layer at `position` in a
        call of `rows` rows, or None when the call runs the stock runner:
        the layer is not served, the call has more than `MAX_ROWS` rows, or
        the stock call's GEMMs use another token tile than the front's. The
        tactic is the stock call's (`stock_tactic`), so the routed path
        passes `FALLBACK` where the stock call does and the launcher
        resolves both alike. Every served layer has the same shapes, so a
        tuned tactic is cached by row count; the fallback is looked up again
        at the next call, as the stock call does, until the autotuner has a
        tactic (vLLM tunes after the memory profiling run)."""
        from ..moe.front import TILE

        params = self.params[position]
        if params is None or not 0 < rows <= MAX_ROWS:
            return None
        tactic = self._tactics.get(rows)
        if tactic is None:
            shapes = self._shapes
            tactic = stock_tactic(
                params,
                self._logits[:rows],
                shapes.x_fp4[:rows],
                shapes.x_scale[:rows].view(torch.float8_e4m3fn),
                shapes.metadata.ids[:rows],
                shapes.metadata.weights[:rows],
                self._output[:rows],
            )
            if tactic != FALLBACK:
                self._tactics[rows] = tactic
        tile = (
            fallback_tile(rows, params.top_k, params.num_experts)
            if tactic == FALLBACK
            else tactic[0]
        )
        logger.info_once(
            "Qwen4Exp specialized decode: %d-row MoE calls %s (stock tactic %s).",
            rows,
            "take the routed front" if tile == TILE else "keep the stock routing",
            str(tactic),
        )
        return tactic if tile == TILE else None

    def routed(
        self,
        position: int,
        logits: torch.Tensor,
        x: torch.Tensor,
        tactic: list[int],
        deferred: Callable[[UnfinalizedMoEOutput], None] | None = None,
    ) -> torch.Tensor | None:
        """The routed output of the layer at `position` for router
        `logits` and input `x` with `tactic` (`tactic(position,
        rows)`): the front, then the expert body, on the layer's record.
        With `deferred`, the body stops after GEMM2 and hands its
        unfinalized output to `deferred` right after enqueuing it (as the
        stock experts call their `post_gemm2_hook`), and None is returned."""
        params = self.params[position]
        record = self.records[position]
        assert params is not None and record is not None
        record.front(logits, x, params.input_scale)
        output = record.body(params, x.shape[0], tactic, deferred is None)
        if deferred is None:
            assert isinstance(output, torch.Tensor)
            return output
        assert isinstance(output, UnfinalizedMoEOutput)
        deferred(output)
        return None

    def forward(
        self,
        position: int,
        mlp: "Qwen4ExpSparseMoeBlock",
        x: torch.Tensor,
        tactic: list[int],
        deferred: Callable[[UnfinalizedMoEOutput], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`mlp`'s runner body (`MoERunner._forward_impl` at TP 1) on the
        routed path: the shared expert as the runner runs it (on its side
        stream when it overlaps it, else on this stream after the gate), the
        gate, then `routed` (with `deferred`). Returns the shared and the
        routed output (None with `deferred`)."""
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )

        runner = mlp.experts
        runner.routed_experts._ensure_moe_quant_config_init()
        shared_experts = runner.shared_experts
        assert shared_experts is not None and runner.gate is not None
        assert shared_experts._determine_shared_experts_order(x) in (
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            SharedExpertsOrder.NO_OVERLAP,
        )
        overlapping = shared_experts.maybe_forward_async(x)
        logits, _ = runner.gate(x)
        if not overlapping:
            shared_experts(x, SharedExpertsOrder.NO_OVERLAP)
        routed = self.routed(position, logits, x, tactic, deferred)
        if overlapping:
            shared_experts.wait()
        return shared_experts.output, routed
