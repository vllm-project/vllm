# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's only calls into FlashInfer; no other module of the
package names the library:

- the routed experts: the TRT-LLM-gen MoE op the stock layer runs, with the stock
  layer's own weights and arguments, stopping after GEMM2 (``do_finalize=False``)
  so that the tail kernel finalizes. NVFP4 layers take routing prepared by the
  caller; the BF16 layers route inside the op from the router logits, as the
  stock layer does.
- on NVFP4 layers, the op's own runner at the op's tactic, run from routing
  metadata the routing kernel wrote (``MuseRouting``) instead
  of the op's routing kernel, so the output is the op's bit for bit; a tactic
  without a prepared body takes the public op.
- the TP16 all-reduce, through vLLM's own wrapper (the TP communicator's MNNVL
  all-reduce). TP4 and TP8 fuse their all-reduce into the attention boundary,
  summing in the order of the backend vLLM's FlashInfer all-reduce resolved to;
  across trays (TP8) build() still checks that it is the MNNVL one.
- with expert TP (``build(..., use_expert_tp=True)``), the NVFP4 layers' routed
  op runs on TP experts (all experts at 1/tp of the intermediate width,
  resharded by ``expert_tp``) at one pinned tactic (``EXPERT_TP_TACTICS``); their
  GEMM2 rows are K-partials, which ``TpTail`` sums across ranks.

The library is imported lazily inside the expert calls, as the stock experts do,
so the package imports without it.
"""

import time
from typing import Any, NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    convert_flashinfer_moe_output,
)

from ..cutedsl.common import policy
from . import expert_tp
from .cute_ops import (
    METADATA_MAX_TILE,
    ROUTING_MAX_TOKENS,
    RoutingMetadata,
    metadata_size,
)
from .providers import (
    BATCHES,
    HIDDEN,
    NUM_EXPERTS,
    POOLED,
    PREPARED_MAX,
    TOP_K,
    ExpertsInput,
    RoutedTopk,
)

logger = init_logger(__name__)

# The library functions this module reaches, by public name.
NVFP4_EXPERTS = "fused_moe.trtllm_fp4_block_scale_routed_moe"
# The entry below it that takes prepared routing metadata, and the autotuner
# identity of the op, under which its tactics are tuned.
NVFP4_EXPERTS_FROM_METADATA = "fused_moe.core.TrtllmMoERunner.forward"
NVFP4_OP = "flashinfer::trtllm_fp4_block_scale_moe"
BF16_EXPERTS = "fused_moe.trtllm_bf16_moe"
ALL_REDUCE = "comm.allreduce_fusion"
# The tensor-parallel sizes whose all-reduce goes through this module (TP4 and
# TP8 fuse theirs into the attention boundary).
ALL_REDUCE_TPS = (16,)
# The tensor-parallel sizes whose stock all-reduce must be the MNNVL one: the TP
# groups that span trays. At every size the stock all-reduce must be vLLM's
# FlashInfer one: TP4 and TP8 fuse it in its backend's order, TP16 calls it.
MNNVL_TPS = (8, 16)
ALL_REDUCE_BACKENDS = ("auto", "mnnvl")
# The sum order of each backend of vLLM's FlashInfer all-reduce, the order the
# fused all-reduces reproduce: TRT-LLM's node-local one-shot adds in BF16,
# MNNVL's one-shot and two-shot sum in FP32.
BACKEND_SUM_ORDERS = {"trtllm": policy.BF16_ADDS, "mnnvl": policy.FP32_ADDS}


def expected_calls(tp: int, bf16_layers: bool) -> frozenset[str]:
    """The library functions one decode step reaches through this module (NVFP4
    layers: the entry below the public op, expert TP or not; the public op only
    where a tactic has no prepared body, which is logged)."""
    calls = {NVFP4_EXPERTS_FROM_METADATA}
    if bf16_layers:
        calls.add(BF16_EXPERTS)
    if tp in ALL_REDUCE_TPS:
        calls.add(ALL_REDUCE)
    return frozenset(calls)


# The attention backend whose metadata and KV cache layout the decode kernels read.
ATTENTION_BACKEND = "FLASHINFER"


def check_attention(attention: nn.Module) -> None:
    """Raises unless the stock attention layer runs the FlashInfer backend: the
    executor reads its decode metadata (block table, sequence lengths) and its
    KV cache layout."""
    name = attention.attn_backend.get_name()
    if name != ATTENTION_BACKEND:
        raise ValueError(
            f"Muse Spark specialized decode needs the {ATTENTION_BACKEND} attention "
            f"backend, got {name}"
        )


def bf16_kv_insert(attn: nn.Module, num_kv_heads: int, head_dim: int) -> bool:
    """Whether this attention layer stores K/V with FlashInfer's bf16
    ``reshape_and_cache_flash`` (``FlashInferImpl.do_kv_cache_update``), whose
    HND cache layout the draft's fused context K/V store writes."""
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    impl = getattr(attn, "impl", None)
    return (
        isinstance(impl, FlashInferImpl)
        and type(impl).do_kv_cache_update is FlashInferImpl.do_kv_cache_update
        and impl.kv_sharing_target_layer_name is None
        and not impl.is_kvcache_nvfp4
        and not impl.nvfp4_fa2
        and impl.cache_dtype in ("auto", "bfloat16")
        and impl.num_kv_heads == num_kv_heads
        and impl.head_size == head_dim
    )


def library_rope(rotary_emb: nn.Module) -> bool:
    """Whether a rotary embedding runs FlashInfer's RoPE instead of the stock
    CUDA op."""
    return bool(rotary_emb.use_flashinfer)


def trtllm_decode(metadata) -> bool:
    """Whether a layer's decode metadata is the TRT-LLM form attn2 reads (block
    table and sequence lengths), not the native decode wrapper's."""
    from vllm.v1.attention.backends.flashinfer import FlashInferTrtllmAPIDecode

    return isinstance(metadata.decode, FlashInferTrtllmAPIDecode)


def unsupported(vllm_config) -> str | None:
    """The deployment requirements that name the library (the attention backend,
    vLLM's FlashInfer all-reduce and, at TP8/16, its MNNVL backend), or None."""
    attention_config = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention_config, "backend", None)
    # attn2 reads the TRT-LLM decode metadata, which this setting turns off.
    if getattr(attention_config, "use_trtllm_attention", None) is False:
        return "FlashInfer attention without its TRT-LLM decode"
    if backend is not None and getattr(backend, "name", str(backend)) != (
        ATTENTION_BACKEND
    ):
        return f"the {getattr(backend, 'name', backend)} attention backend"
    tp = vllm_config.parallel_config.tensor_parallel_size
    if not envs.VLLM_ALLREDUCE_USE_FLASHINFER:
        return f"TP{tp} with VLLM_ALLREDUCE_USE_FLASHINFER=0"
    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    if tp in MNNVL_TPS and backend not in ALL_REDUCE_BACKENDS:
        return f"TP{tp} with VLLM_FLASHINFER_ALLREDUCE_BACKEND={backend}"
    return None


def nvfp4_arguments(experts, routed: nn.Module) -> dict[str, Any]:
    """The per-layer arguments of the NVFP4 routed experts op, read from the stock
    MuseSparkTrtLlmNvFp4Experts and its RoutedExperts as apply_routed passes them
    (vllm/models/muse_spark/nvidia/muse_moe_runner.py, apply_routed)."""
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    quant = experts.quant_config
    if experts.post_gemm2_hook is None or experts.expects_unquantized_inputs:
        raise RuntimeError(
            "The stock NVFP4 experts do not take prepared input with a deferred "
            "finalize, so the routed experts op would differ from the stock one."
        )
    if experts.topk != TOP_K or quant.w1_scale is None or quant.w2_scale is None:
        raise RuntimeError("Unexpected stock NVFP4 experts configuration.")
    return dict(
        routing_bias=None,
        gemm1_weights=routed.w13_weight,
        gemm1_weights_scale=quant.w1_scale.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=experts.gemm1_alpha,
        gemm1_beta=experts.gemm1_beta,
        gemm1_clamp_limit=experts.gemm1_clamp_limit,
        gemm2_weights=routed.w2_weight,
        gemm2_weights_scale=quant.w2_scale.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=experts.g1_scale_c,
        output1_scale_gate_scalar=quant.g1_alphas,
        output2_scale_scalar=quant.g2_alphas,
        num_experts=routed.global_num_experts,
        top_k=experts.topk,
        n_group=0,
        topk_group=0,
        intermediate_size=experts.intermediate_size_per_partition,
        local_expert_offset=experts.ep_rank * experts.local_num_experts,
        local_num_experts=experts.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=experts.routing_method_type,
        do_finalize=False,
        activation_type=activation_to_flashinfer_int(routed.activation),
        tune_max_num_tokens=fi_moe_largest_bucket(experts.moe_config),
    )


def stock_shards(args: dict[str, Any], experts) -> expert_tp.StockNvfp4:
    """One NVFP4 layer's stock EP experts, as ``nvfp4_arguments`` bound them,
    for the expert-TP reshard."""
    quant = experts.quant_config
    return expert_tp.StockNvfp4(
        w13=args["gemm1_weights"],
        w13_sf=args["gemm1_weights_scale"],
        w2=args["gemm2_weights"],
        w2_sf=args["gemm2_weights_scale"],
        g1_alphas=args["output1_scale_gate_scalar"],
        g1_scale_c=args["output1_scale_scalar"],
        g2_alphas=args["output2_scale_scalar"],
        a1_gscale=quant.a1_gscale,
        a2_gscale=quant.a2_gscale,
        first_expert=args["local_expert_offset"],
    )


def tp_arguments(args: dict[str, Any], tp: expert_tp.TpExperts) -> dict[str, Any]:
    """``args`` (``nvfp4_arguments``) on the layer's TP experts: their weights,
    block scales and per-expert scalars, every expert local, the intermediate
    slice, and ``tune_max_num_tokens`` at ROUTING_MAX_TOKENS, the most tokens
    the executor runs them at. The rest (routing, activation, global expert
    count) is the stock layer's."""
    return {
        **args,
        "gemm1_weights": tp.w13,
        "gemm1_weights_scale": tp.w13_sf.view(torch.float8_e4m3fn),
        "gemm2_weights": tp.w2,
        "gemm2_weights_scale": tp.w2_sf.view(torch.float8_e4m3fn),
        "output1_scale_scalar": tp.g1_scale_c,
        "output1_scale_gate_scalar": tp.g1_alphas,
        "output2_scale_scalar": tp.g2_alphas,
        "intermediate_size": tp.intermediate,
        "local_expert_offset": 0,
        "local_num_experts": tp.num_experts,
        "tune_max_num_tokens": ROUTING_MAX_TOKENS,
    }


# FP4BlockScaleLauncher::getSupportedTileNums for NVFP4 activations and weights.
FP4_TILES = (8, 16, 32, 64, 128, 192, 256)

# The expert-TP routed GEMMs' tactics as (tile, config), most preferred first,
# per trtllm-gen batched-GEMM cubin package (FlashInfer's
# ArtifactPath.TRTLLM_GEN_BMM). Every rank takes the first one valid on all
# ranks, so every start runs the same kernels. A config id names a kernel pair
# only within the package it was chosen for; under any other package nothing is
# pinned and every rank takes the op's untuned default.
EXPERT_TP_TACTICS: dict[str, tuple[tuple[int, int], ...]] = {
    # FlashInfer 0.7.2.post1: the fastest pairs at the TP8 decode batch.
    "8ee504560a86b6c72b8c941d3003e8fb1804b1b6/batched_gemm-fdba669-16ba8ed/": (
        (8, 199),
        (8, 194),
        (8, 319),
        (8, 79),
        (8, 74),
        (8, 314),
        (8, 200),
        (8, 202),
    ),
}


def default_tile(num_tokens: int, num_local_experts: int) -> int:
    """The tile the op runs an untuned tactic at (the launcher's
    selectDefaultTileN): the smallest of its candidates, the first tile that
    covers the average routes per local expert and its neighbours."""
    average = num_tokens * TOP_K / num_local_experts
    first = next(
        (i for i, tile in enumerate(FP4_TILES) if tile >= average), len(FP4_TILES) - 1
    )
    return FP4_TILES[max(first - 1, 0)]


def tactic_key(tactic: Any, num_tokens: int, num_local_experts: int) -> tuple:
    """(tile, config) of the op's tactic as its launcher resolves it: an
    untuned tactic (-1 anywhere) runs its default config at the default tile."""
    if isinstance(tactic, int):
        tactic = (tactic, tactic)
    tile, config = (int(v) for v in tactic)
    if tile == -1 or config == -1:
        return default_tile(num_tokens, num_local_experts), -1
    return tile, config


def gemm2_rows(num_tokens: int, tile: int, hidden_out: int) -> int:
    """The GEMM2 output rows the op allocates (bf16): the padded permuted rows,
    at least 128 KiB."""
    routes = num_tokens * TOP_K
    filled = min(NUM_EXPERTS, routes)
    ctas = filled + (routes - filled) // tile
    return max(ctas * tile, -(-(128 * 1024 * 8) // (hidden_out * 16)))


class Nvfp4Call:
    """One NVFP4 layer's routed experts op as the op runs them: its runner,
    inputs, tuning config and runner kwargs, built as
    ``trtllm_fp4_block_scale_moe_op`` builds them for the arguments
    ``nvfp4_arguments`` binds, so the autotuner picks the op's tactic and the
    runner launches the op's GEMMs."""

    def __init__(self, args: dict[str, Any]) -> None:
        self.args = args
        self.runner: Any = None
        self.kwargs: dict[str, Any] = {}
        # Per token count: the op's output placeholder (zero-width without
        # the finalize) and its tuning config, which depend on the shapes only.
        self.fixed: dict[int, tuple[torch.Tensor, Any]] = {}

    def bind(self, device: torch.device, max_tokens: int) -> None:
        """Builds the runner and, for 1 to ``max_tokens`` tokens, the
        output placeholder and tuning config, so that ``prepare`` only
        assembles a call's inputs."""
        for b in range(1, max_tokens + 1):
            self.prepare(
                ExpertsInput(
                    torch.zeros((b, POOLED // 2), dtype=torch.uint8, device=device),
                    torch.zeros((b, POOLED // 16), device=device).to(
                        torch.float8_e4m3fn
                    ),
                ),
                RoutedTopk(
                    torch.zeros((b, TOP_K), dtype=torch.int32, device=device),
                    torch.zeros((b, TOP_K), dtype=torch.float32, device=device),
                ),
            )

    def prepare(
        self, inputs: ExpertsInput, topk: RoutedTopk
    ) -> tuple[Any, list, Any, dict[str, Any]]:
        """(runner, inputs, tuning config, runner kwargs) for these inputs;
        allocates and builds nothing for a token count ``bind`` covered."""
        from flashinfer.fused_moe import core
        from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
        from flashinfer.tllm_enums import (
            Fp8QuantizationType,
            RoutingInputMode,
            WeightLayout,
        )

        args = self.args
        a = inputs.a
        num_tokens = a.shape[0]
        a_sf = inputs.a_sf.view(torch.float8_e4m3fn).reshape(num_tokens, -1)
        hidden_size = a.shape[-1] * 2
        device = a.device
        mode = RoutingInputMode.UnpackedPrecomputed
        if self.runner is None:
            self.runner = core.TrtllmMoERunner(
                core.get_trtllm_moe_sm100_module().moe_op,
                top_k=args["top_k"],
                num_local_experts=args["local_num_experts"],
                dtype_act=core.deduce_trtllm_gen_tensor_dtype(a, a_sf),
                dtype_weights=core.deduce_trtllm_gen_tensor_dtype(
                    args["gemm1_weights"], args["gemm1_weights_scale"]
                ),
                fp8_quantization_type=Fp8QuantizationType.NoneFp8,
                hidden_size=hidden_size,
                hidden_size_output=hidden_size,
                intermediate_size=args["intermediate_size"],
                activation_type=args["activation_type"],
                weight_layout=WeightLayout.MajorK,
                use_shuffled_weight=True,
                use_per_token_scaling=False,
                num_experts=args["num_experts"],
                num_fused_shared_experts=0,
            )
            self.kwargs = self.runner_kwargs(core.device_support_pdl(device))
        fixed = self.fixed.get(num_tokens)
        output = (
            core._alloc_trtllm_moe_output(
                num_tokens,
                core._trtllm_moe_output_width(hidden_size, None),
                False,
                device,
            )
            if fixed is None
            else fixed[0]
        )
        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=None,
            topk_ids=topk.ids,
            expert_weights=topk.weights,
            hidden_states=a,
            hidden_states_scale=a_sf,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        if fixed is None:
            fixed = (
                output,
                self.runner._make_tuning_config(
                    moe_inputs,
                    tune_max_num_tokens=args["tune_max_num_tokens"],
                    routing_input_mode=mode,
                    hidden_states_scale_layout=None,
                    use_cold_l2_cache=True,
                    use_cuda_graph=True,
                ),
            )
            self.fixed[num_tokens] = fixed
        return self.runner, moe_inputs.to_list(), fixed[1], self.kwargs

    def runner_kwargs(self, enable_pdl: bool) -> dict[str, Any]:
        """The runner's keyword arguments, as the op passes them."""
        from flashinfer.tllm_enums import RoutingInputMode

        args = self.args
        return {
            "routing_input_mode": int(RoutingInputMode.UnpackedPrecomputed),
            "num_experts": args["num_experts"],
            "routing_bias": None,
            "gemm1_weights": args["gemm1_weights"],
            "gemm1_weights_scale": args["gemm1_weights_scale"],
            "gemm1_bias": None,
            "gemm1_alpha": args["gemm1_alpha"],
            "gemm1_beta": args["gemm1_beta"],
            "gemm1_clamp_limit": args["gemm1_clamp_limit"],
            "gemm2_weights": args["gemm2_weights"],
            "gemm2_weights_scale": args["gemm2_weights_scale"],
            "gemm2_bias": None,
            "output1_scale_scalar": args["output1_scale_scalar"],
            "output1_scale_gate_scalar": args["output1_scale_gate_scalar"],
            "output2_scale_scalar": args["output2_scale_scalar"],
            "n_group": args["n_group"],
            "topk_group": args["topk_group"],
            "local_expert_offset": args["local_expert_offset"],
            "routed_scaling_factor": None,
            "routing_method_type": args["routing_method_type"],
            # The op's PDL default.
            "enable_pdl": enable_pdl,
            "do_finalize": False,
            "activation_type": args["activation_type"],
            "num_fused_shared_experts": 0,
            "norm_topk_prob": True,
            "routing_replay_out": None,
            "valid_hidden_size": None,
            "valid_intermediate_size": None,
        }


class Nvfp4Plan(NamedTuple):
    """How one NVFP4 call runs below the public op: the op's tactic for these
    inputs as (tile, config), where the routing writes the metadata at that
    tile, and the prepared call (runner, inputs, runner kwargs)."""

    tactic: tuple[int, int]
    metadata: RoutingMetadata
    call: tuple[Any, list, dict[str, Any]]


def bf16_arguments(experts, routed: nn.Module) -> dict[str, Any]:
    """The per-layer arguments of the BF16 experts op, read from the stock
    TrtLlmBf16ExpertsMonolithic and its RoutedExperts as the stock layer passes them
    (RoutedExperts.forward_monolithic -> apply_monolithic -> apply, with the
    post-expert-norm hook attached, so do_finalize=False)."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        view_as_block_major_k,
    )
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    if experts.post_gemm2_hook is None:
        raise RuntimeError(
            "The stock BF16 experts finalize inside the op, so the experts op "
            "would differ from the stock one."
        )
    if routed.routing_sink is not None:
        raise RuntimeError("Routed experts capture is not supported.")
    if experts.topk != TOP_K:
        raise RuntimeError("Unexpected stock BF16 experts configuration.")
    return dict(
        routing_bias=routed.e_score_correction_bias,
        gemm1_weights=view_as_block_major_k(routed.w13_weight),
        gemm2_weights=view_as_block_major_k(routed.w2_weight),
        num_experts=routed.global_num_experts,
        top_k=experts.topk,
        n_group=routed.num_expert_group,
        topk_group=routed.topk_group,
        intermediate_size=experts.intermediate_size_per_partition,
        local_expert_offset=experts.ep_rank * experts.local_num_experts,
        local_num_experts=experts.local_num_experts,
        routed_scaling_factor=routed.routed_scaling_factor,
        routing_method_type=experts.routing_method_type,
        activation_type=activation_to_flashinfer_int(routed.activation),
        tune_max_num_tokens=fi_moe_largest_bucket(experts.moe_config),
        routing_replay_out=None,
        do_finalize=False,
    )


def stock_experts(moe: nn.Module) -> tuple[Any, nn.Module]:
    """The stock experts kernel and RoutedExperts of a MuseSpark1xMoE."""
    routed = moe.experts.routed_experts
    kernel = getattr(routed.quant_method, "moe_kernel", None)
    if kernel is None:
        raise RuntimeError("The stock MoE layer has no experts kernel.")
    return kernel.fused_experts, routed


def stock_all_reduce(communicator) -> tuple[Any, str | None]:
    """The TP communicator's FlashInfer all-reduce wrapper, checked to take the
    decode's [PREPARED_MAX, HIDDEN] bf16 rows (so the stock layers' all-reduce
    is this one), and the backend its workspace resolved to
    (VLLM_FLASHINFER_ALLREDUCE_BACKEND; auto: MNNVL, on one node TRT-LLM where
    MNNVL has no multicast). Creates the workspace on first use, collectively
    over the TP group, as the stock path does; vLLM keeps one per process, so
    every caller reads the same backend."""
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )

    wrapper = getattr(communicator, "fi_ar_comm", None)
    if wrapper is None or wrapper.disabled:
        raise RuntimeError(
            "The specialized decode's all-reduce needs vLLM's FlashInfer all-reduce."
        )
    probe = torch.empty(
        (PREPARED_MAX, HIDDEN), dtype=torch.bfloat16, device=communicator.device
    )
    if not wrapper.should_use_fi_ar(probe):
        raise RuntimeError(
            f"vLLM's FlashInfer all-reduce does not take [{PREPARED_MAX}, {HIDDEN}] "
            "bf16 here."
        )
    workspace = get_fi_ar_workspace(
        world_size=wrapper.world_size,
        rank=wrapper.rank,
        max_token_num=wrapper.max_num_tokens,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        group=wrapper.group,
    )
    return wrapper, getattr(workspace, "backend", None)


def stock_sum_order(communicator) -> str:
    """The sum order of the stock TP all-reduce (``stock_all_reduce``'s backend,
    ``BACKEND_SUM_ORDERS``), the one the fused all-reduces over the TP group
    take. Raises for a backend whose order they do not reproduce. Collective
    over the TP group on the process's first call."""
    _, backend = stock_all_reduce(communicator)
    order = BACKEND_SUM_ORDERS.get(backend) if backend is not None else None
    if order is None:
        raise RuntimeError(
            f"The fused all-reduce has no sum order for vLLM's {backend!r} "
            "FlashInfer all-reduce."
        )
    return order


def bind_all_reduce(communicator):
    """The TP communicator's FlashInfer all-reduce wrapper, checked to be the MNNVL
    all-reduce the stock layers take for [PREPARED_MAX, HIDDEN] bf16
    (``stock_all_reduce``)."""
    backend = envs.VLLM_FLASHINFER_ALLREDUCE_BACKEND
    if backend not in ALL_REDUCE_BACKENDS:
        raise RuntimeError(
            f"The TP8/16 all-reduce needs the MNNVL backend, got {backend!r}."
        )
    wrapper, resolved = stock_all_reduce(communicator)
    if resolved != "mnnvl":
        raise RuntimeError(
            "vLLM's FlashInfer all-reduce did not resolve to the MNNVL backend."
        )
    return wrapper


def tp_coordinator(group):
    """The TP GroupCoordinator, given it or its device group."""
    tp = get_tp_group()
    if group is not tp and group is not tp.device_group:
        raise ValueError("FlashInferOps.build takes the TP group.")
    return tp


def _finite(rows: torch.Tensor) -> str:
    """How finite ``rows`` are, and their largest finite magnitude."""
    finite = torch.isfinite(rows)
    largest = rows.float().abs().masked_fill(~finite, 0).max() if rows.numel() else 0
    return f"{int(finite.sum())}/{rows.numel()} finite (max |x| {float(largest):.3g})"


# The expert-TP build probe: rel L2 of a layer's TP route rows, summed across
# ranks, against its EP rows (expected ~3e-3: one more BF16 rounding per rank;
# a wrong slice or scale gives O(1)).
EXPERT_TP_PROBE_MAX = 1e-2


class FlashInferOps:
    """``RoutedExpertsOps``: the routed experts of every MoE layer and the TP16
    all-reduce.

    build() binds every per-layer argument and checks every requirement once, then
    runs each op once at b = 8 (rank-symmetric; the all-reduce workspace is created
    lazily otherwise). The calls are graph-capturable and read no per-step state.
    """

    def __init__(self) -> None:
        self.nvfp4: dict[int, dict[str, Any]] = {}
        self.bf16: dict[int, dict[str, Any]] = {}
        self.all_reduce_wrapper = None
        # The NVFP4 calls below the public op (prepare_from_metadata).
        self.nvfp4_calls: dict[int, Nvfp4Call] = {}
        self.prepared: set[tuple[int, int]] = set()
        self.body_workspace: tuple[torch.Tensor, ...] = ()
        # Per token count, the (tile, config) keys of the tactics the NVFP4
        # op takes (prepare_from_metadata).
        self.valid_tactics: dict[int, set[tuple[int, int]]] = {}
        # Expert TP: the NVFP4 layers on TP experts (their arguments in
        # nvfp4), the TP experts' bytes and each layer's build probe.
        self.tp_experts: set[int] = set()
        self.tp_bytes = 0
        self.probe_rel_l2: dict[int, float] = {}
        # Their tactic (pin_tp_tactic) and the preferences it came from; None:
        # the op's untuned default.
        self.tp_preferred: tuple[tuple[int, int], ...] = ()
        self.tp_tactic: tuple[int, int] | None = None

    def build(
        self, group, layers: tuple[nn.Module, ...], use_expert_tp: bool = False
    ) -> None:
        """``layers``: the stock decoder layers (layer_idx, feed_forward).

        With ``use_expert_tp`` (collective over the TP group, which must equal the
        stock layers' EP group), every NVFP4 layer runs on TP experts resharded
        from the stock ones (finalized by the package's TP tail); BF16 layers
        stay EP (stock weights, the EP tail)."""
        from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
            TrtLlmBf16ExpertsMonolithic,
        )
        from vllm.models.muse_spark.nvidia.muse_moe_runner import (
            MuseSparkTrtLlmNvFp4Experts,
        )

        tp = tp_coordinator(group)
        for layer in layers:
            experts, routed = stock_experts(layer.feed_forward)
            if isinstance(experts, MuseSparkTrtLlmNvFp4Experts):
                self.nvfp4[layer.layer_idx] = nvfp4_arguments(experts, routed)
            elif isinstance(experts, TrtLlmBf16ExpertsMonolithic):
                self.bf16[layer.layer_idx] = bf16_arguments(experts, routed)
            else:
                raise RuntimeError(
                    f"Layer {layer.layer_idx} runs {type(experts).__name__}, not "
                    "the TRT-LLM-gen experts the specialized decode calls."
                )
        seconds = 0.0
        if use_expert_tp:
            # Before the calls below the op are prepared: they bind the
            # arguments, one shape for every NVFP4 layer.
            start = time.perf_counter()
            self.shard_experts(tp, layers)
            seconds = time.perf_counter() - start
        self.prepare_from_metadata()
        if use_expert_tp:
            # After the calls are prepared: the tactic needs a prepared body.
            self.pin_tp_tactic(tp)
        self.warm_up()
        if use_expert_tp:
            self.log_expert_tp(tp.world_size, seconds)
        if tp.world_size in MNNVL_TPS:
            # At TP8 only the check: the attention boundary sums in this
            # all-reduce's order.
            communicator = tp.device_communicator
            wrapper = bind_all_reduce(communicator)
            if tp.world_size in ALL_REDUCE_TPS:
                self.all_reduce_wrapper = wrapper
                x = torch.zeros(
                    (BATCHES[0], HIDDEN),
                    dtype=torch.bfloat16,
                    device=communicator.device,
                )
                self.all_reduce(x)

    # ------------------------------------------------------------- expert TP
    def shard_experts(self, tp, layers: tuple[nn.Module, ...]) -> None:
        """Replaces every NVFP4 layer's arguments by its TP experts': reshard
        (collective) and a probe against the stock EP op. Collective over
        ``tp``: the placement check is agreed across the ranks before any
        rank raises, so none is left waiting in the reshard."""
        if not self.nvfp4:
            raise RuntimeError("Expert TP needs NVFP4 layers.")
        group = tp.device_group
        nvfp4_layers = [layer for layer in layers if layer.layer_idx in self.nvfp4]
        ep_ranks = {
            stock_experts(layer.feed_forward)[0].ep_rank for layer in nvfp4_layers
        }
        wrong = sorted(ep_ranks - {tp.rank_in_group})
        reason = (
            f"got EP rank {wrong[0]} on TP rank {tp.rank_in_group}" if wrong else None
        )
        reason = expert_tp.agree(tp.cpu_group, reason, "cpu")
        if reason is not None:
            raise RuntimeError(
                f"Expert TP needs the EP rank to be the TP rank: {reason}."
            )
        for layer in nvfp4_layers:
            index = layer.layer_idx
            experts, routed = stock_experts(layer.feed_forward)
            ep_args = self.nvfp4[index]
            shards = expert_tp.reshard_nvfp4(
                stock_shards(ep_args, experts), group, routed.global_num_experts
            )
            tp_args = tp_arguments(ep_args, shards)
            self.probe_rel_l2[index] = self.probe(
                index, tp_args, group, experts.quant_config.a1_gscale
            )
            self.nvfp4[index] = tp_args
            self.tp_experts.add(index)
            self.tp_bytes += sum(
                t.numel() * t.element_size()
                for t in shards
                if isinstance(t, torch.Tensor)
            )

    def log_expert_tp(self, tp: int, seconds: float) -> None:
        """Logs the expert-TP build: what was resharded, its memory, the build
        probes and the tactic the decode batch takes (None: no body prepared
        for it, so the public op runs)."""
        args = self.nvfp4[min(self.tp_experts)]
        num_local = args["local_num_experts"]
        tactic = tactic_key(self.tp_tactic or -1, BATCHES[0], num_local)
        probes = self.probe_rel_l2.values()
        pinned = (
            "the op's default: no preferred tactic valid on every rank"
            if self.tp_tactic is None
            else f"pinned: preference {self.tp_preferred.index(self.tp_tactic) + 1}"
            f" of {len(self.tp_preferred)}, on every rank"
        )
        logger.info(
            "Muse Spark expert TP built: TP%d, %d NVFP4 layers (%d BF16 layers "
            "EP), %d experts at intermediate %d per rank, +%.2f GiB per rank, "
            "probe rel L2 %.3g..%.3g, tactic %s at %d tokens (%s), %.1f s",
            tp,
            len(self.tp_experts),
            len(self.bf16),
            num_local,
            args["intermediate_size"],
            self.tp_bytes / 2**30,
            min(probes),
            max(probes),
            tactic if tactic in self.prepared else None,
            BATCHES[0],
            pinned,
            seconds,
        )

    def probe_inputs(
        self, device: torch.device, global_scale: torch.Tensor
    ) -> tuple[ExpertsInput, RoutedTopk]:
        """The build probe's input at the decode batch, the same on every rank
        (seeded): unit-RMS rows quantized with ``global_scale`` (the layer's
        NVFP4 input global scale), so they stay in the range the layer's scales
        were calibrated for, and routes spread over all experts, so every EP
        rank serves some."""
        from vllm import _custom_ops as ops

        b = BATCHES[0]
        g = torch.Generator(device=device).manual_seed(20261023)
        x = torch.randn((b, POOLED), generator=g, device=device)
        a, a_sf = ops.scaled_fp4_quant(
            x.to(torch.bfloat16), global_scale, is_sf_swizzled_layout=False
        )
        inputs = ExpertsInput(a, a_sf.view(torch.float8_e4m3fn).reshape(b, -1))
        routes = torch.arange(b * TOP_K, dtype=torch.int32, device=device)
        stride = NUM_EXPERTS // (b * TOP_K)
        topk = RoutedTopk(
            (routes * stride + routes // TOP_K % stride).view(b, TOP_K),
            torch.full((b, TOP_K), 1.0 / TOP_K, device=device),
        )
        return inputs, topk

    def probe(
        self,
        layer: int,
        tp_args: dict[str, Any],
        group,
        global_scale: torch.Tensor,
    ) -> float:
        """Rel L2 of the layer's TP route rows, summed across ranks, against its
        stock EP rows on the same stock-quantized input (collective); raises
        above EXPERT_TP_PROBE_MAX on every rank."""
        inputs, topk = self.probe_inputs(tp_args["gemm1_weights"].device, global_scale)
        outputs = []
        for args in (self.nvfp4[layer], tp_args):
            out = self.public_routed_experts(args, inputs, topk)
            outputs.append((out.gemm2_permuted, out.expanded_idx_to_permuted_idx))
        rel = expert_tp.probe_rel_l2(outputs[0], outputs[1], group)
        if not rel < EXPERT_TP_PROBE_MAX:
            ep_rows, ep_local = expert_tp.route_rows(*outputs[0])
            tp_rows, _ = expert_tp.route_rows(*outputs[1])
            raise RuntimeError(
                f"Layer {layer}: the TP experts differ from the stock EP experts "
                f"by rel L2 {rel:.3g} (at most {EXPERT_TP_PROBE_MAX}); on this "
                f"rank the EP rows of its {int(ep_local.sum())} routes are "
                f"{_finite(ep_rows[ep_local])}, its TP partial rows "
                f"{_finite(tp_rows)}."
            )
        logger.debug("Muse Spark expert TP: layer %d probe rel L2 %.3g", layer, rel)
        return rel

    @staticmethod
    def expert_tp_tactics() -> tuple[tuple[int, int], ...]:
        """The preferred expert-TP tactics measured on the installed FlashInfer's
        batched-GEMM cubin package; none for any other package."""
        from flashinfer.artifacts import ArtifactPath

        return EXPERT_TP_TACTICS.get(ArtifactPath.TRTLLM_GEN_BMM, ())

    def pin_tp_tactic(self, tp) -> None:
        """The TP experts' tactic: the first of expert_tp_tactics() that the op
        takes for 1 to ROUTING_MAX_TOKENS tokens and that has a prepared body,
        on every rank of the TP group, else the op's untuned default.
        Collective over ``tp``'s CPU group (every rank runs the same FlashInfer
        build, so the same list); nothing is timed, so the same weights and
        FlashInfer build take the same tactic at every start."""
        self.tp_preferred = self.expert_tp_tactics()
        self.tp_tactic = None
        if not self.tp_preferred:
            from flashinfer.artifacts import ArtifactPath

            logger.warning(
                "Muse Spark expert TP: no tactics measured for FlashInfer's "
                "batched-GEMM cubins %s; the TP experts run the op's untuned "
                "default tactic.",
                ArtifactPath.TRTLLM_GEN_BMM,
            )
            return
        usable = torch.tensor(
            [
                int(
                    tactic in self.prepared
                    and all(tactic in valid for valid in self.valid_tactics.values())
                )
                for tactic in self.tp_preferred
            ],
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            usable, op=torch.distributed.ReduceOp.MIN, group=tp.cpu_group
        )
        first = next((i for i, ok in enumerate(usable.tolist()) if ok), None)
        if first is None:
            logger.warning(
                "Muse Spark expert TP: none of the tactics %s is valid on every "
                "rank; the TP experts run the op's untuned default tactic.",
                self.tp_preferred,
            )
            return
        self.tp_tactic = self.tp_preferred[first]

    def public_routed_experts(
        self, args: dict[str, Any], inputs: ExpertsInput, topk: RoutedTopk
    ) -> UnfinalizedMoEOutput:
        """The public routed op on ``args``."""
        import flashinfer

        num_tokens = inputs.a.shape[0]
        output = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk.ids, topk.weights),
            hidden_states=inputs.a,
            hidden_states_scale=inputs.a_sf.view(torch.float8_e4m3fn).reshape(
                num_tokens, -1
            ),
            **args,
        )
        return convert_flashinfer_moe_output(
            output, do_finalize=False, num_tokens=num_tokens, top_k=TOP_K
        )

    def is_expert_tp(self, layer: int) -> bool:
        """Whether the layer's experts are TP (finalized by the TP tail)
        rather than EP (the EP tail)."""
        return layer in self.tp_experts

    def warm_up(self) -> None:
        """One call of each bound op at b = 8 on its first layer."""
        b = BATCHES[0]
        if self.nvfp4:
            layer = min(self.nvfp4)
            device = self.nvfp4[layer]["gemm1_weights"].device
            # Spread over all experts, so every EP rank runs its GEMMs.
            ids = torch.arange(b * TOP_K, dtype=torch.int32, device=device)
            topk = RoutedTopk(
                (ids * (NUM_EXPERTS // (b * TOP_K))).view(b, TOP_K),
                torch.full((b, TOP_K), 1.0 / TOP_K, device=device),
            )
            inputs = ExpertsInput(
                torch.zeros((b, POOLED // 2), dtype=torch.uint8, device=device),
                torch.zeros((b, POOLED // 16), device=device).to(torch.float8_e4m3fn),
            )
            self.routed_experts(layer, inputs, topk)
        if self.bf16:
            layer = min(self.bf16)
            device = self.bf16[layer]["gemm1_weights"].device
            x = torch.zeros((b, POOLED), dtype=torch.bfloat16, device=device)
            logits = torch.zeros((b, NUM_EXPERTS), device=device)
            self.routed_experts_bf16(layer, x, logits)

    # -------------------------------------- NVFP4 experts from routing metadata
    def prepare_from_metadata(self) -> None:
        """Prepares the NVFP4 layers' calls below the public op for 1 to
        ROUTING_MAX_TOKENS tokens: the routing metadata buffer, the routing
        scratch and one expert body workspace, each sized for the op's largest
        tactic, so a captured step's plan builds nothing. One buffer and one
        workspace serve every layer: a layer's routing waits for the previous
        layer's tail, the last reader."""
        if not self.nvfp4:
            return
        from flashinfer.fused_moe import core

        self.nvfp4_calls = {n: Nvfp4Call(args) for n, args in self.nvfp4.items()}
        call = self.nvfp4_calls[min(self.nvfp4)]
        device = call.args["gemm1_weights"].device
        for layer_call in self.nvfp4_calls.values():
            layer_call.bind(device, ROUTING_MAX_TOKENS)
        num_local = call.args["local_num_experts"]
        # -1 everywhere: the guard element and the unwritten rows stay defined.
        self.metadata = torch.full(
            (metadata_size(METADATA_MAX_TILE),), -1, dtype=torch.int32, device=device
        )
        self.histogram = torch.zeros(
            (2 * NUM_EXPERTS,), dtype=torch.int32, device=device
        )
        self.expert_counts = torch.zeros(
            (NUM_EXPERTS,), dtype=torch.int32, device=device
        )
        workspace: list[torch.Tensor] = []
        for b in range(ROUTING_MAX_TOKENS, 0, -1):
            inputs = ExpertsInput(
                torch.zeros((b, POOLED // 2), dtype=torch.uint8, device=device),
                torch.zeros((b, POOLED // 16), device=device).to(torch.float8_e4m3fn),
            )
            topk = RoutedTopk(
                torch.zeros((b, TOP_K), dtype=torch.int32, device=device),
                torch.zeros((b, TOP_K), dtype=torch.float32, device=device),
            )
            runner, flat, _, kwargs = call.prepare(inputs, topk)
            valid = list(runner.get_valid_tactics(flat, None))
            self.valid_tactics[b] = {tactic_key(t, b, num_local) for t in valid}
            tactics = [(-1, -1)] + valid
            for tactic in tactics:
                key = tactic_key(tactic, b, num_local)
                if key in self.prepared or key[0] > METADATA_MAX_TILE:
                    continue
                metadata = self.routing_metadata(call, key[0])
                fields = runner.forward(
                    flat,
                    tactic=list(key),
                    do_preparation=True,
                    da_routing_metadata=self.slot(metadata, topk.weights),
                    **kwargs,
                )
                fields = [core._torch_view_of_ffi_tensor(f) for f in fields]
                if not workspace:
                    workspace = fields
                if [(f.dtype, f.device) for f in fields] != [
                    (w.dtype, w.device) for w in workspace
                ]:
                    raise RuntimeError("The expert body workspace differs by tactic.")
                workspace = [
                    max(w, f, key=torch.numel) for w, f in zip(workspace, fields)
                ]
                self.prepared.add(key)
        self.body_workspace = tuple(workspace)

    def routing_metadata(self, call: Nvfp4Call, tile: int) -> RoutingMetadata:
        return RoutingMetadata(
            self.metadata,
            call.args["local_expert_offset"],
            call.args["local_num_experts"],
            tile,
        )

    def slot(self, metadata: RoutingMetadata, weights: torch.Tensor) -> list:
        """The op's nine routing metadata tensors, in its order, for the
        tokens of ``weights`` (fp32 [b, TOP_K], the routing's)."""
        f = metadata.fields(weights.shape[0])
        return [
            f["total_num_padded_tokens"],
            f["expanded_idx_to_permuted_idx"],
            f["permuted_idx_to_token_idx"],
            weights,
            self.histogram,
            self.expert_counts,
            f["cta_idx_xy_to_batch_idx"],
            f["cta_idx_xy_to_mn_limit"],
            f["num_non_exiting_ctas"],
        ]

    def nvfp4_plan(
        self, layer: int, inputs: ExpertsInput, topk: RoutedTopk
    ) -> Nvfp4Plan | None:
        """Host only, before the routing launches: the tactic for these inputs
        (on an expert-TP layer the pinned one, never the autotuner's; else the
        one the public op's autotuner takes) and the routing metadata it needs,
        or None (the public op runs, routing included) when no body was
        prepared for that tactic."""
        from flashinfer.autotuner import AutoTuner

        call = self.nvfp4_calls.get(layer)
        if call is None:
            return None
        runner, flat, tuning_config, kwargs = call.prepare(inputs, topk)
        if self.is_expert_tp(layer):
            tactic: Any = -1 if self.tp_tactic is None else self.tp_tactic
        else:
            _, tactic = AutoTuner.get().choose_one(
                NVFP4_OP, [runner], tuning_config, flat, **kwargs
            )
        key = tactic_key(tactic, inputs.a.shape[0], call.args["local_num_experts"])
        if key not in self.prepared:
            logger.warning_once(
                "Muse Spark specialized decode: no prepared expert body for tactic "
                "%s; the routed experts op routes itself.",
                key,
            )
            return None
        return Nvfp4Plan(
            key, self.routing_metadata(call, key[0]), (runner, flat, kwargs)
        )

    def is_bf16(self, layer: int) -> bool:
        """Whether the layer's experts are BF16 (they take normed BF16 rows and
        the router logits) rather than NVFP4 (``ExpertsInput`` and
        ``RoutedTopk``)."""
        return layer in self.bf16

    def routed_experts(
        self,
        layer: int,
        inputs: ExpertsInput,
        topk: RoutedTopk,
        plan: Nvfp4Plan | None = None,
    ) -> UnfinalizedMoEOutput:
        """The NVFP4 expert input and its routed top-k -> the unfinalized
        expert output, on an NVFP4 layer. With ``plan`` (``nvfp4_plan`` for
        these inputs), from the metadata the routing wrote for it."""
        if plan is not None:
            return self.routed_experts_from_metadata(inputs, topk, plan)
        return self.public_routed_experts(self.nvfp4[layer], inputs, topk)

    def routed_experts_from_metadata(
        self, inputs: ExpertsInput, topk: RoutedTopk, plan: Nvfp4Plan
    ) -> UnfinalizedMoEOutput:
        """The op's GEMMs at its tactic, from the routing metadata, without
        its routing kernel. Returns the op's unfinalized output at its extents:
        the GEMM2 rows it allocates, the routing weights, the permutation."""
        from flashinfer.fused_moe import core

        runner, flat, kwargs = plan.call
        num_tokens = inputs.a.shape[0]
        out = runner.forward(
            flat,
            tactic=list(plan.tactic),
            da_routing_metadata=self.slot(plan.metadata, topk.weights),
            da_body_workspace=self.body_workspace,
            **kwargs,
        )
        rows, weights, permuted = (core._torch_view_of_ffi_tensor(t) for t in out)
        rows = rows[: gemm2_rows(num_tokens, plan.tactic[0], rows.shape[1])]
        return convert_flashinfer_moe_output(
            [rows, weights, permuted],
            do_finalize=False,
            num_tokens=num_tokens,
            top_k=TOP_K,
        )

    def routed_experts_bf16(
        self, layer: int, x: torch.Tensor, router_logits: torch.Tensor
    ) -> UnfinalizedMoEOutput:
        """The normed rows (bf16 [b, POOLED]) + the router logits (fp32
        [b, NUM_EXPERTS]) -> the unfinalized expert output, on a BF16 layer; the
        op routes, so its expert weights are bf16."""
        import flashinfer

        output = flashinfer.fused_moe.trtllm_bf16_moe(
            routing_logits=router_logits, hidden_states=x, **self.bf16[layer]
        )
        return convert_flashinfer_moe_output(
            output, do_finalize=False, num_tokens=x.shape[0], top_k=TOP_K
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """The TP16 all-reduce of bf16 [b <= PREPARED_MAX, HIDDEN]: the stock
        layers' MNNVL all-reduce through vLLM's wrapper."""
        if self.all_reduce_wrapper is None:
            raise RuntimeError("FlashInferOps.all_reduce serves TP16 only.")
        return self.all_reduce_wrapper.all_reduce(x)


def build(
    group, layers: tuple[nn.Module, ...], use_expert_tp: bool = False
) -> FlashInferOps:
    """The routed expert ops for ``layers`` (the stock decoder layers), built
    and warmed up."""
    ops = FlashInferOps()
    ops.build(group, layers, use_expert_tp=use_expert_tp)
    return ops
