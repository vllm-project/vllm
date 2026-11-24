# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer (routed experts only, NVFP4 or MXFP8).

Experts are sharded contiguously over the TP group (EP size = TP size); each
rank returns the full all-reduced bf16 output. Checkpoint names under
`model.llm.layers.N.mlp` map 1:1 to parameters.

Blackwell (SM10x) and SM12x route into tile-padded expert rows with quantized
activations (`nvidia/ops/moe.py`). Everywhere else the flow is Hopper's:
`select_experts`, then a `FlatExperts` backend for the expert GEMMs on the
flat rows (Humming on CUDA), then `combine` (a backend whose `combines`
is True sums the rows itself).
"""

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling2.nvidia.ops.activation import swiglu_quantize
from vllm.models.inkling2.nvidia.ops.block_scales import (
    shuffle_rows,
    swizzle_block_scales,
)
from vllm.models.inkling2.nvidia.ops.moe import NUM_COUNTERS, combine_push, route
from vllm.models.inkling2.nvidia.ops.moe_humming import HummingNVFP4Experts
from vllm.models.inkling2.nvidia.ops.moe_sm90 import HummingExperts
from vllm.models.inkling2.nvidia.ops.moe_sm120 import bmm as bmm_sm120
from vllm.models.inkling2.nvidia.ops.seq_parallel import SeqParallel
from vllm.models.inkling2.nvidia.ops.trtllm_bmm import bmm, expert_plan, load_kernels
from vllm.platforms import current_platform

from .ops.moe import combine, select_experts
from .ops.platform import use_sm100_kernels

QuantFormat = Literal["nvfp4", "mxfp8"]


class FlatExperts(Protocol):
    """Expert GEMMs over the flat (token * topk + slot) rows of
    `select_experts` (`HummingExperts` is the reference): built after
    loading as `cls(weights, shapes, nvfp4)` from the checkpoint-layout
    weights, block scales and NVFP4 global scales of `w13` (interleaved
    gate / up rows) and `w2`. A call returns the bf16 `[T * topk, H]`
    down-projection rows of the local slots of `ids` (-1: not local), with
    the softcapped SiTU (`gate_cap`, `up_cap`) of the `w13` output scaled by
    the routing `weights`; the other rows are left unset.

    A backend whose `combines` is True (e.g. the SYCL fused MoE on XPU)
    returns the combined bf16 `[T, H]` output instead: the routing weight
    scales each down-projection row in fp32 and the sum is rounded once to
    bf16, as `combine` does. It serves only the reduced output (no
    `reduce=False` rows, no `push`). Without the attribute it is False.

    A backend whose `keeps_parameters` is True (the XPU ones) is built from
    the layer's Parameters instead of their `.data` and keeps them, in the
    checkpoint layout or rewritten in place into its kernel layout; the
    other backends get the `.data` tensors and the Parameters are emptied
    after the build."""

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor: ...


def flat_experts_class(
    quant_format: QuantFormat, topk: int, gate_cap: float, up_cap: float
) -> Callable[..., FlatExperts]:
    """This platform's `FlatExperts` constructor for `quant_format` experts
    with `topk` slots per token and the SiTU softcaps of the layer: the XPU
    backends on XPU, Humming elsewhere."""
    if current_platform.is_xpu():
        from vllm.models.inkling2.xpu.experts import xpu_flat_experts

        return xpu_flat_experts(quant_format, topk, gate_cap, up_cap)
    return HummingExperts


class ExpertsLoRA(Protocol):
    """Adds the routed experts' LoRA deltas (`lora.Inkling2MoEWithLoRA`),
    for an experts implementation over expert-sorted rows: row `r` is token
    `perm[r] // topk`, and this rank's experts own rows
    `[offsets[0], offsets[-1])`. The NVFP4 / MXFP8 experts here (tile-padded
    rows, quantized activations) do not support it; the BF16 experts a model
    may plug in instead (SkyRL's `Inkling2BF16Experts`) do.
    """

    def add_w13(
        self,
        gate_up: torch.Tensor,
        x: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...

    def add_w2(
        self,
        down: torch.Tensor,
        activated: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
        # Completion counters of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32), persistent=False
        )


class Inkling2Experts(nn.Module):
    # Expert LoRA (`ExpertsLoRA`) needs unquantized experts; experts that
    # support it (e.g. BF16 ones plugged in by a model) need not define this.
    supports_lora = False

    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            # Rows are shuffled and block scales swizzled (``block_scales``).
            self._register(
                f"{name}_weight",
                (n, k // pack),
                weight_dtype,
                loader=self._load_weight,
            )
            self._register(
                f"{name}_weight_scale",
                (n, k // group),
                scale_dtype,
                loader=self._load_block_scales,
            )
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        # Rows per local expert of the routing kernels (reset by the kernels).
        self.register_buffer(
            "counts", torch.zeros(self.num_experts, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "ones", torch.ones(self.num_experts, dtype=torch.float32), persistent=False
        )
        # Humming weights are repacked after loading.
        self.sm100 = use_sm100_kernels()
        self.sm120 = current_platform.is_device_capability_family(120)
        vllm_config = get_current_vllm_config_or_none()
        self.humming_nvfp4 = (
            self.sm120
            and self.nvfp4
            and vllm_config is not None
            and vllm_config.kernel_config.moe_backend == "humming"
        )
        self.humming: HummingNVFP4Experts | None = None
        # Experts of the flat `select_experts` rows (neither SM10x nor SM12x),
        # built after loading; unsupported formats fail here.
        self.flat_experts: FlatExperts | None = None
        self.flat_experts_cls = (
            None
            if self.sm100 or self.sm120
            else flat_experts_class(quant_format, self.topk, self.gate_cap, self.up_cap)
        )
        if self.sm100:
            load_kernels(
                (self.w13_weight, self.w2_weight),
                (self.w13_weight_scale, self.w2_weight_scale),
                self.topk,
            )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
        loader=None,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if loader is None:
            loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _refuse_reload(self, param: nn.Parameter) -> None:
        """Refuse a weight load into a layer whose flat backend already
        computes on its Parameters (`keeps_parameters`), before anything is
        written: they hold the backend's kernel layout (and NVFP4 global
        scales were taken at the build), so a reload, direct or layerwise
        (whose copy-back writes the loaded layout into them), would corrupt
        the experts."""
        if self.flat_experts is not None and self._flat_experts_keep_parameters():
            raise RuntimeError(
                f"Inkling2Experts: {type(self.flat_experts).__name__} computes on "
                "the routed experts' Parameters in its own layout; reloading "
                "their weights into a built layer is not supported "
                f"(parameter of shape {tuple(param.shape)})."
            )

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        if self.sm100:
            param.data.copy_(shuffle_rows(param.data))

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        if self.sm100:
            param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))

    def _load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._refuse_reload(param)
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def _flat_experts_keep_parameters(self) -> bool:
        """Whether the flat backend is built from the Parameters themselves
        (its `keeps_parameters`): it keeps them, rewritten in place into its
        kernel layout where it needs one, so they stay module state."""
        backend = self.flat_experts_cls
        # The XPU constructors are `functools.partial`s of the backend.
        backend = getattr(backend, "func", backend)
        return bool(getattr(backend, "keeps_parameters", False))

    def process_weights_after_loading(self) -> None:
        if self.sm100 or (self.sm120 and not self.humming_nvfp4):
            return
        # Only the backends that keep the Parameters in their kernel layout
        # (the XPU ones) refuse a second build; the CUDA paths are unchanged.
        if self.flat_experts is not None and self._flat_experts_keep_parameters():
            raise RuntimeError(
                "Inkling2Experts.process_weights_after_loading was already "
                "called: the expert kernels were built from the loaded "
                "weights, which now hold the kernel layout. Reloading the "
                "routed experts' weights into a built layer is not supported."
            )
        keep = not self.humming_nvfp4 and self._flat_experts_keep_parameters()
        weights = {}
        for name in ("w13", "w2"):
            amax, denominator = self.scales(name)
            global_scale = None
            if amax is not None and denominator is not None:
                global_scale = (
                    self.ones if self.humming_nvfp4 else amax.data / denominator.data
                )
            weight = getattr(self, f"{name}_weight")
            scale = getattr(self, f"{name}_weight_scale")
            if not keep:
                weight, scale = weight.data, scale.data
            weights[name] = (weight, scale, global_scale)
        if self.humming_nvfp4:
            self.humming = HummingNVFP4Experts(weights, self.shapes)
        else:
            assert self.flat_experts_cls is not None
            self.flat_experts = self.flat_experts_cls(weights, self.shapes, self.nvfp4)
        if keep:
            return
        # The repacked copies replace the checkpoint layout.
        for name in ("w13", "w2"):
            for suffix in ("weight", "weight_scale"):
                param = getattr(self, f"{name}_{suffix}")
                param.data = param.data.new_empty((0,))

    def plan(self, tokens: int):
        return expert_plan(tokens, self.topk, self.num_experts, self.nvfp4, self.shapes)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Outer scale amax and denominator of a projection's NVFP4 weights."""
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def _gemm(self, quantized, name: str, tiles, plan, projection: int):
        if self.humming_nvfp4:
            assert isinstance(self.humming, HummingNVFP4Experts)
            return self.humming.bmm(quantized, name, tiles, plan)
        act, act_scale, row_scale = quantized
        gemm = bmm_sm120 if self.sm120 else bmm
        return gemm(
            act,
            act_scale,
            row_scale,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            self.ones,
            tiles,
            plan,
            projection,
        )

    def forward(
        self,
        x: torch.Tensor,
        routing,
        quantized,
        plan,
        reduce: bool = True,
        push: SeqParallel | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """`routing` and `quantized` (`x` in the expert layout) are from
        `ops.moe.route` (on Hopper, `routing` is from
        `ops.moe_sm90.select_experts` and `quantized` is None). The routing
        weight is applied inside the SwiGLU, before the down projection,
        except by a `combines` flat backend, which applies it to the
        down-projection rows in fp32 while summing them.
        Without `reduce`, returns the local expert rows and `src2dst` for
        the consumer to combine and reduce, or pushes their combine to the
        token shard owners."""
        if self.sm100 or self.sm120:
            src2dst, gammas, rows, tiles = routing
            gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
            m = plan.max_ctas
            activated = swiglu_quantize(
                gate_up,
                gammas,
                self.gate_cap,
                self.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]),
                plan.tile,
                *self.scales("w2"),
            )
            del gate_up
            down = self._gemm(activated, "w2", tiles, plan, 1)
        else:
            assert self.flat_experts is not None, "process_weights_after_loading"
            ids, gammas, src2dst = routing
            down = self.flat_experts(
                x,
                ids,
                gammas,
                src2dst,
                self.num_experts * self.ep_size,
                self.gate_cap,
                self.up_cap,
            )
            if getattr(self.flat_experts, "combines", False):
                if push is not None or not reduce:
                    raise NotImplementedError(
                        f"{type(self.flat_experts).__name__} combines the expert "
                        "rows itself; it has no unreduced rows to return or push"
                    )
                # The fp32 combine was rounded once to bf16 by the experts.
                if self.ep_size == 1:
                    return down
                return tensor_model_parallel_all_reduce(down)
        if push is not None:
            # Each rank's fp32 combine is rounded once to bf16.
            combine_push(down, src2dst, self.topk, push)
            return None
        if not reduce:
            return down, src2dst
        # Each rank's fp32 combine is rounded once to bf16 before the all-reduce.
        out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)


class Inkling2RoutingCapture(nn.Module):
    """A routed-experts capture row (`RoutedExpertsCaptureSource`): with
    `enable_return_routed_experts`, vLLM binds `capture_fn` to record each
    token's expert ids at row `layer_id` of the capture buffer."""

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn: Callable[[torch.Tensor], None] | None = None


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, quant_format: QuantFormat, visit_ids: Sequence[int] = (0,)
    ) -> None:
        """`visit_ids`: the routed-experts capture row of each pass of the
        layer (`Inkling2ModelConfig.layer_visit_ids`)."""
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)
        self.routing_captures = nn.ModuleList(
            Inkling2RoutingCapture(visit) for visit in visit_ids
        )
        self.register_buffer(
            "all_experts",
            torch.tensor([0, config.n_routed_experts], dtype=torch.int32),
            persistent=False,
        )

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Each token's top-k expert ids `[T, topk]` over all experts: the
        selection of the routing kernels (`select_experts`, `route`)."""
        gate = self.gate
        ids, _, _ = select_experts(
            x, gate.weight, gate.bias, gate.topk, gate.route_scale, self.all_experts
        )
        return ids

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push: SeqParallel | None = None,
        lora: ExpertsLoRA | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """bf16 [T, H] -> bf16 [T, H], reduced over the expert-parallel group;
        without `reduce`, the unreduced `(down, src2dst)` of
        `Inkling2Experts`, or (with `push`) pushes this rank's partial sums
        to the token shard owners.
        Padding tokens (`is_padding` of the forward context) are routed to
        no expert and come out as 0. `lora` (`Inkling2MoEWithLoRA`) adds
        per-expert LoRA deltas inside the expert GEMMs, before any reduce or
        push; the NVFP4 / MXFP8 experts here do not support it. With
        routed-experts capture, the expert ids are recorded at the row of
        loop pass `pass_idx`."""
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(self.routed_expert_ids(x.contiguous()))
        if lora is not None:
            raise NotImplementedError(
                "Inkling2 expert LoRA needs BF16 experts; the NVFP4 / MXFP8 "
                "experts do not support it"
            )
        x = x.contiguous()
        gate, experts = self.gate, self.experts
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        if not (experts.sm100 or experts.sm120):
            selected = select_experts(
                x,
                gate.weight,
                gate.bias,
                gate.topk,
                gate.route_scale,
                experts.expert_bounds,
                is_padding,
            )
            return experts(x, selected, None, None, reduce, push)
        plan = experts.plan(x.shape[0])
        routing, quantized = route(
            x,
            gate.weight,
            gate.bias,
            gate.counters,
            experts.counts,
            gate.topk,
            gate.route_scale,
            experts.expert_bounds,
            plan,
            *experts.scales("w13"),
            is_padding,
        )
        return experts(x, routing, quantized, plan, reduce, push)
