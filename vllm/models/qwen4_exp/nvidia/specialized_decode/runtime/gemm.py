# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode GEMM plans of the specialized decode.

The executor runs the stock layers' BF16 projections (the GDN input and output
projections, QSA's fused input projection and output projection, the MoE
router and the shared expert's projections). Stock picks each GEMM from the
GPU's plan table (`QWEN4_EXP_SM103_GEMM_PLANS` on GB300, in
`nvidia/low_latency_gemm.py`): a skinny CuTe GEMM at the shapes and row
counts it lists, cuBLAS (`F.linear`) otherwise. That table covers few of the
row counts a speculative decode step has (four rows a request with three draft
tokens) and none of the router's, so most of these GEMMs run cuBLAS, often
split-K with a separate reduction kernel.

The specialized decode has its own table, `QWEN4_EXP_DECODE_GEMM_PLANS`, of
skinny GEMM configurations measured at its row counts, which only the
specialized decode reads: the stock path, and every forward the executor does
not take, keeps the stock table. This module also holds the MTP draft front's
plans for its two projections (`QWEN4_EXP_DRAFT_FRONT_GEMM_PLANS`, read only
by `draft_front.py`).

`PlannedGemms` finds the layers whose local `(N, K)` the table lists among
the layers it is built over: the target's decoder layers for the executor
(`SpecializedDecode`), the MTP draft model's layers for the draft passes
that take the fused front (`DraftFront`). Within `PlannedGemms.active`,
each of those layers' quant method is a `PlannedLinearMethod`: the planned
skinny GEMM for the row counts its plan lists, the layer's own method for any
other call (row counts without a plan, a bias, an activation layout the
kernel does not take). Outside it the layers are as built. The executor and
the draft front enter it only while they capture a FULL CUDA graph of a pure
decode step, so the plans are baked into those graphs and nothing else
changes. The layer's own method is the one it has when the block is entered,
so a method installed after `build` (the stock table's, which the MTP draft
model installs after building its layers) is the one other calls take.

A planned GEMM launches with programmatic dependent launch (PDL): before its
`griddepcontrol.wait` it only loads weight tiles and writes nothing, so the
kernel before it may trigger it early, and it writes its output before it
triggers the kernel after it. The swap changes the layers'
`quant_method` in place, so it assumes one thread runs the forwards and
nothing traces them: the deployment check rejects micro-batching and
`torch.compile` of the model.

The skinny GEMM accumulates in FP32 like cuBLAS but sums K in another order,
so its outputs are not bitwise equal to cuBLAS's: they differ by BF16
rounding (the table's comment states the bound).

Warm-up. `PlannedGemms.warm_up` compiles and runs every planned
configuration once; the executor and the draft front call it when they warm
up their own kernels, before memory profiling and every CUDA graph capture,
so no capture compiles a GEMM.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

# The stock table's operand check, so a plan runs exactly where a stock plan
# of the same configuration would.
from ...low_latency_gemm import _runtime_ok

# The MTP draft front's 2560 x 2560 projections (fc_embedding and fc_hidden on
# full weights at every TP size) in the specialized decode, by row count. Only
# the draft front (``draft_front.py``) reads these plans, so the stock layers
# keep their GEMMs; other row counts use the standard linear implementation.
# GB300 (CUDA graph replay over 16 weight copies): 3.1 against 5.7 us at 1 row
# and 4.0 against 5.7 us at 4 rows; at 8 rows the best plan gains under 5 % and
# at 16 rows it loses.
QWEN4_EXP_DRAFT_FRONT_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    (10, 3): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, static_k=2560),
    },
}


def draft_front_gemm_plans() -> dict[int, SkinnyGemmConfig]:
    """The draft front's GEMM plans on this device, by row count."""
    if not shape_dynamic_skinny_gemm.is_available():
        return {}
    capability = current_platform.get_device_capability()
    if capability is None:
        return {}
    return dict(QWEN4_EXP_DRAFT_FRONT_GEMM_PLANS.get(capability, {}))


def draft_front_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    plans: dict[int, SkinnyGemmConfig],
) -> torch.Tensor:
    """`x @ weight.T` with the planned skinny GEMM for `x`'s row count,
    else the standard linear implementation.

    Args:
        x: BF16 `[rows, k]` activations.
        weight: Contiguous BF16 `[n, k]` weight.
        plans: `draft_front_gemm_plans()`.

    """
    config = plans.get(x.shape[0])
    if config is not None and _runtime_ok(x, weight, config):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


# The specialized decode's plans for the stock layers' projections in its
# FULL-graph decode steps and MTP draft passes (``PlannedGemms`` below), by
# device, local (N, K) and row count. Only the specialized decode reads them, so
# the stock layers keep the stock tables (``low_latency_gemm.py``); other row
# counts take the layer's own method. A plan is listed where it beats that
# method (the stock table's plan or the standard linear implementation) by at
# least 0.25 us and 5 % in CUDA graph replay over weight copies larger than L2,
# both launched back to back and each after a producer kernel (the gains below
# are the smaller of the two, in us per call), and where the decode step agrees:
# plans at 8 and 16 rows that won the replay made the TP4 step of 4 requests
# 2.3 % slower and the TP1 step of 4 requests no faster, so only 1 and 4 rows
# are listed. Outputs are within BF16 rounding of the standard
# implementation's: the largest difference measured was 0.4 % of the largest
# output; the tests bound it at 1 % of the largest output, and the difference
# from an FP32 reference at 2**-7 of it (one BF16 unit in the last place).
QWEN4_EXP_DECODE_GEMM_PLANS: dict[
    tuple[int, int], dict[tuple[int, int], dict[int, SkinnyGemmConfig]]
] = {
    (10, 3): {
        # GDN fused QKVZ projection, TP=4: 5.3 against 7.2 us at 4 rows.
        (4096, 2560): {
            4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=2560),
        },
        # GDN fused B/A projection, TP=4: 0.6 us under the stock table.
        (24, 2560): {
            4: SkinnyGemmConfig(4, 64, 1, vector_width=4, static_k=2560),
        },
        # QSA output projection of the draft passes, TP=4: 0.3 us under the
        # stock table at 1 row.
        (2560, 1536): {
            1: SkinnyGemmConfig(1, 32, 4, vector_width=4, static_k=1536),
        },
        # MoE router (replicated): 1.8 and 2.0 against 4.4 and 4.6 us at 1
        # and 4 rows.
        (512, 2560): {
            1: SkinnyGemmConfig(1, 64, 1, vector_width=4, static_k=2560),
            4: SkinnyGemmConfig(4, 64, 1, vector_width=4, static_k=2560),
        },
        # Shared-expert gate (replicated): 0.9 and 1.1 against 2.9 us.
        (1, 2560): {
            1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=2560),
            4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),
        },
        # GDN and QSA output projections, TP=1: 6.8 against 8.2 us at 4 rows.
        (2560, 6144): {
            4: SkinnyGemmConfig(4, 64, 4, static_k=6144),
        },
        # QSA fused QKV/gate + replicated indexer Q/K, TP=1: 13.4 against
        # 14.4 us at 4 rows.
        (13952, 2560): {
            4: SkinnyGemmConfig(4, 32, 4, static_k=2560),
        },
        # Shared-expert down projection, TP=1: 2.6 against 3.5 us at 4 rows.
        (2560, 640): {
            4: SkinnyGemmConfig(4, 32, 4, vector_width=4, static_k=640),
        },
    },
}


def decode_gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    """The specialized decode's GEMM plans on this device, by local
    `(N, K)` and row count."""
    if not shape_dynamic_skinny_gemm.is_available():
        return {}
    capability = current_platform.get_device_capability()
    if capability is None:
        return {}
    return {
        shape: dict(plan)
        for shape, plan in QWEN4_EXP_DECODE_GEMM_PLANS.get(capability, {}).items()
    }


def planned_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    plans: dict[int, SkinnyGemmConfig],
) -> torch.Tensor | None:
    """`x @ weight.T` with the planned skinny GEMM for `x`'s row count, or
    None when `plans` has none for it or the kernel does not take the
    operands.

    Args:
        x: BF16 `[rows, k]` activations.
        weight: BF16 `[n, k]` weight.
        plans: Skinny GEMM configurations by row count.

    """
    config = plans.get(x.shape[0]) if x.dim() == 2 else None
    if config is None or not _runtime_ok(x, weight, config):
        return None
    return shape_dynamic_skinny_gemm(x, weight, config)


class PlannedLinearMethod:
    """A layer's quant method within `PlannedGemms.active`: the planned
    skinny GEMM for the row counts of `plans`, `stock` otherwise.

    Args:
        stock: The layer's own (unquantized) linear method.
        plans: The layer's skinny GEMM configurations, by row count.

    """

    def __init__(
        self, stock: UnquantizedLinearMethod, plans: dict[int, SkinnyGemmConfig]
    ) -> None:
        self.stock = stock
        self.plans = plans

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`x @ layer.weight.T (+ bias)`: the planned GEMM when one applies,
        the stock method otherwise."""
        if bias is None:
            output = planned_gemm(x, layer.weight, self.plans)
            if output is not None:
                return output
        return self.stock.apply(layer, x, bias)


class PlannedGemms:
    """The planned layers among some modules, and their plans.

    Args:
        sites: Each planned layer with its skinny GEMM configurations, by row
            count.

    """

    def __init__(
        self, sites: list[tuple[nn.Module, dict[int, SkinnyGemmConfig]]]
    ) -> None:
        self.sites = sites

    @classmethod
    def build(cls, modules: Iterable[nn.Module]) -> "PlannedGemms":
        """The unquantized BF16 linear layers in `modules` whose local
        `(N, K)` has a plan on this device (none off GB300)."""
        plans = decode_gemm_plans()
        sites: list[tuple[nn.Module, dict[int, SkinnyGemmConfig]]] = []
        seen: set[int] = set()
        for module in modules:
            for child in module.modules():
                if id(child) in seen or not isinstance(child, LinearBase):
                    continue
                seen.add(id(child))
                method = child.quant_method
                weight = getattr(child, "weight", None)
                if (
                    not isinstance(method, UnquantizedLinearMethod)
                    or not isinstance(weight, torch.Tensor)
                    or weight.dim() != 2
                    or weight.dtype != torch.bfloat16
                ):
                    continue
                plan = plans.get((weight.shape[0], weight.shape[1]))
                if plan:
                    sites.append((child, plan))
        return cls(sites)

    @contextmanager
    def active(self) -> Iterator[None]:
        """Within the block, the planned layers take their plans."""
        saved = [(layer, layer.quant_method) for layer, _ in self.sites]
        try:
            for (layer, plans), (_, method) in zip(self.sites, saved):
                # A layer whose method changed to a quantized one since build
                # keeps it.
                if isinstance(method, UnquantizedLinearMethod):
                    layer.quant_method = PlannedLinearMethod(method, plans)
            yield
        finally:
            for layer, method in saved:
                layer.quant_method = method

    def configs(self) -> list[tuple[SkinnyGemmConfig, torch.Tensor]]:
        """Each distinct planned configuration with one weight it serves."""
        found: dict[SkinnyGemmConfig, torch.Tensor] = {}
        for layer, plans in self.sites:
            for config in plans.values():
                found.setdefault(config, layer.weight)
        return list(found.items())

    def warm_up(self) -> list[SkinnyGemmConfig]:
        """Compile and run every planned configuration once; returns them."""
        configs = self.configs()
        with torch.inference_mode():
            for config, weight in configs:
                x = torch.zeros(
                    config.num_rows,
                    weight.shape[1],
                    dtype=weight.dtype,
                    device=weight.device,
                )
                output = planned_gemm(x, weight, {config.num_rows: config})
                assert output is not None, f"{config} does not take its weight"
        if configs:
            torch.accelerator.synchronize(configs[0][1].device)
        return [config for config, _ in configs]
