# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash MoE layer of up to 32 decode tokens on SM90 as two kernels:
the fused router (`moe_router`: gate GEMM, top-8, FP8 input) and the routed
plus shared experts (`moe_experts`), launched as its programmatic dependent.

`HopperMoe` replaces the whole stock `Glm5NextMoE.forward` of a TP4 rank (gate
GEMM, grouped top-k, the shared expert on the aux stream, input quant, the
Triton FP8 routed experts and the final add) and returns the BF16 MoE output,
routing-weighted routes plus the shared expert, still TP-partial: unlike the
GB300 kernels it finalizes the top-k reduction itself (in a fixed order, so the
output is deterministic), and the caller all-reduces it like any other
sublayer output.

It reads the stock Hopper weights in place: the routed experts' FP8 w13
`[288, 1024, 4096]` (gate rows, then up rows) and w2 `[288, 4096, 512]` with FP32
128 x 128 block scales, the shared expert's FP8 gate_up and down linears with
their block scales, the BF16 gate weight and the FP32 score correction bias.
Activations are quantized per 128 (input) and per 64 (hidden) with amax / 448
scales. Only these kernels read the FP8 activations, so the scales need not be
powers of two as DeepGEMM's E8M0 path wants them: a power-of-two scale leaves up
to a bit of the FP8 range unused, which shows on the massive-activation columns
of the shared expert (on the last layer's real inputs one such column lands
9% low with power-of-two hidden scales, 1% with exact ones).

Every layer on a device shares one workspace: calls are ordered on the
forward stream, and the experts kernel resets its counters when it ends, so
CUDA graph replays need no reset.

The TMA descriptors are built once from the weights' addresses (packed UE8M0
scales are converted to FP32 then), so weights replaced by new tensors after
the layer is built are not seen.
"""

from typing import NamedTuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HIDDEN = 4096
INTER = 512
EXPERTS = 288
TOP_K = 8
SWIGLU_LIMIT = 10.0
# Up to 4 token groups of 8 (the MMA N).
MAX_TOKENS = 32


class _Workspace:
    """The buffers of one call, shared by every layer on a device."""

    def __init__(self, device: torch.device):
        from .moe_experts import SLOTS

        def zeros(*shape, dtype=torch.uint8):
            return torch.zeros(*shape, dtype=dtype, device=device)

        f32, i32 = torch.float32, torch.int32
        groups = MAX_TOKENS // 8
        # Router outputs.
        self.logits = zeros(MAX_TOKENS, EXPERTS, dtype=f32)
        self.ids = zeros(MAX_TOKENS, TOP_K, dtype=i32)
        self.weights = zeros(MAX_TOKENS, TOP_K, dtype=f32)
        # The FP8 input, [token / 8][128-byte K chunk][token % 8][16 bytes].
        self.xq = zeros(groups, HIDDEN // 16, 8, 16)
        self.xs = zeros(MAX_TOKENS, HIDDEN // 128, dtype=f32)
        self.router_counter = zeros(64, dtype=i32)
        # The FP8 hidden of each slot, [slot][group][128-byte K chunk][8][16]
        # (at the call's group count), and its scales [slot][block][rows].
        self.iq = zeros(SLOTS * groups * (INTER // 16) * 8 * 16)
        self.isc = zeros(SLOTS * (INTER // 64) * MAX_TOKENS, dtype=f32)
        # One FP32 row per (token, k): 8 routed slots and the shared expert.
        self.slots = zeros(MAX_TOKENS, TOP_K + 1, HIDDEN, dtype=f32)
        self.counters = zeros(256, dtype=i32)
        self.sync = zeros(SLOTS + 4, dtype=i32)
        self.grid = 2 * torch.cuda.get_device_properties(device).multi_processor_count

    def maps(self, groups: int):
        """TMA specs of the FP8 input and hidden at `groups` token groups."""
        import cuda.bindings.driver as cuda

        from .moe_experts import SLOTS

        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        none = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        chunks = INTER // 16
        return [
            (self.xq.data_ptr(), u8, [128, HIDDEN // 16, groups],
             [128, 128 * (HIDDEN // 16)], [128, 8, groups], none, l2),
            (self.iq.data_ptr(), u8, [128, chunks, groups, SLOTS],
             [128, 128 * chunks, 128 * chunks * groups], [128, 8, groups, 1], none,
             l2),
        ]  # fmt: skip


_workspaces: dict[int, _Workspace] = {}


def _workspace(device: torch.device) -> _Workspace:
    index = torch.device(device).index
    if index not in _workspaces:
        _workspaces[index] = _Workspace(device)
    return _workspaces[index]


def _fp32_scales(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 `[N / 128, K / 128]` block scales of a linear, contiguous."""
    if scale.dtype == torch.float32:
        return scale.contiguous()
    from ..shared_expert import block_scales

    return block_scales(scale, *weight.shape)


def _weight_specs(w13, w2, shared13, shared2):
    """TMA specs of the routed and shared expert weights (128B swizzle)."""
    import cuda.bindings.driver as cuda

    u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B

    def spec(t, rows, k):
        return (t.data_ptr(), u8, [k, rows, t.numel() // (rows * k)],
                [k, k * rows], [128, 64, 1], swizzle, l2)  # fmt: skip

    return (
        spec(w13, 2 * INTER, HIDDEN),
        spec(w2, HIDDEN, INTER),
        spec(shared13, 2 * INTER, HIDDEN),
        spec(shared2, HIDDEN, INTER),
    )


def unsupported(moe, config) -> list[str]:
    """Every reason the stock `Glm5NextMoE` layer `moe` (with `config`, the
    text config) cannot run as `HopperMoe`."""
    from vllm.platforms import current_platform

    reasons = []
    expected = {
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "hidden_size": HIDDEN,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "swiglu_limit": SWIGLU_LIMIT,
        "hidden_act": "silu",
    }
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            reasons.append(f"{name}={getattr(config, name, None)} (expects {value})")
    if not current_platform.is_device_capability(90):
        reasons.append("a GPU other than SM90")
    if getattr(moe, "use_mega_moe", False) or getattr(
        moe, "is_fused_shared_expert_enabled", False
    ):
        reasons.append("MegaMoE or fused shared experts")
    if getattr(moe, "is_sequence_parallel", False):
        reasons.append("sequence-parallel MoE")
    shared = getattr(moe, "shared_experts", None)
    routed = getattr(getattr(moe, "experts", None), "routed_experts", None)
    if shared is None or routed is None:
        return reasons + ["no separate shared expert or routed experts"]
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

    # Other backends may reorder w13 (FlashInfer swaps it to [up; gate]) or
    # requantize it without changing its shape.
    backend = getattr(getattr(routed, "quant_method", None), "fp8_backend", None)
    if backend != Fp8MoeBackend.TRITON:
        backend_name = getattr(backend, "value", backend)
        reasons.append(f"FP8 MoE backend {backend_name} (expects TRITON)")
    fp8 = torch.float8_e4m3fn
    tensors = {
        "w13_weight": (routed.w13_weight, (EXPERTS, 2 * INTER, HIDDEN), fp8),
        "w2_weight": (routed.w2_weight, (EXPERTS, HIDDEN, INTER), fp8),
        "w13_weight_scale_inv": (
            getattr(routed, "w13_weight_scale_inv", None),
            (EXPERTS, 2 * INTER // 128, HIDDEN // 128),
            torch.float32,
        ),
        "w2_weight_scale_inv": (
            getattr(routed, "w2_weight_scale_inv", None),
            (EXPERTS, HIDDEN // 128, INTER // 128),
            torch.float32,
        ),
        "shared gate_up": (shared.gate_up_proj.weight, (2 * INTER, HIDDEN), fp8),
        "shared down": (shared.down_proj.weight, (HIDDEN, INTER), fp8),
        "gate": (moe.gate.weight, (EXPERTS, HIDDEN), torch.bfloat16),
    }
    for name, (t, shape, dtype) in tensors.items():
        if t is None or tuple(t.shape) != shape or t.dtype != dtype:
            got = None if t is None else (tuple(t.shape), t.dtype)
            reasons.append(f"{name} {got} (expects {shape}, {dtype})")
        elif not t.is_contiguous():
            reasons.append(f"non-contiguous {name}")
    for linear in (shared.gate_up_proj, shared.down_proj):
        if getattr(linear, "weight_scale_inv", None) is None:
            reasons.append("shared expert without block scales")
    bias = moe.gate.e_score_correction_bias
    if bias is None or tuple(bias.shape) != (EXPERTS,):
        reasons.append("no score correction bias")
    if moe.gate.out_dtype != torch.float32:
        reasons.append("router logits other than FP32")
    return reasons


class MoeWeights(NamedTuple):
    """One layer's tensors as the stock Hopper layer holds them."""

    gate: torch.Tensor  # BF16 [288, 4096]
    bias: torch.Tensor  # FP32 [288] score correction bias
    w13: torch.Tensor  # FP8 [288, 1024, 4096], gate rows then up rows
    w13_scale: torch.Tensor  # FP32 [288, 8, 32]
    w2: torch.Tensor  # FP8 [288, 4096, 512]
    w2_scale: torch.Tensor  # FP32 [288, 32, 4]
    shared13: torch.Tensor  # FP8 [1024, 4096]
    shared13_scale: torch.Tensor  # FP32 [8, 32]
    shared2: torch.Tensor  # FP8 [4096, 512]
    shared2_scale: torch.Tensor  # FP32 [32, 4]


class HopperMoe:
    """One `Glm5NextMoE` layer's decode of up to `MAX_TOKENS` tokens."""

    MAX_TOKENS = MAX_TOKENS

    def __init__(self, weights: MoeWeights, scale: float, ue8m0: bool):
        """`scale`: the routed scaling factor; `ue8m0`: power-of-two activation
        scales."""
        from ..cutedsl.host import tensor_maps

        w = weights
        assert all(t.is_contiguous() for t in (w.w13, w.w2, w.shared13, w.shared2))
        self.device = w.w13.device
        self.workspace = _workspace(self.device)
        self.scale, self.ue8m0 = float(scale), ue8m0
        self.gate = w.gate.contiguous()
        self.bias = w.bias.float().contiguous()
        self.scales = tuple(
            t.float().contiguous()
            for t in (w.w13_scale, w.w2_scale, w.shared13_scale, w.shared2_scale)
        )
        weight_specs = _weight_specs(w.w13, w.w2, w.shared13, w.shared2)
        # Per token group count: routed w13, w2, the FP8 input and hidden, the
        # shared w13 and w2 (the order of moe_experts.MAP_*).
        self.maps = {}
        for groups in range(1, MAX_TOKENS // 8 + 1):
            xq, iq = self.workspace.maps(groups)
            specs = [*weight_specs[:2], xq, iq, *weight_specs[2:]]
            self.maps[groups] = tensor_maps(specs, self.device)
        # Keep the tensors the maps address alive with the layer's.
        self._weights = (w.w13, w.w2, w.shared13, w.shared2)

    @classmethod
    def of(cls, moe, config) -> "HopperMoe | None":
        """The `HopperMoe` of a stock `Glm5NextMoE` layer after its weights are
        processed (`config`: the text config), or None (logging why) for any
        other layout."""
        reasons = unsupported(moe, config)
        if reasons:
            logger.warning_once(
                "GLM-5.3-Flash Hopper MoE disabled: %s", "; ".join(reasons)
            )
            return None
        routed = moe.experts.routed_experts
        gate_up, down = moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj
        weights = MoeWeights(
            gate=moe.gate.weight,
            bias=moe.gate.e_score_correction_bias.data,
            w13=routed.w13_weight,
            w13_scale=routed.w13_weight_scale_inv,
            w2=routed.w2_weight,
            w2_scale=routed.w2_weight_scale_inv,
            shared13=gate_up.weight,
            shared13_scale=_fp32_scales(gate_up.weight_scale_inv, gate_up.weight),
            shared2=down.weight,
            shared2_scale=_fp32_scales(down.weight_scale_inv, down.weight),
        )
        return cls(weights, moe.routed_scaling_factor, ue8m0=False)

    def supports(self, tokens: int) -> bool:
        return 0 < tokens <= MAX_TOKENS

    def _kernels(self, groups: int):
        from . import moe_experts, moe_router

        return (
            moe_router.compiled(groups, self.scale, self.ue8m0),
            moe_experts.compiled(groups, self.ue8m0, self.device.index),
        )

    def prepare(self, tokens: list[int]) -> None:
        """Compiles and runs once for these token counts before graph capture."""
        for m in sorted(set(tokens)):
            if self.supports(m):
                x = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
                self(x)
        torch.accelerator.synchronize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The TP-partial BF16 `[M, 4096]` MoE output of the normed BF16 tokens
        `x` `[M, 4096]`: the routed experts weighted (routing weights times the
        routed scaling factor) plus the shared expert."""
        from cutlass import Int32, Int64

        from ..cutedsl.host import stream

        m = x.shape[0]
        assert self.supports(m), m
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        assert x.shape[1] == HIDDEN
        groups = (m + 7) // 8
        ws = self.workspace
        router, experts = self._kernels(groups)
        out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=x.device)
        st = stream(x.device)

        def address(t: torch.Tensor) -> Int64:
            return Int64(t.data_ptr())

        router(
            address(x), address(self.gate), address(self.bias), address(ws.logits),
            address(ws.ids), address(ws.weights), address(ws.xq), address(ws.xs),
            address(ws.router_counter), Int32(m), st,
        )  # fmt: skip
        experts(
            address(self.maps[groups]), *(address(t) for t in self.scales),
            address(ws.xs), address(ws.ids), address(ws.weights), address(ws.iq),
            address(ws.isc), address(ws.slots), address(out), address(ws.counters),
            address(ws.sync), Int32(m), Int32(ws.grid), st,
        )  # fmt: skip
        return out

    def routing(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The last call's int32 `[m, 8]` ids and FP32 `[m, 8]` weights (views of
        the workspace, for tests)."""
        return self.workspace.ids[:m], self.workspace.weights[:m]
