# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch MoE of the Hopper executor.

``HopperLowBatchMoE`` runs one DeepSeek-V4.1 MoE layer's routed experts, and up
to ``SHARED_MAX_ROWS`` rows its shared expert, as one persistent CuTe DSL launch
(``moe_cute.LowBatchMoEKernel``) for up to ``MAX_ROWS`` rows. It reads the
layer's stock Humming tensors in place: the routed experts' SM90 W4A8 weights
(MXFP4 x block-FP8 group-128 activations) and the shared expert's W8A16 linears.

The kernel routes the step itself from the stock router's FP32 logits
(sqrt-softplus plus the correction bias, top-k, renormalized, times the routed
scaling factor; image sentinel tokens with ``bias_vl``), or takes the stock
router's top-k ids and weights on hash-routed layers. It stops before the top-k
reduction: its output is an ``UnfinalizedMoEOutput`` (unweighted BF16 rows per
(token, expert), FP32 routing weights, int32 route-to-row map) that the next
boundary's fused all-reduce + mHC finalizes (``AllReduceMHC.finalize``), with
the shared expert's BF16 output (fused, or from the stock shared expert).

The expert count and top-k are the layer's (384 / 6 in the backbone, 128 / 3
in the DSpark drafter). Every layer shares one ``MoEWorkspace``: calls are
ordered on the forward stream, and each boundary consumes its MoE output
before the next MoE layer runs. The kernel's counters are epoch-stamped or
reset by their last user, so CUDA graph replays need no reset.
"""

from functools import cache
from typing import NamedTuple

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

# Rows of the largest call: at 144 rows the kernel with its fused shared
# expert ties the stock MoE with the shared expert on the side stream (the
# kernel holds 256).
MAX_ROWS = 144
# Calls of up to this many rows give each active expert one chunk of all rows,
# and split GEMM1's K over CTA pairs that reduce through distributed shared
# memory (CLUSTER_MAX_ROWS).
DENSE_MAX_ROWS = 16
CLUSTER_MAX_ROWS = 16
# The fused shared expert beats or ties the stock one on its side stream up to
# here.
SHARED_MAX_ROWS = 144
_HIDDEN = 5120


def _routed_shapes(experts: int) -> dict[str, tuple[int, ...]]:
    return {
        "w13_weight": (experts, 80, 10240),
        "w13_weight_scale": (experts, 160, 1280),
        "w13_weight_scale_2": (experts,),
        "w2_weight": (experts, 10, 40960),
        "w2_weight_scale": (experts, 20, 5120),
        "w2_weight_scale_2": (experts,),
    }


def _tensor_map(dtype, address: int, dims, strides, box) -> list[int]:
    import cuda.bindings.driver as cuda

    err, tensor_map = cuda.cuTensorMapEncodeTiled(
        dtype,
        len(dims),
        address,
        [cuda.cuuint64_t(d) for d in dims],
        [cuda.cuuint64_t(s) for s in strides],
        [cuda.cuuint32_t(b) for b in box],
        [cuda.cuuint32_t(1)] * len(dims),
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    return [v if v < 2**63 else v - 2**64 for v in map(int, tensor_map.opaque)]


def _layer_maps(
    routed: dict[str, torch.Tensor],
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    bias_vl: torch.Tensor | None,
    sentinel_lo: int,
) -> torch.Tensor:
    """The kernel's [9, 16] int64 layer blob: TMA tensor maps of the routed
    w13 / s13 / w2 / s2 and shared gate-up / its scales / down / its scales;
    row 8: (address, bytes) of the four shared tensors, then the gate weight
    (unused), ``bias_vl`` addresses and the image sentinel."""
    import cuda.bindings.driver as cuda

    u64 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64
    u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    e = routed["w13_weight"].shape[0]
    w13, s13 = routed["w13_weight"], routed["w13_weight_scale"].view(torch.uint8)
    w2, s2 = routed["w2_weight"], routed["w2_weight_scale"].view(torch.uint8)
    blob = torch.zeros(9, 16, dtype=torch.int64)
    maps = [
        # w13 [e * 80 + k64][half][pair][2 KB]
        (u64, w13, [256, 9, 2, e * 80], [2048, 18432, 40960], [256, 1, 2, 4]),
        # s13 [e * 160 + k32][half][pair][64 B]
        (u8, s13, [64, 9, 2, e * 160], [64, 576, 1280], [64, 1, 2, 8]),
        # w2 [e * 10 + k64][block][2 KB]
        (u64, w2, [256, 80, e * 10], [2048, 163840], [256, 2, 3]),
        # s2 [e * 20 + k32][block pair][128 B]
        (u8, s2, [128, 40, e * 20], [128, 5120], [128, 1, 6]),
    ]
    if shared is not None:
        gate_up, gate_up_scale, down, down_scale = shared
        maps += [
            (u64, gate_up, [128, 9, 2, 320], [1024, 9216, 20480], [128, 1, 2, 8]),
            (u8, gate_up_scale, [64, 9, 2, 160], [64, 576, 1280], [64, 1, 2, 4]),
            (u64, down, [128, 80, 40], [1024, 81920], [128, 2, 6]),
            (u8, down_scale, [5120, 20], [5120], [128, 3]),
        ]
        # The shared tensors' (address, bytes) for the kernel's L2 prefetch.
        for i, t in enumerate(shared):
            blob[8, 2 * i] = t.data_ptr()
            blob[8, 2 * i + 1] = t.numel() * t.element_size()
    for i, (dtype, t, dims, strides, box) in enumerate(maps):
        blob[i] = torch.tensor(_tensor_map(dtype, t.data_ptr(), dims, strides, box))
    blob[8, 9] = 0 if bias_vl is None else bias_vl.data_ptr()
    blob[8, 10] = sentinel_lo
    return blob.to(w13.device)


class MoEWorkspace:
    """The buffers of one MoE call, shared by every layer on a device."""

    def __init__(self, device: torch.device) -> None:
        from .moe_cute import (
            CHUNK_CAP,
            ENTP,
            KT1,
            MAX_TOPK,
            MAXM,
            NG,
            REC_POS,
            REC_WORDS,
            SYNC_WORDS,
        )

        def zeros(*shape, dtype=torch.uint8):
            return torch.zeros(*shape, dtype=dtype, device=device)

        f32, i32 = torch.float32, torch.int32
        self.xg = zeros(KT1 // 4, NG, 4, 4, 8, 16)
        self.xsg = zeros(KT1 // 4, ENTP, 2, dtype=f32)
        self.rec = zeros(REC_WORDS, dtype=i32)
        self.iq = zeros(3, NG, 3, 4, 8, 16)
        self.isc = zeros(3, ENTP, 3, dtype=f32)
        self.part = zeros(CHUNK_CAP, 32, 128, dtype=f32)
        self.cnt = zeros(32, dtype=i32)
        self.sync = zeros(SYNC_WORDS, dtype=i32)
        self.rows = zeros(ENTP, _HIDDEN, dtype=torch.bfloat16)
        self.out = zeros(MAXM, _HIDDEN, dtype=torch.bfloat16)
        self.ids = zeros(MAXM * MAX_TOPK, dtype=i32)
        self.wts = zeros(MAXM * MAX_TOPK, dtype=f32)
        self.xgs = zeros(40, MAXM // 8, 8, 8, 16)
        self.xsgs = zeros(40, MAXM, dtype=f32)
        self.iqs = zeros(6, MAXM // 8, 12, 8, 16)
        self.iscs = zeros(6, MAXM, 3, dtype=f32)
        self.shared = zeros(MAXM, _HIDDEN, dtype=torch.bfloat16)
        self._rec_pos = REC_POS

    def route_to_row(self, rows: int, top_k: int) -> torch.Tensor:
        """``expanded_idx_to_permuted_idx`` [rows, top_k] (a view of the
        record)."""
        start = self._rec_pos
        return self.rec[start : start + rows * top_k].view(rows, top_k)


_workspaces: dict[torch.device, MoEWorkspace] = {}


def workspace(device: torch.device) -> MoEWorkspace:
    """The device's workspace, allocated on first use (outside graph capture)."""
    if device not in _workspaces:
        _workspaces[device] = MoEWorkspace(device)
    return _workspaces[device]


class LaunchConfig(NamedTuple):
    """A compiled variant of the kernel."""

    experts: int
    top_k: int
    ids_mode: bool
    dense: bool
    shared: bool
    image: bool
    # Split GEMM1's K over CTA pairs (dense calls only).
    cluster: bool
    # K chunks of the fused shared expert's GEMM1.
    shared_chunks: int
    # GEMM2 ring stages.
    gemm2_stages: int


def launch_config(
    experts: int, top_k: int, rows: int, ids_mode: bool, shared: bool, image: bool
) -> LaunchConfig:
    dense = rows <= DENSE_MAX_ROWS
    cluster = dense and rows <= CLUSTER_MAX_ROWS
    return LaunchConfig(
        experts=experts,
        top_k=top_k,
        ids_mode=ids_mode,
        dense=dense,
        shared=shared,
        image=image,
        cluster=cluster,
        shared_chunks=10 if cluster else 20,
        # Above 8 rows the cluster's partial slot takes the fifth stage.
        gemm2_stages=4 if cluster and rows > 8 else 5,
    )


@cache
def _compiled(config: LaunchConfig):
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .moe_cute import LowBatchMoEKernel

    kernel = LowBatchMoEKernel(
        ns2=config.gemm2_stages,
        ids_mode=config.ids_mode,
        dense=config.dense,
        shared=config.shared,
        image=config.image,
        cluster=config.cluster,
        nchs=config.shared_chunks,
        experts=config.experts,
        top_k=config.top_k,
    )
    zero = Int64(0)
    return cute.compile(kernel, *([zero] * 23), Int32(1), Int32(1), make_fake_stream())


class HopperLowBatchMoE:
    """The low-batch MoE call of one ``DeepseekV4MoE`` layer."""

    def __init__(
        self,
        routed: dict[str, torch.Tensor],
        shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        bias: torch.Tensor | None,
        bias_vl: torch.Tensor | None,
        sentinel_lo: int,
        hash_routed: bool,
        top_k: int,
    ) -> None:
        self.device = routed["w13_weight"].device
        self.experts = routed["w13_weight"].shape[0]
        self.top_k = top_k
        self.workspace = workspace(self.device)
        self.blob = _layer_maps(routed, shared, bias_vl, sentinel_lo)
        self.scale13 = routed["w13_weight_scale_2"]
        self.scale2 = routed["w2_weight_scale_2"]
        self.has_shared = shared is not None
        # Keep the tensors the blob addresses alive with the layer's.
        self._tensors = (routed, shared, bias_vl)
        self.bias = (
            bias.float().contiguous()
            if bias is not None
            else torch.zeros(self.experts, dtype=torch.float32, device=self.device)
        )
        self.image = bias_vl is not None and sentinel_lo > 0
        self.hash_routed = hash_routed
        self._grid = (
            2 * torch.cuda.get_device_properties(self.device).multi_processor_count
        )

    @classmethod
    def of(cls, ffn: nn.Module) -> "HopperLowBatchMoE | None":
        """The call of a stock ``DeepseekV4MoE`` on Humming W4A8 experts at TP4
        (a multiple of 32 experts up to 384, top-k up to 6, sqrt-softplus,
        renormalized, routed scaling 1.5, SwiGLU limit 10), or None for any
        other layer."""
        from .moe_cute import MAX_EXPERTS, MAX_TOPK, ROUTED_SCALING, SWIGLU_LIMIT

        runner = getattr(ffn, "experts", None)
        experts = getattr(runner, "routed_experts", None)
        if experts is None or getattr(ffn, "use_native_mega_moe", False):
            return None
        params = dict(experts.named_parameters())
        num_experts = getattr(params.get("w13_weight"), "shape", (0,))[0]
        top_k = ffn.n_activated_experts
        if not (
            0 < num_experts <= MAX_EXPERTS
            and num_experts % 32 == 0
            and 0 < top_k <= MAX_TOPK
        ):
            return None
        shapes = _routed_shapes(num_experts)
        for name, shape in shapes.items():
            p = params.get(name)
            if p is None or tuple(p.shape) != shape or not p.is_contiguous():
                return None
        if (
            params["w13_weight"].dtype != torch.int32
            or params["w13_weight_scale_2"].dtype != torch.float32
            or params["w2_weight_scale_2"].dtype != torch.float32
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
            or ffn.routed_scaling_factor != ROUTED_SCALING
            or ffn.swiglu_limit != SWIGLU_LIMIT
        ):
            return None
        shared = None
        mlp = getattr(ffn, "shared_experts", None)
        if mlp is not None:
            tensors = (
                mlp.gate_up_proj.weight.view(torch.int32),
                mlp.gate_up_proj.weight_scale.view(torch.uint8),
                mlp.down_proj.weight.view(torch.int32),
                mlp.down_proj.weight_scale.view(torch.uint8),
            )
            shared_shapes = ((320, 5120), (160, 1280), (40, 20480), (20, 5120))
            if all(tuple(t.shape) == s for t, s in zip(tensors, shared_shapes)):
                shared = tensors
        gate = ffn.gate
        return cls(
            {name: params[name].data for name in shapes},
            shared,
            getattr(gate, "e_score_correction_bias", None),
            getattr(gate, "bias_vl", None),
            int(getattr(ffn, "image_sentinel_lo", 0)),
            getattr(gate, "tid2eid", None) is not None,
            top_k,
        )

    def serves(self, rows: int) -> bool:
        return 0 < rows <= MAX_ROWS

    def fuses_shared(self, rows: int) -> bool:
        return self.has_shared and rows <= SHARED_MAX_ROWS

    def warm_up(self, rows: list[int]) -> None:
        """Compile the variants calls of ``rows`` rows launch."""
        for m in rows:
            if self.serves(m):
                _compiled(self._config(m, self.hash_routed, self.image))

    def _config(self, rows: int, ids_mode: bool, image: bool) -> LaunchConfig:
        return launch_config(
            self.experts,
            self.top_k,
            rows,
            ids_mode,
            self.fuses_shared(rows),
            image,
        )

    def __call__(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[UnfinalizedMoEOutput, torch.Tensor | None]:
        """The unfinalized routed output of the BF16 MoE input ``x`` [rows,
        5120], routed from FP32 ``router_logits`` [rows, experts] or, on a
        hash-routed layer, by the stock router's ``topk_ids`` / ``topk_weights``
        (scaled); and the fused shared expert's output, or None above
        ``SHARED_MAX_ROWS`` rows. Both are views of the workspace."""
        from cutlass import Int32, Int64

        from ...ops.cute_dsl.primitives import current_cu_stream

        rows = x.shape[0]
        assert self.serves(rows) and x.is_contiguous() and x.dtype == torch.bfloat16
        ws = self.workspace
        shared = self.fuses_shared(rows)

        def address(t: torch.Tensor) -> Int64:
            return Int64(t.data_ptr())

        ids_mode = self.hash_routed
        image = self.image and not ids_mode and input_ids is not None
        if ids_mode:
            assert topk_ids is not None and topk_weights is not None
            ids = topk_ids.to(torch.int32).contiguous()
            weights = topk_weights.to(torch.float32).contiguous()
            logits = bias = x
        else:
            assert router_logits is not None and router_logits.shape == (
                rows,
                self.experts,
            )
            size = rows * self.top_k
            ids = ws.ids[:size].view(rows, self.top_k)
            weights = ws.wts[:size].view(rows, self.top_k)
            logits, bias = router_logits.float().contiguous(), self.bias
        # Image routing reads the token ids where the FP8 shared activations
        # would go (the W8A16 shared expert converts its own).
        xgs = ws.xgs
        if image:
            assert input_ids is not None
            xgs = input_ids.to(torch.int32).contiguous()
        _compiled(self._config(rows, ids_mode, image))(
            address(self.blob),
            address(self.scale13),
            address(self.scale2),
            address(ws.xg),
            address(ws.xsg),
            address(ws.rec),
            address(ws.iq),
            address(ws.isc),
            address(ws.part),
            address(ws.cnt),
            address(ws.sync),
            address(x),
            address(ids),
            address(weights),
            address(logits),
            address(bias),
            address(ws.rows),
            address(ws.out),
            address(xgs),
            address(ws.xsgs),
            address(ws.iqs),
            address(ws.iscs),
            address(ws.shared),
            Int32(rows),
            Int32(self._grid),
            current_cu_stream(),
        )
        routed = UnfinalizedMoEOutput(
            gemm2_permuted=ws.rows,
            expert_weights=weights[:rows],
            expanded_idx_to_permuted_idx=ws.route_to_row(rows, self.top_k),
        )
        return routed, ws.shared[:rows] if shared else None
