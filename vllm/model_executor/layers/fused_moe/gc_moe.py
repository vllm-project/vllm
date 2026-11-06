# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper GC MoE with packed MXFP4 routed and fused FP8 shared experts."""

from typing import Any, ClassVar

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
    _linear_weight_scale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _gc_prepare_inputs_kernel(
    X,
    IDS,
    WEIGHTS,
    ROUTED,
    ROUTED_SF,
    SHARED,
    SHARED_SF,
    OUT_IDS,
    OUT_WEIGHTS,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    IDS_STRIDE: tl.constexpr,
    WEIGHTS_STRIDE: tl.constexpr,
    IDS_INNER_STRIDE: tl.constexpr,
    WEIGHTS_INNER_STRIDE: tl.constexpr,
    NATIVE_QUANT: tl.constexpr,
    GROUPS: tl.constexpr,
):
    blocks_per_row = tl.cdiv(HIDDEN // 128, GROUPS)
    block = tl.program_id(0)
    row = (block // blocks_per_row).to(tl.int64)
    column = block % blocks_per_row
    groups = column * GROUPS + tl.arange(0, GROUPS)
    shared_groups = groups[:, None] * 4 + tl.arange(0, 4)[None, :]
    offsets = shared_groups[:, :, None] * 32 + tl.arange(0, 32)[None, None, :]
    valid = offsets < HIDDEN
    values = tl.load(X + row * X_STRIDE + offsets, valid, 0).to(tl.float32)
    maxima = tl.maximum(tl.max(tl.abs(values), 2), 1e-4)
    routed_max = tl.max(maxima, 1)
    # Match the native CUDA and noncontiguous Triton quantizers' rounding.
    if NATIVE_QUANT:
        routed_scale = tl.div_rn(routed_max, 448.0)
        shared_scale = tl.div_rn(maxima, 448.0)
    else:
        routed_scale = routed_max * (1.0 / 448.0)
        shared_scale = maxima * (1.0 / 448.0)
    bits = shared_scale.to(tl.uint32, bitcast=True)
    # Round positive K32 scales up to powers of two, stored as FP32 by GC.
    shared_scale = ((bits + 0x7FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
    if NATIVE_QUANT:
        routed_values = values * tl.div_rn(1.0, routed_scale)[:, None, None]
        shared_values = values * tl.div_rn(1.0, shared_scale)[:, :, None]
    else:
        routed_values = values / routed_scale[:, None, None]
        shared_values = values / shared_scale[:, :, None]
    tl.store(ROUTED + row * HIDDEN + offsets, tl.clamp(routed_values, -448, 448), valid)
    tl.store(SHARED + row * HIDDEN + offsets, tl.clamp(shared_values, -448, 448), valid)
    tl.store(
        ROUTED_SF + row * (HIDDEN // 128) + groups,
        routed_scale,
        groups < HIDDEN // 128,
    )
    tl.store(
        SHARED_SF + row * (HIDDEN // 32) + shared_groups,
        shared_scale,
        shared_groups < HIDDEN // 32,
    )
    if column == 0:
        routes = tl.arange(0, triton.next_power_of_2(TOP_K))
        ids = tl.load(
            IDS + row * IDS_STRIDE + routes * IDS_INNER_STRIDE, routes < TOP_K, -1
        )
        weights = tl.load(
            WEIGHTS + row * WEIGHTS_STRIDE + routes * WEIGHTS_INNER_STRIDE,
            routes < TOP_K,
            0,
        )
        tl.store(OUT_IDS + row * TOP_K + routes, ids, routes < TOP_K)
        tl.store(OUT_WEIGHTS + row * TOP_K + routes, weights, routes < TOP_K)


def _prepare_gc_inputs(hidden_states, topk_weights, topk_ids, buffer):
    tokens, hidden = hidden_states.shape
    if not tokens:
        return
    assert hidden_states.stride(-1) == 1
    groups = 4
    _gc_prepare_inputs_kernel[(tokens * triton.cdiv(hidden // 128, groups),)](
        hidden_states,
        topk_ids,
        topk_weights,
        buffer.x,
        buffer.x_sf,
        buffer.shared_l1_acts,
        buffer.shared_l1_acts_sf,
        buffer.topk_idx,
        buffer.topk_weights,
        hidden,
        topk_ids.shape[1],
        hidden_states.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(1),
        hidden_states.is_contiguous(),
        groups,
        num_warps=1,
    )


def _get_num_tokens_bound(capacity: int) -> int:
    if not is_forward_context_available():
        return capacity
    context = get_forward_context()
    # Keep eager/mixed batches on the capacity geometry: arbitrary token-count
    # pairs would otherwise multiply GC preparation and child-graph cache keys.
    # The new GPU runner captures FULL graphs with runtime mode NONE.
    if (
        context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        and not torch.cuda.is_current_stream_capturing()
    ):
        return capacity
    metadata = context.dp_metadata
    if metadata is None:
        return capacity
    # DP coordination includes graph padding. These CPU counts give every rank
    # the same bound, also conservatively covering TP sequence shards.
    tokens = int(metadata.num_tokens_across_dp_cpu.max().item())
    return max(1, min(capacity, tokens))


class GcMoEExperts(DeepGemmMegaMoEExperts):
    """Reuse native model/loading contracts with GC-owned layouts and launches.

    Workspaces are shared across layers on the worker's execution stream.
    DBO, EPLB, weight updates and multi-node EP are deliberately unsupported.
    """

    _symm_buffers: ClassVar[dict[tuple, Any]] = {}
    _synchronized_ep_groups: ClassVar[set[tuple[int, int]]] = set()

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        parallel = vllm_config.parallel_config
        if (
            parallel.enable_eplb
            or parallel.eplb_config.num_redundant_experts
            or parallel.use_ubatching
            or parallel.expert_placement_strategy != "linear"
            or parallel.nnodes != 1
            or parallel.data_parallel_size != parallel.data_parallel_size_local
        ):
            raise ValueError(
                "gc_moe requires single-node linear EP without EPLB or ubatching"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError("gc_moe does not yet support speculative decoding")
        if vllm_config.weight_transfer_config is not None:
            raise ValueError("gc_moe does not support runtime weight transfer")
        import gc_moe

        self._gc = gc_moe
        self._prepared = False
        super().__init__(vllm_config, **kwargs)
        if self.num_experts != self.num_logical_experts:
            raise ValueError("gc_moe does not support redundant experts")

    @staticmethod
    def check_supported(device, hidden_size, intermediate_size):
        if not current_platform.is_device_capability(90, device_id=device.index or 0):
            raise NotImplementedError("gc_moe requires SM90 GPUs")
        if not (
            512 <= hidden_size <= 8192
            and hidden_size % 512 == 0
            and 256 <= intermediate_size <= 4096
            and intermediate_size % 256 == 0
        ):
            raise ValueError("Unsupported gc_moe hidden or intermediate size")

    def _fusable_shared_experts(self, shared_experts):
        if shared_experts is None:
            raise ValueError("gc_moe requires fused, unsharded FP8 shared experts")
        gate, down = shared_experts.gate_up_proj, shared_experts.down_proj
        shared_size = down.weight.shape[-1]
        for linear in (gate, down):
            sf = _linear_weight_scale(linear)
            if (
                linear.weight.dtype != torch.float8_e4m3fn
                or sf is None
                or sf.ndim != 2
                or sf.dtype not in (torch.float32, torch.uint8, torch.float8_e8m0fnu)
            ):
                raise ValueError("gc_moe shared experts require block-scaled FP8")
        if (
            tuple(gate.weight.shape) != (2 * shared_size, self.hidden_size)
            or tuple(down.weight.shape) != (self.hidden_size, shared_size)
            or shared_size % self.intermediate_size
        ):
            raise ValueError("gc_moe shared expert weights must be unsharded")
        return shared_size // self.intermediate_size, MegaMoEWeightSpec(
            torch.float8_e4m3fn, scale_group_k=32, scale_dtype=torch.float32
        )

    def _register_weight(self, name, shape, spec):
        *outer, n, k = shape
        self.register_parameter(
            f"{name}_weight",
            nn.Parameter(
                torch.zeros(*outer, n, k // spec.values_per_element, dtype=spec.dtype),
                requires_grad=False,
            ),
        )
        scale_dtype = torch.float32 if name.startswith("shared_") else torch.uint8
        self.register_parameter(
            f"{name}_weight_scale",
            nn.Parameter(
                torch.zeros(*outer, n, k // 32, dtype=scale_dtype), requires_grad=False
            ),
        )

    def _load_weight(self, dst, src, shard_id, is_scale):
        if self._prepared:
            raise RuntimeError("gc_moe weights cannot be reloaded after preparation")
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard {shard_id}")
        if shard_id != "w2":
            half = dst.shape[0] // 2
            dst = dst[:half] if shard_id == "w1" else dst[half:]
        if is_scale:
            if dst.dtype == torch.float32:
                if src.dtype in (torch.uint8, torch.float8_e8m0fnu):
                    src = torch.exp2(src.view(torch.uint8).float() - 127)
                else:
                    src = src.float()
            else:
                if src.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                    raise ValueError("gc_moe routed scales must be UE8M0")
                src = src.view(torch.uint8)
            if dst.shape[0] % src.shape[0] or dst.shape[1] % src.shape[1]:
                raise ValueError("Checkpoint scales do not tile gc_moe weights")
            src = src.repeat_interleave(dst.shape[0] // src.shape[0], 0)
            src = src.repeat_interleave(dst.shape[1] // src.shape[1], 1)
        else:
            src = src.view(dst.dtype)
        if src.shape != dst.shape:
            raise ValueError(f"gc_moe shard shape {src.shape} != {dst.shape}")
        dst.copy_(src)

    def get_symm_buffer(self):
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
        )
        if key not in self._symm_buffers:
            self._symm_buffers[key] = self._gc.get_symm_buffer_for_gc_moe_sm90(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=self.num_shared_experts,
            )
        return self._symm_buffers[key]

    def process_weights_after_loading(self):
        """Convert loaded weights once, before any graph captures their pointers."""
        if self._prepared:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up gc_moe before CUDA Graph capture")
        convert = self._gc.transform_weights_for_gc_moe_sm90
        self._routed_weights = convert(self.l1_weights, self.l2_weights)
        self._shared_weights = convert(
            self._kernel_weights("shared_l1"),
            self._kernel_weights("shared_l2"),
            shared_recipe=(1, 1, 32),
        )
        # Keep the checkpoint parameter names while retiring the raw storage.
        for name, values in zip(
            ("w13", "w2", "shared_l1", "shared_l2"),
            (*self._routed_weights, *self._shared_weights),
        ):
            getattr(self, f"{name}_weight").data = values[0]
            getattr(self, f"{name}_weight_scale").data = values[1]
        self.get_symm_buffer()
        self._prepared = True

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        if shared_experts is not None:
            raise ValueError("gc_moe owns the fused shared expert")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("gc_moe requires BF16 hidden states")
        tokens = hidden_states.shape[0]
        if tokens > self.max_num_tokens:
            raise ValueError("gc_moe token count exceeds workspace capacity")
        self.process_weights_after_loading()
        self._synchronize_first_launch()
        buffer = self.get_symm_buffer()
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        _prepare_gc_inputs(hidden_states, topk_weights, topk_ids, buffer)
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        self._gc.sm90_fp8_fp4_gc_moe(
            output,
            *self._routed_weights,
            buffer,
            shared_l1_weights=self._shared_weights[0],
            shared_l2_weights=self._shared_weights[1],
            shared_recipe=(1, 1, 32),
            activation_clamp=self.activation.clamp,
            num_tokens_bound=_get_num_tokens_bound(buffer.num_max_tokens_per_rank),
            fast_math=self.fast_math,
        )
        return output
