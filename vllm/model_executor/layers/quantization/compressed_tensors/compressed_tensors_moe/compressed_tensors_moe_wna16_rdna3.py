# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A16 using the fused RDNA3 (gfx1100) HIP kernel.

Uses ``moe_gptq_gemm_rdna3`` — a single HIP kernel launch per GEMM that
handles expert routing + W4A16 dequant + dot product with atomic output.

Weight format (per expert, same as dense RDNA3 W4A16):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` in activation dtype
  - Zero points ``[E, groups, N/8]`` packed int32 (synthesized)
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


def _synthesize_qzeros(
    groups: int, out_features: int, device: torch.device
) -> torch.Tensor:
    """Create packed zero-point tensor for symmetric quant.

    GPTQv1 +1 quirk: kernel adds 1 to stored zeros, so encode
    (bias - 1) = 7 for uint4b8 (bias=8).
    """
    zeros = torch.full(
        (groups, out_features),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    return pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)


def _expert_cache_size() -> int:
    """Cold-expert offload cache size (# experts kept resident per layer/rank).

    0 (default) disables offloading -> all experts resident on GPU (unchanged).
    """
    try:
        return int(os.environ.get("VLLM_MOE_EXPERT_CACHE_SIZE", "0"))
    except ValueError:
        return 0


class CompressedTensorsWNA16RDNA3MoEMethod(CompressedTensorsWNA16MoEMethod):
    """W4A16 MoE using the fused RDNA3 HIP kernel (moe_gptq_gemm_rdna3).

    Weights are in RDNA3 format (shuffled int32 [E, K/8, N]),
    NOT Triton format (transposed uint8). apply() dispatches through
    the fused HIP kernel directly.

    When ``VLLM_MOE_EXPERT_CACHE_SIZE > 0`` the experts are offloaded to pinned
    CPU RAM and a fixed GPU cache of that many experts is streamed in per step
    by the in-graph gather kernel (see expert_gather.ExpertOffloadCache).
    """

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Offload: allocate the (large) expert params on CPU pinned so the full
        # E-expert set never has to fit in VRAM during load. torch.device('cpu')
        # context makes the base create_weights' torch.empty land on CPU.
        if _expert_cache_size() > 0:
            with torch.device("cpu"):
                super().create_weights(
                    layer,
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    params_dtype,
                    **extra_weight_attrs,
                )
            for name in (
                "w13_weight_packed",
                "w2_weight_packed",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                p = getattr(layer, name)
                p.data = p.data.pin_memory()
        else:
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _expert_cache_size() > 0:
            return self._process_weights_offload(layer)
        device = layer.w13_weight_packed.device
        num_experts = layer.w13_weight_packed.shape[0]
        empty_g_idx = torch.empty(0, dtype=torch.int32, device=device)

        # Shuffle weights in-place per expert (exllama nibble interleave)
        for e in range(num_experts):
            w13_e = layer.w13_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w13_e, empty_g_idx, 4)
            layer.w13_weight_packed.data[e] = w13_e
            w2_e = layer.w2_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w2_e, empty_g_idx, 4)
            layer.w2_weight_packed.data[e] = w2_e

        # Keep scales as [E, groups, N] in activation dtype
        act_dtype = layer.w13_weight_scale.dtype
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )

        # Synthesize packed zero points: [E, groups, N/8] int32
        w13_groups = (layer.w13_weight_packed.shape[1] * 8) // self.group_size
        w13_N = layer.w13_weight_packed.shape[2]
        w2_groups = (layer.w2_weight_packed.shape[1] * 8) // self.group_size
        w2_N = layer.w2_weight_packed.shape[2]

        w13_qz = _synthesize_qzeros(w13_groups, w13_N, device)
        w2_qz = _synthesize_qzeros(w2_groups, w2_N, device)
        layer.w13_qzeros = torch.nn.Parameter(
            w13_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
            requires_grad=False,
        )
        layer.w2_qzeros = torch.nn.Parameter(
            w2_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
            requires_grad=False,
        )

        # Pre-allocate reusable buffers for decode (sizes based on top_k=8)
        N_gate_up = w13_N
        hidden_size = w2_N
        intermediate = N_gate_up // 2  # gated activation
        # Max tokens we expect in decode; prefill will re-allocate if needed
        max_decode_tokens = 16
        top_k = 8  # conservative default
        buf_size = max_decode_tokens * top_k
        layer.rdna3_w1_buf = torch.zeros(
            buf_size, N_gate_up, dtype=act_dtype, device=device
        )
        layer.rdna3_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna3_out_buf = torch.zeros(
            max_decode_tokens, hidden_size, dtype=act_dtype, device=device
        )
        layer.rdna3_empty_tw = torch.empty(0, device=device)
        layer.expert_cache = None

    def _process_weights_offload(self, layer: torch.nn.Module) -> None:
        """Move experts to pinned CPU masters + build the GPU cache.

        vLLM's ``device_loading_context`` has already moved this layer's params
        onto the GPU before we run, so we shuffle the packed weights IN-PLACE on
        the GPU (fast, no per-expert H2D), then move each prepared plane to a
        CPU pinned master and FREE the GPU param (``p.data = empty``). Freeing is
        essential: if we kept a reference to the GPU tensor, the 48 layers would
        pile up on the GPU (the device_loading_context move-back can't reclaim a
        referenced tensor). Zero-points are identical across experts -> a single
        static [C, ...] GPU tensor, not gathered.
        """
        from vllm.model_executor.layers.fused_moe.expert_gather import (
            ExpertOffloadCache,
        )

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        E = layer.w13_weight_packed.shape[0]
        C = min(_expert_cache_size(), E)
        act_dtype = layer.w13_weight_scale.dtype
        empty_g_idx = torch.empty(0, dtype=torch.int32, device=device)

        gw13 = (layer.w13_weight_packed.shape[1] * 8) // self.group_size
        w13_N = layer.w13_weight_packed.shape[2]
        gw2 = (layer.w2_weight_packed.shape[1] * 8) // self.group_size
        w2_N = layer.w2_weight_packed.shape[2]

        # shuffle packed weights in-place on the GPU-resident params
        for e in range(E):
            ops.gptq_shuffle(layer.w13_weight_packed.data[e], empty_g_idx, 4)
            ops.gptq_shuffle(layer.w2_weight_packed.data[e], empty_g_idx, 4)

        # move each prepared plane to a CPU pinned master, then free the GPU param
        cache = ExpertOffloadCache(E, C, max_sel=C, device=device)
        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            p = getattr(layer, name)
            src = p.data
            if name.endswith("scale") and src.dtype != act_dtype:
                src = src.to(act_dtype)
            master = src.to("cpu").pin_memory()
            p.data = torch.empty(0, device=device)  # free GPU (avoid pile-up)
            shape = master.shape[1:]
            cache.add_plane(
                name,
                master,
                torch.empty(C, *shape, dtype=master.dtype, device=device),
            )
        cache.warm(list(range(C)))
        layer.expert_cache = cache

        # zero-points: identical for every expert -> one static [C, ...] tensor
        qz13 = _synthesize_qzeros(gw13, w13_N, device)
        qz2 = _synthesize_qzeros(gw2, w2_N, device)
        layer.w13_qzeros = qz13.unsqueeze(0).expand(C, -1, -1).contiguous()
        layer.w2_qzeros = qz2.unsqueeze(0).expand(C, -1, -1).contiguous()

        # decode scratch buffers
        intermediate = w13_N // 2
        buf = 16 * 8
        layer.rdna3_w1_buf = torch.zeros(buf, w13_N, dtype=act_dtype, device=device)
        layer.rdna3_act_buf = torch.empty(
            buf, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna3_out_buf = torch.zeros(16, w2_N, dtype=act_dtype, device=device)
        layer.rdna3_empty_tw = torch.empty(0, device=device)
        # per-layer resident cache bytes (cache tensors already sized [C, ...])
        per_layer = (
            sum(
                p["cache"].numel() * p["cache"].element_size()
                for p in cache.planes.values()
            )
            + layer.w13_qzeros.numel() * 4
            + layer.w2_qzeros.numel() * 4
        )
        logger.info_once(
            "[expert-offload] cache C=%d / E=%d experts, "
            "%.0f MB/layer (%.1f GB/rank over %d layers est.)",
            C,
            E,
            per_layer / 1e6,
            per_layer * 48 / 1e9,
            48,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        return _rdna3_fused_moe(
            x,
            topk_weights,
            topk_ids,
            layer=layer,
            activation=activation,
            apply_router_weight_on_input=(layer.apply_router_weight_on_input),
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )


def _rdna3_fused_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer: RoutedExperts,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Fused MoE forward using the RDNA3 W4A16 HIP kernel.

    Optimizations vs naive dispatch:
      - BLOCK_SIZE_M=1 for decode (no padding waste, bf16 fast path)
      - Pre-allocated buffers (no torch.zeros per call)
      - Inline token sorting for small M (skip moe_align_block_size)
      - moe_sum fused into output accumulation
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]

    # --- cold-expert offload: chunk tokens so a step's working set <= C, then
    # gather the needed experts into cache slots and remap topk to slot space ---
    ec = getattr(layer, "expert_cache", None)
    if ec is not None:
        max_tok = max(1, ec.C // top_k)
        if num_tokens > max_tok:
            outs = [
                _rdna3_fused_moe(
                    hidden_states[i : i + max_tok],
                    topk_weights[i : i + max_tok],
                    topk_ids[i : i + max_tok],
                    layer,
                    activation,
                    apply_router_weight_on_input,
                    global_num_experts,
                    expert_map,
                )
                for i in range(0, num_tokens, max_tok)
            ]
            return torch.cat(outs, dim=0)
        topk_ids = ec.ensure(topk_ids)  # -> slot space [0, C)
        w13_packed = ec.cache_tensor("w13_weight_packed")
        w2_packed = ec.cache_tensor("w2_weight_packed")
        w13_scale = ec.cache_tensor("w13_weight_scale")
        w2_scale = ec.cache_tensor("w2_weight_scale")
        w13_qzeros = layer.w13_qzeros  # static [C, ...] (shared across experts)
        w2_qzeros = layer.w2_qzeros
        global_num_experts = ec.C
        expert_map = None
    else:
        w13_packed = layer.w13_weight_packed
        w2_packed = layer.w2_weight_packed
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_qzeros = layer.w13_qzeros
        w2_qzeros = layer.w2_qzeros

    total_tokens = num_tokens * top_k
    N_gate_up = w13_packed.shape[2]
    hidden_size = w2_packed.shape[2]
    dtype = hidden_states.dtype
    device = hidden_states.device

    intermediate_size = N_gate_up // 2 if activation.is_gated else N_gate_up

    if global_num_experts <= 0:
        global_num_experts = w13_packed.shape[0]

    # BLOCK_SIZE_M=1 for decode (small M), 4 for prefill
    block_size_m = 1 if num_tokens <= 4 else 4

    # --- Token routing ---
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
    )

    # --- Reuse pre-allocated buffers when possible ---
    if total_tokens <= layer.rdna3_w1_buf.shape[0]:
        w1_out = layer.rdna3_w1_buf[:total_tokens]
        w1_out.zero_()
        act_out = layer.rdna3_act_buf[:total_tokens]
    else:
        w1_out = torch.zeros(
            total_tokens,
            N_gate_up,
            dtype=dtype,
            device=device,
        )
        act_out = torch.empty(
            total_tokens,
            intermediate_size,
            dtype=dtype,
            device=device,
        )

    # --- topk weights (pre-cast to float32 for kernel) ---
    topk_w_float = topk_weights.view(-1).float()
    empty_tw = layer.rdna3_empty_tw

    # --- w1 GEMM: [M, K] -> [M*top_k, N_gate_up] ---
    ops.moe_gptq_gemm_rdna3(
        hidden_states,
        w1_out,
        w13_packed,
        w13_scale,
        w13_qzeros,
        topk_w_float if apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        block_size_m,
        apply_router_weight_on_input,
    )

    # --- Activation (silu_and_mul etc.) ---
    apply_moe_activation(activation, act_out, w1_out)

    # --- w2 GEMM: [M*top_k, intermediate] -> [M, hidden] (fused reduce) ---
    # output_topk=top_k: kernel writes to out[token_id / top_k] directly,
    # fusing moe_sum into the atomic accumulation — saves one kernel launch
    # and the w2_out intermediate buffer.
    out = torch.zeros(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    ops.moe_gptq_gemm_rdna3(
        act_out,
        out,
        w2_packed,
        w2_scale,
        w2_qzeros,
        topk_w_float if not apply_router_weight_on_input else empty_tw,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
        block_size_m,
        not apply_router_weight_on_input,
        output_topk=top_k,
    )
    return out
