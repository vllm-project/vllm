# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 quantization methods on NVIDIA GPUs before SM100.

`DeepseekV4FP8Config` asks this module, on CUDA only, for the methods of
the MXFP8 linear layers and of the MXFP4 routed experts:

- On A100 (SM80) and H100 (SM90) the MXFP8 linear layers run as BF16 GEMMs on
  weights dequantized once at load, and decode steps of a few rows on an FP8
  GEMV kernel that reads a packed copy of the FP8 weights.
- On A100 and H100 the Marlin MXFP4 experts pad the per-rank intermediate
  size to a multiple of 64 instead of 128, and leave their top-k reduction to
  the next mHC boundary. vLLM picks the experts' backend: Marlin on A100,
  Humming on H100 unless `--moe-backend marlin` is given.

Every other GPU gets the stock methods.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinExperts,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        ExpertTokensMetadata,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod

logger = init_logger(__name__)


def mxfp8_linear_as_bf16() -> bool:
    """Whether the MXFP8 linear layers run as BF16 GEMMs: on A100 (SM80) and
    H100 (SM90), whose MXFP8 kernels (Marlin, Humming) are slower than cuBLAS
    on the weights dequantized once at load at decode sizes. SM80 always does
    (its attention reads the BF16 `wo_a` weight); SM90 only when no MXFP8
    linear backend is requested explicitly."""
    if current_platform.is_device_capability(80):
        return True
    if not current_platform.is_device_capability(90):
        return False
    kernel_config = get_current_vllm_config().kernel_config
    overrides = kernel_config.linear_backend_per_quant or {}
    return (overrides.get("mxfp8") or kernel_config.linear_backend) == "auto"


# Decode steps of at most this many rows run the MXFP8 linear layers on the
# FP8 GEMV kernel (``specialized_decode/ampere/gemv.py``), which reads the FP8
# weights, half the bytes of the BF16 copy. Larger steps and prefill keep
# cuBLAS on the BF16 copy. On H100 at TP8 a layer's dense GEMMs take 0.65x
# cuBLAS's time at 6 rows, 0.71x at 12 and 0.77x at 24, and about the same
# time at 48. The packed copy costs N x K bytes per layer (about 29 MB per
# decoder layer, 1.2 GB per GPU at TP8), taken from the KV cache.
GEMV_MAX_ROWS = 24


def pack_for_gemv(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The GEMV kernel's copy of an MXFP8 weight `[N, K]` with row-expanded
    UE8M0 scales `[N, K / 32]`, or None for a weight it cannot take: a
    shape that is not whole 32 x 32 blocks, scales that differ within a block's
    rows, or a scale above `2^7`."""
    from .specialized_decode.ampere.gemv import GROUP, MAX_SCALE_CODE, pack_weight

    n, k = weight.shape
    if weight.element_size() != 1 or n % GROUP or k % GROUP:
        return None
    codes = scales[:n, : k // GROUP].contiguous().view(torch.uint8)
    blocks = codes.view(n // GROUP, GROUP, k // GROUP)
    if not torch.equal(blocks, blocks[:, :1].expand_as(blocks)):
        return None
    if int(codes.max()) > MAX_SCALE_CODE:
        return None
    return pack_weight(weight, codes)


@functools.cache
def bf16_mxfp8_linear_method_cls() -> type[ModelOptLinearMethod]:
    """`ModelOptLinearMethod` with the MXFP8 weights dequantized to BF16 at
    load (the emulation kernel) instead of the platform's MXFP8 kernel, and
    decode steps of at most `GEMV_MAX_ROWS` rows on the FP8 GEMV kernel."""
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
    )

    from .specialized_decode.ampere.gemv import mxfp8_gemv

    class DecodeGemvMxfp8LinearKernel(EmulationMxfp8LinearKernel):
        """BF16 cuBLAS on the weights dequantized at load, and the FP8 GEMV
        on a packed copy of the FP8 weights for small decode steps."""

        # The packed weights, their scale factors and their own ticket
        # counters (one per 16 rows; projections of one layer run on several
        # streams at once), or None when the GEMV cannot take the weight.
        gemv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            weight, scales = layer.weight, layer.weight_scale
            assert isinstance(weight, torch.Tensor)
            assert isinstance(scales, torch.Tensor)
            packed = None
            if self.config.bmm_batch_size is None:
                packed = pack_for_gemv(weight.data, scales.data)
            super().process_weights_after_loading(layer)
            if self.gemv is not None:
                # A reload of the weights (reload_weights, a weight update in
                # RL) runs this again, and captured CUDA graphs read the first
                # packed copy by its address: write the new one into it. (A
                # reload in the kernel's own format copies only the layer's
                # parameters and leaves the packed copy as it was.)
                if packed is None:
                    raise RuntimeError(
                        "reloaded MXFP8 weights no longer fit the FP8 GEMV"
                    )
                self.gemv[0].copy_(packed[0])
                self.gemv[1].copy_(packed[1])
            elif packed is not None:
                blocks, factors = packed
                counters = torch.zeros(
                    blocks.shape[0], dtype=torch.int32, device=blocks.device
                )
                self.gemv = (blocks, factors, counters)

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                self.gemv is not None
                and bias is None
                and x.dim() == 2
                and 0 < x.shape[0] <= GEMV_MAX_ROWS
                and x.dtype == torch.bfloat16
                and x.stride(1) == 1
                and x.stride(0) % 8 == 0
                and x.data_ptr() % 16 == 0
            ):
                blocks, factors, counters = self.gemv
                return mxfp8_gemv(x, blocks, factors, counters=counters)
            return super().apply_weights(layer, x, bias)

    class BF16Mxfp8LinearMethod(ModelOptLinearMethod):
        """The MXFP8 linear layer as a BF16 GEMM on dequantized weights."""

        # As declared by `ModelOptLinearMethod`, which holds any selected
        # kernel; mypy would otherwise type it from the assignment below.
        kernel: Any

        def create_weights(self, layer, *args, **kwargs) -> None:
            super().create_weights(layer, *args, **kwargs)
            logger.info_once(
                "DeepSeek-V4.1 MXFP8 linear layers: BF16 GEMMs on weights "
                "dequantized at load, FP8 GEMV up to %d rows, in place of %s",
                GEMV_MAX_ROWS,
                type(self.kernel).__name__,
            )
            self.kernel = DecodeGemvMxfp8LinearKernel(
                Mxfp8LinearLayerConfig(
                    weight_shape=self.fmt.kernel_weight_shape(layer),
                    bmm_batch_size=getattr(layer, "bmm_batch_size", None),
                )
            )

    return BF16Mxfp8LinearMethod


def mxfp8_linear_method_cls() -> type[ModelOptLinearMethod]:
    """The method of the MXFP8 linear layers (`mxfp8_linear_as_bf16`)."""
    if mxfp8_linear_as_bf16():
        return bf16_mxfp8_linear_method_cls()
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod

    return ModelOptLinearMethod


@functools.cache
def _identity_index(device: torch.device, length: int) -> torch.Tensor:
    """`arange(length)` as INT32 on `device`, one copy per process: a
    weight reload rebuilds the experts, and the captured CUDA graphs keep
    reading this address."""
    return torch.arange(length, dtype=torch.int32, device=device)


class DeferringMarlinExperts(MarlinExperts):
    """Marlin's experts that can leave the top-k reduction to the next mHC
    boundary's fused all-reduce (`AllReduceMHC.finalize`), as SM100's
    TRT-LLM-Gen experts do: when the layer's config defers a call, GEMM2
    writes its rows unweighted, one per route, and the boundary weights and
    sums them and adds the shared output inside the all-reduce. The rows are
    in route order, so the boundary's row index is the identity: one copy
    per device (`_identity_index`) serves every deferred call (Marlin's
    `post_gemm2_hook` path would build it with a kernel per call)."""

    route_index: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Take the deferred calls' row index, outside any CUDA graph."""
        super().process_weights_after_loading(layer)
        if self.moe_config.use_deferred_moe_finalize:
            max_num_tokens = self.moe_config.defer_moe_finalize_max_num_tokens
            assert max_num_tokens > 0 and self.w1_scale is not None
            self.route_index = _identity_index(
                self.w1_scale.device, max_num_tokens * self.moe_config.experts_per_token
            )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> UnfinalizedMoEOutput | None:
        # The layer routes the output by the same answer, so a call it defers
        # must defer.
        if not self.moe_config.should_defer_moe_finalize(hidden_states.shape[0]):
            super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
            return None
        if (
            self.post_gemm2_hook is not None
            or self._lora_context is not None
            or apply_router_weight_on_input
            or expert_map is not None
        ):
            raise NotImplementedError(
                "A deferred Marlin call takes no post-GEMM2 hook, LoRA, router "
                "weights on the input or expert map."
            )
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.route_index is not None
        routes: list[torch.Tensor] = []
        # As MarlinExperts.apply, but GEMM2 leaves its rows unweighted and
        # moe_sum keeps them.
        fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            bias1=self.w1_bias,
            bias2=self.w2_bias,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_scale1=self.g1_alphas,
            global_scale2=self.g2_alphas,
            w1_zeros=self.w1_zp,
            w2_zeros=self.w2_zp,
            quant_type_id=self.quant_type_id,
            global_num_experts=global_num_experts,
            activation=activation,
            activation_func=self.activation,
            activation_config=self.activation_config,
            moe_sum=lambda rows, *_: routes.append(rows),
            output=output,
            # Swapped as in MarlinExperts.apply (see workspace_shapes).
            intermediate_cache13=workspace2,
            intermediate_cache2=workspace13,
            mul_topk_weights_in_gemm2=False,
        )
        num_routes = topk_ids.numel()
        return UnfinalizedMoEOutput(
            gemm2_permuted=routes[0].view(num_routes, -1),
            expert_weights=topk_weights,
            expanded_idx_to_permuted_idx=self.route_index[:num_routes].view_as(
                topk_ids
            ),
            route_major=True,
        )


class MarlinMxfp4MoEMethod(Mxfp4MoEMethod):
    """The MXFP4 routed experts on SM80 and SM90: with the Marlin backend, the
    per-rank intermediate size is padded to a multiple of 64 instead of 128.

    Marlin's MXFP4 path rounds the intermediate size up to 128, but Marlin
    needs only 64, its smallest N tile of GEMM1 and K tile of GEMM2 (as
    `marlin_moe_padded_intermediate` states). At TP8 the 2304-wide experts
    leave 288 columns per rank: 384 with 128, so a quarter of every expert's
    bytes are zeros, and 320 with 64. The padded weights are zero, so the
    products are unchanged; only FP32 summation orders change (GEMM2's K
    split, and how the GEMMs split their work across SMs).

    The backend is vLLM's choice: Marlin on SM80; on SM90, Humming with block
    FP8 activations unless `--moe-backend marlin` is given."""

    def __init__(self, moe: FusedMoEConfig, activation_key: QuantKey | None = None):
        super().__init__(moe, activation_key=activation_key)
        if (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability(80)
        ) and self.experts_cls is MarlinExperts:
            # SM90 fuses the TP all-reduce into the mHC boundaries and SM80
            # reduces at them with its own kernel; both can take the top-k
            # reduction too.
            self.experts_cls = DeferringMarlinExperts

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        """The stock sizes, except the Marlin intermediate size (class
        docstring)."""
        hidden_size, rounded = super().maybe_roundup_sizes(
            hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config
        )
        if self.mxfp4_backend == Mxfp4MoeBackend.MARLIN:
            rounded = round_up(intermediate_size_per_partition, 64)
        return hidden_size, rounded


def mxfp4_moe_method(moe: FusedMoEConfig) -> Mxfp4MoEMethod:
    """The method of the MXFP4 routed experts: `MarlinMxfp4MoEMethod`
    on A100 (SM80) and H100 (SM90), the stock method elsewhere, with the
    checkpoint's MXFP8 activation."""
    method_cls = (
        MarlinMxfp4MoEMethod
        if current_platform.is_device_capability(80)
        or current_platform.is_device_capability(90)
        else Mxfp4MoEMethod
    )
    return method_cls(moe, activation_key=kMxfp8Dynamic)
