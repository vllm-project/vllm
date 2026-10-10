# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MegaMoE backends: fused EP dispatch, expert GEMMs and combine."""

from inspect import signature
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class DeepGemmMegaMoEBackend:
    """Abstract MegaMoE backend that hides deep_gemm-specific details.

    A backend advertises its expected hidden-state quantization layout via
    ``hidden_quant`` and its expected MMA type via ``mma_type``. It owns the
    kernel weight layout: :meth:`create_weights` registers parameters in that
    layout and :meth:`load_weight` writes each checkpoint tensor into it, so
    loading (and reloading) needs no post-load transform. It also runs the
    MegaMoE kernel. Runtime capability checks live in
    :func:`get_deep_gemm_mega_moe_backend`: obtaining a backend already implies
    that the current device and shapes are supported.
    """

    # Human-readable MMA type tag used by deep_gemm helpers. SM100 default.
    mma_type: str = "fp8xfp4"
    hidden_quant: QuantKey = kMxfp8Dynamic

    def create_weights(
        self,
        layer: nn.Module,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int,
    ) -> None:
        """Register the expert (and fused shared-expert) kernel-layout parameters:
        ``w13_weight``, ``w13_weight_scale``, ``w2_weight``, ``w2_weight_scale``
        and, if ``num_shared_experts``, ``shared_l{1,2}_weight[_scale]``. Routed
        parameters are indexed by local expert on dim 0, which EPLB moves."""
        raise NotImplementedError

    def load_weight(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> None:
        """Write checkpoint shard ``src`` (``w1``/``w2``/``w3``) into ``dst``, one
        expert's slice or a shared-expert parameter, in the kernel layout."""
        raise NotImplementedError

    def supports_shared_experts(self, shared_experts: nn.Module) -> bool:
        """Whether ``shared_experts`` can be fused into the MegaMoE kernel.

        Decided at construction from the installed kernel API and the shared
        MLP's parameters, so the fused layout is fixed before any weight loads.
        """
        return False

    def run_mega_moe(
        self,
        *,
        y: torch.Tensor,
        l1_weights: tuple[torch.Tensor, torch.Tensor],
        l2_weights: tuple[torch.Tensor, torch.Tensor],
        symm_buffer,
        activation_clamp: float | None,
        fast_math: bool,
        activation: str | None = None,
        activation_alpha: float | None = None,
        activation_beta: float | None = None,
        shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        raise NotImplementedError

    def supports_bf16_mega_gate(self) -> bool:
        """Whether this backend can run the DeepGEMM bf16 MegaMoE gate kernel.

        Defaults to ``False`` (the symbol is not guaranteed to exist in every
        deep_gemm build). Backends that target a deep_gemm build exposing the
        ``bf16_mega_gate`` symbol override this and :meth:`bf16_mega_gate`.
        """
        return False

    def bf16_mega_gate(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        num_topk: int,
        scoring_func: str,
        routed_scaling_factor: float,
        ep_rank: int,
        bias: torch.Tensor | None = None,
        image_bias: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        fix_routing_mask: torch.Tensor | None = None,
        unmapped_topk_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DeepGemmSm100MegaMoEBackend(DeepGemmMegaMoEBackend):
    """Default MegaMoE backend targeting SM100 via deep_gemm fp8_fp4_mega_moe."""

    mma_type = "fp8xfp4"
    hidden_quant = kMxfp8Dynamic

    def create_weights(
        self,
        layer: nn.Module,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int,
    ) -> None:
        def register(name: str, data: torch.Tensor) -> None:
            layer.register_parameter(name, nn.Parameter(data, requires_grad=False))

        # FP4 weights two per byte; UE8M0 scales MN-major, four per int32 along K.
        e, h, i = num_local_experts, hidden_size, intermediate_size
        register("w13_weight", torch.zeros(e, 2 * i, h // 2, dtype=torch.int8))
        register(
            "w13_weight_scale",
            torch.zeros(e, h // 128, 2 * i, dtype=torch.int32).transpose(1, 2),
        )
        register("w2_weight", torch.zeros(e, h, i // 2, dtype=torch.int8))
        register(
            "w2_weight_scale",
            torch.zeros(e, i // 128, h, dtype=torch.int32).transpose(1, 2),
        )
        if num_shared_experts == 0:
            return
        # The shared expert is FP8 with the layout of one routed expert slice.
        s, fp8 = i * num_shared_experts, torch.float8_e4m3fn
        register("shared_l1_weight", torch.zeros(2 * s, h, dtype=fp8))
        register(
            "shared_l1_weight_scale",
            torch.zeros(h // 128, 2 * s, dtype=torch.int32).transpose(0, 1),
        )
        register("shared_l2_weight", torch.zeros(h, s, dtype=fp8))
        register(
            "shared_l2_weight_scale",
            torch.zeros(s // 128, h, dtype=torch.int32).transpose(0, 1),
        )

    def load_weight(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
        if is_scale:
            src = self._pack_scale(src, rows, dst.shape[1] * 4)
        else:
            src = src.view(dst.dtype)
            if tuple(src.shape) != (rows, dst.shape[1]):
                raise ValueError(
                    f"MegaMoE {shard_id} weight shape {tuple(src.shape)} does not "
                    f"match the kernel shard ({rows}, {dst.shape[1]})"
                )
        dst, src = self._kernel_layout_shard(dst, src, shard_id, is_scale)
        dst.copy_(src)

    @staticmethod
    def _pack_scale(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Expand block UE8M0 scales to one per 1x32 and pack four along K per int32."""
        sf = sf.view(torch.uint8)
        if rows % sf.shape[0] or cols % sf.shape[1]:
            raise ValueError(
                f"Scale shape {tuple(sf.shape)} does not tile ({rows}, {cols})"
            )
        sf = sf.repeat_interleave(rows // sf.shape[0], 0)
        sf = sf.repeat_interleave(cols // sf.shape[1], 1)
        return sf.contiguous().clone().view(torch.int32)

    @staticmethod
    def _kernel_layout_shard(
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair a checkpoint shard with its destination in the MegaMoE layout."""
        k = src.shape[-1]
        if shard_id == "w2":
            if not is_scale:
                return dst, src
            # UTCCP: scale row 128*b + 32*j + i -> 128*b + 4*i + j.
            return dst.view(-1, 32, 4, k).transpose(1, 2), src.reshape(-1, 4, 32, k)
        half = 0 if shard_id == "w1" else 1
        if not is_scale:
            # Interleave gate/up by 8 rows: r -> (r // 8) * 16 + 8 * half + r % 8.
            return dst.view(-1, 2, 8, k)[:, half], src.reshape(-1, 8, k)
        # Interleave + UTCCP: 64b + 16j + 8q + m -> 128b + 64q + 32half + 4m + j.
        dst = dst.view(-1, 2, 2, 8, 4, k)[:, :, half].permute(0, 3, 1, 2, 4)
        return dst, src.reshape(-1, 4, 2, 8, k)

    def supports_shared_experts(self, shared_experts: nn.Module) -> bool:
        for linear in (shared_experts.gate_up_proj, shared_experts.down_proj):
            weight = linear.weight
            scale = getattr(linear, "weight_scale", None)
            if scale is None:
                scale = getattr(linear, "weight_scale_inv", None)
            if not self._tiles_1x32(weight, scale):
                logger.warning_once(
                    "Disabling native MegaMoE shared-expert fusion: needs FP8 "
                    "weights with UE8M0 block scales that tile to 1x32; got "
                    "weight %s/%s and scale %s/%s.",
                    tuple(weight.shape),
                    weight.dtype,
                    None if scale is None else tuple(scale.shape),
                    None if scale is None else scale.dtype,
                )
                return False

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        try:
            api = {
                *signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters,
                *signature(deep_gemm.fp8_fp4_mega_moe).parameters,
            }
        except (AttributeError, TypeError, ValueError):
            api = set()
        if not (
            hasattr(deep_gemm, "get_block_m_for_mega_moe")
            and {"num_shared_experts", "shared_l1_weights", "shared_l2_weights"} <= api
        ):
            logger.warning_once(
                "Disabling native MegaMoE shared-expert fusion because the "
                "installed DeepGEMM Python API is older than the vLLM "
                "source. Rebuild the vendored _deep_gemm_C extension to enable it.",
            )
            return False
        return True

    @staticmethod
    def _tiles_1x32(weight: torch.Tensor, scale: torch.Tensor | None) -> bool:
        if scale is None or weight.dim() != 2 or scale.dim() != 2:
            return False
        (rows, cols), (scale_rows, scale_cols) = weight.shape, scale.shape
        return (
            weight.dtype == torch.float8_e4m3fn
            and scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
            and rows % scale_rows == 0
            and cols % scale_cols == 0
            and (cols // scale_cols) % 32 == 0
        )

    def run_mega_moe(
        self,
        *,
        y: torch.Tensor,
        l1_weights: tuple[torch.Tensor, torch.Tensor],
        l2_weights: tuple[torch.Tensor, torch.Tensor],
        symm_buffer,
        activation_clamp: float | None,
        fast_math: bool,
        activation: str | None = None,
        activation_alpha: float | None = None,
        activation_beta: float | None = None,
        shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        kwargs: dict[str, Any] = {
            name: value
            for name, value in (
                ("activation", activation),
                ("activation_alpha", activation_alpha),
                ("activation_beta", activation_beta),
            )
            if value is not None
        }
        if shared_l1_weights is not None and shared_l2_weights is not None:
            kwargs["shared_l1_weights"] = shared_l1_weights
            kwargs["shared_l2_weights"] = shared_l2_weights
        deep_gemm.fp8_fp4_mega_moe(
            y,
            l1_weights,
            l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            **kwargs,
        )

    def supports_bf16_mega_gate(self) -> bool:
        """SM100 deep_gemm build is expected to provide the bf16 gate kernel."""
        return True

    def bf16_mega_gate(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        num_topk: int,
        scoring_func: str,
        routed_scaling_factor: float,
        ep_rank: int,
        bias: torch.Tensor | None = None,
        image_bias: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
        fix_routing_mask: torch.Tensor | None = None,
        unmapped_topk_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.utils.deep_gemm import bf16_mega_gate

        return bf16_mega_gate(
            x,
            weight,
            num_topk,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            ep_rank=ep_rank,
            bias=bias,
            image_bias=image_bias,
            image_token_mask=image_token_mask,
            fix_routing_mask=fix_routing_mask,
            unmapped_topk_idx=unmapped_topk_idx,
        )


def get_deep_gemm_mega_moe_backend(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
) -> DeepGemmMegaMoEBackend:
    """Return the DeepGEMM MegaMoE backend for ``device``.

    A successful return implies the backend can run on this device with the
    given ``hidden_size`` / ``intermediate_size``. All runtime capability
    checks (arch + shape divisibility) live here; there is no separate check
    on the returned backend.

    Currently only the SM100 deep_gemm backend is registered here. SM90 and
    other backends are expected to be provided by new backends or plugins
    that override this factory.

    Raises:
        NotImplementedError: If the device architecture is unsupported.
        ValueError: If the shapes are unsupported.

    """
    if not current_platform.is_device_capability_family(
        100, device_id=device.index or 0
    ):
        raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "DeepGEMM MegaMoE requires hidden and intermediate sizes "
            "to be multiples of 128."
        )
    return DeepGemmSm100MegaMoEBackend()
