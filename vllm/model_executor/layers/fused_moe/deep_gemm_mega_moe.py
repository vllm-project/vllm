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


def ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    """Reinterpret a uint8 UE8M0 exponent tensor as float32 scale values."""
    return (sf.to(torch.int32) << 23).view(torch.float32)


_FP8_MAX = 448.0
_SF_GRAN_K = 32


def requant_block_fp8_to_ue8m0(
    weight: torch.Tensor,
    scale: torch.Tensor,
    chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize block-FP8 weights to E4M3 with one UE8M0 scale per 1x32.

    The SM100 MegaMoE feeds its scales to the block-scaled MMA, so they have to
    be powers of two, one per row and 32 columns. A checkpoint with float32
    scales per block (e.g. 128x128) is rounded once more to get there.

    Args:
        weight: ``(..., mn, k)`` E4M3 values.
        scale: ``(..., mn / block_m, k / block_k)`` float32 dequantization
            scales (``real = weight * scale``).
        chunk: Leading-dimension entries converted at a time, bounding the
            float32 workspace.

    Returns:
        ``(weight, sf)``: the requantized E4M3 weights and their float32
        power-of-two scales, ``(..., mn, k / 32)``.

    """
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    assert scale.dtype == torch.float32, scale.dtype
    *lead, mn, k = weight.shape
    block_m, block_k = mn // scale.shape[-2], k // scale.shape[-1]
    assert k % _SF_GRAN_K == 0, k
    assert tuple(scale.shape) == (*lead, mn // block_m, k // block_k) and (
        mn % block_m == 0 and k % block_k == 0
    ), (tuple(weight.shape), tuple(scale.shape))
    w = weight.reshape(-1, mn, k)
    s = scale.reshape(-1, scale.shape[-2], scale.shape[-1])
    out = torch.empty_like(w)
    sf = torch.empty(
        w.shape[0], mn, k // _SF_GRAN_K, dtype=torch.float32, device=w.device
    )
    for start in range(0, w.shape[0], chunk):
        end = min(start + chunk, w.shape[0])
        s_full = (
            s[start:end]
            .repeat_interleave(block_m, dim=1)
            .repeat_interleave(block_k, dim=2)
        )
        real = w[start:end].to(torch.float32) * s_full
        groups = real.view(end - start, mn, k // _SF_GRAN_K, _SF_GRAN_K)
        amax = groups.abs().amax(dim=-1).clamp_(min=1e-30)
        # Smallest power of two holding the group's largest value in E4M3.
        new_sf = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
        out[start:end] = ((groups / new_sf.unsqueeze(-1)).view(end - start, mn, k)).to(
            torch.float8_e4m3fn
        )
        sf[start:end] = new_sf
    return out.view(*lead, mn, k), sf.view(*lead, mn, k // _SF_GRAN_K)


class DeepGemmMegaMoEBackend:
    """Abstract MegaMoE backend that hides deep_gemm-specific details.

    A backend advertises its expected hidden-state quantization layout via
    ``hidden_quant`` and its expected MMA type via ``mma_type``. It is
    responsible for the routed-expert weight transform run at load time, the
    optional shared-expert weight transform, and the actual MegaMoE kernel
    invocation at inference time. Runtime capability checks live in
    :func:`get_deep_gemm_mega_moe_backend`: obtaining a backend already implies
    that the current device and shapes are supported.
    """

    # Human-readable MMA type tag used by deep_gemm helpers. SM100 default.
    mma_type: str = "fp8xfp4"
    hidden_quant: QuantKey = kMxfp8Dynamic

    def transform_weights(
        self,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        raise NotImplementedError

    def supports_shared_experts(self, deep_gemm) -> bool:
        return False

    def transform_shared_expert_weights(
        self,
        *,
        shared_experts: nn.Module,
        num_shared_experts: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str,
    ) -> (
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None
    ):
        """Return ``(l1, l2)`` transformed shared-expert weights, or ``None``.

        ``shared_experts`` must expose ``gate_up_proj`` and ``down_proj``
        linear layers. A ``None`` return means the backend cannot fuse shared
        experts for this module; the caller should fall back to a separate
        shared MLP.
        """
        return None

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

    def transform_weights(
        self,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        w13_scale = deep_gemm.transform_sf_into_required_layout(
            ue8m0_uint8_to_float(w13_weight_scale).contiguous(),
            2 * intermediate_size,
            hidden_size,
            (1, 32),
            num_local_experts,
        )
        w2_scale = deep_gemm.transform_sf_into_required_layout(
            ue8m0_uint8_to_float(w2_weight_scale).contiguous(),
            hidden_size,
            intermediate_size,
            (1, 32),
            num_local_experts,
        )
        kwargs = {} if activation is None else {"activation": activation}
        return deep_gemm.transform_weights_for_mega_moe(
            (w13_weight.view(torch.int8).contiguous(), w13_scale),
            (w2_weight.view(torch.int8).contiguous(), w2_scale),
            **kwargs,
        )

    def supports_shared_experts(self, deep_gemm) -> bool:
        """Check the Python API before touching a symmetric-memory group.

        This also gives users of an older precompiled vLLM wheel a safe serial
        fallback instead of failing halfway through multi-rank buffer setup.
        """
        try:
            buffer_params = signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters
            kernel_params = signature(deep_gemm.fp8_fp4_mega_moe).parameters
        except (TypeError, ValueError):
            return False
        return (
            hasattr(deep_gemm, "get_block_m_for_mega_moe")
            and hasattr(deep_gemm, "transform_weights_for_mega_moe")
            and "num_shared_experts" in buffer_params
            and "shared_l1_weights" in kernel_params
            and "shared_l2_weights" in kernel_params
        )

    def transform_shared_expert_weights(
        self,
        *,
        shared_experts: nn.Module,
        num_shared_experts: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str,
    ) -> (
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None
    ):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        gate_up = shared_experts.gate_up_proj
        down = shared_experts.down_proj
        gate_up_weight = gate_up.weight.data
        gate_up_scale = (
            gate_up.weight_scale
            if hasattr(gate_up, "weight_scale")
            else gate_up.weight_scale_inv
        ).data
        down_weight = down.weight.data
        down_scale = (
            down.weight_scale
            if hasattr(down, "weight_scale")
            else down.weight_scale_inv
        ).data

        # MegaMoE's shared FP8 MMA consumes a 1x32 scale for every weight row,
        # while the checkpoint uses coarser block-FP8 scales (usually
        # 128x128). Build a dedicated, numerically equivalent scale view.
        ue8m0_scale_dtypes = (torch.float8_e8m0fnu, torch.uint8, torch.int32)
        if (
            gate_up_scale.dtype in ue8m0_scale_dtypes
            and down_scale.dtype in ue8m0_scale_dtypes
        ):
            gate_up_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                gate_up,
                gate_up_scale,
                gate_up_weight.shape[0],
                gate_up_weight.shape[1],
                prefix,
            )
            down_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                down,
                down_scale,
                down_weight.shape[0],
                down_weight.shape[1],
                prefix,
            )

        if gate_up_scale is None or down_scale is None:
            return None

        shared_intermediate_size = intermediate_size * num_shared_experts
        expected_gate_up_shape = (
            2 * shared_intermediate_size,
            hidden_size,
        )
        expected_down_shape = (hidden_size, shared_intermediate_size)
        if (
            gate_up_weight.dtype != torch.float8_e4m3fn
            or down_weight.dtype != torch.float8_e4m3fn
            or gate_up_scale.dtype != torch.int32
            or down_scale.dtype != torch.int32
            or tuple(gate_up_weight.shape) != expected_gate_up_shape
            or tuple(down_weight.shape) != expected_down_shape
        ):
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: expected "
                "replicated block-FP8 weights with gate_up=%s, down=%s, and "
                "DeepGEMM int32 scales; got gate_up=%s/%s/%s and down=%s/%s/%s.",
                prefix,
                expected_gate_up_shape,
                expected_down_shape,
                tuple(gate_up_weight.shape),
                gate_up_weight.dtype,
                gate_up_scale.dtype,
                tuple(down_weight.shape),
                down_weight.dtype,
                down_scale.dtype,
            )
            return None

        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
            (gate_up_weight, gate_up_scale),
            (down_weight, down_scale),
        )
        # L1 interleaving allocates a full copy. Re-home the loader Parameter on
        # that storage so the original 2*intermediate*hidden FP8 tensor can be
        # released instead of adding roughly 0.7 GiB per rank on DSV4-Flash.
        # The generic linear post-load hook may still repack the serial scales,
        # but this shared MLP is never called after native fusion is enabled.
        gate_up.weight.data = transformed_l1[0]
        transformed_l1 = (gate_up.weight.data, transformed_l1[1])
        return transformed_l1, transformed_l2

    @staticmethod
    def _prepare_shared_expert_scale(
        deep_gemm,
        linear: nn.Module,
        scale: torch.Tensor,
        mn: int,
        k: int,
        prefix: str,
    ) -> torch.Tensor | None:
        block_size = getattr(linear, "weight_block_size", None)
        if block_size is None or len(block_size) != 2:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "shared FP8 weight block size is unavailable.",
                prefix,
            )
            return None

        block_m, block_k = block_size
        num_k_blocks = (k + block_k - 1) // block_k
        # The linear hook's DeepGEMM layout packs four k-block bytes per int32.
        packed = scale.dtype == torch.int32
        expected_shape = (
            (mn, (num_k_blocks + 3) // 4)
            if packed
            else ((mn + block_m - 1) // block_m, num_k_blocks)
        )
        if block_k % 32 != 0 or tuple(scale.shape) != expected_shape:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "cannot convert shared scale shape %s with block size %s "
                "to MegaMoE's 1x32 layout for weight (%d, %d).",
                prefix,
                tuple(scale.shape),
                tuple(block_size),
                mn,
                k,
            )
            return None

        if packed:
            row_scale = scale.flatten().view(torch.uint8).view(mn, -1)[:, :num_k_blocks]
        else:
            row_scale = scale.view(torch.uint8).repeat_interleave(block_m, dim=0)[:mn]
        scale_1x32 = (
            ue8m0_uint8_to_float(row_scale)
            .repeat_interleave(block_k // 32, dim=1)[:, : k // 32]
            .contiguous()
        )
        # The grouped API is used with a singleton dimension to request the
        # MN-major, TMA-aligned packed-UE8M0 strides, then squeezed back to the
        # 2D layout required for a shared expert.
        return deep_gemm.transform_sf_into_required_layout(
            scale_1x32.unsqueeze(0),
            mn,
            k,
            (1, 32),
            1,
        ).squeeze(0)

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


class DeepGemmSm100Fp8MegaMoEBackend(DeepGemmSm100MegaMoEBackend):
    """SM100 MegaMoE over block-FP8 experts (``fp8_fp4_mega_moe`` with E4M3
    weights). Routed and shared expert weights arrive with float32 block scales
    and are requantized to the kernel's UE8M0 1x32 scales."""

    mma_type = "fp8xfp8"

    @staticmethod
    def _to_kernel_format(
        deep_gemm, weight: torch.Tensor, scale: torch.Tensor, num_groups: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight, sf = requant_block_fp8_to_ue8m0(weight, scale)
        mn, k = weight.shape[-2:]
        sf = deep_gemm.transform_sf_into_required_layout(
            sf.contiguous(), mn, k, (1, _SF_GRAN_K), num_groups
        )
        return weight, sf

    def transform_weights(
        self,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        kwargs = {} if activation is None else {"activation": activation}
        return deep_gemm.transform_weights_for_mega_moe(
            self._to_kernel_format(
                deep_gemm, w13_weight, w13_weight_scale, num_local_experts
            ),
            self._to_kernel_format(
                deep_gemm, w2_weight, w2_weight_scale, num_local_experts
            ),
            **kwargs,
        )

    def transform_shared_expert_weights(
        self,
        *,
        shared_experts: nn.Module,
        num_shared_experts: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str,
    ) -> (
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None
    ):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        shared_size = intermediate_size * num_shared_experts
        pairs = []
        for linear, shape in (
            (shared_experts.gate_up_proj, (2 * shared_size, hidden_size)),
            (shared_experts.down_proj, (hidden_size, shared_size)),
        ):
            weight = linear.weight
            scale = getattr(linear, "weight_scale_inv", None)
            if scale is None:
                scale = getattr(linear, "weight_scale", None)
            if (
                scale is None
                or weight.dtype != torch.float8_e4m3fn
                or scale.dtype != torch.float32
                or tuple(weight.shape) != shape
            ):
                logger.warning(
                    "Disabling native MegaMoE shared-expert fusion for %s: "
                    "expected replicated block-FP8 weights of shape %s with "
                    "float32 block scales.",
                    prefix,
                    shape,
                )
                return None
            # Requantized copies: the shared MLP keeps its own storage, which
            # the generic post-load hooks may still rewrite.
            w, sf = self._to_kernel_format(
                deep_gemm, weight.data.unsqueeze(0), scale.data.unsqueeze(0), 1
            )
            pairs.append((w.squeeze(0), sf.squeeze(0)))
        return deep_gemm.transform_weights_for_mega_moe(*pairs)


def get_deep_gemm_mega_moe_backend(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
    mma_type: str = "fp8xfp4",
) -> DeepGemmMegaMoEBackend:
    """Return the DeepGEMM MegaMoE backend for ``device``.

    ``mma_type`` selects by expert weight format: ``fp8xfp4`` for MXFP4
    experts, ``fp8xfp8`` for block-FP8 experts.

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
    if mma_type == "fp8xfp8":
        return DeepGemmSm100Fp8MegaMoEBackend()
    if mma_type != "fp8xfp4":
        raise ValueError(f"Unsupported DeepGEMM MegaMoE mma_type: {mma_type!r}")
    return DeepGemmSm100MegaMoEBackend()
