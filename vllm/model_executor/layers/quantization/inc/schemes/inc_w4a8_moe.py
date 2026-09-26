# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
from vllm.model_executor.utils import replace_parameter

from .inc_wna16_moe import INCARKWNA16MoEMethod

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig

logger = init_logger(__name__)

W4A8_MOE_SYMBOLS = ("moe_w4a8_prepack", "moe_gemm_w4a8")
W4A16_DECODE_SYMBOLS = ("moe_gemm_decode",)
_MOE_AUTO_DECODE_MAX_TOTAL_TOKENS = 128
_EAGER_PREPACK_VALUES = {"1", "true", "yes", "on"}
_EAGER_PREPACK_FORCE_VALUES = {"force", "always"}


def _has_ark_symbols(is_ark_available: bool, ark, symbols: tuple[str, ...]) -> bool:
    xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
    return (
        is_ark_available
        and ark is not None
        and xpu_lib is not None
        and all(hasattr(ark, symbol) and hasattr(xpu_lib, symbol) for symbol in symbols)
    )


def has_ark_w4a8_moe_kernel(is_ark_available: bool, ark) -> bool:
    return _has_ark_symbols(
        is_ark_available,
        ark,
        W4A8_MOE_SYMBOLS + W4A16_DECODE_SYMBOLS,
    )


def _tensor_parallel_world_size() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except (AssertionError, RuntimeError, ValueError):
        return 1


def _w4a8_eager_prepack_enabled() -> bool:
    value = os.environ.get("ARK_MOE_W4A8_EAGER_PREPACK", "")
    normalized = value.strip().lower()
    if normalized in _EAGER_PREPACK_FORCE_VALUES:
        return True
    if normalized not in _EAGER_PREPACK_VALUES:
        return False

    if _tensor_parallel_world_size() > 1:
        return True

    logger.info_once(
        "ARK W4A8 MoE eager prepack is disabled for "
        "tensor_parallel_size=1; use "
        "ARK_MOE_W4A8_EAGER_PREPACK=force to override."
    )
    return False


def _moe_auto_decode_max_total_tokens() -> int:
    value = os.environ.get("ARK_MOE_AUTO_DECODE_MAX_TOKENS")
    if value is None:
        return _MOE_AUTO_DECODE_MAX_TOTAL_TOKENS
    try:
        threshold = int(value.strip())
    except ValueError:
        return _MOE_AUTO_DECODE_MAX_TOTAL_TOKENS
    if threshold <= 0:
        return _MOE_AUTO_DECODE_MAX_TOTAL_TOKENS
    return threshold


def _first_attention_metadata():
    if not is_forward_context_available():
        return None

    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):
        entries = attn_metadata
    else:
        entries = [attn_metadata]

    for entry in entries:
        if isinstance(entry, dict):
            for metadata in entry.values():
                if metadata is not None:
                    return metadata
        elif entry is not None:
            return entry
    return None


def _prefill_decode_split(num_rows: int) -> tuple[int, int] | None:
    metadata = _first_attention_metadata()
    if metadata is None:
        return None

    num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
    num_prefill_tokens = getattr(metadata, "num_prefill_tokens", None)
    if num_decode_tokens is None or num_prefill_tokens is None:
        return None

    num_decode_tokens = int(num_decode_tokens)
    num_prefill_tokens = int(num_prefill_tokens)
    if num_decode_tokens < 0 or num_prefill_tokens < 0:
        return None
    if num_decode_tokens + num_prefill_tokens != num_rows:
        return None
    return num_decode_tokens, num_prefill_tokens


def _effective_moe_group_size(
    layer_config: "INCLayerConfig",
    hidden_size: int,
    intermediate_size: int,
    prefix: str,
) -> int:
    group_size = layer_config.group_size
    while intermediate_size % group_size or hidden_size % group_size:
        group_size //= 2
        if group_size < 32:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires hidden and "
                "intermediate sizes divisible by a derived group size of at "
                f"least 32. Layer: {prefix}."
            )
    return group_size


def check_xpu_moe_w4a8_supported(
    layer: "torch.nn.Module",
    layer_config: "INCLayerConfig",
    prefix: str,
) -> int:
    moe_config = layer.moe_config
    hidden_size = moe_config.hidden_dim
    intermediate_size = moe_config.intermediate_size_per_partition
    group_size = _effective_moe_group_size(
        layer_config,
        hidden_size,
        intermediate_size,
        prefix,
    )

    if group_size <= 0 or group_size % 8 != 0:
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires a MoE group size that "
            f"is a positive multiple of 8, got {group_size}. Layer: {prefix}."
        )

    gemm_shapes = (
        (
            "w13",
            moe_config.w13_num_shards * intermediate_size,
            hidden_size,
        ),
        ("w2", hidden_size, intermediate_size),
    )
    for name, out_features, in_features in gemm_shapes:
        if out_features % 16 != 0 or in_features % 64 != 0:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires MoE GEMM shapes "
                "with N multiple of 16 and K multiple of 64, got "
                f"{name}: N={out_features}, K={in_features}. Layer: {prefix}."
            )
        if in_features % group_size != 0:
            raise NotImplementedError(
                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires K to be divisible "
                f"by group_size, got {name}: K={in_features}, "
                f"group_size={group_size}. Layer: {prefix}."
            )
    return group_size


class INCARKW4A8MoEMethod(INCARKWNA16MoEMethod):
    kernel_name = "W4A8"
    log_message = "Selected ARK XPU W4A8 prefill/W4A16 decode MoE method."
    prefill_kernel_log_message = "Using ARK XPU W4A8 MoE kernel for prefill."
    decode_kernel_log_message = "Using ARK XPU W4A16 MoE kernel for decode."
    _eager_prepack_layer_count = 0

    def __init__(
        self,
        quant_config: MoeWNA16Config,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(quant_config, moe)
        self.w13_moe_w4a8: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.w2_moe_w4a8: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.w13_moe_w4a16: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.w2_moe_w4a16: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.w13_moe_w4a8_prepacked: tuple[
            torch.Tensor, torch.Tensor, int
        ] | None = None
        self.w2_moe_w4a8_prepacked: tuple[
            torch.Tensor, torch.Tensor, int
        ] | None = None
        self._use_w4a8_for_current_apply = False

    def _has_ark_moe_kernel(self, is_available, ark, xpu_lib) -> bool:
        del xpu_lib
        return has_ark_w4a8_moe_kernel(is_available, ark)

    @staticmethod
    def _signed_w4a8_qweight(qweight: torch.Tensor) -> torch.Tensor:
        qweight = qweight.detach()
        if qweight.is_contiguous():
            qweight.bitwise_xor_(0x88)
            return qweight
        return torch.bitwise_xor(qweight.contiguous(), 0x88)

    @staticmethod
    def _contiguous_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if tensor.is_contiguous():
            return tensor
        return tensor.contiguous()

    def _make_moe_weight(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        signed_qweight = self._signed_w4a8_qweight(qweight)
        contiguous_scales = self._contiguous_tensor(scales)
        return signed_qweight, contiguous_scales, group_size

    def process_weights_after_loading(self, layer) -> None:
        self._setup_common_moe_state(layer)

        w13_qweight, w13_scales, group_size = self._make_moe_weight(
            layer.w13_qweight,
            layer.w13_scales,
            layer.group_size,
        )
        replace_parameter(layer, "w13_qweight", w13_qweight)
        replace_parameter(layer, "w13_scales", w13_scales)

        w2_qweight, w2_scales, _ = self._make_moe_weight(
            layer.w2_qweight,
            layer.w2_scales,
            layer.group_size,
        )
        replace_parameter(layer, "w2_qweight", w2_qweight)
        replace_parameter(layer, "w2_scales", w2_scales)

        w13_packed = (layer.w13_qweight, layer.w13_scales, group_size)
        w2_packed = (layer.w2_qweight, layer.w2_scales, group_size)
        self.w13_moe_w4a8 = w13_packed
        self.w2_moe_w4a8 = w2_packed
        self.w13_moe_w4a16 = w13_packed
        self.w2_moe_w4a16 = w2_packed

        if _w4a8_eager_prepack_enabled():
            type(self)._eager_prepack_layer_count += 1
            layer_count = type(self)._eager_prepack_layer_count
            logger.info(
                "ARK W4A8 MoE eager prepack started for layer %d: "
                "w13=%s, w2=%s",
                layer_count,
                tuple(w13_packed[0].shape),
                tuple(w2_packed[0].shape),
            )
            self.w13_moe_w4a8_prepacked = self._prepack_w4a8_moe_weight(
                w13_packed
            )
            self.w2_moe_w4a8_prepacked = self._prepack_w4a8_moe_weight(w2_packed)
            logger.info(
                "ARK W4A8 MoE eager prepack finished for layer %d",
                layer_count,
            )

    def _check_moe_weights_loaded(self) -> None:
        assert self.w13_moe_w4a8 is not None
        assert self.w2_moe_w4a8 is not None
        assert self.w13_moe_w4a16 is not None
        assert self.w2_moe_w4a16 is not None

    def _prepack_w4a8_moe_weight(
        self,
        packed: tuple[torch.Tensor, torch.Tensor, int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        qweight, scales, group_size = packed
        return self.ark.moe_w4a8_prepack(
            qweight,
            scales,
            group_size=group_size,
        )

    def _apply_w4a8_moe_prefill(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
        packed: tuple[torch.Tensor, torch.Tensor, int],
        prepacked: tuple[torch.Tensor, torch.Tensor, int] | None,
    ) -> torch.Tensor:
        logger.info_once(self.prefill_kernel_log_message)
        if prepacked is None:
            prepacked = self._prepack_w4a8_moe_weight(packed)
        weights_s8, wscales, block = prepacked
        return self.ark.moe_gemm_w4a8(
            x,
            weights_s8,
            wscales,
            rows_per_expert,
            rescale_block_size=block,
            phase="prefill",
        )

    def _apply_w4a16_moe_decode(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
        packed: tuple[torch.Tensor, torch.Tensor, int],
    ) -> torch.Tensor:
        logger.info_once(self.decode_kernel_log_message)
        qweight, scales, group_size = packed
        return self.ark.moe_gemm_decode(
            x,
            qweight,
            rows_per_expert,
            scales=scales,
            weight_bits=4,
            group_size=group_size,
        )

    def _apply_w13_moe(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_w4a8_for_current_apply:
            assert self.w13_moe_w4a8 is not None
            return self._apply_w4a8_moe_prefill(
                x,
                rows_per_expert,
                self.w13_moe_w4a8,
                self.w13_moe_w4a8_prepacked,
            )
        assert self.w13_moe_w4a16 is not None
        return self._apply_w4a16_moe_decode(x, rows_per_expert, self.w13_moe_w4a16)

    def _apply_w2_moe(
        self,
        x: torch.Tensor,
        rows_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_w4a8_for_current_apply:
            assert self.w2_moe_w4a8 is not None
            return self._apply_w4a8_moe_prefill(
                x,
                rows_per_expert,
                self.w2_moe_w4a8,
                self.w2_moe_w4a8_prepacked,
            )
        assert self.w2_moe_w4a16 is not None
        return self._apply_w4a16_moe_decode(x, rows_per_expert, self.w2_moe_w4a16)

    def _apply_with_prefill_kernel(
        self,
        use_w4a8_prefill: bool,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        previous = self._use_w4a8_for_current_apply
        self._use_w4a8_for_current_apply = use_w4a8_prefill
        try:
            return super().apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        finally:
            self._use_w4a8_for_current_apply = previous

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        split = _prefill_decode_split(x.shape[0])
        if split is None:
            use_w4a8_prefill = (
                x.shape[0] * topk_ids.shape[1]
                > _moe_auto_decode_max_total_tokens()
            )
            return self._apply_with_prefill_kernel(
                use_w4a8_prefill,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

        num_decode_tokens, num_prefill_tokens = split
        if num_decode_tokens == 0:
            return self._apply_with_prefill_kernel(
                True,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        if num_prefill_tokens == 0:
            return self._apply_with_prefill_kernel(
                False,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

        output = torch.empty_like(x)
        decode_output = self._apply_with_prefill_kernel(
            False,
            layer,
            x[:num_decode_tokens],
            topk_weights[:num_decode_tokens],
            topk_ids[:num_decode_tokens],
            shared_experts,
            shared_experts_input,
        )
        output[:num_decode_tokens].copy_(decode_output)
        del decode_output

        prefill_output = self._apply_with_prefill_kernel(
            True,
            layer,
            x[num_decode_tokens:],
            topk_weights[num_decode_tokens:],
            topk_ids[num_decode_tokens:],
            shared_experts,
            shared_experts_input,
        )
        output[num_decode_tokens:].copy_(prefill_output)
        del prefill_output
        return output
