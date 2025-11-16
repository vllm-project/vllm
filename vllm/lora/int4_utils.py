"""
INT4 Unpacking Utilities for LoRA Compatibility in vLLM.

This module provides utilities to unpack INT4 quantized weights to floating-point
format, enabling LoRA adapter injection on compressed models.
"""


import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["INT4Unpacker", "get_unpacker"]


class INT4Unpacker:
    """
    Manages unpacking and caching of INT4 weights for LoRA compatibility.

    This class handles the conversion of packed INT4 weights (stored as uint8)
    back to floating-point tensors that can be used with LoRA adapters.
    """

    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def unpack_int4_weights(
        self,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None = None,
        group_size: int | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Unpack INT4 quantized weights to floating-point format.

        INT4 weights are stored with 2 values per byte in a uint8 tensor.
        This function unpacks them and dequantizes using provided scales
        and zero points.

        Args:
            packed_weights: Packed INT4 weights as uint8,
                shape [out_features, in_features // 2]
            scales: Quantization scales
                - Per-tensor: shape [1]
                - Per-channel: shape [out_features]
                - Grouped: shape [out_features, num_groups]
            zero_points: Optional zero points for asymmetric quantization
            group_size: Group size for grouped quantization (e.g., 128)
            output_dtype: Output dtype (default: torch.float16)

        Returns:
            Unpacked and dequantized weights with shape [out_features, in_features]
        """
        if packed_weights.dtype != torch.uint8:
            raise ValueError(
                f"packed_weights must be uint8, got {packed_weights.dtype}"
            )

        out_features, packed_in_features = packed_weights.shape
        in_features = packed_in_features * 2

        # Unpack: extract two INT4 values from each uint8 byte
        # Lower 4 bits: value & 0x0F (even indices)
        # Upper 4 bits: (value >> 4) & 0x0F (odd indices)
        unpacked = torch.zeros(
            (out_features, in_features),
            dtype=torch.uint8,
            device=packed_weights.device,
        )
        unpacked[:, 0::2] = packed_weights & 0x0F
        unpacked[:, 1::2] = (packed_weights >> 4) & 0x0F

        # Convert to signed INT4 range: [0, 15] -> [-8, 7]
        unpacked_signed = unpacked.to(torch.int8) - 8

        # Convert to floating point
        unpacked_fp = unpacked_signed.to(output_dtype)

        # Apply zero points (for asymmetric quantization)
        if zero_points is not None:
            if zero_points.numel() == 1:
                # Per-tensor zero point
                unpacked_fp = unpacked_fp - zero_points.to(output_dtype)
            elif zero_points.shape[0] == out_features and zero_points.ndim == 1:
                # Per-channel zero point
                unpacked_fp = unpacked_fp - zero_points.view(-1, 1).to(output_dtype)
            elif zero_points.ndim == 2:
                # Grouped zero point
                if group_size is None:
                    raise ValueError(
                        "group_size must be provided for grouped zero points"
                    )
                zp_expanded = zero_points.unsqueeze(2).repeat(1, 1, group_size)
                zp_flat = zp_expanded.view(out_features, -1)[:, :in_features].to(
                    output_dtype
                )
                unpacked_fp = unpacked_fp - zp_flat

        # Apply scales
        if scales.numel() == 1:
            # Per-tensor scale
            unpacked_fp = unpacked_fp * scales.to(output_dtype)
        elif scales.shape[0] == out_features and scales.ndim == 1:
            # Per-channel scale
            unpacked_fp = unpacked_fp * scales.view(-1, 1).to(output_dtype)
        elif scales.ndim == 2:
            # Grouped scale
            if group_size is None:
                raise ValueError(
                    "group_size must be provided for grouped quantization"
                )
            scales_expanded = scales.unsqueeze(2).repeat(1, 1, group_size)
            scales_flat = scales_expanded.view(out_features, -1)[:, :in_features].to(
                output_dtype
            )
            unpacked_fp = unpacked_fp * scales_flat
        else:
            raise ValueError(f"Unsupported scales shape: {scales.shape}")

        return unpacked_fp

    def unpack_module(
        self,
        module: torch.nn.Module,
        module_name: str,
        force: bool = False,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor | None:
        """
        Unpack INT4 weights from a module, with caching.

        Args:
            module: PyTorch module with packed weights
            module_name: Unique name for caching
            force: If True, bypass cache and re-unpack
            output_dtype: Output dtype for unpacked weights

        Returns:
            Unpacked FP16 weights, or None if module is not quantized
        """
        # Check cache first
        if not force and module_name in self._cache:
            self._cache_hits += 1
            logger.debug("Cache hit for %s", module_name)
            return self._cache[module_name]

        self._cache_misses += 1

        # Check if module has packed weights
        # compressed-tensors can use either 'weight_packed'
        # or 'weight' (when compressed)
        packed_weights = None
        if hasattr(module, "weight_packed"):
            packed_weights = module.weight_packed
        elif hasattr(module, "weight") and module.weight.dtype == torch.uint8:
            packed_weights = module.weight
        else:
            logger.debug(
                "Module %s does not have packed INT4 weights", module_name
            )
            return None

        # Get quantization parameters
        scales = getattr(module, "weight_scale", None)
        zero_points = getattr(module, "weight_zero_point", None)

        if scales is None:
            logger.warning(
                "Module %s missing weight_scale for dequantization", module_name
            )
            return None

        # Infer group size from scales shape
        group_size = None
        if scales.ndim == 2:
            out_features, num_groups = scales.shape
            in_features_packed = packed_weights.shape[1]
            in_features = in_features_packed * 2
            group_size = in_features // num_groups
            logger.debug(
                "Inferred group_size=%d from scales shape %s",
                group_size,
                scales.shape,
            )

        try:
            unpacked = self.unpack_int4_weights(
                packed_weights=packed_weights,
                scales=scales,
                zero_points=zero_points,
                group_size=group_size,
                output_dtype=output_dtype,
            )

            # Cache the result
            self._cache[module_name] = unpacked
            logger.info(
                "Unpacked and cached INT4 weights for %s: %s -> %s",
                module_name,
                packed_weights.shape,
                unpacked.shape,
            )

            return unpacked

        except Exception as e:
            logger.error(
                "Failed to unpack INT4 weights for %s: %s", module_name, e
            )
            return None

    def is_int4_quantized(self, module: torch.nn.Module) -> bool:
        """
        Check if a module has INT4 quantized weights.

        Args:
            module: PyTorch module to check

        Returns:
            True if module has packed INT4 weights
        """
        has_packed = hasattr(module, "weight_packed") or (
            hasattr(module, "weight")
            and hasattr(module.weight, "dtype")
            and module.weight.dtype == torch.uint8
        )

        has_scales = hasattr(module, "weight_scale")

        return has_packed and has_scales

    def clear_cache(self):
        """Clear the unpacked weights cache to free memory."""
        num_entries = len(self._cache)
        self._cache.clear()
        logger.info(
            "Cleared INT4 unpacking cache (%d entries). "
            "Cache stats - hits: %d, misses: %d",
            num_entries,
            self._cache_hits,
            self._cache_misses,
        )
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
        }


# Global unpacker instance
_global_unpacker: INT4Unpacker | None = None


def get_unpacker() -> INT4Unpacker:
    """
    Get the global INT4 unpacker instance.

    Returns:
        The global INT4Unpacker instance (creates one if it doesn't exist)
    """
    global _global_unpacker
    if _global_unpacker is None:
        _global_unpacker = INT4Unpacker()
        logger.info("Initialized global INT4 unpacker")
    return _global_unpacker
