# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.util import find_spec
from typing import Final, cast

import torch

from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_CONCH_SUPPORTED_WEIGHT_TYPES: Final = [
    scalar_types.uint4,
    scalar_types.uint8,
    scalar_types.uint4b8,
    scalar_types.uint8b128,
]
_CONCH_SUPPORTED_GROUP_SIZES: Final = [-1, 128]


def _conch_padded_width(N: int) -> int:
    """Columns added to reach conch's block_n tile width (0 when aligned)."""
    block_n, _ = _conch_tile_sizes()
    return (-N) % block_n


def _conch_tile_sizes() -> tuple[int, int]:
    """Returns conch's (block_n, block_k), falling back to (64, 64).

    Imported lazily (conch is optional) and must match what conch passes to
    _gemm_kernel; the tuning helper is private, hence the fallback.
    """
    try:
        from conch.kernels.quantization.gemm import _get_tuning_parameters

        tuning = _get_tuning_parameters()
        return int(tuning["cxpr_block_size_n"]), int(tuning["cxpr_block_size_k"])
    except Exception:
        return 64, 64


class ConchLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_type not in _CONCH_SUPPORTED_WEIGHT_TYPES:
            error_msg = (
                f"Weight type ({c.weight_type}) not supported by "
                "ConchLinearKernel, supported types are: "
                f"{_CONCH_SUPPORTED_WEIGHT_TYPES}"
            )
            return False, error_msg

        if c.group_size not in _CONCH_SUPPORTED_GROUP_SIZES:
            error_msg = (
                f"Group size ({c.group_size}) not supported by "
                "ConchLinearKernel, supported group sizes are: "
                f"{_CONCH_SUPPORTED_GROUP_SIZES}"
            )
            return False, error_msg

        if find_spec("conch") is None:
            error_msg = (
                "conch-triton-kernels is not installed, please "
                "install it via `pip install conch-triton-kernels` "
                "and try again!"
            )
            return False, error_msg

        # conch's B and scale loads are unmasked, so an unaddressable K cannot be
        # padded away (N can) and is a hard reject.
        K = c.partition_weight_shape[0]
        gs = K if c.group_size == -1 else c.group_size
        block_n, block_k = _conch_tile_sizes()
        if K % block_k != 0:
            error_msg = f"K ({K}) must be divisible by conch's block_k ({block_k})"
            return False, error_msg
        if K % gs != 0:
            error_msg = f"K ({K}) must be divisible by the group size ({gs})"
            return False, error_msg
        if gs % block_k != 0:
            error_msg = (
                f"group size ({gs}) must be divisible by conch's block_k ({block_k})"
            )
            return False, error_msg

        # The pad is copied from the unpacked launch-layout weight, which is
        # ceil-packed along N, so a partial packing group cannot be padded.
        N = c.partition_weight_shape[1]
        pack_factor = 32 // c.weight_type.size_bits
        if N % pack_factor != 0:
            error_msg = (
                f"N ({N}) must be divisible by the {pack_factor}-column packing "
                "group, otherwise the unpacked weight is wider than N"
            )
            return False, error_msg

        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    #  `weight_zero_point` is: {input_dim = 1, output_dim = 0, packed_dim = 0}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The pads below are derived state a reload cannot refresh: it re-registers
        # the pre-reload buffers, so a reloaded padded layer would serve stale
        # weights. A second pass over a padded layer therefore raises. The marker is
        # a plain attribute on purpose: the reload deletes registered tensors, and
        # its metadata predates the pads, so the attribute is the only thing left.
        N = self.config.partition_weight_shape[1]
        pad_width = _conch_padded_width(N)
        if pad_width and getattr(layer, "_conch_padded_once", False):
            raise NotImplementedError(
                f"conch W4A16 pads this layer's output width from N={N} to "
                f"{N + pad_width} to satisfy its 64-column tile, and that "
                "padded copy cannot be refreshed by a weight reload (the reload "
                "re-registers the pre-reload buffers). Reload with "
                "--linear-backend triton, or use a checkpoint whose output width "
                "is a multiple of conch's tile."
            )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = x.data.contiguous()
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        def transform_w_zp(x):
            # Zero points are stored PACKED as [N//pack_factor, K//G]
            # The Conch kernel expects UNPACKED zeros: [K//G, N]
            # We need to unpack and reorder
            assert isinstance(x, BasevLLMParameter)
            packed = x.data  # shape: [N//pack_factor, K//G], dtype: int32

            # Determine packing based on weight bit width
            size_bits = self.config.weight_type.size_bits
            pack_factor = 32 // size_bits  # 8 for 4-bit, 4 for 8-bit
            mask = (1 << size_bits) - 1  # 0xF for 4-bit, 0xFF for 8-bit

            n_packed, k_groups = packed.shape
            n_full = n_packed * pack_factor

            # Unpack using vectorized bitwise ops
            # shifts = [0, size_bits, 2*size_bits, ...] for each packed position
            shifts = torch.arange(
                0, 32, size_bits, dtype=torch.int32, device=packed.device
            )
            # packed: [N//pack_factor, K//G] -> [N//pack_factor, K//G, 1]
            # shifts: [pack_factor] -> [1, 1, pack_factor]
            # Result: [N//pack_factor, K//G, pack_factor]
            unpacked = (packed.unsqueeze(-1) >> shifts) & mask

            # Permute to [K//G, N//pack_factor, pack_factor] then reshape to [K//G, N]
            unpacked = unpacked.permute(1, 0, 2).reshape(k_groups, n_full)

            x.data = unpacked.to(torch.uint8).contiguous()

            # Update metadata - zeros are no longer packed
            if hasattr(x, "_input_dim"):
                x._input_dim = 0
            if hasattr(x, "_output_dim"):
                x._output_dim = 1
            if hasattr(x, "_packed_factor"):
                x._packed_factor = 1
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if self.config.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        elif self.w_zp_name is not None:
            layer.register_parameter(self.w_zp_name, None)

        # conch's B/scale loads are unmasked, so pad N up to the tile width.
        # Non-persistent, one tensor-or-None buffer per tensor (Module._apply
        # passes buffers through fn, so a tuple would break Module.to()): the pads
        # stay out of state_dict() and keep a stable address for CUDA graph replay,
        # which is also why a reload cannot refresh them (see the guard above).
        N = self.config.partition_weight_shape[1]
        pad = _conch_padded_width(N)
        if pad == 0:
            layer.register_buffer("_conch_padded_w_q", None, persistent=False)
            layer.register_buffer("_conch_padded_w_s", None, persistent=False)
            layer.register_buffer("_conch_padded_w_zp", None, persistent=False)
        else:
            w_q, w_s, w_zp = self._get_weight_params(layer)

            def pad_cols(t: torch.Tensor) -> torch.Tensor:
                out = t.new_zeros(t.shape[0], N + pad)
                out[:, :N] = t
                return out

            layer.register_buffer("_conch_padded_w_q", pad_cols(w_q), persistent=False)
            layer.register_buffer("_conch_padded_w_s", pad_cols(w_s), persistent=False)
            layer.register_buffer(
                "_conch_padded_w_zp",
                pad_cols(w_zp) if w_zp is not None else None,
                persistent=False,
            )
        # Survives the reload path's tensor deletion; guards the second pass.
        layer._conch_padded_once = pad != 0

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from conch.ops.quantization.gemm import mixed_precision_gemm

        w_q, w_s, w_zp = self._get_weight_params(layer)
        padded_w_q = getattr(layer, "_conch_padded_w_q", None)
        if padded_w_q is not None:
            w_q = padded_w_q
            w_s = cast(torch.Tensor, getattr(layer, "_conch_padded_w_s", None))
            w_zp = getattr(layer, "_conch_padded_w_zp", None)
        else:
            w_q = w_q.data
            w_s = w_s.data
            w_zp = w_zp.data if w_zp is not None else None

        # Map channelwise group_size=-1 to the actual input dimension K.
        # The conch kernel computes stride_mul = block_k / group_size;
        # passing -1 produces a negative stride that reads out-of-bounds
        # scale values for all K-blocks after the first.
        group_size = self.config.group_size
        if group_size == -1:
            group_size = x.shape[-1]

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.config.partition_weight_shape[1],)

        output = mixed_precision_gemm(
            x=x_2d,
            w_q_packed=w_q,
            w_s=w_s,
            w_zp=w_zp,
            weight_size_bits=self.config.weight_type.size_bits,
            weight_bias=self.config.weight_type.bias,
            group_size=group_size,
        )

        if padded_w_q is not None:
            # Drop the zero-padded tail columns before bias add and reshape.
            output = output[:, : self.config.partition_weight_shape[1]]

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
