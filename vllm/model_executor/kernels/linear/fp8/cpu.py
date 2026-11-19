# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _cpu_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from .base import (
    Fp8LinearKernel,
)
from .block_base import (
    Fp8BlockLinearKernel,
    Fp8LinearLayerConfig,
)

# BLOCK_K in csrc/cpu/sgl-kernels/gemm.h — the AMX kernel's fixed K-block size.
_BLOCK_SIZE_K = 128


class CPUFp8BlockLinearKernel(Fp8BlockLinearKernel):
    """FP8 W8A16 block-quantized GEMM via AMX BRGEMM on CPU."""

    # Input stays BF16 — no FP8 activation quantization.
    apply_input_quant = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU platform."
        if not torch.cpu._is_amx_tile_supported():
            return False, "requires AMX tile support (Sapphire Rapids or newer)."
        if not ops._supports_cpu_fp8_w8a16:
            return False, "fp8_scaled_mm_cpu op not available."
        return True, None

    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return None

    @classmethod
    def can_implement(cls, config: Fp8LinearLayerConfig) -> tuple[bool, str | None]:
        # Validate weight block shape
        weight_gs = config.weight_quant_key.scale.group_shape
        if weight_gs.col <= 0 or weight_gs.col != 128:
            return False, (
                "CPU FP8 kernel requires K-dimension block size of 128, "
                f"got {weight_gs.col}."
            )
        if weight_gs.row <= 0 or weight_gs.row % 32 != 0:
            return False, (
                "CPU FP8 kernel requires N-dimension block size to be "
                f"a positive multiple of 32, got {weight_gs.row}."
            )
        if config.out_dtype not in (torch.bfloat16, torch.float32):
            return False, "Only bfloat16/float32 output dtype supported."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_cpu_skip_gemm_dispatch", False):
            return
        # Skip the base class process (FP8 padding / fnuz normalization)
        # which is GPU-oriented.  Instead, VNNI-prepack weights for AMX.
        params = self._get_layer_params(layer)

        # MLA's kv_b_proj (see `_cpu_skip_gemm_dispatch`): MLA reads this
        # weight to derive W_UK/W_UV, so keep the checkpoint [N, K] layout and
        # let the kernel pack on the fly instead.
        if not getattr(layer, "_cpu_skip_gemm_dispatch", False):
            packed_weight = torch.ops._C.convert_weight_packed(params.weight)
            replace_parameter(
                layer,
                params.WEIGHT,
                torch.nn.Parameter(packed_weight, requires_grad=False),
            )

        # Re-wrap scale as a plain Parameter so the kernel can read it
        # without weight-loader metadata interfering.
        scale_attr = (
            params.WEIGHT_SCALE_INV
            if params.weight_scale_inv is not None
            else params.WEIGHT_SCALE
        )
        weight_scale = (
            params.weight_scale_inv
            if params.weight_scale_inv is not None
            else params.weight_scale
        )
        assert weight_scale is not None
        if weight_scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
            weight_scale = _upcast_e8m0_to_fp32(weight_scale.data)
        replace_parameter(
            layer,
            scale_attr,
            torch.nn.Parameter(weight_scale.data.contiguous(), requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale_inv
            if params.weight_scale_inv is not None
            else params.weight_scale
        )

        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        out = torch.ops._C.fp8_scaled_mm_cpu(
            x_2d,
            params.weight,
            weight_scale,
            list(self.weight_group_shape),
            bias,
            x.dtype,
            not getattr(layer, "_cpu_skip_gemm_dispatch", False),  # is_vnni
        )
        return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CPUFp8BlockLinearKernel overrides apply_weights directly."
        )


class CPUFp8PerTensorLinearKernel(Fp8LinearKernel):
    """FP8 W8A16 per-tensor-scaled GEMM via AMX BRGEMM on CPU.

    Reuses the block-scaled AMX kernel (fp8_scaled_mm_cpu) with a single
    synthetic block spanning the whole weight, so activations stay BF16/FP32
    — no FP8 activation quantization, unlike PerTensorTorchFp8LinearKernel.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU platform."
        if not torch.cpu._is_amx_tile_supported():
            return False, "requires AMX tile support (Sapphire Rapids or newer)."
        if not ops._supports_cpu_fp8_w8a16:
            return False, "fp8_scaled_mm_cpu op not available."
        return True, None

    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return None

    @classmethod
    def can_implement(cls, c: Fp8LinearLayerConfig) -> tuple[bool, str | None]:
        if not c.weight_quant_key.scale.group_shape.is_per_tensor():
            return False, "requires a per-tensor weight scale."
        if not c.activation_quant_key.scale.group_shape.is_per_tensor():
            return False, "requires a per-tensor activation scale."
        if c.out_dtype not in (torch.bfloat16, torch.float32):
            return False, "Only bfloat16/float32 output dtype supported."
        n = c.weight_shape[0]
        if n % 32 != 0:
            # The AMX tinygemm kernel tiles N in chunks of 32 and cannot
            # handle a remainder tile smaller than that.
            return False, f"requires weight output dim (N={n}) to be a multiple of 32."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_name, w_s_name, _, _ = self.layer_param_names
        # Fp8LinearMethod transposes weight to (K, N) for torch._scaled_mm-style
        # kernels; convert_weight_packed/fp8_scaled_mm_cpu expect (N, K), as
        # produced by CUDA nn.Linear-style checkpoints.
        weight = getattr(layer, w_name).t().contiguous()
        n, k = weight.shape

        packed_weight = torch.ops._C.convert_weight_packed(weight)
        replace_parameter(
            layer, w_name, torch.nn.Parameter(packed_weight, requires_grad=False)
        )

        # Synthesize a single-block "block scale" tensor from the scalar
        # per-tensor weight scale so fp8_scaled_mm_cpu can be reused as-is.
        weight_scale = getattr(layer, w_s_name)
        self._block_size_n = -(-n // 32) * 32  # round up to a multiple of 32
        num_k_blocks = -(-k // _BLOCK_SIZE_K)
        block_scale = weight_scale.reshape(()).expand(1, num_k_blocks).contiguous()
        replace_parameter(
            layer, w_s_name, torch.nn.Parameter(block_scale, requires_grad=False)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_name, w_s_name, _, _ = self.layer_param_names
        weight = getattr(layer, w_name)
        weight_scale = getattr(layer, w_s_name)

        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        out = torch.ops._C.fp8_scaled_mm_cpu(
            x_2d,
            weight,
            weight_scale,
            [self._block_size_n, _BLOCK_SIZE_K],
            bias,
            x.dtype,
            True,  # is_vnni (weight already prepacked)
        )
        return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CPUFp8PerTensorLinearKernel overrides apply_weights directly."
        )


class CPUFp8ChannelWiseLinearKernel(CPUFp8PerTensorLinearKernel):
    """FP8 W8A16 per-channel-scaled GEMM via AMX BRGEMM on CPU.

    fp8_scaled_mm_cpu only supports block scales whose N extent is a multiple
    of 32, so the GEMM runs with unit block scales and the per-output-channel
    weight scale is applied to the output afterwards. Activations stay
    BF16, regardless of the checkpoint's activation scheme.
    """

    @classmethod
    def can_implement(cls, c: Fp8LinearLayerConfig) -> tuple[bool, str | None]:
        if not c.weight_quant_key.scale.group_shape.is_per_channel():
            return False, "requires a per-channel weight scale."
        if c.out_dtype != torch.bfloat16:
            return False, "Only bfloat16 output dtype supported."
        n = c.weight_shape[0]
        if n % 32 != 0:
            return False, f"requires weight output dim (N={n}) to be a multiple of 32."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_name, w_s_name, _, _ = self.layer_param_names
        weight = getattr(layer, w_name).t().contiguous()
        n, k = weight.shape

        packed_weight = torch.ops._C.convert_weight_packed(weight)
        replace_parameter(
            layer, w_name, torch.nn.Parameter(packed_weight, requires_grad=False)
        )

        self._block_size_n = n
        num_k_blocks = -(-k // _BLOCK_SIZE_K)
        self._unit_block_scale = torch.ones(1, num_k_blocks, dtype=torch.float32)
        channel_scale = getattr(layer, w_s_name).reshape(-1).to(torch.float32)
        replace_parameter(
            layer, w_s_name, torch.nn.Parameter(channel_scale, requires_grad=False)
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_name, w_s_name, _, _ = self.layer_param_names
        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        out = torch.ops._C.fp8_scaled_mm_cpu(
            x_2d,
            getattr(layer, w_name),
            self._unit_block_scale,
            [self._block_size_n, _BLOCK_SIZE_K],
            None,
            x.dtype,
            True,  # is_vnni (weight already prepacked)
        )
        out = (out.float() * getattr(layer, w_s_name)).to(x.dtype)
        if bias is not None:
            out += bias
        return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out


class CPUFp8W8A8BlockLinearKernel(Fp8BlockLinearKernel):
    """FP8 W8A8 block-quantized GEMM with dynamic per-token activation quant on CPU."""

    # Activation quant is done inside fp8_scaled_mm_with_quant;
    # the base class must not apply a second quant step.
    apply_input_quant = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU platform."
        if not torch.cpu._is_amx_tile_supported():
            return False, "requires AMX tile support (Sapphire Rapids or newer)."
        if not ops.cpu_has_amx_fp8():
            return False, (
                "requires native AMX-FP8 hardware; falls back "
                "to CPUFp8BlockLinearKernel (W8A16) otherwise."
            )
        return True, None

    @classmethod
    def can_implement(cls, config: Fp8LinearLayerConfig) -> tuple[bool, str | None]:
        # Same block-shape requirements as the W8A16 sibling kernel.
        weight_gs = config.weight_quant_key.scale.group_shape
        if weight_gs.col <= 0 or weight_gs.col != 128:
            return False, (
                "CPU FP8 W8A8 block kernel requires K-dimension block size of 128, "
                f"got {weight_gs.col}."
            )
        if weight_gs.row <= 0 or weight_gs.row % 32 != 0:
            return False, (
                "CPU FP8 W8A8 block kernel requires N-dimension block size to be "
                f"a positive multiple of 32, got {weight_gs.row}."
            )
        if config.out_dtype not in (torch.bfloat16, torch.float32):
            return False, "Only bfloat16/float32 output dtype supported."
        n = config.weight_shape[0]
        if n % 32 != 0:
            # float8_linear_prepack_cpu tiles N in chunks of 32 and cannot
            # handle a remainder tile smaller than that.
            return False, f"requires weight output dim (N={n}) to be a multiple of 32."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepack block-quantized FP8 weight for W8A8 AMX BRGEMM.

        For block-quant FP8 models weight is stored as [N, K] (NOT transposed,
        unlike non-block FP8 where vLLM stores [K, N]).

        float8_linear_prepack_cpu expects:
          weight:  [N, K]
          scales:  [N, G]  where G = K / block_k

        weight_scale_inv from the checkpoint has shape [ceil(N/block_n), G],
        so we expand rows from ceil(N/block_n) to N via repeat_interleave.
        """
        params = self._get_layer_params(layer)
        weight = params.weight  # [N, K]

        # MLA's kv_b_proj (see `_cpu_skip_gemm_dispatch`): MLA reads this
        # weight to derive W_UK/W_UV, so keep the checkpoint [N, K] layout and
        # its block scales; apply_weights falls back to the W8A16 op.
        if getattr(layer, "_cpu_skip_gemm_dispatch", False):
            return

        weight_scale = (
            params.weight_scale_inv
            if params.weight_scale_inv is not None
            else params.weight_scale
        )  # [ceil(N/block_n), ceil(K/block_k)]
        assert weight_scale is not None, "FP8 W8A8 block kernel requires weight scale."

        N = weight.shape[0]
        block_n = self.weight_group_shape.row  # e.g. 128

        # Expand scale rows: [ceil(N/block_n), G] → [N, G]
        if weight_scale.size(0) != N:
            weight_scale_expanded = weight_scale.repeat_interleave(block_n, dim=0)[
                :N, :
            ].contiguous()
        else:
            weight_scale_expanded = weight_scale.contiguous()

        packed_weight, packed_scales = torch.ops._C.float8_linear_prepack_cpu(
            weight, weight_scale_expanded.to(torch.float32)
        )

        scale_attr = (
            params.WEIGHT_SCALE_INV
            if params.weight_scale_inv is not None
            else params.WEIGHT_SCALE
        )
        replace_parameter(
            layer,
            params.WEIGHT,
            torch.nn.Parameter(packed_weight, requires_grad=False),
        )
        replace_parameter(
            layer,
            scale_attr,
            torch.nn.Parameter(packed_scales, requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale_inv
            if params.weight_scale_inv is not None
            else params.weight_scale
        )

        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        if getattr(layer, "_cpu_skip_gemm_dispatch", False):
            # Unpacked weight: fp8_scaled_mm_with_quant has no is_vnni option,
            # so fall back to the W8A16 op (BF16 activation, block-scaled FP8
            # weight) for this layer.
            out = torch.ops._C.fp8_scaled_mm_cpu(
                x_2d,
                params.weight,
                weight_scale,
                list(self.weight_group_shape),
                bias,
                x.dtype,
                False,  # is_vnni
            )
            return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

        # act_scales=None + channelwise=True → dynamic per-token activation quant
        out = ops.fp8_scaled_mm_with_quant(
            x_2d,
            None,
            True,
            params.weight,
            weight_scale,
            bias,
            x.dtype,
        )
        return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CPUFp8W8A8BlockLinearKernel overrides apply_weights directly."
        )


class CPUFp8W8A8LinearKernel(Fp8LinearKernel):
    """FP8 W8A8 GEMM with dynamic per-token activation quantization on CPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU platform."
        if not torch.cpu._is_amx_tile_supported():
            return False, "requires AMX tile support (Sapphire Rapids or newer)."
        if not ops.cpu_has_amx_fp8():
            return False, (
                "requires native AMX-FP8 hardware (Diamond Rapids); falls back "
                "to the Torch FP8 kernels otherwise."
            )
        return True, None

    @classmethod
    def can_implement(cls, config: Fp8LinearLayerConfig) -> tuple[bool, str | None]:
        if config.out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False, "Only bfloat16/float16/float32 output dtype supported."
        # Reject block-quantized weights — those go to CPUFp8BlockLinearKernel
        weight_gs = config.weight_quant_key.scale.group_shape
        if weight_gs.col > 0 and weight_gs.row > 0:
            return False, "Block-quantized weights not handled by FP8 W8A8 kernel."
        n = config.weight_shape[0]
        if n % 32 != 0:
            # float8_linear_prepack_cpu tiles N in chunks of 32 and cannot
            # handle a remainder tile smaller than that.
            return False, f"requires weight output dim (N={n}) to be a multiple of 32."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepack weights with float8_linear_prepack_cpu (VNNI + block layout)."""
        w_name, ws_name, xs_name, _ = self.layer_param_names
        weight = getattr(layer, w_name)
        weight_scale = getattr(layer, ws_name)

        # vLLM stores linear weight as [K, N], while the CPU prepack op expects [N, K].
        N = weight.shape[-1]

        # Ensure scale is 2D [N, G]
        ws = weight_scale
        if ws is None:
            raise RuntimeError("FP8 W8A8 kernel requires weight scale.")
        ws = ws.to(torch.float32)

        # Fused modules may carry per-shard scales (e.g. QKV/MLP shards).
        # Expand those to per-output-channel scales before prepack.
        if (
            len(getattr(layer, "logical_widths", [])) > 1
            and ws.dim() <= 1
            and ws.numel() == len(layer.logical_widths)
        ):
            ws = convert_to_channelwise(ws.reshape(-1), layer.logical_widths)

        if ws.dim() == 0:
            ws = ws.view(1, 1).expand(N, 1)
        elif ws.dim() == 1:
            if ws.numel() == 1:
                ws = ws.view(1, 1).expand(N, 1)
            elif ws.numel() == N:
                ws = ws.unsqueeze(1)  # [N] -> [N, 1]
            else:
                raise RuntimeError(
                    "FP8 W8A8 kernel got unsupported 1D weight scale shape: "
                    f"{tuple(ws.shape)} for N={N}."
                )

        # Align scale rows to output channels N.
        if ws.size(0) != N:
            if ws.size(0) == 1:
                ws = ws.expand(N, ws.size(1)).contiguous()
            else:
                raise RuntimeError(
                    "FP8 W8A8 kernel got mismatched weight scale rows: "
                    f"ws_rows={ws.size(0)}, expected N={N}."
                )

        w_2d = weight.t().contiguous()  # [N, K]
        packed_weight, packed_scales = torch.ops._C.float8_linear_prepack_cpu(w_2d, ws)

        replace_parameter(
            layer,
            w_name,
            torch.nn.Parameter(packed_weight, requires_grad=False),
        )
        replace_parameter(
            layer,
            ws_name,
            torch.nn.Parameter(packed_scales, requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        w_name, ws_name, xs_name, _ = self.layer_param_names
        weight = getattr(layer, w_name)
        weight_scale = getattr(layer, ws_name)
        # For static quantization, use pre-computed activation scale
        act_scales = getattr(layer, xs_name, None) if xs_name else None

        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        # channelwise=True: per-token (per-row) quantization when act_scales is None
        # channelwise=False: per-tensor quantization when act_scales is provided
        channelwise = act_scales is None
        out = ops.fp8_scaled_mm_with_quant(
            x_2d,
            act_scales,
            channelwise,
            weight,
            weight_scale,
            bias,
            x.dtype,
        )
        return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

    def apply_scaled_mm(self, A, B, out_dtype, As, Bs, bias=None, **kwargs):
        raise NotImplementedError("CPUFp8W8A8LinearKernel uses apply_weights directly.")
