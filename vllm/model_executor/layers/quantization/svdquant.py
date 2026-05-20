# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SVDQuant W4A4 quantization with low-rank correction.

SVDQuant (https://arxiv.org/abs/2411.05007) is a 4-bit weight, 4-bit
activation quantization scheme paired with a low-rank residual that
absorbs the quantization error. It is the dominant practical
quantization method for diffusion transformers, delivering >2x
speedup vs BF16 with minimal quality loss.

The in-tree GEMM path uses the external `nunchaku` package, covering
consumer NVIDIA GPUs (Turing SM_75 through consumer Blackwell
SM_120). Hopper SM_90 is unsupported; datacenter Blackwell SM_100/103
is out of scope here (planned via FlashInfer).

Diffusion-specific weight key remapping (e.g. diffusers naming
conventions) is not handled here; downstream pipelines remap before
loading. Checkpoints are expected to already store gated-activation
halves in `[gate; hidden]` order — produce that ordering at
quantization time, not at runtime.
"""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.layers.quantization.utils.svdquant_dispatch import (
    SVDQuantPrecision,
    assert_svdquant_supported,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.nunchaku import (
    svdq_gemm_w4a4,
    svdq_quantize_w4a4_act_fuse_lora,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

# Group sizes are dictated by the kernel's scaled-MMA tile:
#   * NVFP4 uses tcgen05's 16-element scale block.
#   * INT4 uses Nunchaku's 64-element block.
_GROUP_SIZE_BY_PRECISION: dict[str, int] = {"int4": 64, "nvfp4": 16}


class SVDQuantConfig(QuantizationConfig):
    """Configuration for SVDQuant W4A4 quantization.

    Parameters mirror what's on disk in a Nunchaku-produced checkpoint:

    Args:
        rank: SVD low-rank correction dimension. Typical values are
            16, 32, or 64; the checkpoint dictates the value.
        precision: 4-bit format, either "int4" or "nvfp4".
        act_unsigned: Whether activations are quantized as unsigned
            (saves the sign bit at a small accuracy cost). Per
            checkpoint config.
        modules_to_not_convert: Layer names (or substring patterns)
            that should keep their unquantized weight, e.g. embedders
            and adaLN-modulation projections in diffusion models.
    """

    def __init__(
        self,
        rank: int = 32,
        precision: SVDQuantPrecision = "int4",
        act_unsigned: bool = False,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if precision not in _GROUP_SIZE_BY_PRECISION:
            raise ValueError(
                f"SVDQuant precision must be one of "
                f"{set(_GROUP_SIZE_BY_PRECISION)}; got {precision!r}"
            )
        self.rank = rank
        self.precision = precision
        self.group_size = _GROUP_SIZE_BY_PRECISION[precision]
        self.act_unsigned = act_unsigned
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return (
            f"SVDQuantConfig(rank={self.rank}, precision={self.precision!r}, "
            f"act_unsigned={self.act_unsigned})"
        )

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM_75 (Turing) is the floor; the dispatcher rejects SM_90 and
        # routes SM_100+ separately.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SVDQuantConfig":
        return cls(
            rank=config.get("rank", 32),
            precision=config.get("precision", "int4"),
            act_unsigned=config.get("act_unsigned", False),
            modules_to_not_convert=config.get("modules_to_not_convert"),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            skip_with_substr=True,
        ):
            return UnquantizedLinearMethod()
        return SVDQuantLinearMethod(self)


class SVDQuantLinearMethod(LinearMethodBase):
    """Linear method for SVDQuant W4A4.

    The same parameter layout serves both the int4 and nvfp4 paths;
    only the dtypes of `wscales` and the LoRA matrices differ. The
    active platform is checked at `__init__` time and an unsupported
    GPU raises here, before any weights are allocated.
    """

    _hardware_logged = False

    def __init__(self, quant_config: SVDQuantConfig) -> None:
        self.quant_config = quant_config
        assert_svdquant_supported(quant_config.precision)
        if not SVDQuantLinearMethod._hardware_logged:
            logger.info(
                "Using nunchaku backend for SVDQuantLinearMethod (precision=%s)",
                quant_config.precision,
            )
            SVDQuantLinearMethod._hardware_logged = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del extra_weight_attrs  # weight_loader is set explicitly per-param.
        output_size_per_partition = sum(output_partition_sizes)

        config = self.quant_config
        rank = config.rank
        group_size = config.group_size
        precision = config.precision

        # The LoRA matrices and the smooth factor must be in the same
        # dtype as the kernel's accumulator. Nunchaku's nvfp4 path
        # locks this to bf16 regardless of the model's params_dtype;
        # the int4 path inherits params_dtype.
        lora_dtype = torch.bfloat16 if precision == "nvfp4" else params_dtype

        wscales_dtype = (
            torch.float8_e4m3fn if precision == "nvfp4" else params_dtype
        )

        # qweight: 4-bit weights packed two-per-byte along the input
        # axis. Shape (out_per_partition, in_per_partition // 2).
        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        _set_attrs(
            qweight,
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )

        # wscales: per-(group_size) input-column scale,
        # shape (in_per_partition // group_size, out_per_partition).
        wscales = Parameter(
            torch.empty(
                input_size_per_partition // group_size,
                output_size_per_partition,
                dtype=wscales_dtype,
            ),
            requires_grad=False,
        )
        _set_attrs(
            wscales,
            input_dim=0,
            output_dim=1,
            weight_loader=default_weight_loader,
        )

        # SVD low-rank correction matrices.
        proj_down = Parameter(
            torch.empty(input_size_per_partition, rank, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            proj_down,
            input_dim=0,
            output_dim=1,
            weight_loader=default_weight_loader,
        )

        proj_up = Parameter(
            torch.empty(output_size_per_partition, rank, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            proj_up,
            input_dim=1,
            output_dim=0,
            weight_loader=default_weight_loader,
        )

        # Smooth-quant factors. Live on the input axis: replicated for
        # column-parallel layers, sharded for row-parallel.
        smooth_factor = Parameter(
            torch.empty(input_size_per_partition, dtype=lora_dtype),
            requires_grad=False,
        )
        _set_attrs(
            smooth_factor,
            input_dim=0,
            weight_loader=default_weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("smooth_factor", smooth_factor)

        if precision == "nvfp4":
            # Per-output-channel BF16 scale; sharded with the output dim.
            wcscales = Parameter(
                torch.ones(output_size_per_partition, dtype=lora_dtype),
                requires_grad=False,
            )
            _set_attrs(
                wcscales,
                output_dim=0,
                weight_loader=default_weight_loader,
            )
            # Per-tensor global scale (shape (1,) on disk).
            wtscale = Parameter(
                torch.ones(1, dtype=lora_dtype),
                requires_grad=False,
            )
            _set_attrs(wtscale, weight_loader=default_weight_loader)
            layer.register_parameter("wcscales", wcscales)
            layer.register_parameter("wtscale", wtscale)
        else:
            # Keep the attributes present so apply() can branch
            # uniformly without `hasattr` checks.
            layer.wcscales = None
            layer.wtscale = None

        # Stash for apply().
        layer.in_features = input_size
        layer.out_features = output_size
        layer.out_features_per_partition = output_size_per_partition
        layer.precision = precision
        layer.act_unsigned = config.act_unsigned

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-load weight prep.

        On-disk format is canonical row-major NVFP4; the nunchaku kernel
        wants a PTX-MMA fragment-permuted layout, so for NVFP4 we repack
        here once. Bit-preserving pack/unpack pair (see
        `utils/svdquant_nvfp4_layout.py`); round-trip verified bit-exact.

        Also caches the kernel's `alpha` from the per-tensor `wtscale`.
        Do NOT collapse `wcscales` into a scalar `alpha` — the kernel
        applies them as `(accumulator * alpha) * wcscales`, and
        conflating the two double-counts the per-channel factors.

        All parameters are produced by our quantization pipeline and
        must be loaded by the time we get here; a meta tensor at this
        point is a checkpoint bug, not a missing-shard case to paper
        over.
        """
        if layer.precision == "nvfp4":
            self._pack_nvfp4_to_nunchaku_fragment(layer)

        alpha: float = 1.0
        wtscale = getattr(layer, "wtscale", None)
        if wtscale is not None:
            value = float(wtscale.detach().cpu().item())
            if abs(value - 1.0) > 1e-6:
                alpha = value
        layer._svdquant_alpha = alpha

    @staticmethod
    def _pack_nvfp4_to_nunchaku_fragment(layer: torch.nn.Module) -> None:
        """Repack row-major NVFP4 params in-place to nunchaku fragment layout.

        On-disk (canonical row-major):
          * qweight   : [N, K/2] int8/uint8 (FP4 nibbles, low = even-k)
          * wscales   : [K/16, N] fp8_e4m3fn
          * proj_up   : [N, R]
          * proj_down : [K, R]

        After repack (nunchaku PTX-MMA fragment):
          * qweight   : [N, K/2] int8 (permuted into MMA fragment)
          * wscales   : [K/16, N] fp8 (permuted into MMA fragment)
          * proj_up   : [N, R] (permuted into MMA fragment)
          * proj_down : [K, R] (permuted into MMA fragment)
        """
        # Lazy imports: nunchaku is a soft dep on non-consumer hardware,
        # and the layout helpers pull in torch ops we only need here.
        from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight

        from vllm.model_executor.layers.quantization.utils.svdquant_nvfp4_layout import (  # noqa: E501
            _unpack_nibbles,
            pack_nunchaku_qweight_fp4,
            pack_nunchaku_wscales_fp4,
        )

        device = layer.qweight.device

        # qweight: stored as [N, K/2] packed-nibble bytes (low = even-k).
        # `pack_nunchaku_qweight_fp4` expects [N, K] one-nibble-per-byte —
        # unpack to that form first, then pack to nunchaku fragment.
        qw_rm_packed = layer.qweight.data.view(torch.uint8)               # [N, K/2]
        qw_rm_nibs = _unpack_nibbles(qw_rm_packed)                        # [N, K]
        layer.qweight.data = pack_nunchaku_qweight_fp4(qw_rm_nibs).to(device)

        # wscales: pack pair operates in fp8_e4m3fn.
        layer.wscales.data = pack_nunchaku_wscales_fp4(layer.wscales.data).to(device)

        # proj_up: row-major [N, R] → nunchaku frag [N, R]. down=False.
        layer.proj_up.data = pack_lowrank_weight(layer.proj_up.data, down=False).to(device)

        # proj_down: canonical row-major is [K, R]; nunchaku's pack expects
        # [R, K] (transpose-quirk on the down=True path). Transpose then pack;
        # output is fragment [K, R].
        pd = layer.proj_down.data
        pd_rk = pd.transpose(0, 1).contiguous()
        layer.proj_down.data = pack_lowrank_weight(pd_rk, down=True).to(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])

        is_fp4 = layer.precision == "nvfp4"
        out_features = layer.out_features_per_partition

        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora(
            x_2d,
            lora_down=layer.proj_down,
            smooth=layer.smooth_factor,
            fp4=is_fp4,
            pad_size=256,
        )

        # The quantize kernel may pad the batch dim up to a multiple
        # of `pad_size`; the GEMM consumes the padded shape, then we
        # trim back to the real batch size below.
        out_2d = torch.empty(
            quantized_x.shape[0],
            out_features,
            dtype=layer.proj_up.dtype,
            device=x_2d.device,
        )

        svdq_gemm_w4a4(
            act=quantized_x,
            wgt=layer.qweight,
            out=out_2d,
            ascales=ascales,
            wscales=layer.wscales,
            lora_act_in=lora_act_out,
            lora_up=layer.proj_up,
            bias=bias,
            fp4=is_fp4,
            alpha=getattr(layer, "_svdquant_alpha", 1.0),
            wcscales=layer.wcscales,
            act_unsigned=layer.act_unsigned,
        )

        actual_batch = x_2d.shape[0]
        if out_2d.shape[0] > actual_batch:
            out_2d = out_2d[:actual_batch]

        return out_2d.reshape(*orig_shape[:-1], out_features)


def _set_attrs(param: torch.nn.Parameter, **attrs: Any) -> None:
    for key, value in attrs.items():
        setattr(param, key, value)


__all__ = ["SVDQuantConfig", "SVDQuantLinearMethod"]
