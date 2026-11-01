# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``w4afp8`` quantization: PhalaCloud W4A8 MoE checkpoints.

Routed experts are translated into the compressed-tensors W4A8 FP8 MoE path
(Humming / CUTLASS W4A8 oracle, vllm-project/vllm#58427), so no kernel or
build changes are involved. Dense Linear layers reuse the native FP8
block-128 path, because these checkpoints serialize dense scales under
``weight_scale_inv`` (as :class:`Fp8Config` expects) rather than
``weight_scale``.

Per routed expert the checkpoint (e.g. ``PhalaCloud/GLM-5.3-W4AFP8``) holds:

* ``{gate,up,down}_proj.weight``: signed INT4, two nibbles per ``int8`` byte.
* ``{gate,up,down}_proj.weight_scale_inv``: ``bf16`` group-128 scales.
* ``w{1,2,3}.input_scale``: static per-tensor FP8 activation scales, loaded
  only so that the stock loader finds a parameter for them. The W4A8 FP8
  kernels consume dynamic per-token activations, so the values are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8 import (  # noqa: E501
    CompressedTensorsW4A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.utils import (
    move_parameter,
    replace_parameter,
    retire_parameter,
    set_weight_attrs,
)

if TYPE_CHECKING:
    from compressed_tensors.quantization import QuantizationArgs

    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# Routed experts of GLM/DeepSeek-style checkpoints. Layers that do not match
# stay unquantized, which is why `get_quant_method` warns instead of silently
# accepting them.
_MOE_EXPERTS_TARGET = r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"

# Checkpoint packs two INT4 nibbles per byte; the W4A8 oracle packs eight per
# int32 word (4 bytes).
_INT4_PER_BYTE = 2


def _repack_signed_int4_to_uint4b8(weight: torch.Tensor) -> torch.Tensor:
    """Repack signed INT4 (``int8``, 2 per byte) into ``uint4b8`` (``int32``).

    XOR-ing every byte with ``0x88`` flips the sign bit of both nibbles, which
    maps each two's-complement int4 value ``v`` to ``v + 8`` modulo 16, i.e.
    the offset-by-8 unsigned encoding the W4A8 MoE kernels consume
    (compressed-tensors publishes such checkpoints directly, which is why the
    oracle performs no conversion of its own). ``view(torch.int32)`` then
    folds the trailing byte axis by four without disturbing those bits.
    """
    assert weight.dtype is torch.int8, f"expected int8 weights, got {weight.dtype}"
    flipped = weight.clone().view(torch.uint8)
    assert flipped.shape[-1] % 4 == 0, (
        f"trailing dim {flipped.shape[-1]} does not pack into whole int32 words"
    )
    flipped.bitwise_xor_(0x88)
    return flipped.view(torch.int32)


class _W4AFP8MoEAdapter(CompressedTensorsW4A8Fp8MoEMethod):
    """Bridge the on-disk ``w4afp8`` MoE layout to the W4A8 FP8 path.

    Parameters are created under the checkpoint's own names and widths
    (``w{13,2}_weight`` as ``int8`` pack=2, ``w{13,2}_weight_scale_inv``,
    ``w{13,2}_input_scale``) so that the stock
    ``fused_moe_make_expert_params_mapping`` loader can fill them without any
    model-side change. :meth:`process_weights_after_loading` then repacks them
    into the ``int32`` pack=8 / ``*_weight_scale`` layout and hands the layer
    to the parent, which runs the usual oracle / backend / kernel setup.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(weight_quant, input_quant, moe, layer_name)
        # The repack below assumes the parent consumes 8xint4 per int32.
        assert self.packed_factor == 8, (
            f"expected 8 int4 per int32, got packed_factor={self.packed_factor}"
        )

    def create_weights(  # type: ignore[override]
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # CUTLASS reorder_tensor requires 256-alignment on both axes.
        assert hidden_size % 256 == 0, f"{hidden_size=} must be divisible by 256"
        assert intermediate_size_per_partition % 256 == 0, (
            f"{intermediate_size_per_partition=} must be divisible by 256"
        )

        inter = intermediate_size_per_partition
        n_out = self.moe.w13_num_shards * inter

        # int8 pack=2 signed INT4, as published by PhalaCloud.
        for name, shape in (
            ("w13_weight", (num_experts, n_out, hidden_size // _INT4_PER_BYTE)),
            ("w2_weight", (num_experts, hidden_size, inter // _INT4_PER_BYTE)),
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int8), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        per_group = {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        for name, shape, attrs in (
            (
                "w13_weight_scale_inv",
                (num_experts, n_out, hidden_size // self.group_size),
                per_group,
            ),
            (
                "w2_weight_scale_inv",
                (num_experts, hidden_size, inter // self.group_size),
                per_group,
            ),
            # Static activation scales: one scalar per expert. `<E>` (not
            # `<E, 2>`) is enough because w1 and w3 overwrite each other and
            # the value is discarded after loading anyway.
            ("w13_input_scale", (num_experts,), {"is_w4afp8_input_scale": True}),
            ("w2_input_scale", (num_experts,), {"is_w4afp8_input_scale": True}),
        ):
            param = torch.nn.Parameter(
                torch.ones(shape, dtype=params_dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, {**extra_weight_attrs, **attrs})

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:  # type: ignore[override]
        for name in ("w13", "w2"):
            replace_parameter(
                layer,
                f"{name}_weight",
                _repack_signed_int4_to_uint4b8(getattr(layer, f"{name}_weight").data),
            )
            # The parent reads `*_weight_packed` and aliases it back onto
            # `*_weight`. Keep the checkpoint name registered as well, so a
            # later reload still loads through it.
            move_parameter(layer, f"{name}_weight", f"{name}_weight_packed")

            # `*_scale_inv` -> `*_scale`: same data, the oracle's name. Aliased
            # rather than renamed, for the same reload reason.
            move_parameter(layer, f"{name}_weight_scale_inv", f"{name}_weight_scale")

        # Static activation scales are never read by the W4A8 kernels. Retire
        # them instead of deleting, so reloading stays possible.
        retire_parameter(layer, "w13_input_scale")
        retire_parameter(layer, "w2_input_scale")
        super().process_weights_after_loading(layer)


class W4AFP8Config(CompressedTensorsConfig):
    """Config for ``w4afp8`` checkpoints.

    Inheriting from :class:`CompressedTensorsConfig` keeps the routed-expert
    plumbing (scheme lookup, backend selection, W4A8 oracle, kernel
    construction) untouched; only :meth:`from_config`, :meth:`apply_vllm_mapper`
    and :meth:`get_quant_method` are overridden to pick the right method per
    layer type.
    """

    _dense_fp8_config: Fp8Config

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "w4afp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W4AFP8Config:
        if config.get("quant_method") != "w4afp8":
            raise ValueError(
                f"Expected quant_method='w4afp8', got {config.get('quant_method')!r}"
            )

        group_size = int(config.get("group_size", 128))
        ignore = list(
            config.get("modules_to_not_convert") or config.get("ignore") or []
        )

        # Fabricate the compressed-tensors view of the *routed experts* only;
        # dense Linear layers are handled by `_dense_fp8_config` below.
        experts_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "ignore": ignore,
            "config_groups": {
                "group_0": {
                    "targets": [_MOE_EXPERTS_TARGET],
                    "format": "pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": group_size,
                        "dynamic": False,
                        "observer": "minmax",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                        "observer": None,
                    },
                },
            },
        }

        inst = cast("W4AFP8Config", super().from_config(experts_config))

        # Dense layers are FP8 block-128 w8a8 with `weight_scale_inv`, which is
        # exactly the native FP8 serialization format.
        inst._dense_fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=ignore,
            weight_block_size=[128, 128],
        )

        logger.info_once(
            "Loading w4afp8 checkpoint: routed experts via the W4A8 FP8 oracle "
            "(INT4 g=%d x FP8-dynamic), dense Linear via FP8 block-128.",
            group_size,
        )
        return inst

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        self._dense_fp8_config.ignored_layers = hf_to_vllm_mapper.apply_list(
            self._dense_fp8_config.ignored_layers
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return self._dense_fp8_config.get_quant_method(layer, prefix)

        if isinstance(layer, RoutedExperts):
            # Resolve the scheme through compressed-tensors, then swap in the
            # adapter. The intermediate method object is cheap (backend
            # selection only) and is discarded immediately.
            base = super().get_quant_method(layer, prefix)
            if isinstance(base, CompressedTensorsW4A8Fp8MoEMethod):
                return _W4AFP8MoEAdapter(
                    weight_quant=base.weight_quant,
                    input_quant=base.input_quant,
                    moe=layer.moe_config,
                    layer_name=prefix,
                )
            if base is None:
                # Running these experts unquantized would produce silent
                # garbage, so make the mismatch visible.
                logger.warning_once(
                    "w4afp8: routed experts of %s are not quantized; the "
                    "checkpoint layout does not match %s",
                    prefix,
                    _MOE_EXPERTS_TARGET,
                )
            return base

        return super().get_quant_method(layer, prefix)
