# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from compressed_tensors.transform import TransformConfig, TransformLocation
from compressed_tensors.utils import is_match

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    get_schemes_args,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.module import (  # noqa: E501
    HadamardTransform,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (  # noqa: E501
    TransformTuple,
)


def _projection_sides(layer: RoutedExperts) -> dict[str, str]:
    sides = {
        layer.ckpt_gate_proj_name: "w13",
        layer.ckpt_up_proj_name: "w13",  # None for non-gated experts
        layer.ckpt_down_proj_name: "w2",
    }
    return {proj: side for proj, side in sides.items() if proj is not None}


def create_moe_transforms(
    layer: RoutedExperts, prefix: str, config: TransformConfig | None
) -> "CompressedTensorsMoETransforms | None":
    """Resolve the online input transforms of the w13 and w2 expert GEMMs, which
    must be shared by all experts (and by gate/up)."""
    if config is None:
        return None

    proxy = torch.nn.Linear(1, 1, device="meta")  # experts match "Linear" targets
    online = [TransformTuple(*t) for t in get_schemes_args(config) if t[2].is_online()]
    found: dict[str, TransformTuple | None] = {}
    for expert in range(layer.moe_config.num_logical_experts):
        for proj, side in _projection_sides(layer).items():
            name = f"{prefix}.{expert}.{proj}"
            matches = [
                t
                for t in online
                if is_match(name, proxy, t.args.targets, t.args.ignore)
            ]
            if len(matches) > 1:
                raise NotImplementedError(f"Multiple online transforms on {name}")
            tfm = matches[0] if matches else None
            if found.setdefault(side, tfm) != tfm:
                raise ValueError(
                    f"Expert {side} input transforms must be identical across experts"
                )

    transforms = {}
    for side, tfm in found.items():
        if tfm is None:
            continue
        scheme_name, scheme = tfm.scheme_name, tfm.scheme
        if tfm.args.location != TransformLocation.INPUT:
            raise NotImplementedError(
                f"Expert transform {scheme_name}: {tfm.args.location} transforms "
                "are not supported"
            )
        if scheme.randomize:
            raise NotImplementedError(
                f"Expert transform {scheme_name}: randomized transforms are not "
                "supported"
            )
        if scheme.type not in ("hadamard", "random-hadamard"):
            raise NotImplementedError(
                f"Expert transform {scheme_name}: type {scheme.type} is not supported"
            )
        if scheme.type == "hadamard" and scheme.requires_grad:
            # hadacore would ignore the learned matrix
            raise NotImplementedError(
                f"Expert transform {scheme_name}: learned hadamard transforms are "
                "not supported"
            )
        transforms[side] = tfm

    return CompressedTensorsMoETransforms(layer, transforms) if transforms else None


class CompressedTensorsMoETransforms(torch.nn.Module):
    """Input transforms of the w13 GEMM (applied in `forward`, after routing) and
    the w2 GEMM (installed as the experts' post-activation transform)."""

    def __init__(self, layer: RoutedExperts, transforms: dict[str, TransformTuple]):
        super().__init__()
        if layer.expert_map_manager.num_fused_shared_experts:
            raise NotImplementedError(
                "Expert transforms do not support fused shared-expert slots"
            )
        self.sides = _projection_sides(layer)
        config = layer.moe_config
        assert config.hidden_dim_unpadded is not None
        assert config.intermediate_size_per_partition_unpadded is not None
        sizes = {
            "w13": config.hidden_dim_unpadded,
            "w2": config.intermediate_size_per_partition_unpadded,
        }
        self.w13: HadamardTransform | None = None
        self.w2: HadamardTransform | None = None
        for side, tfm in transforms.items():
            transform = HadamardTransform(
                {0: tfm},
                layer,
                lambda *args, **kwargs: None,  # loaded by `load_weight`
                sizes[side],
                [sizes[side]],
            )
            setattr(self, side, transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.w13 is None else self.w13(x)

    def load_weight(self, name: str, weight: torch.Tensor) -> str | None:
        """Load `<expert>.<proj>.<scheme>_input.weight` (any expert) into the
        shared matrix, before the expert projection names are rewritten."""
        parts = name.split(".")
        if len(parts) != 4 or parts[3] != "weight":
            return None
        side = self.sides.get(parts[1])
        module = getattr(self, side) if side else None
        if module is None or parts[2] != f"{module.transforms[0].scheme_name}_input":
            return None
        module.weight.load_column_parallel_weight(weight)
        return f"ct_transforms.{side}.weight"

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        config = layer.moe_config
        if (
            config.hidden_dim != config.hidden_dim_unpadded
            or config.intermediate_size_per_partition
            != config.intermediate_size_per_partition_unpadded
        ):
            raise NotImplementedError(
                "Expert transforms do not support backend feature padding"
            )
        kernel = layer.quant_method.moe_kernel
        experts = kernel.fused_experts if kernel is not None else None
        if type(experts) not in (TritonExperts, TritonWNA16Experts, MarlinExperts):
            raise NotImplementedError(
                "Compressed-tensors expert transforms require Triton or Marlin "
                f"experts; selected {type(experts).__name__}"
            )
        assert isinstance(experts, FusedMoEExpertsModular)

        for module in (self.w13, self.w2):
            if module is None:
                continue
            tfm = module.transforms[0]
            # hadacore does not read the matrix; others need a saved copy, which
            # may be tied to a non-expert layer
            if tfm.scheme.type != "hadamard" and not module.weight.partitions[0].any():
                raise ValueError(
                    f"Missing checkpoint matrix for expert transform {tfm.scheme_name}"
                )
            module.process_weights_after_loading()
        experts.post_activation_transform = self.w2
