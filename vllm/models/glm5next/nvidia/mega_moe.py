# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
)

# (local expert id, shard id); the fused shared expert has expert id None.
Shard = tuple[int | None, str]


def requant_block_fp8_to_ue8m0(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize block-FP8 weights to E4M3 with one UE8M0 scale per 1x32.

    The SM100 MegaMoE feeds its scales to the block-scaled MMA, so they have to
    be powers of two, one per row and 32 columns. A checkpoint with float32
    scales per block (e.g. 128x128) is rounded once more to get there.

    Args:
        weight: ``(n, mn, k)`` E4M3 values.
        scale: ``(n, mn / block_m, k / block_k)`` float32 dequantization
            scales (``real = weight * scale``).

    Returns:
        ``(weight, sf)``: the requantized E4M3 weights and their UE8M0 scale
        bytes, ``(n, mn, k / 32)`` uint8.

    """
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    assert scale.dtype == torch.float32, scale.dtype
    n, mn, k = weight.shape
    _, blocks_m, blocks_k = scale.shape
    assert scale.shape[0] == n and mn % blocks_m == 0 and k % blocks_k == 0, (
        tuple(weight.shape),
        tuple(scale.shape),
    )
    real = weight.to(torch.float32).view(n, blocks_m, -1, blocks_k, k // blocks_k)
    real.mul_(scale[:, :, None, :, None])
    return _mxfp8_e4m3_quantize_torch(real.view(n, mn, k))


class Glm5NextMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MXFP8 MegaMoE experts loaded from GLM-5.3-Flash's block-FP8 checkpoint.

    The checkpoint stores E4M3 weights with one float32 scale per
    ``weight_block_size`` block; the kernel takes power-of-two UE8M0 scales per
    1x32. Loading writes each weight shard straight into the kernel layout and
    stages its block scale in ``*_weight_scale_inv``. Once a load is complete,
    the model's ``process_weights_after_loading`` calls
    :meth:`requantize_block_fp8`, which rounds every newly loaded shard to the
    kernel scales in place. A block-FP8 shared expert is fused the same way.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        weight_block_size: tuple[int, int],
        **kwargs,
    ):
        block_m, block_k = weight_block_size
        self.weight_block_size = (block_m, block_k)
        # Shards whose weight / block scale arrived since the last requantize.
        self._new_weights: set[Shard] = set()
        self._new_scales: set[Shard] = set()
        super().__init__(vllm_config, weight_format=MEGA_MOE_MXFP8, **kwargs)
        h, i = self.hidden_size, self.intermediate_size
        if h % block_m or h % block_k or i % block_m or i % block_k:
            raise ValueError(
                "GLM-5.3 MegaMoE requires hidden and intermediate sizes that are "
                f"multiples of the FP8 block size {self.weight_block_size}."
            )

    def _shared_linear_fits(self, linear: nn.Module, spec: MegaMoEWeightSpec) -> bool:
        """A block-FP8 linear loads into the routed MXFP8 layout by requantizing."""
        weight = linear.weight
        scale = getattr(linear, "weight_scale_inv", None)
        if spec != self.weight_format.routed or weight.dtype != torch.float8_e4m3fn:
            return False
        if weight.dim() != 2 or scale is None or scale.dtype != torch.float32:
            return False
        (rows, cols), (block_m, block_k) = weight.shape, self.weight_block_size
        return tuple(scale.shape) == (rows // block_m, cols // block_k)

    def _create_weights(self) -> None:
        super()._create_weights()
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        self._register_block_scale("w13", (e, 2 * i, h))
        self._register_block_scale("w2", (e, h, i))
        if self.has_fused_shared_experts:
            s = i * self.num_shared_experts
            self._register_block_scale("shared_l1", (2 * s, h))
            self._register_block_scale("shared_l2", (h, s))

    def _register_block_scale(self, name: str, shape: tuple[int, ...]) -> None:
        """Stage the checkpoint's float32 block scales of ``name``. The kernel's
        ``{name}_weight_scale`` is derived from them and never loaded."""
        *outer, n, k = shape
        block_m, block_k = self.weight_block_size
        scale = torch.zeros(*outer, n // block_m, k // block_k, dtype=torch.float32)
        self.register_parameter(
            f"{name}_weight_scale_inv", nn.Parameter(scale, requires_grad=False)
        )
        getattr(self, f"{name}_weight_scale").weight_loader_numel = 0

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if weight_name.endswith("weight_scale_inv"):
            param_prefix = "w2_" if shard_id == "w2" else "w13_"
            loaded = bool(local_expert_ids) and param_prefix in weight_name
            for local_expert_id in local_expert_ids if loaded else ():
                self._stage_block_scale(
                    param.data[local_expert_id], loaded_weight, shard_id
                )
            new = self._new_scales
        else:
            loaded = bool(
                super().weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
            )
            new = self._new_weights
        if loaded:
            new.update(
                (local_expert_id, shard_id) for local_expert_id in local_expert_ids
            )
        return loaded if return_success else None

    def _shared_param_name(self, name: str) -> str:
        param_name = super()._shared_param_name(name)
        # Block scales go to the staging parameter, not the kernel scale.
        return param_name + "_inv" if param_name.endswith("_scale") else param_name

    def shared_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str
    ) -> None:
        shard_id = self._shared_shard_id(weight_name)
        if "weight_scale" in weight_name:
            self._stage_block_scale(param.data, loaded_weight, shard_id)
            self._new_scales.add((None, shard_id))
        else:
            super().shared_weight_loader(param, loaded_weight, weight_name)
            self._new_weights.add((None, shard_id))

    @staticmethod
    def _shard_rows(t: torch.Tensor, shard_id: str) -> torch.Tensor:
        """Rows of ``t`` (gate rows, then up rows, for w13) that hold ``shard_id``."""
        if shard_id == "w2":
            return t
        half = t.shape[0] // 2
        return t[:half] if shard_id == "w1" else t[half:]

    def _stage_block_scale(
        self, dst: torch.Tensor, src: torch.Tensor, shard_id: str
    ) -> None:
        dst = self._shard_rows(dst, shard_id)
        if tuple(src.shape) != tuple(dst.shape):
            raise ValueError(
                f"{self.prefix}: {shard_id} block scale shape {tuple(src.shape)} "
                f"does not match {tuple(dst.shape)} for FP8 blocks "
                f"{self.weight_block_size}"
            )
        dst.copy_(src)

    @torch.no_grad()
    def requantize_block_fp8(self, chunk: int = 8) -> None:
        """Round every shard loaded since the last call to UE8M0 1x32 scales,
        in place.

        Runs once a load is complete. A weight update may send a weight and its
        block scale in separate ``load_weights`` calls, but both have to be in
        by then.
        """
        unpaired = self._new_weights ^ self._new_scales
        if unpaired:
            raise ValueError(
                f"{self.prefix}: block-FP8 shards (local expert, shard; None is "
                "the shared expert) loaded without their weight or block scale: "
                f"{sorted(unpaired, key=str)}"
            )
        for shard_id in ("w1", "w3", "w2"):
            experts = sorted(
                e for e, s in self._new_weights if s == shard_id and e is not None
            )
            for start in range(0, len(experts), chunk):
                self._requantize_shards(
                    [
                        self._shard_tensors(e, shard_id)
                        for e in experts[start : start + chunk]
                    ],
                    shard_id,
                )
            if (None, shard_id) in self._new_weights:
                self._requantize_shards([self._shard_tensors(None, shard_id)], shard_id)
        self._new_weights.clear()
        self._new_scales.clear()

    def _shard_tensors(
        self, local_expert_id: int | None, shard_id: str
    ) -> tuple[torch.Tensor, ...]:
        """Kernel weight, kernel scale and staged block scale of one routed
        expert, or of the shared expert for ``local_expert_id`` None."""
        if local_expert_id is None:
            name = "shared_l2" if shard_id == "w2" else "shared_l1"
        else:
            name = "w2" if shard_id == "w2" else "w13"
        tensors = tuple(
            getattr(self, f"{name}_{suffix}").data
            for suffix in ("weight", "weight_scale", "weight_scale_inv")
        )
        if local_expert_id is None:
            return tensors
        return tuple(t[local_expert_id] for t in tensors)

    def _requantize_shards(
        self, tensors: list[tuple[torch.Tensor, ...]], shard_id: str
    ) -> None:
        """Requantize ``shard_id`` of each kernel-layout weight in place."""
        raw = torch.stack(
            [
                self._weight_shard_view(w, shard_id).reshape(-1, w.shape[-1])
                for w, _, _ in tensors
            ]
        )
        block_scale = torch.stack(
            [self._shard_rows(s, shard_id) for _, _, s in tensors]
        )
        weight, sf = requant_block_fp8_to_ue8m0(raw, block_scale)
        for (dst, dst_scale, _), w, s in zip(tensors, weight, sf):
            self._load_weight(dst, w, shard_id, False)
            self._load_weight(dst_scale, s, shard_id, True)


Glm5NextMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
