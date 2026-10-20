# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
)

logger = init_logger(__name__)


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


def _is_block_fp8_linear(linear: nn.Module, block: tuple[int, int]) -> bool:
    weight = linear.weight
    scale = getattr(linear, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2:
        return False
    if scale is None or scale.dtype != torch.float32:
        return False
    rows, cols = weight.shape
    return (
        rows % block[0] == 0
        and cols % block[1] == 0
        and tuple(scale.shape) == (rows // block[0], cols // block[1])
    )


class Glm5NextMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MXFP8 MegaMoE experts loaded from GLM-5.3-Flash's block-FP8 checkpoint.

    The checkpoint stores E4M3 weights with one float32 scale per
    ``weight_block_size`` block; the kernel takes power-of-two UE8M0 scales per
    1x32. Weight shards are written straight into the kernel layout and their
    float32 block scales are staged in ``*_weight_scale_inv``. Once every shard
    is in, :meth:`requantize_block_fp8` rounds each freshly loaded shard to the
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
        # Shards whose raw weight / block scale arrived since the last
        # requantize: (local expert, shard) and shared-expert shards.
        self._new_weights: set[tuple[int, str]] = set()
        self._new_scales: set[tuple[int, str]] = set()
        self._new_shared_weights: set[str] = set()
        self._new_shared_scales: set[str] = set()
        super().__init__(vllm_config, weight_format=MEGA_MOE_MXFP8, **kwargs)
        h, i = self.hidden_size, self.intermediate_size
        if h % block_m or h % block_k or i % block_m or i % block_k:
            raise ValueError(
                "GLM-5.3 MegaMoE requires hidden and intermediate sizes that are "
                f"multiples of the FP8 block size {self.weight_block_size}."
            )

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        """Fuse an unsharded block-FP8 shared expert, requantized to the routed
        MXFP8 layout after loading; any other one stays a separate MLP."""
        if shared_experts is None:
            return 0, None
        gate_up, down = shared_experts.gate_up_proj, shared_experts.down_proj
        hidden, inter = self.hidden_size, self.intermediate_size
        shared_size = down.weight.shape[-1]
        spec = self.weight_format.routed
        if (
            all(
                _is_block_fp8_linear(linear, self.weight_block_size)
                for linear in (gate_up, down)
            )
            and tuple(gate_up.weight.shape) == (2 * shared_size, hidden)
            and tuple(down.weight.shape) == (hidden, shared_size)
            and shared_size % inter == 0
            and self._kernel_fuses_shared_experts(spec)
        ):
            return shared_size // inter, spec
        logger.warning_once(
            "GLM-5.3 MegaMoE fuses only an unsharded block-FP8 shared expert "
            "(and needs a DeepGEMM build that fuses shared experts); running "
            "the shared expert as a separate MLP."
        )
        return 0, None

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
        """Staging for the checkpoint's float32 block scales of ``name``."""
        *outer, n, k = shape
        block_m, block_k = self.weight_block_size
        scale = torch.zeros(*outer, n // block_m, k // block_k, dtype=torch.float32)
        self.register_parameter(
            f"{name}_weight_scale_inv", nn.Parameter(scale, requires_grad=False)
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not weight_name.endswith("weight_scale_inv"):
            success = super().weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success and weight_name.endswith("weight"):
                self._new_weights.update(
                    (local_expert_id, shard_id)
                    for local_expert_id in self._map_global_expert_id(expert_id)
                )
            return success if return_success else None
        local_expert_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_expert_ids or param_prefix not in weight_name:
            return False if return_success else None
        for local_expert_id in local_expert_ids:
            self._stage_block_scale(
                param.data[local_expert_id], loaded_weight, shard_id
            )
            self._new_scales.add((local_expert_id, shard_id))
        return True if return_success else None

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

    def load_shared_expert_weight(self, name: str, loaded_weight: torch.Tensor) -> str:
        """Load one block-FP8 shared-expert tensor; return the parameter name."""
        if ".gate_proj." in name:
            shard_id, param_name = "w1", "shared_l1_weight"
        elif ".up_proj." in name:
            shard_id, param_name = "w3", "shared_l1_weight"
        elif ".down_proj." in name:
            shard_id, param_name = "w2", "shared_l2_weight"
        else:
            raise ValueError(f"Unexpected shared-expert tensor: {name}")
        if name.endswith("weight_scale_inv"):
            param_name += "_scale_inv"
            self._stage_block_scale(
                getattr(self, param_name).data, loaded_weight, shard_id
            )
            self._new_shared_scales.add(shard_id)
        else:
            self._load_weight(
                getattr(self, param_name).data, loaded_weight, shard_id, False
            )
            self._new_shared_weights.add(shard_id)
        return f"{name.partition('shared_experts.')[0]}experts.{param_name}"

    @torch.no_grad()
    def requantize_block_fp8(self, chunk: int = 8) -> None:
        """Round every shard whose weight and block scale both arrived since
        the last call to UE8M0 1x32 scales, in place.

        Called after every ``load_weights``. A shard still missing its weight or
        scale waits for a later call, so a weight update may send the two in any
        order and in separate calls; it has to send both.
        """
        if self.w13_weight.is_meta:
            self._new_weights.clear()
            self._new_scales.clear()
            self._new_shared_weights.clear()
            self._new_shared_scales.clear()
            return
        ready = self._new_weights & self._new_scales
        shared_ready = self._new_shared_weights & self._new_shared_scales
        self._check_scales_loaded(
            [
                (f"local expert {e}", shard_id, self._expert_tensors(e, shard_id)[2])
                for e, shard_id in sorted(self._new_weights - ready)
            ]
            + [
                ("the shared expert", shard_id, self._shared_tensors(shard_id)[2])
                for shard_id in sorted(self._new_shared_weights - shared_ready)
            ]
        )
        for shard_id in ("w1", "w3", "w2"):
            local_ids = sorted(e for e, s in ready if s == shard_id)
            for start in range(0, len(local_ids), chunk):
                ids = local_ids[start : start + chunk]
                self._requantize_shards(
                    [self._expert_tensors(e, shard_id) for e in ids],
                    shard_id,
                    f"local experts {ids}",
                )
            if shard_id in shared_ready:
                self._requantize_shards(
                    [self._shared_tensors(shard_id)], shard_id, "the shared expert"
                )
        self._new_weights -= ready
        self._new_scales -= ready
        self._new_shared_weights -= shared_ready
        self._new_shared_scales -= shared_ready

    def _expert_tensors(
        self, local_expert_id: int, shard_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel weight, kernel scale and staged block scale of one expert."""
        name = "w2" if shard_id == "w2" else "w13"
        return (
            getattr(self, f"{name}_weight").data[local_expert_id],
            getattr(self, f"{name}_weight_scale").data[local_expert_id],
            getattr(self, f"{name}_weight_scale_inv").data[local_expert_id],
        )

    def _shared_tensors(
        self, shard_id: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = "shared_l2" if shard_id == "w2" else "shared_l1"
        return (
            getattr(self, f"{name}_weight").data,
            getattr(self, f"{name}_weight_scale").data,
            getattr(self, f"{name}_weight_scale_inv").data,
        )

    def _check_scales_loaded(
        self, waiting: list[tuple[str, str, torch.Tensor]]
    ) -> None:
        """Reject weights whose block scale was never loaded.

        A weight waiting on a later scale is fine, but staging that is still
        zero means the checkpoint has no ``weight_scale_inv`` for it.
        """
        for what, shard_id, block_scale in waiting:
            if not bool(self._shard_rows(block_scale, shard_id).any()):
                raise ValueError(
                    f"{self.prefix}: {shard_id} block scales of {what} were not "
                    "loaded; the checkpoint must ship weight_scale_inv for every "
                    "block-FP8 expert weight."
                )

    def _requantize_shards(
        self,
        tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        shard_id: str,
        what: str,
    ) -> None:
        """Requantize ``shard_id`` of each kernel-layout weight in place."""
        weights = [w for w, _, _ in tensors]
        k = weights[0].shape[-1]
        if shard_id == "w2":
            raw = torch.stack(weights)
        else:
            # Undo the 8-row gate/up interleave to get checkpoint row order.
            half = 0 if shard_id == "w1" else 1
            raw = torch.stack(
                [w.view(-1, 2, 8, k)[:, half].reshape(-1, k) for w in weights]
            )
        block_scale = torch.stack(
            [self._shard_rows(s, shard_id) for _, _, s in tensors]
        )
        if not bool((block_scale > 0).all()):
            raise ValueError(
                f"{self.prefix}: {shard_id} block scales of {what} must be positive."
            )
        weight, sf = requant_block_fp8_to_ue8m0(raw, block_scale)
        for (dst, dst_scale, _), w, s in zip(tensors, weight, sf):
            self._load_weight(dst, w, shard_id, False)
            self._load_weight(dst_scale, s, shard_id, True)


Glm5NextMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
