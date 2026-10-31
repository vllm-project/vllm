# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 MegaMoE experts with weights in GPU memory locality domains."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEWeightFormat,
    MegaMoEWeightSpec,
)
from vllm.utils.deep_gemm import _import_deep_gemm


class DeepseekV4LocalizedMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts whose weight rows live in DeepGEMM locality domains.

    Weights are allocated in the ``deep_gemm.localize(w, dim=-2)`` layout,
    ``(domains, *outer, rows / domains, k)``, and checkpoint shards are loaded
    into it directly, so only one copy is ever resident. Scales keep the base
    layout. The kernel output is bitwise identical to the base class.
    """

    def __init__(
        self, vllm_config: VllmConfig, *, weight_format: MegaMoEWeightFormat, **kwargs
    ):
        if weight_format.uses_nvfp4_api:
            raise ValueError("Localized MegaMoE weights do not support NVFP4.")
        if torch.get_default_device().type != "cuda":
            raise ValueError("Localized MegaMoE weights require CUDA allocation.")
        if not _import_deep_gemm().is_localization_available():
            raise ValueError("DeepGEMM locality-domain allocation is unavailable.")
        try:
            super().__init__(vllm_config, weight_format=weight_format, **kwargs)
        finally:
            # Allocations stay valid; the SM locality map is probed and cached.
            _import_deep_gemm().destroy_localizer()

    def _register_weight(
        self, name: str, shape: tuple[int, ...], spec: MegaMoEWeightSpec
    ) -> None:
        dg = _import_deep_gemm()
        domains = dg.get_num_locality_domains()
        *outer, n, k = shape
        # Each domain must hold whole 16-row gate/up interleave groups.
        if n % (16 * domains):
            raise ValueError(f"{name}: {n} rows do not split into {domains} domains")
        weight = dg.locality_domain.empty_per_domain(
            (*outer, n // domains, k // spec.values_per_element), spec.dtype
        ).zero_()
        self.register_parameter(
            f"{name}_weight", nn.Parameter(weight, requires_grad=False)
        )
        if spec.scale_group_k is not None:
            # Same block-scale layout as the base class.
            scale_k = k // (4 * spec.scale_group_k)
            scale = torch.zeros(*outer, scale_k, n, dtype=torch.int32)
            self.register_parameter(
                f"{name}_weight_scale",
                nn.Parameter(scale.transpose(-1, -2), requires_grad=False),
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
        if param.ndim == 4:
            # (domains, experts, ...) -> (experts, domains, ...)
            param = param.transpose(0, 1)
        return super().weight_loader(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success
        )

    def _load_weight(
        self, dst: torch.Tensor, src: torch.Tensor, shard_id: str, is_scale: bool
    ) -> None:
        if dst.ndim == 2:
            super()._load_weight(dst, src, shard_id, is_scale)
            return
        # (domains, rows / domains, k): each domain takes its share of rows.
        for domain_dst, domain_src in zip(dst, src.chunk(dst.shape[0])):
            super()._load_weight(domain_dst, domain_src, shard_id, is_scale)

    def get_expert_weights(self) -> list[torch.Tensor]:
        """Contiguous ``(experts, -1)`` views; one per domain for weights."""
        e = self.num_local_experts
        views: list[torch.Tensor] = []
        for name, param in self.named_parameters(recurse=False):
            if name.startswith("shared_"):
                continue
            if param.ndim == 4:
                views.extend(domain.view(e, -1) for domain in param)
            else:
                # Block scales are MN-major.
                views.append(param.transpose(1, 2).view(e, -1))
        return views


DeepseekV4LocalizedMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
