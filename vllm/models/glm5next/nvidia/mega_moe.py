# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM FP8xFP8 MegaMoE experts for GLM-5.3-Flash.

The checkpoint's experts are block FP8 (E4M3, one float32 scale per 128x128
block). DeepGEMM's SM100 MegaMoE feeds the scales to the block-scaled MMA, so
it needs power-of-two (UE8M0) scales, one per row and 32 columns. The weights
are requantized once at load to that format; routed and shared experts then run
in one kernel that also dispatches and combines across the EP group.
"""

from inspect import signature

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

MMA_TYPE = "fp8xfp8"
_FP8_MAX = 448.0
_SF_GRAN_K = 32


def requant_block_fp8_to_ue8m0(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    chunk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize block-FP8 weights to E4M3 with one UE8M0 scale per 1x32.

    Args:
        weight: ``(..., mn, k)`` E4M3 values.
        scale: ``(..., mn / block_m, k / block_k)`` float32 dequantization
            scales (``real = weight * scale``).
        block_size: ``(block_m, block_k)`` of the checkpoint's quantization.
        chunk: Leading-dimension entries converted at a time, bounding the
            float32 workspace.

    Returns:
        ``(weight, sf)``: the requantized E4M3 weights and their float32
        power-of-two scales, ``(..., mn, k / 32)``.

    """
    assert weight.dtype == torch.float8_e4m3fn, weight.dtype
    assert scale.dtype == torch.float32, scale.dtype
    block_m, block_k = block_size
    *lead, mn, k = weight.shape
    assert k % _SF_GRAN_K == 0, k
    assert tuple(scale.shape) == (*lead, -(-mn // block_m), -(-k // block_k)), (
        tuple(weight.shape),
        tuple(scale.shape),
    )
    w = weight.reshape(-1, mn, k)
    s = scale.reshape(-1, scale.shape[-2], scale.shape[-1])
    out = torch.empty_like(w)
    sf = torch.empty(
        w.shape[0], mn, k // _SF_GRAN_K, dtype=torch.float32, device=w.device
    )
    for start in range(0, w.shape[0], chunk):
        end = min(start + chunk, w.shape[0])
        s_full = (
            s[start:end]
            .repeat_interleave(block_m, dim=1)[:, :mn]
            .repeat_interleave(block_k, dim=2)[:, :, :k]
        )
        real = w[start:end].to(torch.float32) * s_full
        groups = real.view(end - start, mn, k // _SF_GRAN_K, _SF_GRAN_K)
        amax = groups.abs().amax(dim=-1).clamp_(min=1e-30)
        # Smallest power of two holding the group's largest value in E4M3.
        new_sf = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
        out[start:end] = ((groups / new_sf.unsqueeze(-1)).view(end - start, mn, k)).to(
            torch.float8_e4m3fn
        )
        sf[start:end] = new_sf
    return out.view(*lead, mn, k), sf.view(*lead, mn, k // _SF_GRAN_K)


class Glm5NextMegaMoEExperts(nn.Module):
    """Routed (and optionally the shared) experts of one GLM-5.3-Flash MoE
    layer, run by DeepGEMM's FP8xFP8 MegaMoE kernel."""

    _symm_buffer_cache: dict[tuple, object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        weight_block_size: tuple[int, int],
        num_shared_experts: int = 0,
        prefix: str = "",
        num_logical_experts: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weight_block_size = weight_block_size
        self.num_shared_experts = num_shared_experts
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )
        self.eplb_state = EplbLayerState()

        block_m, block_k = weight_block_size
        if (
            hidden_size % 128
            or intermediate_size % 128
            or intermediate_size % block_m
            or hidden_size % block_m
            or hidden_size % block_k
            or intermediate_size % block_k
        ):
            raise ValueError(
                "GLM-5.3 MegaMoE requires hidden and intermediate sizes that are "
                f"multiples of 128 and of the FP8 block size {weight_block_size}."
            )

        def param(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
            p = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, {"weight_loader": self.weight_loader})
            return p

        e, h, i = num_local_experts, hidden_size, intermediate_size
        # Loader-side parameters, under the names the expert mapping gives the
        # block-FP8 checkpoint tensors (``experts.routed_experts.w13_weight``...).
        loader = nn.Module()
        loader.w13_weight = param((e, 2 * i, h), torch.float8_e4m3fn)
        loader.w13_weight_scale_inv = param(
            (e, 2 * i // block_m, h // block_k), torch.float32
        )
        loader.w2_weight = param((e, h, i), torch.float8_e4m3fn)
        loader.w2_weight_scale_inv = param(
            (e, h // block_m, i // block_k), torch.float32
        )
        self.routed_experts: nn.Module | None = loader

        self._l1: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l2: tuple[torch.Tensor, torch.Tensor] | None = None
        self._shared_l1: tuple[torch.Tensor, torch.Tensor] | None = None
        self._shared_l2: tuple[torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------- loading
    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        return [
            p - self.experts_start_idx
            for p in range(self.experts_start_idx, self.experts_end_idx)
            if p % self.num_logical_experts == expert_id
        ]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        loaded_any = False
        for local_expert_id in self._map_global_expert_id(expert_id):
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                # gate (w1) rows first, then up (w3); scales shard the same way.
                half = expert_data.shape[0] // 2
                expert_data = expert_data.narrow(
                    0, 0 if shard_id == "w1" else half, half
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")
            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"GLM-5.3 MegaMoE expert weight shape mismatch for {weight_name}: "
                    f"parameter shard {tuple(expert_data.shape)} vs checkpoint "
                    f"{tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        return loaded_any if return_success else None

    @staticmethod
    def _deep_gemm():
        from vllm.utils.deep_gemm import _import_deep_gemm

        return _import_deep_gemm()

    def _check_runtime_supported(self, deep_gemm, device: torch.device) -> None:
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        params = signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters
        if "mma_type" not in params or "num_shared_experts" not in params:
            raise NotImplementedError(
                "The installed DeepGEMM does not expose FP8xFP8 MegaMoE "
                "(get_symm_buffer_for_mega_moe lacks mma_type)."
            )

    def _to_kernel_format(
        self, deep_gemm, weight: torch.Tensor, scale: torch.Tensor, num_groups: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Requantize ``(G, mn, k)`` block-FP8 weights; return them with the
        scales in DeepGEMM's packed, MN-major layout."""
        weight, sf = requant_block_fp8_to_ue8m0(weight, scale, self.weight_block_size)
        mn, k = weight.shape[-2:]
        sf = deep_gemm.transform_sf_into_required_layout(
            sf.contiguous(), mn, k, (1, _SF_GRAN_K), num_groups
        )
        return weight, sf

    def finalize_weights(self, shared_experts: nn.Module | None = None) -> None:
        """Turn the loaded checkpoint weights into the kernel's. Call once the
        whole checkpoint is loaded and before the generic post-load hooks touch
        the shared MLP."""
        if self._l1 is not None:
            return
        deep_gemm = self._deep_gemm()
        loader = self.routed_experts
        assert loader is not None
        self._check_runtime_supported(deep_gemm, loader.w13_weight.device)

        e = self.num_local_experts
        w13, w13_sf = self._to_kernel_format(
            deep_gemm, loader.w13_weight.data, loader.w13_weight_scale_inv.data, e
        )
        w2, w2_sf = self._to_kernel_format(
            deep_gemm, loader.w2_weight.data, loader.w2_weight_scale_inv.data, e
        )
        self._l1, self._l2 = deep_gemm.transform_weights_for_mega_moe(
            (w13, w13_sf), (w2, w2_sf)
        )
        # The kernel reads only the transformed tensors.
        self.routed_experts = None

        if shared_experts is None or self.num_shared_experts == 0:
            self.num_shared_experts = 0
            return
        self._finalize_shared_expert_weights(deep_gemm, shared_experts)

    def _finalize_shared_expert_weights(
        self, deep_gemm, shared_experts: nn.Module
    ) -> None:
        gate_up, down = shared_experts.gate_up_proj, shared_experts.down_proj

        def raw(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
            scale = getattr(linear, "weight_scale_inv", None)
            if scale is None:
                scale = getattr(linear, "weight_scale", None)
            weight = linear.weight
            mn, k = weight.shape
            block_m, block_k = self.weight_block_size
            if (
                scale is None
                or weight.dtype != torch.float8_e4m3fn
                or scale.dtype != torch.float32
                or tuple(scale.shape) != (-(-mn // block_m), -(-k // block_k))
            ):
                return None
            return weight.data, scale.data

        shared_i = self.intermediate_size * self.num_shared_experts
        gate_up_raw, down_raw = raw(gate_up), raw(down)
        if (
            gate_up_raw is None
            or down_raw is None
            or tuple(gate_up_raw[0].shape) != (2 * shared_i, self.hidden_size)
            or tuple(down_raw[0].shape) != (self.hidden_size, shared_i)
        ):
            logger.warning(
                "Disabling MegaMoE shared-expert fusion for %s: expected replicated "
                "block-FP8 gate_up/down weights with float32 %s scales.",
                self.prefix,
                self.weight_block_size,
            )
            self.num_shared_experts = 0
            return

        # Requantized copies: the shared MLP keeps its own storage, which the
        # generic post-load hooks may still rewrite.
        def to_kernel(pair: tuple[torch.Tensor, torch.Tensor]):
            weight, sf = self._to_kernel_format(
                deep_gemm, pair[0].unsqueeze(0), pair[1].unsqueeze(0), 1
            )
            return weight.squeeze(0), sf.squeeze(0)

        self._shared_l1, self._shared_l2 = deep_gemm.transform_weights_for_mega_moe(
            to_kernel(gate_up_raw), to_kernel(down_raw)
        )

    @property
    def has_fused_shared_experts(self) -> bool:
        return self._shared_l1 is not None

    # ------------------------------------------------------------- runtime
    def get_symm_buffer(self):
        deep_gemm = self._deep_gemm()
        group = get_ep_group().device_group
        num_shared = self.num_shared_experts if self.has_fused_shared_experts else 0
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            num_shared,
            MMA_TYPE,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=num_shared,
                mma_type=MMA_TYPE,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def update_expert_map(self) -> None:
        pass

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"GLM-5.3 MegaMoE got {num_tokens} tokens, but the symmetric buffer "
                f"was sized for {self.max_num_tokens}."
            )
        assert self._l1 is not None and self._l2 is not None, (
            "finalize_weights() has not run"
        )
        deep_gemm = self._deep_gemm()
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        symm_buffer = self.get_symm_buffer()

        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        shared_x_sf = None
        shared_block_m = None
        if self.has_fused_shared_experts:
            shared_x_sf = symm_buffer.shared_l1_acts_sf
            shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                get_ep_group().world_size,
                self.num_experts,
                symm_buffer.num_max_tokens_per_rank,
                num_tokens,
                self.top_k,
                MMA_TYPE,
            )

        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
            shared_x_sf=shared_x_sf,
            shared_block_m=shared_block_m,
        )

        kwargs = {}
        if self.has_fused_shared_experts:
            kwargs = dict(
                shared_l1_weights=self._shared_l1, shared_l2_weights=self._shared_l2
            )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._l1,
            self._l2,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            **kwargs,
        )
        return y


Glm5NextMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
