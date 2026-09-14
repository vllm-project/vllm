# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` expert module for DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
from vllm.utils.flashinfer_moe_ep import (
    build_fi_mega_layer,
    ensure_fi_moe_ep_runtime,
    fi_moe_ep_backend_spec,
    make_fi_moe_ep_bootstrap,
    mega_moe_weight_pack_from_params,
)

if TYPE_CHECKING:
    from flashinfer.moe_ep import MoEEpMegaLayer

    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP

_MOE_SKIP_PADDING: bool | None = None


def ckpt_uses_nvfp4_experts(vllm_config: VllmConfig) -> bool:
    """True when the loaded checkpoint quantizes experts with modelopt NVFP4
    (e2m1 + fp8-e4m3 per-16 block scales + per-tensor weight_scale_2), i.e.
    the recipe the nvfp4_cutedsl prequantized-weights path consumes verbatim.

    DeepSeek-V4 NVFP4-expert checkpoints declare ``moe_quant_algo: NVFP4`` in
    their ``quantization_config``, which DeepseekV4FP8Config surfaces as
    ``moe_quant_algo``.
    """
    return getattr(vllm_config.quant_config, "moe_quant_algo", "") == "NVFP4"


def resolve_mega_moe_is_padding(num_tokens: int) -> torch.Tensor | None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    global _MOE_SKIP_PADDING
    if _MOE_SKIP_PADDING is None:
        import vllm.envs as envs

        _MOE_SKIP_PADDING = bool(envs.VLLM_MOE_SKIP_PADDING)
    if not _MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None:
        return None
    return is_padding[:num_tokens]


def apply_mega_moe_routing_preprocess(
    topk_ids: torch.Tensor,
    *,
    is_padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Padding-only routing preprocess (EPLB hooks go here later)."""
    if is_padding is not None:
        topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
    return topk_ids


def nvfp4_prequant_pack_and_alphas(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,  # (E_local, 2) fp32: [:,0]=gate(w1), [:,1]=up(w3)
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,  # (E_local,) fp32
    *,
    intermediate_size: int,
):
    """NVFP4 checkpoint params -> (MoEWeightPack, fc1_alpha, fc2_alpha).

    Activation quantization is fully dynamic, so the checkpoint's static
    ``input_scale`` drops out and each GEMM's epilogue alpha reduces to the
    weight's per-tensor ``weight_scale_2``.

    fc1_alpha is one scalar per expert, but gate (w1) and up (w3) carry their
    own ``weight_scale_2``. When they differ, the ratio is folded into the up
    half's e4m3 block scales — only if the fold round-trips exactly, since a
    lossy rescale would silently change the model.
    """
    from flashinfer.moe_ep import MoEWeightPack

    inter = intermediate_size
    gate_s2 = w13_weight_scale_2[:, 0].float()
    up_s2 = w13_weight_scale_2[:, 1].float()
    if (gate_s2 <= 0).any() or (up_s2 <= 0).any() or (w2_weight_scale_2 <= 0).any():
        raise ValueError(
            "nvfp4 prequant: non-positive weight_scale_2 loaded — checkpoint "
            "scale tensors missing or loader routed them wrong."
        )

    w13_scale = w13_weight_scale
    if not torch.equal(gate_s2, up_s2):
        ratio = up_s2 / gate_s2  # (E_local,)
        up_sf = w13_scale[:, inter:, :].float()
        folded = up_sf * ratio[:, None, None]
        folded_e4m3 = folded.to(torch.float8_e4m3fn)
        if not torch.equal(folded_e4m3.float(), folded):
            raise ValueError(
                "NVFP4 MoE cannot merge the gate and up weight_scale_2 values "
                "because their ratio is not exactly representable in "
                "float8_e4m3fn. Use a checkpoint quantized with a shared "
                "gate/up weight_scale_2."
            )
        w13_scale = w13_scale.clone()
        w13_scale[:, inter:, :] = folded_e4m3

    fc1_alpha = gate_s2.clone().contiguous()
    fc2_alpha = w2_weight_scale_2.float().clone().contiguous()
    pack = MoEWeightPack(
        w13=w13_weight,
        w2=w2_weight,
        w13_scale=w13_scale,
        w2_scale=w2_weight_scale,
    )
    return pack, fc1_alpha, fc2_alpha


class DeepseekV4MegaMoEExpertsFI(DeepseekV4MegaMoEExperts):
    """Same weight layout/loader as the native mega experts, FI compute path."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        activation_clamp: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vllm_config, **kwargs)
        self._vllm_config = vllm_config
        self._activation_clamp = activation_clamp
        self._mega_layer: MoEEpMegaLayer | None = None
        self._fast_ctx: tuple[Any, Any, Any, int, bool] | None = None
        self._epilogue_alphas: tuple[torch.Tensor, torch.Tensor] | None = None
        self._nvfp4_prequant = ckpt_uses_nvfp4_experts(vllm_config)
        if self._nvfp4_prequant:
            megakernel = fi_moe_ep_backend_spec(
                vllm_config.kernel_config.moe_backend
            ).megakernel
            if megakernel != "nvfp4_cutedsl":
                raise ValueError(
                    "NVFP4-quantized expert checkpoint requires "
                    "moe_backend=flashinfer_moe_ep_mega_cutedsl, got a "
                    f"backend using megakernel {megakernel!r} "
                    "(deep_gemm consumes the MXFP4 checkpoint instead)."
                )
            self._realloc_nvfp4_params()

    def _realloc_nvfp4_params(self) -> None:
        """Swap the mx-recipe scale params for the NVFP4 checkpoint's:
        fp8-e4m3 per-16 block scales plus the per-tensor second-level
        scales (weight_scale_2, input_scale) modelopt exports."""
        n_e = self.num_local_experts
        inter = self.intermediate_size
        hidden = self.hidden_size
        attrs = {"weight_loader": self.weight_loader}

        def _param(shape: tuple, dtype: torch.dtype) -> nn.Parameter:
            p = nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(p, attrs)
            return p

        self.w13_weight_scale = _param(
            (n_e, 2 * inter, hidden // 16), torch.float8_e4m3fn
        )
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight_scale = _param((n_e, hidden, inter // 16), torch.float8_e4m3fn)
        self.w2_weight_scale.quant_method = "block"
        # (E, 2): column 0 = gate (w1), column 1 = up (w3).
        self.w13_weight_scale_2 = _param((n_e, 2), torch.float32)
        self.w2_weight_scale_2 = _param((n_e,), torch.float32)
        # Loaded for completeness; unused — activation quant is dynamic
        # (see nvfp4_prequant_pack_and_alphas).
        self.w13_input_scale = _param((n_e, 2), torch.float32)
        self.w2_input_scale = _param((n_e,), torch.float32)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        # NVFP4 checkpoint second-level scalars route here; everything
        # else (packed weights, block scales) matches the base layout.
        if "weight_scale_2" in weight_name or "input_scale" in weight_name:
            local_expert_ids = self._map_global_expert_id(expert_id)
            if not local_expert_ids:
                return False if return_success else None
            value = loaded_weight.reshape(()).to(torch.float32)
            for local_expert_id in local_expert_ids:
                if shard_id in ("w1", "w3"):
                    if "w13_" not in weight_name:
                        return False if return_success else None
                    param.data[local_expert_id, 0 if shard_id == "w1" else 1] = value
                elif shard_id == "w2":
                    if "w2_" not in weight_name:
                        return False if return_success else None
                    param.data[local_expert_id] = value
                else:
                    raise ValueError(f"Unsupported expert shard id: {shard_id}")
            return True if return_success else None
        return super().weight_loader(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success,
        )

    def finalize_weights(self, shared_experts: DeepseekV4MLP | None = None) -> None:
        # The FlashInfer megakernel has no shared-expert fusion; the caller's
        # serial shared MLP path handles shared_experts.
        if self._mega_layer is not None:
            return
        if self.w13_weight is None:
            return

        self._check_runtime_supported()
        ensure_fi_moe_ep_runtime(self._vllm_config)

        if self._nvfp4_prequant:
            # NVFP4 checkpoint: hand the packed weights + both scale
            # planes straight to the backend (no dequant->requant);
            # per-expert globals become fc1/fc2 epilogue alphas staged
            # at every forward via MoEEpTensors.
            weights, fc1_alpha, fc2_alpha = nvfp4_prequant_pack_and_alphas(
                self.w13_weight.data,
                self.w13_weight_scale.data,
                self.w13_weight_scale_2.data,
                self.w2_weight.data,
                self.w2_weight_scale.data,
                self.w2_weight_scale_2.data,
                intermediate_size=self.intermediate_size,
            )
            self._epilogue_alphas = (fc1_alpha, fc2_alpha)
        else:
            weights = mega_moe_weight_pack_from_params(
                self.w13_weight,
                self.w13_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
                megakernel=fi_moe_ep_backend_spec(
                    self._vllm_config.kernel_config.moe_backend
                ).megakernel,
            )
        self._mega_layer = build_fi_mega_layer(
            make_fi_moe_ep_bootstrap(),
            vllm_config=self._vllm_config,
            num_experts=self.num_experts,
            max_tokens_per_rank=self.max_num_tokens,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            activation_clamp=self._activation_clamp,
            weights=weights,
        )
        del weights
        # Allocate (or attach to) the pooled workspace before first
        # forward so warmup/capture never hits the lazy path.
        self._mega_layer._ensure_workspace()
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        if self._nvfp4_prequant:
            self.w13_weight_scale_2 = None
            self.w2_weight_scale_2 = None
            self.w13_input_scale = None
            self.w2_input_scale = None

    # EPLB is rejected for these backends in validate_fi_moe_ep_config, so
    # nothing constructs an EplbState and none of these run. They are kept
    # as explicit errors rather than deleted because inheriting the native
    # implementations would be worse: set_eplb_state would record a map
    # this forward path never consults, silently splitting routing from
    # weights, and get_expert_weights reads transformed-weight attributes
    # the flashinfer path releases after preprocess.
    _EPLB_UNSUPPORTED = (
        "EPLB is not supported with the flashinfer moe_ep backends; "
        "validate_fi_moe_ep_config should have rejected this "
        "configuration at startup."
    )

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def get_expert_weights(self) -> list[torch.Tensor]:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def update_expert_map(self) -> None:
        raise NotImplementedError(self._EPLB_UNSUPPORTED)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        # fast_math is a native deep_gemm knob; the FI kernels have no
        # equivalent toggle, so it is accepted for signature parity only.
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )

        from flashinfer.moe_ep import MoEEpTensors

        num_tokens = hidden_states.shape[0]
        is_padding = resolve_mega_moe_is_padding(num_tokens)
        topk_ids = apply_mega_moe_routing_preprocess(
            topk_ids,
            is_padding=is_padding,
        )

        # Fast path: after the first successful full forward the layer is
        # immutable, so skip MoEEpMegaLayer.forward()'s per-call validation
        # and go straight to the kernel backend's stage_inputs + compute.
        alphas = self._epilogue_alphas
        fc1_alpha = alphas[0] if alphas is not None else None
        fc2_alpha = alphas[1] if alphas is not None else None

        fast = self._fast_ctx
        if fast is not None:
            kernel, workspace, transformed, hidden_size, zero_copy = fast
            t = MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
            )
            kernel.stage_inputs(t, workspace, quantize_input=True)
            if zero_copy:
                # Zero-copy (cutedsl backends): consume the workspace [:n]
                # view directly — valid under stream ordering until the next
                # MoE layer's launch on the shared workspace.
                return kernel.compute(workspace, transformed, output=None)
            # deep_gemm_mega's compute() requires a real output tensor.
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            return kernel.compute(workspace, transformed, output=out)

        ensure_fi_moe_ep_runtime(self._vllm_config)
        self.finalize_weights()
        assert self._mega_layer is not None

        y = self._mega_layer.forward(
            MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
            )
        )
        layer = self._mega_layer
        if hidden_states.dtype == torch.bfloat16:
            self._fast_ctx = (
                layer._kernel,
                layer._ensure_workspace(),
                layer._transformed,
                layer._fleet_params.token_hidden_size,
                # zero-copy output views are a cutedsl-backend contract
                layer._kernel.kernel_name() != "deep_gemm_mega",
            )
        return y


DeepseekV4MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
