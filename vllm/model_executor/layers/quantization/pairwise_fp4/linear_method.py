# SPDX-License-Identifier: Apache-2.0
"""PairwiseFP4LinearMethod – online rotation + FP4 quantization."""

from __future__ import annotations

import os

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.activation_collector import (
    ActivationCollector,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.channel_monitor import (
    load_risk_scores,
    save_risk_scores,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.fp4_quant_policy import (
    estimate_global_scale,
    quantize_weight_to_fp4,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_applier import (
    apply_givens_rotation,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_plan import (
    RotationPlanBuilder,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
    RotationPlan,
    empty_angles,
    empty_pairs,
    load_plan,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)

# Number of forward passes used to collect activation statistics.
_WARMUP_SAMPLES = 8


class PairwiseFP4LinearMethod(QuantizeMethodBase):
    """Online pairwise-Givens-rotation + FP4 quantization.

    Loads BF16/FP16 weights, applies Givens rotations and quantizes to
    NVFP4 during ``process_weights_after_loading``.  At inference time,
    optionally rotates activations before calling the standard NVFP4 GEMM.

    For ``activation_only`` and ``joint`` modes, activation risk scores are
    collected during a warmup phase (first ``_WARMUP_SAMPLES`` forward
    passes).  Once collected, the activation rotation plan is built
    on-the-fly and cached to disk so that subsequent runs skip warmup.
    """

    def __init__(self, quant_config) -> None:
        # Avoid circular import – quant_config is PairwiseFP4Config
        self.quant_config = quant_config
        self.backend = select_nvfp4_linear_backend()

    # ------------------------------------------------------------------
    # create_weights
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size  # unused

        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size

        if input_size_per_partition % (group_size * 2) != 0:
            raise ValueError(
                f"input_size_per_partition ({input_size_per_partition}) must "
                f"be divisible by group_size*2 ({group_size * 2}) for FP4 "
                f"block quantization + packing."
            )

        # Store partition metadata on the layer for later use.
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # BF16 weight – will be rotated + quantised in
        # process_weights_after_loading.
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    # ------------------------------------------------------------------
    # process_weights_after_loading
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        cfg = self.quant_config
        mode = cfg.mode
        group_size = cfg.group_size
        weight = layer.weight.data.float()  # work in fp32

        layer_name = getattr(layer, "prefix", None) or getattr(
            layer, "_layer_name", "unknown",
        )
        # Store for later use during warmup finalization.
        layer._pairwise_layer_name = layer_name

        # ---- 1. Try to build full rotation plan ---------------------
        # For activation_only/joint: check if cached activation risk scores
        # exist.  If so, we can build the full plan now.  Otherwise we
        # defer plan building to the warmup phase.

        needs_act = mode in ("activation_only", "joint")
        act_cache_path = self._act_risk_cache_path(cfg, layer_name)
        act_risk_cached = None

        if needs_act and act_cache_path and os.path.isfile(act_cache_path):
            act_risk_cached, meta = load_risk_scores(act_cache_path)
            logger.info(
                "Loaded cached activation risk scores for %s "
                "(%d channels) from %s",
                layer_name, act_risk_cached.shape[0], act_cache_path,
            )

        plan: RotationPlan | None = None
        if cfg.use_prebuilt_plan and cfg.rotation_plan_path:
            plan = load_plan(cfg.rotation_plan_path)
        elif needs_act and act_risk_cached is None:
            # Cannot build activation plan yet — will do so after warmup.
            # Build weight-only plan for joint mode if possible.
            if mode == "joint":
                # Build a weight-only plan for now; will be rebuilt after
                # warmup with both sides.
                builder = RotationPlanBuilder(cfg.get_plan_builder_config())
                plan = builder.build(
                    layer_index=layer_name,
                    mode="weight_only",
                    weight=weight,
                    activation=None,
                )
            else:
                plan = None  # activation_only: defer entirely
        else:
            # weight_only, or activation data available from cache.
            builder = RotationPlanBuilder(cfg.get_plan_builder_config())
            # For cached activation data, synthesize a small tensor whose
            # risk scores match the cache.  RotationPlanBuilder._get_risk_scores
            # will pick up the cache file automatically.
            act_arg = None
            if act_risk_cached is not None:
                # Create a dummy activation whose per-channel max_abs matches
                # the cached risk scores, so the builder sees it as non-None
                # and computes/loads risk scores correctly.
                act_arg = act_risk_cached.unsqueeze(0)  # (1, C)
            plan = builder.build(
                layer_index=layer_name,
                mode=mode,
                weight=weight if mode in ("weight_only", "joint") else None,
                activation=act_arg,
            )

        # ---- 2. Rotate weight if applicable -------------------------
        if plan is not None and mode in ("weight_only", "joint") and not plan.is_empty:
            weight = apply_givens_rotation(weight, plan.pairs, plan.angles)

        # ---- 3. Quantize weight to packed FP4 -----------------------
        gs = estimate_global_scale(weight, block_size=group_size)
        packed_weight, block_scales_fp8, gs = quantize_weight_to_fp4(
            weight, gs, block_size=group_size,
        )

        # ---- 4. Set kernel-expected layer attributes ----------------
        layer.weight = Parameter(packed_weight, requires_grad=False)
        if self.backend == NvFp4LinearBackend.EMULATION:
            layer.weight_scale = Parameter(
                swizzle_blockscale(block_scales_fp8), requires_grad=False,
            )
        else:
            layer.weight_scale = Parameter(
                block_scales_fp8, requires_grad=False,
            )
        weight_global_scale = gs
        layer.weight_global_scale = Parameter(
            weight_global_scale.clone(), requires_grad=False,
        )

        input_global_scale = torch.tensor(1.0, dtype=torch.float32)
        layer.input_global_scale = Parameter(
            input_global_scale, requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / input_global_scale).to(torch.float32), requires_grad=False,
        )
        layer.alpha = Parameter(
            (input_global_scale / weight_global_scale).to(torch.float32),
            requires_grad=False,
        )

        # ---- 5. Convert to backend kernel format --------------------
        convert_to_nvfp4_linear_kernel_format(self.backend, layer)

        # ---- 6. Store rotation params for activation rotation -------
        if plan is not None and mode in ("activation_only", "joint") and not plan.is_empty:
            layer.register_buffer(
                "rotation_pairs", plan.pairs.to(layer.weight.device),
            )
            layer.register_buffer(
                "rotation_angles", plan.angles.to(layer.weight.device),
            )
        else:
            layer.register_buffer(
                "rotation_pairs", empty_pairs().to(layer.weight.device),
            )
            layer.register_buffer(
                "rotation_angles", empty_angles().to(layer.weight.device),
            )

        # ---- 7. Set up activation warmup if needed ------------------
        if needs_act and act_risk_cached is None:
            # No cached activation risk scores – enter warmup phase.
            input_c = layer.input_size_per_partition
            layer._act_collector = ActivationCollector(
                num_channels=input_c,
                risk_method=cfg.risk_method,
                warmup_samples=_WARMUP_SAMPLES,
                device=layer.weight.device,
            )
            layer._act_warmup_done = False
            logger.info(
                "Activation warmup enabled for %s (mode=%s, %d channels, "
                "%d samples needed)",
                layer_name, mode, input_c, _WARMUP_SAMPLES,
            )
        else:
            layer._act_collector = None
            layer._act_warmup_done = True

    # ------------------------------------------------------------------
    # apply (forward)
    # ------------------------------------------------------------------

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---- Activation warmup: collect stats -----------------------
        collector: ActivationCollector | None = getattr(
            layer, "_act_collector", None,
        )
        if collector is not None and not collector.ready:
            collector.update(x)
            if collector.ready:
                self._finalize_activation_plan(layer)

        # ---- Optionally rotate activations --------------------------
        rotation_pairs = getattr(layer, "rotation_pairs", None)
        if rotation_pairs is not None and rotation_pairs.numel() > 0:
            rotation_angles = layer.rotation_angles
            x = apply_givens_rotation(x, rotation_pairs, rotation_angles)

        return apply_nvfp4_linear(
            backend=self.backend,
            layer=layer,
            x=x,
            bias=bias,
        )

    # ------------------------------------------------------------------
    # Activation warmup finalization
    # ------------------------------------------------------------------

    def _finalize_activation_plan(self, layer: torch.nn.Module) -> None:
        """Build rotation plan from collected activation stats and cache."""
        cfg = self.quant_config
        mode = cfg.mode
        collector: ActivationCollector = layer._act_collector
        layer_name = getattr(layer, "_pairwise_layer_name", "unknown")

        act_risk = collector.finalize()
        logger.info(
            "Activation warmup complete for %s: %d channels, "
            "risk score range [%.4f, %.4f]",
            layer_name, act_risk.shape[0],
            act_risk.min().item(), act_risk.max().item(),
        )

        # Save activation risk scores to cache
        cache_path = self._act_risk_cache_path(cfg, layer_name)
        if cache_path:
            save_risk_scores(
                act_risk,
                cache_path,
                metadata={
                    "layer_index": layer_name,
                    "target": "activation",
                    "method": cfg.risk_method,
                    "shard_id": 0,
                    "num_channels": act_risk.shape[0],
                },
            )
            logger.info(
                "Saved activation risk scores to %s", cache_path,
            )

        # Build rotation plan using cached risk scores.
        # For joint mode we need weight risk too — load from cache
        # or compute from the dummy path.
        builder = RotationPlanBuilder(cfg.get_plan_builder_config())

        # Use the risk score tensor directly as a 1-row activation "tensor"
        # so that _get_risk_scores computes max_abs → identical values.
        act_tensor = act_risk.unsqueeze(0).to(layer.weight.device)

        if mode == "joint":
            # Weight risk: re-derive from builder's cache (was saved during
            # process_weights_after_loading's builder.build call).
            # If weight risk cache exists, builder._get_risk_scores will load it.
            # We create a dummy weight tensor from the risk scores too.
            w_risk_path = builder._risk_cache_path(layer_name, "weight")
            if w_risk_path and os.path.isfile(w_risk_path):
                w_risk, _ = load_risk_scores(w_risk_path)
                weight_tensor = w_risk.unsqueeze(0).to(layer.weight.device)
            else:
                # Fallback: use activation risk for both sides.
                weight_tensor = act_tensor
                logger.warning(
                    "No weight risk cache for %s; joint plan will "
                    "use activation risk for both sides", layer_name,
                )
        else:
            weight_tensor = None

        plan = builder.build(
            layer_index=layer_name,
            mode=mode,
            weight=weight_tensor,
            activation=act_tensor,
        )

        # Install plan on layer
        device = layer.weight.device
        if not plan.is_empty:
            layer.rotation_pairs = plan.pairs.to(device)
            layer.rotation_angles = plan.angles.to(device)
            logger.info(
                "Activation plan installed for %s: %d pairs, "
                "angle range [%.4f, %.4f] rad",
                layer_name, plan.num_pairs,
                plan.angles.min().item(), plan.angles.max().item(),
            )
        else:
            logger.info(
                "Activation plan for %s is empty (no pairs selected)",
                layer_name,
            )

        # Clean up collector
        layer._act_collector = None
        layer._act_warmup_done = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _act_risk_cache_path(cfg, layer_name: str) -> str:
        """Return file path for activation risk score cache."""
        cache_dir = cfg.risk_cache_dir
        if not cache_dir:
            return ""
        safe_name = layer_name.replace("/", "_").replace(".", "_")
        fname = f"{safe_name}__activation__{cfg.risk_method}.json"
        return os.path.join(cache_dir, fname)
