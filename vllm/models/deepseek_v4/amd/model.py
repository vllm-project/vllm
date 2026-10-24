# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import os
import typing
import weakref
from collections.abc import Callable, Iterable
from dataclasses import replace
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    FusedMoEQuantConfig,
    GateLinear,
    RoutedExperts,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_fused_experts,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
    _forward_allows_fused_shared_append,
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
    resolve_layer_fused_shared_expert,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_TILELANG_MHC,
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.amd.mega_moe import (
    DeepseekV4AiterMegaMoEExperts,
    finalize_mega_moe_weights,
)
from vllm.models.deepseek_v4.amd.rocm import DeepseekV4ROCMAiterMLAAttention
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950, on_gfx1250
from vllm.sequence import IntermediateTensors
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream, current_stream
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

logger = init_logger(__name__)


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

        # gate_up_proj B-preshuffle (ColumnParallel -> no all-reduce); set at load.
        self._gateup = rocm_aiter_ops.is_enabled()
        # Block scale for the preshuffled gate_up weight; None = not preshuffled.
        self._gateup_scale: torch.Tensor | None = None

    def prepare_gateup_preshuffle(self) -> None:
        # B-preshuffle the gate_up_proj weight in place (single weight).
        if not self._gateup:
            return
        # aiter's gemm_a8w8_blockscale_bpreshuffle currently disabled on gfx1250.
        # Leaving _gateup_scale as None -> "not preshuffled"
        # forward() falls back to the standard gate_up_proj linear
        if on_gfx1250():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
            get_fp8_block_weight_scale,
        )
        from vllm.model_executor.utils import replace_parameter

        w = getattr(self.gate_up_proj, "weight", None)
        ws = get_fp8_block_weight_scale(self.gate_up_proj)
        if w is None or ws is None or w.dim() != 2:
            return
        # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
        if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
            return
        if ws.dtype == torch.float8_e8m0fnu:
            ws = _upcast_e8m0_to_fp32(ws).contiguous()
        # Skip when the linear kernel already B-preshuffled this weight.
        if not getattr(self.gate_up_proj, "aiter_bpreshuffled", False):
            replace_parameter(
                self.gate_up_proj,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
        self._gateup_scale = ws

    def forward(self, x):
        if self._gateup_scale is not None and x.dim() == 2:
            # gate_up via fp8 group-quant (col-major) + B-preshuffle GEMM.
            x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
            gate_up = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
                x_fp8,
                self.gate_up_proj.weight,
                x_scale,
                self._gateup_scale,
                output_dtype=x.dtype,
            )
        else:
            gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _use_heterogeneous_fhmoe(num_tokens: int) -> bool:
    return num_tokens > 0 and (
        rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert(num_tokens)
        is True
    )


def _validate_heterogeneous_routes(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_per_token: int,
) -> None:
    if (
        topk_weights.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError(
            "Heterogeneous fused MoE routes must have equal two-dimensional shapes."
        )
    expected_columns = experts_per_token + 1
    if topk_weights.shape[1] != expected_columns:
        raise ValueError(
            "Heterogeneous fused MoE routes must have exactly "
            f"{expected_columns} columns, got {topk_weights.shape[1]}."
        )


def _heterogeneous_shared_expert_min_tokens() -> int:
    """Minimum batch size (rows entering the MoE) for the fused
    heterogeneous FHMoE path.

    At or above this size the fused call wins: the fixed overhead of
    AITER's mixed FP8/MXFP4 kernel variants amortizes, and fusing removes
    the separate shared-expert launches and aux-stream syncs from the
    launch-bound prefill chain. Below it those variants regress, so the
    stock overlapped separate shared expert runs instead. 0 restores the
    always-fused behavior.
    """
    raw = os.environ.get("VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS", "16")
    try:
        value = int(raw)
    except ValueError:
        logger.warning_once(
            "VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS must be an integer, got "
            "%r; using 16.",
            raw,
        )
        return 16
    if value < 0:
        logger.warning_once(
            "VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS must be >= 0, got %d; using 0.",
            value,
        )
        return 0
    return value


def _heterogeneous_shared_expert_enabled(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    parallel_config = vllm_config.parallel_config
    reasons: list[str] = []
    mono_prefill_fusion_armed = False

    # Kill switch shared with the FP8->MXFP4 requant fusion: the env var
    # restores the separate shared-expert path everywhere (V4 and V4.1
    # target layers, and the DSpark draft layers).
    if os.environ.get("VLLM_DISABLE_FUSED_SHARED_EXPERTS", "0") == "1":
        logger.warning_once(
            "Fused shared experts (heterogeneous native-FP8) is DISABLED (env override)"
        )
        return False

    # Mono decode kernels hard-require the stock 384/6 structure (separate
    # shared experts, no expert-table remap), so the hetero construction
    # coexists with --monokernel only because every module mono reads stays
    # stock: the appended shared slot sits past the routed experts and is
    # sliced off for the mono weight views (mono_stock_moe_views). The
    # fused hetero FHMoE engages only for prefill-shaped batches (the
    # router's prefill-only append plus
    # VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS), so mono decode steps and the
    # fusion never share a step. VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO=1
    # disables the fusion under --monokernel.
    if vllm_config.kernel_config.monokernel:
        if os.environ.get("VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO", "0") == "1":
            logger.info_once(
                "Fused shared experts (heterogeneous native-FP8) yields to "
                "--monokernel decode (VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO=1; "
                "the mono kernels require the stock 384/6 MoE structure)"
            )
            return False
        # Armed: fall through to the full eligibility chain below. A
        # blocked chain logs "ARMED but unavailable" instead of the
        # engagement warning, which fires only on success. The AITER
        # coordination flag the chain checks next is defaulted ON by the
        # platform for supported checkpoints (rocm.py
        # _maybe_default_fused_shared_experts).
        mono_prefill_fusion_armed = True

    if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
        if mono_prefill_fusion_armed:
            logger.warning_once(
                "prefill fusion above mono ARMED but unavailable: "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is off (the "
                "AITER router/kernel coordination flag the heterogeneous "
                "construction requires). Continuing with the stock "
                "shared-expert path above mono (decode stays mono)."
            )
        return False
    if not on_gfx950():
        reasons.append("the device is not gfx950")
    if parallel_config.enable_expert_parallel:
        reasons.append("expert parallelism is enabled")
    if parallel_config.enable_eplb:
        reasons.append("EPLB is enabled")
    if parallel_config.data_parallel_size != 1:
        reasons.append("data parallelism is enabled")
    if parallel_config.prefill_context_parallel_size != 1:
        reasons.append("prefill context parallelism is enabled")
    if vllm_config.kernel_config.moe_backend != "aiter":
        reasons.append("the MoE backend is not AITER")
    if vllm_config.model_config.dtype != torch.bfloat16:
        reasons.append("the model dtype is not BF16")

    # Two supported shape classes:
    #   dsv4        - big DeepSeek-V4: TP8, block-128 FP8 shared experts,
    #                 AITER's CSV-backed DSV4 I384 FHMoE contract, no
    #                 intermediate padding.
    #   dsv41_flash - DeepSeek-V4.1-Flash: TP4, block-32 FP8 shared experts,
    #                 AITER's default-config two-stage FHMoE path with the
    #                 MXFP4 128-multiple intermediate padding (576 -> 640 per
    #                 rank). The shared expert keeps its native FP8 weights:
    #                 no FP8->MXFP4 requant, so neither the routed-expert nor
    #                 the shared-expert numerics change.
    common_model: dict[str, typing.Any] = {
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "hidden_act": "silu",
        "topk_method": "noaux_tc",
    }
    shape_classes = {
        "dsv4": {
            "hidden_size": 7168,
            "moe_intermediate_size": 3072,
        },
        "dsv41_flash": {
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
        },
    }
    shape_class = None
    for name, sizes in shape_classes.items():
        if all(
            getattr(config, key, None) == value
            for key, value in {**common_model, **sizes}.items()
        ):
            shape_class = name
            break
    if shape_class is None:
        reasons.append("the model is not a supported heterogeneous shared-expert shape")
    elif shape_class == "dsv4":
        if getattr(config, "expert_dtype", None) != "fp4":
            reasons.append("expert_dtype is not 'fp4'")
        if parallel_config.tensor_parallel_size != 8:
            reasons.append("tensor parallelism is not 8")
        if rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert(1) is not True:
            reasons.append(
                "AITER lacks CSV-backed DSV4 I384 heterogeneous shared-expert support"
            )
    else:  # dsv41_flash
        # Flash's text config does not carry expert_dtype; the quant config
        # resolves the same default ("fp4") for FP4-expert checkpoints.
        # Each early rejection skips the AITER shape probe: there is no
        # point asking about configurations already turned down.
        tp_size = parallel_config.tensor_parallel_size
        moe_intermediate = shape_classes[shape_class]["moe_intermediate_size"]
        if getattr(config, "expert_dtype", "fp4") != "fp4":
            reasons.append("expert_dtype is not 'fp4'")
        elif tp_size != 4:
            reasons.append("tensor parallelism is not 4")
        elif moe_intermediate % tp_size != 0:
            reasons.append("the MoE intermediate size does not shard evenly")
        else:
            inter_per_rank = moe_intermediate // tp_size
            padded_inter = (inter_per_rank + 127) // 128 * 128
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if (
                rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert_shape(
                    max_tokens,
                    shape_classes[shape_class]["hidden_size"],
                    padded_inter,
                    common_model["n_routed_experts"] + 1,
                    common_model["num_experts_per_tok"] + 1,
                    padded_inter - inter_per_rank,
                )
                is not True
            ):
                reasons.append(
                    "AITER's default FHMoE configs do not cover this "
                    "DeepSeek-V4.1-Flash TP4 shape"
                )

    if quant_config is None or quant_config.get_name() != "deepseek_v4_fp8":
        reasons.append("the DeepSeek V4 FP8 quantization config is not active")
    else:
        if getattr(quant_config, "moe_quant_algo", "").upper() == "NVFP4":
            reasons.append("routed experts use NVFP4")
        expected_block = [32, 32] if shape_class == "dsv41_flash" else [128, 128]
        if getattr(quant_config, "weight_block_size", None) != expected_block:
            reasons.append(f"shared experts are not block-{expected_block[0]} FP8")
        if not getattr(quant_config, "is_checkpoint_fp8_serialized", False):
            reasons.append("shared experts are not serialized as FP8")
        if not getattr(quant_config, "is_scale_e8m0", False):
            reasons.append("shared-expert scales are not E8M0")
        if getattr(quant_config, "ignored_layers", None):
            reasons.append("the quantization config has ignored layers")

    offload_config = getattr(vllm_config, "offload_config", None)
    if offload_config is not None and (
        getattr(getattr(offload_config, "uva", None), "cpu_offload_gb", 0) > 0
        or getattr(
            getattr(offload_config, "prefetch", None),
            "offload_group_size",
            0,
        )
        > 0
    ):
        reasons.append("weight offloading is enabled")

    if reasons:
        if mono_prefill_fusion_armed:
            logger.warning_once(
                "prefill fusion above mono ARMED but unavailable: %s. "
                "Continuing with the stock shared-expert path above mono "
                "(decode stays mono).",
                "; ".join(reasons),
            )
        logger.debug_once(
            "DeepSeek V4 heterogeneous shared-expert fusion is unavailable: %s. "
            "Continuing with the existing shared-expert path.",
            "; ".join(reasons),
        )
        return False

    if shape_class == "dsv41_flash":
        if mono_prefill_fusion_armed:
            logger.warning_once(
                "prefill fusion above mono engaged: the heterogeneous fused "
                "shared experts (native-FP8) run above --monokernel decode "
                "(prefill-shaped batches of >= %d tokens only; mono decode "
                "keeps the stock 384/6 kernels on decode-only steps; disable "
                "with VLLM_DISABLE_PREFILL_FUSION_ABOVE_MONO=1)",
                _heterogeneous_shared_expert_min_tokens(),
            )
        logger.warning_once(
            "Fused shared experts (heterogeneous native-FP8) is ENABLED for "
            "this DeepSeek-V4.1-Flash checkpoint (AITER FHMoE; the shared "
            "expert keeps its native FP8 block-32 weights - no requant). "
            "Fused execution engages for prefill batches of >= %d tokens "
            "(VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS); decode forwards and "
            "smaller batches keep the overlapped separate shared expert",
            _heterogeneous_shared_expert_min_tokens(),
        )
    else:
        logger.info_once(
            "Using AITER DeepSeek V4 heterogeneous MoE for CSV-covered MoE "
            "input rows (M) per invocation (native intermediate width 384)."
        )
    return True


def _shared_expert_block_scales(linear: nn.Module) -> torch.Tensor:
    """Block-32 scale tensor for a shared-expert linear, (N // 32, K // 32).

    Two checkpoint formats reach the heterogeneous prepare step:
    * DeepSeek V4 (Fp8Config) keeps ``weight_scale_inv`` block scales
      ((N // B, K // B) with B in {32, 128}).
    * DeepSeek V4.1-Flash routes the shared linears through ModelOpt's
      kMxfp8Static scheme: the checkpoint's (32, 32) block scales are
      expanded at load time into the per-row MXFP8 layout (N, K // 32),
      and the linear kernel's post-load folds them back to
      (N // 32, K // 32). Both landings are accepted here.
    """
    scale = getattr(linear, "weight_scale_inv", None)
    if scale is not None:
        return scale
    scale = linear.weight_scale
    if scale.ndim != 2:
        raise ValueError(f"Unexpected shared-expert scale layout {tuple(scale.shape)}.")
    rows, k_blocks = scale.shape
    out_rows = linear.weight.shape[0]
    if rows == out_rows // 32:
        # already folded by the linear kernel's post-load
        return scale
    if rows == out_rows:
        # per-row MXFP8 layout: fold, verifying the block uniformity the
        # load-time expansion guarantees
        blocks = scale.view(rows // 32, 32, k_blocks)
        folded = blocks[:, 0, :].contiguous()
        if not torch.equal(blocks, folded.unsqueeze(1).expand_as(blocks)):
            raise ValueError(
                "The shared-expert per-row scales are not uniform inside "
                "their 32-row blocks; the block scales cannot be recovered."
            )
        return folded
    raise ValueError(f"Unexpected shared-expert scale layout {tuple(scale.shape)}.")


@torch.no_grad()
def _prepare_native_fp8_shared_expert(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    intermediate_size: int,
    padded_intermediate_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand native block-128/block-32 E8M0 scales to FHMoE's 1x32 layout.

    ``intermediate_size`` is the true per-rank shared-expert width (half the
    ``gate_up_proj`` rows). ``padded_intermediate_size`` is the allocated
    routed-expert width (MXFP4 rounds it up to a 128 multiple, e.g.
    DeepSeek-V4.1-Flash TP4: 576 -> 640); when larger, the gate/up slabs and
    the W2 columns are zero-padded so the shared expert matches the FHMoE
    contract, which sizes it off the routed tensors.
    """
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError("Heterogeneous shared-expert weights must be FP8 E4M3.")
    hidden_size = w13.shape[1]
    if w13.shape != (2 * intermediate_size, hidden_size):
        raise ValueError(f"Unexpected shared W13 shape {tuple(w13.shape)}.")
    if w2.shape != (hidden_size, intermediate_size):
        raise ValueError(f"Unexpected shared W2 shape {tuple(w2.shape)}.")

    def scale_bytes(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.float8_e8m0fnu:
            return scale.contiguous().view(torch.uint8)
        if scale.dtype == torch.uint8:
            return scale.contiguous()
        if scale.dtype == torch.float32:
            bits = scale.contiguous().view(torch.int32)
            exponent = (bits >> 23).to(torch.uint8)
            if not torch.equal(bits, exponent.to(torch.int32) << 23):
                raise ValueError("FP32 shared-expert scales are not exact E8M0.")
            return exponent
        raise ValueError(f"Unsupported shared-expert scale dtype {scale.dtype}.")

    w13_scale = scale_bytes(w13_scale)
    w2_scale = scale_bytes(w2_scale)

    block = None
    for candidate in (128, 32):
        if tuple(w13_scale.shape) == (
            2 * intermediate_size // candidate,
            hidden_size // candidate,
        ) and tuple(w2_scale.shape) == (
            hidden_size // candidate,
            intermediate_size // candidate,
        ):
            block = candidate
            break
    if block is None:
        raise ValueError(
            f"Shared-expert scales {tuple(w13_scale.shape)}/"
            f"{tuple(w2_scale.shape)} match neither the block-128 nor the "
            "block-32 FP8 layout."
        )

    if block == 128:
        w13_scale = w13_scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
        w2_scale = w2_scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    else:
        # Block-32 scales: rows expand 32x; columns are already 1x32.
        w13_scale = w13_scale.repeat_interleave(32, dim=0)
        w2_scale = w2_scale.repeat_interleave(32, dim=0)

    if padded_intermediate_size is None:
        padded_intermediate_size = intermediate_size
    if (
        padded_intermediate_size < intermediate_size
        or (padded_intermediate_size - intermediate_size) % 32 != 0
    ):
        raise ValueError(
            f"Padded intermediate size {padded_intermediate_size} is "
            f"incompatible with intermediate size {intermediate_size}."
        )
    if padded_intermediate_size > intermediate_size:
        # Pad the gate and up slabs separately: the GUGU interleave applied
        # by shuffle_weight_a16w4 / shuffle_scale_a16w4 splits the rows as
        # [gate rows; up rows], so a plain bottom append would corrupt the
        # slab boundaries.
        padded_w13 = w13.new_zeros((2, padded_intermediate_size, hidden_size))
        padded_w13[:, :intermediate_size].copy_(
            w13.view(2, intermediate_size, hidden_size)
        )
        w13 = padded_w13.view(2 * padded_intermediate_size, hidden_size)

        w13_scale_cols = w13_scale.shape[1]
        padded_w13_scale = w13_scale.new_full(
            (2, padded_intermediate_size, w13_scale_cols), 0x7F
        )
        padded_w13_scale[:, :intermediate_size].copy_(
            w13_scale.view(2, intermediate_size, w13_scale_cols)
        )
        w13_scale = padded_w13_scale.view(2 * padded_intermediate_size, w13_scale_cols)

        padded_w2 = w2.new_zeros((hidden_size, padded_intermediate_size))
        padded_w2[:, :intermediate_size].copy_(w2)
        w2 = padded_w2

        padded_w2_scale = w2_scale.new_full(
            (hidden_size, padded_intermediate_size // 32), 0x7F
        )
        padded_w2_scale[:, : intermediate_size // 32].copy_(w2_scale)
        w2_scale = padded_w2_scale

    scale_cols = ((w2_scale.shape[1] + 7) // 8) * 8
    if scale_cols != w2_scale.shape[1]:
        padded_w2_scale = torch.full(
            (w2_scale.shape[0], scale_cols),
            0x7F,
            dtype=torch.uint8,
            device=w2_scale.device,
        )
        padded_w2_scale[:, : w2_scale.shape[1]].copy_(w2_scale)
        w2_scale = padded_w2_scale

    return (
        w13.unsqueeze(0),
        w2.unsqueeze(0),
        w13_scale.view(torch.float8_e8m0fnu),
        w2_scale.view(torch.float8_e8m0fnu),
    )


class DeepseekV4HeterogeneousMxfp4MoEMethod(Mxfp4MoEMethod):
    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        assert isinstance(layer, DeepseekV4HeterogeneousSharedRoutedExperts)
        layer.prepare_heterogeneous_shared_expert()


class DeepseekV4HeterogeneousSharedRoutedExperts(RoutedExperts):
    def __init__(
        self,
        *args,
        shared_expert: DeepseekV4MLP,
        shared_expert_id: int,
        **kwargs,
    ):
        self._shared_expert_ref = weakref.ref(shared_expert)
        self.shared_expert_id = shared_expert_id
        self._routed_quant_config: FusedMoEQuantConfig | None = None
        super().__init__(*args, **kwargs)
        if self.expert_map_manager.num_fused_shared_experts != 1:
            raise ValueError("Heterogeneous fusion requires one appended expert.")
        if self.shared_expert_id != self.global_num_experts:
            raise ValueError("The shared expert ID must name the appended slot.")
        self.register_buffer("shared_w1", None, persistent=False)
        self.register_buffer("shared_w2", None, persistent=False)
        self.register_buffer("shared_w1_scale", None, persistent=False)
        self.register_buffer("shared_w2_scale", None, persistent=False)
        # Decode-path shared-expert overlap (see forward_modular): mirrors
        # MoERunner's SharedExperts aux-stream execution. The stream and
        # events are created here, before any CUDA graph capture, and are
        # reused across captured graphs (the same pattern the stock
        # SharedExperts runner uses). The separate shared-expert output is
        # consumed within the same forward call, so a single event pair
        # stays correct under sequential DBO ubatches as well.
        self._overlap_stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available() and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            from vllm.utils.torch_utils import aux_stream

            self._overlap_stream = aux_stream()
            if self._overlap_stream is not None:
                self._overlap_input_ready = torch.cuda.Event()
                self._overlap_output_ready = torch.cuda.Event()
        # Log which decode layout runs below the append threshold;
        # warning_once dedups across the per-layer instances.
        if self._overlap_stream is not None:
            logger.warning_once(
                "Fused shared experts decode layout: decode forwards and "
                "batches below VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS run the "
                "separate shared expert overlapped on the aux stream "
                "(disable with VLLM_DISABLE_SHARED_EXPERTS_STREAM=1)"
            )
        else:
            logger.warning_once(
                "Fused shared experts decode layout: decode forwards and "
                "batches below VLLM_HETERO_SHARED_EXPERT_MIN_TOKENS run the "
                "separate shared expert serially (overlap unavailable: no "
                "CUDA, or VLLM_DISABLE_SHARED_EXPERTS_STREAM=1)"
            )

    def _get_quant_method(self, prefix, quant_config, moe_config):
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)
        if not isinstance(quant_method, Mxfp4MoEMethod):
            raise ValueError("Heterogeneous fusion requires MXFP4 routed experts.")
        return DeepseekV4HeterogeneousMxfp4MoEMethod(moe_config)

    @torch.no_grad()
    def prepare_heterogeneous_shared_expert(self) -> None:
        shared_expert = self._shared_expert_ref()
        if shared_expert is None:
            raise RuntimeError("The native shared-expert module was released.")

        # The shared module carries the true per-rank width; the routed
        # buffers carry the MXFP4-rounded width (DeepSeek-V4.1-Flash TP4:
        # 576 -> 640). FHMoE sizes the shared expert off the routed tensors,
        # so the shared weights are padded to match (see
        # _prepare_native_fp8_shared_expert).
        shared_intermediate = shared_expert.gate_up_proj.weight.shape[0] // 2
        padded_intermediate = self.w13_weight.shape[1] // 2
        if padded_intermediate != self.moe_config.intermediate_size_per_partition:
            raise ValueError(
                "The routed-expert intermediate width "
                f"{self.moe_config.intermediate_size_per_partition} does not "
                f"match the routed weight buffers ({padded_intermediate})."
            )
        shared = _prepare_native_fp8_shared_expert(
            shared_expert.gate_up_proj.weight,
            shared_expert.down_proj.weight,
            _shared_expert_block_scales(shared_expert.gate_up_proj),
            _shared_expert_block_scales(shared_expert.down_proj),
            shared_intermediate,
            padded_intermediate_size=padded_intermediate,
        )
        self.shared_w1 = rocm_aiter_ops.shuffle_weight_a16w4(shared[0], 16, True)
        self.shared_w2 = rocm_aiter_ops.shuffle_weight_a16w4(shared[1], 16, False)
        self.shared_w1_scale = rocm_aiter_ops.shuffle_scale_a16w4(shared[2], 1, True)
        self.shared_w2_scale = rocm_aiter_ops.shuffle_scale(shared[3])

        # Per-invocation dispatch: the DSV4 I384 contract is served from
        # AITER's dedicated FHMoE CSV; every other shape (e.g.
        # DeepSeek-V4.1-Flash TP4) goes through the default-config two-stage
        # path, probed per M against the scheduler's token budget.
        model_dim = self.w2_weight.shape[1]
        experts = self.w13_weight.shape[0]
        topk = self.moe_config.experts_per_token + 1
        self._fhmoe_csv_dispatch = (
            model_dim,
            padded_intermediate,
            experts,
            topk,
        ) == (7168, 384, 385, 7)
        self._fhmoe_shape_key = (
            model_dim,
            padded_intermediate,
            experts,
            topk,
            padded_intermediate - shared_intermediate,
        )

        self._ensure_moe_quant_config_init()
        quant_config = self.quant_method.moe_quant_config
        if quant_config is None:
            raise RuntimeError("The routed MXFP4 quantization config is unavailable.")
        if quant_config.w1_scale is None or quant_config.w2_scale is None:
            raise RuntimeError("The routed MXFP4 scales are unavailable.")

        def routed_scales(scale: torch.Tensor) -> torch.Tensor:
            scale = scale.reshape(self.local_num_experts, -1, scale.shape[-1])[
                : self.shared_expert_id
            ]
            return scale.reshape(-1, scale.shape[-1]).contiguous()

        self._routed_quant_config = replace(
            quant_config,
            _w1=replace(
                quant_config._w1,
                scale=routed_scales(quant_config.w1_scale),
            ),
            _w2=replace(
                quant_config._w2,
                scale=routed_scales(quant_config.w2_scale),
            ),
        )

    def mono_stock_moe_views(self) -> tuple[torch.Tensor, ...]:
        """The stock 384-expert views of the routed buffers for the mono
        decode kernels.

        The appended shared slot (index ``shared_expert_id``, past every
        routed expert) is sliced off; the routed rows and their AITER A8W4
        layout are exactly the stock ones. The appended slot's own weights
        are never read by any path -- the fused hetero call passes the
        native-FP8 shared buffers separately, and the routed-only paths
        slice the slot off the same way -- so the mono kernels reading
        these views see byte-identical weights to a non-hetero build.
        """
        n = self.shared_expert_id
        rows = self.w13_weight.shape[0]
        if n <= 0 or rows != n + 1:
            raise RuntimeError(
                "prefill fusion above mono: the routed buffers are not the "
                "expected (routed + 1 appended) layout (w13 rows "
                f"{rows}, shared_expert_id {n})"
            )
        cols = self.w13_weight_scale.shape[-1]
        w13_s = (
            self.w13_weight_scale.reshape(n + 1, -1, cols)[:n]
            .reshape(-1, cols)
            .contiguous()
        )
        dcols = self.w2_weight_scale.shape[-1]
        w2_s = (
            self.w2_weight_scale.reshape(n + 1, -1, dcols)[:n]
            .reshape(-1, dcols)
            .contiguous()
        )
        w13 = self.w13_weight[:n]
        w2 = self.w2_weight[:n]
        w13.is_shuffled = True
        w2.is_shuffled = True
        return w13, w13_s, w2, w2_s

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shared_experts is not None or shared_experts_input is not None:
            raise ValueError("The heterogeneous expert owns the shared path.")
        if (
            topk_ids.ndim == 2
            and topk_ids.shape[-1] == self.moe_config.experts_per_token
        ):
            # The router skipped the shared-expert append for this
            # decode-sized batch (below the model's
            # shared_expert_append_min_tokens): keep the stock decode
            # layout -- routed experts through the plain kernel with the
            # separate shared expert overlapped on the aux stream.
            return self._forward_overlapped_separate(x, topk_weights, topk_ids)
        if (
            topk_ids.ndim == 2
            and topk_ids.shape[-1] == self.moe_config.experts_per_token + 1
            and not _forward_allows_fused_shared_append()
        ):
            # Belt-and-suspenders for the prefill-only policy: a width-(K+1)
            # route arrived during an unambiguous decode forward (a router
            # path that appended before consulting the mode gate, or a
            # caller that appends unconditionally). Slice the appended
            # shared slot back off and keep the stock decode layout instead
            # of running the fused hetero FHMoE on decode tokens. The
            # router renormalizes the routed top-k BEFORE appending, so the
            # slice is bit-identical to what a no-append router emits.
            return self._forward_overlapped_separate(
                x,
                topk_weights[:, :-1].contiguous(),
                topk_ids[:, :-1].contiguous(),
            )
        _validate_heterogeneous_routes(
            topk_weights,
            topk_ids,
            self.moe_config.experts_per_token,
        )
        if any(
            tensor is None
            for tensor in (
                self.shared_w1,
                self.shared_w2,
                self.shared_w1_scale,
                self.shared_w2_scale,
            )
        ):
            raise RuntimeError("Heterogeneous shared weights were not prepared.")

        if self._fhmoe_csv_dispatch:
            use_fhmoe = _use_heterogeneous_fhmoe(x.shape[0])
        else:
            use_fhmoe = (
                rocm_aiter_ops.fused_moe_supports_heterogeneous_shared_expert_shape(
                    x.shape[0], *self._fhmoe_shape_key
                )
                is True
            )
        if use_fhmoe:
            self._ensure_moe_quant_config_init()
            quant_config = self.quant_method.moe_quant_config
            if quant_config is None:
                raise RuntimeError("The routed MXFP4 config is unavailable.")
            return rocm_aiter_fused_experts(
                hidden_states=x,
                w1=self.w13_weight,
                w2=self.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=self.moe_config,
                activation=self.activation,
                expert_mask=self.expert_map,
                quant_config=quant_config,
                output_dtype=x.dtype,
                moe_sorting_dispatch_policy=(rocm_aiter_ops.get_moe_dispatch_policy()),
                shared_w1=self.shared_w1,
                shared_w2=self.shared_w2,
                shared_w1_scale=self.shared_w1_scale,
                shared_w2_scale=self.shared_w2_scale,
                shared_expert_id=self.shared_expert_id,
            )

        return self._forward_separate_shared(x, topk_weights, topk_ids)

    def _forward_routed_experts_only(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the routed experts through the plain (non-heterogeneous)
        AITER kernel: E=384 sliced weights, routed-width routes. The caller
        owns the separate shared-expert contribution."""
        if self._routed_quant_config is None:
            raise RuntimeError("The routed-only shared-expert path is unavailable.")
        routed_w1 = self.w13_weight[: self.shared_expert_id]
        routed_w2 = self.w2_weight[: self.shared_expert_id]
        routed_w1.is_shuffled = True
        routed_w2.is_shuffled = True
        return rocm_aiter_fused_experts(
            hidden_states=x,
            w1=routed_w1,
            w2=routed_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=self.moe_config,
            activation=self.activation,
            expert_mask=self.expert_map,
            quant_config=self._routed_quant_config,
            output_dtype=x.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        )

    def _forward_overlapped_separate(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Stock decode layout for decode-sized batches.

        The separate shared expert runs on the aux stream (mirroring
        SharedExperts.maybe_forward_async: record input_ready on the current
        stream, run the shared MLP on the aux stream, record output_ready)
        while the routed experts run through the plain kernel on the current
        stream; the streams join before the sum. AITER's heterogeneous
        variants regress at decode-sized M while this overlap hides the
        separate shared expert under the routed chain, so decode keeps this
        layout. Returns the TP-partial sum; the caller's late all-reduce
        reduces it, exactly like the stock path."""
        shared_expert = self._shared_expert_ref()
        if shared_expert is None:
            raise RuntimeError("The native shared-expert module was released.")
        if self._overlap_stream is not None:
            self._overlap_input_ready.record(current_stream())
            with torch.cuda.stream(self._overlap_stream):
                self._overlap_input_ready.wait(self._overlap_stream)
                shared_out = shared_expert(x)
                self._overlap_output_ready.record(self._overlap_stream)
            routed = self._forward_routed_experts_only(x, topk_weights, topk_ids)
            self._overlap_output_ready.wait(current_stream())
        else:
            # Overlap disabled (VLLM_DISABLE_SHARED_EXPERTS_STREAM or no
            # CUDA): serial separate shared expert, like the stock
            # no-stream path.
            routed = self._forward_routed_experts_only(x, topk_weights, topk_ids)
            shared_out = shared_expert(x)
        return routed + shared_out

    def _forward_separate_shared(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Serial fallback when AITER cannot serve the fused heterogeneous
        shape: width-7 routes sliced to routed width, plus the separate
        shared expert, all on the current stream."""
        shared_expert = self._shared_expert_ref()
        if shared_expert is None:
            raise RuntimeError("The separate shared-expert fallback is unavailable.")
        shared_out = shared_expert(x)
        routed = self._forward_routed_experts_only(
            x,
            topk_weights[:, :-1].contiguous(),
            topk_ids[:, :-1].contiguous(),
        )
        return routed + shared_out


def _fuse_shared_experts_enabled(config, vllm_config: "VllmConfig") -> bool:
    # Kill switch for the fused-shared-expert feature (including the
    # FP8->MXFP4 load-time requant path): setting the env var restores the
    # original separate shared-expert execution everywhere (V4 and V4.1
    # target layers, and the DSpark draft layers).
    if os.environ.get("VLLM_DISABLE_FUSED_SHARED_EXPERTS", "0") == "1":
        logger.warning_once(
            "Fused shared experts (FP8->MXFP4) is DISABLED (env override)"
        )
        return False
    if vllm_config.kernel_config.monokernel:
        # Under --monokernel the FP8->MXFP4 requant fusion is refused
        # unconditionally -- it consumes the shared-expert FP8 weights into
        # the routed MXFP4 table, which the mono decode kernels read in
        # their checkpoint layout. The only fused-shared-expert consumer
        # under mono is the heterogeneous native-FP8 fusion
        # (prefill-shaped batches above mono decode), which keeps the
        # shared weights in FP8.
        return False
    parallel_config = vllm_config.parallel_config
    if (
        getattr(config, "n_shared_experts", None)
        and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        and not parallel_config.enable_expert_parallel
        and parallel_config.data_parallel_size > 1
    ):
        # Fused shared experts are not supported under data parallelism:
        # the fused path cannot load the FP8 shared expert into the MXFP4
        # routed slot, which fails obscurely during weight loading. Fail
        # fast with an actionable message instead.
        raise ValueError(
            "DeepSeek-V4 fused shared experts "
            "(VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1) are not supported "
            "with data parallelism (data_parallel_size > 1). Set "
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 to run the shared "
            "expert as an unfused FP8 MLP."
        )
    return bool(
        getattr(config, "n_shared_experts", None)
        and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        and not parallel_config.enable_expert_parallel
    )


def _validate_aiter_mega_moe_config(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError(
            "--moe-backend aiter_mega_moe requires VLLM_ROCM_USE_AITER=1 "
            " and VLLM_ROCM_USE_AITER_MOE=1."
        )
    if not on_gfx950():
        raise NotImplementedError("AITER MegaMoE requires gfx950.")
    if (
        not parallel_config.enable_expert_parallel
        or parallel_config.tensor_parallel_size != 1
        or parallel_config.pipeline_parallel_size != 1
    ):
        raise NotImplementedError(
            "AITER MegaMoE requires data parallel attention with expert "
            "parallel MoE: --data-parallel-size N --enable-expert-parallel "
            "with TP=1 and PP=1."
        )
    if parallel_config.all2all_backend not in (
        "mori_high_throughput",
        "mori_low_latency",
    ):
        raise ValueError(
            "AITER MegaMoE runs on MoRI shmem; add "
            "--all2all-backend mori_high_throughput."
        )
    if parallel_config.enable_eplb:
        raise NotImplementedError("AITER MegaMoE does not support EPLB.")
    if getattr(vllm_config.model_config.hf_config, "expert_dtype", "fp4") != "fp4":
        raise NotImplementedError("AITER MegaMoE only supports fp4 experts.")


class DeepseekV4MoE(nn.Module):
    # None: use the padding vLLM added.
    experts_intermediate_pad: int | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        fuse_heterogeneous_shared_expert: bool = False,
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        self.image_sentinel_lo = (
            IMAGE_SENTINEL_BASE_ID if getattr(config, "vision_n_layers", 0) > 0 else 0
        )
        num_hash_layers = getattr(config, "num_hash_layers", None)  # Set by V4.1
        if num_hash_layers is None:
            num_hash_layers = config.mlp_layer_types.count("hash_moe")
        is_hash_moe = extract_layer_index(prefix) < num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        if getattr(config, "topk_method", None) == "noaux_tc" and (
            not is_hash_moe or getattr(config, "vision_n_layers", 0) > 0
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        self.n_shared_experts = config.n_shared_experts

        # TODO: FSE for DeepSeek-v4 is enabled by checking only
        # `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, despite AITER not
        # being used; this should use `resolve_layer_fused_shared_expert`.
        self.fuse_heterogeneous_shared_expert = fuse_heterogeneous_shared_expert
        fse_requested = (
            _fuse_shared_experts_enabled(config, vllm_config)
            and not self.fuse_heterogeneous_shared_expert
        )
        fse_compatible = False
        if fse_requested:
            fse_compatible, fse_reason = is_shared_expert_quant_fse_compatible(
                quant_config,
                f"{prefix}.experts",
                f"{prefix}.shared_experts",
            )
            if not fse_compatible:
                logger.warning(
                    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
                    "cannot be enabled: %s.",
                    fse_reason,
                )
        self.is_fused_shared_expert_enabled = fse_requested and fse_compatible

        self.use_mega_moe = vllm_config.kernel_config.moe_backend == "aiter_mega_moe"
        fuse_mega_moe_shared_expert = (
            self.use_mega_moe
            and config.n_shared_experts == 1
            and resolve_layer_fused_shared_expert(
                quant_config, prefix, moe_backend="aiter_mega_moe"
            )
        )
        shared_quant_config = quant_config
        if fuse_mega_moe_shared_expert:
            # MegaMoE requantizes the checkpoint weights itself.
            assert quant_config is not None
            shared_quant_config = copy.copy(quant_config)
            shared_quant_config.online_quantization_config = None

        if config.n_shared_experts is None or self.is_fused_shared_expert_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=shared_quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        if self.use_mega_moe:
            _validate_aiter_mega_moe_config(vllm_config)
            self.experts = DeepseekV4AiterMegaMoEExperts(
                vllm_config,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                swiglu_limit=self.swiglu_limit,
                fuse_shared_expert=fuse_mega_moe_shared_expert,
                prefix=f"{prefix}.experts",
            )
            self.shared_experts_stream = (
                aux_stream()
                if self.shared_experts is not None
                and not self.experts.fuse_shared_expert
                and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
                else None
            )
            self.shared_experts_events = (
                (torch.cuda.Event(), torch.cuda.Event())
                if self.shared_experts_stream is not None
                else None
            )
            return

        fuse_shared_into_routed = (
            self.is_fused_shared_expert_enabled or self.fuse_heterogeneous_shared_expert
        )
        self.experts = FusedMoEFactory(
            shared_experts=(None if fuse_shared_into_routed else self.shared_experts),
            n_shared_experts=(
                config.n_shared_experts if fuse_shared_into_routed else None
            ),
            fuse_shared_experts=fuse_shared_into_routed,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            intermediate_pad=self.experts_intermediate_pad,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            routed_experts_cls=(
                DeepseekV4HeterogeneousSharedRoutedExperts
                if self.fuse_heterogeneous_shared_expert
                else None
            ),
            routed_experts_args=(
                {
                    "shared_expert": self.shared_experts,
                    "shared_expert_id": config.n_routed_experts,
                }
                if self.fuse_heterogeneous_shared_expert
                else None
            ),
        )
        if self.fuse_heterogeneous_shared_expert:
            # Prefill-only gating for the heterogeneous fused path: batches
            # below the M threshold skip the shared-slot append in the
            # router (routed-width top-k) and keep the stock overlapped
            # separate shared-expert layout in forward_modular; decode
            # forwards never append regardless of batch size, so the fused
            # hetero FHMoE engages only for prefill-shaped batches (the
            # regime where it wins) and large decode batches keep stock
            # numerics.
            router = typing.cast(FusedTopKBiasRouter, self.experts.router)
            router.shared_expert_append_min_tokens = (
                _heterogeneous_shared_expert_min_tokens()
            )
            router.shared_expert_append_prefill_only = True

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
        if self.gate.bias_vl is not None and input_ids is None:
            raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        if self.use_mega_moe:
            return self._forward_mega_moe(hidden_states, input_ids).view(org_shape)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )

        return final_hidden_states.view(org_shape)

    def _forward_mega_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        shared_experts = self.shared_experts
        if shared_experts is None:
            return self._forward_mega_moe_routed(hidden_states, input_ids)
        events = self.shared_experts_events
        if (
            events is None
            or hidden_states.shape[0] > envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        ):
            return self._forward_mega_moe_routed(
                hidden_states, input_ids
            ) + shared_experts(hidden_states)
        # The unfused shared expert overlaps router + MegaMoE in small steps.
        routed, shared = maybe_execute_in_parallel(
            fn0=lambda: self._forward_mega_moe_routed(hidden_states, input_ids),
            fn1=lambda: shared_experts(hidden_states),
            event0=events[0],
            event1=events[1],
            aux_stream=self.shared_experts_stream,
        )
        return routed + shared

    def _forward_mega_moe_routed(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None
    ) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
        )
        return self.experts(hidden_states, topk_weights, topk_ids)


# Hidden sizes supported by AITER mhc_pre_big_fuse_rmsnorm.
_AITER_MHC_FUSED_RMSNORM_SIZES = frozenset({1280, 2560, 4096, 7168})


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        fuse_heterogeneous_shared_expert: bool = False,
    ):
        super().__init__()

        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.vllm.mhc_pre and mhc_post
        import vllm.model_executor.layers.mhc  # noqa: F401

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4ROCMAiterMLAAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            fuse_heterogeneous_shared_expert=fuse_heterogeneous_shared_expert,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()
        # AITER mhc kernels (pre/post/fused) require hc_mult == 4.
        use_aiter_mhc = (
            HAS_AITER_MHC and self.hidden_size % 256 == 0 and self.hc_mult == 4
        )
        # Prefer AITER fused post+pre when eligible; otherwise TileLang.
        self.use_fused_mhc = use_aiter_mhc or HAS_TILELANG_MHC
        # Fold attn/ffn RMSNorm into MHC only when the active backend's
        # fused-rmsnorm path supports this hidden size.
        if use_aiter_mhc:
            self.fuse_mhc_rmsnorm = self.hidden_size in _AITER_MHC_FUSED_RMSNORM_SIZES
        else:
            self.fuse_mhc_rmsnorm = HAS_TILELANG_MHC and self.use_fused_mhc

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        """Reduce HC residual streams to the next sub-layer input.

        When ``norm_weight`` is set, RMSNorm is fused into the pre kernel.
        """
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    def _forward_fused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_norm_weight = self.attn_norm.weight if self.fuse_mhc_rmsnorm else None
        attn_norm_eps = (
            self.attn_norm.variance_epsilon if self.fuse_mhc_rmsnorm else 0.0
        )
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        if not self.fuse_mhc_rmsnorm:
            x = self.attn_norm(x)
        x = self.attn(positions, x, None)

        ffn_norm_weight = self.ffn_norm.weight if self.fuse_mhc_rmsnorm else None
        ffn_norm_eps = self.ffn_norm.variance_epsilon if self.fuse_mhc_rmsnorm else 0.0
        residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        if not self.fuse_mhc_rmsnorm:
            x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix

    def _forward_unfused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if not self.use_fused_mhc:
            return self._forward_unfused_post_pre(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        return self._forward_fused_post_pre(
            x, positions, input_ids, post_mix, res_mix, residual
        )


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.fuse_heterogeneous_shared_expert = _heterogeneous_shared_expert_enabled(
            vllm_config
        )
        self.use_mega_moe = vllm_config.kernel_config.moe_backend == "aiter_mega_moe"
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                fuse_heterogeneous_shared_expert=(
                    self.fuse_heterogeneous_shared_expert
                ),
            ),
            prefix=f"{prefix}.layers",
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            DeepseekV4MoE,
            "ffn",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_op = HCHeadOp()
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        # EAGLE3 / DSpark / DFlash aux hidden states: reconstructed (post-mhc)
        # hidden state at the configured target layers, averaged over the
        # hc_mult streams to [T, hidden_size]. Empty unless a draft model set
        # aux_hidden_state_layers.
        aux_hidden_states: list[torch.Tensor] = []
        # On the fused path the final layer's hc_post output is reused below
        # (avoids computing hc_post twice when the last layer is also an aux
        # layer).
        final_aux_recon: torch.Tensor | None = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if (idx + 1) in self.aux_hidden_state_layers:
                # On the unfused path the layer already applied hc_post,
                # so hidden_states is the reconstructed stream; on the fused
                # path reconstruct it via hc_post before averaging.
                if layer.use_fused_mhc:
                    aux_recon = layer.hc_post(
                        hidden_states, residual, post_mix, res_mix
                    )
                    final_aux_recon = aux_recon
                else:
                    aux_recon = hidden_states
                aux_hidden_states.append(aux_recon.mean(dim=1))
        if layer is not None and layer.use_fused_mhc:
            # Reuse the last layer's hc_post output if it was already computed
            # for the aux hidden state above; otherwise compute it now.
            if (
                final_aux_recon is not None
                and self.end_layer in self.aux_hidden_state_layers
            ):
                hidden_states = final_aux_recon
            else:
                hidden_states = layer.hc_post(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            # Each DBO ubatch owns a different slice of the batch. Writing
            # from row 0 in both threads would let one ubatch clobber the
            # other's rows in the buffer the MTP drafter reads after the step.
            offset = dbo_current_ubatch_token_offset()
            self._mtp_hidden_buffer[offset : offset + num_tokens].copy_(
                hidden_states.flatten(1)
            )

        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _resolve_param_name(name: str) -> str:
            # Fp8LinearMethod registers block scales as ``weight_scale_inv``.
            # Quark checkpoints / QuarkW8A8Fp8PerBlock use ``weight_scale``.
            # Alias only when the registered param is the ``_inv`` name.
            inv_name = f"{name}_inv"
            if name not in params_dict and inv_name in params_dict:
                return inv_name
            return name

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Use each MoE's own per-layer fusion decision (computed with its prefix
        # at init) as the single source of truth, so the redirect below cannot
        # diverge from how the module was built if per-layer quantization ever
        # mixes fused and non-fused layers.
        fuse_by_layer = {
            extract_layer_index(mod_name): mod.is_fused_shared_expert_enabled
            for mod_name, mod in self.named_modules()
            if isinstance(mod, DeepseekV4MoE)
        }
        n_routed = self.config.n_routed_experts
        # The redirect below maps the single shared-expert tensor group to one
        # appended slot; multiple shared experts would need per-expert slicing
        # (see deepseek_v2.py). DeepSeek-V4 has n_shared_experts == 1.
        if any(fuse_by_layer.values()) and self.config.n_shared_experts != 1:
            raise NotImplementedError(
                "deepseek-v4 fused shared-expert loading supports only "
                f"n_shared_experts == 1, got {self.config.n_shared_experts}"
            )

        for name, loaded_weight in weights:
            # Shared-expert fusion: redirect ``.ffn.shared_experts.w{1,2,3}``
            # into appended routed-expert slot ``.ffn.experts.{n_routed}``
            # so the MXFP4-quantized shared expert loads through the routed
            # expert loader (grouped GEMM). Single shared expert only.
            if ".ffn.shared_experts.w" in name and fuse_by_layer.get(
                extract_layer_index(name), False
            ):
                name = name.replace(
                    ".ffn.shared_experts.w",
                    f".ffn.experts.{n_routed}.w",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param_name = _resolve_param_name(name)
                param = params_dict[param_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(param_name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param_name = _resolve_param_name(name)
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(param_name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # When fusing shared experts, include the appended slots
        # (ids n_routed_experts .. n_routed_experts + n_shared - 1) so the
        # redirected shared-expert weights route through the expert loader.
        n_shared = getattr(self.config, "n_shared_experts", 0) or 0
        num_experts = self.config.n_routed_experts + (
            n_shared if self.is_fused_shared_expert_enabled else 0
        )
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
            routed_experts_prefix="" if self.use_mega_moe else "routed_experts",
        )


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, fuse_shared_experts: bool = False
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Native DeepSeek FP8
        # linear / shared experts use Fp8LinearMethod's ``weight_scale_inv``.
        # Quark linear block-FP8 registers ``weight_scale`` and is left
        # as-is; load_weights aliases ``.weight_scale`` -> ``.weight_scale_inv``
        # only when the latter is the registered parameter.
        #
        #  - DeepSeek native ``.scale``: expert scales -> ``.weight_scale``,
        #    everything else -> ``.weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    # When shared experts are fused into the routed MXFP4 grouped GEMM, the
    # shared_experts tensors are redirected to routed expert slots ; leave
    # their names untouched here.
    orig_to_new_substr: dict[str, str | None] = (
        {}
        if fuse_shared_experts
        else {".shared_experts.w2": ".shared_experts.down_proj"}
    )
    orig_to_new_substr["mtp."] = None
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".input_scale": ".input_scale_2",
        },
        orig_to_new_substr=orig_to_new_substr,
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    model_cls = DeepseekV4Model
    finalizes_weights_during_load = False

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if expert_dtype != "fp4" or self.model.is_fused_shared_expert_enabled:
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
                expert_dtype,
                fuse_shared_experts=self.model.is_fused_shared_expert_enabled,
            )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        # Must precede the gate_up preshuffle of the fused shared experts.
        finalize_mega_moe_weights(self)
        # After per-layer quant finalize, so we preshuffle the final fp8 weights.
        fused_compressor_layers = 0
        for module in self.modules():
            if isinstance(module, DeepseekV4ROCMAiterMLAAttention):
                fused_compressor_layers += module.prepare_compressor_gemm_fusion()
                module.prepare_attn_preshuffle()
            elif isinstance(module, DeepseekV4MLP):
                module.prepare_gateup_preshuffle()
        if fused_compressor_layers:
            logger.info(
                "Fused the C4 compressor GEMMs in %d DeepSeek V4 layers",
                fused_compressor_layers,
            )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
