# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

if TYPE_CHECKING:
    from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import DenseMlp
    from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
        GroupedExperts,
    )

logger = init_logger(__name__)

MAX_TOKENS = 64
# (tiles, weight MMA steps in registers), measured with cold L2 on SM103.
FP8_PROJECTION_TACTICS = {
    (4096, 2048): {1: (1, 8), 2: (2, 8), 4: (2, 8), 8: (2, 8)},
    (6144, 4096): {
        1: (1, 8),
        2: (1, 4),
        4: (1, 4),
        5: (3, 4),
        6: (3, 4),
        7: (3, 4),
        8: (3, 4),
    },
}
GEOMETRY = {
    "hidden_size": 6144,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "num_attention_heads": 64,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "hidden_act": "silu",
    "topk_method": "noaux_tc",
}


@dataclass(frozen=True)
class DecodeConfig:
    tokens: tuple[int, ...]


def specialized_decode_config(vllm_config: VllmConfig, text) -> DecodeConfig | None:
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "stock":
        return None
    # other DSA checkpoints keep their existing implementation by default.
    if backend == "auto" and getattr(text, "model_type", None) not in (
        "glm_moe_dsa",
        "deepseek_mtp",
    ):
        return None
    if backend not in ("auto", "latency_specialized"):
        raise ValueError(
            "Supported decode_backend values are auto/stock/latency_specialized"
        )

    from vllm.config import CompilationMode
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_cutedsl

    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    quant = vllm_config.quant_config
    compilation = vllm_config.compilation_config
    offload = vllm_config.offload_config
    spec = vllm_config.speculative_config
    checks = (
        (
            not current_platform.is_device_capability_family(100),
            "GPUs outside the SM100 family",
        ),
        (not has_cutedsl(), "missing CuTe DSL"),
        (model.dtype != torch.bfloat16, "activation dtypes other than BF16"),
        (
            any(getattr(text, k, None) != v for k, v in GEOMETRY.items()),
            "model geometry other than GLM-5.3",
        ),
        (bool(getattr(text, "mhc", False)), "mHC checkpoints"),
        (
            model.quantization != "fp8"
            or not isinstance(quant, Fp8Config)
            or not quant.is_checkpoint_fp8_serialized
            or quant.activation_scheme != "dynamic"
            or quant.weight_block_size != [128, 128],
            "weights other than native block-FP8 (128x128)",
        ),
        (
            parallel.tensor_parallel_size not in (1, 2, 4, 8),
            "TP other than 1, 2, 4 or 8",
        ),
        # MoE shards across TP x DP and needs the stock dispatch/combine path.
        (parallel.data_parallel_size != 1, "data parallelism"),
        # EPLB and sequence-parallel MoE require EP in ParallelConfig.
        (parallel.enable_expert_parallel, "expert parallelism / EPLB"),
        (parallel.use_ubatching, "DBO"),
        (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1,
            "context parallelism",
        ),
        (model.enable_sleep_mode, "sleep mode"),
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.aux_output_config.enabled, "routed expert capture"),
        (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0,
            "weight offloading",
        ),
        (envs.VLLM_BATCH_INVARIANT, "batch invariant execution"),
        (compilation.mode == CompilationMode.STOCK_TORCH_COMPILE, "torch.compile"),
        (
            compilation.cudagraph_mode is None
            or compilation.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL,
            "execution without FULL decode graphs",
        ),
        (
            spec is not None
            and (spec.method != "mtp" or spec.num_speculative_tokens > 7),
            "speculation other than MTP with up to 7 draft tokens",
        ),
    )
    reasons = [reason for failed, reason in checks if failed]
    if reasons:
        if backend == "auto":
            return None
        raise ValueError(
            "GLM-5.3 specialized decode does not support " + "; ".join(reasons)
        )
    q = 1 + spec.num_speculative_tokens if spec else 1
    sizes = (
        compilation.decode_cudagraph_capture_request_sizes
        or compilation.cudagraph_capture_sizes
        or (1, 2, 4, 8, 16, 32, 64)
    )
    tokens = tuple(
        sorted(
            {
                n * width
                for n in sizes
                for width in (1, q)
                if 0 < n * width <= MAX_TOKENS
            }
        )
    )
    return DecodeConfig(tokens=tokens)


def is_decode_step(tokens: int, config: DecodeConfig) -> bool:
    if tokens not in config.tokens or not is_forward_context_available():
        return False
    context = get_forward_context()
    # V2 captures FULL graphs directly with runtime mode NONE.
    if context.cudagraph_runtime_mode != CUDAGraphMode.FULL and not (
        context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        and torch.cuda.is_current_stream_capturing()
    ):
        return False
    metadata = context.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    logger.info_once("GLM-5.3 specialized decode selected for %d-token graph", tokens)
    return True


class Fp8Projection:
    """The Flash dense MLP's GEMV without its SwiGLU epilogue."""

    def __init__(self, module, tokens):
        from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import (
            MAX_TOKENS as GEMV_MAX_TOKENS,
        )
        from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import _MmaGemv
        from vllm.models.glm5next.nvidia.specialized_decode.shared_expert import (
            block_scales,
        )
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        self.weight = module.weight
        device_props = torch.cuda.get_device_properties(self.weight.device)
        rows, k = self.weight.shape
        scales = module.weight_scale_inv
        self.scales = (
            scales if scales.dtype == torch.float32 else block_scales(scales, rows, k)
        )
        self.tokens = {m for m in tokens if 0 < m <= GEMV_MAX_TOKENS}
        self.kernels = {}
        import cutlass.cute as cute

        from vllm.models.glm5next.nvidia.specialized_decode.cutedsl.host import (
            stream,
            to_cute,
        )

        # compile for the selected tokens
        for m in sorted(self.tokens):
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=self.weight.device)
            out = x.new_empty(m, rows)
            tiles, registers = FP8_PROJECTION_TACTICS.get((rows, k), {}).get(
                m, (min(2, 16384 // k), 4)
            )
            kernel = _MmaGemv(
                rows, k, m, tiles, registers, 1, False, 0.0, is_deep_gemm_e8m0_used()
            )
            if kernel.smem > device_props.shared_memory_per_block_optin:
                self.tokens.remove(m)
                continue
            self.kernels[m] = cute.compile(
                kernel,
                *(to_cute(t, 16) for t in (self.weight, self.scales, x, out)),
                stream(x.device),
            )

    def __call__(self, x):
        from vllm.models.glm5next.nvidia.specialized_decode.cutedsl.host import (
            stream,
            to_cute,
        )

        out = x.new_empty(x.shape[0], self.weight.shape[0])
        self.kernels[x.shape[0]](
            *(to_cute(t, 16) for t in (self.weight, self.scales, x, out)),
            stream(x.device),
        )
        return out


class DecodeLinearMethod(QuantizeMethodBase):
    def __init__(self, inner, projection, tokens):
        self.inner, self.projection, self.tokens = inner, projection, tokens
        self.requires_device_loading = inner.requires_device_loading
        self.supports_pre_processed_weights = inner.supports_pre_processed_weights
        self.uses_meta_device = inner.uses_meta_device

    def create_weights(self, layer, *args, **kwargs):
        return self.inner.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        self.inner.process_weights_after_loading(layer)
        self.projection = None  # rebuilt by the decoder layer's post-load hook

    def dequantize_weight(self, layer):
        return self.inner.dequantize_weight(layer)

    def apply(self, layer, x, bias=None):
        from vllm.models.glm5next.nvidia.specialized_decode.linear import planned

        if (
            self.projection is not None
            and planned()
            and x.dtype == torch.bfloat16
            and x.numel() // x.shape[-1] in self.tokens
        ):
            out = self.projection(x.reshape(-1, x.shape[-1]).contiguous())
            if bias is not None:
                out = (out + bias).to(x.dtype)
            return out.view(*x.shape[:-1], out.shape[-1])
        return self.inner.apply(layer, x, bias)


class LayerDecode:
    mlp: "DenseMlp | GroupedExperts | None"

    def __init__(self, layer, config: DecodeConfig):
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        self.config = config
        self.mlp = None
        self.mlp_tokens: set[int] = set()
        for name in ("fused_qkv_a_proj", "q_b_proj", "o_proj"):
            module = getattr(layer.self_attn, name)
            if module.weight.dtype != torch.float8_e4m3fn:
                continue
            inner = module.quant_method
            if isinstance(inner, DecodeLinearMethod):
                inner = inner.inner
            projection = Fp8Projection(module, config.tokens)
            module.quant_method = DecodeLinearMethod(
                inner, projection, projection.tokens
            )

        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
                GroupedExperts,
                shared_weights,
            )

            routed = mlp.experts.routed_experts
            se = mlp.shared_experts
            # the grouped kernel consumes TRTLLM-Gen's shuffled BlockMajorK
            # weights. A different MoE backend keeps its own implementation.
            if routed.w13_weight.dim() != 4 or se is None:
                logger.info_once(
                    "GLM-5.3 specialized MoE needs TRTLLM-Gen BlockMajorK weights"
                )
                return
            gate_up, down = se.gate_up_proj, se.down_proj
            from vllm.models.glm5next.nvidia.specialized_decode.shared_expert import (
                block_scales,
            )

            def scales(module):
                value = module.weight_scale_inv
                return (
                    value
                    if value.dtype == torch.float32
                    else block_scales(value, *module.weight.shape)
                )

            shared = shared_weights(
                gate_up.weight, scales(gate_up), down.weight, scales(down)
            )
            self.mlp = GroupedExperts(
                routed.w13_weight,
                routed.w13_weight_scale_inv,
                routed.w2_weight,
                routed.w2_weight_scale_inv,
                mlp.experts.moe_config.experts_per_token,
                float("inf"),
                mlp.routed_scaling_factor,
                is_deep_gemm_e8m0_used(),
                shared=shared,
            )
            self.mlp.prepare(
                list(config.tokens), mlp.gate.out_dtype or torch.bfloat16, config.tokens
            )
            self.mlp_tokens = set(config.tokens)
        else:
            from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import (
                MAX_TOKENS as GEMV_MAX_TOKENS,
            )
            from vllm.models.glm5next.nvidia.specialized_decode.dense_mlp import (
                SLICE,
                DenseMlp,
            )

            gate_up, down = mlp.gate_up_proj, mlp.down_proj
            self.mlp = DenseMlp(
                gate_up.weight,
                gate_up.weight_scale_inv,
                down.weight,
                down.weight_scale_inv,
                float("inf"),
                is_deep_gemm_e8m0_used(),
            )
            # TP8's down GEMV has only three warps per tile: at most six tokens.
            self.mlp_tokens = {
                m
                for m in config.tokens
                if m <= min(GEMV_MAX_TOKENS, 2 * (self.mlp.inter // SLICE))
            }
            self.mlp.prepare(sorted(self.mlp_tokens))
        logger.info_once(
            "GLM-5.3 specialized decode prepared %s for %s-token graphs",
            type(self.mlp).__name__,
            tuple(sorted(self.mlp_tokens)),
        )

    def run_mlp(self, layer, x):
        from vllm.models.glm5next.nvidia.specialized_decode.grouped_moe import (
            GroupedExperts,
        )

        if self.mlp is None or x.shape[0] not in self.mlp_tokens:
            return layer.mlp(x)
        if isinstance(self.mlp, GroupedExperts):
            logits, _ = layer.mlp.gate(x)
            return self.mlp(
                x, logits, layer.mlp.gate.e_score_correction_bias, finalize=True
            )
        return self.mlp(x)
