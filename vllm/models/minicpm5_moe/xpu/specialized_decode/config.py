# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment checks that run before allocating any mono state."""

from dataclasses import dataclass
from os import getenv

import torch

SHAPES = {
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 8192,
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 512,
    "n_routed_experts": 160,
    "num_experts_per_tok": 16,
    "n_shared_experts": 1,
    "use_gated_attention": True,
    "routed_scaling_factor": 3.66,
    "rope_theta": 5000000,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
}
ROWS = tuple(range(1, 9))
MAX_VERIFY_ROWS = 64
MAX_CONTEXT_LEN = 512 * 1024


def native_rows(ops) -> frozenset[int]:
    """Old ABI3 libraries remain usable for their original four shapes."""
    return frozenset(
        ops.supported_rows() if hasattr(ops, "supported_rows") else (1, 2, 4, 8)
    )


def native_context_limit(ops) -> int:
    """Libraries without the capability retain their original 32K limit."""
    return ops.max_context_len() if hasattr(ops, "max_context_len") else 32768


def native_rotary_cache_version(ops) -> int:
    return ops.rotary_cache_version() if hasattr(ops, "rotary_cache_version") else 0


@dataclass(frozen=True)
class DecodeConfig:
    tp: int
    graph: bool = False
    max_context_len: int = 32768
    speculative_tokens: int = 0
    tail_only: bool = False
    tail_rows4: bool = False
    pre_o_split: bool = False
    post_o_split: bool = False
    pre_o_fa2_adaptive_split1: bool = False


def requested(config) -> bool:
    return bool(getattr(config.kernel_config, "monokernel", False)) or (
        isinstance(config.additional_config, dict)
        and config.additional_config.get("decode_backend") == "latency_specialized"
    )


def serves(metadata, rows: int, max_context_len: int = 32768) -> bool:
    """Serve ordinary requests or explicitly qualified causal verify rows."""
    query_len = getattr(metadata, "max_query_len", 0)
    verify = getattr(metadata, "minicpm5_verify_eligible", False)
    query_supported = query_len == 1 or (verify and 1 < query_len <= min(rows, 8))
    return (
        (
            rows in ROWS
            or (verify and 1 < query_len <= 8 and 8 < rows <= MAX_VERIFY_ROWS)
        )
        and metadata is not None
        and getattr(metadata, "minicpm5_decode_eligible", False)
        and query_supported
        and getattr(metadata, "causal", True) is True
        and 0 < getattr(metadata, "num_actual_tokens", 0) <= rows
        and getattr(metadata, "num_prefill_tokens", 0) == 0
        and getattr(metadata, "mm_prefix_query_range_tensor", None) is None
        and getattr(metadata, "rswa_prefix_lens", None) is None
        and 0 < getattr(metadata, "max_seq_len", max_context_len + 1) <= max_context_len
    )


def unsupported(config) -> list[str]:
    mc, pc, cc, cg = (
        config.model_config,
        config.parallel_config,
        config.cache_config,
        config.compilation_config,
    )
    dtype = str(mc.dtype).removeprefix("torch.")
    checks = [
        (pc.tensor_parallel_size not in (1, 2), "TP1 or TP2 required"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        (pc.use_ubatching, "dual-batch overlap"),
        (bool(pc.enable_batch_sharded_sampling), "batch-sharded sampling"),
        (config.lora_config is not None, "LoRA"),
        (config.kv_transfer_config is not None, "KV transfer"),
        (config.weight_transfer_config is not None, "weight updates"),
        (mc.enable_sleep_mode, "sleep mode"),
        (config.offload_config.uva.cpu_offload_gb > 0, "CPU weight offload"),
        (config.offload_config.prefetch.offload_group_size > 0, "prefetch offload"),
        (
            config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        (dtype not in ("bfloat16", "float16"), "BF16 or FP16 activations required"),
        (mc.quantization != "fp8_per_tensor", "online FP8 per-tensor required"),
        (
            cc.cache_dtype not in ("auto", dtype),
            "KV cache must match activation dtype",
        ),
        (
            getattr(cc, "user_specified_block_size", True)
            and cc.block_size is not None
            and cc.block_size % 64 != 0,
            "cache block size must be a multiple of 64",
        ),
        (
            config.speculative_config is not None
            and (
                config.speculative_config.method != "dspark"
                or not 1 <= config.speculative_config.num_speculative_tokens <= 7
            ),
            "speculative decoding requires DSpark with 1..7 draft tokens",
        ),
        (getattr(cg.mode, "name", str(cg.mode)) not in ("NONE", "0"), "torch.compile"),
        (
            mc.max_model_len > MAX_CONTEXT_LEN,
            f"mono context limit is {MAX_CONTEXT_LEN}",
        ),
    ]
    reasons = [why for bad, why in checks if bad]
    priorities = getattr(config.kernel_config, "ir_op_priority", None)
    if priorities is not None:
        for name in ("rms_norm", "fused_add_rms_norm"):
            priority = getattr(priorities, name, [])
            if priority and priority[0] != "vllm_c":
                reasons.append(f"{name} must use the vllm_c rounding contract")
    if not mc.enforce_eager:
        mode = cg.cudagraph_mode
        decode_mode = None if mode is None else mode.decode_mode()
        if getattr(decode_mode, "name", str(decode_mode)) != "FULL":
            reasons.append("FULL decode graph required when eager is disabled")
    if "MiniCPM5MoEForCausalLM" not in mc.architectures:
        reasons.append("MiniCPM5MoEForCausalLM architecture required")
    text = mc.hf_text_config
    reasons.extend(
        f"{key} must be {expected}"
        for key, expected in SHAPES.items()
        if getattr(text, key, None) != expected
    )
    for key, default in (("n_group", 1), ("topk_group", 1), ("norm_topk_prob", True)):
        if getattr(text, key, default) != default:
            reasons.append(f"{key} must be {default}")
    rope = getattr(text, "rope_parameters", None)
    default_rope = not rope or (
        isinstance(rope, dict)
        and rope.get("rope_type", rope.get("type", "default")) == "default"
        and rope.get("rope_theta", SHAPES["rope_theta"]) == SHAPES["rope_theta"]
        and rope.get("partial_rotary_factor", 1.0) == 1.0
    )
    if getattr(text, "attention_bias", False) or not default_rope:
        reasons.append("attention bias or scaled RoPE")
    return reasons


def configure(config) -> DecodeConfig | None:
    if not requested(config):
        return None
    reasons = unsupported(config)
    from vllm import envs
    from vllm.distributed import get_tp_group
    from vllm.platforms import current_platform

    from .peer import agree

    if envs.VLLM_BATCH_INVARIANT:
        reasons.append("batch-invariant normalization")
    if not current_platform.is_xpu():
        reasons.append("Intel XPU required")
    device_index = None
    if not reasons:
        try:
            device_index = torch.xpu.current_device()
            # XPUPlatform already loads this common architecture query. Check
            # the worker's device before importing the optional mono provider.
            if not torch.ops._xpu_C.is_bmg(device_index):
                reasons.append("MiniCPM5 mono requires a Battlemage (Xe2) XPU")
        except (AttributeError, RuntimeError) as error:
            reasons.append(f"cannot verify MiniCPM5 mono XPU architecture: {error}")
    ops = None
    if not reasons:
        try:
            from vllm_xpu_kernels.minicpm5_mono import load

            ops = load()
            limit = native_context_limit(ops)
            if config.model_config.max_model_len > limit:
                reasons.append(
                    f"Rebuild native MiniCPM5 mono for context "
                    f"{config.model_config.max_model_len}; installed limit is {limit}"
                )
            if (
                config.model_config.max_model_len > 32768
                and native_rotary_cache_version(ops) != 1
            ):
                reasons.append(
                    "Rebuild native MiniCPM5 mono with canonical RoPE cache support"
                )
            dtype = str(config.model_config.dtype).removeprefix("torch.")
            if dtype == "float16" and (
                not hasattr(ops, "activation_dtypes")
                or dtype not in ops.activation_dtypes()
            ):
                reasons.append("Rebuild native MiniCPM5 mono with FP16 support")
            if getattr(ops, "router_logits_dtype", lambda: None)() != "float32":
                reasons.append("Rebuild native MiniCPM5 mono with FP32 router logits")
            capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
            missing_rows = sorted(
                set(capture_sizes).intersection(ROWS) - native_rows(ops)
            )
            if not config.model_config.enforce_eager and missing_rows:
                reasons.append(
                    f"Rebuild native MiniCPM5 mono for capture rows {missing_rows}"
                )
            supports_device = getattr(ops, "supports_device", None)
            supported = (
                supports_device(device_index)
                if supports_device is not None
                else ops.supported()
            )
            if not supported:
                reasons.append("in-order Xe2 GPU with >=32 Xe cores required")
        except (ImportError, RuntimeError) as error:
            reasons.append(f"native library: {error}")
    agree(reasons, get_tp_group().cpu_group, "deployment")
    mc = config.model_config
    tp = config.parallel_config.tensor_parallel_size
    monokernel = bool(getattr(config.kernel_config, "monokernel", False))
    ordinary_fp16_graph = (
        str(mc.dtype) == "torch.float16"
        and not mc.enforce_eager
        and config.speculative_config is None
        and config.cache_config.enable_prefix_caching is False
    )
    tail_supported = ordinary_fp16_graph and tp == 1
    o_split_supported = ordinary_fp16_graph and tp == 2
    pre_o_split = o_split_supported and getenv("MINICPM5_MONO_PRE_O_SPLIT", "1") == "1"
    # The existing PRE_O opt-out still disables automatically selected splits.
    post_o_default = "1" if monokernel and pre_o_split else "0"
    post_o_split = (
        o_split_supported
        and getenv("MINICPM5_MONO_POST_O_SPLIT", post_o_default) == "1"
    )
    adaptive_flag = getenv("MINICPM5_MONO_PRE_O_FA2_ADAPTIVE_SPLIT1")
    if adaptive_flag is None:
        post_o_abi = getattr(ops, "post_o_abi_version", None)
        adaptive = bool(
            monokernel
            and post_o_split
            and post_o_abi is not None
            and post_o_abi() == 1
            and hasattr(ops, "decode_post_o")
        )
    else:
        adaptive = (pre_o_split or post_o_split) and adaptive_flag == "1"
    return DecodeConfig(
        tp=tp,
        graph=not mc.enforce_eager,
        max_context_len=mc.max_model_len,
        speculative_tokens=(
            config.speculative_config.num_speculative_tokens
            if config.speculative_config is not None
            else 0
        ),
        tail_only=tail_supported and getenv("MINICPM5_MONO_TAIL_ONLY", "1") == "1",
        tail_rows4=(
            tail_supported
            and getenv("MINICPM5_MONO_TAIL_ROWS4", "1" if monokernel else "0") == "1"
        ),
        pre_o_split=pre_o_split,
        post_o_split=post_o_split,
        pre_o_fa2_adaptive_split1=adaptive,
    )
