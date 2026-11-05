# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 / Qwen3.6 MoE mono decode on two Xe2 GPUs (TP2), enabled by
``--monokernel``.

``specialized_decode_config`` runs in the model's ``__init__``, before the
weights load. It returns None for vLLM's decode, and raises with every reason
from ``unsupported`` when the mono decode is requested for a deployment it
cannot serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``), those that need the caches at every bind
(``decode.SpecializedDecode.bind_caches``)."""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TP = 2
# the 35B-A3B shapes the kernels are built for (hidden 2048, 256 experts top-8
# of inter 512 + one shared expert, GatedDeltaNet 16 k / 32 v heads of 128 with
# a width-4 conv, full attention 16 q / 2 kv heads of 256)
SHAPES = dict(
    hidden_size=2048,
    num_experts=256,
    num_experts_per_tok=8,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=256,
)
ARCHITECTURES = ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
# model dtypes the kernels serve (activations, KV cache and conv state) and
# their cache dtype names
DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
# rows of the decode kernels (one token a request)
DECODE_ROWS = (1, 2, 4, 8)
# (rows, tokens a request) of the verify kernels: rows = requests x tokens
VERIFY = frozenset({(2, 2), (3, 3), (4, 2), (4, 4), (6, 3), (8, 2), (8, 4), (8, 8)})
# rows of the MTP prologue kernel
MTP_ROWS = (1, 2, 3, 4, 6, 8)
# speculative tokens the verify kernels serve: 1 .. 7 (8 tokens a request)
MAX_SPEC_TOKENS = 7
# kernel-side ids of the MTP drafter's layer and prologue
MTP_LAYER_ID, MTP_PROLOGUE_ID = 1000, 2000


@dataclass(frozen=True)
class DecodeConfig:
    # tokens a request verifies (num_speculative_tokens + 1), 1 without
    # speculative decoding
    q: int = 1

    @property
    def verify(self) -> frozenset[tuple[int, int]]:
        """The verify rows served at this q."""
        return frozenset(r for r in VERIFY if r[1] == self.q)


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's mono decode settings, None for vLLM's decode. Raises
    when ``--monokernel`` is set for an unsupported deployment, listing every
    reason."""
    if not vllm_config.kernel_config.monokernel:
        return None
    reasons = unsupported(vllm_config)
    if reasons:
        raise ValueError(
            "Qwen3.5 XPU mono decode (--monokernel) does not support: "
            + "; ".join(reasons)
            + " (without --monokernel vLLM's decode runs)"
        )
    spec = vllm_config.speculative_config
    return DecodeConfig(q=spec.num_speculative_tokens + 1 if spec is not None else 1)


def unsupported(vllm_config, ops=None) -> list[str]:
    """Every reason this deployment cannot run the mono decode (empty: it can).
    Config only: the loaded weights and the caches are checked later. Default
    deny: every feature below is refused, the kernels serve none of them.
    ``ops``: ``torch.ops._xpu_C`` (overridable for tests)."""
    from vllm.config import CompilationMode, CUDAGraphMode

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    spec, oc = vllm_config.speculative_config, vllm_config.offload_config
    cg = vllm_config.compilation_config
    quant = vllm_config.quant_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        # the sampled hidden states run through compute_sample_logits
        (bool(pc.enable_batch_sharded_sampling), "batch-sharded sampling"),
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.kv_transfer_config is not None, "KV transfer connectors"),
        # the kernels read the loaded weights and caches in place
        (mc.enable_sleep_mode, "sleep mode"),
        (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0,
            "CPU weight offload",
        ),
        (vllm_config.weight_transfer_config is not None, "weight updates"),
        (
            vllm_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        (
            mc.dtype not in DTYPES,
            f"dtype {mc.dtype} (float16 or bfloat16 only)",
        ),
        # the weight check confirms per-tensor scales (``guards``)
        (
            quant is None or mc.quantization not in ("fp8", "fp8_per_tensor"),
            f"quantization {mc.quantization} (online fp8 per-tensor only: "
            "--quantization fp8_per_tensor)",
        ),
        # activations, KV cache and conv state in one dtype (no fp16 / bf16 mix)
        (
            cc.cache_dtype not in ("auto", DTYPES.get(mc.dtype)),
            f"KV cache dtype {cc.cache_dtype} (the model dtype only)",
        ),
        (
            cc.mamba_cache_dtype not in ("auto", DTYPES.get(mc.dtype)),
            f"conv state dtype {cc.mamba_cache_dtype} (the model dtype only)",
        ),
        (
            cc.block_size is not None and cc.block_size % 64 != 0,
            f"block size {cc.block_size} (a multiple of 64)",
        ),
        (
            cc.mamba_ssm_cache_dtype not in ("auto", "float32"),
            f"recurrent state dtype {cc.mamba_ssm_cache_dtype} (float32 only)",
        ),
        (
            spec is not None and spec.method != "mtp",
            f"speculative method {spec.method if spec else None!r} (mtp only)",
        ),
        (
            spec is not None
            and not 1 <= spec.num_speculative_tokens <= MAX_SPEC_TOKENS,
            f"{spec.num_speculative_tokens if spec else 0} speculative tokens "
            f"(1 .. {MAX_SPEC_TOKENS})",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        (
            spec is not None and spec.uses_dynamic_speculative_decoding(),
            "dynamic speculative decoding",
        ),
        # the model state builds the executor and binds it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the layer kernels run in forward; a compiled forward
        # would trace them
        (
            cg.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
        # a decode step must be one graph: PIECEWISE graphs replay one capture
        # for any batch of the same size (prefill or mixed included)
        (
            mc.enforce_eager
            or cg.cudagraph_mode is None
            or cg.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL,
            f"decode graphs other than FULL (cudagraph_mode {cg.cudagraph_mode}; "
            "use FULL_DECODE_ONLY or FULL_AND_PIECEWISE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    archs = mc.architectures or []
    if not any(a in ARCHITECTURES for a in archs):
        reasons.append(f"architecture {archs} ({', '.join(ARCHITECTURES)} only)")
    text = mc.hf_text_config
    for key, want in SHAPES.items():
        if getattr(text, key, None) != want:
            reasons.append(
                f"{key} {getattr(text, key, None)} (the kernels are built for {want})"
            )
    if getattr(text, "layer_scale", False):
        reasons.append("layer_scale")
    platform = platform_reasons(ops)
    if not platform:
        limit = int(_ops(ops).mono_max_context())
        if mc.max_model_len > limit:
            reasons.append(
                f"max_model_len {mc.max_model_len} (the attention kernels serve "
                f"up to {limit} tokens)"
            )
    return reasons + platform


# the vllm-xpu-kernels ops the executor calls (schemas in README.md)
OPS = (
    "mono_decode_supported",
    "mono_max_context",
    "mono_health",
    "mono_ipc_create",
    "mono_ipc_export",
    "mono_ipc_open",
    "mono_ipc_ptr",
    "mono_mbox_bytes",
    "mono_register_layer",
    "mono_decode_layer",
    "mono_decode_layer_spec",
    "mono_register_mtp",
    "mono_mtp_prologue",
    "mono_lmhead_mbox_bytes",
    "mono_shard_topk_max_k",
    "mono_shard_topk",
)


def _ops(ops):
    return torch.ops._xpu_C if ops is None else ops


def platform_reasons(ops=None) -> list[str]:
    """Why this process cannot launch the kernels (empty: it can)."""
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        return ["this platform (XPU only)"]
    ops = _ops(ops)
    missing = [name for name in OPS if not hasattr(ops, name)]
    if missing:
        return [
            "vllm-xpu-kernels without the mono decode kernels (built with "
            f"MONO_DECODE_KERNELS_ENABLED; missing {', '.join(missing)})"
        ]
    if not ops.mono_decode_supported(-1):
        return ["this GPU (Xe2 / Battlemage only)"]
    return []
