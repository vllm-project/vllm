# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-effect-free Mistral Large 4 specialized-decode configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from vllm.config import CompilationMode, VllmConfig

AUTO, SPECIALIZED = "auto", "latency_specialized"
# the kernel's step rows at most (``mono/kernel.py``: ``1 <= S <= 12``)
MAX_ROWS = 12
# speculative methods whose verify step the kernels serve: a uniform block of
# ``num_speculative_tokens + 1`` causal rows per request, the target's residual
# stream tapped at ``aux_hidden_state_layers``
SPEC_METHODS = ("dspark",)


@dataclass(frozen=True)
class DecodeConfig:
    tp_size: int = 4  # the fused indexer's q_b split needs 16 local heads
    blocks: int = 256  # one CTA per CU, all co-resident
    # rows a launch serves (one decode token per request): vLLM's FULL-graph
    # capture sizes, so graph replays pad decode batches to them; eager steps
    # of other sizes and batches above 4 rows run stock. The 8-row launch
    # builds and is correct but measured slower than vLLM's layers (super-linear
    # up/gate and down stages), so it is not served.
    widths: tuple[int, ...] = (1, 2, 4)
    topk: int = 2048
    # rows a request takes in a step: 1, or num_speculative_tokens + 1 for a
    # speculative verify block (the kernels' ``index_request_width``)
    q: int = 1

    def refusal_reasons(self, config: VllmConfig) -> list[str]:
        """Default-deny: every deployment feature the kernels do not serve."""
        pc = config.parallel_config
        cc = config.cache_config
        reasons: list[str] = []
        if pc.tensor_parallel_size != self.tp_size:
            reasons.append(f"requires tensor parallel size {self.tp_size}")
        if pc.pipeline_parallel_size != 1:
            reasons.append("does not support pipeline parallelism")
        if pc.data_parallel_size != 1:
            reasons.append("does not support data parallelism")
        if pc.decode_context_parallel_size != 1:
            reasons.append("does not support decode context parallelism")
        if pc.prefill_context_parallel_size != 1:
            reasons.append("does not support prefill context parallelism")
        if pc.use_ubatching:
            reasons.append("does not support DBO or ubatching")
        if pc.enable_expert_parallel:
            reasons.append("does not support expert parallelism")
        if pc.enable_eplb:
            reasons.append("does not support EPLB")
        if str(cc.cache_dtype) not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            reasons.append("requires a bf16 or fp8 (e4m3) KV cache")
        if cc.block_size is not None and cc.block_size % 16:
            reasons.append("requires a KV cache block size that is a multiple of 16")
        if cc.kv_offloading_size is not None:
            reasons.append("does not support KV cache offloading")
        if config.lora_config is not None:
            reasons.append("does not support LoRA")
        transfer = config.kv_transfer_config
        if transfer is not None and transfer.kv_connector is not None:
            reasons.append("does not support a KV connector")
        if not config.use_v2_model_runner:
            reasons.append("requires Model Runner V2")
        spec = config.speculative_config
        if spec is not None:
            if getattr(spec, "method", None) not in SPEC_METHODS:
                reasons.append(
                    "supports only " + "/".join(SPEC_METHODS) + " speculative decoding"
                )
            dynamic = getattr(spec, "uses_dynamic_speculative_decoding", None)
            if dynamic is not None and dynamic():
                reasons.append("does not support dynamic speculative decoding")
            if (getattr(spec, "num_speculative_tokens", 0) or 0) + 1 > MAX_ROWS:
                reasons.append(f"supports at most {MAX_ROWS - 1} speculative tokens")
        if config.model_config.enable_sleep_mode:
            reasons.append("does not support sleep mode")
        if config.weight_transfer_config is not None:
            reasons.append("does not support weight transfer or updates")
        offload = config.offload_config
        if offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0:
            reasons.append("does not support model weight offloading")
        if config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            reasons.append("does not support stock torch.compile")
        aux = config.aux_output_config
        if getattr(aux, "enable_return_routed_experts", False):
            reasons.append("does not support routed-expert capture")
        quant = config.model_config.quantization
        if quant == "fp8" and (
            os.environ.get("VLLM_FP8_MOE_REQUANT", "").lower() != "mxfp4"
            or os.environ.get("VLLM_FP8_MOE_REQUANT_ACT", "a8w4").lower() != "a4w4"
        ):
            reasons.append(
                "requires MXFP4 a4w4 routed experts (VLLM_FP8_MOE_REQUANT=mxfp4 "
                "VLLM_FP8_MOE_REQUANT_ACT=a4w4)"
            )
        return reasons


DECODE_CONFIG = DecodeConfig()


def decode_backend(config: VllmConfig) -> str:
    """The requested decode backend: ``--additional-config
    '{"decode_backend": "latency_specialized"}'``, or ``--monokernel`` where the
    kernel config has it."""
    if getattr(config.kernel_config, "monokernel", False):
        return SPECIALIZED
    extra = config.additional_config
    return extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO


def specialized_decode_config(config: VllmConfig) -> DecodeConfig | None:
    """The deployment's specialized decode, None for vLLM's decode. Raises with
    every reason when it is requested for an unsupported deployment."""
    backend = decode_backend(config)
    if backend == AUTO:
        return None
    if backend != SPECIALIZED:
        raise ValueError(
            f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}"
        )
    reasons = DECODE_CONFIG.refusal_reasons(config)
    if reasons:
        raise ValueError(
            "Mistral Large 4 specialized decode unsupported: " + "; ".join(reasons)
        )
    spec = config.speculative_config
    if spec is None:
        return DECODE_CONFIG
    # one request's verify block (q rows) per launch: two requests (8 rows at
    # q=4) build and are correct but measured slower than vLLM's layers, as the
    # 8-row launch without speculation (see ``widths``)
    q = spec.num_speculative_tokens + 1
    return replace(DECODE_CONFIG, q=q, widths=(q,))
