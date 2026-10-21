# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-effect-free configuration for MiniMax-M3 specialized decode."""

from __future__ import annotations

from dataclasses import dataclass, replace

from vllm.config import CompilationMode, VllmConfig


@dataclass(frozen=True)
class DecodeConfig:
    tp_size: int = 4
    blocks: int = 256
    max_rows: int = 16
    # every physical row count up to max_rows: eager steps and FULL capture sizes
    widths: tuple[int, ...] = tuple(range(1, 17))
    first_layer: int = 3
    last_layer: int = 59

    def refusal_reasons(self, config: VllmConfig) -> list[str]:
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
        if cc.cache_dtype not in ("fp8", "fp8_e4m3"):
            reasons.append("requires an FP8 KV cache")
        if cc.block_size != 128:
            reasons.append("requires KV cache block size 128")
        if cc.kv_offloading_size is not None:
            reasons.append("does not support KV cache offloading")

        index_dtype = config.attention_config.resolve_indexer_kv_dtype("bf16")
        if index_dtype != "fp8":
            reasons.append('requires indexer_kv_dtype "fp8"')
        if getattr(config, "lora_config", None) is not None:
            reasons.append("does not support LoRA")
        transfer = getattr(config, "kv_transfer_config", None)
        if transfer is not None and transfer.kv_connector is not None:
            reasons.append("does not support a KV connector")
        if not getattr(config, "use_v2_model_runner", False):
            reasons.append("requires Model Runner V2")
        if config.speculative_config is not None:
            reasons.append("does not support speculative decoding")
        if config.model_config.enable_sleep_mode:
            reasons.append("does not support sleep mode")
        if config.weight_transfer_config is not None:
            reasons.append("does not support weight transfer or updates")
        offload = config.offload_config
        if offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0:
            reasons.append("does not support model weight offloading")
        if config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            reasons.append("does not support stock torch.compile")
        text_config = config.model_config.hf_text_config
        if getattr(text_config, "use_index_cache", False):
            reasons.append("does not support cross-layer index reuse")
        if getattr(text_config, "index_topk_freq", 1) != 1:
            reasons.append("requires the default index_topk_freq")
        aux = getattr(config, "aux_output_config", None)
        if getattr(aux, "enable_return_routed_experts", False):
            reasons.append("does not support routed-expert capture")
        if config.kernel_config.moe_backend != "aiter":
            reasons.append("requires --moe-backend aiter")
        sparse = getattr(text_config, "sparse_attention_config", None) or {}
        layers = text_config.num_hidden_layers
        frequencies = sparse.get("sparse_attention_freq", ())[:layers]
        sparse_layers = [i for i, enabled in enumerate(frequencies) if enabled]
        if (
            self.last_layer != layers - 1
            or layers <= self.first_layer
            or sparse_layers != list(range(self.first_layer, layers))
        ):
            reasons.append(
                f"requires sparse attention on layers {self.first_layer} through "
                "the last"
            )

        return reasons


DECODE_CONFIG = DecodeConfig()


def specialized_decode_config(config: VllmConfig) -> DecodeConfig | None:
    if not config.kernel_config.specialized_decode:
        return None
    layers = config.model_config.hf_text_config.num_hidden_layers
    spec = replace(DECODE_CONFIG, last_layer=layers - 1)
    reasons = spec.refusal_reasons(config)
    if reasons:
        raise ValueError(
            "MiniMax-M3 specialized decode unsupported: " + "; ".join(reasons)
        )
    return spec
