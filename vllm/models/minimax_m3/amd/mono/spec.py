# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contract for the MiniMax-M3 ROCm MonoKernel."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.config import VllmConfig


@dataclass(frozen=True)
class MiniMaxM3MonoSpec:
    tp_size: int = 4
    blocks: int = 256
    max_rows: int = 16
    widths: tuple[int, ...] = (1, 2, 4, 8, 16)
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
        if pc.enable_expert_parallel:
            reasons.append("does not support expert parallelism")
        if pc.enable_eplb:
            reasons.append("does not support EPLB")
        if cc.cache_dtype not in ("fp8", "fp8_e4m3"):
            reasons.append("requires an FP8 KV cache")
        if cc.block_size != 128:
            reasons.append("requires KV cache block size 128")
        if cc.enable_prefix_caching:
            reasons.append("does not yet support prefix caching")

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
        text_config = config.model_config.hf_text_config
        if getattr(text_config, "use_index_cache", False):
            reasons.append("does not support cross-layer index reuse")
        if config.kernel_config.moe_backend != "aiter":
            reasons.append("requires --moe-backend aiter")
        sparse = getattr(text_config, "sparse_attention_config", None)
        if text_config.num_hidden_layers != 60 or not sparse:
            reasons.append("requires the 60-layer MiniMax M3 sparse schema")
        else:
            frequencies = sparse.get("sparse_attention_freq", ())
            sparse_layers = [i for i, enabled in enumerate(frequencies) if enabled]
            if sparse_layers != list(range(self.first_layer, self.last_layer + 1)):
                reasons.append("requires sparse attention on layers 3 through 59")

        return reasons

    def extend_final_graph_widths(self, final_sizes: list[int]) -> list[int]:
        """Add mono widths after vLLM resolves its default graph-size list."""
        return sorted(set(final_sizes).union(self.widths))


MONO_SPEC = MiniMaxM3MonoSpec()
