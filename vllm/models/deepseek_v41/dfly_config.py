# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of the standalone DeepSeek-V4.1 DFly2 checkpoint."""

from copy import deepcopy
from functools import partial

from transformers import PreTrainedConfig

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dfly.config import DFlySpeculatorConfig


def derive_dfly_config(
    hf_config: PreTrainedConfig, *, target_config: PreTrainedConfig
) -> PreTrainedConfig:
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    if hf_config.architectures != ["DFlyDeepseekV41DraftModel"]:
        raise ValueError(
            "DFly2 requires a DFlyDeepseekV41DraftModel checkpoint, "
            f"got {hf_config.architectures!r}"
        )
    config = deepcopy(hf_config)
    dfly = getattr(config, "dfly_config", None)
    if not isinstance(dfly, dict):
        raise ValueError("DFly2 checkpoint is missing dfly_config")
    target = target_config.get_text_config()
    if target.model_type != "deepseek_v41":
        raise ValueError("This DFly2 checkpoint requires a DeepSeek-V4.1 target")
    for field in (
        "vocab_size",
        "hidden_size",
        "head_dim",
        "qk_rope_head_dim",
        "num_attention_heads",
        "o_groups",
        "q_lora_rank",
        "o_lora_rank",
        "sliding_window",
        "hc_mult",
        "index_topk",
    ):
        if getattr(config, field) != getattr(target, field):
            raise ValueError(f"DFly2 and target must have the same {field}")
    if config.head_dim != 512 or (config.index_topk + config.sliding_window) % 128:
        raise ValueError("DFly2 requires 512-wide MLA and a 128-aligned context")
    for name in ("target_kv_layers", "target_index_layers"):
        selected = dfly.get(name)
        if (
            not isinstance(selected, list)
            or not selected
            or any(
                type(i) is not int or not 0 <= i < target.num_hidden_layers
                for i in selected
            )
        ):
            raise ValueError(f"DFly2 {name} must contain valid target layers")
    kv_layers = dfly["target_kv_layers"]
    index_layers = dfly["target_index_layers"]
    if len(kv_layers) != len(index_layers):
        raise ValueError("DFly2 requires one KV/index pair per draft layer")
    for kv, index in zip(kv_layers, index_layers):
        if (
            target.compress_ratios[kv] != 1
            or index not in target.index_source_layer_ids
        ):
            raise ValueError("DFly2 requires ratio-1 KV and target index-source layers")
        if max(s for s in target.kv_source_layer_ids if s <= kv) != max(
            s for s in target.kv_source_layer_ids if s <= index
        ):
            raise ValueError("DFly2 KV and index layers must share a compressed cache")
    block_size = dfly.get("block_size")
    if type(block_size) is not int or not 2 <= block_size <= 16:
        raise ValueError("DFly2 block_size must be in [2, 16]")
    if dfly.get("causal_block") is not False or dfly.get("mtp_fusion") != "slot0":
        raise ValueError("DFly2 requires a noncausal block and slot0 MTP fusion")
    if dfly.get("dense_dtype") != "bfloat16":
        raise ValueError("DFly2 requires BF16 dense weights")
    mask = dfly.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < config.vocab_size:
        raise ValueError("DFly2 mask_token_id is outside the vocabulary")
    if dfly.get("index_topk") != target.index_topk:
        raise ValueError("DFly2 index_topk must match the target")
    rope = dfly.get("rope", {})
    target_rope = target.rope_parameters
    target_rope = target_rope.get("compress", target_rope)
    for name, expected in (
        ("theta", target.compress_rope_theta),
        ("original_seq_len", target_rope.get("original_max_position_embeddings")),
        ("factor", target_rope.get("factor")),
        ("beta_fast", target_rope.get("beta_fast", 32)),
        ("beta_slow", target_rope.get("beta_slow", 1)),
    ):
        if rope.get(name) != expected:
            raise ValueError(f"DFly2 RoPE {name} disagrees with the target")
    correction = dfly.get("attentive_correction", {})
    if any(
        correction.get(k) != v
        for k, v in (
            ("block_keys", True),
            ("block_mode", "gated"),
            ("block_kv_in", "hidden_embed"),
            ("max_block_slots", 16),
        )
    ):
        raise ValueError("DFly2 requires gated hidden/embedding block correction")
    source = correction.get("source_layer")
    if type(source) is not int or not -len(kv_layers) <= source < len(kv_layers):
        raise ValueError("DFly2 correction source_layer is outside the draft")
    intermediate = dfly.get("hidden_correction", {}).get("intermediate_size")
    if type(intermediate) is not int or intermediate <= 0:
        raise ValueError("DFly2 hidden correction intermediate_size must be positive")
    config.num_hidden_layers = len(kv_layers)
    # The released config retains the backbone's MoE fields. DFly2's exported
    # three-layer stack uses the 128-expert, top-3 training configuration.
    config.n_routed_experts = dfly.get("n_routed_experts", 128)
    config.num_experts_per_tok = dfly.get("num_experts_per_tok", 3)
    if (
        type(config.n_routed_experts) is not int
        or type(config.num_experts_per_tok) is not int
        or not 1 <= config.num_experts_per_tok <= config.n_routed_experts
    ):
        raise ValueError("DFly2 requires valid routed/active expert counts")
    config.num_nextn_predict_layers = 0
    config.vision_n_layers = 0
    config.engram_layer_ids = []
    config.mask_token_id = mask
    config.target_index_layer = index_layers
    config.target_kv_layers = kv_layers
    config.hidden_correction_intermediate_size = intermediate
    return config


class DeepseekV41DFlySpeculatorConfig(DFlySpeculatorConfig):
    speculator_class = (
        "vllm.models.deepseek_v41.nvidia.dfly_speculator.DeepseekV41DFlySpeculator"
    )

    @classmethod
    def default_model(cls, spec):
        raise ValueError("DeepSeek-V4.1 DFly2 requires a separate draft model")

    @classmethod
    def draft_hf_overrides(cls, spec):
        return partial(
            derive_dfly_config, target_config=spec.target_model_config.hf_config
        )

    @classmethod
    def adapt_draft_model_config(cls, spec):
        config = spec.draft_model_config.hf_config
        parallel = spec.target_parallel_config
        for field in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "data_parallel_size",
        ):
            if getattr(parallel, field) != 1:
                raise ValueError(f"DeepSeek-V4.1 DFly2 requires {field}=1")
        if spec.draft_tensor_parallel_size not in (None, parallel.tensor_parallel_size):
            raise ValueError("DFly2 requires matching target and draft TP")
        if config.num_attention_heads // parallel.tensor_parallel_size > 128:
            raise ValueError("DFly2 supports at most 128 attention heads per rank")
        for field in (
            "num_attention_heads",
            "o_groups",
            "moe_intermediate_size",
            "hidden_correction_intermediate_size",
        ):
            if getattr(config, field) % parallel.tensor_parallel_size:
                raise ValueError(f"DFly2 {field} must be divisible by TP")
        if str(spec.target_model_config.dtype) != "torch.bfloat16":
            raise ValueError("DFly2 requires bfloat16 computation")
        if spec.draft_model_config.quantization not in (None, "deepseek_v4_fp8"):
            raise ValueError(
                "DFly2 supports BF16 dense weights and native MXFP4 experts"
            )
        quant = getattr(config, "quantization_config", None)
        if quant is not None and (
            quant.get("expert_dtype") != "fp4"
            or quant.get("weight_block_size") != [32, 32]
        ):
            raise ValueError("DFly2 requires the native MXFP4 checkpoint format")

    @classmethod
    def resolve_num_speculative_tokens(cls, spec):
        block_size = spec.draft_model_config.hf_config.dfly_config["block_size"]
        if spec.num_speculative_tokens is None:
            spec.num_speculative_tokens = block_size - 1
        if spec.num_speculative_tokens != block_size - 1:
            raise ValueError("DFly2 num_speculative_tokens must equal block_size - 1")

    @classmethod
    def apply_options(cls, spec):
        if (
            not current_platform.is_cuda()
            or not current_platform.is_device_capability_family(90)
        ):
            raise NotImplementedError("DeepSeek-V4.1 DFly2 requires NVIDIA SM90")
        if spec.kv_cache_dtype not in (None, "auto", "fp8_ds_mla"):
            raise ValueError("DFly2 requires the target's fp8_ds_mla KV layout")
        if spec.attention_backend is not None:
            raise ValueError("DFly2 reads target KV; omit the draft attention backend")
        super().apply_options(spec)
