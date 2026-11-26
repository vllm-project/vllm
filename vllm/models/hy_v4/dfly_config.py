# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

from transformers import PreTrainedConfig

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dfly.config import DFlySpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


def _validate_checkpoint_quantization(config: object) -> None:
    error = (
        "HYV4 requires native blockwise FP8 or compressed-tensors FP8 "
        "(128x128 weights, dynamic group-128 activations) and/or weight-only "
        "MXFP4 (group-32 weights, uint8 scales), without checkpoint "
        "kv_cache_scheme or transforms"
    )
    if not isinstance(config, dict) or (
        config.get("kv_cache_scheme") is not None
        or config.get("transform_config") is not None
    ):
        raise ValueError(error)
    if config.get("quant_method") == "fp8":
        if (
            config.get("activation_scheme") != "dynamic"
            or config.get("weight_block_size") != [128, 128]
            or config.get("fmt") not in (None, "e4m3")
            or config.get("store_dtype") not in (None, "fp8")
            or config.get("expert_dtype") not in (None, "fp8")
        ):
            raise ValueError(error)
        return
    quant_format = config.get("format")
    if config.get("quant_method") != "compressed-tensors" or quant_format not in (
        "float-quantized",
        "mxfp4-pack-quantized",
        "mixed-precision",
    ):
        raise ValueError(error)
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    expected_weight = dict(
        num_bits=8,
        type="float",
        strategy="block",
        block_structure=[128, 128],
        symmetric=True,
        dynamic=False,
    )
    expected_activation = dict(
        num_bits=8,
        type="float",
        strategy="group",
        group_size=128,
        symmetric=True,
        dynamic=True,
    )
    expected_mxfp4_weight = dict(
        num_bits=4,
        type="float",
        strategy="group",
        group_size=32,
        symmetric=True,
        dynamic=False,
    )
    for group in groups.values():
        if not isinstance(group, dict):
            raise ValueError(error)
        weights = group.get("weights")
        activations = group.get("input_activations")
        if not isinstance(weights, dict) or group.get("output_activations") is not None:
            raise ValueError(error)
        group_format = group.get("format") or quant_format
        is_fp8 = (
            group_format == "float-quantized"
            and isinstance(activations, dict)
            and all(weights.get(k) == v for k, v in expected_weight.items())
            and all(activations.get(k) == v for k, v in expected_activation.items())
        )
        is_mxfp4 = (
            group_format == "mxfp4-pack-quantized"
            and activations is None
            and all(weights.get(k) == v for k, v in expected_mxfp4_weight.items())
            and weights.get("scale_dtype") in (None, "uint8", "torch.uint8")
        )
        if not (is_fp8 or is_mxfp4):
            raise ValueError(error)


def _validate_dfly_quantization(config: object) -> None:
    if isinstance(config, dict) and config.get("quant_method") == "compressed-tensors":
        # Any layer scheme works for the target; draft layers check their own.
        # An empty transform_config means no transforms.
        if config.get("kv_cache_scheme") is not None or config.get("transform_config"):
            raise ValueError(
                "DFly does not support compressed-tensors kv_cache_scheme or transforms"
            )
        return
    _validate_checkpoint_quantization(config)


def _derive_dfly_config(
    hf_config: PreTrainedConfig,
    *,
    target_config: PreTrainedConfig,
    target_model_type: str = "hy_v4",
) -> PreTrainedConfig:
    """Derive HYV4's text-only draft without changing the target configuration."""
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    config = deepcopy(target_config.get_text_config())
    if config.model_type != target_model_type:
        raise ValueError(f"DFly requires a {target_model_type} target model")
    num_layers = getattr(target_config, "dfly_num_hidden_layers", None)
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("DFly requires a positive dfly_num_hidden_layers")
    config.target_num_hidden_layers = config.num_hidden_layers
    config.target_hidden_size = config.hidden_size
    config.dfly_intermediate_size = getattr(
        target_config, "dfly_intermediate_size", None
    )
    if config.dfly_intermediate_size is not None and (
        type(config.dfly_intermediate_size) is not int
        or config.dfly_intermediate_size < 1
    ):
        raise ValueError("DFly intermediate size must be positive")
    for name, default in (
        ("enable_ihc", False),
        ("mtp_fusion", "slot0"),
        ("target_index_topk", config.index_topk),
        ("context_local_window", None),
        ("enable_hidden_correction", True),
        ("mlp_type", "dense"),
        ("intermediate_size", config.intermediate_size),
        ("attn_post_proj_norm", None),
        ("mask_token_id", None),
        ("target_index_layer", None),
        ("target_kv_layers", None),
        ("hidden_correction_intermediate_size", config.hidden_size),
    ):
        setattr(config, name, deepcopy(getattr(target_config, f"dfly_{name}", default)))
    if (
        config.attn_post_proj_norm is not None
        and type(config.attn_post_proj_norm) is not bool
    ):
        raise ValueError("DFly attention post-projection norm must be boolean")
    if config.mlp_type != "dense" or config.enable_ihc is not False:
        raise ValueError("DFly requires dense FFNs and dfly_enable_ihc=false")
    if getattr(target_config, "dfly_attn_causal", True) is not True:
        raise ValueError("DFly requires causal MLA attention")
    if config.mtp_fusion != "slot0" or config.enable_hidden_correction is not True:
        raise ValueError("DFly requires slot0 fusion and hidden correction")
    if type(config.mask_token_id) is not int or not (
        0 <= config.mask_token_id < config.vocab_size
    ):
        raise ValueError("DFly dfly_mask_token_id is outside the vocabulary")
    if (
        type(config.target_index_topk) is not int
        or config.target_index_topk <= 0
        or (
            config.context_local_window is not None
            and (
                type(config.context_local_window) is not int
                or config.context_local_window < 0
            )
        )
    ):
        raise ValueError("DFly needs positive target top-k and nonnegative window")
    if (
        type(config.hidden_correction_intermediate_size) is not int
        or config.hidden_correction_intermediate_size <= 0
    ):
        raise ValueError("DFly hidden correction intermediate size must be positive")
    config.model_type = "hy_v4_dfly"
    config.architectures = ["HYV4DFlyModel"]
    config.num_hidden_layers = num_layers
    config.n_routed_experts = 0
    config.n_shared_experts = 0
    config.num_experts = 0
    config.num_nextn_predict_layers = 0
    config.n_predict = 0
    config.mlp_layer_types = ["dense"] * num_layers
    config.layer_types = ["deepseek_sparse_attention"] * num_layers
    config.indexer_types = ["full"] * num_layers
    return config


def verify_rocm_dfly_kv_cache_dtype(cache_dtype: str | None) -> None:
    """Reject KV caches the ROCm DFly sparse MLA path cannot apply sinks to."""
    if cache_dtype not in (None, "auto", "bfloat16"):
        raise ValueError(
            f"HYV4 DFly on ROCm requires a BF16 KV cache, got {cache_dtype!r}"
        )


class HYV4DFlySpeculatorConfig(DFlySpeculatorConfig):
    """HYV4's configuration and implementation of DFly speculative decoding."""

    # Resolved per platform by the `vllm.models.hy_v4` entry point.
    speculator_class = "vllm.models.hy_v4.HYV4DFlySpeculator"
    target_model_type = "hy_v4"

    @classmethod
    def derive_draft_hf_config(cls, hf_config, *, target_config):
        return _derive_dfly_config(
            hf_config,
            target_config=target_config,
            target_model_type=cls.target_model_type,
        )

    @classmethod
    def target_rope_parameters(cls, target_text_config, draft_config):
        return target_text_config.rope_parameters

    @classmethod
    def verify_target_parallel_config(cls, spec: "SpeculativeConfig") -> None:
        for field in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(spec.target_parallel_config, field) != 1:
                raise ValueError(f"DFly requires {field}=1")

    @classmethod
    def draft_hf_overrides(
        cls, spec: "SpeculativeConfig"
    ) -> Callable[[PreTrainedConfig], PreTrainedConfig]:
        target = spec.target_model_config
        if spec.model != target.model:
            raise ValueError("DFly weights must be in the target checkpoint")
        target_config = deepcopy(target.hf_config)
        return partial(cls.derive_draft_hf_config, target_config=target_config)

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        target = spec.target_model_config
        draft = spec.draft_model_config
        draft.model_weights = target.model_weights
        config = draft.hf_config
        if config.model_type != "hy_v4_dfly":
            raise ValueError("method='dfly' requires a HYV4 DFly checkpoint")
        if target.hf_text_config.model_type != cls.target_model_type:
            raise ValueError(f"DFly requires a {cls.target_model_type} target model")
        if str(target.dtype) != "torch.bfloat16":
            raise ValueError("DFly requires a BF16 target dtype")
        checkpoint_quantized = target.quantization in ("fp8", "compressed-tensors")
        if checkpoint_quantized:
            _validate_dfly_quantization(target.model_arch_config.quantization_config)
        elif target.quantization is not None:
            from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8Static128BlockSym,
            )

            quantization_config = target.quantization_config
            if isinstance(quantization_config, dict):
                quantization_config = QuantizationConfigArgs(**quantization_config)
            if target.quantization != "online" or quantization_config != (
                QuantizationConfigArgs(moe=QuantSpec(weight=kFp8Static128BlockSym))
            ):
                raise ValueError(
                    "DFly supports only unquantized targets, compressed-tensors or "
                    "blockwise FP8 checkpoints, or online MoE-only fp8_per_block "
                    "quantization"
                )
        if checkpoint_quantized:
            if draft.quantization != target.quantization or spec.quantization not in (
                None,
                target.quantization,
            ):
                raise ValueError("DFly must use the checkpoint's quantization")
            _validate_dfly_quantization(draft.model_arch_config.quantization_config)
        elif draft.quantization is not None or spec.quantization is not None:
            raise ValueError("DFly does not support quantized draft weights")
        parallel = spec.target_parallel_config
        cls.verify_target_parallel_config(spec)
        if spec.draft_tensor_parallel_size not in (None, parallel.tensor_parallel_size):
            raise ValueError("DFly draft and target must use the same TP size")
        for size in (
            config.num_attention_heads,
            config.dfly_intermediate_size,
            config.hidden_correction_intermediate_size,
        ):
            if size is not None and size % parallel.tensor_parallel_size:
                raise ValueError(
                    "DFly heads and FFN dimensions must be divisible by TP"
                )
        for field in ("vocab_size", "hidden_size", "kv_lora_rank", "qk_rope_head_dim"):
            if getattr(config, field) != getattr(target.hf_text_config, field):
                raise ValueError(f"DFly and target must have the same {field}")
        if config.target_hidden_size != target.get_hidden_size():
            raise ValueError("DFly target_hidden_size disagrees with the target")
        if config.target_index_topk != target.hf_text_config.index_topk:
            raise ValueError("DFly target_index_topk disagrees with the target")
        if config.rope_parameters != cls.target_rope_parameters(
            target.hf_text_config, config
        ):
            raise ValueError("DFly and target must use identical RoPE parameters")
        layers = target.hf_text_config.num_hidden_layers
        if config.target_num_hidden_layers not in (None, layers):
            raise ValueError("DFly target_num_hidden_layers disagrees with the target")
        indices = config.target_index_layer
        if isinstance(indices, int) and not isinstance(indices, bool):
            indices = [indices] * config.num_hidden_layers
        for name, selected in (
            ("target_index_layer", indices),
            ("target_kv_layers", config.target_kv_layers),
        ):
            if (
                not isinstance(selected, list)
                or len(selected) != config.num_hidden_layers
                or any(type(i) is not int or not 1 <= i < layers for i in selected)
            ):
                raise ValueError(
                    f"DFly {name} needs one valid target layer per draft layer"
                )
        if len(set(config.target_kv_layers)) != config.num_hidden_layers:
            raise ValueError("DFly requires distinct target KV layers")
        config.target_index_layer = indices
        super().adapt_draft_model_config(spec)

    @classmethod
    def verify_platform(cls, spec: "SpeculativeConfig") -> None:
        if not current_platform.is_rocm():
            return
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            raise NotImplementedError("HYV4 DFly on ROCm requires gfx950")
        verify_rocm_dfly_kv_cache_dtype(spec.kv_cache_dtype)

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        cls.verify_platform(spec)
        if spec.kv_cache_dtype not in (None, "auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla draft MLA KV cache")
        if spec.attention_backend is not None:
            raise ValueError("DFly does not support a draft attention backend override")
        super().apply_options(spec)
